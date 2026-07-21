"""Train the learned residual closure on freshly sampled random MLPs.

Example:

    python -m repro_external.aicrowd_whestbench.learned_residual.train \
      --config repro_external/aicrowd_whestbench/learned_residual/configs/smoke.json

For multi-GPU training, launch the same command with ``torchrun``.  Batch size
in the JSON config is per GPU.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel

from .model import ModelConfig, ResidualClosureNet, architecture_dict, export_model


@dataclass
class TrainConfig:
    seed: int = 20260721
    width: int = 256
    depth: int = 32
    hidden_dim: int = 32
    blocks: int = 2
    residual_scale: float = 0.10
    steps: int = 3_000
    batch_size: int = 4
    mc_samples: int = 1_024
    mc_chunk_size: int = 512
    antithetic: bool = True
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_steps: int = 200
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    auxiliary_loss_weight: float = 0.10
    log_every: int = 20
    validate_every: int = 250
    checkpoint_every: int = 500
    val_networks: int = 16
    val_batch_size: int = 2
    val_mc_samples: int = 16_384
    val_mc_chunk_size: int = 1_024
    device: str = "auto"
    matmul_precision: str = "high"
    compile_model: bool = False
    output_dir: str = "outputs/whestbench_learned_residual/pilot"

    @classmethod
    def from_json(cls, path: str | Path) -> "TrainConfig":
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        unknown = sorted(set(values) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"unknown training config fields: {unknown}")
        return cls(**values)

    def model_config(self) -> ModelConfig:
        return ModelConfig(
            width=self.width,
            depth=self.depth,
            hidden_dim=self.hidden_dim,
            blocks=self.blocks,
            residual_scale=self.residual_scale,
        )

    def validate(self) -> None:
        self.model_config().validate()
        positive = {
            "steps": self.steps,
            "batch_size": self.batch_size,
            "mc_samples": self.mc_samples,
            "mc_chunk_size": self.mc_chunk_size,
            "learning_rate": self.learning_rate,
            "validate_every": self.validate_every,
            "checkpoint_every": self.checkpoint_every,
            "val_networks": self.val_networks,
            "val_batch_size": self.val_batch_size,
            "val_mc_samples": self.val_mc_samples,
            "val_mc_chunk_size": self.val_mc_chunk_size,
        }
        bad = [name for name, value in positive.items() if value <= 0]
        if bad:
            raise ValueError(f"these fields must be positive: {bad}")
        if self.antithetic and (self.mc_samples % 2 or self.val_mc_samples % 2):
            raise ValueError("MC sample counts must be even when antithetic=true")
        if self.min_learning_rate < 0.0 or self.min_learning_rate > self.learning_rate:
            raise ValueError("min_learning_rate must be in [0, learning_rate]")


def _distributed_context() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        if not torch.cuda.is_available():
            raise RuntimeError("multi-process training currently requires CUDA/NCCL")
        dist.init_process_group(backend="nccl")
    return rank, local_rank, world_size


def _resolve_device(requested: str, local_rank: int) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            requested = f"cuda:{local_rank}"
        elif torch.backends.mps.is_available():
            requested = "mps"
        else:
            requested = "cpu"
    device = torch.device(requested)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return device


def _generator(device: torch.device, seed: int) -> torch.Generator | None:
    # MPS does not consistently support device-local Generator objects across
    # PyTorch versions. Global seeding still makes the smoke path reproducible.
    if device.type == "mps":
        torch.manual_seed(seed)
        return None
    generator_device = device if device.type == "cuda" else torch.device("cpu")
    result = torch.Generator(device=generator_device)
    result.manual_seed(seed)
    return result


def _randn(
    shape: Iterable[int], device: torch.device, generator: torch.Generator | None
) -> Tensor:
    return torch.randn(tuple(shape), device=device, dtype=torch.float32, generator=generator)


def sample_he_weights(
    batch_size: int,
    depth: int,
    width: int,
    *,
    device: torch.device,
    generator: torch.Generator | None,
) -> Tensor:
    return _randn((batch_size, depth, width, width), device, generator) * math.sqrt(
        2.0 / width
    )


@torch.inference_mode()
def monte_carlo_targets(
    weights: Tensor,
    n_samples: int,
    chunk_size: int,
    *,
    antithetic: bool,
    generator: torch.Generator | None,
) -> Tensor:
    """Estimate every layer mean using memory-bounded Gaussian samples."""

    if n_samples <= 0 or chunk_size <= 0:
        raise ValueError("n_samples and chunk_size must be positive")
    if antithetic and n_samples % 2:
        raise ValueError("n_samples must be even for antithetic sampling")
    batch, depth, width, _ = weights.shape
    sums = torch.zeros((depth, batch, width), device=weights.device, dtype=torch.float32)
    samples_done = 0

    if antithetic:
        independent_total = n_samples // 2
        independent_done = 0
        while independent_done < independent_total:
            independent = min(max(1, chunk_size // 2), independent_total - independent_done)
            positive = _randn((batch, independent, width), weights.device, generator)
            activations = torch.cat((positive, -positive), dim=1)
            for layer_index in range(depth):
                activations = torch.relu(torch.bmm(activations, weights[:, layer_index]))
                sums[layer_index].add_(activations.sum(dim=1))
            independent_done += independent
            samples_done += 2 * independent
    else:
        while samples_done < n_samples:
            current = min(chunk_size, n_samples - samples_done)
            activations = _randn((batch, current, width), weights.device, generator)
            for layer_index in range(depth):
                activations = torch.relu(torch.bmm(activations, weights[:, layer_index]))
                sums[layer_index].add_(activations.sum(dim=1))
            samples_done += current

    return (sums / float(samples_done)).permute(1, 0, 2).contiguous()


def _losses(
    prediction: Tensor,
    baseline: Tensor,
    target: Tensor,
    auxiliary_weight: float,
) -> tuple[Tensor, dict[str, float]]:
    final_error = (prediction[:, -1] - target[:, -1]).square().mean()
    baseline_final = (baseline[:, -1] - target[:, -1]).square().mean()
    if prediction.shape[1] > 1:
        layer_weights = torch.linspace(
            0.0, 1.0, prediction.shape[1], device=prediction.device
        ).square()
        auxiliary = (
            (prediction - target).square() * layer_weights[None, :, None]
        ).mean()
    else:
        auxiliary = final_error
    loss = final_error + auxiliary_weight * auxiliary
    metrics = {
        "loss": float(loss.detach()),
        "final_mse": float(final_error.detach()),
        "baseline_final_mse": float(baseline_final.detach()),
        "auxiliary_mse": float(auxiliary.detach()),
    }
    return loss, metrics


def _learning_rate(config: TrainConfig, step: int) -> float:
    if step <= config.warmup_steps and config.warmup_steps > 0:
        return config.learning_rate * step / config.warmup_steps
    progress = (step - config.warmup_steps) / max(config.steps - config.warmup_steps, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
    return config.min_learning_rate + (config.learning_rate - config.min_learning_rate) * cosine


def _unwrap(model: nn.Module) -> ResidualClosureNet:
    while True:
        if hasattr(model, "module"):
            model = model.module  # type: ignore[assignment,union-attr]
            continue
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod  # type: ignore[assignment,union-attr]
            continue
        break
    return model  # type: ignore[return-value]


@torch.inference_mode()
def _build_validation_set(
    config: TrainConfig,
    device: torch.device,
) -> list[tuple[Tensor, Tensor]]:
    weight_generator = _generator(device, config.seed + 80_000_001)
    input_generator = _generator(device, config.seed + 90_000_001)
    batches: list[tuple[Tensor, Tensor]] = []
    remaining = config.val_networks
    while remaining:
        current = min(config.val_batch_size, remaining)
        weights = sample_he_weights(
            current,
            config.depth,
            config.width,
            device=device,
            generator=weight_generator,
        )
        targets = monte_carlo_targets(
            weights,
            config.val_mc_samples,
            config.val_mc_chunk_size,
            antithetic=config.antithetic,
            generator=input_generator,
        )
        batches.append((weights.cpu(), targets.cpu()))
        remaining -= current
    return batches


@torch.inference_mode()
def _validate(
    model: nn.Module,
    validation: list[tuple[Tensor, Tensor]],
    device: torch.device,
    auxiliary_weight: float,
) -> dict[str, float]:
    raw_model = _unwrap(model)
    raw_model.eval()
    totals = {"loss": 0.0, "final_mse": 0.0, "baseline_final_mse": 0.0, "auxiliary_mse": 0.0}
    count = 0
    for weights_cpu, targets_cpu in validation:
        weights = weights_cpu.to(device)
        targets = targets_cpu.to(device)
        prediction, baseline = raw_model(weights)
        _, metrics = _losses(prediction, baseline, targets, auxiliary_weight)
        batch_size = weights.shape[0]
        for name in totals:
            totals[name] += metrics[name] * batch_size
        count += batch_size
    raw_model.train()
    return {name: value / count for name, value in totals.items()}


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    step: int,
    path: Path,
) -> None:
    torch.save(
        {
            "step": step,
            "config": asdict(config),
            "model": _unwrap(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
        },
        path,
    )


def train(config: TrainConfig, *, resume: str | None = None) -> Path:
    config.validate()
    rank, local_rank, world_size = _distributed_context()
    device = _resolve_device(config.device, local_rank)
    torch.set_float32_matmul_precision(config.matmul_precision)
    seed = config.seed + rank * 1_000_003
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    output_dir = Path(config.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(output_dir / "resolved_config.json", asdict(config))

    model: nn.Module = ResidualClosureNet(config.model_config()).to(device)
    if config.compile_model:
        model = torch.compile(model)  # type: ignore[assignment]
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    start_step = 0
    if resume:
        checkpoint = torch.load(resume, map_location=device, weights_only=False)
        _unwrap(model).load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["step"])

    if rank == 0:
        print(
            json.dumps(
                {
                    "event": "start",
                    "device": str(device),
                    "world_size": world_size,
                    "architecture": architecture_dict(_unwrap(model)),
                    "output_dir": str(output_dir),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        validation = _build_validation_set(config, device)
    else:
        validation = []
    if world_size > 1:
        dist.barrier()

    weight_generator = _generator(device, seed + 10_000_001)
    input_generator = _generator(device, seed + 20_000_001)
    metrics_path = output_dir / "metrics.jsonl"
    best_validation = math.inf
    interval_start = time.perf_counter()

    for step in range(start_step + 1, config.steps + 1):
        learning_rate = _learning_rate(config, step)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate

        weights = sample_he_weights(
            config.batch_size,
            config.depth,
            config.width,
            device=device,
            generator=weight_generator,
        )
        targets = monte_carlo_targets(
            weights,
            config.mc_samples,
            config.mc_chunk_size,
            antithetic=config.antithetic,
            generator=input_generator,
        )
        prediction, baseline = model(weights)
        loss, metrics = _losses(
            prediction, baseline, targets, config.auxiliary_loss_weight
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        if rank == 0 and (step == 1 or step % config.log_every == 0):
            now = time.perf_counter()
            record = {
                "event": "train",
                "step": step,
                "learning_rate": learning_rate,
                "grad_norm": float(grad_norm),
                "seconds_per_log_interval": now - interval_start,
                **metrics,
            }
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            print(json.dumps(record, sort_keys=True), flush=True)
            interval_start = now

        should_validate = step % config.validate_every == 0 or step == config.steps
        if should_validate:
            if world_size > 1:
                dist.barrier()
            if rank == 0:
                validation_metrics = _validate(
                    model, validation, device, config.auxiliary_loss_weight
                )
                record = {"event": "validation", "step": step, **validation_metrics}
                with metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                print(json.dumps(record, sort_keys=True), flush=True)
                if validation_metrics["final_mse"] < best_validation:
                    best_validation = validation_metrics["final_mse"]
                    _save_checkpoint(
                        model, optimizer, config, step, output_dir / "checkpoint_best.pt"
                    )
                    export_model(_unwrap(model), output_dir / "learned_residual_weights_best.npz")
            if world_size > 1:
                dist.barrier()

        if rank == 0 and (
            step % config.checkpoint_every == 0 or step == config.steps
        ):
            _save_checkpoint(model, optimizer, config, step, output_dir / "checkpoint_last.pt")
            export_model(_unwrap(model), output_dir / "learned_residual_weights_last.npz")

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()
    return output_dir / "learned_residual_weights_best.npz"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="JSON TrainConfig file")
    parser.add_argument("--output-dir", help="override config output_dir")
    parser.add_argument("--steps", type=int, help="override config steps")
    parser.add_argument("--device", help="override config device")
    parser.add_argument("--resume", help="checkpoint_last.pt to resume")
    args = parser.parse_args()

    config = TrainConfig.from_json(args.config)
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    if args.steps is not None:
        config.steps = args.steps
    if args.device is not None:
        config.device = args.device
    train(config, resume=args.resume)


if __name__ == "__main__":
    main()
