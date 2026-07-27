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
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel

from .model import ModelConfig, ResidualClosureNet, architecture_dict, export_model


_SQRT_2 = math.sqrt(2.0)


@dataclass
class TrainConfig:
    seed: int = 20260721
    width: int = 256
    depth: int = 32
    hidden_dim: int = 64
    blocks: int = 3
    residual_scale: float = 0.50
    final_calibration: float = 0.9921
    steps: int = 1_000_000
    max_train_seconds: float | None = 43_200.0
    batch_size: int = 8
    mc_samples: int = 4_096
    mc_chunk_size: int = 1_024
    antithetic: bool = True
    randomized_lattice_teacher: bool = True
    radial_rao_blackwell: bool = True
    exact_first_layer_target: bool = True
    learning_rate: float = 2e-4
    min_learning_rate: float = 1e-5
    warmup_steps: int = 500
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    auxiliary_loss_weight: float = 0.05
    ema_decay: float = 0.999
    log_every: int = 20
    validate_every: int = 250
    checkpoint_every: int = 500
    full100_eval_every: int = 25_000
    full100_eval_n_mlps: int = 100
    full100_eval_at_start: bool = True
    val_networks: int = 128
    val_batch_size: int = 4
    val_mc_samples: int = 65_536
    val_mc_chunk_size: int = 2_048
    qualification_final_mse: float = 3.5e-5
    leaderboard_final_mse: float = 3.5e-6
    device: str = "auto"
    matmul_precision: str = "highest"
    compile_model: bool = False
    output_dir: str = "outputs/whestbench_learned_residual/production"

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
            final_calibration=self.final_calibration,
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
            "qualification_final_mse": self.qualification_final_mse,
            "leaderboard_final_mse": self.leaderboard_final_mse,
        }
        bad = [name for name, value in positive.items() if value <= 0]
        if bad:
            raise ValueError(f"these fields must be positive: {bad}")
        if self.val_mc_samples % 2:
            raise ValueError("val_mc_samples must be even for paired validation")
        if self.antithetic and (self.mc_samples % 2 or self.val_mc_samples % 4):
            raise ValueError(
                "mc_samples must be even and val_mc_samples divisible by four "
                "when antithetic=true"
            )
        if self.min_learning_rate < 0.0 or self.min_learning_rate > self.learning_rate:
            raise ValueError("min_learning_rate must be in [0, learning_rate]")
        if self.warmup_steps < 0 or self.warmup_steps >= self.steps:
            raise ValueError("warmup_steps must be in [0, steps)")
        if self.log_every <= 0:
            raise ValueError("log_every must be positive")
        if self.max_train_seconds is not None and self.max_train_seconds <= 0.0:
            raise ValueError("max_train_seconds must be positive or null")
        if self.full100_eval_every < 0:
            raise ValueError("full100_eval_every must be non-negative")
        if self.full100_eval_every and self.full100_eval_n_mlps <= 0:
            raise ValueError("full100_eval_n_mlps must be positive when enabled")
        if self.grad_clip <= 0.0:
            raise ValueError("grad_clip must be positive")
        if self.weight_decay < 0.0 or self.auxiliary_loss_weight < 0.0:
            raise ValueError("weight_decay and auxiliary_loss_weight must be non-negative")
        if not 0.0 <= self.ema_decay < 1.0:
            raise ValueError("ema_decay must be in [0, 1)")
        if self.matmul_precision not in {"highest", "high", "medium"}:
            raise ValueError("matmul_precision must be highest, high, or medium")


@dataclass
class ValidationBatch:
    weights: Tensor
    targets: Tensor
    teacher_noise_final_mse: float


@dataclass
class Full100Monitor:
    step: int
    process: subprocess.Popen[Any]
    log_handle: Any
    report_path: Path
    snapshot_path: Path


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


def _mean_chi(dimension: int) -> float:
    return math.exp(
        0.5 * math.log(2.0)
        + math.lgamma(0.5 * (dimension + 1))
        - math.lgamma(0.5 * dimension)
    )


@lru_cache(maxsize=None)
def _first_primes(count: int) -> tuple[int, ...]:
    primes: list[int] = []
    candidate = 2
    while len(primes) < count:
        is_prime = True
        for prime in primes:
            if prime * prime > candidate:
                break
            if candidate % prime == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(candidate)
        candidate += 1
    return tuple(primes)


def _lattice_normal_samples(
    batch: int,
    start: int,
    count: int,
    width: int,
    *,
    device: torch.device,
    shift: Tensor,
) -> Tensor:
    """Randomly shifted rank-1 lattice points mapped through the normal PPF."""

    roots = torch.sqrt(
        torch.tensor(_first_primes(width), dtype=torch.float64, device=device)
    )
    lattice_generator = roots - torch.floor(roots)
    indices = torch.arange(
        start, start + count, dtype=torch.float64, device=device
    )
    uniforms = torch.frac(
        indices[None, :, None] * lattice_generator[None, None, :]
        + shift[:, None, :]
    ).clamp(1e-7, 1.0 - 1e-7)
    return (_SQRT_2 * torch.erfinv(2.0 * uniforms - 1.0)).to(torch.float32)


def _radial_rao_blackwellize(samples: Tensor) -> Tensor:
    """Replace sampled Gaussian radii by E[chi_d] without changing the mean."""

    radius = torch.linalg.vector_norm(samples, dim=-1, keepdim=True).clamp_min(1e-12)
    return samples * (_mean_chi(samples.shape[-1]) / radius)


@torch.inference_mode()
def monte_carlo_targets(
    weights: Tensor,
    n_samples: int,
    chunk_size: int,
    *,
    antithetic: bool,
    generator: torch.Generator | None,
    randomized_lattice: bool = False,
    radial_rao_blackwell: bool = False,
    exact_first_layer: bool = False,
) -> Tensor:
    """Estimate every layer mean with optional unbiased variance reduction."""

    if n_samples <= 0 or chunk_size <= 0:
        raise ValueError("n_samples and chunk_size must be positive")
    if antithetic and n_samples % 2:
        raise ValueError("n_samples must be even for antithetic sampling")
    batch, depth, width, _ = weights.shape
    sums = torch.zeros((depth, batch, width), device=weights.device, dtype=torch.float32)
    samples_done = 0
    lattice_shift = None
    if randomized_lattice:
        lattice_shift = torch.rand(
            (batch, width),
            device=weights.device,
            dtype=torch.float64,
            generator=generator,
        )

    if antithetic:
        independent_total = n_samples // 2
        independent_done = 0
        while independent_done < independent_total:
            independent = min(max(1, chunk_size // 2), independent_total - independent_done)
            if lattice_shift is None:
                positive = _randn(
                    (batch, independent, width), weights.device, generator
                )
            else:
                positive = _lattice_normal_samples(
                    batch,
                    independent_done,
                    independent,
                    width,
                    device=weights.device,
                    shift=lattice_shift,
                )
            if radial_rao_blackwell:
                positive = _radial_rao_blackwellize(positive)
            activations = torch.cat((positive, -positive), dim=1)
            for layer_index in range(depth):
                activations = torch.relu(torch.bmm(activations, weights[:, layer_index]))
                sums[layer_index].add_(activations.sum(dim=1))
            independent_done += independent
            samples_done += 2 * independent
    else:
        while samples_done < n_samples:
            current = min(chunk_size, n_samples - samples_done)
            if lattice_shift is None:
                activations = _randn(
                    (batch, current, width), weights.device, generator
                )
            else:
                activations = _lattice_normal_samples(
                    batch,
                    samples_done,
                    current,
                    width,
                    device=weights.device,
                    shift=lattice_shift,
                )
            if radial_rao_blackwell:
                activations = _radial_rao_blackwellize(activations)
            for layer_index in range(depth):
                activations = torch.relu(torch.bmm(activations, weights[:, layer_index]))
                sums[layer_index].add_(activations.sum(dim=1))
            samples_done += current

    result = (sums / float(samples_done)).permute(1, 0, 2).contiguous()
    if exact_first_layer:
        result[:, 0] = torch.sqrt(weights[:, 0].square().sum(dim=1)) / math.sqrt(
            2.0 * math.pi
        )
    return result


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


def _learning_rate(
    config: TrainConfig, step: int, elapsed_train_seconds: float = 0.0
) -> float:
    if step <= config.warmup_steps and config.warmup_steps > 0:
        return config.learning_rate * step / config.warmup_steps
    if config.max_train_seconds is not None:
        progress = elapsed_train_seconds / config.max_train_seconds
    else:
        progress = (step - config.warmup_steps) / max(
            config.steps - config.warmup_steps, 1
        )
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
) -> list[ValidationBatch]:
    weight_generator = _generator(device, config.seed + 80_000_001)
    input_generator_a = _generator(device, config.seed + 90_000_001)
    input_generator_b = _generator(device, config.seed + 91_000_001)
    batches: list[ValidationBatch] = []
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
        targets_a = monte_carlo_targets(
            weights,
            config.val_mc_samples // 2,
            config.val_mc_chunk_size,
            antithetic=config.antithetic,
            generator=input_generator_a,
            randomized_lattice=config.randomized_lattice_teacher,
            radial_rao_blackwell=config.radial_rao_blackwell,
            exact_first_layer=config.exact_first_layer_target,
        )
        targets_b = monte_carlo_targets(
            weights,
            config.val_mc_samples // 2,
            config.val_mc_chunk_size,
            antithetic=config.antithetic,
            generator=input_generator_b,
            randomized_lattice=config.randomized_lattice_teacher,
            radial_rao_blackwell=config.radial_rao_blackwell,
            exact_first_layer=config.exact_first_layer_target,
        )
        # If each half estimate has variance v, their average has variance v/2
        # while E[(a-b)^2] = 2v, hence the factor 1/4.
        teacher_noise = float(
            (targets_a[:, -1] - targets_b[:, -1]).square().mean() * 0.25
        )
        batches.append(
            ValidationBatch(
                weights=weights.cpu(),
                targets=(0.5 * (targets_a + targets_b)).cpu(),
                teacher_noise_final_mse=teacher_noise,
            )
        )
        remaining -= current
    return batches


@torch.inference_mode()
def _validate(
    model: nn.Module,
    validation: list[ValidationBatch],
    device: torch.device,
    config: TrainConfig,
) -> dict[str, float]:
    raw_model = _unwrap(model)
    was_training = raw_model.training
    raw_model.eval()
    totals = {"loss": 0.0, "final_mse": 0.0, "baseline_final_mse": 0.0, "auxiliary_mse": 0.0}
    oracle_bound_total = 0.0
    saturation_total = 0.0
    teacher_noise_total = 0.0
    baseline_finals: list[Tensor] = []
    target_finals: list[Tensor] = []
    count = 0
    for batch in validation:
        weights = batch.weights.to(device)
        targets = batch.targets.to(device)
        prediction, baseline, correction_limit = raw_model(weights)
        _, metrics = _losses(
            prediction, baseline, targets, config.auxiliary_loss_weight
        )
        batch_size = weights.shape[0]
        for name in totals:
            totals[name] += metrics[name] * batch_size
        residual = targets[:, -1] - baseline[:, -1]
        limit = correction_limit[:, -1]
        oracle = (baseline[:, -1] + torch.clamp(residual, -limit, limit)).clamp_min(0.0)
        oracle_bound_total += float((oracle - targets[:, -1]).square().mean()) * batch_size
        saturation_total += float((residual.abs() > limit).float().mean()) * batch_size
        teacher_noise_total += batch.teacher_noise_final_mse * batch_size
        baseline_finals.append(baseline[:, -1].detach().cpu().flatten())
        target_finals.append(targets[:, -1].detach().cpu().flatten())
        count += batch_size
    raw_model.train(was_training)
    result = {name: value / count for name, value in totals.items()}
    baseline_final = torch.cat(baseline_finals).double()
    target_final = torch.cat(target_finals).double()
    scale = torch.dot(baseline_final, target_final) / torch.dot(
        baseline_final, baseline_final
    ).clamp_min(1e-24)
    design = torch.stack((baseline_final, torch.ones_like(baseline_final)), dim=1)
    affine = torch.linalg.lstsq(design, target_final.unsqueeze(1)).solution[:, 0]
    scale_mse = (scale * baseline_final - target_final).square().mean()
    affine_mse = (design @ affine - target_final).square().mean()
    result.update(
        {
            "oracle_bound_final_mse": oracle_bound_total / count,
            "bound_saturation_fraction": saturation_total / count,
            "teacher_noise_final_mse": teacher_noise_total / count,
            "oracle_global_scale": float(scale),
            "oracle_global_scale_final_mse": float(scale_mse),
            "oracle_affine_scale": float(affine[0]),
            "oracle_affine_bias": float(affine[1]),
            "oracle_affine_final_mse": float(affine_mse),
            "relative_to_baseline": result["final_mse"]
            / max(result["baseline_final_mse"], 1e-24),
            "qualification_gate_pass": float(
                result["final_mse"] <= config.qualification_final_mse
            ),
            "leaderboard_gate_pass": float(
                result["final_mse"] <= config.leaderboard_final_mse
            ),
        }
    )
    return result


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _save_checkpoint(
    model: nn.Module,
    ema_model: ResidualClosureNet,
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    step: int,
    best_validation: float,
    elapsed_train_seconds: float,
    path: Path,
) -> None:
    cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.save(
        {
            "step": step,
            "best_validation": best_validation,
            "elapsed_train_seconds": elapsed_train_seconds,
            "config": asdict(config),
            "model": _unwrap(model).state_dict(),
            "ema_model": ema_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": cuda_rng_state,
        },
        path,
    )


@torch.inference_mode()
def _update_ema(
    ema_model: ResidualClosureNet,
    model: nn.Module,
    configured_decay: float,
    step: int,
) -> None:
    decay = min(configured_decay, (step + 1.0) / (step + 10.0))
    raw_model = _unwrap(model)
    for ema_parameter, parameter in zip(
        ema_model.parameters(), raw_model.parameters(), strict=True
    ):
        ema_parameter.mul_(decay).add_(parameter, alpha=1.0 - decay)


def _step_generator(
    device: torch.device, base_seed: int, step: int
) -> torch.Generator | None:
    """Make online data a deterministic function of (rank, step)."""

    return _generator(device, base_seed + step * 1_000_000_007)


def _append_metric(metrics_path: Path, record: dict[str, Any]) -> None:
    with metrics_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    print(json.dumps(record, sort_keys=True), flush=True)


def _full100_environment(repo_root: Path) -> dict[str, str]:
    environment = os.environ.copy()
    deps_path = environment.get("WHEST_DEPS_PATH", "").strip()
    if not deps_path:
        for candidate in (Path("/tmp/whest-official-deps"), Path("/tmp/whest-test-deps")):
            if (candidate / "flopscope").is_dir() and (candidate / "whestbench").is_dir():
                deps_path = str(candidate)
                break
    python_path = [part for part in (deps_path, str(repo_root)) if part]
    if environment.get("PYTHONPATH"):
        python_path.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(python_path)
    environment.setdefault("HF_HOME", "/tmp/hf-whest-cache")
    environment["CUDA_VISIBLE_DEVICES"] = ""
    for name in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        environment[name] = "1"
    return environment


def _launch_full100_monitor(
    model: ResidualClosureNet,
    config: TrainConfig,
    output_dir: Path,
    metrics_path: Path,
    step: int,
) -> Full100Monitor | None:
    repo_root = Path(__file__).resolve().parents[3]
    step_dir = output_dir / "full100" / f"step_{step:07d}"
    step_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = export_model(model, step_dir / "learned_residual_weights.npz")
    report_path = step_dir / "report.json"
    log_path = step_dir / "evaluator.log"
    command = [
        sys.executable,
        str(repo_root / "repro/aicrowd_whestbench/evaluate_full100_direct.py"),
        str(
            repo_root
            / "repro_external/aicrowd_whestbench/learned_residual/submission/estimator.py"
        ),
        str(report_path),
        "--n-mlps",
        str(config.full100_eval_n_mlps),
        "--asset",
        str(snapshot_path),
    ]
    log_handle = log_path.open("w", encoding="utf-8")
    try:
        process = subprocess.Popen(
            command,
            cwd=repo_root,
            env=_full100_environment(repo_root),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
    except Exception as exc:
        log_handle.close()
        _append_metric(
            metrics_path,
            {
                "event": "full100_monitor_error",
                "step": step,
                "error": repr(exc),
                "log_path": str(log_path),
            },
        )
        return None
    _append_metric(
        metrics_path,
        {
            "event": "full100_monitor_start",
            "step": step,
            "pid": process.pid,
            "report_path": str(report_path),
            "log_path": str(log_path),
        },
    )
    return Full100Monitor(step, process, log_handle, report_path, snapshot_path)


def _finish_full100_monitor(
    monitor: Full100Monitor | None,
    metrics_path: Path,
    *,
    wait: bool,
) -> Full100Monitor | None:
    if monitor is None:
        return None
    return_code = monitor.process.wait() if wait else monitor.process.poll()
    if return_code is None:
        return monitor
    monitor.log_handle.close()
    record: dict[str, Any] = {
        "event": "full100_monitor",
        "step": monitor.step,
        "return_code": return_code,
        "report_path": str(monitor.report_path),
    }
    if return_code == 0 and monitor.report_path.exists():
        summary = json.loads(monitor.report_path.read_text(encoding="utf-8"))["summary"]
        record.update({f"full100_{name}": value for name, value in summary.items()})
    else:
        record["status"] = "failed_monitor_only"
    _append_metric(metrics_path, record)
    return None


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
    ema_model = ResidualClosureNet(config.model_config()).to(device)
    ema_model.load_state_dict(_unwrap(model).state_dict())
    ema_model.eval()
    start_step = 0
    best_validation = math.inf
    elapsed_before_resume = 0.0
    if resume:
        checkpoint = torch.load(resume, map_location=device, weights_only=False)
        checkpoint_model_config = TrainConfig(**checkpoint["config"]).model_config()
        if checkpoint_model_config != config.model_config():
            raise ValueError(
                "resume checkpoint architecture does not match the requested config"
            )
        _unwrap(model).load_state_dict(checkpoint["model"])
        ema_model.load_state_dict(checkpoint.get("ema_model", checkpoint["model"]))
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["step"])
        best_validation = float(checkpoint.get("best_validation", math.inf))
        elapsed_before_resume = float(checkpoint.get("elapsed_train_seconds", 0.0))
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if torch.cuda.is_available() and checkpoint.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(
                [state.cpu() for state in checkpoint["cuda_rng_state"]]
            )

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

    metrics_path = output_dir / "metrics.jsonl"
    interval_start = time.perf_counter()

    # Preserve the calibrated deterministic closure before any noisy SGD step.
    # Production training is only allowed to improve this artifact.
    if start_step == 0:
        if rank == 0:
            initial_metrics = _validate(ema_model, validation, device, config)
            best_validation = initial_metrics["final_mse"]
            record = {"event": "validation", "step": 0, **initial_metrics}
            _append_metric(metrics_path, record)
            _save_checkpoint(
                model,
                ema_model,
                optimizer,
                config,
                0,
                best_validation,
                0.0,
                output_dir / "checkpoint_best.pt",
            )
            export_model(ema_model, output_dir / "learned_residual_weights_best.npz")
        if world_size > 1:
            dist.barrier()

    active_full100: Full100Monitor | None = None
    full100_launched_steps: set[int] = set()
    if (
        rank == 0
        and config.full100_eval_every
        and config.full100_eval_at_start
        and start_step == 0
    ):
        active_full100 = _launch_full100_monitor(
            ema_model, config, output_dir, metrics_path, 0
        )
        if active_full100 is not None:
            full100_launched_steps.add(0)

    training_started_at = time.perf_counter()
    elapsed_train_seconds = elapsed_before_resume
    last_step = start_step
    for step in range(start_step + 1, config.steps + 1):
        learning_rate = _learning_rate(config, step, elapsed_train_seconds)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate

        weights = sample_he_weights(
            config.batch_size,
            config.depth,
            config.width,
            device=device,
            generator=_step_generator(device, seed + 10_000_001, step),
        )
        targets = monte_carlo_targets(
            weights,
            config.mc_samples,
            config.mc_chunk_size,
            antithetic=config.antithetic,
            generator=_step_generator(device, seed + 20_000_001, step),
            randomized_lattice=config.randomized_lattice_teacher,
            radial_rao_blackwell=config.radial_rao_blackwell,
            exact_first_layer=config.exact_first_layer_target,
        )
        prediction, baseline, _ = model(weights)
        loss, metrics = _losses(
            prediction, baseline, targets, config.auxiliary_loss_weight
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        _update_ema(ema_model, model, config.ema_decay, step)
        last_step = step

        rank_zero_elapsed = elapsed_before_resume + (
            time.perf_counter() - training_started_at
        )
        if world_size > 1:
            elapsed_tensor = torch.tensor(
                [rank_zero_elapsed if rank == 0 else 0.0],
                dtype=torch.float64,
                device=device,
            )
            dist.broadcast(elapsed_tensor, src=0)
            elapsed_train_seconds = float(elapsed_tensor.item())
        else:
            elapsed_train_seconds = rank_zero_elapsed
        time_limit_reached = (
            config.max_train_seconds is not None
            and elapsed_train_seconds >= config.max_train_seconds
        )

        if rank == 0 and (step == 1 or step % config.log_every == 0):
            now = time.perf_counter()
            record = {
                "event": "train",
                "step": step,
                "learning_rate": learning_rate,
                "grad_norm": float(grad_norm),
                "elapsed_train_seconds": elapsed_train_seconds,
                "seconds_per_log_interval": now - interval_start,
                **metrics,
            }
            _append_metric(metrics_path, record)
            interval_start = now
            active_full100 = _finish_full100_monitor(
                active_full100, metrics_path, wait=False
            )

        should_validate = (
            step % config.validate_every == 0
            or step == config.steps
            or time_limit_reached
        )
        if should_validate:
            if world_size > 1:
                dist.barrier()
            if rank == 0:
                validation_metrics = _validate(
                    ema_model, validation, device, config
                )
                record = {"event": "validation", "step": step, **validation_metrics}
                _append_metric(metrics_path, record)
                if validation_metrics["final_mse"] < best_validation:
                    best_validation = validation_metrics["final_mse"]
                    _save_checkpoint(
                        model,
                        ema_model,
                        optimizer,
                        config,
                        step,
                        best_validation,
                        elapsed_train_seconds,
                        output_dir / "checkpoint_best.pt",
                    )
                    export_model(ema_model, output_dir / "learned_residual_weights_best.npz")
            if world_size > 1:
                dist.barrier()

        if rank == 0 and (
            step % config.checkpoint_every == 0
            or step == config.steps
            or time_limit_reached
        ):
            _save_checkpoint(
                model,
                ema_model,
                optimizer,
                config,
                step,
                best_validation,
                elapsed_train_seconds,
                output_dir / "checkpoint_last.pt",
            )
            export_model(ema_model, output_dir / "learned_residual_weights_last.npz")

        if rank == 0 and config.full100_eval_every and (
            step % config.full100_eval_every == 0
        ):
            active_full100 = _finish_full100_monitor(
                active_full100, metrics_path, wait=False
            )
            if active_full100 is None:
                active_full100 = _launch_full100_monitor(
                    ema_model, config, output_dir, metrics_path, step
                )
                if active_full100 is not None:
                    full100_launched_steps.add(step)
            else:
                _append_metric(
                    metrics_path,
                    {
                        "event": "full100_monitor_skipped",
                        "step": step,
                        "reason": "previous_monitor_still_running",
                        "active_step": active_full100.step,
                    },
                )

        if time_limit_reached:
            if rank == 0:
                _append_metric(
                    metrics_path,
                    {
                        "event": "time_limit_reached",
                        "step": step,
                        "elapsed_train_seconds": elapsed_train_seconds,
                    },
                )
            break

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()
    if rank == 0 and config.full100_eval_every:
        active_full100 = _finish_full100_monitor(
            active_full100, metrics_path, wait=True
        )
        if last_step not in full100_launched_steps:
            active_full100 = _launch_full100_monitor(
                ema_model, config, output_dir, metrics_path, last_step
            )
            if active_full100 is not None:
                full100_launched_steps.add(last_step)
                _finish_full100_monitor(active_full100, metrics_path, wait=True)
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
