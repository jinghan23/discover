"""PyTorch model and export helpers for the learned WhestBench estimator.

The model keeps diagonal Gaussian mean/variance propagation as a deterministic
base estimator.  A small permutation-equivariant message-passing network sees
the same weights and analytic states, and predicts a bounded residual for each
post-ReLU mean.  Parameters are shared across depth, so the model cannot attach
special meaning to a particular neuron index.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn


_EPS = 1e-12
_FEATURE_DIM = 10
_SQRT_2 = 2.0**0.5
_SQRT_2PI = (2.0 * torch.pi) ** 0.5
_FORMAT_VERSION = 1


@dataclass(frozen=True)
class ModelConfig:
    """Architecture fields needed by both training and submission inference."""

    width: int = 256
    depth: int = 32
    hidden_dim: int = 32
    blocks: int = 2
    residual_scale: float = 0.10

    def validate(self) -> None:
        if self.width <= 0 or self.depth <= 0:
            raise ValueError("width and depth must be positive")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.blocks < 0:
            raise ValueError("blocks must be non-negative")
        if not 0.0 < self.residual_scale <= 1.0:
            raise ValueError("residual_scale must be in (0, 1]")


def _normal_pdf(value: Tensor) -> Tensor:
    return torch.exp(-0.5 * value.square()) / _SQRT_2PI


def _normal_cdf(value: Tensor) -> Tensor:
    return 0.5 * (1.0 + torch.erf(value / _SQRT_2))


def diagonal_gaussian_step(mean: Tensor, variance: Tensor, weights: Tensor) -> tuple[Tensor, ...]:
    """Propagate diagonal moments through one linear-ReLU layer.

    Args:
        mean: ``(batch, width)`` input means.
        variance: ``(batch, width)`` input marginal variances.
        weights: ``(batch, width, width)`` weight matrices using ``x @ W``.
    """

    pre_mean = torch.bmm(mean.unsqueeze(1), weights).squeeze(1)
    pre_variance = torch.bmm(
        variance.unsqueeze(1), weights.square()
    ).squeeze(1).clamp_min(_EPS)
    pre_std = torch.sqrt(pre_variance)
    alpha = pre_mean / pre_std
    density = _normal_pdf(alpha)
    probability = _normal_cdf(alpha)

    output_mean = pre_mean * probability + pre_std * density
    second_moment = (
        (pre_mean.square() + pre_variance) * probability
        + pre_mean * pre_std * density
    )
    output_variance = (second_moment - output_mean.square()).clamp_min(0.0)
    return output_mean, output_variance, pre_mean, pre_std, alpha


def analytic_features(
    input_mean: Tensor,
    input_variance: Tensor,
    output_mean: Tensor,
    output_variance: Tensor,
    pre_std: Tensor,
    alpha: Tensor,
    weights: Tensor,
    layer_fraction: float,
) -> Tensor:
    """Build stable, dimensionless per-neuron features."""

    input_rms = torch.sqrt(
        torch.mean(input_mean.square() + input_variance, dim=1, keepdim=True).clamp_min(_EPS)
    )
    column_energy = weights.square().sum(dim=1).clamp_min(_EPS)
    column_sum = weights.sum(dim=1)
    fraction = torch.full_like(output_mean, float(layer_fraction))

    features = (
        (input_mean / input_rms).clamp(-8.0, 8.0) / 4.0,
        (torch.sqrt(input_variance.clamp_min(0.0)) / input_rms).clamp(0.0, 8.0) / 4.0,
        alpha.clamp(-8.0, 8.0) / 4.0,
        (output_mean / pre_std).clamp(0.0, 8.0) / 4.0,
        (torch.sqrt(output_variance) / pre_std).clamp(0.0, 8.0) / 4.0,
        torch.log((pre_std / input_rms).clamp_min(_EPS)).clamp(-6.0, 6.0) / 3.0,
        column_sum.clamp(-4.0 * _SQRT_2, 4.0 * _SQRT_2) / (4.0 * _SQRT_2),
        (8.0 * (torch.sqrt(0.5 * column_energy) - 1.0)).clamp(-4.0, 4.0) / 4.0,
        fraction,
        fraction.square(),
    )
    return torch.stack(features, dim=-1)


class ResidualClosureNet(nn.Module):
    """Shared forward message-passing network over MLP neurons.

    The head predicts a number in ``[-1, 1]``.  At layer ``l`` it is converted
    to ``residual_scale * (l / (depth - 1)) * pre_std`` and added to the
    analytic mean.  Consequently the exact first-layer Gaussian expectation is
    never changed, and an untrained/zero-head model is exactly the analytic
    baseline.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        config.validate()
        self.config = config
        input_dim = _FEATURE_DIM + 3 * config.hidden_dim
        self.input_projection = nn.Linear(input_dim, config.hidden_dim)
        self.update_blocks = nn.ModuleList(
            nn.Linear(config.hidden_dim, config.hidden_dim) for _ in range(config.blocks)
        )
        self.residual_head = nn.Linear(config.hidden_dim, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.input_projection.weight)
        nn.init.zeros_(self.input_projection.bias)
        for layer in self.update_blocks:
            nn.init.orthogonal_(layer.weight, gain=0.5)
            nn.init.zeros_(layer.bias)
        # Starting from the deterministic baseline makes optimization and
        # regression tests substantially easier to interpret.
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    def forward(self, weights: Tensor) -> tuple[Tensor, Tensor]:
        if weights.ndim != 4:
            raise ValueError("weights must have shape (batch, depth, width, width)")
        batch, depth, width, width_again = weights.shape
        if depth != self.config.depth or width != self.config.width or width_again != width:
            raise ValueError(
                f"expected (*, {self.config.depth}, {self.config.width}, "
                f"{self.config.width}), got {tuple(weights.shape)}"
            )

        mean = torch.zeros((batch, width), dtype=weights.dtype, device=weights.device)
        variance = torch.ones_like(mean)
        state = torch.zeros(
            (batch, width, self.config.hidden_dim),
            dtype=weights.dtype,
            device=weights.device,
        )
        predictions: list[Tensor] = []
        baselines: list[Tensor] = []

        for layer_index in range(depth):
            layer_weights = weights[:, layer_index]
            signed_message = torch.bmm(layer_weights.transpose(1, 2), state) / _SQRT_2
            energy_message = torch.bmm(
                layer_weights.square().transpose(1, 2), state
            ) * 0.5
            global_context = state.mean(dim=1, keepdim=True).expand_as(state)

            input_mean = mean
            input_variance = variance
            mean, variance, _, pre_std, alpha = diagonal_gaussian_step(
                input_mean, input_variance, layer_weights
            )
            layer_fraction = layer_index / max(depth - 1, 1)
            features = analytic_features(
                input_mean,
                input_variance,
                mean,
                variance,
                pre_std,
                alpha,
                layer_weights,
                layer_fraction,
            )
            network_input = torch.cat(
                (features, signed_message, energy_message, global_context), dim=-1
            )
            state = torch.tanh(self.input_projection(network_input))
            for block in self.update_blocks:
                state = state + 0.5 * torch.tanh(block(state))

            standardized_residual = torch.tanh(self.residual_head(state).squeeze(-1))
            correction = (
                self.config.residual_scale
                * layer_fraction
                * pre_std
                * standardized_residual
            )
            predictions.append((mean + correction).clamp_min(0.0))
            baselines.append(mean)

        return torch.stack(predictions, dim=1), torch.stack(baselines, dim=1)


def parameter_count(config: ModelConfig) -> int:
    """Return the exact number of learned scalar parameters."""

    return sum(parameter.numel() for parameter in ResidualClosureNet(config).parameters())


def export_model(model: ResidualClosureNet, output_path: str | Path) -> Path:
    """Export a pickle-free archive consumed by the flopscope estimator."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    config = model.config
    block_weights = np.stack(
        [layer.weight.detach().cpu().numpy().T for layer in model.update_blocks], axis=0
    ) if model.update_blocks else np.empty((0, config.hidden_dim, config.hidden_dim), np.float32)
    block_biases = np.stack(
        [layer.bias.detach().cpu().numpy() for layer in model.update_blocks], axis=0
    ) if model.update_blocks else np.empty((0, config.hidden_dim), np.float32)

    arrays: dict[str, Any] = {
        "format_version": np.asarray(_FORMAT_VERSION, dtype=np.int64),
        "width": np.asarray(config.width, dtype=np.int64),
        "depth": np.asarray(config.depth, dtype=np.int64),
        "hidden_dim": np.asarray(config.hidden_dim, dtype=np.int64),
        "blocks": np.asarray(config.blocks, dtype=np.int64),
        "residual_scale": np.asarray(config.residual_scale, dtype=np.float32),
        "feature_dim": np.asarray(_FEATURE_DIM, dtype=np.int64),
        # Export transposed Linear weights so inference is simply x @ W.
        "input_weight": model.input_projection.weight.detach().cpu().numpy().T,
        "input_bias": model.input_projection.bias.detach().cpu().numpy(),
        "block_weights": block_weights,
        "block_biases": block_biases,
        "output_weight": model.residual_head.weight.detach().cpu().numpy().T,
        "output_bias": model.residual_head.bias.detach().cpu().numpy(),
    }
    np.savez(output_path, **arrays)
    return output_path


def architecture_dict(model: ResidualClosureNet) -> dict[str, Any]:
    return {**asdict(model.config), "parameters": sum(p.numel() for p in model.parameters())}
