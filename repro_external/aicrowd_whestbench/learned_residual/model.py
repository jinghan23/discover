"""Competitive learned closure model for WhestBench.

The deterministic base estimator propagates a dense Gaussian covariance.  The
first ReLU covariance is exact (the input is jointly Gaussian and centred),
and later layers use the non-central bivariate Gaussian closure.  A compact
permutation-equivariant network then predicts the remaining non-Gaussian
residual.  The learned state receives both ordinary weight messages and a
covariance-aware relational message.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn


_EPS = 1e-12
_MAX_ABS_CORRELATION = 1.0 - 1e-6
_FEATURE_DIM = 13
_SQRT_2 = 2.0**0.5
_SQRT_2PI = (2.0 * torch.pi) ** 0.5
_TWO_PI = 2.0 * torch.pi
_FORMAT_VERSION = 2

# Eight-point Gauss--Legendre quadrature on [0, 1].  Plackett's identity
# expresses the bivariate normal CDF as an integral over correlation.
_QUADRATURE_NODES = (
    0.019855071751231884,
    0.10166676129318664,
    0.2372337950418355,
    0.4082826787521751,
    0.5917173212478248,
    0.7627662049581645,
    0.8983332387068134,
    0.9801449282487681,
)
_QUADRATURE_WEIGHTS = (
    0.05061426814518813,
    0.11119051722668724,
    0.15685332293894366,
    0.181341891689181,
    0.181341891689181,
    0.15685332293894366,
    0.11119051722668724,
    0.05061426814518813,
)


@dataclass(frozen=True)
class ModelConfig:
    """Architecture fields needed by training and submission inference."""

    width: int = 256
    depth: int = 32
    hidden_dim: int = 64
    blocks: int = 3
    residual_scale: float = 0.50
    final_calibration: float = 0.9921

    def validate(self) -> None:
        if self.width <= 0 or self.depth <= 0:
            raise ValueError("width and depth must be positive")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.blocks < 0:
            raise ValueError("blocks must be non-negative")
        if not 0.0 < self.residual_scale <= 2.0:
            raise ValueError("residual_scale must be in (0, 2]")
        if not 0.9 <= self.final_calibration <= 1.1:
            raise ValueError("final_calibration must be in [0.9, 1.1]")


def _normal_pdf(value: Tensor) -> Tensor:
    return torch.exp(-0.5 * value.square()) / _SQRT_2PI


def _normal_cdf(value: Tensor) -> Tensor:
    return 0.5 * (1.0 + torch.erf(value / _SQRT_2))


def _replace_diagonal(matrix: Tensor, diagonal: Tensor) -> Tensor:
    old_diagonal = torch.diagonal(matrix, dim1=-2, dim2=-1)
    return matrix + torch.diag_embed(diagonal - old_diagonal)


def _correlation(covariance: Tensor, standard_deviation: Tensor) -> Tensor:
    scale = standard_deviation.unsqueeze(-1) * standard_deviation.unsqueeze(-2)
    return (covariance / scale.clamp_min(_EPS)).clamp(
        -_MAX_ABS_CORRELATION, _MAX_ABS_CORRELATION
    )


def _zero_mean_relu_second_moment(
    covariance: Tensor, standard_deviation: Tensor
) -> Tensor:
    """Exact E[ReLU(X_i) ReLU(X_j)] for a centred Gaussian vector."""

    scale = standard_deviation.unsqueeze(-1) * standard_deviation.unsqueeze(-2)
    correlation = (covariance / scale.clamp_min(_EPS)).clamp(-1.0, 1.0)
    root = torch.sqrt((1.0 - correlation.square()).clamp_min(0.0))
    kernel = (
        root + (torch.pi - torch.arccos(correlation)) * correlation
    ) / _TWO_PI
    return scale * kernel


def _bivariate_normal_cdf(
    alpha: Tensor, correlation: Tensor, probability: Tensor
) -> Tensor:
    alpha_i = alpha.unsqueeze(-1)
    alpha_j = alpha.unsqueeze(-2)
    joint = probability.unsqueeze(-1) * probability.unsqueeze(-2)
    integral = torch.zeros_like(correlation)
    for node, weight in zip(_QUADRATURE_NODES, _QUADRATURE_WEIGHTS):
        integration_correlation = node * correlation
        denominator = (1.0 - integration_correlation.square()).clamp_min(_EPS)
        exponent = -(
            alpha_i.square()
            - 2.0 * integration_correlation * alpha_i * alpha_j
            + alpha_j.square()
        ) / (2.0 * denominator)
        integral = integral + weight * torch.exp(exponent) / (
            _TWO_PI * torch.sqrt(denominator)
        )
    joint = joint + correlation * integral
    upper = torch.minimum(probability.unsqueeze(-1), probability.unsqueeze(-2))
    return torch.minimum(joint, upper).clamp_min(0.0)


def _noncentral_relu_second_moment(
    pre_mean: Tensor,
    pre_covariance: Tensor,
    pre_std: Tensor,
    alpha: Tensor,
    density: Tensor,
    probability: Tensor,
) -> Tensor:
    """Pairwise ReLU second moment under a non-central Gaussian closure."""

    correlation = _correlation(pre_covariance, pre_std)
    scale = pre_std.unsqueeze(-1) * pre_std.unsqueeze(-2)
    alpha_i = alpha.unsqueeze(-1)
    alpha_j = alpha.unsqueeze(-2)
    root = torch.sqrt((1.0 - correlation.square()).clamp_min(_EPS))
    boundary_j_given_i = (alpha_j - correlation * alpha_i) / root
    boundary_i_given_j = (alpha_i - correlation * alpha_j) / root
    joint_probability = _bivariate_normal_cdf(alpha, correlation, probability)

    mean_i = pre_mean.unsqueeze(-1)
    mean_j = pre_mean.unsqueeze(-2)
    std_i = pre_std.unsqueeze(-1)
    std_j = pre_std.unsqueeze(-2)
    density_i = density.unsqueeze(-1)
    density_j = density.unsqueeze(-2)
    second_moment = (
        (mean_i * mean_j + pre_covariance) * joint_probability
        + mean_i
        * std_j
        * density_j
        * _normal_cdf(boundary_i_given_j)
        + mean_j
        * std_i
        * density_i
        * _normal_cdf(boundary_j_given_i)
        + scale
        * root
        * density_i
        * _normal_pdf(boundary_j_given_i)
    )
    return (0.5 * (second_moment + second_moment.transpose(-1, -2))).clamp_min(0.0)


def gaussian_closure_step(
    mean: Tensor,
    covariance: Tensor,
    weights: Tensor,
    *,
    exact_centered_input: bool,
) -> tuple[Tensor, ...]:
    """Propagate mean and dense covariance through one linear-ReLU layer."""

    pre_mean = torch.bmm(mean.unsqueeze(1), weights).squeeze(1)
    pre_covariance = torch.bmm(
        weights.transpose(1, 2), torch.bmm(covariance, weights)
    )
    pre_covariance = 0.5 * (pre_covariance + pre_covariance.transpose(-1, -2))
    pre_variance = torch.diagonal(
        pre_covariance, dim1=-2, dim2=-1
    ).clamp_min(_EPS)
    pre_std = torch.sqrt(pre_variance)
    alpha = pre_mean / pre_std
    density = _normal_pdf(alpha)
    probability = _normal_cdf(alpha)

    output_mean = pre_mean * probability + pre_std * density
    marginal_second = (
        (pre_mean.square() + pre_variance) * probability
        + pre_mean * pre_std * density
    )
    output_variance = (marginal_second - output_mean.square()).clamp_min(_EPS)

    if exact_centered_input:
        pairwise_second = _zero_mean_relu_second_moment(pre_covariance, pre_std)
    else:
        pairwise_second = _noncentral_relu_second_moment(
            pre_mean,
            pre_covariance,
            pre_std,
            alpha,
            density,
            probability,
        )
    output_covariance = pairwise_second - (
        output_mean.unsqueeze(-1) * output_mean.unsqueeze(-2)
    )
    output_covariance = 0.5 * (
        output_covariance + output_covariance.transpose(-1, -2)
    )
    output_covariance = _replace_diagonal(output_covariance, output_variance)
    return (
        output_mean,
        output_covariance,
        pre_mean,
        pre_covariance,
        pre_std,
        alpha,
    )


def analytic_features(
    input_mean: Tensor,
    input_covariance: Tensor,
    output_mean: Tensor,
    output_covariance: Tensor,
    pre_covariance: Tensor,
    pre_std: Tensor,
    alpha: Tensor,
    weights: Tensor,
    layer_fraction: float,
) -> Tensor:
    """Build output-indexed or permutation-invariant dimensionless features."""

    input_variance = torch.diagonal(input_covariance, dim1=-2, dim2=-1)
    input_rms = torch.sqrt(
        torch.mean(input_mean.square() + input_variance, dim=1, keepdim=True).clamp_min(_EPS)
    )
    input_mean_average = input_mean.mean(dim=1, keepdim=True)
    input_mean_spread = torch.sqrt(
        (input_mean - input_mean_average).square().mean(dim=1, keepdim=True).clamp_min(_EPS)
    )
    input_std_average = torch.sqrt(input_variance.clamp_min(0.0)).mean(
        dim=1, keepdim=True
    )
    column_energy = weights.square().sum(dim=1).clamp_min(_EPS)
    column_sum = weights.sum(dim=1)
    output_variance = torch.diagonal(output_covariance, dim1=-2, dim2=-1)

    pre_correlation = _correlation(pre_covariance, pre_std)
    width = pre_correlation.shape[-1]
    offdiag_count = max(width - 1, 1)
    diagonal = torch.diagonal(pre_correlation, dim1=-2, dim2=-1)
    correlation_mean = (pre_correlation.sum(dim=-1) - diagonal) / offdiag_count
    correlation_rms = torch.sqrt(
        ((pre_correlation.square().sum(dim=-1) - diagonal.square()) / offdiag_count).clamp_min(0.0)
    )
    correlation_scale = float(width) ** 0.5
    fraction = torch.full_like(output_mean, float(layer_fraction))

    def broadcast(value: Tensor) -> Tensor:
        return value.expand_as(output_mean)

    features = (
        broadcast((input_mean_average / input_rms).clamp(-4.0, 4.0) / 2.0),
        broadcast((input_mean_spread / input_rms).clamp(0.0, 4.0) / 2.0),
        broadcast((input_std_average / input_rms).clamp(0.0, 4.0) / 2.0),
        alpha.clamp(-8.0, 8.0) / 4.0,
        (output_mean / pre_std).clamp(0.0, 8.0) / 4.0,
        (torch.sqrt(output_variance) / pre_std).clamp(0.0, 8.0) / 4.0,
        torch.log((pre_std / input_rms).clamp_min(_EPS)).clamp(-6.0, 6.0) / 3.0,
        column_sum.clamp(-4.0 * _SQRT_2, 4.0 * _SQRT_2) / (4.0 * _SQRT_2),
        (8.0 * (torch.sqrt(0.5 * column_energy) - 1.0)).clamp(-4.0, 4.0) / 4.0,
        (correlation_scale * correlation_mean).clamp(-4.0, 4.0) / 4.0,
        (correlation_scale * correlation_rms).clamp(0.0, 4.0) / 4.0,
        fraction,
        fraction.square(),
    )
    return torch.stack(features, dim=-1)


class ResidualClosureNet(nn.Module):
    """Dense-covariance closure plus a shared equivariant residual network."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        config.validate()
        self.config = config
        input_dim = _FEATURE_DIM + 4 * config.hidden_dim
        self.input_projection = nn.Linear(input_dim, config.hidden_dim)
        self.update_blocks = nn.ModuleList(
            nn.Linear(config.hidden_dim, config.hidden_dim) for _ in range(config.blocks)
        )
        self.local_residual_head = nn.Linear(config.hidden_dim, 1)
        self.global_residual_head = nn.Linear(config.hidden_dim, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.input_projection.weight)
        nn.init.zeros_(self.input_projection.bias)
        for layer in self.update_blocks:
            nn.init.orthogonal_(layer.weight, gain=0.5)
            nn.init.zeros_(layer.bias)
        # Zero heads make the initial model exactly equal to the strong closure.
        nn.init.zeros_(self.local_residual_head.weight)
        nn.init.zeros_(self.local_residual_head.bias)
        nn.init.zeros_(self.global_residual_head.weight)
        nn.init.zeros_(self.global_residual_head.bias)

    def forward(self, weights: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if weights.ndim != 4:
            raise ValueError("weights must have shape (batch, depth, width, width)")
        batch, depth, width, width_again = weights.shape
        if depth != self.config.depth or width != self.config.width or width_again != width:
            raise ValueError(
                f"expected (*, {self.config.depth}, {self.config.width}, "
                f"{self.config.width}), got {tuple(weights.shape)}"
            )

        mean = torch.zeros((batch, width), dtype=weights.dtype, device=weights.device)
        covariance = torch.eye(width, dtype=weights.dtype, device=weights.device)
        covariance = covariance.unsqueeze(0).expand(batch, -1, -1).clone()
        state = torch.zeros(
            (batch, width, self.config.hidden_dim),
            dtype=weights.dtype,
            device=weights.device,
        )
        predictions: list[Tensor] = []
        baselines: list[Tensor] = []
        correction_limits: list[Tensor] = []

        for layer_index in range(depth):
            layer_weights = weights[:, layer_index]
            signed_message = torch.bmm(layer_weights.transpose(1, 2), state) / _SQRT_2
            energy_message = torch.bmm(
                layer_weights.square().transpose(1, 2), state
            ) * 0.5
            input_variance = torch.diagonal(covariance, dim1=-2, dim2=-1)
            input_std = torch.sqrt(input_variance.clamp_min(_EPS))
            input_correlation = _correlation(covariance, input_std)
            relational_state = torch.bmm(input_correlation, state) / (float(width) ** 0.5)
            covariance_message = torch.bmm(
                layer_weights.transpose(1, 2), relational_state
            ) / _SQRT_2
            global_context = state.mean(dim=1, keepdim=True).expand_as(state)

            input_mean = mean
            input_covariance = covariance
            (
                mean,
                covariance,
                _,
                pre_covariance,
                pre_std,
                alpha,
            ) = gaussian_closure_step(
                input_mean,
                input_covariance,
                layer_weights,
                exact_centered_input=layer_index == 0,
            )
            layer_fraction = layer_index / max(depth - 1, 1)
            features = analytic_features(
                input_mean,
                input_covariance,
                mean,
                covariance,
                pre_covariance,
                pre_std,
                alpha,
                layer_weights,
                layer_fraction,
            )
            network_input = torch.cat(
                (
                    features,
                    signed_message,
                    energy_message,
                    covariance_message,
                    global_context,
                ),
                dim=-1,
            )
            state = torch.tanh(self.input_projection(network_input))
            for block in self.update_blocks:
                state = state + 0.5 * torch.tanh(block(state))

            local_logit = self.local_residual_head(state).squeeze(-1)
            global_logit = self.global_residual_head(state.mean(dim=1)).expand_as(local_logit)
            standardized_residual = torch.tanh(local_logit + global_logit)
            if layer_index == 0:
                correction_limit = torch.zeros_like(pre_std)
            else:
                correction_limit = self.config.residual_scale * pre_std
            correction = correction_limit * standardized_residual
            baseline_mean = mean
            if layer_index == depth - 1 and layer_index > 0:
                baseline_mean = self.config.final_calibration * baseline_mean
            predictions.append((baseline_mean + correction).clamp_min(0.0))
            baselines.append(baseline_mean)
            correction_limits.append(correction_limit)

        return (
            torch.stack(predictions, dim=1),
            torch.stack(baselines, dim=1),
            torch.stack(correction_limits, dim=1),
        )


def parameter_count(config: ModelConfig) -> int:
    return sum(parameter.numel() for parameter in ResidualClosureNet(config).parameters())


def export_model(model: ResidualClosureNet, output_path: str | Path) -> Path:
    """Export a pickle-free archive consumed by the flopscope estimator."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    config = model.config
    block_weights = (
        np.stack([layer.weight.detach().cpu().numpy().T for layer in model.update_blocks], axis=0)
        if model.update_blocks
        else np.empty((0, config.hidden_dim, config.hidden_dim), np.float32)
    )
    block_biases = (
        np.stack([layer.bias.detach().cpu().numpy() for layer in model.update_blocks], axis=0)
        if model.update_blocks
        else np.empty((0, config.hidden_dim), np.float32)
    )
    arrays: dict[str, Any] = {
        "format_version": np.asarray(_FORMAT_VERSION, dtype=np.int64),
        "width": np.asarray(config.width, dtype=np.int64),
        "depth": np.asarray(config.depth, dtype=np.int64),
        "hidden_dim": np.asarray(config.hidden_dim, dtype=np.int64),
        "blocks": np.asarray(config.blocks, dtype=np.int64),
        "residual_scale": np.asarray(config.residual_scale, dtype=np.float32),
        "final_calibration": np.asarray(config.final_calibration, dtype=np.float32),
        "feature_dim": np.asarray(_FEATURE_DIM, dtype=np.int64),
        "input_weight": model.input_projection.weight.detach().cpu().numpy().T,
        "input_bias": model.input_projection.bias.detach().cpu().numpy(),
        "block_weights": block_weights,
        "block_biases": block_biases,
        "local_output_weight": model.local_residual_head.weight.detach().cpu().numpy().T,
        "local_output_bias": model.local_residual_head.bias.detach().cpu().numpy(),
        "global_output_weight": model.global_residual_head.weight.detach().cpu().numpy().T,
        "global_output_bias": model.global_residual_head.bias.detach().cpu().numpy(),
    }
    np.savez(output_path, **arrays)
    return output_path


def architecture_dict(model: ResidualClosureNet) -> dict[str, Any]:
    return {
        **asdict(model.config),
        "baseline": "dense_noncentral_gaussian_covariance",
        "feature_dim": _FEATURE_DIM,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }
