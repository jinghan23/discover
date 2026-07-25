"""WhestBench entrypoint for the competitive learned covariance closure.

Copy an exported ``learned_residual_weights.npz`` next to this file before
validation or packaging.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import BaseEstimator


# Dense covariance operations intentionally drop symmetry metadata after
# elementwise normalization; the result is explicitly re-symmetrized below.
# This only controls local diagnostics (the hosted evaluator ignores configure).
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    flops.configure(symmetry_warnings=False)


_EPS = 1e-12
_MAX_ABS_CORRELATION = 1.0 - 1e-6
_SQRT_2 = 2.0**0.5
_TWO_PI = 2.0 * fnp.pi
_EXPECTED_FORMAT_VERSION = 2
_EXPECTED_FEATURE_DIM = 13
_WEIGHTS_NAME = "learned_residual_weights.npz"
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


def _correlation(covariance, standard_deviation):
    scale = standard_deviation[:, None] * standard_deviation[None, :]
    return fnp.clip(
        covariance / fnp.maximum(scale, _EPS),
        -_MAX_ABS_CORRELATION,
        _MAX_ABS_CORRELATION,
    )


def _zero_mean_relu_second_moment(covariance, standard_deviation):
    scale = standard_deviation[:, None] * standard_deviation[None, :]
    correlation = fnp.clip(covariance / fnp.maximum(scale, _EPS), -1.0, 1.0)
    root = fnp.sqrt(fnp.maximum(1.0 - correlation * correlation, 0.0))
    kernel = (
        root + (fnp.pi - fnp.arccos(correlation)) * correlation
    ) / _TWO_PI
    return scale * kernel


def _bivariate_normal_cdf(alpha, correlation, probability):
    alpha_i = alpha[:, None]
    alpha_j = alpha[None, :]
    integral = fnp.zeros_like(correlation)
    for node, weight in zip(_QUADRATURE_NODES, _QUADRATURE_WEIGHTS):
        integration_correlation = node * correlation
        denominator = fnp.maximum(
            1.0 - integration_correlation * integration_correlation, _EPS
        )
        exponent = -(
            alpha_i * alpha_i
            - 2.0 * integration_correlation * alpha_i * alpha_j
            + alpha_j * alpha_j
        ) / (2.0 * denominator)
        integral = integral + weight * fnp.exp(exponent) / (
            _TWO_PI * fnp.sqrt(denominator)
        )
    joint = probability[:, None] * probability[None, :] + correlation * integral
    upper = fnp.minimum(probability[:, None], probability[None, :])
    return fnp.maximum(fnp.minimum(joint, upper), 0.0)


def _noncentral_relu_second_moment(
    pre_mean,
    pre_covariance,
    pre_std,
    alpha,
    density,
    probability,
):
    correlation = _correlation(pre_covariance, pre_std)
    scale = pre_std[:, None] * pre_std[None, :]
    alpha_i = alpha[:, None]
    alpha_j = alpha[None, :]
    root = fnp.sqrt(fnp.maximum(1.0 - correlation * correlation, _EPS))
    boundary_j_given_i = (alpha_j - correlation * alpha_i) / root
    boundary_i_given_j = (alpha_i - correlation * alpha_j) / root
    joint_probability = _bivariate_normal_cdf(alpha, correlation, probability)
    second_moment = (
        (pre_mean[:, None] * pre_mean[None, :] + pre_covariance)
        * joint_probability
        + pre_mean[:, None]
        * pre_std[None, :]
        * density[None, :]
        * flops.stats.norm.cdf(boundary_i_given_j)
        + pre_mean[None, :]
        * pre_std[:, None]
        * density[:, None]
        * flops.stats.norm.cdf(boundary_j_given_i)
        + scale
        * root
        * density[:, None]
        * flops.stats.norm.pdf(boundary_j_given_i)
    )
    return fnp.maximum(0.5 * (second_moment + second_moment.T), 0.0)


def _analytic_step(mean, covariance, weights, exact_centered_input):
    pre_mean = weights.T @ mean
    pre_covariance = weights.T @ covariance @ weights
    pre_covariance = 0.5 * (pre_covariance + pre_covariance.T)
    pre_variance = fnp.maximum(fnp.diag(pre_covariance), _EPS)
    pre_std = fnp.sqrt(pre_variance)
    alpha = pre_mean / pre_std
    density = flops.stats.norm.pdf(alpha)
    probability = flops.stats.norm.cdf(alpha)
    output_mean = pre_mean * probability + pre_std * density
    marginal_second = (
        (pre_mean * pre_mean + pre_variance) * probability
        + pre_mean * pre_std * density
    )
    output_variance = fnp.maximum(marginal_second - output_mean * output_mean, _EPS)

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
    output_covariance = pairwise_second - output_mean[:, None] * output_mean[None, :]
    output_covariance = 0.5 * (output_covariance + output_covariance.T)
    fnp.fill_diagonal(output_covariance, output_variance)
    return output_mean, output_covariance, pre_covariance, pre_std, alpha


def _analytic_features(
    input_mean,
    input_covariance,
    output_mean,
    output_covariance,
    pre_covariance,
    pre_std,
    alpha,
    weights,
    layer_fraction,
):
    input_variance = fnp.diag(input_covariance)
    input_rms = fnp.sqrt(
        fnp.maximum(fnp.mean(input_mean * input_mean + input_variance), _EPS)
    )
    input_mean_average = fnp.mean(input_mean)
    input_mean_spread = fnp.sqrt(
        fnp.maximum(fnp.mean((input_mean - input_mean_average) ** 2), _EPS)
    )
    input_std_average = fnp.mean(fnp.sqrt(fnp.maximum(input_variance, 0.0)))
    column_energy = fnp.maximum(fnp.sum(weights * weights, axis=0), _EPS)
    column_sum = fnp.sum(weights, axis=0)
    output_variance = fnp.diag(output_covariance)
    pre_correlation = _correlation(pre_covariance, pre_std)
    width = output_mean.shape[0]
    offdiag_count = max(width - 1, 1)
    diagonal = fnp.diag(pre_correlation)
    correlation_mean = (fnp.sum(pre_correlation, axis=1) - diagonal) / offdiag_count
    correlation_rms = fnp.sqrt(
        fnp.maximum(
            (fnp.sum(pre_correlation * pre_correlation, axis=1) - diagonal * diagonal)
            / offdiag_count,
            0.0,
        )
    )
    correlation_scale = float(width) ** 0.5
    fraction = fnp.full(output_mean.shape, layer_fraction, dtype=fnp.float32)
    ones = fnp.ones_like(output_mean)
    features = (
        ones * fnp.clip(input_mean_average / input_rms, -4.0, 4.0) / 2.0,
        ones * fnp.clip(input_mean_spread / input_rms, 0.0, 4.0) / 2.0,
        ones * fnp.clip(input_std_average / input_rms, 0.0, 4.0) / 2.0,
        fnp.clip(alpha, -8.0, 8.0) / 4.0,
        fnp.clip(output_mean / pre_std, 0.0, 8.0) / 4.0,
        fnp.clip(fnp.sqrt(output_variance) / pre_std, 0.0, 8.0) / 4.0,
        fnp.clip(fnp.log(fnp.maximum(pre_std / input_rms, _EPS)), -6.0, 6.0) / 3.0,
        fnp.clip(column_sum, -4.0 * _SQRT_2, 4.0 * _SQRT_2)
        / (4.0 * _SQRT_2),
        fnp.clip(8.0 * (fnp.sqrt(0.5 * column_energy) - 1.0), -4.0, 4.0) / 4.0,
        fnp.clip(correlation_scale * correlation_mean, -4.0, 4.0) / 4.0,
        fnp.clip(correlation_scale * correlation_rms, 0.0, 4.0) / 4.0,
        fraction,
        fraction * fraction,
    )
    return fnp.stack(features, axis=-1)


class Estimator(BaseEstimator):
    def setup(self, context):
        if context.submission_dir is None:
            raise ValueError("submission_dir is required to load learned weights")
        archive_path = Path(context.submission_dir) / _WEIGHTS_NAME
        if not archive_path.exists():
            raise FileNotFoundError(
                f"missing {archive_path}; copy an exported .npz to {_WEIGHTS_NAME}"
            )
        archive = fnp.load(str(archive_path))
        if int(archive["format_version"]) != _EXPECTED_FORMAT_VERSION:
            raise ValueError("unsupported learned residual weight format")
        if int(archive["feature_dim"]) != _EXPECTED_FEATURE_DIM:
            raise ValueError("learned weights use an incompatible feature definition")
        if int(archive["width"]) != context.width or int(archive["depth"]) != context.depth:
            raise ValueError("learned weights do not match evaluator width/depth")

        self.hidden_dim = int(archive["hidden_dim"])
        self.blocks = int(archive["blocks"])
        self.residual_scale = float(archive["residual_scale"])
        self.final_calibration = float(archive["final_calibration"])
        self.input_weight = archive["input_weight"]
        self.input_bias = archive["input_bias"]
        self.block_weights = archive["block_weights"]
        self.block_biases = archive["block_biases"]
        self.local_output_weight = archive["local_output_weight"]
        self.local_output_bias = archive["local_output_bias"]
        self.global_output_weight = archive["global_output_weight"]
        self.global_output_bias = archive["global_output_bias"]

    def predict(self, mlp, budget):
        del budget
        mean = fnp.zeros(mlp.width, dtype=fnp.float32)
        covariance = fnp.eye(mlp.width, dtype=fnp.float32)
        state = fnp.zeros((mlp.width, self.hidden_dim), dtype=fnp.float32)
        rows = []

        for layer_index, weights in enumerate(mlp.weights):
            signed_message = (weights.T @ state) / _SQRT_2
            energy_message = ((weights * weights).T @ state) * 0.5
            input_std = fnp.sqrt(fnp.maximum(fnp.diag(covariance), _EPS))
            input_correlation = _correlation(covariance, input_std)
            relational_state = (input_correlation @ state) / (float(mlp.width) ** 0.5)
            covariance_message = (weights.T @ relational_state) / _SQRT_2
            global_context = fnp.broadcast_to(
                fnp.mean(state, axis=0)[None, :], state.shape
            )

            input_mean = mean
            input_covariance = covariance
            mean, covariance, pre_covariance, pre_std, alpha = _analytic_step(
                input_mean,
                input_covariance,
                weights,
                layer_index == 0,
            )
            layer_fraction = layer_index / max(mlp.depth - 1, 1)
            features = _analytic_features(
                input_mean,
                input_covariance,
                mean,
                covariance,
                pre_covariance,
                pre_std,
                alpha,
                weights,
                layer_fraction,
            )
            network_input = fnp.concatenate(
                (
                    features,
                    signed_message,
                    energy_message,
                    covariance_message,
                    global_context,
                ),
                axis=-1,
            )
            state = fnp.tanh(network_input @ self.input_weight + self.input_bias)
            for block_index in range(self.blocks):
                state = state + 0.5 * fnp.tanh(
                    state @ self.block_weights[block_index] + self.block_biases[block_index]
                )

            local_logit = (
                state @ self.local_output_weight
            )[:, 0] + self.local_output_bias[0]
            global_state = fnp.mean(state, axis=0)
            global_logit = (
                global_state @ self.global_output_weight
            )[0] + self.global_output_bias[0]
            standardized_residual = fnp.tanh(local_logit + global_logit)
            if layer_index == 0:
                correction = fnp.zeros_like(pre_std)
            else:
                correction = self.residual_scale * pre_std * standardized_residual
            baseline_mean = mean
            if layer_index == mlp.depth - 1 and layer_index > 0:
                baseline_mean = self.final_calibration * baseline_mean
            rows.append(fnp.maximum(baseline_mean + correction, 0.0))

        return fnp.stack(rows, axis=0)
