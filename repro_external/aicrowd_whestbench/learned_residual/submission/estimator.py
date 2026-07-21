"""WhestBench submission entrypoint for the learned residual closure.

Place ``learned_residual_weights.npz`` next to this file before validating or
packaging the containing directory.
"""

from __future__ import annotations

from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import BaseEstimator


_EPS = 1e-12
_SQRT_2 = 2.0**0.5
_EXPECTED_FORMAT_VERSION = 1
_EXPECTED_FEATURE_DIM = 10
_WEIGHTS_NAME = "learned_residual_weights.npz"


def _analytic_step(mean, variance, weights):
    pre_mean = weights.T @ mean
    pre_variance = fnp.maximum((weights * weights).T @ variance, _EPS)
    pre_std = fnp.sqrt(pre_variance)
    alpha = pre_mean / pre_std
    density = flops.stats.norm.pdf(alpha)
    probability = flops.stats.norm.cdf(alpha)
    output_mean = pre_mean * probability + pre_std * density
    second_moment = (
        (pre_mean * pre_mean + pre_variance) * probability
        + pre_mean * pre_std * density
    )
    output_variance = fnp.maximum(second_moment - output_mean * output_mean, 0.0)
    return output_mean, output_variance, pre_std, alpha


def _analytic_features(
    input_mean,
    input_variance,
    output_mean,
    output_variance,
    pre_std,
    alpha,
    weights,
    layer_fraction,
):
    input_rms = fnp.sqrt(
        fnp.maximum(fnp.mean(input_mean * input_mean + input_variance), _EPS)
    )
    column_energy = fnp.maximum(fnp.sum(weights * weights, axis=0), _EPS)
    column_sum = fnp.sum(weights, axis=0)
    fraction = fnp.full(output_mean.shape, layer_fraction, dtype=fnp.float32)
    features = (
        fnp.clip(input_mean / input_rms, -8.0, 8.0) / 4.0,
        fnp.clip(fnp.sqrt(fnp.maximum(input_variance, 0.0)) / input_rms, 0.0, 8.0)
        / 4.0,
        fnp.clip(alpha, -8.0, 8.0) / 4.0,
        fnp.clip(output_mean / pre_std, 0.0, 8.0) / 4.0,
        fnp.clip(fnp.sqrt(output_variance) / pre_std, 0.0, 8.0) / 4.0,
        fnp.clip(fnp.log(fnp.maximum(pre_std / input_rms, _EPS)), -6.0, 6.0)
        / 3.0,
        fnp.clip(column_sum, -4.0 * _SQRT_2, 4.0 * _SQRT_2)
        / (4.0 * _SQRT_2),
        fnp.clip(8.0 * (fnp.sqrt(0.5 * column_energy) - 1.0), -4.0, 4.0)
        / 4.0,
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
        self.input_weight = archive["input_weight"]
        self.input_bias = archive["input_bias"]
        self.block_weights = archive["block_weights"]
        self.block_biases = archive["block_biases"]
        self.output_weight = archive["output_weight"]
        self.output_bias = archive["output_bias"]

    def predict(self, mlp, budget):
        del budget
        mean = fnp.zeros(mlp.width, dtype=fnp.float32)
        variance = fnp.ones(mlp.width, dtype=fnp.float32)
        state = fnp.zeros((mlp.width, self.hidden_dim), dtype=fnp.float32)
        rows = []

        for layer_index, weights in enumerate(mlp.weights):
            signed_message = (weights.T @ state) / _SQRT_2
            energy_message = ((weights * weights).T @ state) * 0.5
            global_context = fnp.broadcast_to(
                fnp.mean(state, axis=0)[None, :], state.shape
            )

            input_mean = mean
            input_variance = variance
            mean, variance, pre_std, alpha = _analytic_step(
                input_mean, input_variance, weights
            )
            layer_fraction = layer_index / max(mlp.depth - 1, 1)
            features = _analytic_features(
                input_mean,
                input_variance,
                mean,
                variance,
                pre_std,
                alpha,
                weights,
                layer_fraction,
            )
            network_input = fnp.concatenate(
                (features, signed_message, energy_message, global_context), axis=-1
            )
            state = fnp.tanh(network_input @ self.input_weight + self.input_bias)
            for block_index in range(self.blocks):
                state = state + 0.5 * fnp.tanh(
                    state @ self.block_weights[block_index]
                    + self.block_biases[block_index]
                )

            standardized_residual = fnp.tanh(
                (state @ self.output_weight)[:, 0] + self.output_bias[0]
            )
            correction = (
                self.residual_scale
                * layer_fraction
                * pre_std
                * standardized_residual
            )
            rows.append(fnp.maximum(mean + correction, 0.0))

        return fnp.stack(rows, axis=0)
