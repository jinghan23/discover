"""Whitened antithetic estimator with a tuned hidden mean correction."""

from __future__ import annotations

import flopscope.numpy as fnp
from whestbench import BaseEstimator


_BUDGET_FRACTION = 0.116398
_MIN_SAMPLES = 32
_MAX_SAMPLES = 1_000_000
_ARRAY_BYTES_LIMIT = 100 * 1024 * 1024
_WORST_CASE_ITEMSIZE = 8
_SETUP_SEED_OFFSET = 3

_INV_SQRT_2PI = 0.3989422804014327
_INV_2PI = 0.15915494309189535
_PI = 3.141592653589793
_CDF_POLY_P = 0.2316419
_CDF_POLY_B1 = 0.319381530
_CDF_POLY_B2 = -0.356563782
_CDF_POLY_B3 = 1.781477937
_CDF_POLY_B4 = -1.821255978
_CDF_POLY_B5 = 1.330274429

_HIDDEN_BLEND_START = 10
_HIDDEN_BLEND_STOP = 12
_LAYER2_BLEND = -0.25
_LAYER3_BLEND = -0.005
_LAYER4_BLEND = -0.11
_LAYER5_BLEND = 0.0
_LAYER6_BLEND = -0.060
_LAYER7_BLEND = 0.182
_LAYER8_BLEND = 0.16
_LAYER9_BLEND = 0.12
_HIDDEN_BLEND_FIRST = 0.18666666666666668
_HIDDEN_BLEND_LAST = 0.40
_HIDDEN_BLEND_STEP = (_HIDDEN_BLEND_LAST - _HIDDEN_BLEND_FIRST) / (
    _HIDDEN_BLEND_STOP - _HIDDEN_BLEND_START
)
_LAYER1_BLEND = 0.255
_FINAL_BLEND = -0.010
_SECOND_MOMENT_BLEND = 0.39


def _sample_count(budget: int, width: int, depth: int) -> int:
    per_sample = depth * (2 * width * width + width) + width * width
    fixed = 9 * width**3 + 2 * (2 * width**3)
    k = (int(_BUDGET_FRACTION * budget) - fixed) // max(per_sample, 1)
    max_k_by_bytes = _ARRAY_BYTES_LIMIT // max(width * _WORST_CASE_ITEMSIZE, 1)
    k = int(max(_MIN_SAMPLES, min(_MAX_SAMPLES, max_k_by_bytes, k)))
    return k - (k & 1)


def _gaussian_relu_mean(mean, variance):
    std = fnp.sqrt(fnp.maximum(variance, 1e-12))
    alpha = fnp.clip(mean / std, -8.0, 8.0)
    density = _INV_SQRT_2PI * fnp.exp(-0.5 * alpha * alpha)
    abs_alpha = fnp.abs(alpha)
    t = 1.0 / (1.0 + _CDF_POLY_P * abs_alpha)
    tail = density * (
        t
        * (
            _CDF_POLY_B1
            + t
            * (
                _CDF_POLY_B2
                + t
                * (_CDF_POLY_B3 + t * (_CDF_POLY_B4 + t * _CDF_POLY_B5))
            )
        )
    )
    cdf = fnp.where(alpha >= 0.0, 1.0 - tail, tail)
    return std * density + mean * cdf


def _first_relu_covariance(weight, variance, mean):
    covariance = weight.T @ weight
    scale = fnp.sqrt(fnp.maximum(variance[:, None] * variance[None, :], 1e-24))
    rho = fnp.clip(covariance / scale, -1.0, 1.0)
    relu_second = scale * _INV_2PI * (
        fnp.sqrt(fnp.maximum(1.0 - rho * rho, 0.0))
        + (_PI - fnp.arccos(rho)) * rho
    )
    return relu_second - mean[:, None] * mean[None, :]


class Estimator(BaseEstimator):
    def setup(self, context):
        self._width = int(context.width)
        self._depth = int(context.depth)
        self._samples = self._make_samples(
            self._width,
            _sample_count(int(context.flop_budget), self._width, self._depth),
            int(context.seed) + _SETUP_SEED_OFFSET,
        )

    def _make_samples(self, width, k, seed):
        rng = fnp.random.default_rng(seed)
        half_count = k // 2
        u = rng.standard_normal((half_count, width), dtype=fnp.float32)
        u = u - fnp.mean(u, axis=0)
        covariance = (u.T @ u) / float(half_count)
        eigenvalues, eigenvectors = fnp.linalg.eigh(covariance)
        eigenvalues = fnp.maximum(eigenvalues, 1e-6)
        inverse_sqrt = (eigenvectors / fnp.sqrt(eigenvalues)) @ eigenvectors.T
        half = fnp.asarray(u @ inverse_sqrt, dtype=fnp.float32)
        return fnp.concatenate([half, -half], axis=0)

    def predict(self, mlp, budget):
        width = mlp.width
        depth = mlp.depth
        if (
            not hasattr(self, "_samples")
            or self._samples.shape[1] != width
            or getattr(self, "_depth", depth) != depth
        ):
            self._samples = self._make_samples(width, _sample_count(budget, width, depth), mlp.seed)
            self._depth = depth

        first_weight = mlp.weights[0]
        first_variance = fnp.sum(first_weight * first_weight, axis=0)
        first_mean = fnp.sqrt(first_variance) * _INV_SQRT_2PI
        if depth == 1:
            zero_rows = fnp.zeros((depth - 1, width), dtype=fnp.float32)
            return fnp.concatenate([zero_rows, first_mean[None, :]], axis=0)

        target_variance = first_variance * (0.5 - _INV_SQRT_2PI * _INV_SQRT_2PI)
        first_covariance = _first_relu_covariance(first_weight, first_variance, first_mean)
        x = fnp.maximum(self._samples @ first_weight, 0.0)
        sample_mean = fnp.mean(x, axis=0)
        sample_second = fnp.mean(x * x, axis=0)
        sample_variance = fnp.maximum(sample_second - sample_mean * sample_mean, 1e-6)
        x = (x - sample_mean) * fnp.sqrt(target_variance / sample_variance) + first_mean

        for layer_index, weight in enumerate(mlp.weights[1:-1], start=1):
            z = x @ weight
            if layer_index == 1:
                exact_z_mean = first_mean @ weight
                exact_z_variance = fnp.maximum(
                    fnp.sum(weight * (first_covariance @ weight), axis=0),
                    1e-8,
                )
                z_mean = fnp.mean(z, axis=0)
                z_second = fnp.mean(z * z, axis=0)
                z_variance = fnp.maximum(z_second - z_mean * z_mean, 1e-8)
                target_z_mean = exact_z_mean
                target_z_variance = z_variance + (
                    exact_z_variance - z_variance
                ) * _SECOND_MOMENT_BLEND
                z = (z - z_mean) * fnp.sqrt(
                    fnp.maximum(target_z_variance, 1e-8) / z_variance
                ) + target_z_mean
            x = fnp.maximum(z, 0.0)
            if layer_index == 1:
                x_mean = fnp.mean(x, axis=0)
                gaussian_mean = _gaussian_relu_mean(exact_z_mean, exact_z_variance)
                x = x + (gaussian_mean - x_mean) * _LAYER1_BLEND
            elif layer_index == 2:
                x_mean = fnp.mean(x, axis=0)
                z_mean = fnp.mean(z, axis=0)
                z_second = fnp.mean(z * z, axis=0)
                z_variance = fnp.maximum(z_second - z_mean * z_mean, 1e-8)
                gaussian_mean = _gaussian_relu_mean(z_mean, z_variance)
                x = x + (gaussian_mean - x_mean) * _LAYER2_BLEND
            elif layer_index == 3:
                x_mean = fnp.mean(x, axis=0)
                z_mean = fnp.mean(z, axis=0)
                z_second = fnp.mean(z * z, axis=0)
                z_variance = fnp.maximum(z_second - z_mean * z_mean, 1e-8)
                gaussian_mean = _gaussian_relu_mean(z_mean, z_variance)
                x = x + (gaussian_mean - x_mean) * _LAYER3_BLEND
            elif layer_index == 4:
                x_mean = fnp.mean(x, axis=0)
                z_mean = fnp.mean(z, axis=0)
                z_second = fnp.mean(z * z, axis=0)
                z_variance = fnp.maximum(z_second - z_mean * z_mean, 1e-8)
                gaussian_mean = _gaussian_relu_mean(z_mean, z_variance)
                x = x + (gaussian_mean - x_mean) * _LAYER4_BLEND
            elif layer_index == 5:
                x_mean = fnp.mean(x, axis=0)
                z_mean = fnp.mean(z, axis=0)
                z_second = fnp.mean(z * z, axis=0)
                z_variance = fnp.maximum(z_second - z_mean * z_mean, 1e-8)
                gaussian_mean = _gaussian_relu_mean(z_mean, z_variance)
                x = x + (gaussian_mean - x_mean) * _LAYER5_BLEND
            elif layer_index == 6:
                x_mean = fnp.mean(x, axis=0)
                z_mean = fnp.mean(z, axis=0)
                z_second = fnp.mean(z * z, axis=0)
                z_variance = fnp.maximum(z_second - z_mean * z_mean, 1e-8)
                gaussian_mean = _gaussian_relu_mean(z_mean, z_variance)
                x = x + (gaussian_mean - x_mean) * _LAYER6_BLEND
            elif layer_index == 7:
                x_mean = fnp.mean(x, axis=0)
                z_mean = fnp.mean(z, axis=0)
                z_second = fnp.mean(z * z, axis=0)
                z_variance = fnp.maximum(z_second - z_mean * z_mean, 1e-8)
                gaussian_mean = _gaussian_relu_mean(z_mean, z_variance)
                x = x + (gaussian_mean - x_mean) * _LAYER7_BLEND
            elif layer_index == 8:
                x_mean = fnp.mean(x, axis=0)
                z_mean = fnp.mean(z, axis=0)
                z_second = fnp.mean(z * z, axis=0)
                z_variance = fnp.maximum(z_second - z_mean * z_mean, 1e-8)
                gaussian_mean = _gaussian_relu_mean(z_mean, z_variance)
                x = x + (gaussian_mean - x_mean) * _LAYER8_BLEND
            elif layer_index == 9:
                x_mean = fnp.mean(x, axis=0)
                z_mean = fnp.mean(z, axis=0)
                z_second = fnp.mean(z * z, axis=0)
                z_variance = fnp.maximum(z_second - z_mean * z_mean, 1e-8)
                gaussian_mean = _gaussian_relu_mean(z_mean, z_variance)
                x = x + (gaussian_mean - x_mean) * _LAYER9_BLEND
            elif _HIDDEN_BLEND_START <= layer_index <= _HIDDEN_BLEND_STOP:
                x_mean = fnp.mean(x, axis=0)
                z_mean = fnp.mean(z, axis=0)
                z_second = fnp.mean(z * z, axis=0)
                z_variance = fnp.maximum(z_second - z_mean * z_mean, 1e-8)
                gaussian_mean = _gaussian_relu_mean(z_mean, z_variance)
                blend = _HIDDEN_BLEND_FIRST + _HIDDEN_BLEND_STEP * (
                    layer_index - _HIDDEN_BLEND_START
                )
                x = x + (gaussian_mean - x_mean) * blend

        z = x @ mlp.weights[-1]
        x = fnp.maximum(z, 0.0)
        sampled_mean = fnp.asarray(
            fnp.mean(fnp.asarray(x, dtype=fnp.float64), axis=0),
            dtype=fnp.float32,
        )
        z_mean = fnp.mean(z, axis=0)
        z_second = fnp.mean(z * z, axis=0)
        z_variance = fnp.maximum(z_second - z_mean * z_mean, 1e-8)
        gaussian_mean = _gaussian_relu_mean(z_mean, z_variance)
        final_mean = sampled_mean + (gaussian_mean - sampled_mean) * _FINAL_BLEND
        zero_rows = fnp.zeros((depth - 1, width), dtype=fnp.float32)
        return fnp.concatenate([zero_rows, final_mean[None, :]], axis=0)
