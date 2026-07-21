"""Whitened antithetic estimator with a tuned hidden mean correction."""

from __future__ import annotations

import flopscope.numpy as fnp
from whestbench import BaseEstimator


_BUDGET_FRACTION = 0.17135
_MIN_SAMPLES = 32
_MAX_SAMPLES = 1_000_000
_ARRAY_BYTES_LIMIT = 100 * 1024 * 1024
_WORST_CASE_ITEMSIZE = 8
_SETUP_SEED_OFFSET = 3
_FIRST_COV_BLEND = 0.0
_FIRST_MOMENT_BLEND = 1.65

_INV_SQRT_2PI = 0.3989422804014327
_INV_2PI = 0.15915494309189535
_CDF_POLY_P = 0.2316419
_CDF_POLY_B1 = 0.319381530
_CDF_POLY_B2 = -0.356563782
_CDF_POLY_B3 = 1.781477937
_CDF_POLY_B4 = -1.821255978
_CDF_POLY_B5 = 1.330274429

_HIDDEN_BLEND_START = 2
_HIDDEN_BLEND_STOP = 12
_HIDDEN_BLEND_FIRST = -0.11
_HIDDEN_BLEND_LAST = 0.27
_HIDDEN_BLEND_STEP = (_HIDDEN_BLEND_LAST - _HIDDEN_BLEND_FIRST) / (
    _HIDDEN_BLEND_STOP - _HIDDEN_BLEND_START
)
_HIDDEN_VAR_BLEND = 0.0
_FINAL_BLEND = 0.005


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


def _gaussian_relu_second(mean, variance):
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
    return (mean * mean + variance) * cdf + mean * std * density


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
        x = fnp.maximum(self._samples @ first_weight, 0.0)
        sample_mean = fnp.mean(x, axis=0)
        sample_second = fnp.mean(x * x, axis=0)
        sample_variance = fnp.maximum(sample_second - sample_mean * sample_mean, 1e-6)
        if _FIRST_MOMENT_BLEND == 1.0:
            x = (x - sample_mean) * fnp.sqrt(target_variance / sample_variance) + first_mean
        else:
            first_scale = fnp.sqrt(
                fnp.maximum(
                    1.0
                    + (target_variance / sample_variance - 1.0) * _FIRST_MOMENT_BLEND,
                    1e-6,
                )
            )
            first_center = sample_mean + (first_mean - sample_mean) * _FIRST_MOMENT_BLEND
            x = (x - sample_mean) * first_scale + first_center
        if _FIRST_COV_BLEND != 0.0:
            centered = x - first_mean
            sample_covariance = (centered.T @ centered) / float(x.shape[0])
            std = fnp.sqrt(fnp.maximum(first_variance, 1e-12))
            pre_covariance = first_weight.T @ first_weight
            correlation = fnp.clip(pre_covariance / (std[:, None] * std[None, :]), -1.0, 1.0)
            root = fnp.sqrt(fnp.maximum(1.0 - correlation * correlation, 0.0))
            angle = fnp.arccos(correlation)
            exact_second = (
                std[:, None]
                * std[None, :]
                * (root + (3.141592653589793 - angle) * correlation)
                * _INV_2PI
            )
            exact_covariance = exact_second - first_mean[:, None] * first_mean[None, :]
            target_covariance = sample_covariance + (
                exact_covariance - sample_covariance
            ) * _FIRST_COV_BLEND
            sample_values, sample_vectors = fnp.linalg.eigh(sample_covariance)
            target_values, target_vectors = fnp.linalg.eigh(target_covariance)
            sample_values = fnp.maximum(sample_values, 1e-7)
            target_values = fnp.maximum(target_values, 1e-7)
            inverse_sample_sqrt = (
                sample_vectors / fnp.sqrt(sample_values)
            ) @ sample_vectors.T
            target_sqrt = (
                target_vectors * fnp.sqrt(target_values)
            ) @ target_vectors.T
            x = fnp.asarray(centered @ (inverse_sample_sqrt @ target_sqrt) + first_mean, dtype=fnp.float32)

        for layer_index, weight in enumerate(mlp.weights[1:-1], start=1):
            z = x @ weight
            x = fnp.maximum(z, 0.0)
            if _HIDDEN_BLEND_START <= layer_index <= _HIDDEN_BLEND_STOP:
                x_mean = fnp.mean(x, axis=0)
                z_mean = fnp.mean(z, axis=0)
                z_second = fnp.mean(z * z, axis=0)
                z_variance = fnp.maximum(z_second - z_mean * z_mean, 1e-8)
                gaussian_mean = _gaussian_relu_mean(z_mean, z_variance)
                blend = _HIDDEN_BLEND_FIRST + _HIDDEN_BLEND_STEP * (
                    layer_index - _HIDDEN_BLEND_START
                )
                if _HIDDEN_VAR_BLEND != 0.0:
                    x_variance = fnp.maximum(sample_second - x_mean * x_mean, 1e-8)
                    gaussian_second = _gaussian_relu_second(z_mean, z_variance)
                    gaussian_variance = fnp.maximum(
                        gaussian_second - gaussian_mean * gaussian_mean, 1e-8
                    )
                    variance_ratio = gaussian_variance / x_variance
                    scale = fnp.sqrt(
                        fnp.maximum(1.0 + (variance_ratio - 1.0) * _HIDDEN_VAR_BLEND, 1e-6)
                    )
                    x = (x - x_mean) * scale + x_mean
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
