"""Setup-cached whitened antithetic submission candidate."""

from __future__ import annotations

import flopscope.numpy as fnp
from whestbench import BaseEstimator


_BUDGET_FRACTION = 0.143
_MIN_SAMPLES = 32
_MAX_SAMPLES = 1_000_000
_ARRAY_BYTES_LIMIT = 100 * 1024 * 1024
_WORST_CASE_ITEMSIZE = 8
_SETUP_SEED_OFFSET = 3
_INV_SQRT_2PI = 0.3989422804014327
_PI = 3.141592653589793
_TWO_PI = 6.283185307179586
_FIRST_VARIANCE_FACTOR = 1.0
_SECOND_VARIANCE_FACTOR = 1.0


def _sample_count(budget: int, width: int, depth: int) -> int:
    per_sample = (2 * depth - 1) * width * width + depth * width + width * width
    fixed = 9 * width**3 + 2 * (2 * width**3)
    k = (int(_BUDGET_FRACTION * budget) - fixed) // max(per_sample, 1)
    max_k_by_bytes = _ARRAY_BYTES_LIMIT // max(width * _WORST_CASE_ITEMSIZE, 1)
    k = int(max(_MIN_SAMPLES, min(_MAX_SAMPLES, max_k_by_bytes, k)))
    return k - (k & 1)


def _first_relu_covariance(weight, variance, mean):
    gram = weight.T @ weight
    std = fnp.sqrt(fnp.maximum(variance, 1e-12))
    scale = std[:, None] * std[None, :]
    rho = fnp.clip(gram / fnp.maximum(scale, 1e-12), -1.0, 1.0)
    root = fnp.sqrt(fnp.maximum(1.0 - rho * rho, 0.0))
    second = scale * (root + (_PI - fnp.arccos(rho)) * rho) / _TWO_PI
    return second - mean[:, None] * mean[None, :]


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
        u = rng.standard_normal((k // 2, width), dtype=fnp.float32)
        u64 = fnp.asarray(u, dtype=fnp.float64)
        covariance = (u64.T @ u64) / float(k // 2)
        eigenvalues, eigenvectors = fnp.linalg.eigh(covariance)
        eigenvalues = fnp.maximum(eigenvalues, 1e-6)
        inverse_sqrt = (eigenvectors / fnp.sqrt(eigenvalues)) @ eigenvectors.T
        half = fnp.asarray(u64 @ inverse_sqrt, dtype=fnp.float32)
        return half

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
        first_projection = self._samples @ first_weight
        x = fnp.concatenate(
            [fnp.maximum(first_projection, 0.0), fnp.maximum(-first_projection, 0.0)],
            axis=0,
        )
        if depth == 1:
            final_mean = first_mean
            zero_rows = fnp.zeros((depth - 1, width), dtype=fnp.float32)
            return fnp.concatenate([zero_rows, final_mean[None, :]], axis=0)
        sample_mean = fnp.mean(x, axis=0)
        sample_second = fnp.mean(x * x, axis=0)
        sample_variance = fnp.maximum(sample_second - sample_mean * sample_mean, 1e-6)
        target_variance = (
            first_variance
            * (0.5 - _INV_SQRT_2PI * _INV_SQRT_2PI)
            * _FIRST_VARIANCE_FACTOR
        )
        x = (x - sample_mean) * fnp.sqrt(target_variance / sample_variance) + first_mean
        if depth > 1:
            first_covariance = _first_relu_covariance(first_weight, first_variance, first_mean)
            second_weight = mlp.weights[1]
            z = x @ second_weight
            target_mean = first_mean @ second_weight
            covariance_weight = first_covariance @ second_weight
            target_variance = fnp.maximum(
                fnp.sum(second_weight * covariance_weight, axis=0) * _SECOND_VARIANCE_FACTOR,
                1e-8,
            )
            z_mean = fnp.mean(z, axis=0)
            z_second = fnp.mean(z * z, axis=0)
            z_variance = fnp.maximum(z_second - z_mean * z_mean, 1e-8)
            x = fnp.maximum(
                (z - z_mean) * fnp.sqrt(target_variance / z_variance) + target_mean,
                0.0,
            )

        for weight in mlp.weights[2:]:
            x = fnp.maximum(x @ weight, 0.0)

        final_mean = fnp.asarray(
            fnp.mean(fnp.asarray(x, dtype=fnp.float64), axis=0),
            dtype=fnp.float32,
        )
        zero_rows = fnp.zeros((depth - 1, width), dtype=fnp.float32)
        return fnp.concatenate([zero_rows, final_mean[None, :]], axis=0)
