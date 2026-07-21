from __future__ import annotations

import flopscope.numpy as fnp
from whestbench import BaseEstimator


_INV_SQRT_2PI = 0.3989422804014327
_SQRT_2_OVER_PI = 0.7978845608028654
_BUDGET_FRACTION = 0.0925
_SEED_OFFSET = -1
_WHITEN_POWER = 0.458
_RADIUS_POWER = 1.0
_FIRST_MEAN_POWER = 0.94
_FIRST_SECOND_POWER = 0.18
_FINAL_GAUSSIAN_BLEND = 0.11
_FINAL_SKEW_WEIGHT = 3.8
_FINAL_SCALE = 0.999895
_RADIAL_BLEND = 2.70
_K_ADJUST = -20
_SAMPLE_DTYPE = fnp.float32
_MIN_SAMPLES = 32
_MAX_SAMPLES = 1_000_000
_ARRAY_BYTES_LIMIT = 160 * 1024 * 1024
_WORST_CASE_ITEMSIZE = 8


def _normal_cdf_approx(x):
    y = _SQRT_2_OVER_PI * (x + 0.044715 * x * x * x)
    return 0.5 * (1.0 + fnp.tanh(y))


def _gaussian_relu_mean(mu, var):
    var = fnp.maximum(var, 1e-20)
    sigma = fnp.sqrt(var)
    alpha = mu / sigma
    phi = _INV_SQRT_2PI * fnp.exp(-0.5 * alpha * alpha)
    return sigma * phi + mu * _normal_cdf_approx(alpha)


def _sample_count(budget: int, width: int, depth: int) -> int:
    per_sample = depth * (2 * width * width + width) + width * width
    fixed = 9 * width**3 + 2 * (2 * width**3)
    k = (int(_BUDGET_FRACTION * budget) - fixed) // max(per_sample, 1)
    max_k_by_bytes = _ARRAY_BYTES_LIMIT // max(width * _WORST_CASE_ITEMSIZE, 1)
    k = int(max(_MIN_SAMPLES, min(_MAX_SAMPLES, max_k_by_bytes, k)))
    k = k + _K_ADJUST
    return k - (k & 1)


class Estimator(BaseEstimator):
    def predict(self, mlp, budget):
        width = mlp.width
        depth = mlp.depth
        k = _sample_count(budget, width, depth)
        rng = fnp.random.default_rng(int(mlp.seed) + _SEED_OFFSET)

        batch_half = k // 2
        first_var = fnp.sum(mlp.weights[0] * mlp.weights[0], axis=0)
        exact_first_mean = fnp.sqrt(fnp.maximum(first_var, 1e-20)) * _INV_SQRT_2PI
        u = rng.standard_normal((batch_half, width), dtype=_SAMPLE_DTYPE)
        covariance = (u.T @ u) / float(batch_half)
        eigenvalues, eigenvectors = fnp.linalg.eigh(covariance)
        eigenvalues = fnp.maximum(eigenvalues, 1e-6)
        inverse_power = (eigenvectors / (eigenvalues ** _WHITEN_POWER)) @ eigenvectors.T
        y = u @ inverse_power
        original_radius = fnp.sqrt(fnp.maximum(fnp.sum(u * u, axis=1, keepdims=True), 1e-20))
        whitened_radius = fnp.sqrt(fnp.maximum(fnp.sum(y * y, axis=1, keepdims=True), 1e-20))
        y = y * ((original_radius / whitened_radius) ** _RADIUS_POWER)
        input_radius = fnp.sqrt(fnp.maximum(fnp.sum(y * y, axis=1, keepdims=True), 1e-20))
        radial_scale = fnp.mean(original_radius) / input_radius
        row_radial_scale = fnp.concatenate([radial_scale, radial_scale], axis=0)
        row_weight = (1.0 - _RADIAL_BLEND) + _RADIAL_BLEND * row_radial_scale

        z = y @ mlp.weights[0]
        x = fnp.concatenate([fnp.maximum(z, 0.0), fnp.maximum(-z, 0.0)], axis=0)
        sample_first_mean = fnp.mean(x, axis=0)
        first_scale = exact_first_mean / fnp.maximum(sample_first_mean, 1e-20)
        x = x * (first_scale ** _FIRST_MEAN_POWER)
        exact_first_second = first_var * 0.5
        sample_first_second = fnp.mean(x * x, axis=0)
        second_scale = fnp.sqrt(exact_first_second / fnp.maximum(sample_first_second, 1e-20))
        x = x * (second_scale ** _FIRST_SECOND_POWER)

        for weight in mlp.weights[1:-1]:
            x = fnp.maximum(x @ weight, 0.0)

        z = x @ mlp.weights[-1]
        z = z * row_weight
        z_mean = fnp.mean(z, axis=0)
        z_second = fnp.mean(z * z, axis=0)
        z_var = fnp.maximum(z_second - z_mean * z_mean, 1e-20)
        gaussian_final_mean = _gaussian_relu_mean(z_mean, z_var)
        final_relu = fnp.maximum(z, 0.0)
        sample_final_mean = fnp.mean(final_relu, axis=0)
        z_centered = z - z_mean
        z_skew = fnp.abs(fnp.mean(z_centered * z_centered * z_centered, axis=0))
        z_skew = z_skew / (z_var * fnp.sqrt(z_var))
        final_blend = _FINAL_GAUSSIAN_BLEND / (1.0 + _FINAL_SKEW_WEIGHT * z_skew)
        final_mean = (
            (1.0 - final_blend) * sample_final_mean
            + final_blend * gaussian_final_mean
        )
        final_mean = final_mean * _FINAL_SCALE
        zero_rows = fnp.zeros((depth - 1, width), dtype=fnp.float32)
        return fnp.concatenate([zero_rows, final_mean[None, :]], axis=0)
