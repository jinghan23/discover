from __future__ import annotations

import flopscope.numpy as fnp
from whestbench import BaseEstimator


_SAMPLE_FRACTION = 0.105
_INV_SQRT_2PI = 0.3989422804014327
_MIN_SAMPLES = 32
_MAX_SAMPLES = 1_000_000
_ARRAY_BYTES_LIMIT = 100 * 1024 * 1024
_WORST_CASE_ITEMSIZE = 8


def _sample_count(fraction: float, budget: int, width: int, depth: int) -> int:
    per_sample = (2 * depth - 1) * width * width + depth * width + width * width
    fixed = 9 * width**3 + 2 * (2 * width**3)
    k = (int(fraction * budget) - fixed) // max(per_sample, 1)
    max_k_by_bytes = _ARRAY_BYTES_LIMIT // max(width * _WORST_CASE_ITEMSIZE, 1)
    min_k = max(_MIN_SAMPLES, 2 * width + 2)
    k = int(max(min_k, min(_MAX_SAMPLES, max_k_by_bytes, k)))
    return k - (k & 1)


def _estimate_block(mlp, rng, k: int):
    width = mlp.width
    u = rng.standard_normal((k // 2, width), dtype=fnp.float32)
    u = u[:, ::-1]

    covariance = (u.T @ u) / float(k // 2)
    cholesky = fnp.linalg.cholesky(covariance)
    folded_first_weight = fnp.linalg.solve(cholesky.T, mlp.weights[0])

    h = u @ folded_first_weight
    x = fnp.concatenate([fnp.maximum(h, 0.0), fnp.maximum(-h, 0.0)], axis=0)
    for weight in mlp.weights[1:]:
        x = fnp.maximum(x @ weight, 0.0)
    return fnp.mean(x, axis=0)


class Estimator(BaseEstimator):
    def predict(self, mlp, budget):
        width = mlp.width
        depth = mlp.depth

        if depth == 1:
            final_mean = _INV_SQRT_2PI * fnp.sqrt(
                fnp.sum(mlp.weights[0] * mlp.weights[0], axis=0)
            )
            return final_mean[None, :]

        rng = fnp.random.default_rng(mlp.seed)

        k_main = _sample_count(_SAMPLE_FRACTION, budget, width, depth)
        main = _estimate_block(mlp, rng, k_main)

        zero_rows = fnp.zeros((depth - 1, width), dtype=fnp.float32)
        return fnp.concatenate([zero_rows, main[None, :]], axis=0)
