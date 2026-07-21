from __future__ import annotations

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import BaseEstimator


_BUDGET_FRACTION = 0.22
_MIN_SAMPLES = 32
_MAX_SAMPLES = 1_000_000
_ARRAY_BYTES_LIMIT = 100 * 1024 * 1024
_WORST_CASE_ITEMSIZE = 8


def _sample_count(budget: int, width: int, depth: int) -> int:
    per_sample = depth * (2 * width * width + width) + width * width + 48 * width
    fixed = 9 * width**3 + 2 * (2 * width**3)
    k = (int(_BUDGET_FRACTION * budget) - fixed) // max(per_sample, 1)
    max_k_by_bytes = _ARRAY_BYTES_LIMIT // max(width * _WORST_CASE_ITEMSIZE, 1)
    k = int(max(_MIN_SAMPLES, min(_MAX_SAMPLES, max_k_by_bytes, k)))
    return k - (k & 1)


class Estimator(BaseEstimator):
    def predict(self, mlp, budget):
        width = mlp.width
        depth = mlp.depth
        k = _sample_count(budget, width, depth)
        half = k // 2
        rng = fnp.random.default_rng(mlp.seed)

        ranks = fnp.argsort(rng.random((half, width)), axis=0)
        probs = (fnp.asarray(ranks, dtype=fnp.float32) + 0.5) / float(half)
        u = flops.stats.norm.ppf(probs).astype(fnp.float32)
        x = fnp.concatenate([u, -u], axis=0)

        covariance = (u.T @ u) / float(half)
        eigenvalues, eigenvectors = fnp.linalg.eigh(covariance)
        eigenvalues = fnp.maximum(eigenvalues, 1e-6)
        inverse_sqrt = (eigenvectors / fnp.sqrt(eigenvalues)) @ eigenvectors.T
        folded_first_weight = inverse_sqrt @ mlp.weights[0]

        x = fnp.maximum(x @ folded_first_weight, 0.0)
        for weight in mlp.weights[1:]:
            x = fnp.maximum(x @ weight, 0.0)

        final_mean = fnp.mean(x, axis=0)
        zero_rows = fnp.zeros((depth - 1, width), dtype=fnp.float32)
        return fnp.concatenate([zero_rows, final_mean[None, :]], axis=0)
