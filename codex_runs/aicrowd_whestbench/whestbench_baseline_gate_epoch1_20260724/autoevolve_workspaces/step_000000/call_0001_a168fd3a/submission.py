from __future__ import annotations

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import BaseEstimator


class Estimator(BaseEstimator):
    """Antithetic, moment-matched Latin-hypercube integration."""

    def predict(self, mlp, budget):
        del budget
        width = mlp.width
        half = 3072
        rng = fnp.random.default_rng(mlp.seed)

        # One point in every marginal stratum, paired with its exact negative.
        order = fnp.argsort(rng.random((half, width)), axis=0)
        lower = (order + 0.5) / (2.0 * half)
        choose_upper = rng.integers(0, 2, size=(half, width))
        u = fnp.where(choose_upper > 0, 1.0 - lower, lower)
        x = flops.stats.norm.ppf(u)

        sample_cov = (x.T @ x) / half
        eigval, eigvec = fnp.linalg.eigh(sample_cov)
        invsqrt = (eigvec * (1.0 / fnp.sqrt(fnp.maximum(eigval, 1e-12)))) @ eigvec.T
        x = x @ invsqrt
        x = fnp.concatenate((x, -x), axis=0)

        rows = []
        for layer, w in enumerate(mlp.weights):
            pre = x @ w
            x = fnp.maximum(pre, 0.0)
            sample_mean = fnp.mean(x, axis=0)
            if layer == 0:
                pre_var = fnp.sum(w * w, axis=0)
                exact_mean = fnp.sqrt(pre_var) * flops.stats.norm.pdf(0.0)
                exact_var = pre_var * (0.5 - 1.0 / (2.0 * fnp.pi))
                sample_var = fnp.mean((x - sample_mean) * (x - sample_mean), axis=0)
                x = exact_mean + (x - sample_mean) * fnp.sqrt(
                    exact_var / fnp.maximum(sample_var, 1e-12)
                )
                sample_mean = exact_mean
            elif layer == mlp.depth - 1:
                pre_mean = fnp.mean(pre, axis=0)
                pre_var = fnp.mean(
                    (pre - pre_mean) * (pre - pre_mean), axis=0
                )
                pre_std = fnp.sqrt(fnp.maximum(pre_var, 1e-12))
                alpha = pre_mean / pre_std
                sample_mean = (
                    pre_mean * flops.stats.norm.cdf(alpha)
                    + pre_std * flops.stats.norm.pdf(alpha)
                )
            rows.append(sample_mean)
        return fnp.stack(rows, axis=0)
