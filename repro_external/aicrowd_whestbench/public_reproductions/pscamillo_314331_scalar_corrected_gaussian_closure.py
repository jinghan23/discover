"""Reproduction of pscamillo submission #314331.

The public write-up describes the official second-order Gaussian-closure
(full-covariance gain) estimator with a scalar correction on the scored final
layer.  The cross-validated coefficient reported in the write-up is 0.9921.
"""

from __future__ import annotations

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import BaseEstimator


FINAL_LAYER_CORRECTION = 0.9921
_EPS = 1e-12


class _GaussianClosureEstimator(BaseEstimator):
    final_layer_correction = 1.0

    def predict(self, mlp, budget: int) -> fnp.ndarray:
        del budget

        mu = fnp.zeros(mlp.width)
        covariance = fnp.eye(mlp.width)
        rows = []

        for layer_index, weights in enumerate(mlp.weights):
            pre_mean = weights.T @ mu
            pre_covariance = fnp.einsum(
                "ij,ia,jb->ab", covariance, weights, weights
            )
            pre_variance = fnp.maximum(fnp.diag(pre_covariance), _EPS)
            pre_std = fnp.sqrt(pre_variance)

            alpha = pre_mean / pre_std
            density = flops.stats.norm.pdf(alpha)
            probability = flops.stats.norm.cdf(alpha)

            mu = pre_mean * probability + pre_std * density
            second_moment = (
                (pre_mean * pre_mean + pre_variance) * probability
                + pre_mean * pre_std * density
            )
            post_variance = fnp.maximum(second_moment - mu * mu, 0.0)

            # The gain approximation is the official k=2 covariance baseline:
            # exact marginal ReLU variances and Gaussian linear-response gains
            # for off-diagonal covariance.
            gain = fnp.where(pre_std > _EPS, probability, 0.0)
            covariance = fnp.outer(gain, gain) * pre_covariance
            fnp.fill_diagonal(covariance, post_variance)

            if layer_index == mlp.depth - 1:
                rows.append(mu * self.final_layer_correction)
            else:
                rows.append(mu)

        return fnp.stack(rows, axis=0)


class UncorrectedEstimator(_GaussianClosureEstimator):
    """The k=2 baseline used in the public write-up's ablation."""


class Estimator(_GaussianClosureEstimator):
    """Submission #314331: k=2 closure with the 0.9921 final-layer factor."""

    final_layer_correction = FINAL_LAYER_CORRECTION
