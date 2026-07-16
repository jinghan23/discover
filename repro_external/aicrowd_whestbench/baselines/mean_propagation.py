"""Diagonal mean/variance propagation baseline for ARC WhestBench.

This is the starter-kit default described in ``docs/how-to/algorithm-ideas.md``.
It treats each layer's pre-activations as independent Gaussians and propagates
the exact first two marginal moments through ReLU.
"""

from __future__ import annotations

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import BaseEstimator


_MIN_VARIANCE = 1e-12


class Estimator(BaseEstimator):
    """Propagate per-neuron means and diagonal variances."""

    def predict(self, mlp, budget: int) -> fnp.ndarray:
        del budget

        mean = fnp.zeros(mlp.width)
        variance = fnp.ones(mlp.width)
        layer_means = []

        for weights in mlp.weights:
            pre_mean = weights.T @ mean
            pre_variance = fnp.maximum(
                (weights * weights).T @ variance,
                _MIN_VARIANCE,
            )
            pre_std = fnp.sqrt(pre_variance)

            alpha = pre_mean / pre_std
            density = flops.stats.norm.pdf(alpha)
            probability = flops.stats.norm.cdf(alpha)

            mean = pre_mean * probability + pre_std * density
            second_moment = (
                (pre_mean * pre_mean + pre_variance) * probability
                + pre_mean * pre_std * density
            )
            variance = fnp.maximum(second_moment - mean * mean, 0.0)
            layer_means.append(mean)

        return fnp.stack(layer_means, axis=0)
