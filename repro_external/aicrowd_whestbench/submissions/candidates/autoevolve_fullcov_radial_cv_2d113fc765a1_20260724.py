from __future__ import annotations

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import BaseEstimator


class Estimator(BaseEstimator):
    """Full linear covariance with a fast first-order ReLU closure."""

    def predict(self, mlp, budget):
        del budget
        width = mlp.width
        mu = fnp.zeros(width)
        cov = fnp.eye(width)
        rows = []

        # The network is positively homogeneous.  Integrating the Gaussian
        # radius analytically leaves only the direction to be sampled.
        rng = fnp.random.default_rng(mlp.seed)
        half_samples = 3000
        probes = rng.standard_normal((half_samples, width))
        radius_mean = fnp.sqrt(float(width)) * (
            1.0
            - 1.0 / (4.0 * width)
            + 1.0 / (32.0 * width * width)
            + 5.0 / (128.0 * width * width * width)
        )
        norms = fnp.sqrt(fnp.sum(probes * probes, axis=1, keepdims=True))
        probes = probes * (radius_mean / fnp.maximum(norms, 1e-12))
        samples = fnp.concatenate((probes, -probes), axis=0)
        radial_second_ratio = (radius_mean * radius_mean) / float(width)

        for layer_index, w in enumerate(mlp.weights):
            sample_pre = samples @ w
            samples = fnp.maximum(sample_pre, 0.0)

            mu_pre = w.T @ mu
            cov_pre = w.T @ cov @ w
            cov_pre = 0.5 * (cov_pre + cov_pre.T)

            var_pre = fnp.maximum(fnp.diagonal(cov_pre), 1e-12)
            sigma = fnp.sqrt(var_pre)
            a = mu_pre / sigma
            cdf = flops.stats.norm.cdf(a)
            pdf = flops.stats.norm.pdf(a)

            gaussian_mu = mu_pre * cdf + sigma * pdf
            ez2 = (mu_pre * mu_pre + var_pre) * cdf + mu_pre * sigma * pdf

            # ReLU's best degree-two Gaussian control variate.  The second
            # moment is adjusted for fixed-radius directional sampling.
            sample_pre_mean = fnp.mean(sample_pre, axis=0)
            sample_relu_mean = fnp.mean(samples, axis=0)
            sample_center2 = fnp.mean(
                (sample_pre - mu_pre) * (sample_pre - mu_pre), axis=0
            )
            directional_center2 = (
                radial_second_ratio * (var_pre + mu_pre * mu_pre)
                - mu_pre * mu_pre
            )
            cv_mean = (
                sample_relu_mean
                - cdf * (sample_pre_mean - mu_pre)
                - (pdf / (2.0 * sigma))
                * (sample_center2 - directional_center2)
            )
            if layer_index == 0:
                mu = gaussian_mu
            else:
                mu = 0.2 * gaussian_mu + 0.8 * cv_mean
            var = fnp.maximum(ez2 - gaussian_mu * gaussian_mu, 0.0)

            cov = cov_pre * fnp.outer(cdf, cdf)
            cov = cov - fnp.diag(fnp.diagonal(cov)) + fnp.diag(var)
            cov = 0.5 * (cov + cov.T)
            rows.append(mu)

        return fnp.stack(rows, axis=0)
