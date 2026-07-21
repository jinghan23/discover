from __future__ import annotations

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import BaseEstimator


class Estimator(BaseEstimator):
    def predict(self, mlp, budget):
        width = int(mlp.width)
        depth = int(mlp.depth)
        rng = fnp.random.default_rng(int(mlp.seed))

        per_sample = max(1, depth * (2 * width * width + 4 * width))
        n_samples = max(1024, int((0.0937 * float(budget)) / float(per_sample)))
        n_samples = min(n_samples, 8192)
        n_samples = max(2, 2 * ((n_samples + 1) // 2))

        mu = fnp.zeros(width)
        var = fnp.ones(width)
        act_cov = None
        diag_rows = []
        diag_vars = []

        for layer_idx, w in enumerate(mlp.weights):
            mu_pre = w.T @ mu
            if act_cov is None:
                var_pre = fnp.maximum((w * w).T @ var, 1e-12)
            else:
                cov_w = act_cov @ w
                var_pre = fnp.maximum(fnp.sum(w * cov_w, axis=0), 1e-12)
                act_cov = None

            sigma = fnp.sqrt(var_pre)
            alpha = mu_pre / sigma
            phi = flops.stats.norm.pdf(alpha)
            cdf = flops.stats.norm.cdf(alpha)
            mu = mu_pre * cdf + sigma * phi
            ez2 = (mu_pre * mu_pre + var_pre) * cdf + mu_pre * sigma * phi
            var = fnp.maximum(ez2 - mu * mu, 0.0)

            if layer_idx == 0:
                pre_cov = w.T @ w
                denom = sigma[:, None] * sigma[None, :]
                rho = fnp.maximum(fnp.minimum(pre_cov / fnp.maximum(denom, 1e-12), 1.0), -1.0)
                root = fnp.sqrt(fnp.maximum(1.0 - rho * rho, 0.0))
                relu_second = denom * (root + (3.141592653589793 - fnp.arccos(rho)) * rho) / (
                    2.0 * 3.141592653589793
                )
                act_cov = fnp.maximum(relu_second - mu[:, None] * mu[None, :], 0.0)

            diag_rows.append(mu)
            diag_vars.append(var)

        half = n_samples // 2
        base = fnp.asarray(rng.standard_normal((half, width), dtype=fnp.float32))
        cov = (base.T @ base) / float(half)
        x = fnp.concatenate((base, -base), axis=0)
        chol = fnp.linalg.cholesky(cov)
        x = fnp.linalg.solve(chol, x.T).T

        for i, w in enumerate(mlp.weights[:-1]):
            x = fnp.maximum(x @ w, 0.0)
            if i == 0:
                x_mean = fnp.mean(x, axis=0)
                x_var = fnp.maximum(fnp.mean(x * x, axis=0) - x_mean * x_mean, 1e-12)
                x_scale = fnp.sqrt(fnp.maximum(diag_vars[i], 1e-12) / x_var)
                x = x * (-1.65 + 2.65 * x_scale) + 2.65 * (diag_rows[i] - x_mean * x_scale)
            elif i == 1:
                x_mean = fnp.mean(x, axis=0)
                x_var = fnp.maximum(fnp.mean(x * x, axis=0) - x_mean * x_mean, 1e-12)
                x_scale = fnp.sqrt(fnp.maximum(diag_vars[i], 1e-12) / x_var)
                x = x * (0.695 + 0.305 * x_scale) + 0.305 * (diag_rows[i] - x_mean * x_scale)
            elif i == 2:
                x_mean = fnp.mean(x, axis=0)
                x_var = fnp.maximum(fnp.mean(x * x, axis=0) - x_mean * x_mean, 1e-12)
                x_scale = fnp.sqrt(fnp.maximum(diag_vars[i], 1e-12) / x_var)
                x = x * (0.965 + 0.035 * x_scale) + 0.035 * (diag_rows[i] - x_mean * x_scale)
            elif i == 3:
                x_mean = fnp.mean(x, axis=0)
                x_var = fnp.maximum(fnp.mean(x * x, axis=0) - x_mean * x_mean, 1e-12)
                x_scale = fnp.sqrt(fnp.maximum(diag_vars[i], 1e-12) / x_var)
                x = x * (1.05 - 0.05 * x_scale) - 0.05 * (diag_rows[i] - x_mean * x_scale)
            elif i == 4:
                x_mean = fnp.mean(x, axis=0)
                x_var = fnp.maximum(fnp.mean(x * x, axis=0) - x_mean * x_mean, 1e-12)
                x_scale = fnp.sqrt(fnp.maximum(diag_vars[i], 1e-12) / x_var)
                x = x * (0.9825 + 0.0175 * x_scale) + 0.0175 * (diag_rows[i] - x_mean * x_scale)

        z = x @ mlp.weights[-1]
        final_mc = fnp.asarray(fnp.mean(fnp.maximum(z, 0.0), axis=0), dtype=fnp.float32)

        diag = fnp.asarray(fnp.stack(diag_rows, axis=0), dtype=fnp.float32)
        final_pred = fnp.asarray(final_mc + 0.001 * (diag[-1] - final_mc), dtype=fnp.float32)
        return fnp.asarray(fnp.concatenate((diag[:-1], final_pred[None, :]), axis=0), dtype=fnp.float32)
