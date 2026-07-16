from __future__ import annotations

import flopscope.numpy as fnp
import flopscope as flops
from whestbench import BaseEstimator


class Estimator(BaseEstimator):
    def _diag(self, mlp):
        mu = fnp.zeros(mlp.width)
        var = fnp.ones(mlp.width)
        rows = []
        for w in mlp.weights:
            mu_pre = w.T @ mu
            var_pre = (w * w).T @ var
            var_pre = fnp.maximum(var_pre, 1e-12)
            sigma_pre = fnp.sqrt(var_pre)
            alpha = mu_pre / sigma_pre
            phi = flops.stats.norm.pdf(alpha)
            cdf = flops.stats.norm.cdf(alpha)
            mu = mu_pre * cdf + sigma_pre * phi
            ez2 = (mu_pre * mu_pre + var_pre) * cdf + mu_pre * sigma_pre * phi
            var = fnp.maximum(ez2 - mu * mu, 0.0)
            rows.append(mu)
        return fnp.stack(rows, axis=0)

    def _cov(self, mlp):
        width = int(mlp.width)
        pi = 3.141592653589793
        inv2pi = 1.0 / (2.0 * pi)
        denom = 0.5 - inv2pi
        eye = fnp.eye(width, dtype=fnp.float64)
        mu = fnp.zeros(width, dtype=fnp.float64)
        cov = eye
        rows = []
        for w0 in mlp.weights:
            w = fnp.asarray(w0, dtype=fnp.float64)
            mu_pre = w.T @ mu
            cov_pre = w.T @ cov @ w
            var_pre = fnp.maximum(fnp.diagonal(cov_pre), 1e-12)
            sigma_pre = fnp.sqrt(var_pre)
            alpha = mu_pre / sigma_pre
            phi = flops.stats.norm.pdf(alpha)
            cdf = flops.stats.norm.cdf(alpha)
            mu = mu_pre * cdf + sigma_pre * phi
            ez2 = (mu_pre * mu_pre + var_pre) * cdf + mu_pre * sigma_pre * phi
            var = fnp.maximum(ez2 - mu * mu, 1e-12)

            scale = sigma_pre[:, None] * sigma_pre[None, :]
            rho = fnp.clip(cov_pre / fnp.maximum(scale, 1e-12), -0.999999, 0.999999)
            root = fnp.sqrt(fnp.maximum(1.0 - rho * rho, 0.0))
            kernel = (root + (pi - fnp.arccos(rho)) * rho) * inv2pi
            corr = (kernel - inv2pi) / denom
            out_scale = fnp.sqrt(var[:, None] * var[None, :])
            cov = corr * out_scale
            cov = cov * (1.0 - eye) + eye * var[None, :]
            rows.append(mu)
        return fnp.asarray(fnp.stack(rows, axis=0), dtype=fnp.float32)

    def _cov_quad(self, mlp):
        width = int(mlp.width)
        nodes = fnp.array(
            [
                -4.1445471861258945,
                -2.802485861287542,
                -1.6365190424351082,
                -0.5390798113513752,
                0.5390798113513752,
                1.6365190424351082,
                2.802485861287542,
                4.1445471861258945,
            ],
            dtype=fnp.float64,
        )
        weights = fnp.array(
            [
                0.0001126145383753679,
                0.009635220120788263,
                0.117239907661759,
                0.3730122576790775,
                0.3730122576790775,
                0.117239907661759,
                0.009635220120788263,
                0.0001126145383753679,
            ],
            dtype=fnp.float64,
        )
        eye = fnp.eye(width, dtype=fnp.float64)
        mu = fnp.zeros(width, dtype=fnp.float64)
        cov = eye
        rows = []
        for w0 in mlp.weights:
            w = fnp.asarray(w0, dtype=fnp.float64)
            mu_pre = w.T @ mu
            cov_pre = w.T @ cov @ w
            var_pre = fnp.maximum(fnp.diagonal(cov_pre), 1e-12)
            sigma = fnp.sqrt(var_pre)
            alpha = mu_pre / sigma
            phi = flops.stats.norm.pdf(alpha)
            cdf = flops.stats.norm.cdf(alpha)
            mu = mu_pre * cdf + sigma * phi
            ez2 = (mu_pre * mu_pre + var_pre) * cdf + mu_pre * sigma * phi
            var = fnp.maximum(ez2 - mu * mu, 1e-12)

            rho = cov_pre / fnp.maximum(sigma[:, None] * sigma[None, :], 1e-12)
            rho = fnp.clip(rho, -0.999999, 0.999999)
            cond_sigma = sigma[None, :] * fnp.sqrt(fnp.maximum(1.0 - rho * rho, 1e-12))
            second = fnp.zeros((width, width), dtype=fnp.float64)
            for node, weight in zip(nodes, weights):
                xplus = fnp.maximum(mu_pre + sigma * node, 0.0)
                cond_mu = mu_pre[None, :] + rho * (sigma[None, :] * node)
                cond_alpha = cond_mu / cond_sigma
                cond_relu = cond_mu * flops.stats.norm.cdf(cond_alpha) + cond_sigma * flops.stats.norm.pdf(
                    cond_alpha
                )
                second = second + weight * xplus[:, None] * cond_relu
            second = 0.5 * (second + second.T)
            second = second * (1.0 - eye) + eye * ez2[None, :]
            cov = second - mu[:, None] * mu[None, :]
            cov = 0.5 * (cov + cov.T)
            rows.append(mu)
        return fnp.asarray(fnp.stack(rows, axis=0), dtype=fnp.float32)

    def predict(self, mlp, budget):
        width = int(mlp.width)
        depth = int(mlp.depth)
        rng = fnp.random.default_rng(int(mlp.seed))

        # About 10% of the FLOP budget for width=256/depth=32.  Below that
        # floor the official multiplier is fixed at 0.1, so this is the best
        # direct-MC point before extra compute starts scaling the score.
        per_sample = max(1, depth * (2 * width * width + 4 * width))
        n_samples = max(1024, int((0.0909908329 * float(budget)) / float(per_sample)))
        n_samples = min(n_samples, 8192)
        d_mu = fnp.zeros(width)
        d_var = fnp.ones(width)
        diag_rows = []
        diag_vars = []
        pre_mus = []
        pre_vars = []
        for w in mlp.weights:
            d_mu_pre = w.T @ d_mu
            d_var_pre = (w * w).T @ d_var
            d_var_pre = fnp.maximum(d_var_pre, 1e-12)
            d_sigma = fnp.sqrt(d_var_pre)
            d_alpha = d_mu_pre / d_sigma
            d_phi = flops.stats.norm.pdf(d_alpha)
            d_cdf = flops.stats.norm.cdf(d_alpha)
            d_mu = d_mu_pre * d_cdf + d_sigma * d_phi
            d_ez2 = (d_mu_pre * d_mu_pre + d_var_pre) * d_cdf + d_mu_pre * d_sigma * d_phi
            d_var = fnp.maximum(d_ez2 - d_mu * d_mu, 0.0)
            pre_mus.append(d_mu_pre)
            pre_vars.append(d_var_pre)
            diag_rows.append(d_mu)
            diag_vars.append(d_var)
        sums = [fnp.zeros(width, dtype=fnp.float64) for _ in range(depth)]
        x = fnp.asarray(rng.standard_normal((n_samples, width), dtype=fnp.float32))
        x = x - fnp.mean(x, axis=0)
        x = x / fnp.sqrt(fnp.maximum(fnp.mean(x * x, axis=0), 1e-12))
        for i, w in enumerate(mlp.weights[:-1]):
            z = x @ w
            if i < 1:
                z_center = z - fnp.mean(z, axis=0)
                sample_cov = (z_center.T @ z_center) / float(n_samples)
                s_vals, s_vecs = fnp.linalg.eigh(sample_cov)
                t_cov = w.T @ w
                t_vals, t_vecs = fnp.linalg.eigh(t_cov)
                inv_sqrt = s_vecs @ (fnp.diag(1.0 / fnp.sqrt(fnp.maximum(s_vals, 1e-12))) @ s_vecs.T)
                target_sqrt = t_vecs @ (fnp.diag(fnp.sqrt(fnp.maximum(t_vals, 1e-12))) @ t_vecs.T)
                z = z_center @ (inv_sqrt @ target_sqrt)
            x = fnp.maximum(z, 0.0)
            if i == 0:
                x_center = x - fnp.mean(x, axis=0)
                x_var = fnp.maximum(fnp.mean(x_center * x_center, axis=0), 1e-12)
                x_matched = x_center * fnp.sqrt(fnp.maximum(diag_vars[i], 1e-12) / x_var) + diag_rows[i]
                x = -1.6 * x + 2.6 * x_matched
            sums[i] = fnp.sum(fnp.asarray(x, dtype=fnp.float64), axis=0)
        z = x @ mlp.weights[-1]
        final_act = fnp.asarray(fnp.maximum(z, 0.0), dtype=fnp.float64)
        sums[-1] = fnp.sum(final_act, axis=0)

        mc = fnp.asarray(fnp.stack(sums, axis=0) / float(n_samples), dtype=fnp.float32)
        diag = fnp.asarray(fnp.stack(diag_rows, axis=0), dtype=fnp.float32)
        return fnp.asarray(0.967 * mc + 0.033 * diag, dtype=fnp.float32)
