from __future__ import annotations

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import BaseEstimator

_FRAC = 0.09
_BETA0 = 0.50
_BETA1 = 0.60
_RADIAL = True
_SEED_OFFSET = 0
_FULL_VAR1 = True
_SKEW1 = 1.0
_SKEW2 = 1.0


def _first_relu_cov(w, width):
    cov_pre = w.T @ w
    var_pre = fnp.maximum(fnp.diag(cov_pre), 1e-12)
    sigma = fnp.sqrt(var_pre)
    denom = fnp.maximum(sigma[:, None] * sigma[None, :], 1e-12)
    rho = fnp.maximum(fnp.minimum(cov_pre / denom, 0.999999), -0.999999)
    kernel = (
        fnp.sqrt(fnp.maximum(1.0 - rho * rho, 0.0))
        + (3.141592653589793 - fnp.arccos(rho)) * rho
    ) * 0.15915494309189535
    second = denom * kernel
    mean = sigma * 0.3989422804014327
    return second - mean[:, None] * mean[None, :]


class Estimator(BaseEstimator):
    def predict(self, mlp, budget):
        width = int(mlp.width)
        depth = int(mlp.depth)
        rng = fnp.random.default_rng(int(mlp.seed) + _SEED_OFFSET)

        per_sample = max(1, depth * (2 * width * width + 4 * width))
        n_samples = max(1024, int((_FRAC * float(budget)) / float(per_sample)))
        n_samples = min(n_samples, 8192)
        n_samples = max(2, 2 * (n_samples // 2))

        mu = fnp.zeros(width)
        var = fnp.ones(width)
        diag_rows = []
        diag_vars = []
        first_cov = None
        prev_tau3 = None
        for layer_i, w in enumerate(mlp.weights):
            mu_pre = w.T @ mu
            if layer_i == 1 and first_cov is not None:
                var_pre = fnp.maximum(fnp.sum(w * (first_cov @ w), axis=0), 1e-12)
            else:
                var_pre = fnp.maximum((w * w).T @ var, 1e-12)
            sigma = fnp.sqrt(var_pre)
            alpha = mu_pre / sigma
            phi = flops.stats.norm.pdf(alpha)
            cdf = flops.stats.norm.cdf(alpha)
            mu = mu_pre * cdf + sigma * phi
            skew = None
            if layer_i == 1 and prev_tau3 is not None and _SKEW1 != 0.0:
                kappa3 = fnp.sum((w * w * w) * prev_tau3[:, None], axis=0)
                skew = kappa3 / fnp.maximum(var_pre * sigma, 1e-12)
                mu = mu - _SKEW1 * (mu_pre * phi * skew) * 0.16666666666666666
            ez2 = (mu_pre * mu_pre + var_pre) * cdf + mu_pre * sigma * phi
            if layer_i == 1 and prev_tau3 is not None and _SKEW2 != 0.0:
                if skew is None:
                    kappa3 = fnp.sum((w * w * w) * prev_tau3[:, None], axis=0)
                    skew = kappa3 / fnp.maximum(var_pre * sigma, 1e-12)
                ez2 = ez2 + _SKEW2 * skew * var_pre * phi * 0.3333333333333333
            var = fnp.maximum(ez2 - mu * mu, 0.0)
            diag_rows.append(mu)
            diag_vars.append(var)
            if layer_i == 0 and _FULL_VAR1:
                first_cov = _first_relu_cov(w, width)
                prev_tau3 = (
                    sigma
                    * sigma
                    * sigma
                    * 0.3989422804014327
                    * (0.5 + 0.3183098861837907)
                )

        if depth == 1:
            return fnp.asarray(fnp.stack(diag_rows, axis=0), dtype=fnp.float32)

        half = n_samples // 2
        base = fnp.asarray(rng.standard_normal((half, width), dtype=fnp.float32))
        x = fnp.concatenate((base, -base), axis=0)
        x = x - fnp.mean(x, axis=0)

        q, _ = fnp.linalg.qr(x, mode="reduced")
        if _RADIAL:
            raw_norm = fnp.sqrt(fnp.maximum(fnp.sum(x * x, axis=1), 1e-12))[:, None]
            q_norm = fnp.sqrt(fnp.maximum(fnp.sum(q * q, axis=1), 1e-12))[:, None]
            x = q * (raw_norm / q_norm)
            x = x - fnp.mean(x, axis=0)
            x_var = fnp.maximum(fnp.mean(x * x, axis=0), 1e-12)
            x = x * fnp.sqrt(1.0 / x_var)
        else:
            x = q * fnp.sqrt(float(n_samples))

        for i, w in enumerate(mlp.weights[:-1]):
            x = fnp.maximum(x @ w, 0.0)
            if i == 0:
                x_center = x - fnp.mean(x, axis=0)
                x_var = fnp.maximum(fnp.mean(x_center * x_center, axis=0), 1e-12)
                x_match = (
                    x_center
                    * fnp.sqrt(fnp.maximum(diag_vars[i], 1e-12) / x_var)
                    + diag_rows[i]
                )
                x = (1.0 - _BETA0) * x + _BETA0 * x_match
            elif i == 1:
                x_center = x - fnp.mean(x, axis=0)
                x_var = fnp.maximum(fnp.mean(x_center * x_center, axis=0), 1e-12)
                x_match = (
                    x_center
                    * fnp.sqrt(fnp.maximum(diag_vars[i], 1e-12) / x_var)
                    + diag_rows[i]
                )
                x = (1.0 - _BETA1) * x + _BETA1 * x_match
        final_act = fnp.maximum(x @ mlp.weights[-1], 0.0)
        final_pred = fnp.asarray(
            fnp.mean(fnp.asarray(final_act, dtype=fnp.float64), axis=0),
            dtype=fnp.float32,
        )
        diag = fnp.asarray(fnp.stack(diag_rows, axis=0), dtype=fnp.float32)
        return fnp.asarray(
            fnp.concatenate((diag[:-1], final_pred[None, :]), axis=0),
            dtype=fnp.float32,
        )
