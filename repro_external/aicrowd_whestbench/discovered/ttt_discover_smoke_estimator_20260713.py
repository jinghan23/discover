from __future__ import annotations

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import BaseEstimator


class Estimator(BaseEstimator):
    def predict(self, mlp, budget):
        del budget

        n = mlp.width
        eps = 1e-12

        mu = fnp.zeros(n)
        cov = fnp.eye(n)
        rows = []

        for w in mlp.weights:
            m = w.T @ mu
            cov_z = w.T @ cov @ w
            v = fnp.maximum(fnp.diag(cov_z), eps)
            s = fnp.sqrt(v)

            a = m / s
            pdf = flops.stats.norm.pdf(a)
            cdf = flops.stats.norm.cdf(a)

            mu_y = m * cdf + s * pdf
            rows.append(mu_y)

            # Full Gaussian-closure covariance propagation through ReLU.
            rho = cov_z / (s[:, None] * s[None, :])
            rho = fnp.maximum(fnp.minimum(rho, 0.999999), -0.999999)

            ai = a[:, None]
            aj = a[None, :]
            ri = rho
            omr2 = fnp.maximum(1.0 - ri * ri, eps)
            q = fnp.sqrt(omr2)

            bi = (aj - ri * ai) / q
            bj = (ai - ri * aj) / q

            Phi_i = cdf[:, None]
            Phi_j = cdf[None, :]
            phi_i = pdf[:, None]
            phi_j = pdf[None, :]

            # P[X>0,Y>0] for standardized bivariate normal, using a cheap
            # Plackett integral quadrature over correlation. Accuracy is most
            # important for off-diagonal covariance; diagonal is overwritten.
            r_nodes = fnp.stack(
                (
                    0.06943184420297371 * ri,
                    0.33000947820757187 * ri,
                    0.6699905217924281 * ri,
                    0.9305681557970262 * ri,
                ),
                axis=0,
            )
            r_wts = fnp.array(
                (
                    0.17392742256872692,
                    0.32607257743127305,
                    0.32607257743127305,
                    0.17392742256872692,
                )
            )[:, None, None]
            den = fnp.maximum(1.0 - r_nodes * r_nodes, eps)
            expo = -(
                ai[None, :, :] * ai[None, :, :]
                - 2.0 * r_nodes * ai[None, :, :] * aj[None, :, :]
                + aj[None, :, :] * aj[None, :, :]
            ) / (2.0 * den)
            integ = fnp.sum(r_wts * fnp.exp(expo) / (2.0 * 3.141592653589793 * fnp.sqrt(den)), axis=0)
            ppos = Phi_i * Phi_j + ri * integ

            term = (
                (m[:, None] * m[None, :] + cov_z) * ppos
                + m[:, None] * s[None, :] * phi_j * flops.stats.norm.cdf(bj)
                + m[None, :] * s[:, None] * phi_i * flops.stats.norm.cdf(bi)
                + s[:, None] * s[None, :] * q * phi_i * flops.stats.norm.pdf(bi)
            )

            ez2 = (m * m + v) * cdf + m * s * pdf
            term = term * (1.0 - fnp.eye(n)) + fnp.diag(ez2)

            cov = term - mu_y[:, None] * mu_y[None, :]
            diag = fnp.maximum(fnp.diag(cov), eps)
            cov = cov * (1.0 - fnp.eye(n)) + fnp.diag(diag)
            cov = 0.5 * (cov + cov.T)
            mu = mu_y

        return fnp.stack(rows, axis=0)
