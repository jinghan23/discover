"""Dense covariance plus compressed cross-cumulant propagation for WhestBench.

Linear covariance propagation is exact.  ReLU means, variances, and pairwise
second moments use the corresponding univariate and bivariate Gaussian
formulas.  The first (zero-mean) layer uses the analytic arc-cosine kernel;
later non-central bivariate probabilities use Gauss-Legendre quadrature of
Plackett's identity.  Third- and fourth-order cumulants are carried as
symmetric CP factors, so their all-distinct cross entries survive linear
layers without materializing width**3 or width**4 arrays.  ReLU propagation
uses a delta/Edgeworth closure and injects an exact residual on every marginal.
"""

from __future__ import annotations

from math import comb

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import BaseEstimator


_MIN_VARIANCE = 1e-12
_MAX_ABS_CORRELATION = 1.0 - 1e-7
_MAX_ABS_SKEWNESS = 2.0
_MIN_EXCESS_KURTOSIS = -2.0
_MAX_EXCESS_KURTOSIS = 6.0
_MAX_CROSS_CUMULANT_RANK = 512
_PI = 3.141592653589793
_TWO_PI = 2.0 * _PI

# Eight-point Gauss-Legendre rule mapped from [-1, 1] to [0, 1].  Plackett's
# identity writes Phi_2(a, b; rho) as Phi(a) Phi(b) plus an integral from zero
# to rho, so these nodes work for positive and negative correlations alike.
_QUADRATURE_NODES = (
    0.019855071751231884,
    0.10166676129318664,
    0.2372337950418355,
    0.4082826787521751,
    0.5917173212478248,
    0.7627662049581645,
    0.8983332387068134,
    0.9801449282487681,
)
_QUADRATURE_WEIGHTS = (
    0.05061426814518813,
    0.11119051722668724,
    0.15685332293894366,
    0.181341891689181,
    0.181341891689181,
    0.15685332293894366,
    0.11119051722668724,
    0.05061426814518813,
)

# Coefficients are ordered by ascending power.  The Edgeworth correction uses
# probabilists' Hermite polynomials He_3, He_4, and He_6.
_HERMITE_3 = (0.0, -3.0, 0.0, 1.0)
_HERMITE_4 = (3.0, 0.0, -6.0, 0.0, 1.0)
_HERMITE_6 = (-15.0, 0.0, 45.0, 0.0, -15.0, 0.0, 1.0)


def _correlation_matrix(
    covariance: fnp.ndarray,
    standard_deviation: fnp.ndarray,
    limit: float,
) -> tuple[fnp.ndarray, fnp.ndarray]:
    scale = fnp.outer(standard_deviation, standard_deviation)
    correlation = covariance / fnp.maximum(scale, _MIN_VARIANCE)
    correlation = fnp.maximum(fnp.minimum(correlation, limit), -limit)
    return correlation, scale


def _zero_mean_relu_second_moment(
    covariance: fnp.ndarray,
    standard_deviation: fnp.ndarray,
) -> fnp.ndarray:
    """Return E[ReLU(X_i) ReLU(X_j)] via the arc-cosine kernel."""

    correlation, scale = _correlation_matrix(
        covariance,
        standard_deviation,
        1.0,
    )
    root = fnp.sqrt(fnp.maximum(1.0 - correlation * correlation, 0.0))
    kernel = (
        root + (_PI - fnp.arccos(correlation)) * correlation
    ) / _TWO_PI
    return scale * kernel


def _bivariate_normal_cdf(
    alpha_i: fnp.ndarray,
    alpha_j: fnp.ndarray,
    correlation: fnp.ndarray,
    marginal_probability: fnp.ndarray,
) -> fnp.ndarray:
    """Approximate Phi_2(alpha_i, alpha_j; correlation) pairwise."""

    nodes = fnp.array(_QUADRATURE_NODES)[:, None, None]
    weights = fnp.array(_QUADRATURE_WEIGHTS)[:, None, None]
    integration_correlation = nodes * correlation[None, :, :]
    denominator = fnp.maximum(
        1.0 - integration_correlation * integration_correlation,
        _MIN_VARIANCE,
    )
    exponent = -(
        alpha_i[None, :, :] * alpha_i[None, :, :]
        - 2.0
        * integration_correlation
        * alpha_i[None, :, :]
        * alpha_j[None, :, :]
        + alpha_j[None, :, :] * alpha_j[None, :, :]
    ) / (2.0 * denominator)
    integrand = fnp.exp(exponent) / (_TWO_PI * fnp.sqrt(denominator))
    joint_probability = (
        fnp.outer(marginal_probability, marginal_probability)
        + correlation * fnp.sum(weights * integrand, axis=0)
    )

    # Roundoff in extreme tails can otherwise create a tiny negative
    # probability or make an intersection larger than either marginal.
    upper_bound = fnp.minimum(
        marginal_probability[:, None],
        marginal_probability[None, :],
    )
    return fnp.maximum(fnp.minimum(joint_probability, upper_bound), 0.0)


def _noncentral_relu_second_moment(
    mean: fnp.ndarray,
    covariance: fnp.ndarray,
    standard_deviation: fnp.ndarray,
    alpha: fnp.ndarray,
    density: fnp.ndarray,
    probability: fnp.ndarray,
) -> fnp.ndarray:
    """Return the full pairwise ReLU second moment under a Gaussian closure."""

    correlation, scale = _correlation_matrix(
        covariance,
        standard_deviation,
        _MAX_ABS_CORRELATION,
    )
    alpha_i = alpha[:, None]
    alpha_j = alpha[None, :]
    root = fnp.sqrt(
        fnp.maximum(1.0 - correlation * correlation, _MIN_VARIANCE)
    )
    boundary_j_given_i = (alpha_j - correlation * alpha_i) / root
    boundary_i_given_j = (alpha_i - correlation * alpha_j) / root
    joint_probability = _bivariate_normal_cdf(
        alpha_i,
        alpha_j,
        correlation,
        probability,
    )

    mean_i = mean[:, None]
    mean_j = mean[None, :]
    std_i = standard_deviation[:, None]
    std_j = standard_deviation[None, :]
    density_i = density[:, None]
    density_j = density[None, :]

    second_moment = (
        (mean_i * mean_j + covariance) * joint_probability
        + mean_i
        * std_j
        * density_j
        * flops.stats.norm.cdf(boundary_i_given_j)
        + mean_j
        * std_i
        * density_i
        * flops.stats.norm.cdf(boundary_j_given_i)
        + scale
        * root
        * density_i
        * flops.stats.norm.pdf(boundary_j_given_i)
    )
    return fnp.maximum(0.5 * (second_moment + second_moment.T), 0.0)


def _polynomial_truncation_integral(
    power: int,
    hermite_coefficients: tuple[float, ...],
    mean: fnp.ndarray,
    standard_deviation: fnp.ndarray,
    truncated_standard_moments: list[fnp.ndarray],
) -> fnp.ndarray:
    """Integrate z**power * He_n(t) over z=mean+std*t > 0."""

    result = mean * 0.0
    for mean_power in range(power + 1):
        base_coefficient = comb(power, mean_power)
        base_scale = (
            base_coefficient
            * mean ** (power - mean_power)
            * standard_deviation**mean_power
        )
        for hermite_power, hermite_coefficient in enumerate(
            hermite_coefficients
        ):
            if hermite_coefficient:
                result = (
                    result
                    + hermite_coefficient
                    * base_scale
                    * truncated_standard_moments[
                        mean_power + hermite_power
                    ]
                )
    return result


def _edgeworth_relu_raw_moments(
    mean: fnp.ndarray,
    standard_deviation: fnp.ndarray,
    alpha: fnp.ndarray,
    density: fnp.ndarray,
    probability: fnp.ndarray,
    skewness: fnp.ndarray,
    excess_kurtosis: fnp.ndarray,
) -> tuple[fnp.ndarray, fnp.ndarray, fnp.ndarray, fnp.ndarray]:
    """Return the first four ReLU moments with a marginal Edgeworth closure."""

    lower_bound = -alpha
    truncated_standard_moments = [probability, density]
    lower_bound_power = lower_bound
    for order in range(2, 11):
        truncated_standard_moments.append(
            lower_bound_power * density
            + (order - 1) * truncated_standard_moments[order - 2]
        )
        lower_bound_power = lower_bound_power * lower_bound

    raw_moments = []
    for power in range(1, 5):
        gaussian_moment = _polynomial_truncation_integral(
            power,
            (1.0,),
            mean,
            standard_deviation,
            truncated_standard_moments,
        )
        skew_correction = _polynomial_truncation_integral(
            power,
            _HERMITE_3,
            mean,
            standard_deviation,
            truncated_standard_moments,
        )
        kurtosis_correction = _polynomial_truncation_integral(
            power,
            _HERMITE_4,
            mean,
            standard_deviation,
            truncated_standard_moments,
        )
        skew_squared_correction = _polynomial_truncation_integral(
            power,
            _HERMITE_6,
            mean,
            standard_deviation,
            truncated_standard_moments,
        )
        raw_moments.append(
            gaussian_moment
            + (skewness / 6.0) * skew_correction
            + (excess_kurtosis / 24.0) * kurtosis_correction
            + (skewness * skewness / 72.0) * skew_squared_correction
        )

    return tuple(raw_moments)


def _symmetric_cp_cumulant_diagonal(
    factors: fnp.ndarray,
    coefficients: fnp.ndarray,
    order: int,
) -> fnp.ndarray:
    """Return K[i,...,i] from a symmetric CP cumulant representation."""

    if factors.shape[1] == 0:
        return fnp.zeros(factors.shape[0])
    return factors**order @ coefficients


def _compress_symmetric_cp_cumulant(
    factors: fnp.ndarray,
    coefficients: fnp.ndarray,
    order: int,
    target_diagonal: fnp.ndarray,
) -> tuple[fnp.ndarray, fnp.ndarray]:
    """Keep energetic cross factors and restore every marginal exactly.

    A symmetric order-k cumulant is represented implicitly as
    ``sum_r c[r] * f[:, r]**(outer k)``.  Linear maps and the ReLU delta gain
    act directly on ``f``.  Rank is bounded by pruning on the invariant term
    norm ``abs(c) * ||f||**k``; width diagonal factors are then appended so
    pruning never changes the scalar skewness/kurtosis used at the next layer.
    """

    rank = factors.shape[1]
    if rank > _MAX_CROSS_CUMULANT_RANK:
        squared_norm = fnp.sum(factors * factors, axis=0)
        score = fnp.abs(coefficients) * squared_norm ** (0.5 * order)
        keep = fnp.argsort(score)[-_MAX_CROSS_CUMULANT_RANK:]
        factors = factors[:, keep]
        coefficients = coefficients[keep]

    retained_diagonal = _symmetric_cp_cumulant_diagonal(
        factors,
        coefficients,
        order,
    )
    diagonal_residual = target_diagonal - retained_diagonal
    factors = fnp.concatenate((factors, fnp.eye(factors.shape[0])), axis=1)
    coefficients = fnp.concatenate((coefficients, diagonal_residual), axis=0)
    return factors, coefficients


def _third_cumulant_diagonal(
    factors: tuple[fnp.ndarray, fnp.ndarray, fnp.ndarray],
) -> fnp.ndarray:
    """Return K3[i,i,i] from Sym(sum_r A_ir B_jr C_kr)."""

    if factors[0].shape[1] == 0:
        return fnp.zeros(factors[0].shape[0])
    return fnp.sum(factors[0] * factors[1] * factors[2], axis=1)


def _compress_third_cumulant(
    factors: tuple[fnp.ndarray, fnp.ndarray, fnp.ndarray],
    target_diagonal: fnp.ndarray,
) -> tuple[fnp.ndarray, fnp.ndarray, fnp.ndarray]:
    """Rank-cap general factored K3 while preserving K3[i,i,i] exactly."""

    first, second, third = factors
    rank = first.shape[1]
    if rank > _MAX_CROSS_CUMULANT_RANK:
        score = (
            fnp.sqrt(fnp.sum(first * first, axis=0))
            * fnp.sqrt(fnp.sum(second * second, axis=0))
            * fnp.sqrt(fnp.sum(third * third, axis=0))
        )
        keep = fnp.argsort(score)[-_MAX_CROSS_CUMULANT_RANK:]
        first = first[:, keep]
        second = second[:, keep]
        third = third[:, keep]

    retained_diagonal = fnp.sum(first * second * third, axis=1)
    diagonal_residual = target_diagonal - retained_diagonal
    identity = fnp.eye(first.shape[0])
    return (
        fnp.concatenate((first, identity), axis=1),
        fnp.concatenate((second, identity), axis=1),
        fnp.concatenate(
            (third, identity * diagonal_residual[:, None]),
            axis=1,
        ),
    )


class Estimator(BaseEstimator):
    """Propagate dense covariance and compressed cross k3/k4 cumulants."""

    def predict(self, mlp, budget: int) -> fnp.ndarray:
        del budget

        mean = fnp.zeros(mlp.width)
        covariance = fnp.eye(mlp.width)
        third_factors = (
            fnp.zeros((mlp.width, 0)),
            fnp.zeros((mlp.width, 0)),
            fnp.zeros((mlp.width, 0)),
        )
        fourth_factors = fnp.zeros((mlp.width, 0))
        fourth_coefficients = fnp.zeros(0)
        layer_means = []

        for layer_index, weights in enumerate(mlp.weights):
            pre_mean = weights.T @ mean
            # This symmetry-aware contraction is equivalent to W.T @ cov @ W.
            pre_covariance = fnp.einsum(
                "ij,ia,jb->ab", covariance, weights, weights
            )
            pre_variance = fnp.maximum(
                fnp.diag(pre_covariance),
                _MIN_VARIANCE,
            )
            pre_std = fnp.sqrt(pre_variance)

            # A linear layer maps each cumulant factor independently.  Thus
            # all represented cross entries survive without dense n^3/n^4
            # materialization.
            pre_third_factors = tuple(
                weights.T @ factor for factor in third_factors
            )
            pre_fourth_factors = weights.T @ fourth_factors
            pre_third_cumulant = _third_cumulant_diagonal(
                pre_third_factors,
            )
            pre_fourth_cumulant = _symmetric_cp_cumulant_diagonal(
                pre_fourth_factors,
                fourth_coefficients,
                4,
            )
            skewness = pre_third_cumulant / (pre_std * pre_variance)
            excess_kurtosis = pre_fourth_cumulant / (
                pre_variance * pre_variance
            )
            skewness = fnp.maximum(
                fnp.minimum(skewness, _MAX_ABS_SKEWNESS),
                -_MAX_ABS_SKEWNESS,
            )
            excess_kurtosis = fnp.maximum(
                fnp.minimum(excess_kurtosis, _MAX_EXCESS_KURTOSIS),
                _MIN_EXCESS_KURTOSIS,
            )

            alpha = pre_mean / pre_std
            density = flops.stats.norm.pdf(alpha)
            probability = flops.stats.norm.cdf(alpha)

            (
                mean,
                marginal_second_moment,
                marginal_third_moment,
                marginal_fourth_moment,
            ) = _edgeworth_relu_raw_moments(
                pre_mean,
                pre_std,
                alpha,
                density,
                probability,
                skewness,
                excess_kurtosis,
            )
            mean = fnp.maximum(mean, 0.0)
            post_variance = fnp.maximum(
                marginal_second_moment - mean * mean,
                _MIN_VARIANCE,
            )
            post_third_cumulant = (
                marginal_third_moment
                - 3.0 * mean * marginal_second_moment
                + 2.0 * mean * mean * mean
            )
            fourth_central_moment = (
                marginal_fourth_moment
                - 4.0 * mean * marginal_third_moment
                + 6.0 * mean * mean * marginal_second_moment
                - 3.0 * mean * mean * mean * mean
            )
            post_fourth_cumulant = (
                fourth_central_moment - 3.0 * post_variance * post_variance
            )
            post_std = fnp.sqrt(post_variance)
            post_third_cumulant = fnp.maximum(
                fnp.minimum(
                    post_third_cumulant,
                    _MAX_ABS_SKEWNESS * post_std * post_variance,
                ),
                -_MAX_ABS_SKEWNESS * post_std * post_variance,
            )
            post_fourth_cumulant = fnp.maximum(
                fnp.minimum(
                    post_fourth_cumulant,
                    _MAX_EXCESS_KURTOSIS
                    * post_variance
                    * post_variance,
                ),
                _MIN_EXCESS_KURTOSIS
                * post_variance
                * post_variance,
            )

            # The expected ReLU derivative is Phi(alpha).  Its product on all
            # tensor legs is the first-order multivariate delta propagation.
            # Append diagonal residuals so the Edgeworth marginal k3/k4 above
            # remains exact even after rank compression.
            propagated_third_factors = tuple(
                probability[:, None] * factor
                for factor in pre_third_factors
            )

            # Leading connected Gaussian Wick diagrams of Hermite degrees
            # (2,1,1).  For centered ReLU outputs the generated term is
            #   sum_centers phi_i/std_i * Phi_j*Phi_k*C_ij*C_ik.
            # Sym(A,B,B) averages the three possible centers, hence the factor
            # of three on A.  This creates new all-distinct cross K3 instead of
            # merely multiplying the old cumulant by delta gains.
            branch = (
                pre_covariance * probability[None, :]
            ).T
            center = (
                3.0
                * fnp.eye(mlp.width)
                * (density / pre_std)[:, None]
            )
            third_factors = _compress_third_cumulant(
                tuple(
                    fnp.concatenate((old, new), axis=1)
                    for old, new in zip(
                        propagated_third_factors,
                        (center, branch, branch),
                    )
                ),
                post_third_cumulant,
            )
            fourth_factors, fourth_coefficients = (
                _compress_symmetric_cp_cumulant(
                    probability[:, None] * pre_fourth_factors,
                    fourth_coefficients,
                    4,
                    post_fourth_cumulant,
                )
            )

            if layer_index == 0:
                # The initial normal input has zero mean, so the standard
                # analytic ReLU arc-cosine kernel applies exactly here.
                second_moment = _zero_mean_relu_second_moment(
                    pre_covariance,
                    pre_std,
                )
            else:
                second_moment = _noncentral_relu_second_moment(
                    pre_mean,
                    pre_covariance,
                    pre_std,
                    alpha,
                    density,
                    probability,
                )

            # Keep the bivariate Gaussian covariance off-diagonal, then
            # overwrite its diagonal with the Edgeworth-corrected variance.
            covariance = second_moment - fnp.outer(
                mean,
                mean,
            )
            covariance = 0.5 * (covariance + covariance.T)
            fnp.fill_diagonal(covariance, post_variance)
            layer_means.append(mean)

        return fnp.stack(layer_means, axis=0)
