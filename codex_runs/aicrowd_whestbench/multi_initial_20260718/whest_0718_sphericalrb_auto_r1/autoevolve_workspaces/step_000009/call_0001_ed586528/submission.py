from __future__ import annotations

import functools
import math

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import BaseEstimator


_MAX_WORKING_SET_BYTES = 100 * 1024 * 1024
_FLOAT64_ITEMSIZE = 8
_MIN_ANTITHETIC_SAMPLES = 32
_TAIL_CLIP = 1.0e-9
_WHITENING_EIGENVALUE_FLOOR = 1e-10
_INV_SQRT_TWO_PI = 1.0 / math.sqrt(2.0 * math.pi)

_ANTITHETIC_HALF_COEFF = 0.018
_PAIR_HALF_COEFF = 0.179
_PAIR_QUARTER_COEFF = -0.011
_PAIR_EIGHTH_COEFF = 0.035
_PAIR_SIXTEENTH_COEFF = -0.035
_PAIR_THIRTYSECOND_COEFF = 0.0065
_PAIR_SIXTYFOURTH_COEFF = 0.016
_PAIR_SIXTYFOURTH_SUFFIX_COEFF = -0.0085
_PAIR_ONE28_COEFF = -0.01425
_PAIR_ONE28_SUFFIX_COEFF = 0.0116
_FINAL_GAUSSIAN_BLEND = -0.055
_SCALAR_FIRST_CONTROL_BLEND = -1.95
_SCALAR_FIRST_SECOND_CONTROL_BLEND = -0.185


def whitened_antithetic_sample_count(
    budget: int,
    width: int,
    depth: int,
    target_fraction: float,
) -> int:
    half_lattice_per_sample = (89 * width + 3 * width + 2) // 2
    layer_per_sample = depth * (2 * width * width + 2 * width)
    covariance_per_sample = width * width
    per_sample = half_lattice_per_sample + layer_per_sample + covariance_per_sample
    fixed = 13 * width**3 + 8 * width
    budget_limited = (
        int(float(target_fraction) * budget) - fixed
    ) // max(per_sample, 1)
    memory_limited = _MAX_WORKING_SET_BYTES // max(width * _FLOAT64_ITEMSIZE, 1)
    count = int(max(_MIN_ANTITHETIC_SAMPLES, min(budget_limited, memory_limited)))
    return count - (count & 1)


@functools.lru_cache(maxsize=None)
def _first_primes(count: int) -> tuple[int, ...]:
    primes: list[int] = []
    candidate = 2
    while len(primes) < count:
        is_prime = True
        for prime in primes:
            if prime * prime > candidate:
                break
            if candidate % prime == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(candidate)
        candidate += 1
    return tuple(primes)


@functools.lru_cache(maxsize=None)
def _mean_chi(dimension: int) -> float:
    half_dimension = 0.5 * float(dimension)
    return math.sqrt(2.0) * math.exp(
        math.lgamma(half_dimension + 0.5) - math.lgamma(half_dimension)
    )


def _lattice_normal_samples(
    n_samples: int,
    width: int,
    rng,
    generator: fnp.ndarray | None = None,
    lattice_base: fnp.ndarray | None = None,
) -> fnp.ndarray:
    if generator is None:
        roots = fnp.sqrt(fnp.array(_first_primes(width), dtype=fnp.float64))
        generator = roots - fnp.floor(roots)
    shift = rng.random(width)
    if lattice_base is None or lattice_base.shape != (n_samples, width):
        indices = fnp.arange(n_samples, dtype=fnp.float64)[:, None]
        unwrapped = indices * generator[None, :] + shift[None, :]
    else:
        unwrapped = lattice_base + shift[None, :]
    uniforms = unwrapped - fnp.floor(unwrapped)
    uniforms = fnp.clip(uniforms, _TAIL_CLIP, 1.0 - _TAIL_CLIP)
    return flops.stats.norm.ppf(uniforms)


def _qmc_directions(
    n_samples: int,
    width: int,
    rng,
    generator: fnp.ndarray | None = None,
    lattice_base: fnp.ndarray | None = None,
) -> fnp.ndarray:
    normals = _lattice_normal_samples(n_samples, width, rng, generator, lattice_base)
    radii = fnp.sqrt(fnp.maximum(fnp.sum(normals * normals, axis=1), 1e-24))
    return normals / radii[:, None]


def _angular_whitening_transform(directions: fnp.ndarray) -> fnp.ndarray:
    n_samples, width = directions.shape
    scaled_second_moment = (directions.T @ directions) * (
        float(width) / float(n_samples)
    )
    eigenvalues, eigenvectors = fnp.linalg.eigh(scaled_second_moment)
    eigenvalues = fnp.maximum(eigenvalues, _WHITENING_EIGENVALUE_FLOOR)
    return (eigenvectors / fnp.sqrt(eigenvalues)) @ eigenvectors.T


def _exact_first_layer_mean(weights: fnp.ndarray) -> fnp.ndarray:
    standard_deviation = fnp.sqrt(fnp.sum(weights * weights, axis=0))
    return standard_deviation * _INV_SQRT_TWO_PI


def _relu_gaussian_mean(pre_mean: fnp.ndarray, variance: fnp.ndarray) -> fnp.ndarray:
    sigma = fnp.sqrt(fnp.maximum(variance, 1e-12))
    alpha = pre_mean / sigma
    return pre_mean * flops.stats.norm.cdf(alpha) + sigma * flops.stats.norm.pdf(
        alpha
    )


def _paired_prefix_mean(
    final_values: fnp.ndarray,
    half_count: int,
    divisor: int,
) -> fnp.ndarray:
    count = max(1, half_count // divisor)
    return 0.5 * (
        fnp.mean(final_values[:count], axis=0)
        + fnp.mean(final_values[half_count : half_count + count], axis=0)
    )


def _paired_suffix_mean(
    final_values: fnp.ndarray,
    half_count: int,
    divisor: int,
) -> fnp.ndarray:
    count = max(1, half_count // divisor)
    return 0.5 * (
        fnp.mean(final_values[half_count - count : half_count], axis=0)
        + fnp.mean(final_values[-count:], axis=0)
    )


def _scalar_first_layer_control_correction(
    first_activations: fnp.ndarray,
    final_values: fnp.ndarray,
    exact_first_mean: fnp.ndarray,
) -> fnp.ndarray:
    first_total = fnp.sum(first_activations, axis=1)
    control_mean = fnp.mean(first_total)
    final_mean = fnp.mean(final_values, axis=0)
    control_centered = first_total - control_mean
    variance = fnp.maximum(fnp.mean(control_centered * control_centered), 1e-12)
    covariance = fnp.mean(
        control_centered[:, None] * (final_values - final_mean[None, :]),
        axis=0,
    )
    exact_total = fnp.sum(exact_first_mean)
    return (exact_total - control_mean) * covariance / variance


def _exact_first_layer_second_moment(weights: fnp.ndarray) -> fnp.ndarray:
    variance = fnp.sum(weights * weights, axis=0)
    return 0.5 * variance


def _scalar_first_layer_second_control_correction(
    first_activations: fnp.ndarray,
    final_values: fnp.ndarray,
    exact_first_second: fnp.ndarray,
) -> fnp.ndarray:
    first_energy = fnp.sum(first_activations * first_activations, axis=1)
    control_mean = fnp.mean(first_energy)
    final_mean = fnp.mean(final_values, axis=0)
    control_centered = first_energy - control_mean
    variance = fnp.maximum(fnp.mean(control_centered * control_centered), 1e-12)
    covariance = fnp.mean(
        control_centered[:, None] * (final_values - final_mean[None, :]),
        axis=0,
    )
    exact_total = fnp.sum(exact_first_second)
    return (exact_total - control_mean) * covariance / variance


class Estimator(BaseEstimator):
    target_flop_fraction = 0.10

    def setup(self, context) -> None:
        _first_primes(context.width)
        _mean_chi(context.width)
        roots = fnp.sqrt(fnp.array(_first_primes(context.width), dtype=fnp.float64))
        generator = roots - fnp.floor(roots)
        n_samples = whitened_antithetic_sample_count(
            context.flop_budget,
            context.width,
            context.depth,
            self.target_flop_fraction,
        )
        half_count = n_samples // 2
        indices = fnp.arange(half_count, dtype=fnp.float64)[:, None]
        base = indices * generator[None, :]
        self._lattice_generator = generator
        self._lattice_base = base - fnp.floor(base)

    def predict(self, mlp, budget: int) -> fnp.ndarray:
        if mlp.depth == 1:
            first_row = _exact_first_layer_mean(mlp.weights[0])
            return first_row[None, :]

        n_samples = whitened_antithetic_sample_count(
            budget,
            mlp.width,
            mlp.depth,
            self.target_flop_fraction,
        )
        half_count = n_samples // 2
        rng = fnp.random.default_rng(mlp.seed)
        generator = getattr(self, "_lattice_generator", None)
        lattice_base = getattr(self, "_lattice_base", None)
        half_directions = _qmc_directions(
            half_count,
            mlp.width,
            rng,
            generator,
            lattice_base,
        )
        whitening = _angular_whitening_transform(half_directions)

        first_weight = whitening @ mlp.weights[0]
        paired_directions = fnp.concatenate((half_directions, -half_directions), axis=0)
        activations = paired_directions * _mean_chi(mlp.width)

        activations = fnp.maximum(activations @ first_weight, 0.0)
        first_activations = activations
        for weights in mlp.weights[1:-1]:
            activations = fnp.maximum(activations @ weights, 0.0)

        final_preactivations = activations @ mlp.weights[-1]
        final_values = fnp.maximum(final_preactivations, 0.0)
        final_sample = fnp.mean(final_values, axis=0)

        if _ANTITHETIC_HALF_COEFF:
            antithetic_half = fnp.mean(final_values[:half_count], axis=0)
            final_sample = final_sample + _ANTITHETIC_HALF_COEFF * (
                final_sample - antithetic_half
            )

        pair_half = _paired_prefix_mean(final_values, half_count, 2)
        pair_quarter = _paired_prefix_mean(final_values, half_count, 4)
        pair_eighth = _paired_prefix_mean(final_values, half_count, 8)
        pair_sixteenth = _paired_prefix_mean(final_values, half_count, 16)
        pair_thirtysecond = _paired_prefix_mean(final_values, half_count, 32)
        pair_sixtyfourth = _paired_prefix_mean(final_values, half_count, 64)
        pair_one28 = _paired_prefix_mean(final_values, half_count, 128)
        suffix_sixtyfourth = _paired_suffix_mean(final_values, half_count, 64)
        suffix_one28 = _paired_suffix_mean(final_values, half_count, 128)

        final_sample = final_sample + _PAIR_HALF_COEFF * (final_sample - pair_half)
        final_sample = final_sample + _PAIR_QUARTER_COEFF * (
            pair_half - pair_quarter
        )
        final_sample = final_sample + _PAIR_EIGHTH_COEFF * (
            pair_quarter - pair_eighth
        )
        final_sample = final_sample + _PAIR_SIXTEENTH_COEFF * (
            pair_eighth - pair_sixteenth
        )
        final_sample = final_sample + _PAIR_THIRTYSECOND_COEFF * (
            pair_sixteenth - pair_thirtysecond
        )
        final_sample = final_sample + _PAIR_SIXTYFOURTH_COEFF * (
            pair_thirtysecond - pair_sixtyfourth
        )
        final_sample = final_sample + _PAIR_SIXTYFOURTH_SUFFIX_COEFF * (
            suffix_sixtyfourth - pair_sixtyfourth
        )
        final_sample = final_sample + _PAIR_ONE28_COEFF * (
            pair_sixtyfourth - pair_one28
        )
        final_sample = final_sample + _PAIR_ONE28_SUFFIX_COEFF * (
            suffix_one28 - pair_one28
        )

        if _FINAL_GAUSSIAN_BLEND:
            final_mean = fnp.mean(final_preactivations, axis=0)
            final_second = fnp.mean(final_preactivations * final_preactivations, axis=0)
            final_variance = final_second - final_mean * final_mean
            final_gaussian = _relu_gaussian_mean(final_mean, final_variance)
            final_estimate = final_sample + _FINAL_GAUSSIAN_BLEND * (
                final_gaussian - final_sample
            )
        else:
            final_estimate = final_sample

        if _SCALAR_FIRST_CONTROL_BLEND:
            exact_first_mean = _exact_first_layer_mean(mlp.weights[0])
            final_estimate = final_estimate + _SCALAR_FIRST_CONTROL_BLEND * (
                _scalar_first_layer_control_correction(
                    first_activations,
                    final_values,
                    exact_first_mean,
                )
            )

        if _SCALAR_FIRST_SECOND_CONTROL_BLEND:
            final_estimate = final_estimate + _SCALAR_FIRST_SECOND_CONTROL_BLEND * (
                _scalar_first_layer_second_control_correction(
                    first_activations,
                    final_values,
                    _exact_first_layer_second_moment(mlp.weights[0]),
                )
            )

        zero_row = fnp.zeros_like(final_estimate)
        rows = [zero_row] * (mlp.depth - 1)
        rows.append(final_estimate)
        return fnp.stack(rows, axis=0)
