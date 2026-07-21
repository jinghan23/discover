from __future__ import annotations

import math
from functools import lru_cache

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import BaseEstimator


_TARGET_FLOP_FRACTION = 0.10
_MAX_WORKING_SET_BYTES = 100 * 1024 * 1024
_FLOAT64_ITEMSIZE = 8
_TAIL_CLIP = 1e-11
_MIN_ANTITHETIC_SAMPLES = 32
_WHITENING_EIGENVALUE_FLOOR = 1e-8
_MOMENT_FLOOR = 1e-24


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

    fixed = 18 * width**3 + 16 * width * width + 8 * width

    budget_limited = (int(float(target_fraction) * budget) - fixed) // max(
        per_sample,
        1,
    )
    memory_limited = _MAX_WORKING_SET_BYTES // max(width * _FLOAT64_ITEMSIZE, 1)

    count = int(max(_MIN_ANTITHETIC_SAMPLES, min(budget_limited, memory_limited)))
    return count - (count & 1)


@lru_cache(maxsize=None)
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


@lru_cache(maxsize=None)
def _mean_chi(dimension: int) -> float:
    half_dimension = 0.5 * float(dimension)
    return math.sqrt(2.0) * math.exp(
        math.lgamma(half_dimension + 0.5) - math.lgamma(half_dimension)
    )


def _lattice_normal_samples(n_samples: int, width: int, rng) -> fnp.ndarray:
    roots = fnp.sqrt(fnp.array(_first_primes(width), dtype=fnp.float64))
    generator = roots - fnp.floor(roots)
    shift = rng.random(width)

    indices = fnp.arange(n_samples, dtype=fnp.float64)[:, None]
    unwrapped = indices * generator[None, :] + shift[None, :]
    uniforms = unwrapped - fnp.floor(unwrapped)
    uniforms = fnp.clip(uniforms, _TAIL_CLIP, 1.0 - _TAIL_CLIP)
    return flops.stats.norm.ppf(uniforms)


def _qmc_directions(n_samples: int, width: int, rng) -> fnp.ndarray:
    normals = _lattice_normal_samples(n_samples, width, rng)
    radii = fnp.sqrt(fnp.maximum(fnp.sum(normals * normals, axis=1), _MOMENT_FLOOR))
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
    return standard_deviation / fnp.sqrt(2.0 * fnp.pi)


def _exact_first_layer_particle_second(weights: fnp.ndarray, width: int) -> fnp.ndarray:
    gram = weights.T @ weights
    variance = fnp.maximum(fnp.sum(weights * weights, axis=0), 0.0)
    standard_deviation = fnp.sqrt(variance)

    denom = fnp.maximum(
        standard_deviation[:, None] * standard_deviation[None, :],
        _MOMENT_FLOOR,
    )
    correlation = fnp.clip(gram / denom, -1.0, 1.0)

    relu_kernel = fnp.sqrt(fnp.maximum(1.0 - correlation * correlation, 0.0))
    relu_kernel = relu_kernel + (math.pi - fnp.arccos(correlation)) * correlation

    gaussian_second = denom * relu_kernel / (2.0 * math.pi)
    angular_radial_scale = (_mean_chi(width) * _mean_chi(width)) / float(width)
    return gaussian_second * angular_radial_scale


def _match_column_moments(
    values: fnp.ndarray,
    target_mean: fnp.ndarray,
    target_variance: fnp.ndarray,
) -> fnp.ndarray:
    sample_mean = fnp.mean(values, axis=0)
    centered = values - sample_mean
    sample_variance = fnp.mean(centered * centered, axis=0)

    scale = fnp.sqrt(
        fnp.maximum(target_variance, _MOMENT_FLOOR)
        / fnp.maximum(sample_variance, _MOMENT_FLOOR)
    )
    return centered * scale + target_mean


class Estimator(BaseEstimator):
    def predict(self, mlp, budget: int) -> fnp.ndarray:
        n_samples = whitened_antithetic_sample_count(
            budget,
            mlp.width,
            mlp.depth,
            _TARGET_FLOP_FRACTION,
        )
        half_samples = n_samples // 2

        rng = fnp.random.default_rng(mlp.seed)
        half_directions = _qmc_directions(half_samples, mlp.width, rng)
        whitening = _angular_whitening_transform(half_directions)

        paired_directions = fnp.concatenate((half_directions, -half_directions), axis=0)
        activations = paired_directions * _mean_chi(mlp.width)

        rows = []
        first_weights = mlp.weights[0]
        activations = fnp.maximum(activations @ (whitening @ first_weights), 0.0)

        exact_first_mean = _exact_first_layer_mean(first_weights)
        rows.append(exact_first_mean)

        exact_first_second = None
        if mlp.depth > 1:
            exact_first_second = _exact_first_layer_particle_second(
                first_weights,
                mlp.width,
            )

        for layer_index, weights in enumerate(mlp.weights[1:]):
            preactivations = activations @ weights

            if layer_index == 0 and exact_first_second is not None:
                target_mean = exact_first_mean @ weights
                target_second = fnp.sum((exact_first_second @ weights) * weights, axis=0)
                target_variance = fnp.maximum(
                    target_second - target_mean * target_mean,
                    _MOMENT_FLOOR,
                )
                preactivations = _match_column_moments(
                    preactivations,
                    target_mean,
                    target_variance,
                )

            activations = fnp.maximum(preactivations, 0.0)
            rows.append(fnp.mean(activations, axis=0))

        return fnp.asarray(fnp.stack(rows, axis=0), dtype=fnp.float32)
