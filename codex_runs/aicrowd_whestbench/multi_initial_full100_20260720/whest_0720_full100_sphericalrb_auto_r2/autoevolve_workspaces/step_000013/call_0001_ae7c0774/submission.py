from __future__ import annotations

import math

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import BaseEstimator


_TARGET_FLOP_FRACTION = 0.13
_MAX_WORKING_SET_BYTES = 100 * 1024 * 1024
_FLOAT64_ITEMSIZE = 8
_TAIL_CLIP = 1e-11
_MIN_SAMPLES = 32
_WHITENING_EIGENVALUE_FLOOR = 1e-8
_KOROBOV_MULTIPLIER = 11
_KOROBOV_MODULUS = 104729
_SEED_OFFSET = 0


def _mean_chi(dimension: int) -> float:
    half_dimension = 0.5 * float(dimension)
    return math.sqrt(2.0) * math.exp(
        math.lgamma(half_dimension + 0.5) - math.lgamma(half_dimension)
    )


def _exact_first_layer_mean(weights: fnp.ndarray) -> fnp.ndarray:
    standard_deviation = fnp.sqrt(fnp.sum(weights * weights, axis=0))
    return standard_deviation / fnp.sqrt(2.0 * fnp.pi)


def _sample_count(budget: int, width: int, depth: int) -> int:
    half_lattice_per_sample = (89 * width + 3 * width + 2) // 2
    layer_per_sample = depth * (2 * width * width + 2 * width)
    covariance_per_sample = width * width
    per_sample = half_lattice_per_sample + layer_per_sample + covariance_per_sample
    fixed = 13 * width**3 + 8 * width
    budget_limited = (
        int(_TARGET_FLOP_FRACTION * budget) - fixed
    ) // max(per_sample, 1)
    memory_limited = _MAX_WORKING_SET_BYTES // max(width * _FLOAT64_ITEMSIZE, 1)
    count = int(max(_MIN_SAMPLES, min(budget_limited, memory_limited)))
    return count - (count & 1)


def _lattice_generator(width: int) -> fnp.ndarray:
    values = [
        pow(_KOROBOV_MULTIPLIER, index, _KOROBOV_MODULUS) / _KOROBOV_MODULUS
        for index in range(width)
    ]
    return fnp.array(values, dtype=fnp.float64)


def _lattice_normal_samples(n_samples: int, width: int, rng) -> fnp.ndarray:
    generator = _lattice_generator(width)
    shift = rng.random(width)
    indices = fnp.arange(n_samples, dtype=fnp.float64)[:, None]
    uniforms = indices * generator[None, :] + shift[None, :]
    uniforms = uniforms - fnp.floor(uniforms)
    uniforms = fnp.clip(uniforms, _TAIL_CLIP, 1.0 - _TAIL_CLIP)
    return flops.stats.norm.ppf(uniforms)


def _qmc_directions(n_samples: int, width: int, rng) -> fnp.ndarray:
    normals = _lattice_normal_samples(n_samples, width, rng)
    radii = fnp.sqrt(fnp.maximum(fnp.sum(normals * normals, axis=1), 1e-24))
    return normals / radii[:, None]


def _angular_whitening_transform(directions: fnp.ndarray) -> fnp.ndarray:
    n_samples, width = directions.shape
    scaled_second_moment = (
        directions.T @ directions
    ) * (float(width) / float(n_samples))
    eigenvalues, eigenvectors = fnp.linalg.eigh(scaled_second_moment)
    eigenvalues = fnp.maximum(eigenvalues, _WHITENING_EIGENVALUE_FLOOR)
    return (eigenvectors / fnp.sqrt(eigenvalues)) @ eigenvectors.T


class Estimator(BaseEstimator):
    def predict(self, mlp, budget: int) -> fnp.ndarray:
        n_samples = _sample_count(budget, mlp.width, mlp.depth)
        half_samples = n_samples // 2
        rng = fnp.random.default_rng(int(mlp.seed) + _SEED_OFFSET)

        half_directions = _qmc_directions(half_samples, mlp.width, rng)
        whitening = _angular_whitening_transform(half_directions)
        first_weight = whitening @ mlp.weights[0]
        directions = fnp.concatenate((half_directions, -half_directions), axis=0)
        activations = directions * _mean_chi(mlp.width)

        rows = []
        activations = fnp.maximum(activations @ first_weight, 0.0)
        first_mean = _exact_first_layer_mean(mlp.weights[0])
        sample_first_mean = fnp.maximum(fnp.mean(activations, axis=0), 1e-12)
        activations = activations * (first_mean / sample_first_mean)[None, :]
        rows.append(first_mean)

        for weights in mlp.weights[1:]:
            activations = fnp.maximum(activations @ weights, 0.0)
            rows.append(fnp.mean(activations, axis=0))

        return fnp.asarray(fnp.stack(rows, axis=0), dtype=fnp.float32)
