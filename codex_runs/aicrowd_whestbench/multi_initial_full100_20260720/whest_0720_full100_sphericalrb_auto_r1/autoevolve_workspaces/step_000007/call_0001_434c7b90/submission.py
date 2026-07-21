from __future__ import annotations

import math

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import BaseEstimator


_TARGET_FLOP_FRACTION = 0.1024
_MAX_WORKING_SET_BYTES = 100 * 1024 * 1024
_FLOAT64_ITEMSIZE = 8
_TAIL_CLIP = 1e-9
_MIN_ANTITHETIC_SAMPLES = 32
_WHITENING_EIGENVALUE_FLOOR = 1e-8
_QMC_SHIFTS = 2
_GATE_MEAN_CORRECTION = 1.05
_ANALYTIC_BLEND = 0.05
_SETUP_DIRECTION_SETS = 1
_SETUP_SEED_OFFSET = 0


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
    count = int(max(_MIN_ANTITHETIC_SAMPLES, min(budget_limited, memory_limited)))
    return count - (count & 1)


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


def _lattice_normal_samples(n_samples: int, width: int, rng) -> fnp.ndarray:
    roots = fnp.sqrt(fnp.array(_first_primes(width), dtype=fnp.float64))
    generator = roots - fnp.floor(roots)
    shift = rng.random(width)
    indices = fnp.arange(n_samples, dtype=fnp.float64)[:, None]
    unwrapped = indices * generator[None, :] + shift[None, :]
    uniforms = unwrapped - fnp.floor(unwrapped)
    uniforms = fnp.clip(uniforms, _TAIL_CLIP, 1.0 - _TAIL_CLIP)
    return flops.stats.norm.ppf(uniforms)


def _mean_chi(dimension: int) -> float:
    half_dimension = 0.5 * float(dimension)
    return math.sqrt(2.0) * math.exp(
        math.lgamma(half_dimension + 0.5) - math.lgamma(half_dimension)
    )


def _qmc_directions(n_samples: int, width: int, rng) -> fnp.ndarray:
    normals = _lattice_normal_samples(n_samples, width, rng)
    radii = fnp.sqrt(fnp.maximum(fnp.sum(normals * normals, axis=1), 1e-24))
    return normals / radii[:, None]


def _lattice_half_directions(half_samples: int, width: int, rng) -> fnp.ndarray:
    base_count = half_samples // _QMC_SHIFTS
    remainder = half_samples - base_count * _QMC_SHIFTS
    direction_blocks = []
    for shift_index in range(_QMC_SHIFTS):
        block_count = base_count + (1 if shift_index < remainder else 0)
        direction_blocks.append(_qmc_directions(block_count, width, rng))
    return fnp.concatenate(direction_blocks, axis=0)


def _angular_whitening_transform(directions: fnp.ndarray) -> fnp.ndarray:
    n_samples, width = directions.shape
    scaled_second_moment = directions.T @ directions * (float(width) / float(n_samples))
    eigenvalues, eigenvectors = fnp.linalg.eigh(scaled_second_moment)
    eigenvalues = fnp.maximum(eigenvalues, _WHITENING_EIGENVALUE_FLOOR)
    return (eigenvectors / fnp.sqrt(eigenvalues)) @ eigenvectors.T


def _exact_first_layer_mean(weights: fnp.ndarray) -> fnp.ndarray:
    standard_deviation = fnp.sqrt(fnp.sum(weights * weights, axis=0))
    return standard_deviation / fnp.sqrt(2.0 * fnp.pi)


def _zero_mean_relu_covariance(pre_covariance: fnp.ndarray) -> fnp.ndarray:
    variance = fnp.maximum(fnp.diag(pre_covariance), 1e-24)
    standard_deviation = fnp.sqrt(variance)
    scale = standard_deviation[:, None] * standard_deviation[None, :]
    correlation = fnp.clip(pre_covariance / scale, -1.0, 1.0)
    second_moment = scale * (
        fnp.sqrt(fnp.maximum(1.0 - correlation * correlation, 0.0))
        + (fnp.pi - fnp.arccos(correlation)) * correlation
    ) / (2.0 * fnp.pi)
    mean = standard_deviation / fnp.sqrt(2.0 * fnp.pi)
    return second_moment - mean[:, None] * mean[None, :]


def _linearized_gaussian_rows(weights: list[fnp.ndarray], width: int) -> fnp.ndarray:
    mean = fnp.zeros(width, dtype=fnp.float32)
    covariance = fnp.eye(width, dtype=fnp.float32)
    rows = []
    normalizer = 1.0 / fnp.sqrt(2.0 * fnp.pi)

    for layer_index, weight in enumerate(weights):
        pre_mean = mean @ weight
        pre_covariance = weight.T @ covariance @ weight
        variance = fnp.maximum(fnp.diag(pre_covariance), 1e-24)
        standard_deviation = fnp.sqrt(variance)
        standardized = pre_mean / standard_deviation
        gate = flops.stats.norm.cdf(standardized)
        density = fnp.exp(-0.5 * standardized * standardized) * normalizer

        next_mean = standard_deviation * density + pre_mean * gate
        second_moment = (
            (pre_mean * pre_mean + variance) * gate
            + pre_mean * standard_deviation * density
        )
        next_variance = fnp.maximum(second_moment - next_mean * next_mean, 1e-24)

        if layer_index == 0:
            covariance = _zero_mean_relu_covariance(pre_covariance)
        else:
            linearized_covariance = pre_covariance * gate[:, None] * gate[None, :]
            covariance = linearized_covariance + fnp.diag(
                next_variance - fnp.diag(linearized_covariance)
            )
        mean = next_mean
        rows.append(mean)

    return fnp.stack(rows, axis=0)


class Estimator(BaseEstimator):
    def setup(self, context) -> None:
        self._setup_width = int(context.width)
        self._setup_half_samples = (
            _sample_count(int(context.flop_budget), int(context.width), int(context.depth))
            // 2
        )
        rng = fnp.random.default_rng(
            (int(context.seed) + _SETUP_SEED_OFFSET) & 0xFFFFFFFF
        )
        self._direction_pool = []
        for _ in range(_SETUP_DIRECTION_SETS):
            half_directions = _lattice_half_directions(
                self._setup_half_samples,
                self._setup_width,
                rng,
            )
            whitening = _angular_whitening_transform(half_directions)
            self._direction_pool.append((half_directions, whitening))

    def predict(self, mlp, budget: int) -> fnp.ndarray:
        n_samples = _sample_count(budget, mlp.width, mlp.depth)
        half_samples = n_samples // 2

        if (
            hasattr(self, "_direction_pool")
            and mlp.width == self._setup_width
            and half_samples == self._setup_half_samples
            and self._direction_pool
        ):
            half_directions, whitening = self._direction_pool[
                int(mlp.seed) % len(self._direction_pool)
            ]
            first_weight = whitening @ mlp.weights[0]
        else:
            rng = fnp.random.default_rng(int(mlp.seed))
            half_directions = _lattice_half_directions(half_samples, mlp.width, rng)
            whitening = _angular_whitening_transform(half_directions)
            first_weight = whitening @ mlp.weights[0]

        paired_directions = fnp.concatenate((half_directions, -half_directions), axis=0)
        activations = paired_directions * _mean_chi(mlp.width)

        rows = []
        activations = fnp.maximum(activations @ first_weight, 0.0)
        exact_first_mean = _exact_first_layer_mean(mlp.weights[0])
        rows.append(exact_first_mean)

        mean_correction = exact_first_mean - fnp.mean(activations, axis=0)
        for weights in mlp.weights[1:]:
            preactivations = activations @ weights
            gate_mean = fnp.mean(preactivations > 0.0, axis=0)
            activations = fnp.maximum(preactivations, 0.0)
            mean_correction = (mean_correction @ weights) * gate_mean
            base_row = fnp.mean(activations, axis=0)
            rows.append(base_row + _GATE_MEAN_CORRECTION * mean_correction)

        qmc_rows = fnp.stack(rows, axis=0)
        analytic_rows = _linearized_gaussian_rows(mlp.weights, mlp.width)
        blended_rows = (
            (1.0 - _ANALYTIC_BLEND) * qmc_rows + _ANALYTIC_BLEND * analytic_rows
        )
        return fnp.asarray(blended_rows, dtype=fnp.float32)
