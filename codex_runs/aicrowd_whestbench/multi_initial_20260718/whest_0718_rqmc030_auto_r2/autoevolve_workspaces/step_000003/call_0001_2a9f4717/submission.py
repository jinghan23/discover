"""Antithetic RQMC estimator with a first-layer control variate."""

from __future__ import annotations

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import BaseEstimator


_TARGET_FLOP_FRACTION = 0.40
_SHIFT_COUNT = 4
_FIRST_CONTROL_SCALE = 0.435
_INPUT_SECOND_POWER = 0.835
_TAIL_CLIP = 6.5e-7
_ARRAY_BYTES_LIMIT = 200 * 1024 * 1024
_FLOAT32_ITEMSIZE = 4


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


def _lattice_generator(width: int) -> fnp.ndarray:
    roots = fnp.sqrt(fnp.array(_first_primes(width), dtype=fnp.float64))
    return roots - fnp.floor(roots)


def _lattice_uniforms(
    n_samples: int, generator: fnp.ndarray, shift: fnp.ndarray
) -> fnp.ndarray:
    indices = fnp.arange(n_samples, dtype=fnp.float64)[:, None]
    unwrapped = indices * generator[None, :] + shift[None, :]
    return unwrapped - fnp.floor(unwrapped)


def _antithetic_normal_lattice(
    n_samples: int, generator: fnp.ndarray, shifts: fnp.ndarray
) -> fnp.ndarray:
    paired_count = n_samples // 2
    shift_count = int(shifts.shape[0])
    per_shift = (paired_count + shift_count - 1) // shift_count
    normal_blocks = []
    remaining = paired_count
    for shift_index in range(shift_count):
        block_count = min(per_shift, remaining)
        if block_count <= 0:
            break
        uniforms = _lattice_uniforms(block_count, generator, shifts[shift_index])
        uniforms = fnp.clip(uniforms, _TAIL_CLIP, 1.0 - _TAIL_CLIP)
        normal_blocks.append(fnp.asarray(flops.stats.norm.ppf(uniforms), dtype=fnp.float32))
        remaining -= block_count
    normals = fnp.concatenate(normal_blocks, axis=0)
    activations = fnp.concatenate([normals, -normals], axis=0)
    if n_samples % 2:
        activations = fnp.concatenate([activations, normals[:1]], axis=0)
    return activations


def _exact_first_layer_mean(first_weight: fnp.ndarray) -> fnp.ndarray:
    standard_deviation = fnp.sqrt(fnp.sum(first_weight * first_weight, axis=0))
    return standard_deviation / fnp.sqrt(2.0 * fnp.pi)


def _sample_count(budget: int, width: int, depth: int) -> int:
    lattice_per_sample = 89 * width
    forward_per_sample = depth * (2 * width * width + width)
    later_means_per_sample = max(depth - 1, 0) * width
    per_sample = lattice_per_sample + forward_per_sample + later_means_per_sample
    fixed = 2 * width * width + 8 * width
    n_samples = (int(_TARGET_FLOP_FRACTION * budget) - fixed) // per_sample
    max_samples_by_bytes = _ARRAY_BYTES_LIMIT // max(width * _FLOAT32_ITEMSIZE, 1)
    n_samples = int(max(1, min(n_samples, max_samples_by_bytes)))
    block_size = max(2 * _SHIFT_COUNT, 1)
    return max(block_size, (n_samples // block_size) * block_size)


class Estimator(BaseEstimator):
    def setup(self, context):
        self._generator_cache = {}
        self._zero_cache = {}

    def _generator_for_width(self, width: int) -> fnp.ndarray:
        cache = getattr(self, "_generator_cache", None)
        if cache is None:
            cache = {}
            self._generator_cache = cache
        generator = cache.get(width)
        if generator is None:
            generator = _lattice_generator(width)
            cache[width] = generator
        return generator

    def _zero_for_width(self, width: int) -> fnp.ndarray:
        cache = getattr(self, "_zero_cache", None)
        if cache is None:
            cache = {}
            self._zero_cache = cache
        zero = cache.get(width)
        if zero is None:
            zero = fnp.zeros(width, dtype=fnp.float32)
            cache[width] = zero
        return zero

    def predict(self, mlp, budget):
        n_samples = _sample_count(budget, mlp.width, mlp.depth)
        rng = fnp.random.default_rng(mlp.seed)
        generator = self._generator_for_width(mlp.width)
        shifts = rng.random((_SHIFT_COUNT, mlp.width))
        activations = _antithetic_normal_lattice(n_samples, generator, shifts)

        input_second = fnp.mean(activations * activations, axis=0)
        input_second = fnp.maximum(input_second, 1e-30)
        activations = activations / (input_second ** _INPUT_SECOND_POWER)

        rows = []
        zero_row = self._zero_for_width(mlp.width)
        first_activations = None
        first_exact_mean = None
        first_sample_mean = None

        for layer_index, weight in enumerate(mlp.weights):
            activations = fnp.maximum(activations @ weight, 0.0)

            if layer_index == 0:
                exact_mean = _exact_first_layer_mean(weight)
                first_sample_mean = fnp.mean(activations, axis=0)
                first_activations = activations
                first_exact_mean = exact_mean
                rows.append(exact_mean)
            elif layer_index == mlp.depth - 1:
                sampled_mean = fnp.mean(activations, axis=0)
                centered_first = first_activations - first_sample_mean
                first_variance = fnp.mean(centered_first * centered_first, axis=0)
                correction_weights = (
                    first_exact_mean - first_sample_mean
                ) / fnp.maximum(first_variance, 1e-30)
                control_scores = centered_first @ correction_weights
                correction = (control_scores @ activations) / float(n_samples)
                rows.append(sampled_mean + _FIRST_CONTROL_SCALE * correction)
            else:
                rows.append(zero_row)

        return fnp.stack(rows, axis=0)
