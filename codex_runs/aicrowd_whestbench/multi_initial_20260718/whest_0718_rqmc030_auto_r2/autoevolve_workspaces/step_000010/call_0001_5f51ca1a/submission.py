"""Antithetic RQMC estimator with a first-layer control variate."""

from __future__ import annotations

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import BaseEstimator


_TARGET_FLOP_FRACTION = 0.40
_SHIFT_COUNT = 4
_FIRST_CONTROL_SCALE = 0.34
_OFFDIAGONAL_CONTROL_SCALE = -0.18
_DIAGONAL_BLEND = 0.0015
_SAMPLE_BLOCK_ADJUST = 1
_TAIL_CLIP = 6.8e-5
_ARRAY_BYTES_LIMIT = 200 * 1024 * 1024
_FLOAT32_ITEMSIZE = 4
_INV_SQRT_2PI = 0.3989422804014327


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
        normal_blocks.append(
            fnp.asarray(flops.stats.norm.ppf(uniforms), dtype=fnp.float32)
        )
        remaining -= block_count
    normals = fnp.concatenate(normal_blocks, axis=0)
    activations = fnp.concatenate([normals, -normals], axis=0)
    if n_samples % 2:
        activations = fnp.concatenate([activations, normals[:1]], axis=0)
    return activations


def _exact_first_layer_moments(
    first_weight: fnp.ndarray,
) -> tuple[fnp.ndarray, fnp.ndarray]:
    norm_squared = fnp.sum(first_weight * first_weight, axis=0)
    exact_mean = fnp.sqrt(norm_squared) * _INV_SQRT_2PI
    exact_variance = 0.5 * norm_squared - exact_mean * exact_mean
    return exact_mean, exact_variance


def _diagonal_moment_rows(mlp) -> list[fnp.ndarray]:
    mu = fnp.zeros(mlp.width, dtype=fnp.float32)
    var = fnp.ones(mlp.width, dtype=fnp.float32)
    rows = []
    for weight in mlp.weights:
        mu_pre = weight.T @ mu
        var_pre = (weight * weight).T @ var
        var_pre = fnp.maximum(var_pre, 1e-12)
        sigma_pre = fnp.sqrt(var_pre)
        alpha = mu_pre / sigma_pre
        phi_alpha = flops.stats.norm.pdf(alpha)
        Phi_alpha = flops.stats.norm.cdf(alpha)
        mu = mu_pre * Phi_alpha + sigma_pre * phi_alpha
        second = (
            (mu_pre * mu_pre + var_pre) * Phi_alpha
            + mu_pre * sigma_pre * phi_alpha
        )
        var = fnp.maximum(second - mu * mu, 0.0)
        rows.append(fnp.asarray(mu, dtype=fnp.float32))
    return rows


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
    n_samples = (n_samples // block_size) * block_size
    n_samples = n_samples + _SAMPLE_BLOCK_ADJUST * block_size
    return max(block_size, n_samples)


class Estimator(BaseEstimator):
    def setup(self, context):
        width = int(context.width)
        self._generator_cache = {width: _lattice_generator(width)}
        self._zero_cache = {width: fnp.zeros(width, dtype=fnp.float32)}

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
        activations = activations / input_second

        diagonal_rows = _diagonal_moment_rows(mlp) if _DIAGONAL_BLEND else None
        rows = []
        zero_row = self._zero_for_width(mlp.width)
        first_activations = None
        first_exact_mean = None
        first_exact_variance = None
        first_sample_mean = None

        for layer_index, weight in enumerate(mlp.weights):
            activations = fnp.maximum(activations @ weight, 0.0)

            if layer_index == 0:
                exact_mean, first_exact_variance = _exact_first_layer_moments(weight)
                first_sample_mean = fnp.mean(activations, axis=0)
                first_activations = activations
                first_exact_mean = exact_mean
                rows.append(exact_mean)
            elif layer_index == mlp.depth - 1:
                sampled_mean = fnp.mean(activations, axis=0)
                centered_first = first_activations - first_sample_mean
                correction_weights = (
                    first_exact_mean - first_sample_mean
                ) / fnp.maximum(first_exact_variance, 1e-30)
                base_scores = centered_first @ correction_weights
                covariance_times_base = (centered_first.T @ base_scores) / float(
                    n_samples
                )
                offdiagonal_effect = (
                    covariance_times_base - first_exact_variance * correction_weights
                )
                correction_weights = correction_weights - (
                    _OFFDIAGONAL_CONTROL_SCALE
                    * offdiagonal_effect
                    / fnp.maximum(first_exact_variance, 1e-30)
                )
                control_scores = centered_first @ correction_weights
                correction = (control_scores @ activations) / float(n_samples)
                controlled_mean = sampled_mean + _FIRST_CONTROL_SCALE * correction
                if _DIAGONAL_BLEND:
                    controlled_mean = controlled_mean + _DIAGONAL_BLEND * (
                        diagonal_rows[-1] - controlled_mean
                    )
                rows.append(controlled_mean)
            else:
                rows.append(zero_row)

        return fnp.stack(rows, axis=0)
