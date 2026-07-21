from __future__ import annotations

import math
import flopscope as flops
import flopscope.numpy as fnp
from whestbench import BaseEstimator


_TARGET_FLOP_FRACTION = 0.65
_N_SHIFTS = 2
_SEED_OFFSET = 1
_RADIUS_MULTIPLIER = 1.00013
_INDEX_START = 1
_TAIL_CLIP = 1e-11
_ARRAY_BYTES_LIMIT = 100 * 1024 * 1024
_FLOAT64_ITEMSIZE = 4


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


def _shifted_lattice_uniforms(
    samples_per_shift: int,
    generator: fnp.ndarray,
    shifts: fnp.ndarray,
) -> fnp.ndarray:
    indices = fnp.arange(
        _INDEX_START, samples_per_shift + _INDEX_START, dtype=fnp.float64
    )[:, None, None]
    unwrapped = indices * generator[None, None, :] + shifts[None, :, :]
    uniforms = unwrapped - fnp.floor(unwrapped)
    return fnp.reshape(uniforms, (samples_per_shift * shifts.shape[0], generator.shape[0]))


def _sample_count(budget: int, width: int, depth: int) -> int:
    lattice_per_sample = 89 * width
    forward_per_sample = depth * (2 * width * width + width)
    later_means_per_sample = max(depth - 1, 0) * width
    per_sample = lattice_per_sample + forward_per_sample + later_means_per_sample
    fixed = 2 * width * width + 8 * width
    n_samples = (int(_TARGET_FLOP_FRACTION * budget) - fixed) // per_sample
    max_samples_by_bytes = _ARRAY_BYTES_LIMIT // max(width * _FLOAT64_ITEMSIZE, 1)
    n_samples = int(max(_N_SHIFTS, min(n_samples, max_samples_by_bytes)))
    n_samples = (n_samples // _N_SHIFTS) * _N_SHIFTS
    return max(_N_SHIFTS, n_samples)


def _mean_gaussian_radius(width: int) -> float:
    return math.sqrt(2.0) * math.exp(
        math.lgamma(0.5 * (width + 1.0)) - math.lgamma(0.5 * width)
    )


class Estimator(BaseEstimator):
    def setup(self, context):
        self._width = context.width
        self._depth = context.depth
        self._budget = context.flop_budget
        self._generator = _lattice_generator(context.width)
        self._radius = _RADIUS_MULTIPLIER * _mean_gaussian_radius(context.width)
        self._n_samples = _sample_count(context.flop_budget, context.width, context.depth)
        self._samples_per_shift = self._n_samples // _N_SHIFTS
        self._filler = fnp.zeros(context.width, dtype=fnp.float32)
        self._zero_rows = [self._filler] * (context.depth - 1)

    def predict(self, mlp, budget):
        if (
            getattr(self, "_generator", None) is not None
            and mlp.width == self._width
            and mlp.depth == self._depth
            and budget == self._budget
        ):
            samples_per_shift = self._samples_per_shift
            generator = self._generator
            radius_scale = self._radius
        else:
            n_samples = _sample_count(budget, mlp.width, mlp.depth)
            samples_per_shift = n_samples // _N_SHIFTS
            generator = _lattice_generator(mlp.width)
            radius_scale = _RADIUS_MULTIPLIER * _mean_gaussian_radius(mlp.width)

        shifts = fnp.random.default_rng(mlp.seed + _SEED_OFFSET).random(
            (_N_SHIFTS, mlp.width)
        )
        uniforms = _shifted_lattice_uniforms(samples_per_shift, generator, shifts)
        uniforms = fnp.clip(uniforms, _TAIL_CLIP, 1.0 - _TAIL_CLIP)
        activations = fnp.asarray(flops.stats.norm.ppf(uniforms), dtype=fnp.float32)
        radius = fnp.sqrt(fnp.sum(activations * activations, axis=1))
        radius = fnp.maximum(radius, 1e-30)
        activations = activations * (radius_scale / radius[:, None])
        activations = fnp.asarray(activations, dtype=fnp.float32)

        for weight in mlp.weights:
            activations = fnp.maximum(activations @ weight, 0.0)

        final_mean = fnp.mean(activations, axis=0)

        if (
            getattr(self, "_zero_rows", None) is not None
            and mlp.width == getattr(self, "_width", None)
            and mlp.depth == getattr(self, "_depth", None)
        ):
            rows = self._zero_rows + [final_mean]
        else:
            filler = fnp.zeros(mlp.width, dtype=fnp.float32)
            rows = [filler] * (mlp.depth - 1) + [final_mean]
        return fnp.stack(rows, axis=0)
