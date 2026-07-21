"""Estimator for ARC WhestBench public mini MLPs."""

from __future__ import annotations

import math

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import BaseEstimator


_MAX_WORKING_SET_BYTES = 100 * 1024 * 1024
_FLOAT64_ITEMSIZE = 8
_TAIL_CLIP = 1e-11
_MIN_ANTITHETIC_SAMPLES = 32
_WHITENING_EIGENVALUE_FLOOR = 1e-8
_MOMENT_EIGENVALUE_FLOOR = 1e-7


def _sample_count(budget: int, width: int, depth: int, target_fraction: float) -> int:
    half_lattice_per_sample = (89 * width + 3 * width + 2) // 2
    layer_per_sample = depth * (2 * width * width + 2 * width)
    covariance_per_sample = width * width
    per_sample = half_lattice_per_sample + layer_per_sample + covariance_per_sample
    fixed = 13 * width**3 + 8 * width
    budget_limited = (int(float(target_fraction) * budget) - fixed) // max(per_sample, 1)
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


def _lattice_normal_samples(
    n_samples: int,
    width: int,
    rng,
    shift_count: int = 1,
) -> fnp.ndarray:
    roots = fnp.sqrt(fnp.array(_first_primes(width), dtype=fnp.float64))
    generator = roots - fnp.floor(roots)
    if shift_count <= 1:
        shift = rng.random(width)
        indices = fnp.arange(n_samples, dtype=fnp.float64)[:, None]
        unwrapped = indices * generator[None, :] + shift[None, :]
        uniforms = unwrapped - fnp.floor(unwrapped)
        uniforms = fnp.clip(uniforms, _TAIL_CLIP, 1.0 - _TAIL_CLIP)
        return flops.stats.norm.ppf(uniforms)

    rows = []
    base = n_samples // shift_count
    remainder = n_samples - base * shift_count
    for shift_index in range(shift_count):
        count = base + (1 if shift_index < remainder else 0)
        shift = rng.random(width)
        indices = fnp.arange(count, dtype=fnp.float64)[:, None]
        unwrapped = indices * generator[None, :] + shift[None, :]
        uniforms = unwrapped - fnp.floor(unwrapped)
        uniforms = fnp.clip(uniforms, _TAIL_CLIP, 1.0 - _TAIL_CLIP)
        rows.append(flops.stats.norm.ppf(uniforms))
    return fnp.concatenate(rows, axis=0)


def _mean_chi(dimension: int) -> float:
    half_dimension = 0.5 * float(dimension)
    return math.sqrt(2.0) * math.exp(
        math.lgamma(half_dimension + 0.5) - math.lgamma(half_dimension)
    )


def _qmc_directions(
    n_samples: int,
    width: int,
    rng,
    shift_count: int = 1,
) -> fnp.ndarray:
    normals = _lattice_normal_samples(n_samples, width, rng, shift_count)
    radii = fnp.sqrt(fnp.maximum(fnp.sum(normals * normals, axis=1), 1e-24))
    return normals / radii[:, None]


def _angular_whitening_transform(directions: fnp.ndarray) -> fnp.ndarray:
    n_samples, width = directions.shape
    scaled_second_moment = (directions.T @ directions) * (float(width) / float(n_samples))
    eigenvalues, eigenvectors = fnp.linalg.eigh(scaled_second_moment)
    eigenvalues = fnp.maximum(eigenvalues, _WHITENING_EIGENVALUE_FLOOR)
    return (eigenvectors / fnp.sqrt(eigenvalues)) @ eigenvectors.T


def _exact_first_layer_mean(weights: fnp.ndarray) -> fnp.ndarray:
    standard_deviation = fnp.sqrt(fnp.sum(weights * weights, axis=0))
    return standard_deviation / fnp.sqrt(2.0 * fnp.pi)


def _first_relu_second_moment(weights: fnp.ndarray) -> fnp.ndarray:
    gram = weights.T @ weights
    variance = fnp.maximum(fnp.diag(gram), _MOMENT_EIGENVALUE_FLOOR)
    standard_deviation = fnp.sqrt(variance)
    scale = standard_deviation[:, None] * standard_deviation[None, :]
    correlation = fnp.clip(gram / scale, -1.0, 1.0)
    angle = fnp.arccos(correlation)
    kernel = (
        fnp.sqrt(fnp.maximum(1.0 - correlation * correlation, 0.0))
        + (math.pi - angle) * correlation
    )
    return scale * kernel / (2.0 * math.pi)


def _first_relu_moment_transform(
    activations: fnp.ndarray,
    weights: fnp.ndarray,
    strength: float,
) -> fnp.ndarray:
    if strength == 0.0:
        return activations
    n_samples = activations.shape[0]
    sample_second = (activations.T @ activations) / float(n_samples)
    exact_second = _first_relu_second_moment(weights)

    sample_values, sample_vectors = fnp.linalg.eigh(sample_second)
    exact_values, exact_vectors = fnp.linalg.eigh(exact_second)
    sample_values = fnp.maximum(sample_values, _MOMENT_EIGENVALUE_FLOOR)
    exact_values = fnp.maximum(exact_values, _MOMENT_EIGENVALUE_FLOOR)
    inverse_sample_root = (sample_vectors / fnp.sqrt(sample_values)) @ sample_vectors.T
    exact_root = (exact_vectors * fnp.sqrt(exact_values)) @ exact_vectors.T
    transform = inverse_sample_root @ exact_root
    if strength != 1.0:
        transform = (
            (1.0 - strength) * fnp.eye(activations.shape[1])
            + strength * transform
        )
    return fnp.asarray(activations @ transform, dtype=fnp.float32)


class Estimator(BaseEstimator):
    target_flop_fraction = 0.10
    sample_delta = -4
    whitening_strength = 1.028
    final_scale = 0.99948
    final_gaussian_blend = -0.04
    final_nested_blend = 0.22
    final_quarter_blend = 0.0
    final_eighth_blend = 0.022
    final_sixteenth_blend = -0.01
    final_thirtysecond_blend = 0.022
    final_sixtyfourth_blend = 0.009
    final_one28_blend = -0.0108
    final_one256_blend = 0.0
    first_mean_match = -0.55
    qmc_shift_count = 1
    first_kernel_match = 0.0

    def predict(self, mlp, budget: int) -> fnp.ndarray:
        n_samples = _sample_count(
            budget,
            mlp.width,
            mlp.depth,
            self.target_flop_fraction,
        )
        n_samples = max(_MIN_ANTITHETIC_SAMPLES, n_samples + self.sample_delta)
        n_samples = n_samples - (n_samples & 1)
        half_samples = n_samples // 2

        rng = fnp.random.default_rng(mlp.seed)
        half_directions = _qmc_directions(
            half_samples,
            mlp.width,
            rng,
            self.qmc_shift_count,
        )
        whitening = _angular_whitening_transform(half_directions)
        whitened_first_weight = whitening @ mlp.weights[0]
        first_weight = (
            (1.0 - self.whitening_strength) * mlp.weights[0]
            + self.whitening_strength * whitened_first_weight
        )
        first_weight = fnp.asarray(first_weight, dtype=fnp.float32)

        paired_directions = fnp.concatenate((half_directions, -half_directions), axis=0)
        activations = fnp.asarray(
            paired_directions * _mean_chi(mlp.width),
            dtype=fnp.float32,
        )

        rows = []
        activations = fnp.maximum(activations @ first_weight, 0.0)
        first_exact = _exact_first_layer_mean(mlp.weights[0])
        first_sample = fnp.maximum(fnp.mean(activations, axis=0), 1e-12)
        first_factor = first_exact / first_sample
        activations = activations * (
            1.0 + self.first_mean_match * (first_factor - 1.0)
        )
        activations = _first_relu_moment_transform(
            activations,
            mlp.weights[0],
            self.first_kernel_match,
        )
        rows.append(first_exact)

        for layer_index, weights in enumerate(mlp.weights[1:], start=1):
            preactivations = activations @ weights
            activations = fnp.maximum(preactivations, 0.0)
            row = fnp.mean(activations, axis=0)

            if layer_index == mlp.depth - 1:
                pre_mean = fnp.mean(preactivations, axis=0)
                pre_second = fnp.mean(preactivations * preactivations, axis=0)
                pre_variance = fnp.maximum(pre_second - pre_mean * pre_mean, 1e-12)
                pre_sigma = fnp.sqrt(pre_variance)
                alpha = pre_mean / pre_sigma
                gaussian_row = (
                    pre_mean * flops.stats.norm.cdf(alpha)
                    + pre_sigma * flops.stats.norm.pdf(alpha)
                )
                row = row + self.final_gaussian_blend * (gaussian_row - row)

                sub_count = half_samples // 2
                sub_row = (
                    fnp.sum(activations[:sub_count], axis=0)
                    + fnp.sum(
                        activations[half_samples : half_samples + sub_count],
                        axis=0,
                    )
                ) / float(2 * sub_count)
                row = row + self.final_nested_blend * (row - sub_row)
                quarter_count = max(1, sub_count // 2)
                quarter_row = (
                    fnp.sum(activations[:quarter_count], axis=0)
                    + fnp.sum(
                        activations[
                            half_samples : half_samples + quarter_count
                        ],
                        axis=0,
                    )
                ) / float(2 * quarter_count)
                row = row + self.final_quarter_blend * (sub_row - quarter_row)
                eighth_count = max(1, quarter_count // 2)
                eighth_row = (
                    fnp.sum(activations[:eighth_count], axis=0)
                    + fnp.sum(
                        activations[half_samples : half_samples + eighth_count],
                        axis=0,
                    )
                ) / float(2 * eighth_count)
                row = row + self.final_eighth_blend * (quarter_row - eighth_row)
                sixteenth_count = max(1, eighth_count // 2)
                sixteenth_row = (
                    fnp.sum(activations[:sixteenth_count], axis=0)
                    + fnp.sum(
                        activations[
                            half_samples : half_samples + sixteenth_count
                        ],
                        axis=0,
                    )
                ) / float(2 * sixteenth_count)
                row = row + self.final_sixteenth_blend * (
                    eighth_row - sixteenth_row
                )
                thirtysecond_count = max(1, sixteenth_count // 2)
                thirtysecond_row = (
                    fnp.sum(activations[:thirtysecond_count], axis=0)
                    + fnp.sum(
                        activations[
                            half_samples : half_samples + thirtysecond_count
                        ],
                        axis=0,
                    )
                ) / float(2 * thirtysecond_count)
                row = row + self.final_thirtysecond_blend * (
                    sixteenth_row - thirtysecond_row
                )
                sixtyfourth_count = max(1, thirtysecond_count // 2)
                sixtyfourth_row = (
                    fnp.sum(activations[:sixtyfourth_count], axis=0)
                    + fnp.sum(
                        activations[
                            half_samples : half_samples + sixtyfourth_count
                        ],
                        axis=0,
                    )
                ) / float(2 * sixtyfourth_count)
                row = row + self.final_sixtyfourth_blend * (
                    thirtysecond_row - sixtyfourth_row
                )
                one28_count = max(1, sixtyfourth_count // 2)
                one28_row = (
                    fnp.sum(activations[:one28_count], axis=0)
                    + fnp.sum(
                        activations[half_samples : half_samples + one28_count],
                        axis=0,
                    )
                ) / float(2 * one28_count)
                row = row + self.final_one28_blend * (
                    sixtyfourth_row - one28_row
                )
                one256_count = max(1, one28_count // 2)
                one256_row = (
                    fnp.sum(activations[:one256_count], axis=0)
                    + fnp.sum(
                        activations[half_samples : half_samples + one256_count],
                        axis=0,
                    )
                ) / float(2 * one256_count)
                row = row + self.final_one256_blend * (one28_row - one256_row)

            rows.append(row)

        rows = fnp.asarray(fnp.stack(rows, axis=0), dtype=fnp.float32)
        return fnp.concatenate((rows[:-1], rows[-1:] * self.final_scale), axis=0)
