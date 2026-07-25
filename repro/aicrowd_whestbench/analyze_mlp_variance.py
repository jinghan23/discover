#!/usr/bin/env python3
"""Audit WhestBench MLP generation and measure suite-composition variance.

This program deliberately implements the MLP generator with NumPy instead of
calling ``whestbench.sample_mlp``.  It can:

1. reproduce every weight in the public mini split from its stored root seed;
2. generate fresh, statistically equivalent MLPs and measure how much their
   final activation variance differs;
3. bootstrap stored full-100 reports to estimate how stable method rankings are
   when the evaluation suite contains a different sample of MLPs.

The public dataset audit requires ``whestbench`` and its dependencies.  The
fresh-generation and report-bootstrap paths only require NumPy.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DEFAULT_DATASET = "aicrowd/arc-whestbench-public-2026"
DEFAULT_REVISION = "v1-phase1"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = (
    REPO_ROOT / "repro_external/aicrowd_whestbench/submissions/registry.json"
)
DEFAULT_OFFICIAL_STATUS = (
    REPO_ROOT
    / "repro_external/aicrowd_whestbench/submissions/official_status.json"
)


def _quantiles(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    probabilities = (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)
    return {
        f"q{int(probability * 100):02d}": float(value)
        for probability, value in zip(probabilities, np.quantile(array, probabilities))
    }


def _pearson(left: Iterable[float], right: Iterable[float]) -> float:
    x = np.asarray(list(left), dtype=np.float64)
    y = np.asarray(list(right), dtype=np.float64)
    if x.size < 2 or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def generate_official_weights(
    root_seed: int,
    *,
    width: int = 256,
    depth: int = 32,
) -> np.ndarray:
    """Independently implement seed-protocol 3.0 and He-normal generation.

    Protocol 3.0 expands each per-MLP root seed into three streams.  Stream zero
    generates weights; stream one generates ground-truth inputs; stream two is
    exposed to participant estimators.  Each weight is N(0, 2 / width), rounded
    to float32 after sampling.
    """

    if width <= 0 or depth <= 0:
        raise ValueError("width and depth must be positive")
    weight_seed = np.random.SeedSequence(int(root_seed)).spawn(3)[0]
    rng = np.random.default_rng(weight_seed)
    scale = math.sqrt(2.0 / width)
    # Keep the per-layer calls: this mirrors the official generator exactly and
    # also makes byte-for-byte reproduction checks meaningful.
    return np.stack(
        [
            (rng.standard_normal((width, width)) * scale).astype(np.float32)
            for _ in range(depth)
        ]
    )


@dataclass
class WeightAudit:
    expected_variance: float
    count: int = 0
    total: float = 0.0
    total_squares: float = 0.0
    adjacent_count: int = 0
    adjacent_products: float = 0.0
    per_mlp_std: list[float] = field(default_factory=list)
    log_layer_gain_products: list[float] = field(default_factory=list)

    def add(self, weights: np.ndarray) -> None:
        values = np.asarray(weights, dtype=np.float32)
        values64 = values.astype(np.float64, copy=False)
        self.count += int(values.size)
        self.total += float(np.sum(values64, dtype=np.float64))
        self.total_squares += float(np.sum(values64 * values64, dtype=np.float64))
        self.per_mlp_std.append(float(np.std(values64, dtype=np.float64)))

        if values.shape[0] > 1:
            self.adjacent_count += int(values[:-1].size)
            self.adjacent_products += float(
                np.sum(
                    values64[:-1] * values64[1:],
                    dtype=np.float64,
                )
            )

        layer_variances = np.mean(values64 * values64, axis=(1, 2))
        self.log_layer_gain_products.append(
            float(np.sum(np.log(layer_variances / self.expected_variance)))
        )

    def summary(self) -> dict[str, Any]:
        mean = self.total / self.count
        variance = self.total_squares / self.count - mean * mean
        std = math.sqrt(max(0.0, variance))
        expected_std = math.sqrt(self.expected_variance)
        mean_standard_error = expected_std / math.sqrt(self.count)
        variance_standard_error = self.expected_variance * math.sqrt(
            2.0 / max(1, self.count - 1)
        )
        adjacent_covariance = self.adjacent_products / self.adjacent_count - mean * mean
        return {
            "count": self.count,
            "expected_mean": 0.0,
            "empirical_mean": mean,
            "mean_z_score": mean / mean_standard_error,
            "expected_variance": self.expected_variance,
            "empirical_variance": variance,
            "variance_relative_error": variance / self.expected_variance - 1.0,
            "variance_z_score_approx": (
                variance - self.expected_variance
            )
            / variance_standard_error,
            "expected_std": expected_std,
            "empirical_std": std,
            "per_mlp_std_quantiles": _quantiles(self.per_mlp_std),
            "adjacent_layer_element_correlation": adjacent_covariance / variance,
        }


def _simulate_activation_statistics(
    weights: np.ndarray,
    *,
    sample_seed: np.random.SeedSequence,
    n_samples: int,
    chunk_size: int,
) -> tuple[float, np.ndarray]:
    """Monte-Carlo final variance and mean second moment at every layer."""

    depth, width, _ = weights.shape
    rng = np.random.default_rng(sample_seed)
    layer_sum_squares = np.zeros(depth, dtype=np.float64)
    final_sums = np.zeros(width, dtype=np.float64)
    final_sum_squares = np.zeros(width, dtype=np.float64)
    processed = 0

    while processed < n_samples:
        current = min(chunk_size, n_samples - processed)
        activations = rng.standard_normal((current, width), dtype=np.float32)
        for layer, matrix in enumerate(weights):
            activations = np.maximum(activations @ matrix, np.float32(0.0))
            values64 = activations.astype(np.float64, copy=False)
            layer_sum_squares[layer] += float(
                np.sum(values64 * values64, dtype=np.float64)
            )
        final64 = activations.astype(np.float64, copy=False)
        final_sums += np.sum(final64, axis=0, dtype=np.float64)
        final_sum_squares += np.sum(final64 * final64, axis=0, dtype=np.float64)
        processed += current

    final_means = final_sums / n_samples
    final_variance = float(
        np.mean(final_sum_squares / n_samples - final_means * final_means)
    )
    layer_second_moments = layer_sum_squares / (n_samples * width)
    return final_variance, layer_second_moments


def analyze_generated_mlps(
    *,
    n_mlps: int,
    width: int,
    depth: int,
    n_samples: int,
    repeats: int,
    chunk_size: int,
    seed: int,
) -> dict[str, Any]:
    """Generate fresh MLPs and separate real network spread from MC noise."""

    if n_mlps <= 1:
        raise ValueError("n_mlps must be greater than one")
    if n_samples <= 1 or repeats <= 0 or chunk_size <= 0:
        raise ValueError("n_samples, repeats, and chunk_size must be positive")

    master = np.random.default_rng(seed)
    roots: list[int] = []
    seen: set[int] = set()
    while len(roots) < n_mlps:
        candidate = int(master.integers(0, 2**63, dtype=np.int64))
        if candidate not in seen:
            seen.add(candidate)
            roots.append(candidate)

    weight_audit = WeightAudit(expected_variance=2.0 / width)
    repeat_variances = np.empty((n_mlps, repeats), dtype=np.float64)
    layer_second_moments = np.empty((n_mlps, repeats, depth), dtype=np.float64)
    rows: list[dict[str, Any]] = []

    for mlp_index, root_seed in enumerate(roots):
        weights = generate_official_weights(root_seed, width=width, depth=depth)
        weight_audit.add(weights)
        sample_stream = np.random.SeedSequence(root_seed).spawn(3)[1]
        repeat_streams = sample_stream.spawn(repeats)
        for repeat_index, repeat_stream in enumerate(repeat_streams):
            variance, layer_m2 = _simulate_activation_statistics(
                weights,
                sample_seed=repeat_stream,
                n_samples=n_samples,
                chunk_size=chunk_size,
            )
            repeat_variances[mlp_index, repeat_index] = variance
            layer_second_moments[mlp_index, repeat_index] = layer_m2
        rows.append(
            {
                "mlp_index": mlp_index,
                "root_seed": root_seed,
                "weight_std": weight_audit.per_mlp_std[-1],
                "log_layer_gain_product": weight_audit.log_layer_gain_products[-1],
                "final_variance_repeats": repeat_variances[mlp_index].tolist(),
                "final_variance_mean": float(np.mean(repeat_variances[mlp_index])),
            }
        )

    per_mlp_variance = np.mean(repeat_variances, axis=1)
    within_mc_variance = (
        float(np.mean(np.var(repeat_variances, axis=1, ddof=1)))
        if repeats > 1
        else float("nan")
    )
    observed_between_variance = float(np.var(per_mlp_variance, ddof=1))
    true_between_variance = (
        max(0.0, observed_between_variance - within_mc_variance / repeats)
        if repeats > 1
        else observed_between_variance
    )
    reliability = (
        true_between_variance / (true_between_variance + within_mc_variance)
        if repeats > 1 and true_between_variance + within_mc_variance > 0.0
        else float("nan")
    )

    mean_layer_m2 = np.mean(layer_second_moments, axis=1)
    layer_spread = []
    for layer in range(depth):
        values = mean_layer_m2[:, layer]
        layer_spread.append(
            {
                "layer": layer,
                "mean_second_moment": float(np.mean(values)),
                "coefficient_of_variation_across_mlps": float(
                    np.std(values, ddof=1) / np.mean(values)
                ),
                "q95_over_q05": float(
                    np.quantile(values, 0.95) / np.quantile(values, 0.05)
                ),
            }
        )

    mean_variance = float(np.mean(per_mlp_variance))
    summary = {
        "n_mlps": n_mlps,
        "width": width,
        "depth": depth,
        "mc_samples_per_repeat": n_samples,
        "mc_repeats": repeats,
        "master_seed": seed,
        "weight_audit": weight_audit.summary(),
        "final_activation_variance": {
            "mean": mean_variance,
            "std_across_mlps": float(np.std(per_mlp_variance, ddof=1)),
            "coefficient_of_variation_across_mlps": float(
                np.std(per_mlp_variance, ddof=1) / mean_variance
            ),
            "quantiles": _quantiles(per_mlp_variance),
            "max_over_min": float(np.max(per_mlp_variance) / np.min(per_mlp_variance)),
            "within_mlp_mc_variance": _finite_or_none(within_mc_variance),
            "estimated_true_between_mlp_variance": true_between_variance,
            "between_mlp_reliability": _finite_or_none(reliability),
            "correlation_log_variance_vs_weight_std": _pearson(
                np.log(per_mlp_variance), weight_audit.per_mlp_std
            ),
            "correlation_log_variance_vs_log_layer_gain_product": _pearson(
                np.log(per_mlp_variance), weight_audit.log_layer_gain_products
            ),
        },
        "layer_second_moment_spread": layer_spread,
        "per_mlp": rows,
    }
    return summary


def audit_official_mini(
    *,
    dataset: str,
    revision: str,
    split: str,
    n_mlps: int,
) -> dict[str, Any]:
    """Regenerate public rows from root seeds and compare every float32 weight."""

    try:
        from whestbench import load_dataset
    except ImportError as error:
        raise RuntimeError(
            "The dataset audit needs whestbench. Add the official dependency "
            "directory to PYTHONPATH (in this workspace: /tmp/whest-official-deps)."
        ) from error

    loaded = load_dataset(
        dataset,
        revision=revision,
        split=split,
        streaming=False,
    )
    n_rows = min(n_mlps, len(loaded))
    first_weights = np.asarray(loaded[0]["weights"], dtype=np.float32)
    depth, width, _ = first_weights.shape
    audit = WeightAudit(expected_variance=2.0 / width)
    exact_rows = 0
    mismatched_elements = 0
    stored_dtypes: set[str] = set()
    avg_variances: list[float] = []

    for index in range(n_rows):
        row = loaded[index]
        stored = np.asarray(row["weights"], dtype=np.float32)
        regenerated = generate_official_weights(
            int(row["mlp_seed"]),
            width=width,
            depth=depth,
        )
        equal = stored == regenerated
        exact_rows += int(bool(np.all(equal)))
        mismatched_elements += int(equal.size - np.count_nonzero(equal))
        stored_dtypes.add(str(stored.dtype))
        avg_variances.append(float(row["avg_variance"]))
        audit.add(stored)

    avg_array = np.asarray(avg_variances, dtype=np.float64)
    return {
        "dataset": dataset,
        "revision": revision,
        "split": split,
        "n_mlps_audited": n_rows,
        "width": width,
        "depth": depth,
        "seed_protocol": "3.0: SeedSequence(root).spawn(3)[0] for weights",
        "stored_dtypes_after_schema_cast": sorted(stored_dtypes),
        "byte_exact_regeneration_rows": exact_rows,
        "mismatched_weight_elements": mismatched_elements,
        "all_rows_regenerate_exactly": exact_rows == n_rows and mismatched_elements == 0,
        "weight_audit": audit.summary(),
        "stored_final_activation_variance": {
            "mean": float(np.mean(avg_array)),
            "std_across_mlps": float(np.std(avg_array, ddof=1)),
            "coefficient_of_variation_across_mlps": float(
                np.std(avg_array, ddof=1) / np.mean(avg_array)
            ),
            "quantiles": _quantiles(avg_array),
            "max_over_min": float(np.max(avg_array) / np.min(avg_array)),
        },
    }


def _load_full100_reports(
    registry_path: Path,
) -> tuple[list[str], list[int], list[Path], np.ndarray, list[str]]:
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    labels: list[str] = []
    submission_ids: list[int] = []
    paths: list[Path] = []
    score_rows: list[list[float]] = []
    reference_names: list[str] | None = None
    seen_paths: set[Path] = set()

    for entry in registry["entries"]:
        raw_path = Path(entry["local_eval"])
        report_path = raw_path if raw_path.is_absolute() else REPO_ROOT / raw_path
        report_path = report_path.resolve()
        if report_path in seen_paths or not report_path.exists():
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        per_mlp = report.get("results", {}).get("per_mlp", [])
        if len(per_mlp) != 100:
            continue
        names = [str(row["mlp_name"]) for row in per_mlp]
        if reference_names is None:
            reference_names = names
        if names != reference_names:
            continue
        labels.append(str(entry["method"]))
        submission_ids.append(int(entry["submission_id"]))
        paths.append(report_path)
        score_rows.append(
            [float(row["adjusted_final_layer_score"]) for row in per_mlp]
        )
        seen_paths.add(report_path)

    if not score_rows or reference_names is None:
        raise ValueError(f"no aligned full-100 reports found in {registry_path}")
    return (
        labels,
        submission_ids,
        paths,
        np.asarray(score_rows, dtype=np.float64),
        reference_names,
    )


def bootstrap_report_rankings(
    *,
    registry_path: Path,
    official_status_path: Path | None,
    suite_size: int,
    repeats: int,
    top_methods: int,
    seed: int,
) -> dict[str, Any]:
    """Bootstrap full-100 per-MLP rows into independent pseudo public suites."""

    labels, submission_ids, paths, scores, mlp_names = _load_full100_reports(
        registry_path
    )
    full_means = np.mean(scores, axis=1)
    selected = np.argsort(full_means)
    if top_methods > 0:
        selected = selected[: min(top_methods, selected.size)]
    labels = [labels[index] for index in selected]
    submission_ids = [submission_ids[index] for index in selected]
    paths = [paths[index] for index in selected]
    scores = scores[selected]
    full_means = full_means[selected]

    n_methods, n_mlps = scores.shape
    if n_methods < 2:
        raise ValueError("rank bootstrap needs at least two methods")
    if suite_size <= 0 or repeats <= 0:
        raise ValueError("suite_size and repeats must be positive")

    full_order = np.argsort(full_means)
    full_ranks = np.empty(n_methods, dtype=np.int64)
    full_ranks[full_order] = np.arange(n_methods)
    pair_i, pair_j = np.triu_indices(n_methods, k=1)
    full_pair_sign = np.sign(full_means[pair_i] - full_means[pair_j])
    pair_inversions = np.zeros(pair_i.size, dtype=np.int64)
    spearman_values = np.empty(repeats, dtype=np.float64)
    inversion_values = np.empty(repeats, dtype=np.float64)
    top1_hits = 0
    top3_overlaps = np.empty(repeats, dtype=np.float64)
    rng = np.random.default_rng(seed)

    for repeat in range(repeats):
        sampled_mlps = rng.integers(0, n_mlps, size=suite_size)
        sampled_means = np.mean(scores[:, sampled_mlps], axis=1)
        sampled_order = np.argsort(sampled_means)
        sampled_ranks = np.empty(n_methods, dtype=np.int64)
        sampled_ranks[sampled_order] = np.arange(n_methods)
        rank_delta = sampled_ranks - full_ranks
        spearman_values[repeat] = 1.0 - (
            6.0 * float(np.sum(rank_delta * rank_delta))
            / (n_methods * (n_methods * n_methods - 1))
        )
        sampled_pair_sign = np.sign(sampled_means[pair_i] - sampled_means[pair_j])
        inverted = sampled_pair_sign != full_pair_sign
        pair_inversions += inverted
        inversion_values[repeat] = float(np.mean(inverted))
        top1_hits += int(sampled_order[0] == full_order[0])
        top3_overlaps[repeat] = len(
            set(sampled_order[:3]).intersection(full_order[:3])
        ) / 3.0

    official_scores: np.ndarray | None = None
    if official_status_path is not None and official_status_path.exists():
        status = json.loads(
            official_status_path.read_text(encoding="utf-8")
        ).get("submissions", {})
        selected_status = [status.get(str(identifier), {}) for identifier in submission_ids]
        if all(
            row.get("status") == "graded" and row.get("score") is not None
            for row in selected_status
        ):
            official_scores = np.asarray(
                [float(row["score"]) for row in selected_status],
                dtype=np.float64,
            )

    official_ranks: np.ndarray | None = None
    if official_scores is not None:
        official_order = np.argsort(official_scores)
        official_ranks = np.empty(n_methods, dtype=np.int64)
        official_ranks[official_order] = np.arange(n_methods)

    method_rows = []
    for rank, method_index in enumerate(full_order, start=1):
        per_mlp = scores[method_index]
        row: dict[str, Any] = {
            "full100_rank": rank,
            "method": labels[method_index],
            "submission_id": submission_ids[method_index],
            "full100_mean": float(full_means[method_index]),
            "per_mlp_std": float(np.std(per_mlp, ddof=1)),
            "per_mlp_coefficient_of_variation": float(
                np.std(per_mlp, ddof=1) / np.mean(per_mlp)
            ),
            "estimated_suite_relative_standard_error": float(
                np.std(per_mlp, ddof=1)
                / math.sqrt(suite_size)
                / np.mean(per_mlp)
            ),
            "report": _display_path(paths[method_index]),
        }
        if official_scores is not None and official_ranks is not None:
            row.update(
                {
                    "hosted_public50_rank_among_selected": int(
                        official_ranks[method_index] + 1
                    ),
                    "hosted_public50_score": float(official_scores[method_index]),
                    "hosted_vs_local_relative_delta": float(
                        official_scores[method_index] / full_means[method_index] - 1.0
                    ),
                }
            )
        method_rows.append(row)

    adjacent_rows = []
    inversion_rates = pair_inversions / repeats
    pair_rate = {
        (int(left), int(right)): float(rate)
        for left, right, rate in zip(pair_i, pair_j, inversion_rates)
    }
    for rank in range(n_methods - 1):
        better = int(full_order[rank])
        worse = int(full_order[rank + 1])
        key = (min(better, worse), max(better, worse))
        adjacent_rows.append(
            {
                "better_full100_rank": rank + 1,
                "better_method": labels[better],
                "worse_method": labels[worse],
                "full100_relative_gap": float(
                    full_means[worse] / full_means[better] - 1.0
                ),
                "bootstrap_order_inversion_probability": pair_rate[key],
            }
        )

    result: dict[str, Any] = {
        "registry": str(registry_path),
        "source_mlp_count": n_mlps,
        "source_mlp_names_sha256_note": (
            f"{len(mlp_names)} aligned names; raw names omitted from output"
        ),
        "selected_method_count": n_methods,
        "pseudo_suite_size": suite_size,
        "bootstrap_repeats": repeats,
        "bootstrap_seed": seed,
        "sampling": (
            "MLPs sampled with replacement from aligned local full-100 per-MLP "
            "scores; this estimates suite-composition uncertainty, not grader drift"
        ),
        "spearman_vs_full100": {
            "mean": float(np.mean(spearman_values)),
            "quantiles": _quantiles(spearman_values),
        },
        "pairwise_order_inversion_fraction": {
            "mean": float(np.mean(inversion_values)),
            "quantiles": _quantiles(inversion_values),
        },
        "full100_top1_retained_probability": top1_hits / repeats,
        "mean_top3_overlap_fraction": float(np.mean(top3_overlaps)),
        "methods": method_rows,
        "adjacent_pair_inversions": adjacent_rows,
    }
    if official_scores is not None and official_ranks is not None:
        actual_rank_delta = official_ranks - full_ranks
        actual_spearman = 1.0 - (
            6.0 * float(np.sum(actual_rank_delta * actual_rank_delta))
            / (n_methods * (n_methods * n_methods - 1))
        )
        official_pair_sign = np.sign(
            official_scores[pair_i] - official_scores[pair_j]
        )
        actual_inversion = float(np.mean(official_pair_sign != full_pair_sign))
        result["actual_hosted_public50_comparison"] = {
            "official_status": str(official_status_path),
            "spearman_vs_full100": actual_spearman,
            "spearman_lower_tail_probability_under_bootstrap": float(
                np.mean(spearman_values <= actual_spearman)
            ),
            "pairwise_order_inversion_fraction": actual_inversion,
            "inversion_upper_tail_probability_under_bootstrap": float(
                np.mean(inversion_values >= actual_inversion)
            ),
            "local_full100_top1_hosted_rank_among_selected": int(
                official_ranks[full_order[0]] + 1
            ),
        }
    return result


def _print_summary(report: dict[str, Any]) -> None:
    if "official_mini_audit" in report:
        audit = report["official_mini_audit"]
        weights = audit["weight_audit"]
        variance = audit["stored_final_activation_variance"]
        print(
            "official mini:"
            f" exact_rows={audit['byte_exact_regeneration_rows']}/"
            f"{audit['n_mlps_audited']},"
            f" weight_var={weights['empirical_variance']:.9g}"
            f" (target={weights['expected_variance']:.9g}),"
            f" final_var_CV={variance['coefficient_of_variation_across_mlps']:.3f},"
            f" max/min={variance['max_over_min']:.2f}"
        )
    if "generated_mlp_analysis" in report:
        generated = report["generated_mlp_analysis"]
        variance = generated["final_activation_variance"]
        print(
            "fresh MLPs:"
            f" n={generated['n_mlps']},"
            f" final_var_CV={variance['coefficient_of_variation_across_mlps']:.3f},"
            f" max/min={variance['max_over_min']:.2f},"
            f" reliability={variance['between_mlp_reliability']}"
        )
    if "rank_bootstrap" in report:
        ranks = report["rank_bootstrap"]
        message = (
            "rank bootstrap:"
            f" methods={ranks['selected_method_count']},"
            f" suite={ranks['pseudo_suite_size']},"
            f" mean_spearman={ranks['spearman_vs_full100']['mean']:.3f},"
            f" top1_retained={ranks['full100_top1_retained_probability']:.3f},"
            " mean_pair_inversion="
            f"{ranks['pairwise_order_inversion_fraction']['mean']:.3f}"
        )
        actual = ranks.get("actual_hosted_public50_comparison")
        if actual:
            message += (
                f", actual_hosted_spearman={actual['spearman_vs_full100']:.3f},"
                " actual_hosted_pair_inversion="
                f"{actual['pairwise_order_inversion_fraction']:.3f}"
            )
        print(message)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-official-mini", action="store_true")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--split", default="mini")
    parser.add_argument("--audit-n-mlps", type=int, default=100)
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--generated-mlps", type=int, default=100)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--depth", type=int, default=32)
    parser.add_argument("--mc-samples", type=int, default=8192)
    parser.add_argument("--mc-repeats", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument("--generation-seed", type=int, default=20260723)
    parser.add_argument("--skip-rank-bootstrap", action="store_true")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument(
        "--official-status",
        type=Path,
        default=DEFAULT_OFFICIAL_STATUS,
        help="Cached hosted scores used for an actual local-vs-hosted comparison.",
    )
    parser.add_argument("--suite-size", type=int, default=50)
    parser.add_argument("--bootstrap-repeats", type=int, default=10_000)
    parser.add_argument(
        "--top-methods",
        type=int,
        default=15,
        help="Bootstrap the best N full-100 methods; 0 means all.",
    )
    parser.add_argument("--bootstrap-seed", type=int, default=20260723)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report: dict[str, Any] = {
        "protocol": {
            "weights": "independent N(0, 2/width), float32, no bias",
            "network": "x <- ReLU(x @ W), repeated depth times",
            "input": "independent N(0, 1), float32",
        }
    }
    if args.audit_official_mini:
        report["official_mini_audit"] = audit_official_mini(
            dataset=args.dataset,
            revision=args.revision,
            split=args.split,
            n_mlps=args.audit_n_mlps,
        )
    if not args.skip_generation:
        report["generated_mlp_analysis"] = analyze_generated_mlps(
            n_mlps=args.generated_mlps,
            width=args.width,
            depth=args.depth,
            n_samples=args.mc_samples,
            repeats=args.mc_repeats,
            chunk_size=args.chunk_size,
            seed=args.generation_seed,
        )
    if not args.skip_rank_bootstrap:
        report["rank_bootstrap"] = bootstrap_report_rankings(
            registry_path=args.registry.resolve(),
            official_status_path=args.official_status.resolve(),
            suite_size=args.suite_size,
            repeats=args.bootstrap_repeats,
            top_methods=args.top_methods,
            seed=args.bootstrap_seed,
        )

    _print_summary(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {args.output}")
    else:
        print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
