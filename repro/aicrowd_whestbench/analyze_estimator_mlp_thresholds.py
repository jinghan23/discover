#!/usr/bin/env python3
"""Estimate across-MLP score variance and paired promotion thresholds."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import t


SCORE_KEY = "corrected_flops_only_adjusted_score"


def _quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        name: float(np.quantile(values, quantile))
        for name, quantile in (
            ("min", 0.0),
            ("q25", 0.25),
            ("median", 0.50),
            ("q75", 0.75),
            ("q90", 0.90),
            ("q95", 0.95),
            ("max", 1.0),
        )
    }


def analyze_thresholds(
    artifact: dict[str, Any],
    *,
    target_sizes: tuple[int, ...] = (50, 100, 200, 400),
) -> dict[str, Any]:
    methods = sorted(
        artifact["results"]["methods"],
        key=lambda row: int(row["local_full100_rank"]),
    )
    scores = np.asarray(
        [
            [float(item[SCORE_KEY]) for item in method["per_mlp"]]
            for method in methods
        ],
        dtype=np.float64,
    )
    n_methods, n_mlps = scores.shape
    method_rows = []
    for method, values in zip(methods, scores):
        standard_deviation = float(np.std(values, ddof=1))
        standard_error = standard_deviation / math.sqrt(n_mlps)
        method_rows.append(
            {
                "submission_id": int(method["submission_id"]),
                "method": method["method"],
                "local_full100_rank": int(method["local_full100_rank"]),
                "fresh_rank": int(method["fresh_flops_only_adjusted_rank"]),
                "n_mlps": n_mlps,
                "mean": float(np.mean(values)),
                "sample_variance": float(np.var(values, ddof=1)),
                "sample_standard_deviation": standard_deviation,
                "coefficient_of_variation": float(
                    standard_deviation / np.mean(values)
                ),
                "q05": float(np.quantile(values, 0.05)),
                "median": float(np.median(values)),
                "q95": float(np.quantile(values, 0.95)),
                "maximum": float(np.max(values)),
                "mean_standard_error": standard_error,
                "mean_two_sided_95_half_width": float(
                    t.ppf(0.975, n_mlps - 1) * standard_error
                ),
            }
        )

    pair_rows = []
    for left in range(n_methods):
        for right in range(left + 1, n_methods):
            differences = scores[left] - scores[right]
            difference_std = float(np.std(differences, ddof=1))
            pair_rows.append(
                {
                    "left_submission_id": int(methods[left]["submission_id"]),
                    "right_submission_id": int(methods[right]["submission_id"]),
                    "mean_left_minus_right": float(np.mean(differences)),
                    "difference_standard_deviation": difference_std,
                    "score_correlation": float(
                        np.corrcoef(scores[left], scores[right])[0, 1]
                    ),
                    "one_sided_95_threshold_at_observed_n": float(
                        t.ppf(0.95, n_mlps - 1)
                        * difference_std
                        / math.sqrt(n_mlps)
                    ),
                    "two_sided_95_half_width_at_observed_n": float(
                        t.ppf(0.975, n_mlps - 1)
                        * difference_std
                        / math.sqrt(n_mlps)
                    ),
                }
            )

    pair_stds = np.asarray(
        [row["difference_standard_deviation"] for row in pair_rows],
        dtype=np.float64,
    )
    threshold_by_size: dict[str, Any] = {}
    for target_n in target_sizes:
        one_sided = (
            t.ppf(0.95, target_n - 1) * pair_stds / math.sqrt(target_n)
        )
        two_sided = (
            t.ppf(0.975, target_n - 1) * pair_stds / math.sqrt(target_n)
        )
        bonferroni_15 = (
            t.ppf(1.0 - 0.05 / 15, target_n - 1)
            * pair_stds
            / math.sqrt(target_n)
        )
        bonferroni_100 = (
            t.ppf(1.0 - 0.05 / 100, target_n - 1)
            * pair_stds
            / math.sqrt(target_n)
        )
        threshold_by_size[str(target_n)] = {
            "one_sided_95": _quantiles(one_sided),
            "two_sided_95": _quantiles(two_sided),
            "bonferroni_one_sided_15_candidates": _quantiles(bonferroni_15),
            "bonferroni_one_sided_100_candidates": _quantiles(bonferroni_100),
        }

    winner_index = int(np.argmin(np.mean(scores, axis=1)))
    winner_comparisons = []
    for index, (method, values) in enumerate(zip(methods, scores)):
        if index == winner_index:
            continue
        candidate_minus_winner = values - scores[winner_index]
        difference_std = float(np.std(candidate_minus_winner, ddof=1))
        winner_comparisons.append(
            {
                "submission_id": int(method["submission_id"]),
                "mean_candidate_minus_winner": float(
                    np.mean(candidate_minus_winner)
                ),
                "difference_standard_deviation": difference_std,
                "one_sided_95_threshold": float(
                    t.ppf(0.95, n_mlps - 1)
                    * difference_std
                    / math.sqrt(n_mlps)
                ),
                "two_sided_95_half_width": float(
                    t.ppf(0.975, n_mlps - 1)
                    * difference_std
                    / math.sqrt(n_mlps)
                ),
            }
        )

    return {
        "score_definition": SCORE_KEY,
        "n_methods": n_methods,
        "n_mlps": n_mlps,
        "method_across_mlp_variance": method_rows,
        "pairwise_summary": {
            "n_pairs": len(pair_rows),
            "difference_standard_deviation": _quantiles(pair_stds),
            "score_correlation": _quantiles(
                np.asarray(
                    [row["score_correlation"] for row in pair_rows],
                    dtype=np.float64,
                )
            ),
            "threshold_by_suite_size": threshold_by_size,
        },
        "fresh_winner_submission_id": int(
            methods[winner_index]["submission_id"]
        ),
        "comparisons_against_fresh_winner": winner_comparisons,
        "pairwise_details": pair_rows,
        "interpretation": {
            "promotion_rule": (
                "For paired candidate-minus-incumbent scores, promote only "
                "when mean_delta + threshold < 0."
            ),
            "scope": (
                "These thresholds estimate random-MLP suite uncertainty. "
                "They do not estimate repeated-run randomness on one fixed MLP."
            ),
        },
    }


def format_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Estimator variance across fresh MLPs",
        "",
        "Scores below use corrected FLOPs-only adjusted score. Lower is better.",
        "",
        "| Method | Fresh rank | Mean ×1e7 | Variance ×1e14 | SD ×1e7 | CV | "
        "q05–q95 ×1e7 | 95% mean half-width ×1e7 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["method_across_mlp_variance"]:
        method = f'#{row["submission_id"]} {row["method"]}'.replace("|", r"\|")
        lines.append(
            f"| {method} | {row['fresh_rank']} "
            f"| {row['mean'] * 1e7:.4f} "
            f"| {row['sample_variance'] * 1e14:.4f} "
            f"| {row['sample_standard_deviation'] * 1e7:.4f} "
            f"| {row['coefficient_of_variation']:.3f} "
            f"| {row['q05'] * 1e7:.3f}–{row['q95'] * 1e7:.3f} "
            f"| ±{row['mean_two_sided_95_half_width'] * 1e7:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Paired candidate-versus-incumbent thresholds",
            "",
            "Each entry summarizes all method pairs. Values are score "
            "differences ×1e7.",
            "",
            "| MLPs | One-sided 95% median / q90 | Two-sided 95% median / q90 "
            "| 15-candidate Bonferroni median / q90 |",
            "|---:|---:|---:|---:|",
        ]
    )
    thresholds = result["pairwise_summary"]["threshold_by_suite_size"]
    for size in sorted(thresholds, key=int):
        row = thresholds[size]
        lines.append(
            f"| {size} "
            f"| {row['one_sided_95']['median'] * 1e7:.3f} / "
            f"{row['one_sided_95']['q90'] * 1e7:.3f} "
            f"| {row['two_sided_95']['median'] * 1e7:.3f} / "
            f"{row['two_sided_95']['q90'] * 1e7:.3f} "
            f"| {row['bonferroni_one_sided_15_candidates']['median'] * 1e7:.3f} / "
            f"{row['bonferroni_one_sided_15_candidates']['q90'] * 1e7:.3f} |"
        )
    lines.extend(
        [
            "",
            "Promotion rule: for paired `candidate - incumbent` scores, "
            "promote only if `mean_delta + threshold < 0`.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-markdown", type=Path)
    args = parser.parse_args()
    artifact = json.loads(args.artifact.read_text(encoding="utf-8"))
    result = analyze_thresholds(artifact)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {args.output_json}")
    if args.output_markdown is not None:
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output_markdown.write_text(
            format_markdown(result),
            encoding="utf-8",
        )
        print(f"wrote {args.output_markdown}")
    if args.output_json is None and args.output_markdown is None:
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
