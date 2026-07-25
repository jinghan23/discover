#!/usr/bin/env python3
"""Merge disjoint fresh-MLP WhestBench evaluations and recompute all ranks."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np

from repro.aicrowd_whestbench.evaluate_fresh_mlp_suite import (
    FLOP_BUDGET,
    _pair_inversion_fraction,
    _rank,
    _spearman,
    bootstrap_fresh_methods,
)


def _weighted_mean(parts: list[dict[str, Any]], key: str) -> float:
    weights = np.asarray([part["suite"]["n_mlps"] for part in parts], dtype=float)
    values = np.asarray(
        [part["results"]["truth_noise"][key] for part in parts],
        dtype=float,
    )
    return float(np.average(values, weights=weights))


def _validate_parts(parts: list[dict[str, Any]]) -> None:
    if len(parts) < 2:
        raise ValueError("at least two evaluation parts are required")
    reference = parts[0]
    suite_keys = (
        "width",
        "depth",
        "suite_seed",
        "truth_samples_per_repeat",
        "truth_repeats",
        "truth_method",
        "total_truth_samples_per_mlp",
    )
    method_signature = [
        (
            int(row["submission_id"]),
            str(row["code_sha256"]),
            int(row["local_full100_rank"]),
        )
        for row in reference["results"]["methods"]
    ]
    roots: set[int] = set()
    indices: set[int] = set()
    for part_number, part in enumerate(parts):
        if part["protocol"] != reference["protocol"]:
            raise ValueError(f"part {part_number} has a different protocol")
        for key in suite_keys:
            if part["suite"][key] != reference["suite"][key]:
                raise ValueError(f"part {part_number} differs at suite.{key}")
        signature = [
            (
                int(row["submission_id"]),
                str(row["code_sha256"]),
                int(row["local_full100_rank"]),
            )
            for row in part["results"]["methods"]
        ]
        if signature != method_signature:
            raise ValueError(f"part {part_number} has different estimators")
        offset = int(part["suite"].get("mlp_offset", 0))
        part_indices = set(range(offset, offset + int(part["suite"]["n_mlps"])))
        if roots.intersection(part["suite"]["root_seeds"]):
            raise ValueError(f"part {part_number} repeats an MLP root seed")
        if indices.intersection(part_indices):
            raise ValueError(f"part {part_number} overlaps an MLP index")
        roots.update(int(seed) for seed in part["suite"]["root_seeds"])
        indices.update(part_indices)


def _merge_method_rows(parts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for method_index, first_method in enumerate(parts[0]["results"]["methods"]):
        row = copy.deepcopy(first_method)
        per_mlp: list[dict[str, Any]] = []
        elapsed = 0.0
        for part in parts:
            method = part["results"]["methods"][method_index]
            offset = int(part["suite"].get("mlp_offset", 0))
            elapsed += float(method["elapsed_s"])
            for item in method["per_mlp"]:
                item = copy.deepcopy(item)
                item["suite_mlp_index"] = int(
                    item.get("suite_mlp_index", offset + int(item["mlp_index"]))
                )
                per_mlp.append(item)
        per_mlp.sort(key=lambda item: int(item["suite_mlp_index"]))
        for index, item in enumerate(per_mlp):
            item["mlp_index"] = index

        raw_mse = np.asarray([item["raw_final_mse"] for item in per_mlp], dtype=float)
        corrected_mse = np.asarray(
            [item["corrected_final_mse"] for item in per_mlp],
            dtype=float,
        )
        measured_multiplier = np.asarray(
            [item["measured_score_multiplier"] for item in per_mlp],
            dtype=float,
        )
        measured_adjusted = np.asarray(
            [item["corrected_measured_adjusted_score"] for item in per_mlp],
            dtype=float,
        )
        flops_adjusted = np.asarray(
            [item["corrected_flops_only_adjusted_score"] for item in per_mlp],
            dtype=float,
        )
        effective_compute = np.asarray(
            [item["effective_compute"] for item in per_mlp],
            dtype=float,
        )
        row.update(
            {
                "fresh_raw_final_mse": float(np.mean(raw_mse)),
                "fresh_corrected_final_mse": float(np.mean(corrected_mse)),
                "fresh_official_measured_adjusted_score": float(
                    np.mean(raw_mse * measured_multiplier)
                ),
                "fresh_corrected_measured_adjusted_score": float(
                    np.mean(measured_adjusted)
                ),
                "fresh_corrected_flops_only_adjusted_score": float(
                    np.mean(flops_adjusted)
                ),
                "mean_effective_compute": float(np.mean(effective_compute)),
                "mean_compute_utilization": float(
                    np.mean(effective_compute) / FLOP_BUDGET
                ),
                "n_failed_mlps": int(sum(bool(item["failed"]) for item in per_mlp)),
                "elapsed_s": elapsed,
                "per_mlp": per_mlp,
            }
        )
        merged.append(row)

    local = np.asarray(
        [row["local_full100_adjusted_score"] for row in merged],
        dtype=float,
    )
    hosted = np.asarray(
        [row["hosted_public50_adjusted_score"] for row in merged],
        dtype=float,
    )
    fresh_mse = np.asarray(
        [row["fresh_corrected_final_mse"] for row in merged],
        dtype=float,
    )
    fresh_adjusted = np.asarray(
        [row["fresh_corrected_flops_only_adjusted_score"] for row in merged],
        dtype=float,
    )
    rank_sets = (
        ("hosted_rank_among_selected", hosted),
        ("fresh_corrected_mse_rank", fresh_mse),
        ("fresh_flops_only_adjusted_rank", fresh_adjusted),
    )
    for key, values in rank_sets:
        ranks = _rank(values)
        for index, row in enumerate(merged):
            row[key] = int(ranks[index] + 1)
    return merged


def merge_parts(parts: list[dict[str, Any]]) -> dict[str, Any]:
    _validate_parts(parts)
    methods = _merge_method_rows(parts)
    local = np.asarray(
        [row["local_full100_adjusted_score"] for row in methods],
        dtype=float,
    )
    hosted = np.asarray(
        [row["hosted_public50_adjusted_score"] for row in methods],
        dtype=float,
    )
    fresh_mse = np.asarray(
        [row["fresh_corrected_final_mse"] for row in methods],
        dtype=float,
    )
    fresh = np.asarray(
        [row["fresh_corrected_flops_only_adjusted_score"] for row in methods],
        dtype=float,
    )
    comparisons = {
        "fresh_flops_only_adjusted_vs_local_full100": {
            "spearman": _spearman(fresh, local),
            "pairwise_inversion_fraction": _pair_inversion_fraction(fresh, local),
            "local_top1_fresh_rank": int(_rank(fresh)[np.argmin(local)] + 1),
        },
        "fresh_corrected_mse_vs_local_full100": {
            "spearman": _spearman(fresh_mse, local),
            "pairwise_inversion_fraction": _pair_inversion_fraction(
                fresh_mse, local
            ),
            "local_top1_fresh_rank": int(_rank(fresh_mse)[np.argmin(local)] + 1),
        },
        "fresh_flops_only_adjusted_vs_hosted_public50": {
            "spearman": _spearman(fresh, hosted),
            "pairwise_inversion_fraction": _pair_inversion_fraction(fresh, hosted),
        },
    }
    if len(parts) == 2:
        split_scores = []
        for part in parts:
            split_rows = sorted(
                part["results"]["methods"],
                key=lambda row: int(row["local_full100_rank"]),
            )
            split_scores.append(
                np.asarray(
                    [
                        row["fresh_corrected_flops_only_adjusted_score"]
                        for row in split_rows
                    ],
                    dtype=float,
                )
            )
        comparisons["fresh_part_0_vs_part_1"] = {
            "suite_sizes": [
                int(parts[0]["suite"]["n_mlps"]),
                int(parts[1]["suite"]["n_mlps"]),
            ],
            "spearman": _spearman(split_scores[0], split_scores[1]),
            "pairwise_inversion_fraction": _pair_inversion_fraction(
                split_scores[0], split_scores[1]
            ),
        }
    suite = copy.deepcopy(parts[0]["suite"])
    suite["n_mlps"] = sum(int(part["suite"]["n_mlps"]) for part in parts)
    suite["mlp_offset"] = min(
        int(part["suite"].get("mlp_offset", 0)) for part in parts
    )
    for key in ("root_seeds", "estimator_seeds", "avg_variances"):
        suite[key] = [
            value
            for part in sorted(
                parts,
                key=lambda item: int(item["suite"].get("mlp_offset", 0)),
            )
            for value in part["suite"][key]
        ]
    truth_noise = {
        "selected_noise_floor_mean": _weighted_mean(
            parts, "selected_noise_floor_mean"
        ),
        "selected_noise_floor_max": max(
            float(part["results"]["truth_noise"]["selected_noise_floor_max"])
            for part in parts
        ),
        "mc_analytic_noise_floor_mean": _weighted_mean(
            parts, "mc_analytic_noise_floor_mean"
        ),
        "repeat_difference_noise_floor_mean": _weighted_mean(
            parts, "repeat_difference_noise_floor_mean"
        ),
    }
    evaluation = copy.deepcopy(parts[0]["evaluation"])
    evaluation["elapsed_s_parts_sum"] = sum(
        float(part["evaluation"]["elapsed_s"]) for part in parts
    )
    evaluation["source_parts"] = len(parts)
    return {
        "protocol": copy.deepcopy(parts[0]["protocol"]),
        "suite": suite,
        "evaluation": evaluation,
        "results": {
            "truth_noise": truth_noise,
            "comparisons": comparisons,
            "fresh_suite_bootstrap": bootstrap_fresh_methods(methods),
            "methods": methods,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parts", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    loaded = [
        json.loads(path.read_text(encoding="utf-8")) for path in args.parts
    ]
    merged = merge_parts(loaded)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(merged, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(merged["results"]["comparisons"], indent=2, sort_keys=True))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
