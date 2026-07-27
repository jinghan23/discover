#!/usr/bin/env python3
"""Run the paired three-seed WhestBench gate through an existing server.

This is the pre-submission acceptance path. The blackbox protocol carries only
the submission source and the request state, so sidecar assets cannot be sent;
submissions whose weights live in a separate file (the learned residual, for
one) must be scored with ``evaluate_full100_direct.py`` instead.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ttt_discover import State
from ttt_discover.eval_runners.blackbox import BlackboxRunner


PROBLEM_TYPE = "arc_whestbench_2026"
SEED_OFFSET_METADATA_KEY = "whestbench_estimator_seed_offset"
DEFAULT_SEED_OFFSETS = (0, 1_000_003, 2_000_003)
DEFAULT_GERMAN_CURRENT = (
    ROOT
    / "repro_external/aicrowd_whestbench/public_reproductions"
    / "german_alfaro_current_41493b1.py"
)
_FAILURE_FLAGS = (
    "budget_exhausted",
    "time_exhausted",
    "residual_wall_time_exhausted",
    "combined_budget_exhausted",
)


def _sha256(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _run_failed(row: dict[str, Any]) -> bool:
    return bool(row.get("error_code")) or any(
        bool(row.get(flag)) for flag in _FAILURE_FLAGS
    )


def _validate_rows(
    rows: Any,
    *,
    expected_n_mlps: int,
    expected_seed_offset: int,
) -> list[dict[str, Any]]:
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("blackbox server returned no per-MLP evaluation results")
    if len(rows) != expected_n_mlps:
        raise RuntimeError(
            f"blackbox server returned {len(rows)} MLPs; expected {expected_n_mlps}"
        )
    for row in rows:
        if int(row.get("estimator_seed_offset", -1)) != expected_seed_offset:
            raise RuntimeError(
                "blackbox server did not honor estimator_seed_offset; "
                "ensure it allows request state and uses WhestBenchRewardEvaluator"
            )
    return rows


def _uniform_row_value(
    rows: list[dict[str, Any]],
    key: str,
    convert: type[int] | type[float],
) -> int | float | None:
    values = {convert(row[key]) for row in rows if row.get(key) is not None}
    return next(iter(values)) if len(values) == 1 else None


def _summarize(rows: list[dict[str, Any]], raw_score: float) -> dict[str, Any]:
    final_mse = [float(row["final_layer_mse"]) for row in rows]
    all_layers_mse = [float(row["all_layers_mse"]) for row in rows]
    adjusted = [float(row["adjusted_final_layer_score"]) for row in rows]
    effective_compute = [float(row.get("effective_compute", 0.0)) for row in rows]
    flop_budget = _uniform_row_value(rows, "flop_budget", int)
    score_multipliers = [
        score / mse if mse > 0.0 else 1.0
        for score, mse in zip(adjusted, final_mse)
    ]
    return {
        "adjusted_final_layer_score": raw_score,
        "all_layers_mse": sum(all_layers_mse) / len(all_layers_mse),
        "best_mlp_adjusted_final_layer_score": min(adjusted),
        "final_layer_mse": sum(final_mse) / len(final_mse),
        "max_effective_compute": max(effective_compute),
        "mean_compute_utilization": (
            sum(effective_compute) / len(effective_compute) / flop_budget
            if flop_budget
            else None
        ),
        "mean_effective_compute": sum(effective_compute) / len(effective_compute),
        "mean_score_multiplier": sum(score_multipliers) / len(score_multipliers),
        "n_failed_mlps": sum(_run_failed(row) for row in rows),
        "worst_mlp_adjusted_final_layer_score": max(adjusted),
    }


def _evaluate_seed(
    runner: BlackboxRunner,
    source: str,
    *,
    seed_offset: int,
    expected_n_mlps: int,
) -> dict[str, Any]:
    state = State(
        timestep=-1,
        construction=[],
        code="",
        value=0.0,
        metadata={SEED_OFFSET_METADATA_KEY: seed_offset},
    )
    env = SimpleNamespace(problem_type=PROBLEM_TYPE, state=state)
    started = time.perf_counter()
    result = runner.evaluate(env, source)
    elapsed = time.perf_counter() - started
    rows = _validate_rows(
        result.result_construction,
        expected_n_mlps=expected_n_mlps,
        expected_seed_offset=seed_offset,
    )
    replacement_count = _uniform_row_value(
        rows,
        "estimator_seed_replacement_count",
        int,
    )
    if replacement_count is None:
        raise RuntimeError("blackbox response has inconsistent replacement counts")
    return {
        "elapsed_s": elapsed,
        "estimator_seed_offset": seed_offset,
        "estimator_seed_replacement_count": replacement_count,
        "per_mlp": rows,
        "summary": _summarize(rows, float(result.raw_score)),
    }


def _mlp_key(row: dict[str, Any], position: int = -1) -> tuple[Any, Any]:
    return row.get("mlp_index", position), row.get("mlp_name")


def _aggregate_seed_runs(seed_runs: list[dict[str, Any]]) -> dict[str, Any]:
    if len(seed_runs) != 3:
        raise ValueError(f"expected exactly three seed runs, got {len(seed_runs)}")

    first_rows = seed_runs[0]["per_mlp"]
    ordered_keys = [_mlp_key(row, i) for i, row in enumerate(first_rows)]
    rows_by_seed = []
    for run in seed_runs:
        indexed = {
            _mlp_key(row, i): row for i, row in enumerate(run["per_mlp"])
        }
        if set(indexed) != set(ordered_keys):
            raise RuntimeError("three seed runs returned different MLP sets")
        rows_by_seed.append(indexed)

    seed_offsets = [int(run["estimator_seed_offset"]) for run in seed_runs]
    aggregate_rows = []
    mean_keys = (
        "adjusted_final_layer_score",
        "final_layer_mse",
        "all_layers_mse",
        "flops_used",
        "effective_compute",
    )
    for key in ordered_keys:
        seed_rows = [indexed[key] for indexed in rows_by_seed]
        aggregate_row: dict[str, Any] = {
            "mlp_index": key[0],
            "mlp_name": key[1],
            "estimator_seed_offsets": seed_offsets,
            "n_failed_seed_runs": sum(_run_failed(row) for row in seed_rows),
            "seed_results": seed_rows,
        }
        for metric in mean_keys:
            values = [
                float(row[metric])
                for row in seed_rows
                if row.get(metric) is not None
            ]
            if len(values) == len(seed_rows):
                aggregate_row[metric] = sum(values) / len(values)
        for flag in _FAILURE_FLAGS:
            aggregate_row[flag] = any(bool(row.get(flag)) for row in seed_rows)
        aggregate_row["error_code"] = next(
            (row.get("error_code") for row in seed_rows if row.get("error_code")),
            None,
        )
        aggregate_row["flop_budget"] = _uniform_row_value(
            seed_rows, "flop_budget", int
        )
        aggregate_row["lambda_flops_per_second"] = _uniform_row_value(
            seed_rows, "lambda_flops_per_second", float
        )
        aggregate_rows.append(aggregate_row)

    seed_scores = [
        float(run["summary"]["adjusted_final_layer_score"]) for run in seed_runs
    ]
    mean_score = sum(seed_scores) / len(seed_scores)
    summary = _summarize(aggregate_rows, mean_score)
    summary.update(
        {
            "adjusted_final_layer_score_std": (
                sum((score - mean_score) ** 2 for score in seed_scores)
                / len(seed_scores)
            )
            ** 0.5,
            "estimator_seed_offsets": seed_offsets,
            "n_failed_mlp_seed_runs": sum(
                int(run["summary"]["n_failed_mlps"]) for run in seed_runs
            ),
            "n_mlps_with_any_failed_seed": sum(
                int(row["n_failed_seed_runs"] > 0) for row in aggregate_rows
            ),
            "n_seed_runs": 3,
            "seed_adjusted_final_layer_scores": seed_scores,
            "mean_score_multiplier": sum(
                float(run["summary"]["mean_score_multiplier"]) for run in seed_runs
            )
            / 3,
        }
    )
    return {**summary, "per_mlp": aggregate_rows, "seed_runs": seed_runs}


def _paired_mlp_differences(
    candidate_results: dict[str, Any],
    baseline_results: dict[str, Any],
) -> list[dict[str, Any]]:
    baseline_rows = {
        _mlp_key(row): row for row in baseline_results["per_mlp"]
    }
    candidate_rows = {
        _mlp_key(row): row for row in candidate_results["per_mlp"]
    }
    if set(candidate_rows) != set(baseline_rows):
        raise RuntimeError("candidate and German current returned different MLP sets")

    paired_rows = []
    for mlp_key, candidate_row in candidate_rows.items():
        baseline_row = baseline_rows[mlp_key]
        candidate_seed_results = candidate_row["seed_results"]
        baseline_seed_results = baseline_row["seed_results"]
        candidate_offsets = [
            int(row["estimator_seed_offset"]) for row in candidate_seed_results
        ]
        baseline_offsets = [
            int(row["estimator_seed_offset"]) for row in baseline_seed_results
        ]
        if candidate_offsets != baseline_offsets:
            raise RuntimeError("candidate and German current seed offsets do not match")

        seed_pairs = []
        for offset, candidate_seed, baseline_seed in zip(
            candidate_offsets,
            candidate_seed_results,
            baseline_seed_results,
        ):
            candidate_score = float(candidate_seed["adjusted_final_layer_score"])
            baseline_score = float(baseline_seed["adjusted_final_layer_score"])
            seed_pairs.append(
                {
                    "baseline_adjusted_final_layer_score": baseline_score,
                    "candidate_adjusted_final_layer_score": candidate_score,
                    "difference": candidate_score - baseline_score,
                    "estimator_seed_offset": offset,
                }
            )
        paired_rows.append(
            {
                "mean_paired_adjusted_difference": sum(
                    pair["difference"] for pair in seed_pairs
                )
                / len(seed_pairs),
                "mlp_index": mlp_key[0],
                "mlp_name": mlp_key[1],
                "seed_pairs": seed_pairs,
            }
        )
    return paired_rows


def _percentile(sorted_values: list[float], quantile: float) -> float:
    if not sorted_values:
        raise ValueError("cannot compute a percentile of an empty sequence")
    position = (len(sorted_values) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def _clustered_bootstrap_ci(
    paired_rows: list[dict[str, Any]],
    *,
    samples: int,
    seed: int,
) -> tuple[float, float, float]:
    differences = [
        float(row["mean_paired_adjusted_difference"]) for row in paired_rows
    ]
    if not differences:
        raise ValueError("cannot bootstrap an empty paired result")
    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")

    observed = sum(differences) / len(differences)
    rng = random.Random(seed)
    bootstrap_means = sorted(
        sum(differences[rng.randrange(len(differences))] for _ in differences)
        / len(differences)
        for _ in range(samples)
    )
    return (
        observed,
        _percentile(bootstrap_means, 0.025),
        _percentile(bootstrap_means, 0.975),
    )


def _result_summary(results: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in results.items()
        if key not in {"per_mlp", "seed_runs"}
    }


def _whole_suite_setup_timeout_before_compute(
    run: dict[str, Any],
    *,
    expected_n_mlps: int,
) -> bool:
    """Return true only for a complete, zero-compute worker-start failure."""

    rows = run.get("per_mlp")
    if not isinstance(rows, list) or len(rows) != expected_n_mlps:
        return False
    for row in rows:
        if row.get("error_code") != "SETUP_TIMEOUT":
            return False
        if "flops_used" not in row or float(row["flops_used"]) != 0.0:
            return False
        if "effective_compute" not in row or float(row["effective_compute"]) != 0.0:
            return False
        if any(bool(row.get(flag)) for flag in _FAILURE_FLAGS):
            return False
    return True


def _attempt_failure_summary(run: dict[str, Any]) -> dict[str, Any]:
    rows = run.get("per_mlp", [])
    error_code_counts: dict[str, int] = {}
    for row in rows:
        code = str(row.get("error_code") or "NONE")
        error_code_counts[code] = error_code_counts.get(code, 0) + 1
    return {
        "error_code_counts": error_code_counts,
        "exhaustion_flag_counts": {
            flag: sum(bool(row.get(flag)) for row in rows)
            for flag in _FAILURE_FLAGS
        },
        "n_nonzero_effective_compute": sum(
            float(row.get("effective_compute", 0.0)) != 0.0 for row in rows
        ),
        "n_nonzero_flops": sum(
            float(row.get("flops_used", 0.0)) != 0.0 for row in rows
        ),
        "n_rows": len(rows),
    }


def _evaluate_schedule(
    runner: BlackboxRunner,
    candidate_source: str,
    baseline_source: str,
    *,
    seed_offsets: tuple[int, ...],
    expected_n_mlps: int,
    interleave_baseline: bool,
    inter_run_cooldown_s: float,
    max_infra_retries: int,
    infra_retry_cooldown_s: float,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Evaluate a symmetric, predeclared candidate/baseline request schedule."""

    if interleave_baseline:
        schedule = [
            (label, source, seed_offset)
            for seed_offset in seed_offsets
            for label, source in (
                ("candidate", candidate_source),
                ("baseline", baseline_source),
            )
        ]
    else:
        schedule = [
            ("candidate", candidate_source, seed_offset)
            for seed_offset in seed_offsets
        ] + [
            ("baseline", baseline_source, seed_offset)
            for seed_offset in seed_offsets
        ]

    candidate_runs: list[dict[str, Any]] = []
    baseline_runs: list[dict[str, Any]] = []
    request_order: list[dict[str, Any]] = []
    request_attempts: list[dict[str, Any]] = []
    for label, source, seed_offset in schedule:
        if inter_run_cooldown_s > 0.0:
            time.sleep(inter_run_cooldown_s)
        run = None
        for attempt_number in range(1, max_infra_retries + 2):
            run = _evaluate_seed(
                runner,
                source,
                seed_offset=seed_offset,
                expected_n_mlps=expected_n_mlps,
            )
            whole_suite_timeout = _whole_suite_setup_timeout_before_compute(
                run,
                expected_n_mlps=expected_n_mlps,
            )
            attempt_record = {
                "attempt_number": attempt_number,
                "estimator": label,
                "estimator_seed_offset": seed_offset,
                "failure_summary": _attempt_failure_summary(run),
                "source_sha256": _sha256(source),
                "summary": _result_summary(run.get("summary", {})),
                "used_for_aggregate": False,
                "whole_suite_setup_timeout_before_compute": whole_suite_timeout,
            }
            request_attempts.append(attempt_record)
            if not whole_suite_timeout:
                attempt_record["used_for_aggregate"] = True
                break
            if attempt_number > max_infra_retries:
                attempt_record["used_for_aggregate"] = True
                break
            if infra_retry_cooldown_s > 0.0:
                time.sleep(infra_retry_cooldown_s)
        assert run is not None
        (candidate_runs if label == "candidate" else baseline_runs).append(run)
        request_order.append(
            {"estimator": label, "estimator_seed_offset": seed_offset}
        )
    return candidate_runs, baseline_runs, request_order, request_attempts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("estimator", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=DEFAULT_GERMAN_CURRENT,
        help="German current estimator used by the three-seed gate.",
    )
    parser.add_argument("--n-mlps", type=int, default=100)
    parser.add_argument(
        "--estimator-seed-offsets",
        type=int,
        nargs=3,
        metavar=("SEED_1", "SEED_2", "SEED_3"),
        default=DEFAULT_SEED_OFFSETS,
    )
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_721)
    parser.add_argument(
        "--interleave-baseline",
        action="store_true",
        help="Evaluate candidate/baseline pairs per seed instead of CCCBBB order.",
    )
    parser.add_argument(
        "--inter-run-cooldown-s",
        type=float,
        default=0.0,
        help=(
            "Symmetric client-side cooldown before every suite request. "
            "This does not change the evaluator setup timeout or score."
        ),
    )
    parser.add_argument(
        "--max-infra-retries",
        type=int,
        default=0,
        help=(
            "Maximum retries beyond the initial attempt, only when every MLP "
            "returns SETUP_TIMEOUT before using any compute."
        ),
    )
    parser.add_argument(
        "--infra-retry-cooldown-s",
        type=float,
        default=30.0,
        help="Client-side cooldown before a narrowly eligible setup-timeout retry.",
    )
    parser.add_argument("--socket", dest="socket_path", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--timeout-s", type=float, default=4000.0)
    args = parser.parse_args()

    estimator = args.estimator.expanduser().resolve()
    baseline = args.baseline.expanduser().resolve()
    if not estimator.is_file():
        parser.error(f"missing estimator: {estimator}")
    if not baseline.is_file():
        parser.error(f"missing German current baseline: {baseline}")
    if args.n_mlps <= 0:
        parser.error("--n-mlps must be positive")
    if args.bootstrap_samples <= 0:
        parser.error("--bootstrap-samples must be positive")
    if args.inter_run_cooldown_s < 0.0:
        parser.error("--inter-run-cooldown-s must be non-negative")
    if args.max_infra_retries < 0:
        parser.error("--max-infra-retries must be non-negative")
    if args.infra_retry_cooldown_s < 0.0:
        parser.error("--infra-retry-cooldown-s must be non-negative")
    seed_offsets = tuple(args.estimator_seed_offsets)
    if len(set(seed_offsets)) != 3:
        parser.error("--estimator-seed-offsets must contain three distinct values")

    source = estimator.read_text(encoding="utf-8")
    baseline_source = baseline.read_text(encoding="utf-8")
    runner = BlackboxRunner(
        socket_path=args.socket_path,
        host=args.host,
        port=args.port,
        timeout_s=args.timeout_s,
    )
    started = time.perf_counter()
    (
        candidate_seed_runs,
        baseline_seed_runs,
        request_order,
        request_attempts,
    ) = _evaluate_schedule(
        runner,
        source,
        baseline_source,
        seed_offsets=seed_offsets,
        expected_n_mlps=args.n_mlps,
        interleave_baseline=args.interleave_baseline,
        inter_run_cooldown_s=args.inter_run_cooldown_s,
        max_infra_retries=args.max_infra_retries,
        infra_retry_cooldown_s=args.infra_retry_cooldown_s,
    )
    candidate_results = _aggregate_seed_runs(candidate_seed_runs)
    baseline_results = _aggregate_seed_runs(baseline_seed_runs)
    paired_rows = _paired_mlp_differences(candidate_results, baseline_results)
    observed, ci_low, ci_high = _clustered_bootstrap_ci(
        paired_rows,
        samples=args.bootstrap_samples,
        seed=args.bootstrap_seed,
    )
    elapsed = time.perf_counter() - started

    candidate_scores = [
        float(run["summary"]["adjusted_final_layer_score"])
        for run in candidate_seed_runs
    ]
    baseline_scores = [
        float(run["summary"]["adjusted_final_layer_score"])
        for run in baseline_seed_runs
    ]
    paired_seed_differences = [
        candidate_score - baseline_score
        for candidate_score, baseline_score in zip(candidate_scores, baseline_scores)
    ]
    candidate_failures = sum(
        int(run["summary"]["n_failed_mlps"]) for run in candidate_seed_runs
    )
    baseline_failures = sum(
        int(run["summary"]["n_failed_mlps"]) for run in baseline_seed_runs
    )
    candidate_mean = float(candidate_results["adjusted_final_layer_score"])
    baseline_mean = float(baseline_results["adjusted_final_layer_score"])
    all_three_beat_baseline = all(
        difference < 0.0 for difference in paired_seed_differences
    )
    infra_retry_exhausted = any(
        bool(attempt["used_for_aggregate"])
        and bool(attempt["whole_suite_setup_timeout_before_compute"])
        for attempt in request_attempts
    )
    criteria = {
        "all_three_seed_aggregates_beat_baseline": all_three_beat_baseline,
        "bootstrap_ci_upper_below_zero": ci_high < 0.0,
        "candidate_zero_failures": candidate_failures == 0,
        "german_current_zero_failures": baseline_failures == 0,
        "infrastructure_retries_not_exhausted": not infra_retry_exhausted,
        "three_seed_mean_beats_german_current": candidate_mean < baseline_mean,
    }
    relative_improvement = (
        100.0 * (baseline_mean - candidate_mean) / baseline_mean
        if baseline_mean != 0.0
        else None
    )

    report = {
        "baseline": str(baseline),
        "baseline_code_sha256": _sha256(baseline_source),
        "baseline_mean_adjusted": baseline_mean,
        "baseline_results": baseline_results,
        "baseline_runs": [run["summary"] for run in baseline_seed_runs],
        "base_code_sha256": _sha256(source),
        "bootstrap": {
            "cluster": "mlp_index",
            "mean_paired_adjusted_difference": observed,
            "n_resamples": args.bootstrap_samples,
            "percentile_95_ci": [ci_low, ci_high],
            "random_seed": args.bootstrap_seed,
        },
        "candidate": str(estimator),
        "candidate_code_sha256": _sha256(source),
        "candidate_mean_adjusted": candidate_mean,
        "candidate_runs": [run["summary"] for run in candidate_seed_runs],
        "code_sha256": _sha256(source),
        "comparison": {
            "paired_mlp_results": paired_rows,
            "paired_seed_adjusted_differences": paired_seed_differences,
            "relative_mean_improvement_pct": relative_improvement,
        },
        "dataset": {
            "n_mlps": len(candidate_results["per_mlp"]),
            "owner": "blackbox_server",
        },
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation": {
            "baseline_seed_replacement_counts": [
                int(run["estimator_seed_replacement_count"])
                for run in baseline_seed_runs
            ],
            "candidate_seed_replacement_counts": [
                int(run["estimator_seed_replacement_count"])
                for run in candidate_seed_runs
            ],
            "elapsed_s": elapsed,
            "estimator_seed_offsets": list(seed_offsets),
            "flop_budget": _uniform_row_value(
                candidate_results["per_mlp"], "flop_budget", int
            ),
            "infra_retry_cooldown_s": args.infra_retry_cooldown_s,
            "inter_run_cooldown_s": args.inter_run_cooldown_s,
            "interleave_baseline": args.interleave_baseline,
            "lambda_flops_per_second": _uniform_row_value(
                candidate_results["per_mlp"],
                "lambda_flops_per_second",
                float,
            ),
            "max_infra_retries": args.max_infra_retries,
            "n_seed_runs_per_estimator": 3,
            "n_nominal_suite_evaluations": 6,
            "n_request_attempts": len(request_attempts),
            "n_suite_evaluations": len(request_attempts),
            "request_attempts": request_attempts,
            "request_order": request_order,
            "runner": "existing_blackbox_server",
        },
        "gate": {
            **criteria,
            "all_three_seed_aggregates_beat_german_current": (
                all_three_beat_baseline
            ),
            "baseline_failures": baseline_failures,
            "candidate_failures": candidate_failures,
            "gate_valid": not infra_retry_exhausted,
            "infrastructure_retries_exhausted": infra_retry_exhausted,
            "official_submission_eligible": all(criteria.values()),
            "paired_seed_adjusted_differences": paired_seed_differences,
            "relative_mean_improvement_pct": relative_improvement,
            "zero_failures": candidate_failures == 0 and baseline_failures == 0,
        },
        "label": estimator.stem,
        "n_mlps": len(candidate_results["per_mlp"]),
        "results": candidate_results,
        "seed_offsets": list(seed_offsets),
        "source": str(estimator),
        "summary": _result_summary(candidate_results),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "bootstrap": report["bootstrap"],
                "gate": report["gate"],
                "summary": report["summary"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
