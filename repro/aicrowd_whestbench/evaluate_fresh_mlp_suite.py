#!/usr/bin/env python3
"""Run the locally best WhestBench estimators on newly generated random MLPs.

Unlike ``analyze_mlp_variance.py``'s report bootstrap, this program really runs
each estimator.  It independently generates protocol-3.0 MLPs, estimates their
ground truth with repeated high-sample Monte Carlo, and invokes the official
subprocess runner on every estimator/MLP pair.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np

from repro.aicrowd_whestbench.analyze_mlp_variance import (
    DEFAULT_REGISTRY,
    REPO_ROOT,
    generate_official_weights,
)


DEFAULT_SELECTION = (
    REPO_ROOT
    / "repro_external/aicrowd_whestbench/analysis/"
    "mlp_variance_rank_stability_20260723.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "repro_external/aicrowd_whestbench/analysis/"
    "fresh_mlp_estimator_eval_20260723.json"
)
FLOP_BUDGET = 272_000_000_000
LAMBDA_FLOPS_PER_SECOND = 100_000_000_000.0


@dataclass(frozen=True)
class EstimatorSource:
    full100_rank: int
    method: str
    submission_id: int
    report_path: Path
    code_sha256: str
    code: str
    source_description: str
    local_full100_score: float
    hosted_public50_score: float | None


def _rank(values: np.ndarray) -> np.ndarray:
    ranks = np.empty(values.size, dtype=np.int64)
    ranks[np.argsort(values)] = np.arange(values.size)
    return ranks


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    left_ranks = _rank(left)
    right_ranks = _rank(right)
    n = left.size
    delta = left_ranks - right_ranks
    return float(
        1.0 - 6.0 * np.sum(delta * delta) / (n * (n * n - 1))
    )


def _pair_inversion_fraction(left: np.ndarray, right: np.ndarray) -> float:
    pair_i, pair_j = np.triu_indices(left.size, k=1)
    left_sign = np.sign(left[pair_i] - left[pair_j])
    right_sign = np.sign(right[pair_i] - right[pair_j])
    return float(np.mean(left_sign != right_sign))


def bootstrap_fresh_methods(
    method_rows: list[dict[str, Any]],
    *,
    repeats: int = 20_000,
    seed: int = 20260724,
) -> dict[str, Any]:
    scores = np.asarray(
        [
            [
                float(row["corrected_flops_only_adjusted_score"])
                for row in method["per_mlp"]
            ]
            for method in method_rows
        ],
        dtype=np.float64,
    )
    n_methods, n_mlps = scores.shape
    means = np.mean(scores, axis=1)
    fresh_order = np.argsort(means)
    local_top = next(
        index
        for index, method in enumerate(method_rows)
        if int(method["local_full100_rank"]) == 1
    )
    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty((repeats, n_methods), dtype=np.float64)
    top_counts = np.zeros(n_methods, dtype=np.int64)
    local_top_ranks = np.empty(repeats, dtype=np.int64)
    for repeat in range(repeats):
        sampled = rng.integers(0, n_mlps, size=n_mlps)
        values = np.mean(scores[:, sampled], axis=1)
        bootstrap_means[repeat] = values
        order = np.argsort(values)
        top_counts[order[0]] += 1
        ranks = np.empty(n_methods, dtype=np.int64)
        ranks[order] = np.arange(1, n_methods + 1)
        local_top_ranks[repeat] = ranks[local_top]

    method_intervals = []
    for index, method in enumerate(method_rows):
        low, median, high = np.quantile(
            bootstrap_means[:, index],
            (0.025, 0.5, 0.975),
        )
        method_intervals.append(
            {
                "local_full100_rank": int(method["local_full100_rank"]),
                "fresh_rank": int(_rank(means)[index] + 1),
                "fresh_mean": float(means[index]),
                "fresh_standard_error": float(
                    np.std(scores[index], ddof=1) / math.sqrt(n_mlps)
                ),
                "bootstrap_95pct_interval": [float(low), float(high)],
                "bootstrap_median": float(median),
                "bootstrap_top1_probability": float(top_counts[index] / repeats),
            }
        )

    fresh_top = int(fresh_order[0])
    return {
        "repeats": repeats,
        "seed": seed,
        "suite_size": n_mlps,
        "fresh_top_local_full100_rank": int(
            method_rows[fresh_top]["local_full100_rank"]
        ),
        "fresh_top_beats_local_top_probability": float(
            np.mean(bootstrap_means[:, fresh_top] < bootstrap_means[:, local_top])
        ),
        "local_top_fresh_rank_quantiles": {
            "q05": float(np.quantile(local_top_ranks, 0.05)),
            "q50": float(np.quantile(local_top_ranks, 0.50)),
            "q95": float(np.quantile(local_top_ranks, 0.95)),
        },
        "local_top_probability_fresh_top3": float(
            np.mean(local_top_ranks <= 3)
        ),
        "local_top_probability_fresh_bottom_half": float(
            np.mean(local_top_ranks >= math.ceil(n_methods / 2))
        ),
        "methods": method_intervals,
    }


def _resolve_repo_path(raw: str | Path) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else REPO_ROOT / path


def _read_estimator_from_artifact(artifact_path: Path) -> str:
    with tarfile.open(artifact_path) as archive:
        member = archive.extractfile("estimator.py")
        if member is None:
            raise ValueError(f"{artifact_path} has no estimator.py")
        return member.read().decode("utf-8")


def load_selected_estimators(
    *,
    selection_path: Path,
    registry_path: Path,
    n_estimators: int,
) -> list[EstimatorSource]:
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selected_rows = selection["rank_bootstrap"]["methods"][:n_estimators]
    registry = json.loads(registry_path.read_text(encoding="utf-8"))["entries"]
    registry_by_id = {int(row["submission_id"]): row for row in registry}
    estimators: list[EstimatorSource] = []

    for selected in selected_rows:
        submission_id = int(selected["submission_id"])
        report_path = _resolve_repo_path(selected["report"]).resolve()
        report = json.loads(report_path.read_text(encoding="utf-8"))
        expected_sha = str(report["code_sha256"])
        raw_source = _resolve_repo_path(report["source"])
        code: str | None = None
        description = ""

        if raw_source.exists():
            candidate = raw_source.read_text(encoding="utf-8")
            if hashlib.sha256(candidate.encode("utf-8")).hexdigest() == expected_sha:
                code = candidate
                description = str(raw_source)

        if code is None:
            artifact = _resolve_repo_path(registry_by_id[submission_id]["artifact"])
            candidate = _read_estimator_from_artifact(artifact)
            if hashlib.sha256(candidate.encode("utf-8")).hexdigest() != expected_sha:
                raise ValueError(
                    f"neither source nor artifact matches {expected_sha} for "
                    f"submission {submission_id}"
                )
            code = candidate
            description = f"{artifact}:estimator.py"

        estimators.append(
            EstimatorSource(
                full100_rank=int(selected["full100_rank"]),
                method=str(selected["method"]),
                submission_id=submission_id,
                report_path=report_path,
                code_sha256=expected_sha,
                code=code,
                source_description=description,
                local_full100_score=float(selected["full100_mean"]),
                hosted_public50_score=(
                    float(selected["hosted_public50_score"])
                    if selected.get("hosted_public50_score") is not None
                    else None
                ),
            )
        )
    return estimators


def _simulate_mean_replica(
    weights: np.ndarray,
    *,
    seed: np.random.SeedSequence,
    n_samples: int,
    chunk_size: int,
    method: str,
) -> tuple[np.ndarray, np.ndarray]:
    depth, width, _ = weights.shape
    layer_sums = np.zeros((depth, width), dtype=np.float64)
    final_sum_squares = np.zeros(width, dtype=np.float64)
    if method == "mc":
        generator: Any = np.random.default_rng(seed)
    elif method == "sobol":
        from scipy.stats import qmc

        generator = qmc.Sobol(
            d=width,
            scramble=True,
            seed=int(seed.generate_state(1)[0]),
        )
    else:
        raise ValueError(f"unsupported truth method: {method}")
    processed = 0
    while processed < n_samples:
        current = min(chunk_size, n_samples - processed)
        if method == "mc":
            activations = generator.standard_normal(
                (current, width),
                dtype=np.float32,
            )
        else:
            from scipy.special import ndtri

            uniforms = generator.random(current)
            activations = ndtri(
                np.clip(
                    uniforms,
                    np.finfo(np.float64).eps,
                    1.0 - np.finfo(np.float64).eps,
                )
            ).astype(np.float32)
        for layer, matrix in enumerate(weights):
            activations = np.maximum(activations @ matrix, np.float32(0.0))
            layer_sums[layer] += np.sum(
                activations,
                axis=0,
                dtype=np.float64,
            )
        final64 = activations.astype(np.float64, copy=False)
        final_sum_squares += np.sum(final64 * final64, axis=0, dtype=np.float64)
        processed += current
    return layer_sums / n_samples, final_sum_squares / n_samples


def _bake_one_mlp(
    mlp_index: int,
    root_seed: int,
    width: int,
    depth: int,
    samples_per_repeat: int,
    repeats: int,
    chunk_size: int,
    truth_method: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    weights = generate_official_weights(root_seed, width=width, depth=depth)
    spawned = np.random.SeedSequence(root_seed).spawn(3)
    sample_replicas = spawned[1].spawn(repeats)
    means = np.empty((repeats, depth, width), dtype=np.float64)
    final_second_moments = np.empty((repeats, width), dtype=np.float64)
    for repeat, sample_seed in enumerate(sample_replicas):
        means[repeat], final_second_moments[repeat] = _simulate_mean_replica(
            weights,
            seed=sample_seed,
            n_samples=samples_per_repeat,
            chunk_size=chunk_size,
            method=truth_method,
        )

    combined_means = np.mean(means, axis=0)
    combined_final_m2 = np.mean(final_second_moments, axis=0)
    final_variance = float(
        np.mean(combined_final_m2 - combined_means[-1] ** 2)
    )
    total_samples = samples_per_repeat * repeats
    analytic_mean_noise_floor = final_variance / total_samples
    repeat_mean_noise_floor = (
        float(np.mean(np.var(means[:, -1, :], axis=0, ddof=1)) / repeats)
        if repeats > 1
        else float("nan")
    )
    selected_mean_noise_floor = (
        repeat_mean_noise_floor
        if truth_method == "sobol" and repeats > 1
        else analytic_mean_noise_floor
    )
    estimator_seed = int(spawned[2].generate_state(1)[0])
    return {
        "mlp_index": mlp_index,
        "root_seed": root_seed,
        "estimator_seed": estimator_seed,
        "weights": weights,
        "all_layer_target": combined_means.astype(np.float32),
        "final_target": combined_means[-1].astype(np.float32),
        "avg_variance": final_variance,
        "analytic_mean_noise_floor": analytic_mean_noise_floor,
        "repeat_mean_noise_floor": repeat_mean_noise_floor,
        "selected_mean_noise_floor": selected_mean_noise_floor,
        "elapsed_s": time.perf_counter() - started,
    }


def bake_fresh_suite(
    *,
    n_mlps: int,
    mlp_offset: int,
    width: int,
    depth: int,
    samples_per_repeat: int,
    repeats: int,
    chunk_size: int,
    truth_method: str,
    seed: int,
    workers: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    roots: list[int] = []
    seen: set[int] = set()
    while len(roots) < n_mlps + mlp_offset:
        value = int(rng.integers(0, 2**63, dtype=np.int64))
        if value not in seen:
            seen.add(value)
            roots.append(value)
    roots = roots[mlp_offset:]

    results: list[dict[str, Any] | None] = [None] * n_mlps
    context = get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        futures = {
            pool.submit(
                _bake_one_mlp,
                index + mlp_offset,
                root,
                width,
                depth,
                samples_per_repeat,
                repeats,
                chunk_size,
                truth_method,
            ): index
            for index, root in enumerate(roots)
        }
        completed = 0
        for future in as_completed(futures):
            index = futures[future]
            results[index] = future.result()
            completed += 1
            print(
                f"ground truth {completed}/{n_mlps} "
                f"(mlp={index}, elapsed={results[index]['elapsed_s']:.1f}s)",
                flush=True,
            )
    return [row for row in results if row is not None]


def make_contest_data(
    baked: list[dict[str, Any]],
    *,
    width: int,
    depth: int,
    ground_truth_samples: int,
) -> Any:
    import flopscope.numpy as fnp
    from whestbench.domain import MLP
    from whestbench.naming import assign_unique_names
    from whestbench.scoring import ContestData, ContestSpec

    names = assign_unique_names([int(row["estimator_seed"]) for row in baked])
    mlps = [
        MLP(
            width=width,
            depth=depth,
            weights=[fnp.array(matrix) for matrix in row["weights"]],
            seed=int(row["estimator_seed"]),
            name=names[index],
        )
        for index, row in enumerate(baked)
    ]
    spec = ContestSpec(
        width=width,
        depth=depth,
        n_mlps=len(baked),
        flop_budget=FLOP_BUDGET,
        ground_truth_samples=ground_truth_samples,
        setup_timeout_s=5.0,
        predict_timeout_s=30.0,
        memory_limit_mb=65_536,
        wall_time_limit_s=60.0,
        residual_wall_time_limit_s=None,
        seed=0,
        lambda_flops_per_second=LAMBDA_FLOPS_PER_SECOND,
    )
    return ContestData(
        spec=spec,
        mlps=mlps,
        all_layer_targets=[
            fnp.asarray(row["all_layer_target"], dtype=fnp.float32) for row in baked
        ],
        final_targets=[
            fnp.asarray(row["final_target"], dtype=fnp.float32) for row in baked
        ],
        avg_variances=[float(row["avg_variance"]) for row in baked],
        sampling_budget_breakdown=None,
    )


def _score_estimator(
    estimator: EstimatorSource,
    contest_data: Any,
) -> tuple[EstimatorSource, dict[str, Any], float]:
    from examples.aicrowd_whestbench.env import OfficialSuiteConfig, _score_code

    config = OfficialSuiteConfig(
        dataset="fresh-generated-in-memory",
        revision=None,
        split="fresh",
        n_mlps=len(contest_data.mlps),
        flop_budget=FLOP_BUDGET,
        lambda_flops_per_second=LAMBDA_FLOPS_PER_SECOND,
        seed=0,
        runner="subprocess",
        streaming=False,
    )
    started = time.perf_counter()
    result = _score_code(estimator.code, contest_data, config)
    return estimator, result, time.perf_counter() - started


def run_estimators(
    estimators: list[EstimatorSource],
    contest_data: Any,
    *,
    workers: int,
    checkpoint_dir: Path | None = None,
    checkpoint_key: dict[str, Any] | None = None,
) -> list[tuple[EstimatorSource, dict[str, Any], float]]:
    completed_results: dict[int, tuple[EstimatorSource, dict[str, Any], float]] = {}
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        for estimator in estimators:
            checkpoint_path = checkpoint_dir / f"local_rank_{estimator.full100_rank}.json"
            if not checkpoint_path.exists():
                continue
            payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if (
                payload.get("checkpoint_key") != checkpoint_key
                or payload.get("code_sha256") != estimator.code_sha256
                or int(payload.get("local_full100_rank", -1))
                != estimator.full100_rank
            ):
                raise ValueError(f"stale or mismatched checkpoint: {checkpoint_path}")
            result = payload["result"]
            elapsed = float(payload["elapsed_s"])
            completed_results[estimator.full100_rank] = (
                estimator,
                result,
                elapsed,
            )
            print(
                f"restored estimator local_rank={estimator.full100_rank} "
                f"from {checkpoint_path}",
                flush=True,
            )

    pending = [
        estimator
        for estimator in estimators
        if estimator.full100_rank not in completed_results
    ]
    errors: list[tuple[int, Exception]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_score_estimator, estimator, contest_data): estimator
            for estimator in pending
        }
        completed = len(completed_results)
        for future in as_completed(futures):
            expected = futures[future]
            try:
                estimator, result, elapsed = future.result()
            except Exception as exc:
                errors.append((expected.full100_rank, exc))
                print(
                    f"estimator failed (local_rank={expected.full100_rank}, "
                    f"error={type(exc).__name__}: {exc})",
                    flush=True,
                )
                continue
            completed_results[estimator.full100_rank] = (estimator, result, elapsed)
            completed += 1
            if checkpoint_dir is not None:
                checkpoint_path = (
                    checkpoint_dir / f"local_rank_{estimator.full100_rank}.json"
                )
                temporary_path = checkpoint_path.with_suffix(".json.tmp")
                temporary_path.write_text(
                    json.dumps(
                        {
                            "checkpoint_key": checkpoint_key,
                            "code_sha256": estimator.code_sha256,
                            "local_full100_rank": estimator.full100_rank,
                            "elapsed_s": elapsed,
                            "result": result,
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                temporary_path.replace(checkpoint_path)
            print(
                f"estimator {completed}/{len(estimators)} "
                f"(local_rank={estimator.full100_rank}, "
                f"score={result['adjusted_final_layer_score']:.6g}, "
                f"elapsed={elapsed:.1f}s, failures={result['n_failed_mlps']})",
                flush=True,
            )
    if errors:
        details = ", ".join(
            f"local_rank={rank}: {type(exc).__name__}: {exc}"
            for rank, exc in errors
        )
        raise RuntimeError(f"{len(errors)} estimator(s) failed; {details}")
    return [completed_results[estimator.full100_rank] for estimator in estimators]


def summarize_results(
    scored: list[tuple[EstimatorSource, dict[str, Any], float]],
    baked: list[dict[str, Any]],
) -> dict[str, Any]:
    noise_floors = np.asarray(
        [float(row["selected_mean_noise_floor"]) for row in baked],
        dtype=np.float64,
    )
    method_rows: list[dict[str, Any]] = []

    for estimator, result, elapsed in scored:
        per_mlp_rows = []
        corrected_mses = []
        corrected_measured_adjusted = []
        corrected_flops_adjusted = []
        raw_mses = []
        for index, raw in enumerate(result["per_mlp"]):
            raw_mse = float(raw["final_layer_mse"])
            raw_adjusted = float(raw["adjusted_final_layer_score"])
            corrected_mse = max(0.0, raw_mse - noise_floors[index])
            measured_multiplier = (
                raw_adjusted / raw_mse if raw_mse > 0.0 else 1.0
            )
            failed = any(
                bool(raw.get(flag))
                for flag in (
                    "budget_exhausted",
                    "time_exhausted",
                    "residual_wall_time_exhausted",
                    "combined_budget_exhausted",
                    "error_code",
                )
            )
            flops_multiplier = (
                1.0
                if failed
                else max(0.1, float(raw.get("flops_used", 0)) / FLOP_BUDGET)
            )
            corrected_measured = corrected_mse * measured_multiplier
            corrected_flops = corrected_mse * flops_multiplier
            raw_mses.append(raw_mse)
            corrected_mses.append(corrected_mse)
            corrected_measured_adjusted.append(corrected_measured)
            corrected_flops_adjusted.append(corrected_flops)
            per_mlp_rows.append(
                {
                    "mlp_index": index,
                    "suite_mlp_index": int(baked[index]["mlp_index"]),
                    "mlp_name": raw.get("mlp_name"),
                    "raw_final_mse": raw_mse,
                    "truth_noise_floor": float(noise_floors[index]),
                    "corrected_final_mse": corrected_mse,
                    "measured_score_multiplier": measured_multiplier,
                    "flops_only_score_multiplier": flops_multiplier,
                    "corrected_measured_adjusted_score": corrected_measured,
                    "corrected_flops_only_adjusted_score": corrected_flops,
                    "flops_used": int(raw.get("flops_used", 0)),
                    "effective_compute": float(raw.get("effective_compute", 0.0)),
                    "failed": failed,
                }
            )

        method_rows.append(
            {
                "method": estimator.method,
                "submission_id": estimator.submission_id,
                "source": estimator.source_description,
                "code_sha256": estimator.code_sha256,
                "local_full100_rank": estimator.full100_rank,
                "local_full100_adjusted_score": estimator.local_full100_score,
                "hosted_public50_adjusted_score": estimator.hosted_public50_score,
                "fresh_raw_final_mse": float(np.mean(raw_mses)),
                "fresh_corrected_final_mse": float(np.mean(corrected_mses)),
                "fresh_official_measured_adjusted_score": float(
                    result["adjusted_final_layer_score"]
                ),
                "fresh_corrected_measured_adjusted_score": float(
                    np.mean(corrected_measured_adjusted)
                ),
                "fresh_corrected_flops_only_adjusted_score": float(
                    np.mean(corrected_flops_adjusted)
                ),
                "mean_effective_compute": float(result["mean_effective_compute"]),
                "mean_compute_utilization": float(result["mean_compute_utilization"]),
                "n_failed_mlps": int(result["n_failed_mlps"]),
                "elapsed_s": elapsed,
                "per_mlp": per_mlp_rows,
            }
        )

    local = np.asarray(
        [row["local_full100_adjusted_score"] for row in method_rows],
        dtype=np.float64,
    )
    fresh_mse = np.asarray(
        [row["fresh_corrected_final_mse"] for row in method_rows],
        dtype=np.float64,
    )
    fresh_adjusted = np.asarray(
        [row["fresh_corrected_flops_only_adjusted_score"] for row in method_rows],
        dtype=np.float64,
    )
    hosted_values = [row["hosted_public50_adjusted_score"] for row in method_rows]
    hosted = (
        np.asarray(hosted_values, dtype=np.float64)
        if all(value is not None for value in hosted_values)
        else None
    )

    fresh_mse_ranks = _rank(fresh_mse)
    fresh_adjusted_ranks = _rank(fresh_adjusted)
    for index, row in enumerate(method_rows):
        row["fresh_corrected_mse_rank"] = int(fresh_mse_ranks[index] + 1)
        row["fresh_flops_only_adjusted_rank"] = int(fresh_adjusted_ranks[index] + 1)

    comparisons: dict[str, Any] = {
        "fresh_flops_only_adjusted_vs_local_full100": {
            "spearman": _spearman(fresh_adjusted, local),
            "pairwise_inversion_fraction": _pair_inversion_fraction(
                fresh_adjusted, local
            ),
            "local_top1_fresh_rank": int(fresh_adjusted_ranks[np.argmin(local)] + 1),
        },
        "fresh_corrected_mse_vs_local_full100": {
            "spearman": _spearman(fresh_mse, local),
            "pairwise_inversion_fraction": _pair_inversion_fraction(
                fresh_mse, local
            ),
            "local_top1_fresh_rank": int(fresh_mse_ranks[np.argmin(local)] + 1),
        },
    }
    if hosted is not None:
        hosted_ranks = _rank(hosted)
        for index, row in enumerate(method_rows):
            row["hosted_rank_among_selected"] = int(hosted_ranks[index] + 1)
        comparisons["fresh_flops_only_adjusted_vs_hosted_public50"] = {
            "spearman": _spearman(fresh_adjusted, hosted),
            "pairwise_inversion_fraction": _pair_inversion_fraction(
                fresh_adjusted, hosted
            ),
        }

    return {
        "truth_noise": {
            "selected_noise_floor_mean": float(np.mean(noise_floors)),
            "selected_noise_floor_max": float(np.max(noise_floors)),
            "mc_analytic_noise_floor_mean": float(
                np.mean(
                    [float(row["analytic_mean_noise_floor"]) for row in baked]
                )
            ),
            "repeat_difference_noise_floor_mean": float(
                np.nanmean(
                    [float(row["repeat_mean_noise_floor"]) for row in baked]
                )
            ),
        },
        "comparisons": comparisons,
        "fresh_suite_bootstrap": bootstrap_fresh_methods(method_rows),
        "methods": method_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--n-estimators", type=int, default=15)
    parser.add_argument("--n-mlps", type=int, default=50)
    parser.add_argument(
        "--mlp-offset",
        type=int,
        default=0,
        help="Skip this many deterministic MLP seeds before generating the suite.",
    )
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--depth", type=int, default=32)
    parser.add_argument("--truth-samples-per-repeat", type=int, default=32_768)
    parser.add_argument("--truth-repeats", type=int, default=8)
    parser.add_argument(
        "--truth-method",
        choices=("sobol", "mc"),
        default="sobol",
    )
    parser.add_argument("--truth-chunk-size", type=int, default=8192)
    parser.add_argument("--suite-seed", type=int, default=20260724)
    parser.add_argument("--truth-workers", type=int, default=7)
    parser.add_argument("--estimator-workers", type=int, default=5)
    parser.add_argument(
        "--baked-cache",
        type=Path,
        help="Optional pickle cache for the generated MLPs and truth targets.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        help="Optional directory for resumable per-estimator result checkpoints.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.truth_method == "sobol" and (
        args.truth_samples_per_repeat <= 0
        or args.truth_samples_per_repeat
        & (args.truth_samples_per_repeat - 1)
    ):
        parser.error("--truth-samples-per-repeat must be a power of two for Sobol")
    if args.truth_repeats < 2:
        parser.error("--truth-repeats must be at least two to estimate truth noise")

    started = time.perf_counter()
    estimators = load_selected_estimators(
        selection_path=args.selection.resolve(),
        registry_path=args.registry.resolve(),
        n_estimators=args.n_estimators,
    )
    print(f"loaded {len(estimators)} estimator sources with matching hashes", flush=True)
    cache_key = {
        "n_mlps": args.n_mlps,
        "mlp_offset": args.mlp_offset,
        "width": args.width,
        "depth": args.depth,
        "truth_samples_per_repeat": args.truth_samples_per_repeat,
        "truth_repeats": args.truth_repeats,
        "truth_chunk_size": args.truth_chunk_size,
        "truth_method": args.truth_method,
        "suite_seed": args.suite_seed,
    }
    if args.baked_cache is not None and args.baked_cache.exists():
        with args.baked_cache.open("rb") as stream:
            cache_payload = pickle.load(stream)
        if cache_payload.get("cache_key") != cache_key:
            raise ValueError(f"stale or mismatched baked cache: {args.baked_cache}")
        baked = cache_payload["baked"]
        print(f"restored baked suite from {args.baked_cache}", flush=True)
    else:
        baked = bake_fresh_suite(
            n_mlps=args.n_mlps,
            mlp_offset=args.mlp_offset,
            width=args.width,
            depth=args.depth,
            samples_per_repeat=args.truth_samples_per_repeat,
            repeats=args.truth_repeats,
            chunk_size=args.truth_chunk_size,
            truth_method=args.truth_method,
            seed=args.suite_seed,
            workers=args.truth_workers,
        )
        if args.baked_cache is not None:
            args.baked_cache.parent.mkdir(parents=True, exist_ok=True)
            temporary_cache = args.baked_cache.with_suffix(
                args.baked_cache.suffix + ".tmp"
            )
            with temporary_cache.open("wb") as stream:
                pickle.dump(
                    {"cache_key": cache_key, "baked": baked},
                    stream,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            temporary_cache.replace(args.baked_cache)
            print(f"wrote baked suite cache to {args.baked_cache}", flush=True)
    contest_data = make_contest_data(
        baked,
        width=args.width,
        depth=args.depth,
        ground_truth_samples=args.truth_samples_per_repeat * args.truth_repeats,
    )
    scored = run_estimators(
        estimators,
        contest_data,
        workers=args.estimator_workers,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_key={
            **cache_key,
            "root_seeds": [int(row["root_seed"]) for row in baked],
        },
    )
    summary = summarize_results(scored, baked)
    output = {
        "protocol": {
            "generator": (
                "SeedSequence(root).spawn(3)[0], independent float32 "
                "N(0, 2/width) weights"
            ),
            "input": (
                "independently scrambled Sobol transformed to float32 N(0, 1)"
                if args.truth_method == "sobol"
                else "independent float32 N(0, 1)"
            ),
            "network": "x <- ReLU(x @ W), no bias",
            "runner": "official subprocess",
            "truth_correction": (
                "subtract the across-scramble variance of the averaged truth "
                "from each per-MLP final MSE"
                if args.truth_method == "sobol"
                else "subtract avg_final_variance / total_truth_samples from "
                "each per-MLP final MSE"
            ),
        },
        "suite": {
            "n_mlps": args.n_mlps,
            "mlp_offset": args.mlp_offset,
            "width": args.width,
            "depth": args.depth,
            "suite_seed": args.suite_seed,
            "truth_samples_per_repeat": args.truth_samples_per_repeat,
            "truth_repeats": args.truth_repeats,
            "truth_method": args.truth_method,
            "total_truth_samples_per_mlp": (
                args.truth_samples_per_repeat * args.truth_repeats
            ),
            "root_seeds": [int(row["root_seed"]) for row in baked],
            "estimator_seeds": [int(row["estimator_seed"]) for row in baked],
            "avg_variances": [float(row["avg_variance"]) for row in baked],
        },
        "evaluation": {
            "n_estimators": len(estimators),
            "truth_workers": args.truth_workers,
            "estimator_workers": args.estimator_workers,
            "elapsed_s": time.perf_counter() - started,
            "flop_budget": FLOP_BUDGET,
            "lambda_flops_per_second": LAMBDA_FLOPS_PER_SECOND,
        },
        "results": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary["comparisons"], indent=2, sort_keys=True), flush=True)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
