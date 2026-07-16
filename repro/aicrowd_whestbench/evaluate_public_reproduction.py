#!/usr/bin/env python3
"""Evaluate one public WhestBench reproduction with the official mini suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from examples.aicrowd_whestbench.env import (
    OfficialSuiteConfig,
    _load_contest_data,
    _score_code,
)


DATASET = "aicrowd/arc-whestbench-public-2026"
REVISION = "v1-phase1"
SPLIT = "mini"
FLOP_BUDGET = 272_000_000_000
LAMBDA_FLOPS_PER_SECOND = 100_000_000_000.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("estimator", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--n-mlps", type=int, default=100)
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--split", default=SPLIT)
    parser.add_argument("--flop-budget", type=int, default=FLOP_BUDGET)
    parser.add_argument(
        "--lambda-flops-per-second",
        type=float,
        default=LAMBDA_FLOPS_PER_SECOND,
    )
    args = parser.parse_args()

    source = args.estimator.read_text(encoding="utf-8")
    config = OfficialSuiteConfig(
        dataset=args.dataset,
        revision=args.revision,
        split=args.split,
        n_mlps=args.n_mlps,
        flop_budget=args.flop_budget,
        lambda_flops_per_second=args.lambda_flops_per_second,
        runner="subprocess",
        streaming=False,
    )
    data = _load_contest_data(config)

    started = time.perf_counter()
    results = _score_code(source, data, config)
    elapsed = time.perf_counter() - started

    summary_keys = (
        "adjusted_final_layer_score",
        "final_layer_mse",
        "all_layers_mse",
        "mean_compute_utilization",
        "mean_effective_compute",
        "mean_score_multiplier",
        "n_failed_mlps",
        "best_mlp_adjusted_final_layer_score",
        "worst_mlp_adjusted_final_layer_score",
    )
    summary = {key: results[key] for key in summary_keys}
    summary["max_effective_compute"] = max(
        float(row.get("effective_compute", 0.0)) for row in results["per_mlp"]
    )

    report = {
        "code_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "dataset": {
            "repo": args.dataset,
            "revision": args.revision,
            "split": args.split,
            "n_mlps": args.n_mlps,
            "width": data.spec.width,
            "depth": data.spec.depth,
        },
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation": {
            "elapsed_s": elapsed,
            "flop_budget": args.flop_budget,
            "lambda_flops_per_second": args.lambda_flops_per_second,
            "runner": "subprocess",
        },
        "results": results,
        "source": str(args.estimator),
        "summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
