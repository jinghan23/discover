#!/usr/bin/env python3
"""Run the official AlphaEvolve AC1 final evolved search program.

The public notebook hard-codes a 1000 second search. This runner extracts that
exact notebook cell and only patches the time budget so it is easy to smoke-test
or rerun for the original budget.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np


OFFICIAL_AC_NOTEBOOK_URL = (
    "https://raw.githubusercontent.com/google-deepmind/"
    "alphaevolve_repository_of_problems/main/experiments/"
    "autocorrelation_problems/autocorrelation_problems.ipynb"
)


def evaluate_ac1(sequence: list[float]) -> float:
    if not isinstance(sequence, list) or not sequence:
        return float("inf")
    clean = []
    for x in sequence:
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            return float("inf")
        if np.isnan(x) or np.isinf(x):
            return float("inf")
        clean.append(min(1000.0, max(0.0, float(x))))
    total = float(np.sum(clean))
    if total < 0.01:
        return float("inf")
    conv = np.convolve(clean, clean)
    return float(2 * len(clean) * np.max(conv) / (total * total))


def load_notebook(path: Path | None) -> dict[str, Any]:
    if path is not None:
        return json.loads(path.read_text(encoding="utf-8"))
    local_clone = Path(
        "/tmp/alphaevolve_repository_of_problems/experiments/"
        "autocorrelation_problems/autocorrelation_problems.ipynb"
    )
    if local_clone.exists():
        return json.loads(local_clone.read_text(encoding="utf-8"))
    with urllib.request.urlopen(OFFICIAL_AC_NOTEBOOK_URL, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def patched_source(source: str, budget: int) -> str:
    source = source.replace(
        "while time.time() - start_time < 1000:",
        f"while time.time() - start_time < {budget}:",
    )
    source = source.replace(
        "max(0, 1000 - (time.time() - start_time))",
        f"max(0, {budget} - (time.time() - start_time))",
    )
    source = source.replace(
        "temperature = time_left / 1000.0",
        f"temperature = time_left / {float(budget)}",
    )
    return source


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=60)
    parser.add_argument("--notebook", type=Path, default=None)
    args = parser.parse_args()

    notebook = load_notebook(args.notebook)
    # Cell 34 is "#@title Code of the final evolved program".
    source = "".join(notebook["cells"][34]["source"])
    namespace: dict[str, Any] = {"evaluate_sequence": evaluate_ac1}
    exec(patched_source(source, args.budget), namespace)

    print("initial all-ones n=300 score:", evaluate_ac1([1.0] * 300))
    sequence = namespace["search_for_best_sequence"]()
    score = evaluate_ac1(sequence)
    output = {
        "budget_seconds": args.budget,
        "length": len(sequence),
        "score": score,
        "sequence": sequence,
    }
    output_path = Path(__file__).with_name("official_ac1_evolved_search_result.json")
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"final length: {len(sequence)}")
    print(f"final score: {score}")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
