#!/usr/bin/env python3
"""Run public AlphaResearch agent code outputs through the C3 evaluator."""

from __future__ import annotations

import importlib.util
import json
import traceback
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent


def run_agent_code(path: Path) -> dict:
    try:
        spec = importlib.util.spec_from_file_location("agent_code", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        sequence = module.find_better_c3_upper_bound()
        convolution = np.convolve(sequence, sequence)
        c_upper = abs(2 * len(sequence) * np.max(convolution) / (np.sum(sequence) ** 2))
        return {
            "status": "success",
            "sequence_length": int(len(sequence)),
            "score_inverse_c_upper": float(1.0 / c_upper),
            "c_upper": float(c_upper),
        }
    except Exception as exc:
        return {
            "status": "execution_failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback_tail": traceback.format_exc().splitlines()[-6:],
        }


def main() -> None:
    targets = [
        ROOT / "third_autocorrelation_e436c26a" / "agent_code.py",
        ROOT / "third_autocorrelation_4f4c7847" / "agent_code.py",
    ]
    results = {}
    for target in targets:
        results[target.parent.name] = run_agent_code(target)
    out = ROOT / "run_results.json"
    out.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(results, indent=2, sort_keys=True))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
