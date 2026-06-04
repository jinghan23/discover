#!/usr/bin/env python3
"""Evaluate copied AlphaResearchComp initial programs."""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import sys
import time
from pathlib import Path
from typing import Any


RUNS_ROOT = Path(__file__).resolve().parents[1]


def available_problems() -> list[str]:
    return sorted(
        p.name
        for p in RUNS_ROOT.iterdir()
        if p.is_dir() and not p.name.startswith("_")
    )


def _load_module(path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def evaluate_problem(problem: str) -> dict[str, Any]:
    problem_dir = RUNS_ROOT / problem
    work_dir = problem_dir / "work"
    evaluator_path = work_dir / "evaluator.py"
    program_path = work_dir / "initial_program.py"
    if not evaluator_path.exists():
        raise FileNotFoundError(evaluator_path)
    if not program_path.exists():
        raise FileNotFoundError(program_path)

    stdout = io.StringIO()
    stderr = io.StringIO()
    start = time.time()
    old_argv = sys.argv[:]
    old_path = sys.path[:]
    try:
        sys.argv = [str(program_path)]
        sys.path.insert(0, str(work_dir))
        sys.modules.pop("program", None)
        module_name = f"alpharesearch_eval_{problem}"
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            evaluator = _load_module(evaluator_path, module_name)
            result = _json_safe(evaluator.evaluate(str(program_path)))
        status = "ok"
    except Exception as exc:  # Keep the batch alive for all problems.
        result = {"error": repr(exc)}
        status = "error"
    finally:
        sys.argv = old_argv
        sys.path = old_path
        elapsed = time.time() - start

    payload = {
        "problem": problem,
        "status": status,
        "program_path": str(program_path),
        "evaluator_path": str(evaluator_path),
        "elapsed_sec": elapsed,
        "result": result,
        "captured_stdout": stdout.getvalue(),
        "captured_stderr": stderr.getvalue(),
    }
    out_path = problem_dir / "initial_eval.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("problems", nargs="*")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    problems = available_problems() if args.all or not args.problems else args.problems
    known = set(available_problems())
    unknown = [p for p in problems if p not in known]
    if unknown:
        print(f"Unknown problems: {', '.join(unknown)}", file=sys.stderr)
        print(f"Available: {', '.join(sorted(known))}", file=sys.stderr)
        return 2

    results = [evaluate_problem(problem) for problem in problems]
    summary = []
    for item in results:
        result = item["result"]
        score = result.get("score", result.get("ratio", result.get("error")))
        summary.append(
            {
                "problem": item["problem"],
                "status": item["status"],
                "score": score,
                "elapsed_sec": round(item["elapsed_sec"], 3),
                "result": result,
            }
        )
    (RUNS_ROOT / "initial_eval_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
