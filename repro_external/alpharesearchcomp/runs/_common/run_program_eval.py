#!/usr/bin/env python3
"""Evaluate an arbitrary candidate program for one AlphaResearchComp run folder."""

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


def _load_module(path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("problem")
    parser.add_argument("program", nargs="?", default="work/improved_program.py")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    problem_dir = RUNS_ROOT / args.problem
    evaluator_path = problem_dir / "work" / "evaluator.py"
    program_path = Path(args.program)
    if not program_path.is_absolute():
        program_path = problem_dir / program_path

    stdout = io.StringIO()
    stderr = io.StringIO()
    start = time.time()
    old_argv = sys.argv[:]
    old_path = sys.path[:]
    try:
        sys.argv = [str(program_path)]
        sys.path.insert(0, str(program_path.parent))
        sys.modules.pop("program", None)
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            evaluator = _load_module(evaluator_path, f"alpharesearch_eval_{args.problem}")
            result = _json_safe(evaluator.evaluate(str(program_path)))
        status = "ok"
    except Exception as exc:
        result = {"error": repr(exc)}
        status = "error"
    finally:
        sys.argv = old_argv
        sys.path = old_path

    payload = {
        "problem": args.problem,
        "status": status,
        "program_path": str(program_path),
        "evaluator_path": str(evaluator_path),
        "elapsed_sec": time.time() - start,
        "result": result,
        "captured_stdout": stdout.getvalue(),
        "captured_stderr": stderr.getvalue(),
    }
    out_path = Path(args.out) if args.out else problem_dir / "improved_eval.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
