#!/usr/bin/env python3
"""Compare idea-only reconstructions with the official AlphaEvolve cells.

The reconstructed modules in this directory are intentionally written from
idea summaries rather than by copying the original notebook cells. This script
uses the official cells only as the oracle for comparison.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
import types
import urllib.request
from pathlib import Path
from typing import Any, Callable

import numpy as np


ROOT = Path(__file__).resolve().parent
OFFICIAL_NOTEBOOK_URL = (
    "https://raw.githubusercontent.com/google-deepmind/"
    "alphaevolve_repository_of_problems/main/experiments/"
    "finite_field_kakeya_problem/finite_field_kakeya.ipynb"
)
LOCAL_NOTEBOOK_CANDIDATES = [
    Path("/tmp/alphaevolve_kakeya/finite_field_kakeya.ipynb"),
    Path(
        "/tmp/alphaevolve_repository_of_problems/experiments/"
        "finite_field_kakeya_problem/finite_field_kakeya.ipynb"
    ),
    Path(
        "/Users/bytedance/Documents/workspace/ttt-discover/"
        "repro_external/alpharesearchcomp/_alphaevolve_repo/experiments/"
        "finite_field_kakeya_problem/finite_field_kakeya.ipynb"
    ),
]
EXPERIMENTS = {
    "exp1": {"cell": 2, "module": ROOT / "exp1_reconstructed.py"},
    "exp2": {"cell": 3, "module": ROOT / "exp2_reconstructed.py"},
    "exp3": {"cell": 4, "module": ROOT / "exp3_reconstructed.py"},
}


def install_numba_shim_if_needed() -> None:
    try:
        import numba  # noqa: F401
    except ModuleNotFoundError:
        sys.modules["numba"] = types.SimpleNamespace(
            njit=lambda f=None, **kwargs: (lambda g: g) if f is None else f
        )


def load_notebook(path: Path | None) -> dict[str, Any]:
    if path is not None:
        return json.loads(path.read_text(encoding="utf-8"))

    for candidate in LOCAL_NOTEBOOK_CANDIDATES:
        if candidate.exists():
            return json.loads(candidate.read_text(encoding="utf-8"))

    with urllib.request.urlopen(OFFICIAL_NOTEBOOK_URL, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def exec_cell(nb: dict[str, Any], cell_index: int) -> Callable[[int, int], np.ndarray]:
    namespace: dict[str, Any] = {}
    exec("".join(nb["cells"][cell_index]["source"]), namespace)
    return namespace["search_for_best_construction"]


def load_module_function(path: Path) -> Callable[[int, int], np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.search_for_best_construction


def point_set(points: np.ndarray) -> set[tuple[int, int, int]]:
    return {tuple(map(int, row)) for row in np.asarray(points)}


def projective_directions_3d(p: int):
    for y in range(p):
        for z in range(p):
            yield (1, y, z)
    for z in range(p):
        yield (0, 1, z)
    yield (0, 0, 1)


def first_missing_direction(points: np.ndarray, p: int) -> tuple[int, int, int] | None:
    points_as_set = point_set(points)
    starts = [np.array(start, dtype=np.int64) for start in points_as_set]
    for direction in projective_directions_3d(p):
        v = np.array(direction, dtype=np.int64)
        found_line = False
        for start in starts:
            if all(
                tuple(((start + t * v) % p).tolist()) in points_as_set
                for t in range(p)
            ):
                found_line = True
                break
        if not found_line:
            return direction
    return None


def run_one(
    name: str,
    p: int,
    original_fn: Callable[[int, int], np.ndarray],
    reconstructed_fn: Callable[[int, int], np.ndarray],
) -> dict[str, Any]:
    started = time.time()
    original = original_fn(p, 3)
    original_seconds = time.time() - started

    started = time.time()
    reconstructed = reconstructed_fn(p, 3)
    reconstructed_seconds = time.time() - started

    original_set = point_set(original)
    reconstructed_set = point_set(reconstructed)
    missing_original = first_missing_direction(original, p)
    missing_reconstructed = first_missing_direction(reconstructed, p)

    return {
        "experiment": name,
        "p": p,
        "original_size": len(original_set),
        "reconstructed_size": len(reconstructed_set),
        "size_match": len(original_set) == len(reconstructed_set),
        "point_set_equal": original_set == reconstructed_set,
        "original_is_kakeya": missing_original is None,
        "reconstructed_is_kakeya": missing_reconstructed is None,
        "missing_original_direction": missing_original,
        "missing_reconstructed_direction": missing_reconstructed,
        "original_seconds": original_seconds,
        "reconstructed_seconds": reconstructed_seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--notebook", type=Path, default=None)
    parser.add_argument("--primes", nargs="*", type=int, default=[3, 5, 7, 13])
    parser.add_argument("--experiments", nargs="*", default=list(EXPERIMENTS))
    args = parser.parse_args()

    install_numba_shim_if_needed()
    nb = load_notebook(args.notebook)

    results = []
    for name in args.experiments:
        meta = EXPERIMENTS[name]
        original_fn = exec_cell(nb, int(meta["cell"]))
        reconstructed_fn = load_module_function(Path(meta["module"]))
        for p in args.primes:
            item = run_one(name, p, original_fn, reconstructed_fn)
            results.append(item)
            print(
                f"{name} p={p}: original={item['original_size']} "
                f"reconstructed={item['reconstructed_size']} "
                f"size_match={item['size_match']} "
                f"equal={item['point_set_equal']} "
                f"valid={item['reconstructed_is_kakeya']} "
                f"seconds=({item['original_seconds']:.3f},"
                f"{item['reconstructed_seconds']:.3f})"
            )

    output_path = ROOT / "comparison_results.json"
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
