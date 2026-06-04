#!/usr/bin/env python3
"""Run the literal Kakeya programs marked "Code found by AlphaEvolve".

This is different from ``reproduce_finite_field_kakeya.py``: that file
implements the paper's cleaned-up d=3 formula. This script extracts and runs
the final evolved programs from the official AlphaEvolve problem notebook.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import types
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np


OFFICIAL_NOTEBOOK_URL = (
    "https://raw.githubusercontent.com/google-deepmind/"
    "alphaevolve_repository_of_problems/main/experiments/"
    "finite_field_kakeya_problem/finite_field_kakeya.ipynb"
)


def load_notebook(path: Path | None) -> dict[str, Any]:
    if path is not None:
        return json.loads(path.read_text(encoding="utf-8"))

    local_clone = Path(
        "/tmp/alphaevolve_repository_of_problems/experiments/"
        "finite_field_kakeya_problem/finite_field_kakeya.ipynb"
    )
    if local_clone.exists():
        return json.loads(local_clone.read_text(encoding="utf-8"))

    with urllib.request.urlopen(OFFICIAL_NOTEBOOK_URL, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def install_numba_shim_if_needed() -> None:
    """The construction code only assigns numba.njit; a shim is enough here."""
    try:
        import numba  # noqa: F401
    except ModuleNotFoundError:
        sys.modules["numba"] = types.SimpleNamespace(
            njit=lambda f=None, **kwargs: (lambda g: g) if f is None else f
        )


def exec_cell(nb: dict[str, Any], cell_index: int) -> dict[str, Any]:
    namespace: dict[str, Any] = {}
    exec("".join(nb["cells"][cell_index]["source"]), namespace)
    return namespace


def projective_directions_3d(p: int):
    for y in range(p):
        for z in range(p):
            yield (1, y, z)
    for z in range(p):
        yield (0, 1, z)
    yield (0, 0, 1)


def is_kakeya_set_3d(points: np.ndarray, p: int) -> bool:
    point_set = {tuple(map(int, row)) for row in points}
    for direction in projective_directions_3d(p):
        v = np.array(direction, dtype=np.int64)
        found_line = False
        for start in point_set:
            x = np.array(start, dtype=np.int64)
            if all(tuple(((x + t * v) % p).tolist()) in point_set for t in range(p)):
                found_line = True
                break
        if not found_line:
            return False
    return True


def run_d3_last_evolved(nb: dict[str, Any], primes: list[int]) -> list[dict[str, Any]]:
    # In the 3D section, cell 4 is "#@title Code found by AlphaEvolve (experiment 3)".
    ns = exec_cell(nb, 4)
    results = []
    for p in primes:
        started = time.time()
        construction = ns["search_for_best_construction"](p, 3)
        size = len({tuple(row) for row in construction})
        recorded = round(-float(ns[f"best_score_p{p}_d3_iqhd"]))
        formula = (2 * p**3 + 7 * p**2 - 1) // 8 if p % 4 == 1 else None
        valid = is_kakeya_set_3d(construction, p)
        results.append(
            {
                "section": "3D",
                "cell": 4,
                "program": "Code found by AlphaEvolve (experiment 3)",
                "p": p,
                "dimension": 3,
                "size": size,
                "recorded_size": recorded,
                "matches_recorded": size == recorded,
                "paper_formula_size": formula,
                "matches_paper_formula": formula is not None and size == formula,
                "is_kakeya": valid,
                "seconds": time.time() - started,
            }
        )
    return results


def run_d5_literal_last_evolved(
    nb: dict[str, Any], primes: list[int], verify_primes: set[int]
) -> list[dict[str, Any]]:
    # Cell 17 is the literal last "Code found by AlphaEvolve" cell in the notebook.
    ns = exec_cell(nb, 17)
    verifier_ns = dict(ns)
    exec("".join(nb["cells"][18]["source"]), verifier_ns)

    results = []
    for p in primes:
        started = time.time()
        construction = ns["search_for_best_construction"](p, 5)
        size = len({tuple(row) for row in construction})
        recorded = round(-float(ns[f"best_score_p{p}_d5_iqhd"]))
        valid = None
        if p in verify_primes:
            score = verifier_ns["calculate_score"](construction, p, 5)
            valid = score != -np.inf
        results.append(
            {
                "section": "5D",
                "cell": 17,
                "program": "Code found by AlphaEvolve (experiment 2)",
                "p": p,
                "dimension": 5,
                "size": size,
                "recorded_size": recorded,
                "matches_recorded": size == recorded,
                "is_kakeya": valid,
                "seconds": time.time() - started,
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--notebook", type=Path, default=None)
    parser.add_argument("--d3-primes", nargs="*", type=int, default=[5, 13])
    parser.add_argument(
        "--d5-primes", nargs="*", type=int, default=[3, 5, 7, 11, 13, 17, 19]
    )
    parser.add_argument(
        "--d5-verify-primes",
        nargs="*",
        type=int,
        default=[3, 5],
        help="Run the notebook's slow d=5 verifier only for these primes.",
    )
    args = parser.parse_args()

    install_numba_shim_if_needed()
    nb = load_notebook(args.notebook)

    results = []
    results.extend(run_d3_last_evolved(nb, args.d3_primes))
    results.extend(run_d5_literal_last_evolved(nb, args.d5_primes, set(args.d5_verify_primes)))

    output_path = Path(__file__).with_name("alphaevolve_notebook_program_results.json")
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    for item in results:
        valid = "not_checked" if item["is_kakeya"] is None else item["is_kakeya"]
        print(
            f"{item['section']} cell {item['cell']} p={item['p']}: "
            f"size={item['size']}, recorded={item['recorded_size']}, "
            f"matches_recorded={item['matches_recorded']}, is_kakeya={valid}, "
            f"seconds={item['seconds']:.3f}"
        )
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
