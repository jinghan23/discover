#!/usr/bin/env python3
"""Reproduce an AlphaEvolve finite-field Kakeya construction.

The construction is the explicit d=3 family reported in
Georgiev-Gomez-Serrano-Tao-Wagner, "Mathematical exploration and discovery at
scale", Appendix: Finite field Kakeya and Nikodym sets.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterable

import numpy as np


def quadratic_residues(p: int) -> set[int]:
    return {(i * i) % p for i in range(p)}


def alphaevolve_kakeya_3d(p: int) -> np.ndarray:
    """Builds the AlphaEvolve d=3 Kakeya set for primes p == 1 mod 4."""
    if p % 4 != 1:
        raise ValueError("This explicit family is stated for primes p congruent to 1 mod 4.")

    residues = quadratic_residues(p)
    inv2 = pow(2, -1, p)
    g = ((p - 1) // 4) % p
    points: set[tuple[int, int, int]] = set()

    # Main part:
    # {(x, (q1+q2)/2 - x^2 - g, (q1-q2)/2): x in F_p; q1,q2 in S}
    for x in range(p):
        x2 = (x * x) % p
        for q1 in residues:
            for q2 in residues:
                y = ((q1 + q2) * inv2 - x2 - g) % p
                z = ((q1 - q2) * inv2) % p
                points.add((x, y, z))

    # Planar parts:
    # {(0,y,z): y+z^2 in S} union {(0,y,0): y in F_p}
    for y in range(p):
        for z in range(p):
            if (y + z * z) % p in residues:
                points.add((0, y, z))
        points.add((0, y, 0))

    return np.array(sorted(points), dtype=np.int64)


def expected_size_formula(p: int) -> int:
    """Formula stated in the paper for p == 1 mod 4."""
    numerator = 2 * p**3 + 7 * p**2 - 1
    assert numerator % 8 == 0
    return numerator // 8


def projective_directions_3d(p: int) -> Iterable[tuple[int, int, int]]:
    """Represent each nonzero direction in F_p^3 by first nonzero coordinate 1."""
    for y in range(p):
        for z in range(p):
            yield (1, y, z)
    for z in range(p):
        yield (0, 1, z)
    yield (0, 0, 1)


def is_kakeya_set_3d(points: np.ndarray, p: int) -> bool:
    """Brute-force verification that a set contains one line in every direction."""
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--primes",
        nargs="*",
        type=int,
        default=[5, 13, 17],
        help="Primes p == 1 mod 4 to verify.",
    )
    args = parser.parse_args()

    results = []
    for p in args.primes:
        start = time.time()
        construction = alphaevolve_kakeya_3d(p)
        expected_size = expected_size_formula(p)
        verified = is_kakeya_set_3d(construction, p)
        seconds = time.time() - start
        result = {
            "p": p,
            "dimension": 3,
            "size": int(len(construction)),
            "expected_size_formula": int(expected_size),
            "matches_formula": bool(len(construction) == expected_size),
            "is_kakeya": bool(verified),
            "seconds": seconds,
        }
        results.append(result)
        print(
            f"p={p}: size={len(construction)}, expected={expected_size}, "
            f"matches_formula={len(construction) == expected_size}, "
            f"is_kakeya={verified}, seconds={seconds:.3f}"
        )

    output_path = Path(__file__).with_name("finite_field_kakeya_results.json")
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
