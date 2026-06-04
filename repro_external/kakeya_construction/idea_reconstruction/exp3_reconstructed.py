"""Idea-only reconstruction of AlphaEvolve Kakeya 3D Experiment 3.

This module implements only the construction described in the prompt.  It does
not depend on the official notebook cells.
"""

from __future__ import annotations

from itertools import product

import numpy as np


Point2 = tuple[int, int]
Point3 = tuple[int, int, int]


def _quadratic_residues(p: int) -> tuple[int, ...]:
    return tuple(sorted({(r * r) % p for r in range(p)}))


def _least_qnr(p: int, residues: set[int]) -> int:
    for value in range(1, p):
        if value not in residues:
            return value
    raise ValueError("could not find a nonzero quadratic non-residue")


def _candidate_gammas(p: int, residues: set[int]) -> tuple[int, ...]:
    if p <= 15:
        return tuple(range(p))

    baselines = {
        0,
        1 % p,
        (-1) % p,
        2 % p,
        (-2) % p,
        ((p - 1) // 2) % p,
        ((p + 1) // 2) % p,
        (p // 3) % p,
        (p // 4) % p,
        (p // 5) % p,
        pow(3, -1, p),
        pow(4, -1, p),
        pow(5, -1, p),
        _least_qnr(p, residues),
    }

    candidates: set[int] = set()
    for base in baselines:
        for shift in range(-3, 4):
            candidates.add((base + shift) % p)
    return tuple(sorted(candidates))


def _a0_points(
    p: int,
    residues: tuple[int, ...],
    gamma1: int,
    gamma2: int,
    inv2: int,
) -> set[Point2]:
    points: set[Point2] = set()
    for q1 in residues:
        u = (q1 - gamma1) % p
        for q2 in residues:
            v = (q2 - gamma2) % p
            points.add((((u + v) * inv2) % p, ((u - v) * inv2) % p))
    return points


def _b_points(p: int, residues: tuple[int, ...], gamma3: int) -> set[Point2]:
    points: set[Point2] = set()
    for y in range(p):
        y2 = (y * y) % p
        for q in residues:
            points.add((y, (q - y2 - gamma3) % p))
    return points


def _rows_by_y(points: set[Point2], p: int) -> tuple[frozenset[int], ...]:
    rows: list[set[int]] = [set() for _ in range(p)]
    for y, z in points:
        rows[y].add(z)
    return tuple(frozenset(row) for row in rows)


def _choose_gammas(
    p: int,
    residues: tuple[int, ...],
    candidates: tuple[int, ...],
    inv2: int,
) -> tuple[int, int, int, int]:
    a_data: dict[tuple[int, int], tuple[set[Point2], tuple[frozenset[int], ...]]] = {}
    for gamma1, gamma2 in product(candidates, repeat=2):
        a_set = _a0_points(p, residues, gamma1, gamma2, inv2)
        a_data[(gamma1, gamma2)] = (a_set, _rows_by_y(a_set, p))

    b_data = {gamma3: _b_points(p, residues, gamma3) for gamma3 in candidates}

    ka_size = p * len(residues) * len(residues)
    k0_size = p * len(residues) + p - len(residues)
    best_size: int | None = None
    best_gammas: tuple[int, int, int, int] | None = None

    for gamma1, gamma2 in product(candidates, repeat=2):
        a_set, a_rows = a_data[(gamma1, gamma2)]
        for gamma3 in candidates:
            ab = a_set & b_data[gamma3]
            ab_rows = _rows_by_y(ab, p)
            base_overlap = len(ab)
            for gamma4 in candidates:
                line_overlap = len(a_rows[gamma4] - ab_rows[gamma4])
                union_size = ka_size + k0_size - base_overlap - line_overlap
                gammas = (gamma1, gamma2, gamma3, gamma4)
                if (
                    best_size is None
                    or union_size < best_size
                    or (union_size == best_size and gammas < best_gammas)
                ):
                    best_size = union_size
                    best_gammas = gammas

    if best_gammas is None:
        raise RuntimeError("gamma search did not produce a construction")
    return best_gammas


def _build_construction(
    p: int,
    residues: tuple[int, ...],
    gammas: tuple[int, int, int, int],
    inv2: int,
) -> np.ndarray:
    gamma1, gamma2, gamma3, gamma4 = gammas
    points: set[Point3] = set()

    for x in range(p):
        x2 = (x * x) % p
        for q1 in residues:
            u = (q1 - x2 - gamma1) % p
            for q2 in residues:
                v = (q2 - x2 - gamma2) % p
                y = ((u + v) * inv2) % p
                z = ((u - v) * inv2) % p
                points.add((x, y, z))

    for y in range(p):
        y2 = (y * y) % p
        for q in residues:
            points.add((0, y, (q - y2 - gamma3) % p))

    for z in range(p):
        points.add((0, gamma4, z))

    return np.array(sorted(points), dtype=np.int64)


def search_for_best_construction(p: int, d: int) -> np.ndarray:
    """Return the Experiment 3 Kakeya construction in ``F_p^3``.

    Only ``d=3`` is supported.  For ``p=2``, the coordinate change involving
    division by two degenerates, so this returns the full affine space as a
    compact valid fallback.
    """
    if d != 3:
        raise ValueError("Experiment 3 reconstruction only supports d=3")
    if p < 2:
        raise ValueError("p must be at least 2")
    if p == 2:
        return np.array(list(product(range(p), repeat=3)), dtype=np.int64)
    if p % 2 == 0:
        raise ValueError("odd prime p is required, except for p=2 fallback")

    inv2 = pow(2, -1, p)
    residues = _quadratic_residues(p)
    candidates = _candidate_gammas(p, set(residues))
    gammas = _choose_gammas(p, residues, candidates, inv2)
    return _build_construction(p, residues, gammas, inv2)


if __name__ == "__main__":
    for prime in (2, 3, 5, 7, 13):
        result = search_for_best_construction(prime, 3)
        print(f"p={prime}: {len(result)} points")
