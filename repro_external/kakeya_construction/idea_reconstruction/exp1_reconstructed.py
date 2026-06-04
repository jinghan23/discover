"""Idea-only reconstruction of AlphaEvolve Kakeya 3D Experiment 1.

This module intentionally implements only the construction described in the
prompt.  It does not depend on the official notebook cells.
"""

from __future__ import annotations

from itertools import product

import numpy as np


def _strip_values(p: int, t: int, c: int, d: int) -> tuple[int, ...]:
    """Return V_t(c,d) as a sorted tuple of unique residues in F_p."""
    values = {
        (s * ((2 * t + c - s) % p) + t + d) % p
        for s in range(p)
    }
    return tuple(sorted(values))


def _candidate_constants(p: int) -> tuple[int, ...]:
    inv2 = pow(2, -1, p)
    return tuple(dict.fromkeys((0, 1 % p, inv2, (-1) % p)))


def _build_construction(
    p: int,
    constants: tuple[int, int, int, int, int, int],
    strips: dict[tuple[int, int, int], tuple[int, ...]],
) -> np.ndarray:
    c1, c2, c3, d1, d2, d3 = constants
    inv2 = pow(2, -1, p)
    points: set[tuple[int, int, int]] = set()

    # K1: product of two strips in u=y+z and v=y-z coordinates.
    for x in range(p):
        u_values = strips[(x, c1, d1)]
        v_values = strips[(x, c3, d3)]
        for u in u_values:
            for v in v_values:
                y = ((u + v) * inv2) % p
                z = ((u - v) * inv2) % p
                points.add((x, y, z))

    # K2: a 2D strip in the plane x=0.
    for y in range(p):
        for z in strips[(y, c2, d2)]:
            points.add((0, y, z))

    # K3: the vertical line for direction (0,0,1).
    for z in range(p):
        points.add((0, 0, z))

    return np.array(sorted(points), dtype=np.int64)


def search_for_best_construction(p: int, d: int) -> np.ndarray:
    """Search the constant grid from the idea description and return K in F_p^3.

    Only ``d=3`` is supported.  For the degenerate ``p=2`` case, where the
    u/v coordinate change cannot divide by two, return the full affine space as
    a simple valid Kakeya set.
    """
    if d != 3:
        raise ValueError("Experiment 1 reconstruction only supports d=3")
    if p < 2:
        raise ValueError("p must be at least 2")
    if p == 2:
        return np.array(list(product(range(p), repeat=3)), dtype=np.int64)
    if p % 2 == 0:
        raise ValueError("odd prime p is required, except for p=2 fallback")

    candidates = _candidate_constants(p)
    strips = {
        (t, c, delta): _strip_values(p, t, c, delta)
        for t in range(p)
        for c in candidates
        for delta in candidates
    }

    best: np.ndarray | None = None
    best_size: int | None = None
    for constants in product(candidates, repeat=6):
        construction = _build_construction(p, constants, strips)
        size = len(construction)
        if best_size is None or size < best_size:
            best = construction
            best_size = size

    if best is None:
        raise RuntimeError("search did not produce a construction")
    return best


if __name__ == "__main__":
    for prime in (2, 3, 5, 7):
        result = search_for_best_construction(prime, 3)
        print(f"p={prime}: {len(result)} points")
