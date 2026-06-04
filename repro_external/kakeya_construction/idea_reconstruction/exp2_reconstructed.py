"""Idea-only reconstruction of the AlphaEvolve 3D Kakeya Experiment 2.

The public entry point is ``search_for_best_construction(p, d)``.  It builds
the two-piece construction described in the prompt, searches a small parameter
grid for maximum overlap, and returns the smallest point set found.
"""

from __future__ import annotations

from itertools import product
from typing import Iterable

import numpy as np


TRANSFORMS = ("identity", "rotated", "shear", "diagonal_scaled")


def _constant_candidates(p: int) -> tuple[int, ...]:
    inv2 = pow(2, -1, p)
    inv4 = pow(4, -1, p)
    raw = (0, 1, -1, inv2, -inv2, inv4, -inv4)

    seen: set[int] = set()
    values: list[int] = []
    for value in raw:
        reduced = value % p
        if reduced not in seen:
            seen.add(reduced)
            values.append(reduced)
    return tuple(values)


def _linearity_candidates(p: int) -> tuple[int, ...]:
    return tuple(value % p for value in (0, 1, -1))


def _apply_transform(u: int, v: int, p: int, inv2: int, transform: str) -> tuple[int, int]:
    if transform == "identity":
        return u % p, v % p
    if transform == "rotated":
        return ((u + v) * inv2) % p, ((u - v) * inv2) % p
    if transform == "shear":
        return (u + v) % p, v % p
    if transform == "diagonal_scaled":
        scale_y = 2 % p
        scale_z = 3 % p if p != 3 else 2 % p
        return (scale_y * u) % p, (scale_z * v) % p
    raise ValueError(f"unknown transform: {transform}")


def _encode(x: int, y: int, z: int, p: int) -> int:
    return (x * p + y) * p + z


def _decode_points(codes: Iterable[int], p: int) -> np.ndarray:
    ordered = np.fromiter(sorted(codes), dtype=np.int64)
    if ordered.size == 0:
        return np.empty((0, 3), dtype=np.int64)

    x = ordered // (p * p)
    rem = ordered % (p * p)
    y = rem // p
    z = rem % p
    return np.column_stack((x, y, z)).astype(np.int64, copy=False)


def _build_k1(
    p: int,
    ly: int,
    lz: int,
    cy: int,
    cz: int,
    transform: str,
    square_residues: tuple[int, ...],
    inv2: int,
) -> set[int]:
    points: set[int] = set()
    for x in range(p):
        x2 = (x * x) % p
        u_base = (x2 + ly * x + cy) % p
        v_base = (x2 + lz * x + cz) % p
        u_values = tuple((u_base - square) % p for square in square_residues)
        v_values = tuple((v_base - square) % p for square in square_residues)

        for u in u_values:
            for v in v_values:
                y, z = _apply_transform(u, v, p, inv2, transform)
                points.add(_encode(x, y, z, p))
    return points


def _build_k2(p: int, l2: int, c2: int) -> set[int]:
    points: set[int] = set()
    for b in range(p):
        intercept = (b * b + l2 * b + c2) % p
        for y in range(p):
            z = (2 * b * y - intercept) % p
            points.add(_encode(0, y, z, p))

    for z in range(p):
        points.add(_encode(0, 0, z, p))
    return points


def _fallback_for_p2() -> np.ndarray:
    return np.array(
        [(x, y, z) for x in range(2) for y in range(2) for z in range(2)],
        dtype=np.int64,
    )


def search_for_best_construction(p: int, d: int) -> np.ndarray:
    """Return a deduplicated point set in F_p^3 for the Experiment 2 idea.

    The main path is for odd primes.  For ``p == 2`` the finite-field halves
    used by the construction degenerate, so this returns the whole 2x2x2 space.
    """

    if d != 3:
        raise ValueError("this reconstruction only supports d=3")
    if p < 2:
        raise ValueError("p must be at least 2")
    if p == 2:
        return _fallback_for_p2()
    if p % 2 == 0:
        raise ValueError("p must be an odd prime, or p=2 for the fallback")

    constants = _constant_candidates(p)
    linearities = _linearity_candidates(p)
    inv2 = pow(2, -1, p)
    square_residues = tuple(sorted({(s * s) % p for s in range(p)}))

    k2_options = [
        (l2, c2, _build_k2(p, l2, c2))
        for l2, c2 in product(linearities, constants)
    ]

    best_points: set[int] | None = None
    best_size: int | None = None

    for ly, lz, cy, cz, transform in product(
        linearities, linearities, constants, constants, TRANSFORMS
    ):
        k1 = _build_k1(p, ly, lz, cy, cz, transform, square_residues, inv2)
        k1_size = len(k1)

        for _l2, _c2, k2 in k2_options:
            size = k1_size + len(k2) - len(k1.intersection(k2))
            if best_size is None or size < best_size:
                best_size = size
                best_points = k1 | k2

    if best_points is None:
        raise RuntimeError("parameter search did not produce a construction")
    return _decode_points(best_points, p)


if __name__ == "__main__":
    for prime in (2, 3, 5, 7):
        construction = search_for_best_construction(prime, 3)
        print(f"p={prime}: {len(construction)} points")
