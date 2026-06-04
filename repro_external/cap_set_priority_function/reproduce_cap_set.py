#!/usr/bin/env python3
"""Reproduce public FunSearch cap-set constructions.

This script is intentionally standalone: it copies the priority functions and
explicit 512-cap construction published in google-deepmind/funsearch.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path
from typing import Callable

import numpy as np


PriorityFn = Callable[[tuple[int, ...], int], float]


def solve(n: int, priority_fn: PriorityFn) -> np.ndarray:
    """Greedily constructs a cap set in F_3^n using a priority function."""
    all_vectors = np.array(
        list(itertools.product((0, 1, 2), repeat=n)), dtype=np.int32
    )
    powers = 3 ** np.arange(n - 1, -1, -1)
    priorities = np.array([priority_fn(tuple(vector), n) for vector in all_vectors])

    capset = np.empty(shape=(0, n), dtype=np.int32)
    while np.any(priorities != -np.inf):
        max_index = int(np.argmax(priorities))
        vector = all_vectors[None, max_index]
        blocking = np.einsum("cn,n->c", (-capset - vector) % 3, powers)
        priorities[blocking] = -np.inf
        priorities[max_index] = -np.inf
        capset = np.concatenate([capset, vector], axis=0)

    return capset


def is_cap_set(vectors: np.ndarray) -> bool:
    """Checks the cap-set property in O(c^2 n), matching the FunSearch notebook."""
    if vectors.ndim != 2:
        return False
    _, n = vectors.shape
    powers = np.array([3**j for j in range(n - 1, -1, -1)], dtype=int)
    raveled = np.einsum("in,n->i", vectors, powers)

    is_blocked = np.full(shape=3**n, fill_value=False, dtype=bool)
    for i, (new_vector, new_index) in enumerate(zip(vectors, raveled)):
        if is_blocked[new_index]:
            return False
        if i >= 1:
            blocking = np.einsum(
                "nk,k->n", (-vectors[:i, :] - new_vector[None, :]) % 3, powers
            )
            is_blocked[blocking] = True
        is_blocked[new_index] = True
    return True


def priority_n8(el: tuple[int, ...], n: int) -> float:
    """FunSearch priority function that builds a 512-cap in n=8."""
    score = n
    in_el = 0
    el_count = el.count(0)

    if el_count == 0:
        score += n**2
        if el[1] == el[-1]:
            score *= 1.5
        if el[2] == el[-2]:
            score *= 1.5
        if el[3] == el[-3]:
            score *= 1.5
    else:
        if el[1] == el[-1]:
            score *= 0.5
        if el[2] == el[-2]:
            score *= 0.5

    for e in el:
        if e == 0:
            if in_el == 0:
                score *= n * 0.5
            elif in_el == el_count - 1:
                score *= 0.5
            else:
                score *= n * 0.5**in_el
            in_el += 1
        else:
            score += 1

    if el[1] == el[-1]:
        score *= 1.5
    if el[2] == el[-2]:
        score *= 1.5

    return score


def priority_n9(el: tuple[int, ...], n: int) -> float:
    """FunSearch priority function that builds a 1082-cap in n=9."""
    el_array = np.array(el, dtype=np.float32)
    weight = (el_array @ el_array) % 3
    a = n // 3
    b = n - n // 3
    s_1 = (el_array[:b] @ el_array[:b]) % 3
    s_3 = (2 * (el_array[:a] @ el_array[:a])) % 3
    s_4 = (el_array[:a] @ el_array[a:b]) % 3
    s_5 = np.sum(el_array[:a] == el_array[-1]) % 3
    return -(3**3) * s_1 + 3**2 * weight + 3**3 * s_3 + 3**2 * s_4 + s_5


def build_512_cap() -> list[tuple[int, ...]]:
    """Explicit 512-cap construction derived from the FunSearch n=8 result."""
    n = 8
    vectors = list(itertools.product(range(3), repeat=n))
    support = lambda v: tuple(i for i in range(n) if v[i] != 0)
    reflections = lambda v: sum(1 for i in range(1, n // 2) if v[i] == v[-i])

    weight8_vectors = [
        v for v in vectors if len(support(v)) == 8 and reflections(v) >= 2
    ]

    supports_16 = [
        (0, 1, 2, 3),
        (0, 1, 2, 5),
        (0, 3, 6, 7),
        (0, 5, 6, 7),
        (1, 3, 4, 6),
        (1, 4, 5, 6),
        (2, 3, 4, 7),
        (2, 4, 5, 7),
    ]
    weight4_vectors = [v for v in vectors if support(v) in supports_16]

    supports_8 = [
        (0, 1, 2, 7),
        (0, 1, 2, 6),
        (0, 1, 3, 7),
        (0, 1, 6, 7),
        (0, 1, 5, 7),
        (0, 2, 3, 6),
        (0, 2, 6, 7),
        (0, 2, 5, 6),
        (1, 2, 4, 7),
        (1, 2, 4, 6),
        (1, 3, 4, 7),
        (1, 4, 6, 7),
        (1, 4, 5, 7),
        (2, 3, 4, 6),
        (2, 4, 6, 7),
        (2, 4, 5, 6),
    ]
    weight4_vectors_2 = [
        v for v in vectors if support(v) in supports_8 and reflections(v) == 1
    ]

    allowed_zeros = [
        (0, 4, 7),
        (0, 2, 4),
        (0, 1, 4),
        (0, 4, 6),
        (1, 2, 6),
        (2, 6, 7),
        (1, 2, 7),
        (1, 6, 7),
    ]
    weight5_vectors = [
        v
        for v in vectors
        if tuple(i for i in range(n) if v[i] == 0) in allowed_zeros
        and reflections(v) <= 1
        and (v[1] * v[7]) % 3 != 1
        and (v[2] * v[6]) % 3 != 1
    ]

    return weight8_vectors + weight4_vectors + weight4_vectors_2 + weight5_vectors


def summarize(name: str, vectors: np.ndarray, seconds: float) -> dict[str, object]:
    return {
        "name": name,
        "shape": list(vectors.shape),
        "size": int(vectors.shape[0]),
        "dimension": int(vectors.shape[1]),
        "is_cap_set": bool(is_cap_set(vectors)),
        "seconds": seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-n9", action="store_true", help="Skip the n=9 run.")
    args = parser.parse_args()

    results = []

    start = time.time()
    explicit = np.array(build_512_cap(), dtype=np.int32)
    results.append(summarize("explicit_512_cap_n8", explicit, time.time() - start))

    start = time.time()
    cap_n8 = solve(8, priority_n8)
    n8_result = summarize("funsearch_priority_n8", cap_n8, time.time() - start)
    n8_result["same_as_explicit"] = set(map(tuple, cap_n8)) == set(map(tuple, explicit))
    results.append(n8_result)

    if not args.skip_n9:
        start = time.time()
        cap_n9 = solve(9, priority_n9)
        results.append(summarize("funsearch_priority_n9", cap_n9, time.time() - start))

    output_path = Path(__file__).with_name("cap_set_results.json")
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    for item in results:
        extra = ""
        if "same_as_explicit" in item:
            extra = f", same_as_explicit={item['same_as_explicit']}"
        print(
            f"{item['name']}: size={item['size']}, dim={item['dimension']}, "
            f"is_cap_set={item['is_cap_set']}, seconds={item['seconds']:.3f}{extra}"
        )
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
