from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


DEFAULT_SEQUENCE_PATH = Path("results/mathematics/ttt_erdos_sequence.json")


def load_sequence(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text())
    if "sequence" not in payload:
        raise KeyError(f"{path} does not contain a 'sequence' key")
    return np.asarray(payload["sequence"], dtype=np.float64)


def compute_c5_bound(sequence: np.ndarray) -> float:
    convolution = np.correlate(sequence, 1.0 - sequence, mode="full")
    return float(np.max(convolution) / len(sequence) * 2.0)


def validate_sequence(sequence: np.ndarray, *, atol: float) -> None:
    if sequence.ndim != 1:
        raise ValueError(f"Expected a 1D sequence, got shape {sequence.shape}")
    if sequence.size == 0:
        raise ValueError("Sequence is empty")
    if not np.all(np.isfinite(sequence)):
        raise ValueError("Sequence contains NaN or infinite values")
    if np.any(sequence < -atol) or np.any(sequence > 1.0 + atol):
        raise ValueError(
            f"Sequence must stay in [0, 1], got range "
            f"[{sequence.min()}, {sequence.max()}]"
        )

    target_sum = sequence.size / 2.0
    if not np.isclose(sequence.sum(), target_sum, atol=atol, rtol=0.0):
        raise ValueError(
            f"Sequence sum is {sequence.sum()} but expected {target_sum}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate an Erdős minimum-overlap sequence and compute C5."
    )
    parser.add_argument(
        "--sequence",
        type=Path,
        default=DEFAULT_SEQUENCE_PATH,
        help="JSON file with a top-level 'sequence' array.",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-9,
        help="Absolute tolerance for constraints.",
    )
    args = parser.parse_args()

    sequence = load_sequence(args.sequence)
    validate_sequence(sequence, atol=args.atol)
    c5_bound = compute_c5_bound(sequence)

    print(f"sequence: {args.sequence}")
    print(f"n_points: {sequence.size}")
    print(f"sum(h): {sequence.sum():.12f}")
    print(f"target sum: {sequence.size / 2.0:.12f}")
    print(f"min(h): {sequence.min():.12f}")
    print(f"max(h): {sequence.max():.12f}")
    print(f"c5_bound: {c5_bound:.12f}")


if __name__ == "__main__":
    main()
