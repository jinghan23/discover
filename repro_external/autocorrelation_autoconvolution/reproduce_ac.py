#!/usr/bin/env python3
"""Reproduce public AlphaEvolve and local TTT autocorrelation scores."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np


OFFICIAL_AC_NOTEBOOK_URL = (
    "https://raw.githubusercontent.com/google-deepmind/"
    "alphaevolve_repository_of_problems/main/experiments/"
    "autocorrelation_problems/autocorrelation_problems.ipynb"
)

ONLINE_AC2_BEST_URL = (
    "https://media.githubusercontent.com/media/justinkang221/"
    "second-autocorrelation-inequality/main/solutions/best_2031546.npy"
)


def evaluate_ac1(sequence: list[float]) -> float:
    """AC1 objective used in ttt-discover: lower is better."""
    seq = np.array([min(1000.0, max(0.0, float(x))) for x in sequence], dtype=float)
    if seq.size == 0 or np.sum(seq) < 0.01:
        return float("inf")
    conv = np.convolve(seq, seq)
    return float(2 * len(seq) * np.max(conv) / (np.sum(seq) ** 2))


def evaluate_ac2(sequence: list[float] | np.ndarray) -> float:
    """AC2 lower-bound objective used in ttt-discover: higher is better."""
    seq = np.asarray(sequence, dtype=float)
    seq = np.clip(seq, 0.0, 1000.0)
    if seq.size == 0 or np.sum(seq) < 0.01:
        raise ValueError("Invalid sequence")
    if seq.size > 200_000:
        conv_size = 2 * seq.size - 1
        nfft = 1 << (conv_size - 1).bit_length()
        spectrum = np.fft.rfft(seq, n=nfft)
        conv = np.fft.irfft(spectrum * spectrum, n=nfft)[:conv_size]
    else:
        conv = np.convolve(seq, seq)
    m = len(conv)
    x_points = np.linspace(-0.5, 0.5, m + 2)
    widths = np.diff(x_points)
    y_points = np.concatenate(([0.0], conv, [0.0]))

    l2_norm_squared = 0.0
    for i in range(m + 1):
        y1 = y_points[i]
        y2 = y_points[i + 1]
        h = widths[i]
        l2_norm_squared += (h / 3.0) * (y1**2 + y1 * y2 + y2**2)

    norm_1 = float(np.sum(np.abs(conv)) / (m + 1))
    norm_inf = float(np.max(np.abs(conv)))
    return float(l2_norm_squared / (norm_1 * norm_inf))


def load_json_sequence(path: Path) -> list[float]:
    return json.loads(path.read_text(encoding="utf-8"))["sequence"]


def load_official_notebook(path: Path | None) -> dict[str, Any]:
    if path is not None:
        return json.loads(path.read_text(encoding="utf-8"))

    local_clone = Path(
        "/tmp/alphaevolve_repository_of_problems/experiments/"
        "autocorrelation_problems/autocorrelation_problems.ipynb"
    )
    if local_clone.exists():
        return json.loads(local_clone.read_text(encoding="utf-8"))

    with urllib.request.urlopen(OFFICIAL_AC_NOTEBOOK_URL, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def exec_cell_silently(source: str, namespace: dict[str, Any]) -> None:
    with contextlib.redirect_stdout(io.StringIO()):
        exec(source, namespace)


def official_alphaevolve_ac1(nb: dict[str, Any]) -> tuple[list[float], float]:
    """Extracts the best AC1 sequence from the official problem notebook."""
    ns: dict[str, Any] = {"np": np, "evaluate_sequence": evaluate_ac1}
    # Cell 60 is titled "Data and verification (score = 1.5032)" in the public
    # notebook and contains the best displayed AlphaEvolve AC1 sequence.
    exec_cell_silently("".join(nb["cells"][60]["source"]), ns)
    sequence = [float(x) for x in ns["best_sequence"]]
    return sequence, evaluate_ac1(sequence)


def official_alphaevolve_ac2(nb: dict[str, Any]) -> tuple[list[float], float]:
    """Extracts the AC2 50k-step sequence from the official problem notebook."""
    ns: dict[str, Any] = {"np": np}
    exec_cell_silently("".join(nb["cells"][91]["source"]), ns)
    exec_cell_silently("".join(nb["cells"][92]["source"]), ns)
    sequence = [float(x) for x in ns["heights_sequence_2"]]
    return sequence, float(ns["C_lower_bound"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-notebook", type=Path, default=None)
    parser.add_argument(
        "--include-online-ac2-best",
        action="store_true",
        help="Download and verify the current public EinsteinArena/ClaudeExplorer AC2 best.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    official_nb = load_official_notebook(args.official_notebook)

    ae_ac1_seq, ae_ac1_score = official_alphaevolve_ac1(official_nb)
    ae_ac2_seq, ae_ac2_score = official_alphaevolve_ac2(official_nb)

    ttt_ac1_seq = load_json_sequence(repo_root / "results/mathematics/ttt_ac1_sequence.json")
    ttt_ac2_seq = load_json_sequence(repo_root / "results/mathematics/ttt_ac2_sequence.json")
    ttt_ac1_score = evaluate_ac1(ttt_ac1_seq)
    ttt_ac2_score = evaluate_ac2(ttt_ac2_seq)

    results = [
        {
            "name": "AlphaEvolve official AC1",
            "length": len(ae_ac1_seq),
            "score": ae_ac1_score,
            "direction": "lower_is_better",
        },
        {
            "name": "TTT-Discover local AC1",
            "length": len(ttt_ac1_seq),
            "score": ttt_ac1_score,
            "direction": "lower_is_better",
        },
        {
            "name": "AlphaEvolve official AC2",
            "length": len(ae_ac2_seq),
            "score": ae_ac2_score,
            "direction": "higher_is_better",
        },
        {
            "name": "TTT-Discover local AC2",
            "length": len(ttt_ac2_seq),
            "score": ttt_ac2_score,
            "direction": "higher_is_better",
        },
    ]

    if args.include_online_ac2_best:
        external_dir = Path(__file__).resolve().parent / "external_downloads"
        external_dir.mkdir(exist_ok=True)
        online_path = external_dir / "best_2031546.npy"
        if not online_path.exists():
            urllib.request.urlretrieve(ONLINE_AC2_BEST_URL, online_path)
        online_seq = np.load(online_path)
        results.append(
            {
                "name": "EinsteinArena ClaudeExplorer public AC2",
                "length": int(len(online_seq)),
                "score": evaluate_ac2(online_seq),
                "direction": "higher_is_better",
                "source": ONLINE_AC2_BEST_URL,
            }
        )

    output_path = Path(__file__).with_name("ac_results.json")
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    for result in results:
        print(
            f"{result['name']}: length={result['length']}, "
            f"score={result['score']:.15f}, {result['direction']}"
        )
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
