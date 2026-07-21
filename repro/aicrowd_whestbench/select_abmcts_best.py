#!/usr/bin/env python3
"""Export a ranked valid candidate from the latest AB-MCTS checkpoint."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
from pathlib import Path
from typing import Any


_STEP_RE = re.compile(r"abmcts_sampler_step_(\d+)\.json$")


def _latest_checkpoint(log_dir: Path) -> Path:
    candidates: list[tuple[int, Path]] = []
    for raw_path in glob.glob(str(log_dir / "abmcts_sampler_step_*.json")):
        path = Path(raw_path)
        match = _STEP_RE.search(path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        raise FileNotFoundError(f"No AB-MCTS checkpoint found under {log_dir}")
    return max(candidates, key=lambda item: item[0])[1]


def _state_value(node: dict[str, Any]) -> float:
    value = (node.get("state") or {}).get("value")
    return float(value) if value is not None else float("-inf")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument(
        "--rank",
        type=int,
        default=1,
        help="1-based rank by tree value (default: 1, the best candidate)",
    )
    parser.add_argument(
        "--exclude-root",
        action="store_true",
        help="Rank only generated, non-root candidates",
    )
    args = parser.parse_args()
    if args.rank < 1:
        parser.error("--rank must be >= 1")

    checkpoint = _latest_checkpoint(args.log_dir.expanduser().resolve())
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    nodes = payload.get("nodes") or []
    valid_nodes = [
        node
        for node in nodes
        if isinstance(node, dict)
        and isinstance(node.get("state"), dict)
        and (node["state"].get("code") or "").strip()
        and node["state"].get("value") is not None
    ]
    if not valid_nodes:
        raise ValueError(f"Checkpoint contains no valid candidate states: {checkpoint}")

    ranked_nodes = sorted(
        (
            node
            for node in valid_nodes
            if not args.exclude_root or node.get("parent_id") is not None
        ),
        key=_state_value,
        reverse=True,
    )
    if args.rank > len(ranked_nodes):
        raise ValueError(
            f"Requested rank {args.rank}, but only {len(ranked_nodes)} "
            f"matching candidates exist in {checkpoint}"
        )

    selected = ranked_nodes[args.rank - 1]
    state = selected["state"]
    code = str(state["code"]).strip() + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(code, encoding="utf-8")

    summary_path = args.summary or args.output.with_suffix(".json")
    summary = {
        "checkpoint": str(checkpoint),
        "checkpoint_step": int(payload.get("step", 0)),
        "node_id": int(selected["node_id"]),
        "parent_id": selected.get("parent_id"),
        "depth": int(selected.get("depth", 0)),
        "action": selected.get("action"),
        "candidate_rank": args.rank,
        "candidate_count": len(ranked_nodes),
        "excluded_root": args.exclude_root,
        "state_id": state.get("id"),
        "tree_value": float(state["value"]),
        "adjusted_score": -float(state["value"]),
        "code_sha256": hashlib.sha256(code.encode("utf-8")).hexdigest(),
        "output": str(args.output.resolve()),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
