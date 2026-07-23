#!/usr/bin/env python3
"""Continuously traverse three-letter proposer codes in resumable batches."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import string
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE = REPO_ROOT / "repro/aicrowd_whestbench/run_whest_acronym_pipeline.py"
TOTAL_CODES = 26**3


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def code_from_index(index: int) -> str:
    if not 0 <= index < TOTAL_CODES:
        raise ValueError(f"code index must be in [0, {TOTAL_CODES}): {index}")
    first, remainder = divmod(index, 26 * 26)
    second, third = divmod(remainder, 26)
    alphabet = string.ascii_uppercase
    return alphabet[first] + alphabet[second] + alphabet[third]


def index_from_code(code: str) -> int:
    normalized = code.strip().upper()
    if re.fullmatch(r"[A-Z]{3}", normalized) is None:
        raise ValueError(f"expected a three-letter code, got {code!r}")
    return sum(
        (ord(letter) - ord("A")) * 26**power
        for letter, power in zip(normalized, (2, 1, 0))
    )


def next_codes(
    start_index: int, count: int, skipped: set[str]
) -> tuple[list[str], int]:
    """Select the next lexicographic codes and return the following cursor."""
    if count <= 0:
        raise ValueError("count must be positive")
    cursor = start_index
    selected: list[str] = []
    while cursor < TOTAL_CODES and len(selected) < count:
        code = code_from_index(cursor)
        cursor += 1
        if code not in skipped:
            selected.append(code)
    return selected, cursor


def _load_state(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected an object in {path}")
    return payload


def _initial_state(args: argparse.Namespace, root: Path) -> dict[str, Any]:
    skipped = sorted(set(args.skip))
    return {
        "version": 1,
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "status": "running",
        "root": str(root),
        "batch_size": args.batch_size,
        "filter_keep": args.keep,
        "next_index": index_from_code(args.start),
        "next_code": args.start.upper(),
        "skipped_codes": skipped,
        "traversed_count": 0,
        "completed_batches": 0,
        "failed_batches": 0,
        "consecutive_failures": 0,
        "active_batch": None,
        "batches": [],
    }


def _reconcile_interrupted(state: dict[str, Any]) -> None:
    active = state.get("active_batch")
    if not isinstance(active, dict):
        return
    interrupted = dict(active)
    interrupted.update(
        {
            "status": "interrupted",
            "completed_at": _utc_now(),
            "note": "Supervisor restarted after the prior process released its lock",
        }
    )
    state["batches"].append(interrupted)
    state["failed_batches"] = int(state.get("failed_batches", 0)) + 1
    state["active_batch"] = None


def _run(args: argparse.Namespace) -> int:
    if args.batch_size <= args.keep:
        raise ValueError("--batch-size must be greater than --keep")
    if args.max_consecutive_failures <= 0:
        raise ValueError("--max-consecutive-failures must be positive")
    for code in args.skip:
        index_from_code(code)

    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / "sweep_state.json"
    lock_handle = (root / "sweep.lock").open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock_path = root / "sweep.lock"
        raise RuntimeError(f"another sweep supervisor holds {lock_path}") from exc

    if state_path.exists():
        state = _load_state(state_path)
        if (
            int(state["batch_size"]) != args.batch_size
            or int(state["filter_keep"]) != args.keep
        ):
            raise ValueError(
                "existing sweep batch-size/keep does not match requested values"
            )
        _reconcile_interrupted(state)
        state["status"] = "running"
    else:
        state = _initial_state(args, root)
    _atomic_json(state_path, state)

    batches_this_process = 0
    while args.max_batches == 0 or batches_this_process < args.max_batches:
        skipped = set(str(code) for code in state["skipped_codes"])
        codes, following_index = next_codes(
            int(state["next_index"]), int(state["batch_size"]), skipped
        )
        if not codes:
            state["status"] = "completed_all_codes"
            state["completed_at"] = _utc_now()
            state["next_code"] = None
            state["updated_at"] = _utc_now()
            _atomic_json(state_path, state)
            print(f"[{_utc_now()}] traversal complete", flush=True)
            return 0

        batch_number = len(state["batches"])
        batch_name = (
            f"batch_{batch_number:04d}_{codes[0].lower()}_{codes[-1].lower()}"
        )
        batch_root = root / "batches" / batch_name
        tag = (
            f"{root.name}_{batch_number:04d}_"
            f"{codes[0].lower()}_{codes[-1].lower()}"
        )
        record = {
            "batch": batch_number,
            "name": batch_name,
            "codes": codes,
            "start_index": int(state["next_index"]),
            "following_index": following_index,
            "run_root": str(batch_root),
            "tag": tag,
            "started_at": _utc_now(),
            "status": "running",
        }
        state["active_batch"] = record
        state["updated_at"] = _utc_now()
        _atomic_json(state_path, state)

        env = os.environ.copy()
        env.update(
            {
                "RUN_WHEST_ACRONYM_PROPOSERS": str(len(codes)),
                "RUN_WHEST_ACRONYM_KEEP": str(state["filter_keep"]),
                "RUN_WHEST_ACRONYM_CODES": ",".join(codes),
                "RUN_WHEST_ACRONYM_CODE_SOURCE": "lexicographic_traversal",
                "RUN_WHEST_ACRONYM_TAG": tag,
                "RUN_WHEST_ACRONYM_ROOT": str(batch_root),
            }
        )
        print(
            f"[{_utc_now()}] starting {batch_name}: {', '.join(codes)}",
            flush=True,
        )
        returncode = subprocess.run(
            [sys.executable, str(PIPELINE)], cwd=REPO_ROOT, env=env, check=False
        ).returncode

        manifest_path = batch_root / "manifest.json"
        manifest: dict[str, Any] = {}
        if manifest_path.is_file():
            manifest = _load_state(manifest_path)
        record.update(
            {
                "completed_at": _utc_now(),
                "returncode": returncode,
                "status": "completed" if returncode == 0 else "failed",
                "manifest": str(manifest_path),
                "pipeline_status": manifest.get("status"),
                "selected_codes": manifest.get("filter", {}).get(
                    "selected_codes", []
                ),
            }
        )
        state["batches"].append(record)
        state["active_batch"] = None
        state["next_index"] = following_index
        state["next_code"] = (
            code_from_index(following_index)
            if following_index < TOTAL_CODES
            else None
        )
        state["traversed_count"] = int(state["traversed_count"]) + len(codes)
        if returncode == 0:
            state["completed_batches"] = int(state["completed_batches"]) + 1
            state["consecutive_failures"] = 0
        else:
            state["failed_batches"] = int(state["failed_batches"]) + 1
            state["consecutive_failures"] = int(state["consecutive_failures"]) + 1
        state["updated_at"] = _utc_now()
        _atomic_json(state_path, state)
        batches_this_process += 1
        print(
            f"[{_utc_now()}] {batch_name} exited {returncode}; "
            f"next={state['next_code']}",
            flush=True,
        )

        if int(state["consecutive_failures"]) >= args.max_consecutive_failures:
            state["status"] = "stopped_after_consecutive_failures"
            state["stopped_at"] = _utc_now()
            state["updated_at"] = _utc_now()
            _atomic_json(state_path, state)
            return 1

    state["status"] = "stopped_at_max_batches"
    state["updated_at"] = _utc_now()
    _atomic_json(state_path, state)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--keep", type=int, default=2)
    parser.add_argument("--start", default="AAA")
    parser.add_argument("--skip", action="append", default=[])
    parser.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="0 means continue until ZZZ or the failure circuit breaker",
    )
    parser.add_argument("--max-consecutive-failures", type=int, default=3)
    args = parser.parse_args()
    try:
        return _run(args)
    except Exception as exc:
        print(
            f"sweep failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
