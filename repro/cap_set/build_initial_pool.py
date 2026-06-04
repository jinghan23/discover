from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples.cap_set_priority.env import is_cap_set
from ttt_discover.codex_utils.runtime import State


DEFAULT_COUNTS = {
    400: 1,
    411: 3,
    412: 2,
    413: 1,
    414: 2,
    418: 1,
    420: 1,
    424: 1,
    431: 1,
    512: 1,
}


def _canonical(construction: Any) -> tuple[int, tuple[tuple[int, ...], ...]] | None:
    if not isinstance(construction, list) or not construction:
        return None
    try:
        points = tuple(sorted(tuple(int(x) for x in point) for point in construction))
    except (TypeError, ValueError):
        return None
    dims = {len(point) for point in points}
    if len(dims) != 1:
        return None
    return next(iter(dims)), points


def _fingerprint(canonical: tuple[int, tuple[tuple[int, ...], ...]]) -> str:
    payload = json.dumps(canonical, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _walk_states(obj: Any) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if isinstance(obj, dict):
        if _canonical(obj.get("construction")) is not None:
            found.append(obj)
        for value in obj.values():
            found.extend(_walk_states(value))
    elif isinstance(obj, list):
        for value in obj:
            found.extend(_walk_states(value))
    return found


def _add_candidate(
    representatives: dict[str, dict[str, Any]],
    state: dict[str, Any],
    source: str,
) -> None:
    canonical = _canonical(state.get("construction"))
    if canonical is None:
        return
    dim, points = canonical
    if dim != 8:
        return
    size = len(points)
    if size not in DEFAULT_COUNTS:
        return

    key = _fingerprint(canonical)
    entry = representatives.get(key)
    if entry is None:
        representatives[key] = {
            "hash": key,
            "size": size,
            "state": state,
            "sources": [source],
        }
    else:
        entry["sources"].append(source)
        if not entry["state"].get("code") and state.get("code"):
            entry["state"] = state


def _load_snapshot_candidates(log_root: Path) -> dict[str, dict[str, Any]]:
    representatives: dict[str, dict[str, Any]] = {}
    for path in sorted(log_root.glob("capset*/puct_sampler_step_*.json")):
        try:
            store = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        source = str(path.relative_to(ROOT))
        for state in _walk_states(store):
            _add_candidate(representatives, state, source)
    return representatives


def _load_external_512_state() -> dict[str, Any]:
    path = ROOT / "repro_external/cap_set_priority_function/reproduce_cap_set.py"
    spec = importlib.util.spec_from_file_location("cap_set_priority_function", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    construction = module.build_512_cap()
    source = inspect.getsource(module.priority_n8)
    source = source.replace("def priority_n8(", "def priority(", 1)
    code = f"```python\n{source.rstrip()}\n```"
    return State(
        timestep=-1,
        construction=construction,
        code=code,
        value=512.0,
    ).to_dict()


def _verify_state(state: dict[str, Any]) -> bool:
    construction = state.get("construction")
    if not isinstance(construction, list):
        return False
    return bool(is_cap_set(np.array(construction, dtype=int)))


def _as_root_initial_state(state: dict[str, Any]) -> dict[str, Any]:
    rooted = dict(state)
    rooted["timestep"] = -1
    rooted["parent_values"] = []
    rooted["parents"] = []
    rooted["observation"] = ""
    return rooted


def build_pool(log_root: Path, counts: dict[int, int]) -> dict[str, Any]:
    representatives = _load_snapshot_candidates(log_root)
    _add_candidate(
        representatives,
        _load_external_512_state(),
        "repro_external/cap_set_priority_function/reproduce_cap_set.py:build_512_cap",
    )

    by_size: dict[int, list[dict[str, Any]]] = {}
    for entry in representatives.values():
        by_size.setdefault(int(entry["size"]), []).append(entry)

    selected: list[dict[str, Any]] = []
    selected_entries: list[dict[str, Any]] = []
    for size, count in sorted(counts.items()):
        entries = sorted(by_size.get(size, []), key=lambda item: item["hash"])
        if len(entries) < count:
            raise RuntimeError(
                f"Requested {count} constructions of size {size}, found {len(entries)}"
            )
        selected_entries.extend(entries[:count])
        selected.extend(_as_root_initial_state(entries[i]["state"]) for i in range(count))

    size_counts = Counter(int(state["value"]) for state in selected)
    return {
        "name": "capset_initial_pool_400_to_512",
        "problem_type": "8",
        "selection": {str(size): count for size, count in sorted(counts.items())},
        "size_counts": {str(size): size_counts[size] for size in sorted(size_counts)},
        "entries": [
            {
                "hash": entry["hash"],
                "size": entry["size"],
                "source_count": len(set(entry["sources"])),
                "sources": sorted(set(entry["sources"]))[:8],
            }
            for entry in selected_entries
        ],
        "states": selected,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the cap-set high-score initial pool from existing sampler logs."
    )
    parser.add_argument(
        "--log-root",
        default=str(ROOT / "tinker_log"),
        help="Directory containing capset*/puct_sampler_step_*.json logs.",
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).with_name("initial_pool_400_to_512.json")),
        help="Output JSON path.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Validate all selected constructions before writing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_root = Path(args.log_root)
    if not log_root.is_absolute():
        log_root = ROOT / log_root
    pool = build_pool(log_root, DEFAULT_COUNTS)

    if args.verify:
        invalid = [
            (i, len(state.get("construction") or []))
            for i, state in enumerate(pool["states"])
            if not _verify_state(state)
        ]
        if invalid:
            raise RuntimeError(f"Invalid cap-set states selected: {invalid}")

    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(pool, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Wrote {len(pool['states'])} states to {output}")
    print(json.dumps(pool["size_counts"], sort_keys=True))


if __name__ == "__main__":
    main()
