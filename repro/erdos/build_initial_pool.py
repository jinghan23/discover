from __future__ import annotations

import argparse
import hashlib
import json
import math
import runpy
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ttt_discover.algorithms.state import State


THRESHOLD_C5 = 0.3808694472025862
DEFAULT_SAMPLER = (
    ROOT
    / "tinker_log/erdos-codex-seeded-500s-3-20260601-180030/puct_sampler_step_000053.json"
)
REFERENCE_BASE = (
    ROOT
    / "repro_external/alpharesearchcomp/_loongflow/agents/math_agent/examples/minimum_overlap_problem/reference_programs"
)
DEFAULT_OUTPUT = Path(__file__).with_name("initial_pool_reference_plus_codex_20260603.json")


def _compute_c5(h_values: np.ndarray) -> float:
    h = np.asarray(h_values, dtype=float)
    return float(np.max(np.correlate(h, 1.0 - h, mode="full")) * 2.0 / int(h.size))


def _canonical_h(values: Any) -> tuple[list[float], float]:
    h = np.asarray(values, dtype=float).reshape(-1)
    if h.size == 0:
        raise ValueError("empty construction")
    if not np.all(np.isfinite(h)):
        raise ValueError("construction contains non-finite values")
    if np.any(h < -1e-9) or np.any(h > 1.0 + 1e-9):
        raise ValueError(f"construction outside [0, 1]: {h.min()}..{h.max()}")

    h = np.clip(h, 0.0, 1.0)
    target_sum = h.size / 2.0
    current_sum = float(np.sum(h))
    if not math.isclose(current_sum, target_sum, rel_tol=0.0, abs_tol=1e-8):
        if current_sum <= 0:
            raise ValueError("construction has non-positive sum")
        h = h * (target_sum / current_sum)
        if np.any(h < -1e-9) or np.any(h > 1.0 + 1e-9):
            raise ValueError(
                f"normalized construction outside [0, 1]: {h.min()}..{h.max()}"
            )
        h = np.clip(h, 0.0, 1.0)

    return h.astype(float).tolist(), _compute_c5(h)


def _replay_code(source_name: str) -> str:
    return f'''```python
import numpy as np

# Replays the fixed construction imported from {source_name}.
def run(seed=42, budget_s=500, **kwargs):
    h = np.asarray(initial_h_values, dtype=float).copy()
    n_points = int(h.size)
    dx = 2.0 / n_points
    c5_bound = float(np.max(np.correlate(h, 1.0 - h, mode="full") * dx))
    return h, c5_bound, n_points
```'''


def _state_id(name: str, source_path: Path) -> str:
    key = f"{name}:{source_path}".encode("utf-8")
    return "pool_" + hashlib.sha1(key).hexdigest()[:16]


def _root_state(state: dict[str, Any]) -> dict[str, Any]:
    rooted = dict(state)
    rooted["timestep"] = -1
    rooted["parents"] = []
    rooted["parent_values"] = []
    rooted["observation"] = (rooted.get("observation") or "").strip()
    return rooted


def _add_h_state(
    states: list[dict[str, Any]],
    entries: list[dict[str, Any]],
    *,
    name: str,
    source_path: Path,
    source_type: str,
    h_values: Any,
    enabled: bool = True,
    note: str = "",
) -> None:
    construction, c5 = _canonical_h(h_values)
    entry = {
        "name": name,
        "enabled": bool(enabled),
        "source_type": source_type,
        "source_path": str(source_path),
        "c5": c5,
        "n_points": len(construction),
        "note": note,
    }
    entries.append(entry)
    if not enabled:
        return

    states.append(
        State(
            id=_state_id(name, source_path),
            timestep=-1,
            construction=construction,
            code=_replay_code(source_path.name),
            value=-float(c5),
            observation=(
                f"Imported initial-pool construction from {source_path.name}; "
                f"C5={c5:.16f}; n={len(construction)}."
            ),
        ).to_dict()
    )


def _load_reference_h(path: Path) -> Any:
    if path.suffix == ".py":
        return runpy.run_path(str(path))["h_values"]

    data = json.loads(path.read_text(encoding="utf-8"))
    if "h_values" in data:
        return data["h_values"]
    if data.get("__simpleevolve_type__") == "tuple":
        return data["items"][0]["data"]
    raise ValueError(f"Cannot load h_values from {path}")


def _collect_codex_states(sampler_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    store = json.loads(sampler_path.read_text(encoding="utf-8"))
    selected: list[tuple[float, dict[str, Any]]] = []
    for state in store.get("states", []):
        value = state.get("value")
        if value is None:
            continue
        c5 = -float(value)
        if c5 < THRESHOLD_C5:
            selected.append((c5, state))

    states: list[dict[str, Any]] = []
    entries: list[dict[str, Any]] = []
    for c5, state in sorted(selected, key=lambda item: (item[0], item[1].get("timestep", 0))):
        name = f"codex_step_{int(state.get('timestep', -1)):06d}_{str(state.get('id'))[:8]}"
        rooted = _root_state(state)
        states.append(rooted)
        entries.append(
            {
                "name": name,
                "enabled": True,
                "source_type": "codex_sampler_state",
                "source_path": str(sampler_path),
                "c5": c5,
                "n_points": len(state.get("construction") or []),
                "original_id": state.get("id"),
                "original_timestep": state.get("timestep"),
                "original_parent_count": len(state.get("parents") or []),
            }
        )
    return states, entries


def build_pool(sampler_path: Path, reference_base: Path) -> dict[str, Any]:
    states, entries = _collect_codex_states(sampler_path)

    reference_specs = [
        ("reference_autoevolver_result", "final_constructions/autoevolver_result.json"),
        ("reference_together_ai_2026", "final_constructions/together_ai_2026.py"),
        ("reference_ttt_discover_2026", "final_constructions/ttt_discover_2026.py"),
        ("reference_alphaevolve_2025", "final_constructions/alphaevolve_2025.py"),
        (
            "reference_simpletes_best_construction",
            "final_constructions/erdos_minimum_overlap_best_construction.json",
        ),
    ]
    for name, rel_path in reference_specs:
        path = reference_base / rel_path
        _add_h_state(
            states,
            entries,
            name=name,
            source_path=path,
            source_type="reference_final_construction",
            h_values=_load_reference_h(path),
        )

    for path in sorted((reference_base / "agent4science").glob("*_solution_data.json")):
        h_values, c5 = _canonical_h(json.loads(path.read_text(encoding="utf-8"))["values"])
        _add_h_state(
            states,
            entries,
            name="agent4science_" + path.name.replace("_solution_data.json", ""),
            source_path=path,
            source_type="agent4science_solution_data",
            h_values=h_values,
            enabled=c5 <= 0.382,
            note="enabled close public solution" if c5 <= 0.382 else "disabled weak public solution",
        )

    manifest_path = reference_base / "manifest.json"
    if manifest_path.exists():
        known_sources = {entry["source_path"] for entry in entries}
        source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for item in source_manifest.get("items", []):
            raw_path = item.get("path")
            if raw_path is None:
                continue
            source_path = Path(raw_path)
            if not source_path.is_absolute():
                source_path = reference_base / source_path
            if str(source_path) in known_sources:
                continue
            entries.append(
                {
                    "name": source_path.stem,
                    "enabled": False,
                    "source_type": item.get("kind"),
                    "source_path": str(source_path),
                    "source_url": item.get("source_url"),
                    "upstream_revision": item.get("upstream_revision"),
                    "note": item.get("notes", "listed in SOURCE_SUMMARY/manifest"),
                }
            )

    return {
        "name": "erdos_reference_plus_codex_20260603",
        "problem_type": "",
        "metric": "C5 bound; lower is better; sampler value is -C5",
        "threshold_c5": THRESHOLD_C5,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_summary_path": str(reference_base / "SOURCE_SUMMARY.zh.md"),
        "codex_sampler_path": str(sampler_path),
        "entries": entries,
        "states": states,
        "counts": {
            "states": len(states),
            "entries": len(entries),
            "codex_states_better_than_threshold": sum(
                1 for entry in entries if entry.get("source_type") == "codex_sampler_state"
            ),
            "disabled_entries": sum(1 for entry in entries if not entry.get("enabled")),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Erdős Codex initial pool.")
    parser.add_argument("--sampler", default=str(DEFAULT_SAMPLER))
    parser.add_argument("--reference-base", default=str(REFERENCE_BASE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sampler_path = Path(args.sampler)
    reference_base = Path(args.reference_base)
    output = Path(args.output)
    if not sampler_path.is_absolute():
        sampler_path = ROOT / sampler_path
    if not reference_base.is_absolute():
        reference_base = ROOT / reference_base
    if not output.is_absolute():
        output = ROOT / output

    pool = build_pool(sampler_path, reference_base)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(pool, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {len(pool['states'])} states to {output}")
    print(json.dumps(pool["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
