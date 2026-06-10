# codex/diversity-birth-phase-surface-balance

## Summary

按 timestep % 4 生成 phase_0..phase_3，并把 seed/unknown 单独成面；在 sampling_diversity=birth_phase_surface 时平衡出生相位。

## Branch State

- Worktree: `/opt/tiger/discover-birth-phase-surface-balance`
- HEAD: `2234282`
- Base used for comparison: `2234282`
- Commits ahead of base: `0`
- Commits behind base: `0`
- Group: `completed_19`
- Implementation location: `uncommitted worktree diff`

## Implemented Behavior

- The summary above is based on reading the changed Python code, added config fields, and sampler/prompt hooks in this branch.
- For sampler branches, the common pattern is to keep baseline PUCT ordering inside a chosen surface/bucket, while changing which surface/bucket gets the next parent pick.
- Tests detected: `10` test functions across untracked test files.

## Added Markers

### Config fields

- `codex_sampling_diversity`
- `sampling_diversity`

### Constants

- `BIRTH_PHASE_SURFACES`
- `_BIRTH_PHASE_SURFACE_COUNTS_KEY`
- `_SAMPLING_DIVERSITIES`

### Classes

- None

### Functions

- `_zero_birth_phase_surface_counts`
- `_sanitize_birth_phase_surface_counts`
- `_safe_int_timestep`
- `classify_birth_phase_surface`
- `_birth_phase_surface`
- `_birth_phase_available_counts`
- `_record_birth_phase_selection`
- `_available_puct_entries`
- `_select_birth_phase_entry`
- `_sample_states_birth_phase_surface`

## Diff Summary

- Worktree tracked shortstat: `4 files changed, 272 insertions(+), 4 deletions(-)`
- Untracked files: `3`

### Worktree Status

````text
 M ttt_discover/codex_utils/discovery.py
 M ttt_discover/codex_utils/sampler.py
 M ttt_discover/discovery.py
 M ttt_discover/rl/codex_no_finetune.py
?? repro/gpu_mode/
?? repro/run_discovery.py
?? tests/
````
### Tracked Worktree Files

````text
M	ttt_discover/codex_utils/discovery.py
M	ttt_discover/codex_utils/sampler.py
M	ttt_discover/discovery.py
M	ttt_discover/rl/codex_no_finetune.py
````
### Untracked Files

- `repro/gpu_mode/run_0608_birth_phase_surface_balance.sh (1067 bytes)`
- `repro/run_discovery.py (4804 bytes)`
- `tests/test_birth_phase_surface_balance.py (11742 bytes)`

### Detected Test Functions

- `tests/test_birth_phase_surface_balance.py::test_disabled_mode_matches_baseline_and_does_not_save_counts`
- `tests/test_birth_phase_surface_balance.py::test_classifier_maps_seed_phases_and_bad_timestep`
- `tests/test_birth_phase_surface_balance.py::test_least_sampled_available_surface_beats_higher_puct`
- `tests/test_birth_phase_surface_balance.py::test_equal_counts_use_surface_order_then_baseline_order`
- `tests/test_birth_phase_surface_balance.py::test_multi_parent_sampling_respects_lineage_blocking`
- `tests/test_birth_phase_surface_balance.py::test_fallback_increments_actual_surface_count`
- `tests/test_birth_phase_surface_balance.py::test_counts_persist_resume_missing_and_bad_values_sanitized`
- `tests/test_birth_phase_surface_balance.py::test_config_and_cli_default_off_and_explicit_reaches_sampler`
- `tests/test_birth_phase_surface_balance.py::test_table_and_metrics_appear_only_when_enabled`
- `tests/test_birth_phase_surface_balance.py::test_classifier_ignores_textual_state_contents`

## Untracked File Snapshots

These files are not part of `git diff` yet, but they are part of the worktree implementation state recorded here.

### `repro/gpu_mode/run_0608_birth_phase_surface_balance.sh`

````bash
#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

MODE="run"
if [[ "${1:-}" == "run" || "${1:-}" == "dry-run" ]]; then
  MODE="$1"
  shift
fi

CODEX_MODEL_NAME="${CODEX_MODEL_NAME:-}"

ARGS=(
  --env examples.gpu_mode.env:GpuModeEnv
  --problem-type trimul
  --experiment-name gpu-mode-0608-birth-phase-surface-balance
  --num-epochs 50
  --group-size 1
  --groups-per-batch 1
  --num-cpus-per-task 1
  --eval-timeout 1200
  --codex-cli-command codex
  --codex-cli-sandbox read-only
  --codex-cli-timeout 600
  --codex-max-concurrent-requests 1
  --codex-sampling-diversity birth_phase_surface
)

if [[ -n "$CODEX_MODEL_NAME" ]]; then
  ARGS+=(--codex-model-name "$CODEX_MODEL_NAME")
fi

if [[ "$MODE" == "dry-run" ]]; then
  python repro/run_discovery.py dry-run "${ARGS[@]}" "$@"
else
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}" \
  TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}" \
  python repro/run_discovery.py "${ARGS[@]}" "$@"
fi
````

### `repro/run_discovery.py`

````python
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_object(spec: str) -> Any:
    if ":" not in spec:
        raise ValueError("Object spec must be formatted as module:attribute")
    module_name, attr_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    obj = module
    for part in attr_name.split("."):
        obj = getattr(obj, part)
    return obj


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Codex no-finetune discovery.")
    parser.add_argument("command", nargs="?", choices=("run", "dry-run"), default="run")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved run configuration and exit.",
    )
    parser.add_argument(
        "--env",
        dest="env_spec",
        help="Environment type as module:attribute, for example examples.gpu_mode.env:GpuModeEnv.",
    )
    parser.add_argument("--problem-type", default="")
    parser.add_argument("--experiment-name", default="codex-discovery")
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--group-size", type=int, default=1)
    parser.add_argument("--groups-per-batch", type=int, default=1)
    parser.add_argument("--num-cpus-per-task", type=int, default=1)
    parser.add_argument("--eval-timeout", type=int, default=45)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--codex-model-name", default=None)
    parser.add_argument("--codex-cli-command", default="codex")
    parser.add_argument(
        "--codex-cli-sandbox",
        choices=("read-only", "workspace-write", "danger-full-access"),
        default="read-only",
    )
    parser.add_argument("--codex-cli-timeout", type=float, default=None)
    parser.add_argument("--codex-max-concurrent-requests", type=int, default=4)
    parser.add_argument("--codex-max-output-tokens", type=int, default=8192)
    parser.add_argument("--codex-temperature", type=float, default=None)
    parser.add_argument("--codex-autonomous", action="store_true")
    parser.add_argument(
        "--codex-initial-program",
        action="append",
        default=None,
        help="Seed program path. May be repeated.",
    )
    parser.add_argument(
        "--codex-initial-pool",
        action="append",
        default=None,
        help="Initial-pool JSON path. May be repeated.",
    )
    parser.add_argument(
        "--codex-sampling-diversity",
        choices=("off", "birth_phase_surface"),
        default="off",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dry_run = args.dry_run or args.command == "dry-run"

    config_preview = {
        "runner": "codex_no_finetune",
        "env": args.env_spec,
        "problem_type": args.problem_type,
        "experiment_name": args.experiment_name,
        "num_epochs": args.num_epochs,
        "group_size": args.group_size,
        "groups_per_batch": args.groups_per_batch,
        "codex_sampling_diversity": args.codex_sampling_diversity,
    }
    if dry_run:
        print(json.dumps(config_preview, indent=2, sort_keys=True))
        return

    if not args.env_spec:
        raise SystemExit("--env is required unless using dry-run")

    from ttt_discover.codex_utils.discovery import DiscoverConfig, discover

    discover(
        DiscoverConfig(
            runner="codex_no_finetune",
            env_type=_load_object(args.env_spec),
            problem_type=args.problem_type,
            experiment_name=args.experiment_name,
            wandb_project=args.wandb_project,
            num_epochs=args.num_epochs,
            group_size=args.group_size,
            groups_per_batch=args.groups_per_batch,
            num_cpus_per_task=args.num_cpus_per_task,
            eval_timeout=args.eval_timeout,
            codex_model_name=args.codex_model_name,
            codex_max_output_tokens=args.codex_max_output_tokens,
            codex_temperature=args.codex_temperature,
            codex_cli_command=args.codex_cli_command,
            codex_cli_sandbox=args.codex_cli_sandbox,
            codex_cli_timeout=args.codex_cli_timeout,
            codex_max_concurrent_requests=args.codex_max_concurrent_requests,
            codex_initial_program_paths=tuple(args.codex_initial_program or ()),
            codex_initial_pool_paths=tuple(args.codex_initial_pool or ()),
            codex_autonomous=args.codex_autonomous,
            codex_sampling_diversity=args.codex_sampling_diversity,
        )
    )


if __name__ == "__main__":
    main()
````

### `tests/test_birth_phase_surface_balance.py`

````python
from __future__ import annotations

import inspect
import json
import subprocess
import sys
from pathlib import Path

from ttt_discover.codex_utils.runtime import State
from ttt_discover.codex_utils.sampler import (
    BIRTH_PHASE_SURFACES,
    PUCTSampler,
    classify_birth_phase_surface,
)


class TinyEnv:
    state_type = State
    _counter = 0

    @classmethod
    def create_initial_state(cls, problem_type: str) -> State:
        cls._counter += 1
        return State(
            timestep=-1,
            construction=[],
            code="",
            value=0.0,
            id=f"initial-{cls._counter}",
        )


def make_state(
    state_id: str,
    timestep,
    value: float,
    *,
    parents: list[dict] | None = None,
    code: str | None = None,
    construction: list | None = None,
    observation: str = "",
) -> State:
    return State(
        timestep=timestep,
        construction=construction if construction is not None else [state_id],
        code=code if code is not None else f"code-{state_id}",
        value=value,
        parents=parents or [],
        id=state_id,
        observation=observation,
    )


def make_sampler(
    tmp_path: Path,
    *,
    sampling_diversity: str = "off",
    puct_c: float = 0.0,
    name: str = "puct_sampler.json",
) -> PUCTSampler:
    return PUCTSampler(
        file_path=str(tmp_path / name),
        env_type=TinyEnv,
        batch_size=0,
        puct_c=puct_c,
        topk_children=0,
        sampling_diversity=sampling_diversity,
    )


def sampler_step_path(file_path: Path, step: int) -> Path:
    return file_path.with_name(f"{file_path.stem}_step_{step:06d}.json")


def set_pool(
    sampler: PUCTSampler,
    states: list[State],
    *,
    initial_states: list[State] | None = None,
) -> None:
    sampler._states = states
    sampler._initial_states = initial_states or []


def test_disabled_mode_matches_baseline_and_does_not_save_counts(tmp_path: Path) -> None:
    root = make_state("root", 0, 10.0)
    sibling = make_state("sibling", 2, 8.0)
    child = make_state("child", 1, 9.0, parents=[{"id": "root", "timestep": 0}])
    states = [root, child, sibling]

    default_sampler = make_sampler(tmp_path, name="default.json")
    explicit_off_sampler = make_sampler(tmp_path, sampling_diversity="off", name="off.json")
    set_pool(default_sampler, list(states))
    set_pool(explicit_off_sampler, list(states))

    assert [s.id for s in default_sampler.sample_states(2)] == ["root", "sibling"]
    assert [s.id for s in explicit_off_sampler.sample_states(2)] == ["root", "sibling"]

    explicit_off_sampler.flush(step=1)
    saved = json.loads(sampler_step_path(tmp_path / "off.json", 1).read_text())
    assert "puct_birth_phase_surface_sample_counts" not in saved
    assert explicit_off_sampler._birth_phase_surface_sample_counts == {
        surface: 0 for surface in BIRTH_PHASE_SURFACES
    }


def test_classifier_maps_seed_phases_and_bad_timestep() -> None:
    assert classify_birth_phase_surface(make_state("a", -1, 0), set()) == "seed"
    assert classify_birth_phase_surface(make_state("b", 0, 0), set()) == "phase_0"
    assert classify_birth_phase_surface(make_state("c", 1, 0), set()) == "phase_1"
    assert classify_birth_phase_surface(make_state("d", 4, 0), set()) == "phase_0"
    assert classify_birth_phase_surface(make_state("e", "bad", 0), set()) == "unknown"
    assert classify_birth_phase_surface(make_state("initial", 99, 0), {"initial"}) == "seed"


def test_least_sampled_available_surface_beats_higher_puct(tmp_path: Path) -> None:
    seed = make_state("seed", -1, 100.0)
    phase0 = make_state("phase0", 0, 1.0)
    sampler = make_sampler(tmp_path, sampling_diversity="birth_phase_surface")
    set_pool(sampler, [seed, phase0], initial_states=[seed])
    sampler._birth_phase_surface_sample_counts["seed"] = 10

    picked = sampler.sample_states(1)

    assert [s.id for s in picked] == ["phase0"]
    assert sampler._birth_phase_surface_sample_counts["phase_0"] == 1
    assert sampler._birth_phase_surface_sample_counts["seed"] == 10


def test_equal_counts_use_surface_order_then_baseline_order(tmp_path: Path) -> None:
    phase1 = make_state("phase1", 1, 100.0)
    phase0_best = make_state("phase0-best", 0, 10.0)
    phase0_second = make_state("phase0-second", 4, 5.0)
    sampler = make_sampler(tmp_path, sampling_diversity="birth_phase_surface")
    set_pool(sampler, [phase1, phase0_second, phase0_best])

    picked = sampler.sample_states(1)

    assert [s.id for s in picked] == ["phase0-best"]


def test_multi_parent_sampling_respects_lineage_blocking(tmp_path: Path) -> None:
    root = make_state("root", 0, 10.0)
    child = make_state("child", 1, 9.0, parents=[{"id": "root", "timestep": 0}])
    sibling = make_state("sibling", 2, 8.0)
    sampler = make_sampler(tmp_path, sampling_diversity="birth_phase_surface")
    set_pool(sampler, [root, child, sibling])

    picked = sampler.sample_states(2)

    assert [s.id for s in picked] == ["root", "sibling"]


def test_fallback_increments_actual_surface_count(tmp_path: Path) -> None:
    phase2 = make_state("phase2", 2, 3.0)
    sampler = make_sampler(tmp_path, sampling_diversity="birth_phase_surface")
    set_pool(sampler, [phase2])
    sampler._birth_phase_surface_sample_counts["phase_2"] = 4
    sampler._select_birth_phase_entry = lambda available, initial_ids: None

    picked = sampler.sample_states(1)

    assert [s.id for s in picked] == ["phase2"]
    assert sampler._birth_phase_surface_sample_counts["phase_2"] == 5
    assert sampler._last_birth_phase_surface_rows == [("phase_2", 4, 5, "fallback")]


def test_counts_persist_resume_missing_and_bad_values_sanitized(tmp_path: Path) -> None:
    file_path = tmp_path / "persist.json"
    sampler = make_sampler(
        tmp_path,
        sampling_diversity="birth_phase_surface",
        name=file_path.name,
    )
    phase0 = make_state("phase0", 0, 1.0)
    set_pool(sampler, [phase0])
    sampler.sample_states(1)
    sampler.flush(step=1)

    saved = json.loads(sampler_step_path(file_path, 1).read_text())
    assert saved["puct_birth_phase_surface_sample_counts"]["phase_0"] == 1

    resumed = PUCTSampler(
        file_path=str(file_path),
        env_type=TinyEnv,
        batch_size=0,
        resume_step=1,
        topk_children=0,
        sampling_diversity="birth_phase_surface",
    )
    assert resumed._birth_phase_surface_sample_counts["phase_0"] == 1

    missing_path = tmp_path / "missing.json"
    sampler_step_path(missing_path, 2).write_text(
        json.dumps({"states": [], "initial_states": [], "puct_n": {}, "puct_m": {}, "puct_T": 0})
    )
    missing = PUCTSampler(
        file_path=str(missing_path),
        env_type=TinyEnv,
        batch_size=0,
        resume_step=2,
        topk_children=0,
        sampling_diversity="birth_phase_surface",
    )
    assert missing._birth_phase_surface_sample_counts == {
        surface: 0 for surface in BIRTH_PHASE_SURFACES
    }

    bad_path = tmp_path / "bad.json"
    sampler_step_path(bad_path, 3).write_text(
        json.dumps(
            {
                "states": [],
                "initial_states": [],
                "puct_n": {},
                "puct_m": {},
                "puct_T": 0,
                "puct_birth_phase_surface_sample_counts": {
                    "seed": 0,
                    "phase_0": 5,
                    "phase_1": True,
                    "phase_2": -1,
                    "phase_3": "7",
                    "unknown": 2,
                    "extra": 99,
                },
            }
        )
    )
    bad = PUCTSampler(
        file_path=str(bad_path),
        env_type=TinyEnv,
        batch_size=0,
        resume_step=3,
        topk_children=0,
        sampling_diversity="birth_phase_surface",
    )
    assert bad._birth_phase_surface_sample_counts == {
        "seed": 0,
        "phase_0": 5,
        "phase_1": 0,
        "phase_2": 0,
        "phase_3": 0,
        "unknown": 2,
    }


def test_config_and_cli_default_off_and_explicit_reaches_sampler(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    default_cli = subprocess.run(
        [sys.executable, "repro/run_discovery.py", "dry-run"],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    )
    assert json.loads(default_cli.stdout)["codex_sampling_diversity"] == "off"

    explicit_cli = subprocess.run(
        [
            sys.executable,
            "repro/run_discovery.py",
            "dry-run",
            "--codex-sampling-diversity",
            "birth_phase_surface",
        ],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    )
    assert json.loads(explicit_cli.stdout)["codex_sampling_diversity"] == "birth_phase_surface"

    from ttt_discover.rl.codex_no_finetune import CodexNoFinetuneConfig, _build_sampler

    cfg = CodexNoFinetuneConfig(
        env_type=TinyEnv,
        log_path=str(tmp_path / "logs"),
        sampling_diversity="birth_phase_surface",
    )
    sampler = _build_sampler(cfg, start_batch=0)
    assert isinstance(sampler, PUCTSampler)
    assert sampler.sampling_diversity == "birth_phase_surface"

    codex_discovery_source = (
        repo_root / "ttt_discover/codex_utils/discovery.py"
    ).read_text(encoding="utf-8")
    public_discovery_source = (repo_root / "ttt_discover/discovery.py").read_text(
        encoding="utf-8"
    )
    for source in (codex_discovery_source, public_discovery_source):
        assert (
            'codex_sampling_diversity: Literal["off", "birth_phase_surface"] = "off"'
            in source
        )
        assert "sampling_diversity=config.codex_sampling_diversity" in source


def test_table_and_metrics_appear_only_when_enabled(tmp_path: Path) -> None:
    state = make_state("phase0", 0, 1.0)

    off = make_sampler(tmp_path, sampling_diversity="off", name="off-table.json")
    set_pool(off, [state])
    off.sample_states(1)
    off_metrics = off.get_sample_stats()
    off_columns, _ = off.get_sample_table()
    assert not any("birth_phase_surface" in key for key in off_metrics)
    assert "birth_phase_surface" not in off_columns

    enabled = make_sampler(
        tmp_path,
        sampling_diversity="birth_phase_surface",
        name="enabled-table.json",
    )
    set_pool(enabled, [state])
    enabled.sample_states(1)
    enabled_metrics = enabled.get_sample_stats()
    enabled_columns, enabled_rows = enabled.get_sample_table()
    assert enabled_metrics["puct/birth_phase_surface/enabled"] == 1
    assert enabled_metrics["puct/birth_phase_surface/count_phase_0"] == 1
    assert enabled_metrics["puct/birth_phase_surface/selected_phase_0"] == 1
    assert enabled_metrics["puct/birth_phase_surface/available_phase_0"] == 1
    assert "birth_phase_surface" in enabled_columns
    assert enabled_rows[0][-4:] == ("phase_0", 0, 1, "balanced")


def test_classifier_ignores_textual_state_contents() -> None:
    first = make_state(
        "same",
        3,
        1.0,
        parents=[{"id": "parent-a", "timestep": 2}],
        code="code A",
        construction=["construction A"],
        observation="observation A",
    )
    second = make_state(
        "same",
        3,
        1.0,
        parents=[{"id": "parent-b", "timestep": 2}],
        code="code B",
        construction=["construction B"],
        observation="observation B",
    )
    assert classify_birth_phase_surface(first, set()) == classify_birth_phase_surface(second, set())

    source = inspect.getsource(classify_birth_phase_surface)
    for banned in ("code", "observation", "parents", "construction"):
        assert banned not in source
````

## Diff Stat

### Worktree Tracked Diff Stat

````text
 ttt_discover/codex_utils/discovery.py |   2 +
 ttt_discover/codex_utils/sampler.py   | 270 +++++++++++++++++++++++++++++++++-
 ttt_discover/discovery.py             |   2 +
 ttt_discover/rl/codex_no_finetune.py  |   2 +
 4 files changed, 272 insertions(+), 4 deletions(-)
````
## Raw Diff

<details>
<summary>Tracked worktree patch</summary>

````diff
diff --git a/ttt_discover/codex_utils/discovery.py b/ttt_discover/codex_utils/discovery.py
index b21a80f..280b7be 100644
--- a/ttt_discover/codex_utils/discovery.py
+++ b/ttt_discover/codex_utils/discovery.py
@@ -50,6 +50,7 @@ class DiscoverConfig:
     codex_initial_program_paths: tuple[str, ...] = ()
     codex_initial_pool_paths: tuple[str, ...] = ()
     codex_autonomous: bool = False
+    codex_sampling_diversity: Literal["off", "birth_phase_surface"] = "off"
 
 
 def _run_codex_no_finetune(config: DiscoverConfig) -> None:
@@ -85,6 +86,7 @@ def _run_codex_no_finetune(config: DiscoverConfig) -> None:
         initial_program_paths=config.codex_initial_program_paths,
         initial_pool_paths=config.codex_initial_pool_paths,
         autonomous=config.codex_autonomous,
+        sampling_diversity=config.codex_sampling_diversity,
         wandb_project=config.wandb_project,
         wandb_name=experiment_name,
         log_path=log_path,
diff --git a/ttt_discover/codex_utils/sampler.py b/ttt_discover/codex_utils/sampler.py
index 5c5d4d1..b682c0f 100644
--- a/ttt_discover/codex_utils/sampler.py
+++ b/ttt_discover/codex_utils/sampler.py
@@ -8,7 +8,7 @@ import os
 from pathlib import Path
 import threading
 import time
-from typing import Any, Callable
+from typing import Any, Callable, Literal
 
 import numpy as np
 
@@ -16,6 +16,62 @@ from ttt_discover.codex_utils.runtime import State, state_from_dict, to_json_ser
 
 logger = logging.getLogger(__name__)
 
+BIRTH_PHASE_SURFACES = (
+    "seed",
+    "phase_0",
+    "phase_1",
+    "phase_2",
+    "phase_3",
+    "unknown",
+)
+_BIRTH_PHASE_SURFACE_COUNTS_KEY = "puct_birth_phase_surface_sample_counts"
+_SAMPLING_DIVERSITIES = ("off", "birth_phase_surface")
+
+
+def _zero_birth_phase_surface_counts() -> dict[str, int]:
+    return {surface: 0 for surface in BIRTH_PHASE_SURFACES}
+
+
+def _sanitize_birth_phase_surface_counts(raw: Any) -> dict[str, int]:
+    counts = _zero_birth_phase_surface_counts()
+    if not isinstance(raw, dict):
+        return counts
+    for surface in BIRTH_PHASE_SURFACES:
+        value = raw.get(surface, 0)
+        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
+            counts[surface] = 0
+        else:
+            counts[surface] = int(value)
+    return counts
+
+
+def _safe_int_timestep(state: Any) -> int | None:
+    if not hasattr(state, "timestep"):
+        return None
+    value = getattr(state, "timestep")
+    if isinstance(value, bool):
+        return None
+    if isinstance(value, (int, np.integer)):
+        return int(value)
+    if isinstance(value, (float, np.floating)):
+        if not float(value).is_integer():
+            return None
+        return int(value)
+    try:
+        return int(value)
+    except (TypeError, ValueError, OverflowError):
+        return None
+
+
+def classify_birth_phase_surface(state: Any, initial_ids: set[str]) -> str:
+    timestep = _safe_int_timestep(state)
+    state_id = getattr(state, "id", None)
+    if state_id in initial_ids or (timestep is not None and timestep < 0):
+        return "seed"
+    if timestep is not None and timestep >= 0:
+        return f"phase_{timestep % 4}"
+    return "unknown"
+
 
 @contextmanager
 def _file_lock(lock_path: str, *, poll_s: float = 0.05, stale_s: float = 600.0):
@@ -353,7 +409,10 @@ class PUCTSampler(StateSampler):
         resume_step: int | None = None,
         puct_c: float = 1.0,
         topk_children: int = 2,
+        sampling_diversity: Literal["off", "birth_phase_surface"] = "off",
     ):
+        if sampling_diversity not in _SAMPLING_DIVERSITIES:
+            raise ValueError(f"Unknown sampling_diversity: {sampling_diversity}")
         self.file_path = file_path
         self.env_type = env_type
         self.problem_type = problem_type
@@ -361,6 +420,7 @@ class PUCTSampler(StateSampler):
         self.batch_size = batch_size
         self.topk_children = topk_children
         self.puct_c = float(puct_c)
+        self.sampling_diversity = sampling_diversity
         
         self._states: list[State] = []
         self._initial_states: list[State] = []
@@ -375,6 +435,10 @@ class PUCTSampler(StateSampler):
         self._T: int = 0
         self._last_scale: float = 1.0
         self._last_puct_stats: list[tuple[int, float, float, float, float]] = []
+        self._birth_phase_surface_sample_counts = _zero_birth_phase_surface_counts()
+        self._last_birth_phase_surface_rows: list[tuple[str, int, int, str]] = []
+        self._last_birth_phase_surface_selected = _zero_birth_phase_surface_counts()
+        self._last_birth_phase_surface_available = _zero_birth_phase_surface_counts()
         
         if resume_step is not None:
             self._load(resume_step)
@@ -399,6 +463,12 @@ class PUCTSampler(StateSampler):
         self._n = store.get("puct_n", {}) or {}
         self._m = store.get("puct_m", {}) or {}
         self._T = int(store.get("puct_T", 0) or 0)
+        if self.sampling_diversity == "birth_phase_surface":
+            self._birth_phase_surface_sample_counts = (
+                _sanitize_birth_phase_surface_counts(
+                    store.get(_BIRTH_PHASE_SURFACE_COUNTS_KEY, {})
+                )
+            )
 
     def _save(self, step: int):
         save_path = _sampler_file_for_step(self.file_path, step)
@@ -411,6 +481,10 @@ class PUCTSampler(StateSampler):
             "puct_m": self._m,
             "puct_T": self._T,
         }
+        if self.sampling_diversity == "birth_phase_surface":
+            store[_BIRTH_PHASE_SURFACE_COUNTS_KEY] = (
+                self._birth_phase_surface_sample_counts
+            )
         with _file_lock(f"{save_path}.lock"):
             _atomic_write_json(save_path, store)
 
@@ -489,6 +563,134 @@ class PUCTSampler(StateSampler):
                     queue.append(child_id)
         return lineage
 
+    def _birth_phase_surface(self, state: State, initial_ids: set[str]) -> str:
+        return classify_birth_phase_surface(state, initial_ids)
+
+    def _birth_phase_available_counts(
+        self,
+        entries: list[tuple[float, float, State, int, float, float, float]],
+        initial_ids: set[str],
+    ) -> dict[str, int]:
+        counts = _zero_birth_phase_surface_counts()
+        for entry in entries:
+            counts[self._birth_phase_surface(entry[2], initial_ids)] += 1
+        return counts
+
+    def _record_birth_phase_selection(
+        self,
+        entry: tuple[float, float, State, int, float, float, float],
+        initial_ids: set[str],
+        selection: str,
+    ) -> None:
+        surface = self._birth_phase_surface(entry[2], initial_ids)
+        before = self._birth_phase_surface_sample_counts[surface]
+        after = before + 1
+        self._birth_phase_surface_sample_counts[surface] = after
+        self._last_birth_phase_surface_selected[surface] += 1
+        self._last_birth_phase_surface_rows.append((surface, before, after, selection))
+
+    def _available_puct_entries(
+        self,
+        entries: list[tuple[float, float, State, int, float, float, float]],
+        *,
+        picked_ids: set[str],
+        blocked_ids: set[str],
+        use_lineage_blocking: bool,
+    ) -> list[tuple[float, float, State, int, float, float, float]]:
+        available = []
+        for entry in entries:
+            state = entry[2]
+            if state.id in picked_ids:
+                continue
+            if use_lineage_blocking and state.id in blocked_ids:
+                continue
+            available.append(entry)
+        return available
+
+    def _select_birth_phase_entry(
+        self,
+        available: list[tuple[float, float, State, int, float, float, float]],
+        initial_ids: set[str],
+    ) -> tuple[float, float, State, int, float, float, float] | None:
+        grouped: dict[str, list[tuple[float, float, State, int, float, float, float]]] = {
+            surface: [] for surface in BIRTH_PHASE_SURFACES
+        }
+        for entry in available:
+            grouped[self._birth_phase_surface(entry[2], initial_ids)].append(entry)
+        surfaces = [surface for surface in BIRTH_PHASE_SURFACES if grouped[surface]]
+        if not surfaces:
+            return None
+        surface = min(
+            surfaces,
+            key=lambda item: self._birth_phase_surface_sample_counts[item],
+        )
+        return grouped[surface][0]
+
+    def _sample_states_birth_phase_surface(
+        self,
+        scores: list[tuple[float, float, State, int, float, float, float]],
+        initial_ids: set[str],
+        num_states: int,
+    ) -> tuple[
+        list[State],
+        list[tuple[float, float, State, int, float, float, float]],
+    ]:
+        self._last_birth_phase_surface_rows = []
+        self._last_birth_phase_surface_selected = _zero_birth_phase_surface_counts()
+        use_lineage_blocking = num_states > 1
+        children_map = self._build_children_map() if use_lineage_blocking else {}
+        picked: list[State] = []
+        top_scores: list[tuple[float, float, State, int, float, float, float]] = []
+        picked_ids: set[str] = set()
+        blocked_ids: set[str] = set()
+
+        initial_available = self._available_puct_entries(
+            scores,
+            picked_ids=picked_ids,
+            blocked_ids=blocked_ids,
+            use_lineage_blocking=use_lineage_blocking,
+        )
+        self._last_birth_phase_surface_available = self._birth_phase_available_counts(
+            initial_available,
+            initial_ids,
+        )
+
+        while len(picked) < num_states:
+            available = self._available_puct_entries(
+                scores,
+                picked_ids=picked_ids,
+                blocked_ids=blocked_ids,
+                use_lineage_blocking=use_lineage_blocking,
+            )
+            entry = self._select_birth_phase_entry(available, initial_ids)
+            if entry is None:
+                break
+            state = entry[2]
+            picked.append(state)
+            top_scores.append(entry)
+            picked_ids.add(state.id)
+            self._record_birth_phase_selection(entry, initial_ids, "balanced")
+            if use_lineage_blocking:
+                blocked_ids.update(self._get_full_lineage(state, children_map))
+
+        if len(picked) < num_states:
+            for entry in scores:
+                state = entry[2]
+                if state.id in picked_ids:
+                    continue
+                if use_lineage_blocking and state.id in blocked_ids:
+                    continue
+                picked.append(state)
+                top_scores.append(entry)
+                picked_ids.add(state.id)
+                self._record_birth_phase_selection(entry, initial_ids, "fallback")
+                if use_lineage_blocking:
+                    blocked_ids.update(self._get_full_lineage(state, children_map))
+                if len(picked) >= num_states:
+                    break
+
+        return picked, top_scores
+
     def sample_states(self, num_states: int) -> list[State]:
         initial_ids = {s.id for s in self._initial_states}
         candidates = list(self._states)
@@ -501,6 +703,10 @@ class PUCTSampler(StateSampler):
             self._last_sampled_states = picked
             self._last_sampled_indices = []
             self._last_puct_stats = [(0, 0.0, 0.0, 0.0, 0.0) for _ in picked]
+            if self.sampling_diversity == "birth_phase_surface":
+                self._last_birth_phase_surface_rows = []
+                self._last_birth_phase_surface_selected = _zero_birth_phase_surface_counts()
+                self._last_birth_phase_surface_available = _zero_birth_phase_surface_counts()
             return picked
 
         vals = np.array([float(s.value if s.value is not None else float("-inf")) for s in candidates])
@@ -521,7 +727,13 @@ class PUCTSampler(StateSampler):
 
         scores.sort(key=lambda x: (x[0], x[1]), reverse=True)
 
-        if num_states > 1:
+        if self.sampling_diversity == "birth_phase_surface":
+            picked, top_scores = self._sample_states_birth_phase_surface(
+                scores,
+                initial_ids,
+                num_states,
+            )
+        elif num_states > 1:
             children_map = self._build_children_map()
             picked, top_scores, blocked_ids = [], [], set()
             for entry in scores:
@@ -731,21 +943,67 @@ class PUCTSampler(StateSampler):
         stats.update(_stats(sampled_values, "puct/sampled_value"))
         stats.update(_stats(sampled_timesteps, "puct/sampled_timestep"))
         stats.update(_stats(sampled_constr_lens, "puct/sampled_construction_len"))
+        if self.sampling_diversity == "birth_phase_surface":
+            stats["puct/birth_phase_surface/enabled"] = 1
+            for surface in BIRTH_PHASE_SURFACES:
+                stats[f"puct/birth_phase_surface/count_{surface}"] = (
+                    self._birth_phase_surface_sample_counts[surface]
+                )
+                stats[f"puct/birth_phase_surface/selected_{surface}"] = (
+                    self._last_birth_phase_surface_selected[surface]
+                )
+                stats[f"puct/birth_phase_surface/available_{surface}"] = (
+                    self._last_birth_phase_surface_available[surface]
+                )
         return stats
 
     def get_sample_table(self) -> tuple[list[str], list[tuple]]:
         columns = ["buffer_idx", "timestep", "value", "terminal_value", "parent_value", "construction_len", "observation_len", "n", "Q", "P", "bonus", "score"]
+        if self.sampling_diversity == "birth_phase_surface":
+            columns = columns + [
+                "birth_phase_surface",
+                "birth_phase_surface_count_before",
+                "birth_phase_surface_count_after",
+                "birth_phase_surface_selection",
+            ]
         rows = []
         if not self._last_sampled_states:
             return columns, rows
         indices = self._last_sampled_indices if len(self._last_sampled_indices) == len(self._last_sampled_states) else [-1] * len(self._last_sampled_states)
         stats = self._last_puct_stats if len(self._last_puct_stats) == len(self._last_sampled_states) else [(0, 0.0, 0.0, 0.0, 0.0)] * len(self._last_sampled_states)
-        for idx, state, (n, Q, P, bonus, score) in zip(indices, self._last_sampled_states, stats):
+        birth_rows = (
+            self._last_birth_phase_surface_rows
+            if self.sampling_diversity == "birth_phase_surface"
+            and len(self._last_birth_phase_surface_rows) == len(self._last_sampled_states)
+            else [("unknown", 0, 0, "")] * len(self._last_sampled_states)
+        )
+        for idx, state, (n, Q, P, bonus, score), birth_row in zip(
+            indices,
+            self._last_sampled_states,
+            stats,
+            birth_rows,
+        ):
             parent_val = state.parent_values[0] if state.parent_values else None
             constr = getattr(state, 'construction', None)
             constr_len = len(constr) if constr is not None else 0
             obs_len = len(state.observation) if state.observation else 0
-            rows.append((idx, state.timestep, state.value, 0, parent_val, constr_len, obs_len, n, Q, P, bonus, score))
+            row = (
+                idx,
+                state.timestep,
+                state.value,
+                0,
+                parent_val,
+                constr_len,
+                obs_len,
+                n,
+                Q,
+                P,
+                bonus,
+                score,
+            )
+            if self.sampling_diversity == "birth_phase_surface":
+                row = row + birth_row
+            rows.append(row)
         return columns, rows
 
 
@@ -756,6 +1014,7 @@ def create_sampler(
     batch_size: int = 1,
     resume_step: int | None = None,
     topk_children: int = 2,
+    sampling_diversity: Literal["off", "birth_phase_surface"] = "off",
 ) -> StateSampler:
     """Factory function to create samplers. Pass the env type (from config.env_type)."""
     if not log_path:
@@ -768,6 +1027,7 @@ def create_sampler(
         batch_size=batch_size,
         resume_step=resume_step,
         topk_children=topk_children,
+        sampling_diversity=sampling_diversity,
     )
 
 
@@ -778,6 +1038,7 @@ def get_or_create_sampler_with_default(
     batch_size: int = 1,
     resume_step: int | None = None,
     topk_children: int = 2,
+    sampling_diversity: Literal["off", "birth_phase_surface"] = "off",
 ) -> StateSampler:
     """Get sampler. Initial experience is created via env_type.create_initial_state."""
     return create_sampler(
@@ -787,4 +1048,5 @@ def get_or_create_sampler_with_default(
         batch_size=batch_size,
         resume_step=resume_step,
         topk_children=topk_children,
+        sampling_diversity=sampling_diversity,
     )
diff --git a/ttt_discover/discovery.py b/ttt_discover/discovery.py
index 6cdb434..da91735 100644
--- a/ttt_discover/discovery.py
+++ b/ttt_discover/discovery.py
@@ -63,6 +63,7 @@ class DiscoverConfig:
     codex_initial_program_paths: tuple[str, ...] = ()
     codex_initial_pool_paths: tuple[str, ...] = ()
     codex_autonomous: bool = False
+    codex_sampling_diversity: Literal["off", "birth_phase_surface"] = "off"
 
 
 def init_ray(num_cpus_per_task: int, env_type: str):
@@ -146,6 +147,7 @@ async def discover_impl(config: DiscoverConfig):
             initial_program_paths=config.codex_initial_program_paths,
             initial_pool_paths=config.codex_initial_pool_paths,
             autonomous=config.codex_autonomous,
+            sampling_diversity=config.codex_sampling_diversity,
             wandb_project=config.wandb_project,
             wandb_name=config.experiment_name,
             log_path=log_path,
diff --git a/ttt_discover/rl/codex_no_finetune.py b/ttt_discover/rl/codex_no_finetune.py
index 18a0e9e..1aeb4cf 100644
--- a/ttt_discover/rl/codex_no_finetune.py
+++ b/ttt_discover/rl/codex_no_finetune.py
@@ -137,6 +137,7 @@ class CodexNoFinetuneConfig:
     initial_program_paths: tuple[str, ...] = ()
     initial_pool_paths: tuple[str, ...] = ()
     topk_children: int = 16
+    sampling_diversity: Literal["off", "birth_phase_surface"] = "off"
 
     # False is TTT Discover: non-auto, many short samples. True is AutoEvolve:
     # one Codex deep-dive workspace per sample with file edits and shell access.
@@ -960,6 +961,7 @@ def _build_sampler(cfg: CodexNoFinetuneConfig, start_batch: int) -> StateSampler
         batch_size=cfg.groups_per_batch,
         resume_step=start_batch if start_batch > 0 else None,
         topk_children=cfg.topk_children,
+        sampling_diversity=cfg.sampling_diversity,
     )
 
 
````
</details>

