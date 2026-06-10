# codex/diversity-archive-slot-surface-balance

## Summary

将 archive buffer index 映射到固定 slot surface，并在 parent_surface_mode=archive_slot_surface 下平衡不同 slot；保留 PUCT fallback 标记。

## Branch State

- Worktree: `/opt/tiger/discover-archive-slot-surface-balance`
- HEAD: `2234282`
- Base used for comparison: `2234282`
- Commits ahead of base: `0`
- Commits behind base: `0`
- Group: `completed_19`
- Implementation location: `uncommitted worktree diff`

## Implemented Behavior

- The summary above is based on reading the changed Python code, added config fields, and sampler/prompt hooks in this branch.
- For sampler branches, the common pattern is to keep baseline PUCT ordering inside a chosen surface/bucket, while changing which surface/bucket gets the next parent pick.
- Tests detected: `8` test functions across untracked test files.

## Added Markers

### Config fields

- `codex_parent_surface_mode`
- `parent_surface_mode`

### Constants

- `ARCHIVE_SLOT_SURFACES`
- `ARCHIVE_SLOT_FALLBACK_SURFACE`
- `_ARCHIVE_SLOT_SURFACE_COUNT_KEY`
- `_PARENT_SURFACE_MODES`

### Classes

- None

### Functions

- `_validate_parent_surface_mode`
- `_empty_archive_slot_surface_counts`
- `_sanitize_archive_slot_surface_counts`
- `archive_slot_surface_for_buffer_idx`
- `_buffer_index_maps`
- `_buffer_idx_for_state`
- `_archive_slot_surface_for_state`
- `_select_balanced_archive_slot_entry`
- `_record_archive_slot_surface_pick`
- `_sample_states_archive_slot_surface`
- `append_entry`

## Diff Summary

- Worktree tracked shortstat: `4 files changed, 281 insertions(+), 3 deletions(-)`
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

- `repro/gpu_mode/run_0609_archive_slot_surface_balance.sh (485 bytes)`
- `repro/run_discovery.py (4443 bytes)`
- `tests/test_codex_archive_slot_surface_balance.py (12783 bytes)`

### Detected Test Functions

- `tests/test_codex_archive_slot_surface_balance.py::test_disabled_mode_matches_default_and_has_no_surface_outputs`
- `tests/test_codex_archive_slot_surface_balance.py::test_classifier_uses_archive_index_modulo_only`
- `tests/test_codex_archive_slot_surface_balance.py::test_least_sampled_surface_beats_higher_puct_and_keeps_order_within_surface`
- `tests/test_codex_archive_slot_surface_balance.py::test_equal_counts_use_surface_order_then_puct_order_within_surface`
- `tests/test_codex_archive_slot_surface_balance.py::test_multi_parent_updates_counts_and_preserves_lineage_blocking`
- `tests/test_codex_archive_slot_surface_balance.py::test_fallback_increments_fallback_count_and_respects_lineage_blocking`
- `tests/test_codex_archive_slot_surface_balance.py::test_save_resume_restores_counts_and_sanitizes_missing_bad_values`
- `tests/test_codex_archive_slot_surface_balance.py::test_config_cli_plumbing_and_default_disabled`

## Untracked File Snapshots

These files are not part of `git diff` yet, but they are part of the worktree implementation state recorded here.

### `repro/gpu_mode/run_0609_archive_slot_surface_balance.sh`

````bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${REPO_ROOT}"

MODE="run"
if [[ "${1:-}" == "run" || "${1:-}" == "dry-run" ]]; then
  MODE="$1"
  shift
fi

EXPERIMENT_NAME="${EXPERIMENT_NAME:-gpu-mode-0609-archive-slot-surface-balance}"

python repro/run_discovery.py "${MODE}" \
  --experiment-name "${EXPERIMENT_NAME}" \
  --codex-parent-surface-mode archive_slot_surface \
  "$@"
````

### `repro/run_discovery.py`

````python
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a Codex no-finetune GPUMode discovery job."
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=("run", "dry-run"),
        default="run",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--problem-type", default="trimul")
    parser.add_argument(
        "--experiment-name",
        default="gpu-mode-archive-slot-surface-balance",
    )
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT"))
    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--groups-per-batch", type=int, default=1)
    parser.add_argument("--group-size", type=int, default=1)
    parser.add_argument("--num-cpus-per-task", type=int, default=1)
    parser.add_argument("--eval-timeout", type=int, default=1200)
    parser.add_argument(
        "--codex-backend",
        choices=("cli", "responses"),
        default="cli",
    )
    parser.add_argument("--codex-model-name", default=None)
    parser.add_argument("--codex-max-output-tokens", type=int, default=8192)
    parser.add_argument("--codex-temperature", type=float, default=None)
    parser.add_argument("--codex-cli-command", default="codex")
    parser.add_argument(
        "--codex-cli-sandbox",
        choices=("read-only", "workspace-write", "danger-full-access"),
        default="read-only",
    )
    parser.add_argument("--codex-cli-timeout", type=float, default=600.0)
    parser.add_argument("--codex-max-concurrent-requests", type=int, default=1)
    parser.add_argument(
        "--codex-parent-surface-mode",
        choices=("none", "archive_slot_surface"),
        default="none",
    )
    return parser.parse_args()


def _summary(args: argparse.Namespace) -> dict[str, object]:
    return {
        "runner": "codex_no_finetune",
        "env_type": "examples.gpu_mode.env.GpuModeEnv",
        "problem_type": args.problem_type,
        "experiment_name": args.experiment_name,
        "wandb_project": args.wandb_project,
        "num_epochs": args.num_epochs,
        "groups_per_batch": args.groups_per_batch,
        "group_size": args.group_size,
        "num_cpus_per_task": args.num_cpus_per_task,
        "eval_timeout": args.eval_timeout,
        "codex_backend": args.codex_backend,
        "codex_model_name": args.codex_model_name,
        "codex_max_output_tokens": args.codex_max_output_tokens,
        "codex_temperature": args.codex_temperature,
        "codex_cli_command": args.codex_cli_command,
        "codex_cli_sandbox": args.codex_cli_sandbox,
        "codex_cli_timeout": args.codex_cli_timeout,
        "codex_max_concurrent_requests": args.codex_max_concurrent_requests,
        "codex_parent_surface_mode": args.codex_parent_surface_mode,
    }


def main() -> None:
    args = parse_args()
    dry_run = args.dry_run or args.command == "dry-run"
    summary = _summary(args)
    if dry_run:
        print(json.dumps({"dry_run": True, **summary}, indent=2, sort_keys=True))
        return

    from examples.gpu_mode.env import GpuModeEnv
    from ttt_discover import DiscoverConfig, discover

    config = DiscoverConfig(
        runner="codex_no_finetune",
        env_type=GpuModeEnv,
        problem_type=args.problem_type,
        experiment_name=args.experiment_name,
        wandb_project=args.wandb_project,
        num_epochs=args.num_epochs,
        groups_per_batch=args.groups_per_batch,
        group_size=args.group_size,
        num_cpus_per_task=args.num_cpus_per_task,
        eval_timeout=args.eval_timeout,
        codex_backend=args.codex_backend,
        codex_model_name=args.codex_model_name,
        codex_max_output_tokens=args.codex_max_output_tokens,
        codex_temperature=args.codex_temperature,
        codex_cli_command=args.codex_cli_command,
        codex_cli_sandbox=args.codex_cli_sandbox,
        codex_cli_timeout=args.codex_cli_timeout,
        codex_max_concurrent_requests=args.codex_max_concurrent_requests,
        codex_parent_surface_mode=args.codex_parent_surface_mode,
    )
    discover(config)


if __name__ == "__main__":
    main()
````

### `tests/test_codex_archive_slot_surface_balance.py`

````python
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ttt_discover.codex_utils.runtime import State
from ttt_discover.codex_utils.sampler import (
    ARCHIVE_SLOT_FALLBACK_SURFACE,
    ARCHIVE_SLOT_SURFACES,
    PUCTSampler,
    _ARCHIVE_SLOT_SURFACE_COUNT_KEY,
    _empty_archive_slot_surface_counts,
    _sampler_file_for_step,
    archive_slot_surface_for_buffer_idx,
    create_sampler,
)


class DummyEnv:
    state_type = State

    @staticmethod
    def create_initial_state(problem_type: str) -> State:
        return State(
            timestep=0,
            construction=["initial", problem_type],
            code="initial",
            value=0.0,
            id=f"initial-{problem_type}",
        )


def make_state(
    idx: int,
    value: float,
    *,
    parents: list[dict] | None = None,
    code: str | None = None,
    construction: list | None = None,
    observation: str | None = None,
) -> State:
    return State(
        timestep=idx,
        construction=construction if construction is not None else [idx],
        code=code if code is not None else f"code-{idx}",
        value=value,
        parents=parents or [],
        id=f"s{idx}",
        observation=observation if observation is not None else f"obs-{idx}",
    )


def make_sampler(
    tmp_path: Path,
    *,
    states: list[State],
    parent_surface_mode: str = "none",
) -> PUCTSampler:
    sampler = PUCTSampler(
        file_path=str(tmp_path / "puct_sampler.json"),
        env_type=DummyEnv,
        batch_size=0,
        puct_c=0.0,
        topk_children=0,
        parent_surface_mode=parent_surface_mode,
    )
    sampler._states = states
    sampler._initial_states = []
    return sampler


def sampler_payload(sampler: PUCTSampler, step: int) -> dict:
    sampler._save(step)
    path = Path(_sampler_file_for_step(sampler.file_path, step))
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("num_states", [1, 2])
def test_disabled_mode_matches_default_and_has_no_surface_outputs(
    tmp_path: Path,
    num_states: int,
) -> None:
    explicit_states = [
        make_state(0, 10.0),
        make_state(1, 9.0, parents=[{"id": "s0", "timestep": 0}]),
        make_state(2, 8.0),
    ]
    default_states = [
        make_state(0, 10.0),
        make_state(1, 9.0, parents=[{"id": "s0", "timestep": 0}]),
        make_state(2, 8.0),
    ]
    explicit = make_sampler(
        tmp_path / "explicit",
        states=explicit_states,
        parent_surface_mode="none",
    )
    default = make_sampler(tmp_path / "default", states=default_states)

    explicit_picked = explicit.sample_states(num_states)
    default_picked = default.sample_states(num_states)

    assert [s.id for s in explicit_picked] == [s.id for s in default_picked]
    assert explicit._last_puct_stats == default._last_puct_stats
    columns, rows = explicit.get_sample_table()
    assert columns == [
        "buffer_idx",
        "timestep",
        "value",
        "terminal_value",
        "parent_value",
        "construction_len",
        "observation_len",
        "n",
        "Q",
        "P",
        "bonus",
        "score",
    ]
    assert all(len(row) == len(columns) for row in rows)
    assert not any(key.startswith("puct/archive_slot_surface") for key in explicit.get_sample_stats())
    assert _ARCHIVE_SLOT_SURFACE_COUNT_KEY not in sampler_payload(explicit, 3)


def test_classifier_uses_archive_index_modulo_only(tmp_path: Path) -> None:
    states = [
        make_state(
            idx,
            100.0 - idx,
            parents=[{"id": f"parent-{idx}", "timestep": -1}],
            code=f"unique-code-{idx}",
            construction=[{"content": idx}],
            observation=f"unique-observation-{idx}",
        )
        for idx in range(8)
    ]
    sampler = make_sampler(
        tmp_path,
        states=states,
        parent_surface_mode="archive_slot_surface",
    )
    state_id_to_idx, state_obj_to_idx = sampler._buffer_index_maps()

    assert [archive_slot_surface_for_buffer_idx(i) for i in range(8)] == [
        ARCHIVE_SLOT_SURFACES[i % 4] for i in range(8)
    ]
    before = [
        sampler._archive_slot_surface_for_state(s, state_id_to_idx, state_obj_to_idx)
        for s in states
    ]

    for idx, state in enumerate(states):
        state.value = -1000.0 + idx
        state.code = f"changed-code-{idx}"
        state.construction = ["changed", idx]
        state.observation = f"changed-observation-{idx}"
        state.parents = [{"id": f"changed-parent-{idx}", "timestep": idx}]

    after = [
        sampler._archive_slot_surface_for_state(s, state_id_to_idx, state_obj_to_idx)
        for s in states
    ]
    assert before == after == [ARCHIVE_SLOT_SURFACES[i % 4] for i in range(8)]


def test_least_sampled_surface_beats_higher_puct_and_keeps_order_within_surface(
    tmp_path: Path,
) -> None:
    states = [
        make_state(0, 100.0),
        make_state(1, 10.0),
        make_state(2, 5.0),
        make_state(3, 4.0),
        make_state(4, 90.0),
        make_state(5, 9.0),
    ]
    sampler = make_sampler(
        tmp_path,
        states=states,
        parent_surface_mode="archive_slot_surface",
    )
    sampler._archive_slot_surface_counts.update(
        {
            "slot_mod_0": 5,
            "slot_mod_1": 0,
            "slot_mod_2": 5,
            "slot_mod_3": 5,
        }
    )

    picked = sampler.sample_states(1)

    assert [s.id for s in picked] == ["s1"]
    assert sampler._archive_slot_surface_counts["slot_mod_1"] == 1
    columns, rows = sampler.get_sample_table()
    assert columns[-6:] == [
        "surface_mode",
        "target_surface",
        "selected_surface",
        "surface_count_before",
        "surface_count_after",
        "surface_fallback",
    ]
    assert rows[0][-6:] == (
        "archive_slot_surface",
        "slot_mod_1",
        "slot_mod_1",
        0,
        1,
        False,
    )
    metrics = sampler.get_sample_stats()
    assert metrics["puct/archive_slot_surface/enabled"] == 1
    assert metrics["puct/archive_slot_surface/count/slot_mod_1"] == 1


def test_equal_counts_use_surface_order_then_puct_order_within_surface(
    tmp_path: Path,
) -> None:
    states = [
        make_state(0, 1.0),
        make_state(1, 100.0),
        make_state(2, 90.0),
        make_state(3, 80.0),
        make_state(4, 2.0),
    ]
    sampler = make_sampler(
        tmp_path,
        states=states,
        parent_surface_mode="archive_slot_surface",
    )

    picked = sampler.sample_states(1)

    assert [s.id for s in picked] == ["s4"]
    assert sampler.get_sample_table()[1][0][-6:] == (
        "archive_slot_surface",
        "slot_mod_0",
        "slot_mod_0",
        0,
        1,
        False,
    )


def test_multi_parent_updates_counts_and_preserves_lineage_blocking(
    tmp_path: Path,
) -> None:
    states = [
        make_state(0, 100.0),
        make_state(1, 99.0, parents=[{"id": "s0", "timestep": 0}]),
        make_state(2, 98.0),
        make_state(3, 97.0),
        make_state(4, 96.0),
    ]
    sampler = make_sampler(
        tmp_path,
        states=states,
        parent_surface_mode="archive_slot_surface",
    )

    picked = sampler.sample_states(2)

    assert [s.id for s in picked] == ["s0", "s2"]
    assert sampler._archive_slot_surface_counts["slot_mod_0"] == 1
    assert sampler._archive_slot_surface_counts["slot_mod_1"] == 0
    assert sampler._archive_slot_surface_counts["slot_mod_2"] == 1
    assert sampler.get_sample_table()[1][1][-6:] == (
        "archive_slot_surface",
        "slot_mod_2",
        "slot_mod_2",
        0,
        1,
        False,
    )


def test_fallback_increments_fallback_count_and_respects_lineage_blocking(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    states = [
        make_state(0, 100.0),
        make_state(1, 99.0, parents=[{"id": "s0", "timestep": 0}]),
        make_state(2, 98.0),
    ]
    sampler = make_sampler(
        tmp_path,
        states=states,
        parent_surface_mode="archive_slot_surface",
    )
    monkeypatch.setattr(
        sampler,
        "_select_balanced_archive_slot_entry",
        lambda *args, **kwargs: None,
    )

    picked = sampler.sample_states(2)

    assert [s.id for s in picked] == ["s0", "s2"]
    assert sampler._archive_slot_surface_counts[ARCHIVE_SLOT_FALLBACK_SURFACE] == 2
    rows = sampler.get_sample_table()[1]
    assert rows[0][-6:] == (
        "archive_slot_surface",
        ARCHIVE_SLOT_FALLBACK_SURFACE,
        "slot_mod_0",
        0,
        1,
        True,
    )
    assert rows[1][-6:] == (
        "archive_slot_surface",
        ARCHIVE_SLOT_FALLBACK_SURFACE,
        "slot_mod_2",
        1,
        2,
        True,
    )


def test_save_resume_restores_counts_and_sanitizes_missing_bad_values(
    tmp_path: Path,
) -> None:
    file_path = str(tmp_path / "puct_sampler.json")
    sampler = PUCTSampler(
        file_path=file_path,
        env_type=DummyEnv,
        batch_size=0,
        parent_surface_mode="archive_slot_surface",
    )
    sampler._archive_slot_surface_counts.update(
        {
            "slot_mod_0": 1,
            "slot_mod_1": 2,
            "slot_mod_2": 3,
            "slot_mod_3": 4,
            ARCHIVE_SLOT_FALLBACK_SURFACE: 5,
        }
    )
    sampler._save(7)
    resumed = PUCTSampler(
        file_path=file_path,
        env_type=DummyEnv,
        batch_size=0,
        resume_step=7,
        parent_surface_mode="archive_slot_surface",
    )
    assert resumed._archive_slot_surface_counts == {
        "slot_mod_0": 1,
        "slot_mod_1": 2,
        "slot_mod_2": 3,
        "slot_mod_3": 4,
        ARCHIVE_SLOT_FALLBACK_SURFACE: 5,
    }

    missing_payload = sampler_payload(sampler, 8)
    missing_payload.pop(_ARCHIVE_SLOT_SURFACE_COUNT_KEY)
    Path(_sampler_file_for_step(file_path, 8)).write_text(
        json.dumps(missing_payload),
        encoding="utf-8",
    )
    missing = PUCTSampler(
        file_path=file_path,
        env_type=DummyEnv,
        batch_size=0,
        resume_step=8,
        parent_surface_mode="archive_slot_surface",
    )
    assert missing._archive_slot_surface_counts == _empty_archive_slot_surface_counts()

    bad_payload = sampler_payload(sampler, 9)
    bad_payload[_ARCHIVE_SLOT_SURFACE_COUNT_KEY] = {
        "slot_mod_0": True,
        "slot_mod_1": "2",
        "slot_mod_2": -1,
        "slot_mod_3": 3,
        ARCHIVE_SLOT_FALLBACK_SURFACE: 4.5,
        "unknown": 99,
    }
    Path(_sampler_file_for_step(file_path, 9)).write_text(
        json.dumps(bad_payload),
        encoding="utf-8",
    )
    bad = PUCTSampler(
        file_path=file_path,
        env_type=DummyEnv,
        batch_size=0,
        resume_step=9,
        parent_surface_mode="archive_slot_surface",
    )
    assert bad._archive_slot_surface_counts == {
        "slot_mod_0": 0,
        "slot_mod_1": 0,
        "slot_mod_2": 0,
        "slot_mod_3": 3,
        ARCHIVE_SLOT_FALLBACK_SURFACE: 0,
    }


def test_config_cli_plumbing_and_default_disabled(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    run_discovery = (root / "repro/run_discovery.py").read_text(encoding="utf-8")
    codex_runner = (root / "ttt_discover/rl/codex_no_finetune.py").read_text(
        encoding="utf-8"
    )
    codex_wrapper = (root / "ttt_discover/codex_utils/discovery.py").read_text(
        encoding="utf-8"
    )
    tinker_wrapper = (root / "ttt_discover/discovery.py").read_text(
        encoding="utf-8"
    )

    sampler = PUCTSampler(
        file_path=str(tmp_path / "puct_sampler.json"),
        env_type=DummyEnv,
        batch_size=0,
    )
    assert sampler.parent_surface_mode == "none"
    assert create_sampler(
        log_path=str(tmp_path / "factory"),
        env_type=DummyEnv,
        batch_size=0,
    ).parent_surface_mode == "none"
    with pytest.raises(ValueError):
        create_sampler(
            log_path=str(tmp_path / "bad"),
            env_type=DummyEnv,
            batch_size=0,
            parent_surface_mode="bad",
        )

    assert "--codex-parent-surface-mode" in run_discovery
    assert '"codex_parent_surface_mode": args.codex_parent_surface_mode' in run_discovery
    assert "codex_parent_surface_mode=args.codex_parent_surface_mode" in run_discovery
    assert 'parent_surface_mode: Literal["none", "archive_slot_surface"] = "none"' in codex_runner
    assert "parent_surface_mode=cfg.parent_surface_mode" in codex_runner
    for wrapper in (codex_wrapper, tinker_wrapper):
        assert 'codex_parent_surface_mode: Literal["none", "archive_slot_surface"] = "none"' in wrapper
        assert "parent_surface_mode=config.codex_parent_surface_mode" in wrapper
````

## Diff Stat

### Worktree Tracked Diff Stat

````text
 ttt_discover/codex_utils/discovery.py |   2 +
 ttt_discover/codex_utils/sampler.py   | 278 +++++++++++++++++++++++++++++++++-
 ttt_discover/discovery.py             |   2 +
 ttt_discover/rl/codex_no_finetune.py  |   2 +
 4 files changed, 281 insertions(+), 3 deletions(-)
````
## Raw Diff

<details>
<summary>Tracked worktree patch</summary>

````diff
diff --git a/ttt_discover/codex_utils/discovery.py b/ttt_discover/codex_utils/discovery.py
index b21a80f..9addf8e 100644
--- a/ttt_discover/codex_utils/discovery.py
+++ b/ttt_discover/codex_utils/discovery.py
@@ -50,6 +50,7 @@ class DiscoverConfig:
     codex_initial_program_paths: tuple[str, ...] = ()
     codex_initial_pool_paths: tuple[str, ...] = ()
     codex_autonomous: bool = False
+    codex_parent_surface_mode: Literal["none", "archive_slot_surface"] = "none"
 
 
 def _run_codex_no_finetune(config: DiscoverConfig) -> None:
@@ -85,6 +86,7 @@ def _run_codex_no_finetune(config: DiscoverConfig) -> None:
         initial_program_paths=config.codex_initial_program_paths,
         initial_pool_paths=config.codex_initial_pool_paths,
         autonomous=config.codex_autonomous,
+        parent_surface_mode=config.codex_parent_surface_mode,
         wandb_project=config.wandb_project,
         wandb_name=experiment_name,
         log_path=log_path,
diff --git a/ttt_discover/codex_utils/sampler.py b/ttt_discover/codex_utils/sampler.py
index 5c5d4d1..0675c2e 100644
--- a/ttt_discover/codex_utils/sampler.py
+++ b/ttt_discover/codex_utils/sampler.py
@@ -8,7 +8,7 @@ import os
 from pathlib import Path
 import threading
 import time
-from typing import Any, Callable
+from typing import Any, Callable, Literal
 
 import numpy as np
 
@@ -16,6 +16,50 @@ from ttt_discover.codex_utils.runtime import State, state_from_dict, to_json_ser
 
 logger = logging.getLogger(__name__)
 
+ARCHIVE_SLOT_SURFACES = (
+    "slot_mod_0",
+    "slot_mod_1",
+    "slot_mod_2",
+    "slot_mod_3",
+)
+ARCHIVE_SLOT_FALLBACK_SURFACE = "puct_fallback"
+_ARCHIVE_SLOT_SURFACE_COUNT_KEY = "puct_archive_slot_surface_counts"
+_PARENT_SURFACE_MODES = {"none", "archive_slot_surface"}
+
+
+def _validate_parent_surface_mode(
+    parent_surface_mode: Literal["none", "archive_slot_surface"],
+) -> Literal["none", "archive_slot_surface"]:
+    if parent_surface_mode not in _PARENT_SURFACE_MODES:
+        raise ValueError(
+            "parent_surface_mode must be one of "
+            f"{sorted(_PARENT_SURFACE_MODES)}, got {parent_surface_mode!r}"
+        )
+    return parent_surface_mode
+
+
+def _empty_archive_slot_surface_counts() -> dict[str, int]:
+    return {
+        surface: 0
+        for surface in (*ARCHIVE_SLOT_SURFACES, ARCHIVE_SLOT_FALLBACK_SURFACE)
+    }
+
+
+def _sanitize_archive_slot_surface_counts(raw_counts: Any) -> dict[str, int]:
+    counts = _empty_archive_slot_surface_counts()
+    if not isinstance(raw_counts, dict):
+        return counts
+    for surface in counts:
+        value = raw_counts.get(surface, 0)
+        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
+            continue
+        counts[surface] = value
+    return counts
+
+
+def archive_slot_surface_for_buffer_idx(buffer_idx: int) -> str:
+    return ARCHIVE_SLOT_SURFACES[buffer_idx % len(ARCHIVE_SLOT_SURFACES)]
+
 
 @contextmanager
 def _file_lock(lock_path: str, *, poll_s: float = 0.05, stale_s: float = 600.0):
@@ -353,6 +397,7 @@ class PUCTSampler(StateSampler):
         resume_step: int | None = None,
         puct_c: float = 1.0,
         topk_children: int = 2,
+        parent_surface_mode: Literal["none", "archive_slot_surface"] = "none",
     ):
         self.file_path = file_path
         self.env_type = env_type
@@ -361,6 +406,7 @@ class PUCTSampler(StateSampler):
         self.batch_size = batch_size
         self.topk_children = topk_children
         self.puct_c = float(puct_c)
+        self.parent_surface_mode = _validate_parent_surface_mode(parent_surface_mode)
         
         self._states: list[State] = []
         self._initial_states: list[State] = []
@@ -375,6 +421,10 @@ class PUCTSampler(StateSampler):
         self._T: int = 0
         self._last_scale: float = 1.0
         self._last_puct_stats: list[tuple[int, float, float, float, float]] = []
+        self._archive_slot_surface_counts = _empty_archive_slot_surface_counts()
+        self._last_archive_slot_surface_rows: list[
+            tuple[str, str, str, int, int, bool]
+        ] = []
         
         if resume_step is not None:
             self._load(resume_step)
@@ -399,6 +449,10 @@ class PUCTSampler(StateSampler):
         self._n = store.get("puct_n", {}) or {}
         self._m = store.get("puct_m", {}) or {}
         self._T = int(store.get("puct_T", 0) or 0)
+        if self.parent_surface_mode == "archive_slot_surface":
+            self._archive_slot_surface_counts = _sanitize_archive_slot_surface_counts(
+                store.get(_ARCHIVE_SLOT_SURFACE_COUNT_KEY, {})
+            )
 
     def _save(self, step: int):
         save_path = _sampler_file_for_step(self.file_path, step)
@@ -411,6 +465,12 @@ class PUCTSampler(StateSampler):
             "puct_m": self._m,
             "puct_T": self._T,
         }
+        if self.parent_surface_mode == "archive_slot_surface":
+            store[_ARCHIVE_SLOT_SURFACE_COUNT_KEY] = (
+                _sanitize_archive_slot_surface_counts(
+                    self._archive_slot_surface_counts
+                )
+            )
         with _file_lock(f"{save_path}.lock"):
             _atomic_write_json(save_path, store)
 
@@ -489,6 +549,176 @@ class PUCTSampler(StateSampler):
                     queue.append(child_id)
         return lineage
 
+    def _buffer_index_maps(self) -> tuple[dict[str, int], dict[int, int]]:
+        state_id_to_idx: dict[str, int] = {}
+        state_obj_to_idx: dict[int, int] = {}
+        for idx, state in enumerate(self._states):
+            state_id_to_idx[state.id] = idx
+            state_obj_to_idx[id(state)] = idx
+        return state_id_to_idx, state_obj_to_idx
+
+    def _buffer_idx_for_state(
+        self,
+        state: State,
+        state_id_to_idx: dict[str, int],
+        state_obj_to_idx: dict[int, int],
+    ) -> int:
+        return state_obj_to_idx.get(id(state), state_id_to_idx.get(state.id, -1))
+
+    def _archive_slot_surface_for_state(
+        self,
+        state: State,
+        state_id_to_idx: dict[str, int],
+        state_obj_to_idx: dict[int, int],
+    ) -> str:
+        buffer_idx = self._buffer_idx_for_state(
+            state,
+            state_id_to_idx,
+            state_obj_to_idx,
+        )
+        if buffer_idx < 0:
+            return ARCHIVE_SLOT_FALLBACK_SURFACE
+        return archive_slot_surface_for_buffer_idx(buffer_idx)
+
+    def _select_balanced_archive_slot_entry(
+        self,
+        scores: list[tuple],
+        picked_ids: set[str],
+        blocked_ids: set[str],
+        state_id_to_idx: dict[str, int],
+        state_obj_to_idx: dict[int, int],
+    ) -> tuple[tuple, str] | None:
+        grouped: dict[str, list[tuple]] = {
+            surface: [] for surface in ARCHIVE_SLOT_SURFACES
+        }
+        for entry in scores:
+            state = entry[2]
+            if state.id in picked_ids or state.id in blocked_ids:
+                continue
+            surface = self._archive_slot_surface_for_state(
+                state,
+                state_id_to_idx,
+                state_obj_to_idx,
+            )
+            if surface in grouped:
+                grouped[surface].append(entry)
+
+        available_surfaces = [
+            surface for surface in ARCHIVE_SLOT_SURFACES if grouped[surface]
+        ]
+        if not available_surfaces:
+            return None
+        target_surface = min(
+            available_surfaces,
+            key=lambda surface: self._archive_slot_surface_counts.get(surface, 0),
+        )
+        return grouped[target_surface][0], target_surface
+
+    def _record_archive_slot_surface_pick(
+        self,
+        *,
+        target_surface: str,
+        selected_surface: str,
+        fallback: bool,
+    ) -> None:
+        count_surface = (
+            ARCHIVE_SLOT_FALLBACK_SURFACE if fallback else selected_surface
+        )
+        before = self._archive_slot_surface_counts.get(count_surface, 0)
+        after = before + 1
+        self._archive_slot_surface_counts[count_surface] = after
+        self._last_archive_slot_surface_rows.append(
+            (
+                "archive_slot_surface",
+                target_surface,
+                selected_surface,
+                before,
+                after,
+                fallback,
+            )
+        )
+
+    def _sample_states_archive_slot_surface(
+        self,
+        num_states: int,
+        scores: list[tuple],
+        initial_ids: set[str],
+    ) -> list[State]:
+        state_id_to_idx, state_obj_to_idx = self._buffer_index_maps()
+        children_map = self._build_children_map() if num_states > 1 else {}
+        picked: list[State] = []
+        top_scores: list[tuple] = []
+        picked_ids: set[str] = set()
+        blocked_ids: set[str] = set()
+        self._last_archive_slot_surface_rows = []
+
+        def append_entry(
+            entry: tuple,
+            *,
+            target_surface: str,
+            fallback: bool,
+        ) -> None:
+            state = entry[2]
+            selected_surface = self._archive_slot_surface_for_state(
+                state,
+                state_id_to_idx,
+                state_obj_to_idx,
+            )
+            self._record_archive_slot_surface_pick(
+                target_surface=target_surface,
+                selected_surface=selected_surface,
+                fallback=fallback,
+            )
+            picked.append(state)
+            top_scores.append(entry)
+            picked_ids.add(state.id)
+            if num_states > 1:
+                blocked_ids.update(self._get_full_lineage(state, children_map))
+
+        while len(picked) < num_states:
+            selected = self._select_balanced_archive_slot_entry(
+                scores,
+                picked_ids,
+                blocked_ids if num_states > 1 else set(),
+                state_id_to_idx,
+                state_obj_to_idx,
+            )
+            if selected is None:
+                break
+            entry, target_surface = selected
+            append_entry(
+                entry,
+                target_surface=target_surface,
+                fallback=False,
+            )
+
+        for entry in scores:
+            if len(picked) >= num_states:
+                break
+            state = entry[2]
+            if state.id in picked_ids:
+                continue
+            if num_states > 1 and state.id in blocked_ids:
+                continue
+            append_entry(
+                entry,
+                target_surface=ARCHIVE_SLOT_FALLBACK_SURFACE,
+                fallback=True,
+            )
+
+        self._last_sampled_states = picked
+        self._last_sampled_indices = [
+            self._buffer_idx_for_state(s, state_id_to_idx, state_obj_to_idx)
+            for s in picked
+        ]
+        self._last_puct_stats = [(t[3], t[4], t[5], t[6], t[0]) for t in top_scores]
+
+        for s in picked:
+            if s.id in initial_ids:
+                self._refresh_random_construction(s)
+
+        return picked
+
     def sample_states(self, num_states: int) -> list[State]:
         initial_ids = {s.id for s in self._initial_states}
         candidates = list(self._states)
@@ -501,6 +731,7 @@ class PUCTSampler(StateSampler):
             self._last_sampled_states = picked
             self._last_sampled_indices = []
             self._last_puct_stats = [(0, 0.0, 0.0, 0.0, 0.0) for _ in picked]
+            self._last_archive_slot_surface_rows = []
             return picked
 
         vals = np.array([float(s.value if s.value is not None else float("-inf")) for s in candidates])
@@ -521,6 +752,13 @@ class PUCTSampler(StateSampler):
 
         scores.sort(key=lambda x: (x[0], x[1]), reverse=True)
 
+        if self.parent_surface_mode == "archive_slot_surface":
+            return self._sample_states_archive_slot_surface(
+                num_states,
+                scores,
+                initial_ids,
+            )
+
         if num_states > 1:
             children_map = self._build_children_map()
             picked, top_scores, blocked_ids = [], [], set()
@@ -731,21 +969,50 @@ class PUCTSampler(StateSampler):
         stats.update(_stats(sampled_values, "puct/sampled_value"))
         stats.update(_stats(sampled_timesteps, "puct/sampled_timestep"))
         stats.update(_stats(sampled_constr_lens, "puct/sampled_construction_len"))
+        if self.parent_surface_mode == "archive_slot_surface":
+            counts = _sanitize_archive_slot_surface_counts(
+                self._archive_slot_surface_counts
+            )
+            stats["puct/archive_slot_surface/enabled"] = 1
+            for surface in (*ARCHIVE_SLOT_SURFACES, ARCHIVE_SLOT_FALLBACK_SURFACE):
+                stats[f"puct/archive_slot_surface/count/{surface}"] = counts[surface]
         return stats
 
     def get_sample_table(self) -> tuple[list[str], list[tuple]]:
         columns = ["buffer_idx", "timestep", "value", "terminal_value", "parent_value", "construction_len", "observation_len", "n", "Q", "P", "bonus", "score"]
+        if self.parent_surface_mode == "archive_slot_surface":
+            columns.extend(
+                [
+                    "surface_mode",
+                    "target_surface",
+                    "selected_surface",
+                    "surface_count_before",
+                    "surface_count_after",
+                    "surface_fallback",
+                ]
+            )
         rows = []
         if not self._last_sampled_states:
             return columns, rows
         indices = self._last_sampled_indices if len(self._last_sampled_indices) == len(self._last_sampled_states) else [-1] * len(self._last_sampled_states)
         stats = self._last_puct_stats if len(self._last_puct_stats) == len(self._last_sampled_states) else [(0, 0.0, 0.0, 0.0, 0.0)] * len(self._last_sampled_states)
-        for idx, state, (n, Q, P, bonus, score) in zip(indices, self._last_sampled_states, stats):
+        surface_rows = (
+            self._last_archive_slot_surface_rows
+            if len(self._last_archive_slot_surface_rows) == len(self._last_sampled_states)
+            else [
+                ("archive_slot_surface", "", "", 0, 0, False)
+                for _ in self._last_sampled_states
+            ]
+        )
+        for row_idx, (idx, state, (n, Q, P, bonus, score)) in enumerate(zip(indices, self._last_sampled_states, stats)):
             parent_val = state.parent_values[0] if state.parent_values else None
             constr = getattr(state, 'construction', None)
             constr_len = len(constr) if constr is not None else 0
             obs_len = len(state.observation) if state.observation else 0
-            rows.append((idx, state.timestep, state.value, 0, parent_val, constr_len, obs_len, n, Q, P, bonus, score))
+            row = (idx, state.timestep, state.value, 0, parent_val, constr_len, obs_len, n, Q, P, bonus, score)
+            if self.parent_surface_mode == "archive_slot_surface":
+                row = row + surface_rows[row_idx]
+            rows.append(row)
         return columns, rows
 
 
@@ -756,10 +1023,12 @@ def create_sampler(
     batch_size: int = 1,
     resume_step: int | None = None,
     topk_children: int = 2,
+    parent_surface_mode: Literal["none", "archive_slot_surface"] = "none",
 ) -> StateSampler:
     """Factory function to create samplers. Pass the env type (from config.env_type)."""
     if not log_path:
         raise ValueError("log_path is required when using PUCT sampler")
+    parent_surface_mode = _validate_parent_surface_mode(parent_surface_mode)
     sampler_path = os.path.join(log_path, "puct_sampler.json")
     return PUCTSampler(
         file_path=sampler_path,
@@ -768,6 +1037,7 @@ def create_sampler(
         batch_size=batch_size,
         resume_step=resume_step,
         topk_children=topk_children,
+        parent_surface_mode=parent_surface_mode,
     )
 
 
@@ -778,6 +1048,7 @@ def get_or_create_sampler_with_default(
     batch_size: int = 1,
     resume_step: int | None = None,
     topk_children: int = 2,
+    parent_surface_mode: Literal["none", "archive_slot_surface"] = "none",
 ) -> StateSampler:
     """Get sampler. Initial experience is created via env_type.create_initial_state."""
     return create_sampler(
@@ -787,4 +1058,5 @@ def get_or_create_sampler_with_default(
         batch_size=batch_size,
         resume_step=resume_step,
         topk_children=topk_children,
+        parent_surface_mode=parent_surface_mode,
     )
diff --git a/ttt_discover/discovery.py b/ttt_discover/discovery.py
index 6cdb434..74c2263 100644
--- a/ttt_discover/discovery.py
+++ b/ttt_discover/discovery.py
@@ -63,6 +63,7 @@ class DiscoverConfig:
     codex_initial_program_paths: tuple[str, ...] = ()
     codex_initial_pool_paths: tuple[str, ...] = ()
     codex_autonomous: bool = False
+    codex_parent_surface_mode: Literal["none", "archive_slot_surface"] = "none"
 
 
 def init_ray(num_cpus_per_task: int, env_type: str):
@@ -146,6 +147,7 @@ async def discover_impl(config: DiscoverConfig):
             initial_program_paths=config.codex_initial_program_paths,
             initial_pool_paths=config.codex_initial_pool_paths,
             autonomous=config.codex_autonomous,
+            parent_surface_mode=config.codex_parent_surface_mode,
             wandb_project=config.wandb_project,
             wandb_name=config.experiment_name,
             log_path=log_path,
diff --git a/ttt_discover/rl/codex_no_finetune.py b/ttt_discover/rl/codex_no_finetune.py
index 18a0e9e..7366a38 100644
--- a/ttt_discover/rl/codex_no_finetune.py
+++ b/ttt_discover/rl/codex_no_finetune.py
@@ -137,6 +137,7 @@ class CodexNoFinetuneConfig:
     initial_program_paths: tuple[str, ...] = ()
     initial_pool_paths: tuple[str, ...] = ()
     topk_children: int = 16
+    parent_surface_mode: Literal["none", "archive_slot_surface"] = "none"
 
     # False is TTT Discover: non-auto, many short samples. True is AutoEvolve:
     # one Codex deep-dive workspace per sample with file edits and shell access.
@@ -960,6 +961,7 @@ def _build_sampler(cfg: CodexNoFinetuneConfig, start_batch: int) -> StateSampler
         batch_size=cfg.groups_per_batch,
         resume_step=start_batch if start_batch > 0 else None,
         topk_children=cfg.topk_children,
+        parent_surface_mode=cfg.parent_surface_mode,
     )
 
 
````
</details>

