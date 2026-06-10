# codex/diversity-parent-birth-gap-surface-balance

## Summary

按 child birth timestep 与直接 parent birth timestep 的 gap 分类 root_or_seed/gap_0/gap_1/gap_2_4/gap_5_plus/unknown，并做 surface 计数平衡。

## Branch State

- Worktree: `/opt/tiger/discover-parent-birth-gap-surface-balance`
- HEAD: `2234282`
- Base used for comparison: `2234282`
- Commits ahead of base: `0`
- Commits behind base: `0`
- Group: `completed_19`
- Implementation location: `uncommitted worktree diff`

## Implemented Behavior

- The summary above is based on reading the changed Python code, added config fields, and sampler/prompt hooks in this branch.
- For sampler branches, the common pattern is to keep baseline PUCT ordering inside a chosen surface/bucket, while changing which surface/bucket gets the next parent pick.
- Tests detected: `9` test functions across untracked test files.

## Added Markers

### Config fields

- `codex_sampling_diversity`
- `sampling_diversity`

### Constants

- `PARENT_BIRTH_GAP_SURFACES`
- `PARENT_BIRTH_GAP_SURFACE_SAMPLE_COUNTS_KEY`
- `_PARENT_BIRTH_GAP_SURFACE_ORDER`

### Classes

- None

### Functions

- `_validate_sampling_diversity`
- `_zero_parent_birth_gap_surface_counts`
- `_sanitize_parent_birth_gap_surface_counts`
- `_is_integral_timestep`
- `parent_birth_gap_surface_for_state`
- `_parent_birth_gap_surface_enabled`
- `_reset_parent_birth_gap_surface_tracking`
- `_iter_available_entries`
- `_group_parent_birth_gap_surface_entries`
- `_choose_parent_birth_gap_surface`
- `_record_parent_birth_gap_surface_selection`
- `_sample_states_parent_birth_gap_surface`

## Diff Summary

- Worktree tracked shortstat: `4 files changed, 300 insertions(+), 4 deletions(-)`
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

- `repro/gpu_mode/run_0609_parent_birth_gap_surface_balance.sh (548 bytes)`
- `repro/run_discovery.py (4828 bytes)`
- `tests/test_parent_birth_gap_surface_balance.py (12854 bytes)`

### Detected Test Functions

- `tests/test_parent_birth_gap_surface_balance.py::test_disabled_matches_baseline_order_and_has_no_diversity_observability_or_persistence`
- `tests/test_parent_birth_gap_surface_balance.py::test_classifier_buckets_and_malformed_cases`
- `tests/test_parent_birth_gap_surface_balance.py::test_classifier_ignores_forbidden_text_content_and_value_fields`
- `tests/test_parent_birth_gap_surface_balance.py::test_least_sampled_surface_beats_higher_puct_and_reports_enabled_metrics`
- `tests/test_parent_birth_gap_surface_balance.py::test_equal_counts_use_surface_order_then_baseline_order_within_surface`
- `tests/test_parent_birth_gap_surface_balance.py::test_lineage_blocking_is_preserved_when_sampling_multiple_states`
- `tests/test_parent_birth_gap_surface_balance.py::test_fallback_increments_actual_surface`
- `tests/test_parent_birth_gap_surface_balance.py::test_enabled_persistence_resume_missing_and_bad_values_are_sanitized`
- `tests/test_parent_birth_gap_surface_balance.py::test_config_sources_cli_and_create_sampler_wire_sampling_diversity`

## Untracked File Snapshots

These files are not part of `git diff` yet, but they are part of the worktree implementation state recorded here.

### `repro/gpu_mode/run_0609_parent_birth_gap_surface_balance.sh`

````bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODE="run"
if [[ "${1:-}" == "run" || "${1:-}" == "dry-run" ]]; then
  MODE="$1"
  shift
fi

export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"

python "${REPO_ROOT}/repro/run_discovery.py" "${MODE}" \
  --experiment-name gpu-mode-0609-parent-birth-gap-surface-balance \
  --codex-sampling-diversity parent_birth_gap_surface \
  "$@"
````

### `repro/run_discovery.py`

````python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _none_if_empty(value: str | None) -> str | None:
    if value is None or value == "":
        return None
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the GPUMode Codex no-finetune discovery repro."
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=("run", "dry-run"),
        default="run",
        help="Use dry-run to print the effective config without launching Codex.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--problem-type",
        choices=("trimul", "mla_decode_nvidia"),
        default="trimul",
    )
    parser.add_argument(
        "--experiment-name",
        default="gpu-mode-parent-birth-gap-surface-balance",
    )
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--group-size", type=int, default=1)
    parser.add_argument("--groups-per-batch", type=int, default=1)
    parser.add_argument("--num-cpus-per-task", type=int, default=1)
    parser.add_argument("--eval-timeout", type=int, default=1200)
    parser.add_argument("--codex-model-name", default=None)
    parser.add_argument(
        "--codex-backend",
        choices=("cli", "responses"),
        default="cli",
    )
    parser.add_argument("--codex-max-output-tokens", type=int, default=8192)
    parser.add_argument("--codex-temperature", type=float, default=None)
    parser.add_argument("--codex-cli-command", default="codex")
    parser.add_argument(
        "--codex-cli-sandbox",
        choices=("read-only", "workspace-write", "danger-full-access"),
        default="read-only",
    )
    parser.add_argument("--codex-cli-timeout", type=float, default=600)
    parser.add_argument("--codex-max-concurrent-requests", type=int, default=1)
    parser.add_argument(
        "--codex-sampling-diversity",
        choices=("off", "parent_birth_gap_surface"),
        default="off",
    )
    return parser.parse_args(argv)


def _config_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "env_type": "examples.gpu_mode.env.GpuModeEnv",
        "problem_type": args.problem_type,
        "runner": "codex_no_finetune",
        "experiment_name": args.experiment_name,
        "wandb_project": _none_if_empty(args.wandb_project),
        "num_epochs": args.num_epochs,
        "group_size": args.group_size,
        "groups_per_batch": args.groups_per_batch,
        "num_cpus_per_task": args.num_cpus_per_task,
        "eval_timeout": args.eval_timeout,
        "codex_model_name": _none_if_empty(args.codex_model_name),
        "codex_backend": args.codex_backend,
        "codex_max_output_tokens": args.codex_max_output_tokens,
        "codex_temperature": args.codex_temperature,
        "codex_cli_command": args.codex_cli_command,
        "codex_cli_sandbox": args.codex_cli_sandbox,
        "codex_cli_timeout": args.codex_cli_timeout,
        "codex_max_concurrent_requests": args.codex_max_concurrent_requests,
        "codex_sampling_diversity": args.codex_sampling_diversity,
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    payload = _config_payload(args)
    if args.command == "dry-run" or args.dry_run:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    from examples.gpu_mode.env import GpuModeEnv
    from ttt_discover.codex_utils.discovery import DiscoverConfig, discover

    config = DiscoverConfig(
        env_type=GpuModeEnv,
        problem_type=args.problem_type,
        runner="codex_no_finetune",
        experiment_name=args.experiment_name,
        wandb_project=payload["wandb_project"],
        num_epochs=args.num_epochs,
        group_size=args.group_size,
        groups_per_batch=args.groups_per_batch,
        num_cpus_per_task=payload["num_cpus_per_task"],
        eval_timeout=payload["eval_timeout"],
        codex_model_name=payload["codex_model_name"],
        codex_backend=args.codex_backend,
        codex_max_output_tokens=args.codex_max_output_tokens,
        codex_temperature=args.codex_temperature,
        codex_cli_command=args.codex_cli_command,
        codex_cli_sandbox=args.codex_cli_sandbox,
        codex_cli_timeout=args.codex_cli_timeout,
        codex_max_concurrent_requests=args.codex_max_concurrent_requests,
        codex_sampling_diversity=args.codex_sampling_diversity,
    )
    discover(config)


if __name__ == "__main__":
    main()
````

### `tests/test_parent_birth_gap_surface_balance.py`

````python
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ttt_discover.codex_utils.runtime import State
from ttt_discover.codex_utils.sampler import (
    PARENT_BIRTH_GAP_SURFACES,
    PARENT_BIRTH_GAP_SURFACE_SAMPLE_COUNTS_KEY,
    PUCTSampler,
    create_sampler,
    parent_birth_gap_surface_for_state,
)


class DummyEnv:
    state_type = State
    _next_initial_id = 0

    @classmethod
    def create_initial_state(cls, problem_type: str) -> State:
        del problem_type
        cls._next_initial_id += 1
        return State(
            timestep=-1,
            construction=[f"init-{cls._next_initial_id}"],
            code="",
            value=0.0,
            id=f"init-{cls._next_initial_id}",
        )


def state(
    sid: str,
    *,
    timestep: object,
    value: float = 0.0,
    parent_timestep: object | None = None,
    parent_id: str = "parent",
    parents: list[dict] | None = None,
    code: str = "",
    construction: list[object] | None = None,
    observation: str = "",
) -> State:
    if parents is None:
        parent_edges = (
            []
            if parent_timestep is None
            else [{"id": parent_id, "timestep": parent_timestep}]
        )
    else:
        parent_edges = parents
    return State(
        timestep=timestep,  # type: ignore[arg-type]
        construction=[sid] if construction is None else construction,
        code=code,
        value=value,
        parents=parent_edges,
        id=sid,
        observation=observation,
    )


def make_sampler(
    tmp_path: Path,
    states: list[State],
    *,
    sampling_diversity: str = "off",
    initial_states: list[State] | None = None,
    counts: dict[str, int] | None = None,
) -> PUCTSampler:
    sampler = PUCTSampler(
        file_path=str(tmp_path / "puct_sampler.json"),
        env_type=DummyEnv,
        puct_c=0.0,
        topk_children=0,
        sampling_diversity=sampling_diversity,  # type: ignore[arg-type]
    )
    sampler._states = list(states)
    sampler._initial_states = list(initial_states or [])
    if counts is not None:
        sampler._parent_birth_gap_surface_sample_counts.update(counts)
    return sampler


def sampler_json_path(tmp_path: Path, step: int) -> Path:
    return tmp_path / f"puct_sampler_step_{step:06d}.json"


def test_disabled_matches_baseline_order_and_has_no_diversity_observability_or_persistence(tmp_path: Path):
    low = state("low", timestep=2, parent_timestep=1, value=1.0)
    high = state("high", timestep=3, parent_timestep=1, value=10.0)
    sampler = make_sampler(tmp_path, [low, high], sampling_diversity="off")

    picked = sampler.sample_states(2)

    assert [s.id for s in picked] == ["high", "low"]
    assert sampler._parent_birth_gap_surface_sample_counts == {
        surface: 0 for surface in PARENT_BIRTH_GAP_SURFACES
    }
    stats = sampler.get_sample_stats()
    assert not any("parent_birth_gap_surface" in key for key in stats)
    columns, rows = sampler.get_sample_table()
    assert "parent_birth_gap_surface" not in columns
    assert len(rows) == 2

    sampler.flush(step=1)
    payload = json.loads(sampler_json_path(tmp_path, 1).read_text(encoding="utf-8"))
    assert PARENT_BIRTH_GAP_SURFACE_SAMPLE_COUNTS_KEY not in payload


def test_classifier_buckets_and_malformed_cases():
    initial = state("initial", timestep=100, parent_timestep=99)
    assert parent_birth_gap_surface_for_state(initial, {"initial"}) == "root_or_seed"
    assert parent_birth_gap_surface_for_state(
        state("negative", timestep=-1, parent_timestep=0),
        set(),
    ) == "root_or_seed"
    assert parent_birth_gap_surface_for_state(
        state("no-parent", timestep=None, parents=[]),
        set(),
    ) == "root_or_seed"
    assert parent_birth_gap_surface_for_state(
        state("bad-current", timestep="5", parent_timestep=1),
        set(),
    ) == "unknown"
    assert parent_birth_gap_surface_for_state(
        state("bad-parent", timestep=5, parent_timestep=None),
        set(),
    ) == "root_or_seed"
    assert parent_birth_gap_surface_for_state(
        state("missing-parent-ts", timestep=5, parents=[{}]),
        set(),
    ) == "unknown"
    assert parent_birth_gap_surface_for_state(
        state("bool-current", timestep=True, parent_timestep=1),
        set(),
    ) == "unknown"
    assert parent_birth_gap_surface_for_state(
        state("bool-parent", timestep=2, parent_timestep=False),
        set(),
    ) == "unknown"
    assert parent_birth_gap_surface_for_state(
        state("gap0", timestep=1, parent_timestep=2),
        set(),
    ) == "gap_0"
    assert parent_birth_gap_surface_for_state(
        state("gap1", timestep=np.int64(3), parent_timestep=np.int64(2)),
        set(),
    ) == "gap_1"
    assert parent_birth_gap_surface_for_state(
        state("gap2", timestep=5, parent_timestep=1),
        set(),
    ) == "gap_2_4"
    assert parent_birth_gap_surface_for_state(
        state("gap5", timestep=7, parent_timestep=1),
        set(),
    ) == "gap_5_plus"


def test_classifier_ignores_forbidden_text_content_and_value_fields():
    a = state(
        "a",
        timestep=9,
        parent_timestep=4,
        value=999.0,
        code="raise SystemExit('do not read')",
        construction=["secret", {"score": 123}],
        observation="stdout should not matter",
    )
    b = state(
        "b",
        timestep=9,
        parent_timestep=4,
        value=-999.0,
        code="different code",
        construction=["different"],
        observation="different observation",
    )

    assert parent_birth_gap_surface_for_state(a, set()) == "gap_5_plus"
    assert parent_birth_gap_surface_for_state(a, set()) == parent_birth_gap_surface_for_state(b, set())


def test_least_sampled_surface_beats_higher_puct_and_reports_enabled_metrics(tmp_path: Path):
    high_gap0 = state("high-gap0", timestep=10, parent_timestep=10, value=100.0)
    low_gap1 = state("low-gap1", timestep=11, parent_timestep=10, value=1.0)
    sampler = make_sampler(
        tmp_path,
        [high_gap0, low_gap1],
        sampling_diversity="parent_birth_gap_surface",
        counts={"gap_0": 5, "gap_1": 0},
    )

    picked = sampler.sample_states(1)

    assert [s.id for s in picked] == ["low-gap1"]
    assert sampler._parent_birth_gap_surface_sample_counts["gap_1"] == 1
    stats = sampler.get_sample_stats()
    assert stats["puct/parent_birth_gap_surface/enabled"] == 1
    assert stats["puct/parent_birth_gap_surface/count_gap_0"] == 5
    assert stats["puct/parent_birth_gap_surface/count_gap_1"] == 1
    assert stats["puct/parent_birth_gap_surface/selected_gap_1"] == 1
    assert stats["puct/parent_birth_gap_surface/available_gap_0"] == 1
    assert stats["puct/parent_birth_gap_surface/available_gap_1"] == 1
    columns, rows = sampler.get_sample_table()
    assert columns[-4:] == [
        "parent_birth_gap_surface",
        "parent_birth_gap_surface_count_before",
        "parent_birth_gap_surface_count_after",
        "parent_birth_gap_surface_selection",
    ]
    assert rows[0][-4:] == ("gap_1", 0, 1, "balanced")


def test_equal_counts_use_surface_order_then_baseline_order_within_surface(tmp_path: Path):
    gap1_high = state("gap1-high", timestep=11, parent_timestep=10, value=100.0)
    gap0_high = state("gap0-high", timestep=10, parent_timestep=10, value=50.0)
    gap0_low = state("gap0-low", timestep=9, parent_timestep=10, value=10.0)
    sampler = make_sampler(
        tmp_path,
        [gap1_high, gap0_low, gap0_high],
        sampling_diversity="parent_birth_gap_surface",
    )

    picked = sampler.sample_states(1)

    assert [s.id for s in picked] == ["gap0-high"]


def test_lineage_blocking_is_preserved_when_sampling_multiple_states(tmp_path: Path):
    ancestor = state(
        "ancestor",
        timestep=10,
        parent_timestep=10,
        parent_id="root",
        value=100.0,
    )
    descendant = state(
        "descendant",
        timestep=11,
        parent_timestep=10,
        parent_id="ancestor",
        value=90.0,
    )
    sibling = state(
        "sibling",
        timestep=11,
        parent_timestep=10,
        parent_id="other",
        value=80.0,
    )
    sampler = make_sampler(
        tmp_path,
        [ancestor, descendant, sibling],
        sampling_diversity="parent_birth_gap_surface",
        counts={"gap_1": 10},
    )

    picked = sampler.sample_states(2)

    assert [s.id for s in picked] == ["ancestor", "sibling"]


def test_fallback_increments_actual_surface(tmp_path: Path):
    candidate = state("candidate", timestep=14, parent_timestep=10, value=10.0)
    sampler = make_sampler(
        tmp_path,
        [candidate],
        sampling_diversity="parent_birth_gap_surface",
    )
    sampler._choose_parent_birth_gap_surface = lambda grouped: None  # type: ignore[method-assign]

    picked = sampler.sample_states(1)

    assert [s.id for s in picked] == ["candidate"]
    assert sampler._parent_birth_gap_surface_sample_counts["gap_2_4"] == 1
    assert sampler.get_sample_table()[1][0][-4:] == ("gap_2_4", 0, 1, "fallback")


def test_enabled_persistence_resume_missing_and_bad_values_are_sanitized(tmp_path: Path):
    sampler = make_sampler(
        tmp_path,
        [state("candidate", timestep=2, parent_timestep=1)],
        sampling_diversity="parent_birth_gap_surface",
        counts={"root_or_seed": 2, "gap_1": 3},
    )
    sampler.flush(step=3)

    resumed = PUCTSampler(
        file_path=str(tmp_path / "puct_sampler.json"),
        env_type=DummyEnv,
        resume_step=3,
        sampling_diversity="parent_birth_gap_surface",
    )
    assert resumed._parent_birth_gap_surface_sample_counts["root_or_seed"] == 2
    assert resumed._parent_birth_gap_surface_sample_counts["gap_1"] == 3

    disabled = make_sampler(
        tmp_path,
        [state("disabled", timestep=2, parent_timestep=1)],
        sampling_diversity="off",
    )
    disabled.flush(step=4)
    missing = PUCTSampler(
        file_path=str(tmp_path / "puct_sampler.json"),
        env_type=DummyEnv,
        resume_step=4,
        sampling_diversity="parent_birth_gap_surface",
    )
    assert missing._parent_birth_gap_surface_sample_counts == {
        surface: 0 for surface in PARENT_BIRTH_GAP_SURFACES
    }

    sampler.flush(step=5)
    path = sampler_json_path(tmp_path, 5)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[PARENT_BIRTH_GAP_SURFACE_SAMPLE_COUNTS_KEY] = {
        "root_or_seed": True,
        "gap_0": -2,
        "gap_1": "3",
        "gap_2_4": 4,
        "unknown": 1.2,
        "not_a_surface": 99,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    sanitized = PUCTSampler(
        file_path=str(tmp_path / "puct_sampler.json"),
        env_type=DummyEnv,
        resume_step=5,
        sampling_diversity="parent_birth_gap_surface",
    )
    assert sanitized._parent_birth_gap_surface_sample_counts == {
        "root_or_seed": 0,
        "gap_0": 0,
        "gap_1": 0,
        "gap_2_4": 4,
        "gap_5_plus": 0,
        "unknown": 0,
    }


def test_config_sources_cli_and_create_sampler_wire_sampling_diversity(tmp_path: Path):
    from repro.run_discovery import _config_payload, parse_args

    default_args = parse_args(["dry-run"])
    explicit_args = parse_args(
        ["dry-run", "--codex-sampling-diversity", "parent_birth_gap_surface"]
    )

    assert default_args.codex_sampling_diversity == "off"
    assert explicit_args.codex_sampling_diversity == "parent_birth_gap_surface"
    assert _config_payload(explicit_args)["codex_sampling_diversity"] == "parent_birth_gap_surface"
    default_payload = _config_payload(default_args)
    assert default_payload["codex_model_name"] is None
    assert default_payload["num_cpus_per_task"] == default_args.num_cpus_per_task

    sampler = create_sampler(
        log_path=str(tmp_path),
        env_type=DummyEnv,
        sampling_diversity="parent_birth_gap_surface",
    )
    assert isinstance(sampler, PUCTSampler)
    assert sampler.sampling_diversity == "parent_birth_gap_surface"

    root = Path(__file__).resolve().parents[1]
    codex_runner = (root / "ttt_discover/rl/codex_no_finetune.py").read_text(
        encoding="utf-8"
    )
    codex_public = (root / "ttt_discover/codex_utils/discovery.py").read_text(
        encoding="utf-8"
    )
    public = (root / "ttt_discover/discovery.py").read_text(encoding="utf-8")
    assert 'sampling_diversity: Literal["off", "parent_birth_gap_surface"] = "off"' in codex_runner
    assert "sampling_diversity=cfg.sampling_diversity" in codex_runner
    for source in (codex_public, public):
        assert 'codex_sampling_diversity: Literal["off", "parent_birth_gap_surface"] = "off"' in source
        assert "sampling_diversity=config.codex_sampling_diversity" in source
````

## Diff Stat

### Worktree Tracked Diff Stat

````text
 ttt_discover/codex_utils/discovery.py |   2 +
 ttt_discover/codex_utils/sampler.py   | 298 +++++++++++++++++++++++++++++++++-
 ttt_discover/discovery.py             |   2 +
 ttt_discover/rl/codex_no_finetune.py  |   2 +
 4 files changed, 300 insertions(+), 4 deletions(-)
````
## Raw Diff

<details>
<summary>Tracked worktree patch</summary>

````diff
diff --git a/ttt_discover/codex_utils/discovery.py b/ttt_discover/codex_utils/discovery.py
index b21a80f..855d3e1 100644
--- a/ttt_discover/codex_utils/discovery.py
+++ b/ttt_discover/codex_utils/discovery.py
@@ -50,6 +50,7 @@ class DiscoverConfig:
     codex_initial_program_paths: tuple[str, ...] = ()
     codex_initial_pool_paths: tuple[str, ...] = ()
     codex_autonomous: bool = False
+    codex_sampling_diversity: Literal["off", "parent_birth_gap_surface"] = "off"
 
 
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
index 5c5d4d1..ba98654 100644
--- a/ttt_discover/codex_utils/sampler.py
+++ b/ttt_discover/codex_utils/sampler.py
@@ -4,11 +4,12 @@ from abc import ABC, abstractmethod
 from contextlib import contextmanager
 import json
 import logging
+from numbers import Integral
 import os
 from pathlib import Path
 import threading
 import time
-from typing import Any, Callable
+from typing import Any, Callable, Literal
 
 import numpy as np
 
@@ -16,6 +17,80 @@ from ttt_discover.codex_utils.runtime import State, state_from_dict, to_json_ser
 
 logger = logging.getLogger(__name__)
 
+SamplingDiversity = Literal["off", "parent_birth_gap_surface"]
+
+PARENT_BIRTH_GAP_SURFACES = (
+    "root_or_seed",
+    "gap_0",
+    "gap_1",
+    "gap_2_4",
+    "gap_5_plus",
+    "unknown",
+)
+PARENT_BIRTH_GAP_SURFACE_SAMPLE_COUNTS_KEY = "puct_parent_birth_gap_surface_sample_counts"
+_PARENT_BIRTH_GAP_SURFACE_ORDER = {
+    surface: idx for idx, surface in enumerate(PARENT_BIRTH_GAP_SURFACES)
+}
+
+
+def _validate_sampling_diversity(value: str) -> SamplingDiversity:
+    if value not in ("off", "parent_birth_gap_surface"):
+        raise ValueError(
+            "sampling_diversity must be 'off' or 'parent_birth_gap_surface', "
+            f"got {value!r}"
+        )
+    return value
+
+
+def _zero_parent_birth_gap_surface_counts() -> dict[str, int]:
+    return {surface: 0 for surface in PARENT_BIRTH_GAP_SURFACES}
+
+
+def _sanitize_parent_birth_gap_surface_counts(raw: Any) -> dict[str, int]:
+    counts = _zero_parent_birth_gap_surface_counts()
+    if not isinstance(raw, dict):
+        return counts
+    for surface in PARENT_BIRTH_GAP_SURFACES:
+        value = raw.get(surface, 0)
+        if isinstance(value, bool) or not isinstance(value, Integral):
+            counts[surface] = 0
+        else:
+            counts[surface] = max(0, int(value))
+    return counts
+
+
+def _is_integral_timestep(value: Any) -> bool:
+    return not isinstance(value, bool) and isinstance(value, Integral)
+
+
+def parent_birth_gap_surface_for_state(state: Any, initial_ids: set[str]) -> str:
+    """Classify a parent using only id, timestep, and immediate parent timestep."""
+    state_id = getattr(state, "id", None)
+    if state_id in initial_ids:
+        return "root_or_seed"
+
+    timestep = getattr(state, "timestep", None)
+    if _is_integral_timestep(timestep) and int(timestep) < 0:
+        return "root_or_seed"
+
+    parents = getattr(state, "parents", None)
+    if not parents:
+        return "root_or_seed"
+
+    parent = parents[0]
+    parent_timestep = parent.get("timestep") if isinstance(parent, dict) else None
+    if not _is_integral_timestep(timestep) or not _is_integral_timestep(parent_timestep):
+        return "unknown"
+
+    gap = int(timestep) - int(parent_timestep)
+    if gap <= 0:
+        return "gap_0"
+    if gap == 1:
+        return "gap_1"
+    if 2 <= gap <= 4:
+        return "gap_2_4"
+    return "gap_5_plus"
+
 
 @contextmanager
 def _file_lock(lock_path: str, *, poll_s: float = 0.05, stale_s: float = 600.0):
@@ -353,6 +428,7 @@ class PUCTSampler(StateSampler):
         resume_step: int | None = None,
         puct_c: float = 1.0,
         topk_children: int = 2,
+        sampling_diversity: SamplingDiversity = "off",
     ):
         self.file_path = file_path
         self.env_type = env_type
@@ -361,6 +437,7 @@ class PUCTSampler(StateSampler):
         self.batch_size = batch_size
         self.topk_children = topk_children
         self.puct_c = float(puct_c)
+        self.sampling_diversity = _validate_sampling_diversity(sampling_diversity)
         
         self._states: list[State] = []
         self._initial_states: list[State] = []
@@ -375,6 +452,10 @@ class PUCTSampler(StateSampler):
         self._T: int = 0
         self._last_scale: float = 1.0
         self._last_puct_stats: list[tuple[int, float, float, float, float]] = []
+        self._parent_birth_gap_surface_sample_counts = _zero_parent_birth_gap_surface_counts()
+        self._last_parent_birth_gap_surface_rows: list[tuple[str, int, int, str]] = []
+        self._last_parent_birth_gap_surface_selected_counts = _zero_parent_birth_gap_surface_counts()
+        self._last_parent_birth_gap_surface_available_counts = _zero_parent_birth_gap_surface_counts()
         
         if resume_step is not None:
             self._load(resume_step)
@@ -399,6 +480,12 @@ class PUCTSampler(StateSampler):
         self._n = store.get("puct_n", {}) or {}
         self._m = store.get("puct_m", {}) or {}
         self._T = int(store.get("puct_T", 0) or 0)
+        if self._parent_birth_gap_surface_enabled():
+            self._parent_birth_gap_surface_sample_counts = (
+                _sanitize_parent_birth_gap_surface_counts(
+                    store.get(PARENT_BIRTH_GAP_SURFACE_SAMPLE_COUNTS_KEY, {})
+                )
+            )
 
     def _save(self, step: int):
         save_path = _sampler_file_for_step(self.file_path, step)
@@ -411,6 +498,10 @@ class PUCTSampler(StateSampler):
             "puct_m": self._m,
             "puct_T": self._T,
         }
+        if self._parent_birth_gap_surface_enabled():
+            store[PARENT_BIRTH_GAP_SURFACE_SAMPLE_COUNTS_KEY] = (
+                self._parent_birth_gap_surface_sample_counts
+            )
         with _file_lock(f"{save_path}.lock"):
             _atomic_write_json(save_path, store)
 
@@ -489,6 +580,158 @@ class PUCTSampler(StateSampler):
                     queue.append(child_id)
         return lineage
 
+    def _parent_birth_gap_surface_enabled(self) -> bool:
+        return self.sampling_diversity == "parent_birth_gap_surface"
+
+    def _reset_parent_birth_gap_surface_tracking(self) -> None:
+        self._last_parent_birth_gap_surface_rows = []
+        self._last_parent_birth_gap_surface_selected_counts = _zero_parent_birth_gap_surface_counts()
+        self._last_parent_birth_gap_surface_available_counts = _zero_parent_birth_gap_surface_counts()
+
+    def _iter_available_entries(
+        self,
+        scores: list[tuple],
+        *,
+        picked_ids: set[str],
+        blocked_ids: set[str],
+        block_lineage: bool,
+    ):
+        for entry in scores:
+            state = entry[2]
+            if state.id in picked_ids:
+                continue
+            if block_lineage and state.id in blocked_ids:
+                continue
+            yield entry
+
+    def _group_parent_birth_gap_surface_entries(
+        self,
+        scores: list[tuple],
+        *,
+        initial_ids: set[str],
+        picked_ids: set[str],
+        blocked_ids: set[str],
+        block_lineage: bool,
+    ) -> dict[str, list[tuple]]:
+        grouped: dict[str, list[tuple]] = {
+            surface: [] for surface in PARENT_BIRTH_GAP_SURFACES
+        }
+        for entry in self._iter_available_entries(
+            scores,
+            picked_ids=picked_ids,
+            blocked_ids=blocked_ids,
+            block_lineage=block_lineage,
+        ):
+            surface = parent_birth_gap_surface_for_state(entry[2], initial_ids)
+            grouped[surface].append(entry)
+        return grouped
+
+    def _choose_parent_birth_gap_surface(
+        self,
+        grouped: dict[str, list[tuple]],
+    ) -> str | None:
+        available_surfaces = [
+            surface for surface in PARENT_BIRTH_GAP_SURFACES if grouped.get(surface)
+        ]
+        if not available_surfaces:
+            return None
+        return min(
+            available_surfaces,
+            key=lambda surface: (
+                self._parent_birth_gap_surface_sample_counts.get(surface, 0),
+                _PARENT_BIRTH_GAP_SURFACE_ORDER[surface],
+            ),
+        )
+
+    def _record_parent_birth_gap_surface_selection(
+        self,
+        state: State,
+        *,
+        initial_ids: set[str],
+        selection: str,
+    ) -> None:
+        surface = parent_birth_gap_surface_for_state(state, initial_ids)
+        before = self._parent_birth_gap_surface_sample_counts.get(surface, 0)
+        after = before + 1
+        self._parent_birth_gap_surface_sample_counts[surface] = after
+        self._last_parent_birth_gap_surface_selected_counts[surface] += 1
+        self._last_parent_birth_gap_surface_rows.append(
+            (surface, before, after, selection)
+        )
+
+    def _sample_states_parent_birth_gap_surface(
+        self,
+        num_states: int,
+        scores: list[tuple],
+        initial_ids: set[str],
+    ) -> tuple[list[State], list[tuple]]:
+        self._reset_parent_birth_gap_surface_tracking()
+        picked: list[State] = []
+        top_scores: list[tuple] = []
+        picked_ids: set[str] = set()
+        blocked_ids: set[str] = set()
+        block_lineage = num_states > 1
+        children_map = self._build_children_map() if block_lineage else {}
+
+        initial_grouped = self._group_parent_birth_gap_surface_entries(
+            scores,
+            initial_ids=initial_ids,
+            picked_ids=set(),
+            blocked_ids=set(),
+            block_lineage=block_lineage,
+        )
+        self._last_parent_birth_gap_surface_available_counts = {
+            surface: len(initial_grouped[surface])
+            for surface in PARENT_BIRTH_GAP_SURFACES
+        }
+
+        while len(picked) < num_states:
+            grouped = self._group_parent_birth_gap_surface_entries(
+                scores,
+                initial_ids=initial_ids,
+                picked_ids=picked_ids,
+                blocked_ids=blocked_ids,
+                block_lineage=block_lineage,
+            )
+            surface = self._choose_parent_birth_gap_surface(grouped)
+            if surface is None:
+                break
+            entry = grouped[surface][0]
+            state = entry[2]
+            picked.append(state)
+            top_scores.append(entry)
+            picked_ids.add(state.id)
+            self._record_parent_birth_gap_surface_selection(
+                state,
+                initial_ids=initial_ids,
+                selection="balanced",
+            )
+            if block_lineage:
+                blocked_ids.update(self._get_full_lineage(state, children_map))
+
+        if len(picked) < num_states:
+            for entry in self._iter_available_entries(
+                scores,
+                picked_ids=picked_ids,
+                blocked_ids=blocked_ids,
+                block_lineage=block_lineage,
+            ):
+                state = entry[2]
+                picked.append(state)
+                top_scores.append(entry)
+                picked_ids.add(state.id)
+                self._record_parent_birth_gap_surface_selection(
+                    state,
+                    initial_ids=initial_ids,
+                    selection="fallback",
+                )
+                if block_lineage:
+                    blocked_ids.update(self._get_full_lineage(state, children_map))
+                if len(picked) >= num_states:
+                    break
+
+        return picked, top_scores
+
     def sample_states(self, num_states: int) -> list[State]:
         initial_ids = {s.id for s in self._initial_states}
         candidates = list(self._states)
@@ -501,6 +744,14 @@ class PUCTSampler(StateSampler):
             self._last_sampled_states = picked
             self._last_sampled_indices = []
             self._last_puct_stats = [(0, 0.0, 0.0, 0.0, 0.0) for _ in picked]
+            if self._parent_birth_gap_surface_enabled():
+                self._reset_parent_birth_gap_surface_tracking()
+                for state in picked:
+                    self._record_parent_birth_gap_surface_selection(
+                        state,
+                        initial_ids=initial_ids,
+                        selection="fallback",
+                    )
             return picked
 
         vals = np.array([float(s.value if s.value is not None else float("-inf")) for s in candidates])
@@ -521,7 +772,13 @@ class PUCTSampler(StateSampler):
 
         scores.sort(key=lambda x: (x[0], x[1]), reverse=True)
 
-        if num_states > 1:
+        if self._parent_birth_gap_surface_enabled():
+            picked, top_scores = self._sample_states_parent_birth_gap_surface(
+                num_states,
+                scores,
+                initial_ids,
+            )
+        elif num_states > 1:
             children_map = self._build_children_map()
             picked, top_scores, blocked_ids = [], [], set()
             for entry in scores:
@@ -731,21 +988,50 @@ class PUCTSampler(StateSampler):
         stats.update(_stats(sampled_values, "puct/sampled_value"))
         stats.update(_stats(sampled_timesteps, "puct/sampled_timestep"))
         stats.update(_stats(sampled_constr_lens, "puct/sampled_construction_len"))
+        if self._parent_birth_gap_surface_enabled():
+            prefix = "puct/parent_birth_gap_surface"
+            stats[f"{prefix}/enabled"] = 1
+            for surface in PARENT_BIRTH_GAP_SURFACES:
+                stats[f"{prefix}/count_{surface}"] = int(
+                    self._parent_birth_gap_surface_sample_counts.get(surface, 0)
+                )
+                stats[f"{prefix}/selected_{surface}"] = int(
+                    self._last_parent_birth_gap_surface_selected_counts.get(surface, 0)
+                )
+                stats[f"{prefix}/available_{surface}"] = int(
+                    self._last_parent_birth_gap_surface_available_counts.get(surface, 0)
+                )
         return stats
 
     def get_sample_table(self) -> tuple[list[str], list[tuple]]:
         columns = ["buffer_idx", "timestep", "value", "terminal_value", "parent_value", "construction_len", "observation_len", "n", "Q", "P", "bonus", "score"]
+        if self._parent_birth_gap_surface_enabled():
+            columns = columns + [
+                "parent_birth_gap_surface",
+                "parent_birth_gap_surface_count_before",
+                "parent_birth_gap_surface_count_after",
+                "parent_birth_gap_surface_selection",
+            ]
         rows = []
         if not self._last_sampled_states:
             return columns, rows
         indices = self._last_sampled_indices if len(self._last_sampled_indices) == len(self._last_sampled_states) else [-1] * len(self._last_sampled_states)
         stats = self._last_puct_stats if len(self._last_puct_stats) == len(self._last_sampled_states) else [(0, 0.0, 0.0, 0.0, 0.0)] * len(self._last_sampled_states)
-        for idx, state, (n, Q, P, bonus, score) in zip(indices, self._last_sampled_states, stats):
+        diversity_rows = self._last_parent_birth_gap_surface_rows
+        if len(diversity_rows) != len(self._last_sampled_states):
+            diversity_rows = [
+                ("unknown", 0, 0, "fallback")
+                for _ in self._last_sampled_states
+            ]
+        for row_idx, (idx, state, (n, Q, P, bonus, score)) in enumerate(zip(indices, self._last_sampled_states, stats)):
             parent_val = state.parent_values[0] if state.parent_values else None
             constr = getattr(state, 'construction', None)
             constr_len = len(constr) if constr is not None else 0
             obs_len = len(state.observation) if state.observation else 0
-            rows.append((idx, state.timestep, state.value, 0, parent_val, constr_len, obs_len, n, Q, P, bonus, score))
+            row = (idx, state.timestep, state.value, 0, parent_val, constr_len, obs_len, n, Q, P, bonus, score)
+            if self._parent_birth_gap_surface_enabled():
+                row = row + diversity_rows[row_idx]
+            rows.append(row)
         return columns, rows
 
 
@@ -756,6 +1042,7 @@ def create_sampler(
     batch_size: int = 1,
     resume_step: int | None = None,
     topk_children: int = 2,
+    sampling_diversity: SamplingDiversity = "off",
 ) -> StateSampler:
     """Factory function to create samplers. Pass the env type (from config.env_type)."""
     if not log_path:
@@ -768,6 +1055,7 @@ def create_sampler(
         batch_size=batch_size,
         resume_step=resume_step,
         topk_children=topk_children,
+        sampling_diversity=sampling_diversity,
     )
 
 
@@ -778,6 +1066,7 @@ def get_or_create_sampler_with_default(
     batch_size: int = 1,
     resume_step: int | None = None,
     topk_children: int = 2,
+    sampling_diversity: SamplingDiversity = "off",
 ) -> StateSampler:
     """Get sampler. Initial experience is created via env_type.create_initial_state."""
     return create_sampler(
@@ -787,4 +1076,5 @@ def get_or_create_sampler_with_default(
         batch_size=batch_size,
         resume_step=resume_step,
         topk_children=topk_children,
+        sampling_diversity=sampling_diversity,
     )
diff --git a/ttt_discover/discovery.py b/ttt_discover/discovery.py
index 6cdb434..20935a4 100644
--- a/ttt_discover/discovery.py
+++ b/ttt_discover/discovery.py
@@ -63,6 +63,7 @@ class DiscoverConfig:
     codex_initial_program_paths: tuple[str, ...] = ()
     codex_initial_pool_paths: tuple[str, ...] = ()
     codex_autonomous: bool = False
+    codex_sampling_diversity: Literal["off", "parent_birth_gap_surface"] = "off"
 
 
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
index 18a0e9e..291a22e 100644
--- a/ttt_discover/rl/codex_no_finetune.py
+++ b/ttt_discover/rl/codex_no_finetune.py
@@ -137,6 +137,7 @@ class CodexNoFinetuneConfig:
     initial_program_paths: tuple[str, ...] = ()
     initial_pool_paths: tuple[str, ...] = ()
     topk_children: int = 16
+    sampling_diversity: Literal["off", "parent_birth_gap_surface"] = "off"
 
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

