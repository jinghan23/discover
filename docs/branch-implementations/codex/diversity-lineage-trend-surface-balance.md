# codex/diversity-lineage-trend-surface-balance

## Summary

读取当前 value 与最近 parent_values 的趋势，分类为 no_history/flat/up_streak/down_streak/reversal/mixed，并平衡不同 lineage value trend。

## Branch State

- Worktree: `/opt/tiger/discover-lineage-trend-surface-balance`
- HEAD: `2234282`
- Base used for comparison: `2234282`
- Commits ahead of base: `0`
- Commits behind base: `0`
- Group: `diversity_extra`
- Implementation location: `uncommitted worktree diff`

## Implemented Behavior

- The summary above is based on reading the changed Python code, added config fields, and sampler/prompt hooks in this branch.
- For sampler branches, the common pattern is to keep baseline PUCT ordering inside a chosen surface/bucket, while changing which surface/bucket gets the next parent pick.
- Tests detected: `14` test functions across untracked test files.

## Added Markers

### Config fields

- `codex_lineage_trend_surface_balance`
- `values`
- `signs`
- `lineage_trend_surface_balance`

### Constants

- `LINEAGE_TREND_SURFACES`
- `LINEAGE_TREND_SURFACE_SET`

### Classes

- None

### Functions

- `_usable_trend_values`
- `lineage_trend_surface`
- `_sanitize_lineage_trend_surface_counts`
- `_select_lineage_trend_surface_balanced`
- `eligible`
- `pick`
- `_increment_lineage_trend_surface_counts`

## Diff Summary

- Worktree tracked shortstat: `5 files changed, 218 insertions(+), 16 deletions(-)`
- Untracked files: `3`

### Worktree Status

````text
 M ttt_discover/codex_utils/__init__.py
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
M	ttt_discover/codex_utils/__init__.py
M	ttt_discover/codex_utils/discovery.py
M	ttt_discover/codex_utils/sampler.py
M	ttt_discover/discovery.py
M	ttt_discover/rl/codex_no_finetune.py
````
### Untracked Files

- `repro/gpu_mode/run_0608_lineage_trend_surface_balance.sh (401 bytes)`
- `repro/run_discovery.py (3606 bytes)`
- `tests/test_lineage_trend_surface_balance.py (13119 bytes)`

### Detected Test Functions

- `tests/test_lineage_trend_surface_balance.py::test_no_history_for_zero_or_one_usable_edge`
- `tests/test_lineage_trend_surface_balance.py::test_exact_surfaces`
- `tests/test_lineage_trend_surface_balance.py::test_zero_signs_are_neutral_except_all_zero_flat`
- `tests/test_lineage_trend_surface_balance.py::test_same_sign_pattern_ignores_magnitude`
- `tests/test_lineage_trend_surface_balance.py::test_missing_nan_and_nonfinite_stop_history`
- `tests/test_lineage_trend_surface_balance.py::test_default_off_matches_baseline_order_and_does_not_increment_counts`
- `tests/test_lineage_trend_surface_balance.py::test_enabled_least_sampled_surface_precedes_repeated_high_puct_surface`
- `tests/test_lineage_trend_surface_balance.py::test_within_one_surface_candidate_order_remains_baseline_puct`
- `tests/test_lineage_trend_surface_balance.py::test_full_lineage_blocking_honored_in_balanced_pass`
- `tests/test_lineage_trend_surface_balance.py::test_fallback_fills_underfilled_batch_without_duplicates`
- `tests/test_lineage_trend_surface_balance.py::test_counts_increment_persist_reload_and_old_json_loads`
- `tests/test_lineage_trend_surface_balance.py::test_update_states_sets_parent_values_from_parent_lineage`
- `tests/test_lineage_trend_surface_balance.py::test_stats_and_table_include_trend_surface_values`
- `tests/test_lineage_trend_surface_balance.py::test_config_and_cli_pass_through`

## Untracked File Snapshots

These files are not part of `git diff` yet, but they are part of the worktree implementation state recorded here.

### `repro/gpu_mode/run_0608_lineage_trend_surface_balance.sh`

````bash
#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

if [[ "${1:-}" == "--dry-run" ]]; then
  shift
  python repro/run_discovery.py dry-run \
    --task trimul \
    --codex-lineage-trend-surface-balance \
    "$@"
else
  python repro/run_discovery.py run \
    --task trimul \
    --codex-lineage-trend-surface-balance \
    "$@"
fi
````

### `repro/run_discovery.py`

````python
from __future__ import annotations

import argparse

from ttt_discover import DiscoverConfig, discover


TASK_CHOICES = ("trimul", "mla_decode_nvidia")


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--task", choices=TASK_CHOICES, default="trimul")
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--groups-per-batch", type=int, default=1)
    parser.add_argument("--group-size", type=int, default=1)
    parser.add_argument("--eval-timeout", type=int, default=1200)
    parser.add_argument("--codex-backend", choices=("cli", "responses"), default="cli")
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
    parser.add_argument("--codex-autonomous", action="store_true")
    parser.add_argument(
        "--codex-lineage-trend-surface-balance",
        action="store_true",
        default=False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or dry-run Codex GPUMode discovery.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("dry-run", "run"):
        subparser = subparsers.add_parser(command)
        _add_common_args(subparser)
    return parser.parse_args()


def build_config(args: argparse.Namespace, *, env_type: type | None) -> DiscoverConfig:
    experiment_name = args.experiment_name or f"gpu-mode-{args.task}-codex"
    return DiscoverConfig(
        runner="codex_no_finetune",
        env_type=env_type,
        problem_type=args.task,
        experiment_name=experiment_name,
        wandb_project=args.wandb_project,
        num_epochs=args.num_epochs,
        groups_per_batch=args.groups_per_batch,
        group_size=args.group_size,
        num_cpus_per_task=1,
        eval_timeout=args.eval_timeout,
        codex_backend=args.codex_backend,
        codex_model_name=args.codex_model_name,
        codex_max_output_tokens=args.codex_max_output_tokens,
        codex_temperature=args.codex_temperature,
        codex_cli_command=args.codex_cli_command,
        codex_cli_sandbox=args.codex_cli_sandbox,
        codex_cli_timeout=args.codex_cli_timeout,
        codex_max_concurrent_requests=args.codex_max_concurrent_requests,
        codex_autonomous=args.codex_autonomous,
        codex_lineage_trend_surface_balance=args.codex_lineage_trend_surface_balance,
    )


def print_dry_run(config: DiscoverConfig) -> None:
    print("dry_run: true")
    print(f"task: {config.problem_type}")
    print(f"runner: {config.runner}")
    print(
        "codex_lineage_trend_surface_balance: "
        f"{config.codex_lineage_trend_surface_balance}"
    )


def main() -> None:
    args = parse_args()
    if args.command == "dry-run":
        print_dry_run(build_config(args, env_type=None))
        return

    from examples.gpu_mode.env import GpuModeEnv

    discover(build_config(args, env_type=GpuModeEnv))


if __name__ == "__main__":
    main()
````

### `tests/test_lineage_trend_surface_balance.py`

````python
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from ttt_discover.codex_utils.runtime import State
from ttt_discover.codex_utils.sampler import (
    LINEAGE_TREND_SURFACES,
    PUCTSampler,
    lineage_trend_surface,
)
from ttt_discover.rl.codex_no_finetune import CodexNoFinetuneConfig, _build_sampler


class DummyEnv:
    state_type = State
    _initial_counter = 0

    @classmethod
    def create_initial_state(cls, problem_type: str) -> State:
        cls._initial_counter += 1
        return State(
            timestep=0,
            construction=[f"initial-{cls._initial_counter}"],
            code="",
            value=0.0,
            id=f"initial-{cls._initial_counter}",
        )


def make_state(
    state_id: str,
    value: float | None,
    parent_values: list[float] | None = None,
    *,
    parents: list[dict] | None = None,
    timestep: int = 1,
) -> State:
    return State(
        timestep=timestep,
        construction=[state_id],
        code=state_id,
        value=value,
        parent_values=parent_values or [],
        parents=parents or [],
        id=state_id,
    )


def make_sampler(
    tmp: str,
    states: list[State],
    *,
    enabled: bool = False,
) -> PUCTSampler:
    sampler = PUCTSampler(
        file_path=os.path.join(tmp, "sampler.json"),
        env_type=DummyEnv,
        batch_size=0,
        topk_children=0,
        lineage_trend_surface_balance=enabled,
    )
    sampler._states = list(states)
    sampler._initial_states = []
    return sampler


class LineageTrendSurfaceClassifierTests(unittest.TestCase):
    def test_no_history_for_zero_or_one_usable_edge(self):
        self.assertEqual(lineage_trend_surface(make_state("none", None, [1.0, 2.0])), "trend_no_history")
        self.assertEqual(lineage_trend_surface(make_state("zero", 1.0, [])), "trend_no_history")
        self.assertEqual(lineage_trend_surface(make_state("one", 2.0, [1.0])), "trend_no_history")

    def test_exact_surfaces(self):
        self.assertEqual(lineage_trend_surface(make_state("flat", 1.0, [1.0, 1.0])), "trend_flat")
        self.assertEqual(lineage_trend_surface(make_state("up", 3.0, [2.0, 1.0])), "trend_up_streak")
        self.assertEqual(lineage_trend_surface(make_state("down", 1.0, [2.0, 3.0])), "trend_down_streak")
        self.assertEqual(lineage_trend_surface(make_state("rev_up", 3.0, [2.0, 4.0])), "trend_reversal_up")
        self.assertEqual(lineage_trend_surface(make_state("rev_down", 2.0, [3.0, 1.0])), "trend_reversal_down")
        self.assertEqual(lineage_trend_surface(make_state("mixed", 4.0, [3.0, 5.0, 2.0])), "trend_mixed")

    def test_zero_signs_are_neutral_except_all_zero_flat(self):
        self.assertEqual(lineage_trend_surface(make_state("all_zero", 2.0, [2.0, 2.0])), "trend_flat")
        self.assertEqual(lineage_trend_surface(make_state("leading_zero", 3.0, [3.0, 2.0])), "trend_up_streak")
        self.assertEqual(lineage_trend_surface(make_state("trailing_zero", 3.0, [2.0, 2.0])), "trend_up_streak")
        self.assertEqual(lineage_trend_surface(make_state("middle_zero", 3.0, [2.0, 2.0, 4.0])), "trend_reversal_up")

    def test_same_sign_pattern_ignores_magnitude(self):
        small = make_state("small", 3.0, [2.0, 1.0])
        huge = make_state("huge", 1e100, [0.0, -1e100])
        self.assertEqual(lineage_trend_surface(small), lineage_trend_surface(huge))

    def test_missing_nan_and_nonfinite_stop_history(self):
        self.assertEqual(lineage_trend_surface(make_state("none_stop", 1.0, [None, 0.0])), "trend_no_history")
        self.assertEqual(lineage_trend_surface(make_state("nan_stop", 1.0, [2.0, float("nan"), -1e99])), "trend_no_history")
        self.assertEqual(lineage_trend_surface(make_state("inf_stop", 3.0, [2.0, float("inf"), -1e99])), "trend_no_history")


class LineageTrendSurfaceSamplerTests(unittest.TestCase):
    def test_default_off_matches_baseline_order_and_does_not_increment_counts(self):
        states = [
            make_state("low", 1.0, [0.0, -1.0]),
            make_state("high", 5.0, [4.0, 3.0]),
            make_state("mid", 3.0, [2.0, 1.0]),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            sampler = make_sampler(tmp, states, enabled=False)
            picked = sampler.sample_states(2)
        self.assertEqual([state.id for state in picked], ["high", "mid"])
        self.assertEqual(sampler.lineage_trend_surface_counts, {})

    def test_enabled_least_sampled_surface_precedes_repeated_high_puct_surface(self):
        high = make_state("high_up", 100.0, [90.0, 80.0])
        low = make_state("low_down", 1.0, [2.0, 3.0])
        with tempfile.TemporaryDirectory() as tmp:
            sampler = make_sampler(tmp, [high, low], enabled=True)
            sampler.lineage_trend_surface_counts = {"trend_up_streak": 10}
            picked = sampler.sample_states(1)
        self.assertEqual([state.id for state in picked], ["low_down"])

    def test_within_one_surface_candidate_order_remains_baseline_puct(self):
        states = [
            make_state("up_mid", 8.0, [7.0, 6.0]),
            make_state("up_high", 9.0, [8.0, 7.0]),
            make_state("up_low", 7.0, [6.0, 5.0]),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            sampler = make_sampler(tmp, states, enabled=True)
            picked = sampler.sample_states(2)
        self.assertEqual([state.id for state in picked], ["up_high", "up_mid"])

    def test_full_lineage_blocking_honored_in_balanced_pass(self):
        ancestor = make_state("ancestor", 10.0, [9.0, 8.0])
        child = make_state(
            "child",
            9.0,
            [10.0, 11.0],
            parents=[{"id": "ancestor", "timestep": 1}],
        )
        unrelated = make_state("unrelated", 8.0, [8.0, 8.0])
        with tempfile.TemporaryDirectory() as tmp:
            sampler = make_sampler(tmp, [ancestor, child, unrelated], enabled=True)
            picked = sampler.sample_states(2)
            stats = sampler.get_sample_stats()
        self.assertEqual([state.id for state in picked], ["ancestor", "unrelated"])
        self.assertEqual(stats["puct/lineage_trend_surface_fallback_count_last"], 0)

    def test_fallback_fills_underfilled_batch_without_duplicates(self):
        ancestor = make_state("ancestor", 10.0, [9.0, 8.0])
        child = make_state(
            "child",
            9.0,
            [10.0, 11.0],
            parents=[{"id": "ancestor", "timestep": 1}],
        )
        with tempfile.TemporaryDirectory() as tmp:
            sampler = make_sampler(tmp, [ancestor, child], enabled=True)
            picked = sampler.sample_states(2)
            stats = sampler.get_sample_stats()
        self.assertEqual([state.id for state in picked], ["ancestor", "child"])
        self.assertEqual(len({state.id for state in picked}), 2)
        self.assertEqual(stats["puct/lineage_trend_surface_fallback_count_last"], 1)

    def test_counts_increment_persist_reload_and_old_json_loads(self):
        states = [
            make_state("up", 3.0, [2.0, 1.0]),
            make_state("down", 1.0, [2.0, 3.0]),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            sampler = make_sampler(tmp, states, enabled=True)
            sampler.sample_states(2)
            self.assertEqual(sampler.lineage_trend_surface_counts["trend_up_streak"], 1)
            self.assertEqual(sampler.lineage_trend_surface_counts["trend_down_streak"], 1)
            sampler.flush(step=1)

            loaded = PUCTSampler(
                file_path=os.path.join(tmp, "sampler.json"),
                env_type=DummyEnv,
                batch_size=0,
                resume_step=1,
                lineage_trend_surface_balance=True,
            )
            self.assertEqual(loaded.lineage_trend_surface_counts["trend_up_streak"], 1)
            self.assertEqual(loaded.lineage_trend_surface_counts["trend_down_streak"], 1)

            old_path = Path(tmp) / "old_step_000002.json"
            old_path.write_text(
                json.dumps(
                    {
                        "step": 2,
                        "states": [],
                        "initial_states": [],
                        "puct_n": {},
                        "puct_m": {},
                        "puct_T": 0,
                    }
                ),
                encoding="utf-8",
            )
            old_loaded = PUCTSampler(
                file_path=os.path.join(tmp, "old.json"),
                env_type=DummyEnv,
                batch_size=0,
                resume_step=2,
                lineage_trend_surface_balance=True,
            )
            self.assertEqual(old_loaded.lineage_trend_surface_counts, {})

            bad_path = Path(tmp) / "bad_step_000003.json"
            bad_path.write_text(
                json.dumps(
                    {
                        "step": 3,
                        "states": [],
                        "initial_states": [],
                        "puct_n": {},
                        "puct_m": {},
                        "puct_T": 0,
                        "lineage_trend_surface_counts": {
                            "trend_flat": -5,
                            "trend_up_streak": "2",
                            "trend_mixed": "bad",
                            "unknown": 99,
                        },
                    }
                ),
                encoding="utf-8",
            )
            bad_loaded = PUCTSampler(
                file_path=os.path.join(tmp, "bad.json"),
                env_type=DummyEnv,
                batch_size=0,
                resume_step=3,
                lineage_trend_surface_balance=True,
            )
            self.assertEqual(bad_loaded.lineage_trend_surface_counts["trend_flat"], 0)
            self.assertEqual(bad_loaded.lineage_trend_surface_counts["trend_up_streak"], 2)
            self.assertEqual(bad_loaded.lineage_trend_surface_counts["trend_mixed"], 0)
            self.assertNotIn("unknown", bad_loaded.lineage_trend_surface_counts)

    def test_update_states_sets_parent_values_from_parent_lineage(self):
        parent = make_state("parent", 5.0, [4.0, 3.0])
        child = make_state("child", 6.0, [])
        with tempfile.TemporaryDirectory() as tmp:
            sampler = make_sampler(tmp, [], enabled=True)
            sampler.update_states([child], [parent], save=False)
        self.assertEqual(child.parent_values, [5.0, 4.0, 3.0])

    def test_stats_and_table_include_trend_surface_values(self):
        states = [
            make_state("up", 3.0, [2.0, 1.0]),
            make_state("flat", 2.0, [2.0, 2.0]),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            sampler = make_sampler(tmp, states, enabled=True)
            picked = sampler.sample_states(2)
            stats = sampler.get_sample_stats()
            columns, rows = sampler.get_sample_table()

        for surface in LINEAGE_TREND_SURFACES:
            self.assertIn(f"puct/lineage_trend_surface_count/{surface}", stats)
        self.assertIn("puct/lineage_trend_surface_fallback_count_last", stats)
        self.assertIn("lineage_trend_surface", columns)
        surface_idx = columns.index("lineage_trend_surface")
        self.assertEqual(
            [row[surface_idx] for row in rows],
            [lineage_trend_surface(state) for state in picked],
        )

    def test_config_and_cli_pass_through(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = CodexNoFinetuneConfig(
                env_type=DummyEnv,
                log_path=tmp,
                groups_per_batch=0,
                lineage_trend_surface_balance=True,
            )
            sampler = _build_sampler(cfg, start_batch=0)
            self.assertTrue(sampler.lineage_trend_surface_balance)

        from ttt_discover.codex_utils.discovery import DiscoverConfig as CodexDiscoverConfig

        self.assertTrue(
            CodexDiscoverConfig(codex_lineage_trend_surface_balance=True).codex_lineage_trend_surface_balance
        )

        repo_root = Path(__file__).resolve().parents[1]
        legacy_discovery = (repo_root / "ttt_discover/discovery.py").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "codex_lineage_trend_surface_balance: bool = False",
            legacy_discovery,
        )
        self.assertIn(
            "lineage_trend_surface_balance=config.codex_lineage_trend_surface_balance",
            legacy_discovery,
        )
        dry_run = subprocess.run(
            [
                sys.executable,
                "repro/run_discovery.py",
                "dry-run",
                "--task",
                "trimul",
                "--codex-lineage-trend-surface-balance",
            ],
            cwd=repo_root,
            check=True,
            text=True,
            capture_output=True,
        )
        self.assertIn("codex_lineage_trend_surface_balance: True", dry_run.stdout)


if __name__ == "__main__":
    unittest.main()
````

## Diff Stat

### Worktree Tracked Diff Stat

````text
 ttt_discover/codex_utils/__init__.py  |   2 +
 ttt_discover/codex_utils/discovery.py |   2 +
 ttt_discover/codex_utils/sampler.py   | 226 +++++++++++++++++++++++++++++++---
 ttt_discover/discovery.py             |   2 +
 ttt_discover/rl/codex_no_finetune.py  |   2 +
 5 files changed, 218 insertions(+), 16 deletions(-)
````
## Raw Diff

<details>
<summary>Tracked worktree patch</summary>

````diff
diff --git a/ttt_discover/codex_utils/__init__.py b/ttt_discover/codex_utils/__init__.py
index 1a51326..f275b02 100644
--- a/ttt_discover/codex_utils/__init__.py
+++ b/ttt_discover/codex_utils/__init__.py
@@ -11,6 +11,7 @@ from ttt_discover.codex_utils.sampler import (
     StateSampler,
     create_sampler,
     get_or_create_sampler_with_default,
+    lineage_trend_surface,
 )
 
 __all__ = [
@@ -26,6 +27,7 @@ __all__ = [
     "create_sampler",
     "discover",
     "get_or_create_sampler_with_default",
+    "lineage_trend_surface",
     "state_from_dict",
     "to_json_serializable",
 ]
diff --git a/ttt_discover/codex_utils/discovery.py b/ttt_discover/codex_utils/discovery.py
index b21a80f..09605c8 100644
--- a/ttt_discover/codex_utils/discovery.py
+++ b/ttt_discover/codex_utils/discovery.py
@@ -50,6 +50,7 @@ class DiscoverConfig:
     codex_initial_program_paths: tuple[str, ...] = ()
     codex_initial_pool_paths: tuple[str, ...] = ()
     codex_autonomous: bool = False
+    codex_lineage_trend_surface_balance: bool = False
 
 
 def _run_codex_no_finetune(config: DiscoverConfig) -> None:
@@ -85,6 +86,7 @@ def _run_codex_no_finetune(config: DiscoverConfig) -> None:
         initial_program_paths=config.codex_initial_program_paths,
         initial_pool_paths=config.codex_initial_pool_paths,
         autonomous=config.codex_autonomous,
+        lineage_trend_surface_balance=config.codex_lineage_trend_surface_balance,
         wandb_project=config.wandb_project,
         wandb_name=experiment_name,
         log_path=log_path,
diff --git a/ttt_discover/codex_utils/sampler.py b/ttt_discover/codex_utils/sampler.py
index 5c5d4d1..d69594c 100644
--- a/ttt_discover/codex_utils/sampler.py
+++ b/ttt_discover/codex_utils/sampler.py
@@ -4,6 +4,7 @@ from abc import ABC, abstractmethod
 from contextlib import contextmanager
 import json
 import logging
+import math
 import os
 from pathlib import Path
 import threading
@@ -17,6 +18,68 @@ from ttt_discover.codex_utils.runtime import State, state_from_dict, to_json_ser
 logger = logging.getLogger(__name__)
 
 
+LINEAGE_TREND_SURFACES = (
+    "trend_no_history",
+    "trend_flat",
+    "trend_up_streak",
+    "trend_down_streak",
+    "trend_reversal_up",
+    "trend_reversal_down",
+    "trend_mixed",
+)
+LINEAGE_TREND_SURFACE_SET = set(LINEAGE_TREND_SURFACES)
+
+
+def _usable_trend_values(state: State) -> list[float]:
+    parent_values = state.parent_values if state.parent_values is not None else []
+    raw_values = [state.value] + list(parent_values[:3])
+    values: list[float] = []
+    for value in raw_values:
+        if value is None:
+            break
+        try:
+            numeric = float(value)
+        except (TypeError, ValueError, OverflowError):
+            break
+        if not math.isfinite(numeric):
+            break
+        values.append(numeric)
+    return values
+
+
+def lineage_trend_surface(state: State) -> str:
+    values = _usable_trend_values(state)
+    if len(values) < 3:
+        return "trend_no_history"
+
+    signs: list[str] = []
+    for newer, older in zip(values, values[1:]):
+        if newer > older:
+            signs.append("+")
+        elif newer < older:
+            signs.append("-")
+        else:
+            signs.append("0")
+
+    if all(sign == "0" for sign in signs):
+        return "trend_flat"
+
+    nonzero = [sign for sign in signs if sign != "0"]
+    if all(sign == "+" for sign in nonzero):
+        return "trend_up_streak"
+    if all(sign == "-" for sign in nonzero):
+        return "trend_down_streak"
+
+    changes = sum(
+        1
+        for prev, cur in zip(nonzero, nonzero[1:])
+        if prev != cur
+    )
+    if changes == 1:
+        return "trend_reversal_up" if nonzero[0] == "+" else "trend_reversal_down"
+    return "trend_mixed"
+
+
 @contextmanager
 def _file_lock(lock_path: str, *, poll_s: float = 0.05, stale_s: float = 600.0):
     while True:
@@ -353,6 +416,7 @@ class PUCTSampler(StateSampler):
         resume_step: int | None = None,
         puct_c: float = 1.0,
         topk_children: int = 2,
+        lineage_trend_surface_balance: bool = False,
     ):
         self.file_path = file_path
         self.env_type = env_type
@@ -361,6 +425,8 @@ class PUCTSampler(StateSampler):
         self.batch_size = batch_size
         self.topk_children = topk_children
         self.puct_c = float(puct_c)
+        self.lineage_trend_surface_balance = bool(lineage_trend_surface_balance)
+        self.lineage_trend_surface_counts: dict[str, int] = {}
         
         self._states: list[State] = []
         self._initial_states: list[State] = []
@@ -375,6 +441,7 @@ class PUCTSampler(StateSampler):
         self._T: int = 0
         self._last_scale: float = 1.0
         self._last_puct_stats: list[tuple[int, float, float, float, float]] = []
+        self._last_lineage_trend_surface_fallback_count = 0
         
         if resume_step is not None:
             self._load(resume_step)
@@ -385,6 +452,21 @@ class PUCTSampler(StateSampler):
                 self._states.append(state)
             self._save(self._current_step)
 
+    @staticmethod
+    def _sanitize_lineage_trend_surface_counts(raw_counts: Any) -> dict[str, int]:
+        if not isinstance(raw_counts, dict):
+            return {}
+        counts: dict[str, int] = {}
+        for surface, count in raw_counts.items():
+            if surface not in LINEAGE_TREND_SURFACE_SET:
+                continue
+            try:
+                sanitized = int(count)
+            except (TypeError, ValueError, OverflowError):
+                sanitized = 0
+            counts[str(surface)] = max(0, sanitized)
+        return counts
+
     def _load(self, step: int):
         file_path = _sampler_file_for_step(self.file_path, step)
         if not os.path.exists(file_path):
@@ -399,10 +481,16 @@ class PUCTSampler(StateSampler):
         self._n = store.get("puct_n", {}) or {}
         self._m = store.get("puct_m", {}) or {}
         self._T = int(store.get("puct_T", 0) or 0)
+        self.lineage_trend_surface_counts = self._sanitize_lineage_trend_surface_counts(
+            store.get("lineage_trend_surface_counts", {}) or {}
+        )
 
     def _save(self, step: int):
         save_path = _sampler_file_for_step(self.file_path, step)
         os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
+        self.lineage_trend_surface_counts = self._sanitize_lineage_trend_surface_counts(
+            self.lineage_trend_surface_counts
+        )
         store = {
             "step": step,
             "states": [s.to_dict() for s in self._states],
@@ -410,6 +498,7 @@ class PUCTSampler(StateSampler):
             "puct_n": self._n,
             "puct_m": self._m,
             "puct_T": self._T,
+            "lineage_trend_surface_counts": self.lineage_trend_surface_counts,
         }
         with _file_lock(f"{save_path}.lock"):
             _atomic_write_json(save_path, store)
@@ -489,6 +578,88 @@ class PUCTSampler(StateSampler):
                     queue.append(child_id)
         return lineage
 
+    def _select_lineage_trend_surface_balanced(
+        self,
+        scores: list[tuple],
+        num_states: int,
+    ) -> tuple[list[State], list[tuple], int]:
+        children_map = self._build_children_map()
+        grouped: dict[str, list[tuple[int, tuple]]] = {
+            surface: [] for surface in LINEAGE_TREND_SURFACES
+        }
+        for rank, entry in enumerate(scores):
+            surface = lineage_trend_surface(entry[2])
+            grouped[surface].append((rank, entry))
+
+        picked: list[State] = []
+        top_scores: list[tuple] = []
+        picked_ids: set[str] = set()
+        blocked_ids: set[str] = set()
+        local_counts: dict[str, int] = {}
+
+        def eligible(entry: tuple, *, use_blocking: bool) -> bool:
+            state = entry[2]
+            if state.id in picked_ids:
+                return False
+            return not use_blocking or state.id not in blocked_ids
+
+        def pick(entry: tuple, *, update_blocking: bool) -> None:
+            state = entry[2]
+            picked.append(state)
+            top_scores.append(entry)
+            picked_ids.add(state.id)
+            if update_blocking:
+                blocked_ids.update(self._get_full_lineage(state, children_map))
+
+        while len(picked) < num_states:
+            best: tuple[tuple[int, int], str, tuple] | None = None
+            for surface in LINEAGE_TREND_SURFACES:
+                first_entry: tuple[int, tuple] | None = None
+                for rank, entry in grouped[surface]:
+                    if eligible(entry, use_blocking=True):
+                        first_entry = (rank, entry)
+                        break
+                if first_entry is None:
+                    continue
+                rank, entry = first_entry
+                count = self.lineage_trend_surface_counts.get(surface, 0) + local_counts.get(surface, 0)
+                key = (count, rank)
+                if best is None or key < best[0]:
+                    best = (key, surface, entry)
+            if best is None:
+                break
+            _, surface, entry = best
+            pick(entry, update_blocking=True)
+            local_counts[surface] = local_counts.get(surface, 0) + 1
+
+        fallback_count = 0
+        if len(picked) < num_states:
+            for entry in scores:
+                if len(picked) >= num_states:
+                    break
+                if not eligible(entry, use_blocking=True):
+                    continue
+                pick(entry, update_blocking=True)
+                fallback_count += 1
+
+        if len(picked) < num_states:
+            for entry in scores:
+                if len(picked) >= num_states:
+                    break
+                if not eligible(entry, use_blocking=False):
+                    continue
+                pick(entry, update_blocking=False)
+                fallback_count += 1
+
+        return picked, top_scores, fallback_count
+
+    def _increment_lineage_trend_surface_counts(self, states: list[State]) -> None:
+        for state in states:
+            surface = lineage_trend_surface(state)
+            self.lineage_trend_surface_counts[surface] = (
+                self.lineage_trend_surface_counts.get(surface, 0) + 1
+            )
+
     def sample_states(self, num_states: int) -> list[State]:
         initial_ids = {s.id for s in self._initial_states}
         candidates = list(self._states)
@@ -501,6 +672,9 @@ class PUCTSampler(StateSampler):
             self._last_sampled_states = picked
             self._last_sampled_indices = []
             self._last_puct_stats = [(0, 0.0, 0.0, 0.0, 0.0) for _ in picked]
+            self._last_lineage_trend_surface_fallback_count = 0
+            if self.lineage_trend_surface_balance:
+                self._increment_lineage_trend_surface_counts(picked)
             return picked
 
         vals = np.array([float(s.value if s.value is not None else float("-inf")) for s in candidates])
@@ -521,21 +695,30 @@ class PUCTSampler(StateSampler):
 
         scores.sort(key=lambda x: (x[0], x[1]), reverse=True)
 
-        if num_states > 1:
-            children_map = self._build_children_map()
-            picked, top_scores, blocked_ids = [], [], set()
-            for entry in scores:
-                s = entry[2]
-                if s.id in blocked_ids:
-                    continue
-                picked.append(s)
-                top_scores.append(entry)
-                blocked_ids.update(self._get_full_lineage(s, children_map))
-                if len(picked) >= num_states:
-                    break
+        if self.lineage_trend_surface_balance:
+            picked, top_scores, fallback_count = self._select_lineage_trend_surface_balanced(
+                scores,
+                num_states,
+            )
+            self._last_lineage_trend_surface_fallback_count = fallback_count
+            self._increment_lineage_trend_surface_counts(picked)
         else:
-            top_scores = scores[:num_states]
-            picked = [t[2] for t in top_scores]
+            self._last_lineage_trend_surface_fallback_count = 0
+            if num_states > 1:
+                children_map = self._build_children_map()
+                picked, top_scores, blocked_ids = [], [], set()
+                for entry in scores:
+                    s = entry[2]
+                    if s.id in blocked_ids:
+                        continue
+                    picked.append(s)
+                    top_scores.append(entry)
+                    blocked_ids.update(self._get_full_lineage(s, children_map))
+                    if len(picked) >= num_states:
+                        break
+            else:
+                top_scores = scores[:num_states]
+                picked = [t[2] for t in top_scores]
 
         state_id_to_idx = {s.id: i for i, s in enumerate(self._states)}
         self._last_sampled_states = picked
@@ -724,7 +907,14 @@ class PUCTSampler(StateSampler):
             "puct/sampled_size": len(self._last_sampled_states),
             "puct/T": self._T,
             "puct/scale_last": float(self._last_scale),
+            "puct/lineage_trend_surface_fallback_count_last": int(
+                self._last_lineage_trend_surface_fallback_count
+            ),
         }
+        for surface in LINEAGE_TREND_SURFACES:
+            stats[f"puct/lineage_trend_surface_count/{surface}"] = int(
+                self.lineage_trend_surface_counts.get(surface, 0)
+            )
         stats.update(_stats(buffer_values, "puct/buffer_value"))
         stats.update(_stats(buffer_timesteps, "puct/buffer_timestep"))
         stats.update(_stats(buffer_constr_lens, "puct/buffer_construction_len"))
@@ -734,7 +924,7 @@ class PUCTSampler(StateSampler):
         return stats
 
     def get_sample_table(self) -> tuple[list[str], list[tuple]]:
-        columns = ["buffer_idx", "timestep", "value", "terminal_value", "parent_value", "construction_len", "observation_len", "n", "Q", "P", "bonus", "score"]
+        columns = ["buffer_idx", "timestep", "value", "terminal_value", "parent_value", "construction_len", "observation_len", "n", "Q", "P", "bonus", "score", "lineage_trend_surface"]
         rows = []
         if not self._last_sampled_states:
             return columns, rows
@@ -745,7 +935,7 @@ class PUCTSampler(StateSampler):
             constr = getattr(state, 'construction', None)
             constr_len = len(constr) if constr is not None else 0
             obs_len = len(state.observation) if state.observation else 0
-            rows.append((idx, state.timestep, state.value, 0, parent_val, constr_len, obs_len, n, Q, P, bonus, score))
+            rows.append((idx, state.timestep, state.value, 0, parent_val, constr_len, obs_len, n, Q, P, bonus, score, lineage_trend_surface(state)))
         return columns, rows
 
 
@@ -756,6 +946,7 @@ def create_sampler(
     batch_size: int = 1,
     resume_step: int | None = None,
     topk_children: int = 2,
+    lineage_trend_surface_balance: bool = False,
 ) -> StateSampler:
     """Factory function to create samplers. Pass the env type (from config.env_type)."""
     if not log_path:
@@ -768,6 +959,7 @@ def create_sampler(
         batch_size=batch_size,
         resume_step=resume_step,
         topk_children=topk_children,
+        lineage_trend_surface_balance=lineage_trend_surface_balance,
     )
 
 
@@ -778,6 +970,7 @@ def get_or_create_sampler_with_default(
     batch_size: int = 1,
     resume_step: int | None = None,
     topk_children: int = 2,
+    lineage_trend_surface_balance: bool = False,
 ) -> StateSampler:
     """Get sampler. Initial experience is created via env_type.create_initial_state."""
     return create_sampler(
@@ -787,4 +980,5 @@ def get_or_create_sampler_with_default(
         batch_size=batch_size,
         resume_step=resume_step,
         topk_children=topk_children,
+        lineage_trend_surface_balance=lineage_trend_surface_balance,
     )
diff --git a/ttt_discover/discovery.py b/ttt_discover/discovery.py
index 6cdb434..cddff95 100644
--- a/ttt_discover/discovery.py
+++ b/ttt_discover/discovery.py
@@ -63,6 +63,7 @@ class DiscoverConfig:
     codex_initial_program_paths: tuple[str, ...] = ()
     codex_initial_pool_paths: tuple[str, ...] = ()
     codex_autonomous: bool = False
+    codex_lineage_trend_surface_balance: bool = False
 
 
 def init_ray(num_cpus_per_task: int, env_type: str):
@@ -146,6 +147,7 @@ async def discover_impl(config: DiscoverConfig):
             initial_program_paths=config.codex_initial_program_paths,
             initial_pool_paths=config.codex_initial_pool_paths,
             autonomous=config.codex_autonomous,
+            lineage_trend_surface_balance=config.codex_lineage_trend_surface_balance,
             wandb_project=config.wandb_project,
             wandb_name=config.experiment_name,
             log_path=log_path,
diff --git a/ttt_discover/rl/codex_no_finetune.py b/ttt_discover/rl/codex_no_finetune.py
index 18a0e9e..a3bb2af 100644
--- a/ttt_discover/rl/codex_no_finetune.py
+++ b/ttt_discover/rl/codex_no_finetune.py
@@ -137,6 +137,7 @@ class CodexNoFinetuneConfig:
     initial_program_paths: tuple[str, ...] = ()
     initial_pool_paths: tuple[str, ...] = ()
     topk_children: int = 16
+    lineage_trend_surface_balance: bool = False
 
     # False is TTT Discover: non-auto, many short samples. True is AutoEvolve:
     # one Codex deep-dive workspace per sample with file edits and shell access.
@@ -960,6 +961,7 @@ def _build_sampler(cfg: CodexNoFinetuneConfig, start_batch: int) -> StateSampler
         batch_size=cfg.groups_per_batch,
         resume_step=start_batch if start_batch > 0 else None,
         topk_children=cfg.topk_children,
+        lineage_trend_surface_balance=cfg.lineage_trend_surface_balance,
     )
 
 
````
</details>

