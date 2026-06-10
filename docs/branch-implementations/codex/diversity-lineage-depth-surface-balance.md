# codex/diversity-lineage-depth-surface-balance

## Summary

按 parent 链长度分类 root/shallow/middle/deep，并持久记录各 depth surface 的采样次数来做 PUCT 表面平衡。

## Branch State

- Worktree: `/opt/tiger/discover-lineage-depth-surface-balance`
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

- `codex_sampler_lineage_depth_balance`
- `lineage_depth_balance`
- `sampler_lineage_depth_balance`

### Constants

- `LINEAGE_DEPTH_SURFACES`
- `LINEAGE_DEPTH_BALANCE_COUNTS_KEY`

### Classes

- None

### Functions

- `classify_lineage_depth_surface`
- `_zero_lineage_depth_balance_counts`
- `_sanitize_lineage_depth_balance_counts`

## Diff Summary

- Worktree tracked shortstat: `4 files changed, 138 insertions(+), 3 deletions(-)`
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

- `repro/gpu_mode/run_0609_lineage_depth_balance.sh (440 bytes)`
- `repro/run_discovery.py (4610 bytes)`
- `tests/test_codex_lineage_depth_balance.py (12211 bytes)`

### Detected Test Functions

- `tests/test_codex_lineage_depth_balance.py::test_disabled_parity_no_lineage_observability_or_persistence`
- `tests/test_codex_lineage_depth_balance.py::test_classifier_bins_use_parent_list_depths`
- `tests/test_codex_lineage_depth_balance.py::test_classifier_static_source_only_references_state_parents`
- `tests/test_codex_lineage_depth_balance.py::test_least_count_available_surface_wins_over_higher_puct`
- `tests/test_codex_lineage_depth_balance.py::test_unavailable_zero_count_surfaces_are_ignored`
- `tests/test_codex_lineage_depth_balance.py::test_within_selected_surface_highest_baseline_puct_wins`
- `tests/test_codex_lineage_depth_balance.py::test_batch_lineage_blocking_recomputes_available_surfaces`
- `tests/test_codex_lineage_depth_balance.py::test_persistence_resume_sanitize_and_disabled_resave`
- `tests/test_codex_lineage_depth_balance.py::test_failed_rollout_updates_puct_visits_without_double_counting`
- `tests/test_codex_lineage_depth_balance.py::test_cli_dry_run_flags_and_build_sampler_plumbing`

## Untracked File Snapshots

These files are not part of `git diff` yet, but they are part of the worktree implementation state recorded here.

### `repro/gpu_mode/run_0609_lineage_depth_balance.sh`

````bash
#!/usr/bin/env bash
set -euo pipefail

mode="run"
if [[ "${1:-}" == "run" || "${1:-}" == "dry-run" ]]; then
  mode="$1"
  shift
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"

python "${repo_root}/repro/run_discovery.py" "${mode}" \
  --experiment-name "gpu-mode-trimul-0609-lineage-depth-balance" \
  --problem-type "trimul" \
  --codex-sampler-lineage-depth-balance \
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run or dry-run the Codex GPUMode discovery repro."
    )
    parser.add_argument("mode", nargs="?", choices=("run", "dry-run"), default="run")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--problem-type", default="trimul")
    parser.add_argument(
        "--experiment-name",
        default="gpu-mode-trimul-0609-lineage-depth-balance",
    )
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--group-size", type=int, default=1)
    parser.add_argument("--groups-per-batch", type=int, default=1)
    parser.add_argument("--num-cpus-per-task", type=int, default=1)
    parser.add_argument("--eval-timeout", type=int, default=530)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--codex-model-name", default=None)
    parser.add_argument("--codex-backend", choices=("cli", "responses"), default="cli")
    parser.add_argument("--codex-max-output-tokens", type=int, default=8192)
    parser.add_argument("--codex-temperature", type=float, default=None)
    parser.add_argument("--codex-cli-command", default="codex")
    parser.add_argument(
        "--codex-cli-sandbox",
        choices=("read-only", "workspace-write", "danger-full-access"),
        default="read-only",
    )
    parser.add_argument("--codex-cli-timeout", type=float, default=None)
    parser.add_argument("--codex-max-concurrent-requests", type=int, default=1)
    parser.add_argument(
        "--codex-initial-program",
        action="append",
        default=None,
    )
    parser.add_argument(
        "--codex-initial-pool",
        action="append",
        default=None,
    )
    parser.add_argument(
        "--codex-autonomous",
        action="store_true",
        help="Let Codex use a writable workspace for each sample.",
    )
    parser.set_defaults(codex_sampler_lineage_depth_balance=False)
    parser.add_argument(
        "--codex-sampler-lineage-depth-balance",
        dest="codex_sampler_lineage_depth_balance",
        action="store_true",
    )
    parser.add_argument(
        "--no-codex-sampler-lineage-depth-balance",
        dest="codex_sampler_lineage_depth_balance",
        action="store_false",
    )
    return parser.parse_args()


def _config_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "runner": "codex_no_finetune",
        "problem_type": args.problem_type,
        "experiment_name": args.experiment_name,
        "wandb_project": args.wandb_project,
        "num_epochs": args.num_epochs,
        "group_size": args.group_size,
        "groups_per_batch": args.groups_per_batch,
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
        "codex_initial_program_paths": tuple(args.codex_initial_program or ()),
        "codex_initial_pool_paths": tuple(args.codex_initial_pool or ()),
        "codex_autonomous": args.codex_autonomous,
        "codex_sampler_lineage_depth_balance": args.codex_sampler_lineage_depth_balance,
    }


def _dry_run_payload(args: argparse.Namespace, config_kwargs: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "dry_run": True,
        "mode": "dry-run",
        "env_type": "examples.gpu_mode.env:GpuModeEnv",
    }
    payload.update(config_kwargs)
    return payload


def main() -> None:
    args = parse_args()
    config_kwargs = _config_kwargs(args)
    dry_run = args.dry_run or args.mode == "dry-run"

    if dry_run:
        print(json.dumps(_dry_run_payload(args, config_kwargs), indent=2, sort_keys=True))
        return

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    from examples.gpu_mode.env import GpuModeEnv
    from ttt_discover import DiscoverConfig, discover

    config = DiscoverConfig(env_type=GpuModeEnv, **config_kwargs)
    discover(config)


if __name__ == "__main__":
    main()
````

### `tests/test_codex_lineage_depth_balance.py`

````python
from __future__ import annotations

import inspect
import json
import subprocess
import sys
from pathlib import Path

from ttt_discover.codex_utils.runtime import State
import ttt_discover.codex_utils.sampler as sampler_mod
from ttt_discover.codex_utils.sampler import (
    LINEAGE_DEPTH_BALANCE_COUNTS_KEY,
    LINEAGE_DEPTH_SURFACES,
    PUCTSampler,
    classify_lineage_depth_surface,
)


class DummyEnv:
    state_type = State

    @staticmethod
    def create_initial_state(problem_type: str) -> State:
        return State(
            timestep=0,
            construction=[problem_type],
            code="initial",
            value=0.0,
            id=f"initial-{problem_type or 'default'}",
        )


def make_state(
    state_id: str,
    value: float,
    *,
    depth: int = 0,
    parent_ids: list[str] | None = None,
) -> State:
    if parent_ids is None:
        parent_ids = [f"{state_id}-ancestor-{i}" for i in range(depth)]
    return State(
        timestep=depth,
        construction=[state_id],
        code=f"code-{state_id}",
        value=value,
        parents=[{"id": parent_id, "timestep": i} for i, parent_id in enumerate(parent_ids)],
        id=state_id,
        observation=f"obs-{state_id}",
    )


def make_sampler(
    tmp_path: Path,
    *,
    states: list[State],
    enabled: bool,
    counts: dict[str, int] | None = None,
) -> PUCTSampler:
    sampler = PUCTSampler(
        file_path=str(tmp_path / "puct_sampler.json"),
        env_type=DummyEnv,
        batch_size=0,
        puct_c=0.0,
        topk_children=0,
        lineage_depth_balance=enabled,
    )
    sampler._states = list(states)
    sampler._initial_states = []
    if counts is not None:
        sampler._lineage_depth_balance_counts.update(counts)
    return sampler


def read_sampler_store(file_path: Path, step: int) -> dict:
    step_path = sampler_mod._sampler_file_for_step(str(file_path), step)
    return json.loads(Path(step_path).read_text(encoding="utf-8"))


def write_sampler_store(file_path: Path, step: int, store: dict) -> None:
    step_path = Path(sampler_mod._sampler_file_for_step(str(file_path), step))
    step_path.parent.mkdir(parents=True, exist_ok=True)
    step_path.write_text(json.dumps(store), encoding="utf-8")


def test_disabled_parity_no_lineage_observability_or_persistence(tmp_path: Path) -> None:
    root = make_state("root", 10.0)
    child = make_state("child", 9.0, parent_ids=["root"])
    middle = make_state("middle", 8.0, depth=3)
    sampler = make_sampler(tmp_path, states=[root, child, middle], enabled=False)

    picked = sampler.sample_states(2)

    assert [state.id for state in picked] == ["root", "middle"]
    columns, rows = sampler.get_sample_table()
    assert rows
    assert "lineage_depth_surface" not in columns
    assert "lineage_depth" not in columns
    assert "lineage_depth_surface_count_before" not in columns
    assert not any(
        key.startswith("puct/lineage_depth_balance/")
        for key in sampler.get_sample_stats()
    )

    sampler.flush(step=1)
    saved = read_sampler_store(tmp_path / "puct_sampler.json", 1)
    assert LINEAGE_DEPTH_BALANCE_COUNTS_KEY not in saved


def test_classifier_bins_use_parent_list_depths() -> None:
    expected = {
        0: "root",
        1: "shallow",
        2: "shallow",
        3: "middle",
        5: "middle",
        6: "deep",
    }
    for depth, surface in expected.items():
        assert classify_lineage_depth_surface(make_state(f"s{depth}", 1.0, depth=depth)) == (
            surface,
            depth,
        )


def test_classifier_static_source_only_references_state_parents() -> None:
    source = inspect.getsource(classify_lineage_depth_surface)
    forbidden_state_fields = (
        ".code",
        ".construction",
        ".observation",
        ".output",
        ".evaluator",
        ".reward",
        ".value",
        ".id",
        ".timestep",
        ".parent_values",
        ".birth",
        ".cohort",
        ".family",
    )
    for field in forbidden_state_fields:
        assert field not in source
    assert ".parents" in source


def test_least_count_available_surface_wins_over_higher_puct(tmp_path: Path) -> None:
    shallow_high = make_state("shallow-high", 100.0, depth=1)
    middle_low = make_state("middle-low", 1.0, depth=3)
    sampler = make_sampler(
        tmp_path,
        states=[shallow_high, middle_low],
        enabled=True,
        counts={"shallow": 5, "middle": 0},
    )

    picked = sampler.sample_states(1)

    assert [state.id for state in picked] == ["middle-low"]
    assert sampler._lineage_depth_balance_counts["middle"] == 1
    columns, rows = sampler.get_sample_table()
    assert columns[-3:] == [
        "lineage_depth_surface",
        "lineage_depth",
        "lineage_depth_surface_count_before",
    ]
    assert rows[0][-3:] == ("middle", 3, 0)
    stats = sampler.get_sample_stats()
    assert stats["puct/lineage_depth_balance/count/middle"] == 1
    assert stats["puct/lineage_depth_balance/sampled_distinct_surfaces"] == 1


def test_unavailable_zero_count_surfaces_are_ignored(tmp_path: Path) -> None:
    shallow = make_state("only-shallow", 1.0, depth=1)
    sampler = make_sampler(
        tmp_path,
        states=[shallow],
        enabled=True,
        counts={"root": 0, "shallow": 7, "middle": 0, "deep": 0},
    )

    picked = sampler.sample_states(1)

    assert [state.id for state in picked] == ["only-shallow"]
    assert sampler._lineage_depth_balance_counts["shallow"] == 8


def test_within_selected_surface_highest_baseline_puct_wins(tmp_path: Path) -> None:
    middle_low = make_state("middle-low", 1.0, depth=3)
    middle_high = make_state("middle-high", 10.0, depth=5)
    sampler = make_sampler(
        tmp_path,
        states=[middle_low, middle_high],
        enabled=True,
        counts={"middle": 0},
    )

    picked = sampler.sample_states(1)

    assert [state.id for state in picked] == ["middle-high"]


def test_batch_lineage_blocking_recomputes_available_surfaces(tmp_path: Path) -> None:
    root = make_state("root", 100.0)
    middle_child = make_state("middle-child", 90.0, parent_ids=["root", "a", "b"])
    shallow = make_state("shallow", 80.0, parent_ids=["other-root"])
    deep = make_state("deep", 70.0, depth=6)
    sampler = make_sampler(
        tmp_path,
        states=[root, middle_child, shallow, deep],
        enabled=True,
        counts={"root": 0, "middle": 0, "shallow": 5, "deep": 6},
    )

    picked = sampler.sample_states(2)

    assert [state.id for state in picked] == ["root", "shallow"]
    assert "middle-child" not in [state.id for state in picked]
    assert sampler._lineage_depth_balance_counts["root"] == 1
    assert sampler._lineage_depth_balance_counts["middle"] == 0
    assert sampler._lineage_depth_balance_counts["shallow"] == 6


def test_persistence_resume_sanitize_and_disabled_resave(tmp_path: Path) -> None:
    file_path = tmp_path / "persist" / "puct_sampler.json"
    root = make_state("root", 1.0)
    enabled = PUCTSampler(
        file_path=str(file_path),
        env_type=DummyEnv,
        batch_size=0,
        lineage_depth_balance=True,
    )
    enabled._states = [root]
    enabled._lineage_depth_balance_counts.update(
        {"root": 2, "shallow": 1, "middle": 0, "deep": 4}
    )
    enabled.flush(step=3)

    saved = read_sampler_store(file_path, 3)
    assert set(saved[LINEAGE_DEPTH_BALANCE_COUNTS_KEY]) == set(LINEAGE_DEPTH_SURFACES)
    resumed = PUCTSampler(
        file_path=str(file_path),
        env_type=DummyEnv,
        batch_size=0,
        resume_step=3,
        lineage_depth_balance=True,
    )
    assert resumed._lineage_depth_balance_counts == {
        "root": 2,
        "shallow": 1,
        "middle": 0,
        "deep": 4,
    }

    bad_store = {
        "step": 4,
        "states": [root.to_dict()],
        "initial_states": [],
        "puct_n": {},
        "puct_m": {},
        "puct_T": 0,
        LINEAGE_DEPTH_BALANCE_COUNTS_KEY: {
            "root": 2,
            "shallow": True,
            "middle": -1,
            "deep": 3.5,
            "unknown": 99,
        },
    }
    write_sampler_store(file_path, 4, bad_store)
    sanitized = PUCTSampler(
        file_path=str(file_path),
        env_type=DummyEnv,
        batch_size=0,
        resume_step=4,
        lineage_depth_balance=True,
    )
    assert sanitized._lineage_depth_balance_counts == {
        "root": 2,
        "shallow": 0,
        "middle": 0,
        "deep": 0,
    }
    sanitized._lineage_depth_balance_counts.update(
        {"root": True, "shallow": -2, "middle": "bad", "deep": 5, "unknown": 99}
    )
    sanitized.flush(step=7)
    saved_sanitized = read_sampler_store(file_path, 7)
    assert saved_sanitized[LINEAGE_DEPTH_BALANCE_COUNTS_KEY] == {
        "root": 0,
        "shallow": 0,
        "middle": 0,
        "deep": 5,
    }

    missing_store = dict(bad_store)
    missing_store.pop(LINEAGE_DEPTH_BALANCE_COUNTS_KEY)
    missing_store["step"] = 5
    write_sampler_store(file_path, 5, missing_store)
    missing = PUCTSampler(
        file_path=str(file_path),
        env_type=DummyEnv,
        batch_size=0,
        resume_step=5,
        lineage_depth_balance=True,
    )
    assert missing._lineage_depth_balance_counts == {
        "root": 0,
        "shallow": 0,
        "middle": 0,
        "deep": 0,
    }

    disabled = PUCTSampler(
        file_path=str(file_path),
        env_type=DummyEnv,
        batch_size=0,
        resume_step=4,
        lineage_depth_balance=False,
    )
    disabled.flush(step=6)
    disabled_saved = read_sampler_store(file_path, 6)
    assert LINEAGE_DEPTH_BALANCE_COUNTS_KEY not in disabled_saved


def test_failed_rollout_updates_puct_visits_without_double_counting(tmp_path: Path) -> None:
    parent = make_state("parent", 10.0, parent_ids=["ancestor"])
    sampler = make_sampler(tmp_path, states=[parent], enabled=True)

    assert [state.id for state in sampler.sample_states(1)] == ["parent"]
    counts_after_sample = dict(sampler._lineage_depth_balance_counts)

    sampler.record_failed_rollout(parent)

    assert sampler._lineage_depth_balance_counts == counts_after_sample
    assert sampler._n["parent"] == 1
    assert sampler._n["ancestor"] == 1
    assert sampler._T == 1


def test_cli_dry_run_flags_and_build_sampler_plumbing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "repro" / "run_discovery.py"

    default = subprocess.run(
        [sys.executable, str(script), "dry-run"],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    )
    default_payload = json.loads(default.stdout)
    assert default_payload["codex_sampler_lineage_depth_balance"] is False
    assert default_payload["codex_model_name"] is None

    enabled = subprocess.run(
        [
            sys.executable,
            str(script),
            "--dry-run",
            "--codex-sampler-lineage-depth-balance",
        ],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    )
    assert json.loads(enabled.stdout)["codex_sampler_lineage_depth_balance"] is True

    disabled = subprocess.run(
        [
            sys.executable,
            str(script),
            "dry-run",
            "--codex-sampler-lineage-depth-balance",
            "--no-codex-sampler-lineage-depth-balance",
        ],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    )
    assert json.loads(disabled.stdout)["codex_sampler_lineage_depth_balance"] is False

    from ttt_discover.rl import codex_no_finetune

    captured: dict = {}

    def fake_create_sampler(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(codex_no_finetune, "create_sampler", fake_create_sampler)
    cfg = codex_no_finetune.CodexNoFinetuneConfig(
        env_type=DummyEnv,
        log_path=str(tmp_path),
        sampler_lineage_depth_balance=True,
    )

    assert codex_no_finetune._build_sampler(cfg, start_batch=0) is not None
    assert captured["lineage_depth_balance"] is True
````

## Diff Stat

### Worktree Tracked Diff Stat

````text
 ttt_discover/codex_utils/discovery.py |   2 +
 ttt_discover/codex_utils/sampler.py   | 135 +++++++++++++++++++++++++++++++++-
 ttt_discover/discovery.py             |   2 +
 ttt_discover/rl/codex_no_finetune.py  |   2 +
 4 files changed, 138 insertions(+), 3 deletions(-)
````
## Raw Diff

<details>
<summary>Tracked worktree patch</summary>

````diff
diff --git a/ttt_discover/codex_utils/discovery.py b/ttt_discover/codex_utils/discovery.py
index b21a80f..d4ba039 100644
--- a/ttt_discover/codex_utils/discovery.py
+++ b/ttt_discover/codex_utils/discovery.py
@@ -50,6 +50,7 @@ class DiscoverConfig:
     codex_initial_program_paths: tuple[str, ...] = ()
     codex_initial_pool_paths: tuple[str, ...] = ()
     codex_autonomous: bool = False
+    codex_sampler_lineage_depth_balance: bool = False
 
 
 def _run_codex_no_finetune(config: DiscoverConfig) -> None:
@@ -84,6 +85,7 @@ def _run_codex_no_finetune(config: DiscoverConfig) -> None:
         max_concurrent_requests=config.codex_max_concurrent_requests,
         initial_program_paths=config.codex_initial_program_paths,
         initial_pool_paths=config.codex_initial_pool_paths,
+        sampler_lineage_depth_balance=config.codex_sampler_lineage_depth_balance,
         autonomous=config.codex_autonomous,
         wandb_project=config.wandb_project,
         wandb_name=experiment_name,
diff --git a/ttt_discover/codex_utils/sampler.py b/ttt_discover/codex_utils/sampler.py
index 5c5d4d1..4fe3c96 100644
--- a/ttt_discover/codex_utils/sampler.py
+++ b/ttt_discover/codex_utils/sampler.py
@@ -16,6 +16,40 @@ from ttt_discover.codex_utils.runtime import State, state_from_dict, to_json_ser
 
 logger = logging.getLogger(__name__)
 
+LINEAGE_DEPTH_SURFACES = ("root", "shallow", "middle", "deep")
+LINEAGE_DEPTH_BALANCE_COUNTS_KEY = "puct_lineage_depth_balance_counts"
+
+
+def classify_lineage_depth_surface(state: State) -> tuple[str, int]:
+    parents = state.parents if isinstance(state.parents, list) else []
+    depth = len(parents)
+    if depth == 0:
+        surface = "root"
+    elif depth <= 2:
+        surface = "shallow"
+    elif depth <= 5:
+        surface = "middle"
+    else:
+        surface = "deep"
+    return surface, depth
+
+
+def _zero_lineage_depth_balance_counts() -> dict[str, int]:
+    return {surface: 0 for surface in LINEAGE_DEPTH_SURFACES}
+
+
+def _sanitize_lineage_depth_balance_counts(raw_counts: Any) -> dict[str, int]:
+    counts = _zero_lineage_depth_balance_counts()
+    if not isinstance(raw_counts, dict):
+        return counts
+    for surface in LINEAGE_DEPTH_SURFACES:
+        value = raw_counts.get(surface, 0)
+        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
+            counts[surface] = 0
+        else:
+            counts[surface] = int(value)
+    return counts
+
 
 @contextmanager
 def _file_lock(lock_path: str, *, poll_s: float = 0.05, stale_s: float = 600.0):
@@ -353,6 +387,7 @@ class PUCTSampler(StateSampler):
         resume_step: int | None = None,
         puct_c: float = 1.0,
         topk_children: int = 2,
+        lineage_depth_balance: bool = False,
     ):
         self.file_path = file_path
         self.env_type = env_type
@@ -361,6 +396,7 @@ class PUCTSampler(StateSampler):
         self.batch_size = batch_size
         self.topk_children = topk_children
         self.puct_c = float(puct_c)
+        self.lineage_depth_balance = bool(lineage_depth_balance)
         
         self._states: list[State] = []
         self._initial_states: list[State] = []
@@ -375,6 +411,8 @@ class PUCTSampler(StateSampler):
         self._T: int = 0
         self._last_scale: float = 1.0
         self._last_puct_stats: list[tuple[int, float, float, float, float]] = []
+        self._lineage_depth_balance_counts = _zero_lineage_depth_balance_counts()
+        self._last_lineage_depth_sample_info: list[tuple[str, int, int]] = []
         
         if resume_step is not None:
             self._load(resume_step)
@@ -399,6 +437,10 @@ class PUCTSampler(StateSampler):
         self._n = store.get("puct_n", {}) or {}
         self._m = store.get("puct_m", {}) or {}
         self._T = int(store.get("puct_T", 0) or 0)
+        if self.lineage_depth_balance:
+            self._lineage_depth_balance_counts = _sanitize_lineage_depth_balance_counts(
+                store.get(LINEAGE_DEPTH_BALANCE_COUNTS_KEY)
+            )
 
     def _save(self, step: int):
         save_path = _sampler_file_for_step(self.file_path, step)
@@ -411,6 +453,14 @@ class PUCTSampler(StateSampler):
             "puct_m": self._m,
             "puct_T": self._T,
         }
+        if self.lineage_depth_balance:
+            self._lineage_depth_balance_counts = _sanitize_lineage_depth_balance_counts(
+                self._lineage_depth_balance_counts
+            )
+            store[LINEAGE_DEPTH_BALANCE_COUNTS_KEY] = {
+                surface: int(self._lineage_depth_balance_counts.get(surface, 0))
+                for surface in LINEAGE_DEPTH_SURFACES
+            }
         with _file_lock(f"{save_path}.lock"):
             _atomic_write_json(save_path, store)
 
@@ -501,6 +551,13 @@ class PUCTSampler(StateSampler):
             self._last_sampled_states = picked
             self._last_sampled_indices = []
             self._last_puct_stats = [(0, 0.0, 0.0, 0.0, 0.0) for _ in picked]
+            self._last_lineage_depth_sample_info = []
+            if self.lineage_depth_balance:
+                for state in picked:
+                    surface, depth = classify_lineage_depth_surface(state)
+                    count_before = self._lineage_depth_balance_counts[surface]
+                    self._last_lineage_depth_sample_info.append((surface, depth, count_before))
+                    self._lineage_depth_balance_counts[surface] = count_before + 1
             return picked
 
         vals = np.array([float(s.value if s.value is not None else float("-inf")) for s in candidates])
@@ -521,7 +578,44 @@ class PUCTSampler(StateSampler):
 
         scores.sort(key=lambda x: (x[0], x[1]), reverse=True)
 
-        if num_states > 1:
+        self._last_lineage_depth_sample_info = []
+        if self.lineage_depth_balance:
+            picked, top_scores = [], []
+            children_map = self._build_children_map()
+            blocked_ids = set()
+            while len(picked) < num_states:
+                available_entries = [
+                    entry for entry in scores if entry[2].id not in blocked_ids
+                ]
+                if not available_entries:
+                    break
+                available_surfaces = {
+                    classify_lineage_depth_surface(entry[2])[0]
+                    for entry in available_entries
+                }
+                target_surface = min(
+                    (
+                        surface
+                        for surface in LINEAGE_DEPTH_SURFACES
+                        if surface in available_surfaces
+                    ),
+                    key=lambda surface: self._lineage_depth_balance_counts[surface],
+                )
+                selected_entry = next(
+                    entry
+                    for entry in available_entries
+                    if classify_lineage_depth_surface(entry[2])[0] == target_surface
+                )
+                selected_state = selected_entry[2]
+                picked.append(selected_state)
+                top_scores.append(selected_entry)
+                surface, depth = classify_lineage_depth_surface(selected_state)
+                count_before = self._lineage_depth_balance_counts[surface]
+                self._last_lineage_depth_sample_info.append((surface, depth, count_before))
+                self._lineage_depth_balance_counts[surface] = count_before + 1
+                if num_states > 1:
+                    blocked_ids.update(self._get_full_lineage(selected_state, children_map))
+        elif num_states > 1:
             children_map = self._build_children_map()
             picked, top_scores, blocked_ids = [], [], set()
             for entry in scores:
@@ -731,21 +825,52 @@ class PUCTSampler(StateSampler):
         stats.update(_stats(sampled_values, "puct/sampled_value"))
         stats.update(_stats(sampled_timesteps, "puct/sampled_timestep"))
         stats.update(_stats(sampled_constr_lens, "puct/sampled_construction_len"))
+        if self.lineage_depth_balance:
+            counts = [
+                int(self._lineage_depth_balance_counts.get(surface, 0))
+                for surface in LINEAGE_DEPTH_SURFACES
+            ]
+            for surface, count in zip(LINEAGE_DEPTH_SURFACES, counts, strict=False):
+                stats[f"puct/lineage_depth_balance/count/{surface}"] = count
+            stats["puct/lineage_depth_balance/count_spread"] = max(counts) - min(counts)
+            sampled_surfaces = {
+                info[0]
+                for info in self._last_lineage_depth_sample_info
+                if info[0] in LINEAGE_DEPTH_SURFACES
+            }
+            stats["puct/lineage_depth_balance/sampled_distinct_surfaces"] = len(sampled_surfaces)
         return stats
 
     def get_sample_table(self) -> tuple[list[str], list[tuple]]:
         columns = ["buffer_idx", "timestep", "value", "terminal_value", "parent_value", "construction_len", "observation_len", "n", "Q", "P", "bonus", "score"]
+        if self.lineage_depth_balance:
+            columns = columns + [
+                "lineage_depth_surface",
+                "lineage_depth",
+                "lineage_depth_surface_count_before",
+            ]
         rows = []
         if not self._last_sampled_states:
             return columns, rows
         indices = self._last_sampled_indices if len(self._last_sampled_indices) == len(self._last_sampled_states) else [-1] * len(self._last_sampled_states)
         stats = self._last_puct_stats if len(self._last_puct_stats) == len(self._last_sampled_states) else [(0, 0.0, 0.0, 0.0, 0.0)] * len(self._last_sampled_states)
-        for idx, state, (n, Q, P, bonus, score) in zip(indices, self._last_sampled_states, stats):
+        lineage_infos = self._last_lineage_depth_sample_info
+        if len(lineage_infos) != len(self._last_sampled_states):
+            lineage_infos = [
+                (*classify_lineage_depth_surface(state), 0)
+                for state in self._last_sampled_states
+            ]
+        for row_idx, (idx, state, (n, Q, P, bonus, score)) in enumerate(
+            zip(indices, self._last_sampled_states, stats)
+        ):
             parent_val = state.parent_values[0] if state.parent_values else None
             constr = getattr(state, 'construction', None)
             constr_len = len(constr) if constr is not None else 0
             obs_len = len(state.observation) if state.observation else 0
-            rows.append((idx, state.timestep, state.value, 0, parent_val, constr_len, obs_len, n, Q, P, bonus, score))
+            row = (idx, state.timestep, state.value, 0, parent_val, constr_len, obs_len, n, Q, P, bonus, score)
+            if self.lineage_depth_balance:
+                row = row + lineage_infos[row_idx]
+            rows.append(row)
         return columns, rows
 
 
@@ -756,6 +881,7 @@ def create_sampler(
     batch_size: int = 1,
     resume_step: int | None = None,
     topk_children: int = 2,
+    lineage_depth_balance: bool = False,
 ) -> StateSampler:
     """Factory function to create samplers. Pass the env type (from config.env_type)."""
     if not log_path:
@@ -768,6 +894,7 @@ def create_sampler(
         batch_size=batch_size,
         resume_step=resume_step,
         topk_children=topk_children,
+        lineage_depth_balance=lineage_depth_balance,
     )
 
 
@@ -778,6 +905,7 @@ def get_or_create_sampler_with_default(
     batch_size: int = 1,
     resume_step: int | None = None,
     topk_children: int = 2,
+    lineage_depth_balance: bool = False,
 ) -> StateSampler:
     """Get sampler. Initial experience is created via env_type.create_initial_state."""
     return create_sampler(
@@ -787,4 +915,5 @@ def get_or_create_sampler_with_default(
         batch_size=batch_size,
         resume_step=resume_step,
         topk_children=topk_children,
+        lineage_depth_balance=lineage_depth_balance,
     )
diff --git a/ttt_discover/discovery.py b/ttt_discover/discovery.py
index 6cdb434..b651fbd 100644
--- a/ttt_discover/discovery.py
+++ b/ttt_discover/discovery.py
@@ -63,6 +63,7 @@ class DiscoverConfig:
     codex_initial_program_paths: tuple[str, ...] = ()
     codex_initial_pool_paths: tuple[str, ...] = ()
     codex_autonomous: bool = False
+    codex_sampler_lineage_depth_balance: bool = False
 
 
 def init_ray(num_cpus_per_task: int, env_type: str):
@@ -145,6 +146,7 @@ async def discover_impl(config: DiscoverConfig):
             max_concurrent_requests=config.codex_max_concurrent_requests,
             initial_program_paths=config.codex_initial_program_paths,
             initial_pool_paths=config.codex_initial_pool_paths,
+            sampler_lineage_depth_balance=config.codex_sampler_lineage_depth_balance,
             autonomous=config.codex_autonomous,
             wandb_project=config.wandb_project,
             wandb_name=config.experiment_name,
diff --git a/ttt_discover/rl/codex_no_finetune.py b/ttt_discover/rl/codex_no_finetune.py
index 18a0e9e..da3ac5c 100644
--- a/ttt_discover/rl/codex_no_finetune.py
+++ b/ttt_discover/rl/codex_no_finetune.py
@@ -137,6 +137,7 @@ class CodexNoFinetuneConfig:
     initial_program_paths: tuple[str, ...] = ()
     initial_pool_paths: tuple[str, ...] = ()
     topk_children: int = 16
+    sampler_lineage_depth_balance: bool = False
 
     # False is TTT Discover: non-auto, many short samples. True is AutoEvolve:
     # one Codex deep-dive workspace per sample with file edits and shell access.
@@ -960,6 +961,7 @@ def _build_sampler(cfg: CodexNoFinetuneConfig, start_batch: int) -> StateSampler
         batch_size=cfg.groups_per_batch,
         resume_step=start_batch if start_batch > 0 else None,
         topk_children=cfg.topk_children,
+        lineage_depth_balance=cfg.sampler_lineage_depth_balance,
     )
 
 
````
</details>

