# codex/diversity-id-hash-lane-balance

## Summary

用 state id 的 hash 稳定映射到 id_lane_0..id_lane_3，作为与语义无关的随机 lane 平衡基线。

## Branch State

- Worktree: `/opt/tiger/discover-id-hash-lane-balance`
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

- `codex_parent_balance_surface`
- `parent_balance_surface`

### Constants

- `ID_HASH_LANE_SURFACES`
- `PUCT_ID_HASH_LANE_SAMPLE_COUNTS_KEY`
- `_ID_HASH_LANE_SURFACE_ORDER`

### Classes

- None

### Functions

- `_validate_parent_balance_surface`
- `id_hash_lane_surface_for_state`
- `_zero_id_hash_lane_counts`
- `_sanitize_id_hash_lane_counts`
- `_reset_id_hash_lane_tracking`
- `_iter_available_entries`
- `_group_id_hash_lane_entries`
- `_choose_id_hash_lane_surface`
- `_record_id_hash_lane_selection`
- `_sample_states_id_hash_lane`

## Diff Summary

- Worktree tracked shortstat: `4 files changed, 283 insertions(+), 3 deletions(-)`
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

- `repro/gpu_mode/run_0609_id_hash_lane_balance.sh (536 bytes)`
- `repro/run_discovery.py (4806 bytes)`
- `tests/test_codex_id_hash_lane_balance.py (13934 bytes)`

### Detected Test Functions

- `tests/test_codex_id_hash_lane_balance.py::test_disabled_mode_parity_stats_table_metrics_and_json`
- `tests/test_codex_id_hash_lane_balance.py::test_classifier_uses_only_state_id_and_matches_blake2b`
- `tests/test_codex_id_hash_lane_balance.py::test_least_sampled_lane_fixed_order_and_within_surface_order`
- `tests/test_codex_id_hash_lane_balance.py::test_multi_parent_updates_counts_and_preserves_lineage_blocking`
- `tests/test_codex_id_hash_lane_balance.py::test_fallback_increments_actual_lane_and_respects_blocking`
- `tests/test_codex_id_hash_lane_balance.py::test_save_resume_restores_counts_and_sanitizes_missing_bad_values`
- `tests/test_codex_id_hash_lane_balance.py::test_enabled_only_table_and_metrics`
- `tests/test_codex_id_hash_lane_balance.py::test_cli_and_config_plumbing_defaults_overrides_and_invalid_cli`

## Untracked File Snapshots

These files are not part of `git diff` yet, but they are part of the worktree implementation state recorded here.

### `repro/gpu_mode/run_0609_id_hash_lane_balance.sh`

````bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODE="run"
if [[ "${1:-}" == "run" || "${1:-}" == "dry-run" ]]; then
  MODE="$1"
  shift || true
fi

export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"

python "${REPO_ROOT}/repro/run_discovery.py" "${MODE}" \
  --experiment-name gpu-mode-0609-id-hash-lane-balance \
  --codex-parent-balance-surface id_hash_lane \
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
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GPUMode Codex discovery.")
    parser.add_argument("mode", nargs="?", choices=("run", "dry-run"), default="run")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--problem-type", default="trimul")
    parser.add_argument("--experiment-name", default="gpu-mode-discovery")
    parser.add_argument(
        "--wandb-project",
        default=os.environ.get("WANDB_PROJECT"),
        help="Pass '' to disable wandb.",
    )
    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--group-size", type=int, default=1)
    parser.add_argument("--groups-per-batch", type=int, default=1)
    parser.add_argument("--num-cpus-per-task", type=int, default=1)
    parser.add_argument("--eval-timeout", type=int, default=1200)
    parser.add_argument("--codex-model-name", default=None)
    parser.add_argument("--codex-backend", choices=("cli", "responses"), default="cli")
    parser.add_argument("--codex-max-output-tokens", type=int, default=8192)
    parser.add_argument("--codex-temperature", type=float, default=None)
    parser.add_argument("--codex-cli-command", default="codex")
    parser.add_argument(
        "--codex-cli-sandbox",
        choices=("read-only", "workspace-write", "danger-full-access"),
        default=None,
    )
    parser.add_argument("--codex-cli-timeout", type=float, default=None)
    parser.add_argument("--codex-max-concurrent-requests", type=int, default=1)
    parser.add_argument("--codex-autonomous", action="store_true")
    parser.add_argument(
        "--codex-parent-balance-surface",
        choices=("disabled", "id_hash_lane"),
        default="disabled",
    )
    return parser.parse_args()


def _resolved_options(args: argparse.Namespace) -> dict[str, Any]:
    codex_cli_sandbox = args.codex_cli_sandbox
    if codex_cli_sandbox is None:
        codex_cli_sandbox = "workspace-write" if args.codex_autonomous else "read-only"
    return {
        "env_type": "examples.gpu_mode.env.GpuModeEnv",
        "problem_type": args.problem_type,
        "runner": "codex_no_finetune",
        "experiment_name": args.experiment_name,
        "wandb_project": args.wandb_project or None,
        "num_epochs": args.num_epochs,
        "group_size": args.group_size,
        "groups_per_batch": args.groups_per_batch,
        "num_cpus_per_task": args.num_cpus_per_task,
        "eval_timeout": args.eval_timeout,
        "codex_model_name": args.codex_model_name,
        "codex_backend": args.codex_backend,
        "codex_max_output_tokens": args.codex_max_output_tokens,
        "codex_temperature": args.codex_temperature,
        "codex_cli_command": args.codex_cli_command,
        "codex_cli_sandbox": codex_cli_sandbox,
        "codex_cli_timeout": args.codex_cli_timeout,
        "codex_max_concurrent_requests": args.codex_max_concurrent_requests,
        "codex_autonomous": args.codex_autonomous,
        "codex_parent_balance_surface": args.codex_parent_balance_surface,
    }


def main() -> None:
    args = parse_args()
    options = _resolved_options(args)
    if args.dry_run or args.mode == "dry-run":
        print(json.dumps(options, indent=2, sort_keys=True))
        return

    from examples.gpu_mode.env import GpuModeEnv
    from ttt_discover import DiscoverConfig, discover

    config = DiscoverConfig(
        env_type=GpuModeEnv,
        problem_type=options["problem_type"],
        runner="codex_no_finetune",
        experiment_name=options["experiment_name"],
        wandb_project=options["wandb_project"],
        num_epochs=options["num_epochs"],
        group_size=options["group_size"],
        groups_per_batch=options["groups_per_batch"],
        num_cpus_per_task=options["num_cpus_per_task"],
        eval_timeout=options["eval_timeout"],
        codex_model_name=options["codex_model_name"],
        codex_backend=options["codex_backend"],
        codex_max_output_tokens=options["codex_max_output_tokens"],
        codex_temperature=options["codex_temperature"],
        codex_cli_command=options["codex_cli_command"],
        codex_cli_sandbox=options["codex_cli_sandbox"],
        codex_cli_timeout=options["codex_cli_timeout"],
        codex_max_concurrent_requests=options["codex_max_concurrent_requests"],
        codex_autonomous=options["codex_autonomous"],
        codex_parent_balance_surface=options["codex_parent_balance_surface"],
    )
    discover(config)


if __name__ == "__main__":
    main()
````

### `tests/test_codex_id_hash_lane_balance.py`

````python
from __future__ import annotations

import ast
import hashlib
import inspect
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from ttt_discover.codex_utils.runtime import State
from ttt_discover.codex_utils.sampler import (
    ID_HASH_LANE_SURFACES,
    PUCT_ID_HASH_LANE_SAMPLE_COUNTS_KEY,
    PUCTSampler,
    create_sampler,
    id_hash_lane_surface_for_state,
)
from ttt_discover.rl.codex_no_finetune import CodexNoFinetuneConfig, _build_sampler


REPO_ROOT = Path(__file__).resolve().parents[1]


class DummyEnv:
    state_type = State
    refresh_count = 0

    @classmethod
    def create_initial_state(cls, problem_type: str) -> State:
        return State(
            timestep=-1,
            construction=[],
            code=f"initial-{problem_type}",
            value=0.0,
            id=f"initial-{problem_type}",
        )

    @classmethod
    def refresh_initial_state(cls, state: State, problem_type: str) -> None:
        cls.refresh_count += 1
        state.observation = f"refreshed-{problem_type}-{cls.refresh_count}"


def make_state(
    state_id: str,
    value: float,
    *,
    parents: list[dict] | None = None,
    code: str | None = None,
    construction: list | None = None,
    observation: str = "",
) -> State:
    return State(
        timestep=0,
        construction=[state_id] if construction is None else construction,
        code=state_id if code is None else code,
        value=value,
        parents=[] if parents is None else parents,
        id=state_id,
        observation=observation,
    )


def make_sampler(
    tmp_path: Path,
    *,
    enabled: bool = False,
    name: str = "sampler.json",
    batch_size: int = 0,
    resume_step: int | None = None,
) -> PUCTSampler:
    return PUCTSampler(
        file_path=str(tmp_path / name),
        env_type=DummyEnv,
        problem_type="p",
        batch_size=batch_size,
        resume_step=resume_step,
        parent_balance_surface="id_hash_lane" if enabled else "disabled",
    )


def state_id_for_lane(surface: str, prefix: str) -> str:
    for idx in range(10000):
        candidate = f"{prefix}-{idx}"
        if id_hash_lane_surface_for_state(make_state(candidate, 0.0)) == surface:
            return candidate
    raise AssertionError(f"no id found for {surface}")


def sampler_step_path(base_path: Path, step: int) -> Path:
    return base_path.with_name(f"{base_path.stem}_step_{step:06d}.json")


@pytest.mark.parametrize("num_states", [1, 2])
def test_disabled_mode_parity_stats_table_metrics_and_json(tmp_path: Path, num_states: int) -> None:
    states = [
        make_state("root", 10.0),
        make_state("child", 9.0, parents=[{"id": "root", "timestep": 0}]),
        make_state("other", 8.0),
    ]
    left = make_sampler(tmp_path, name=f"left-{num_states}.json")
    right = make_sampler(tmp_path, name=f"right-{num_states}.json")
    for sampler in (left, right):
        sampler._states = [
            make_state(s.id, s.value, parents=list(s.parents), code=s.code)
            for s in states
        ]
        sampler._initial_states = []
        sampler._n = {"root": 1}
        sampler._m = {"root": 11.0}
        sampler._T = 1

    left_ids = [state.id for state in left.sample_states(num_states)]
    right_ids = [state.id for state in right.sample_states(num_states)]

    assert left_ids == right_ids
    assert left._last_puct_stats == right._last_puct_stats
    columns, rows = left.get_sample_table()
    assert "surface" not in columns
    assert all(len(row) == len(columns) for row in rows)
    assert not any(key.startswith("puct/id_hash_lane/") for key in left.get_sample_stats())

    left.flush(step=1)
    store = json.loads(sampler_step_path(tmp_path / f"left-{num_states}.json", 1).read_text())
    assert PUCT_ID_HASH_LANE_SAMPLE_COUNTS_KEY not in store


def test_classifier_uses_only_state_id_and_matches_blake2b() -> None:
    for state_id in ("fixed", "42", "abc-123"):
        expected_digest = hashlib.blake2b(
            str(state_id).encode("utf-8"),
            digest_size=8,
            person=b"puct_id_lane",
        ).digest()
        expected = ID_HASH_LANE_SURFACES[int.from_bytes(expected_digest, "big") % 4]
        assert id_hash_lane_surface_for_state(make_state(state_id, 1.0)) == expected

    state_id = "stable-id"
    base = id_hash_lane_surface_for_state(make_state(state_id, 1.0))
    changed = make_state(
        state_id,
        999.0,
        parents=[{"id": "parent", "timestep": 4}],
        code="different code",
        construction=["different", "construction"],
        observation="different observation",
    )
    assert id_hash_lane_surface_for_state(changed) == base

    source = textwrap.dedent(inspect.getsource(id_hash_lane_surface_for_state))
    tree = ast.parse(source)
    state_attrs = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "state"
    }
    assert state_attrs == {"id"}


def test_least_sampled_lane_fixed_order_and_within_surface_order(tmp_path: Path) -> None:
    lane0_hi = state_id_for_lane("id_lane_0", "lane0-hi")
    lane0_lo = state_id_for_lane("id_lane_0", "lane0-lo")
    lane1_hi = state_id_for_lane("id_lane_1", "lane1-hi")
    lane2_lo = state_id_for_lane("id_lane_2", "lane2-lo")

    sampler = make_sampler(tmp_path, enabled=True, name="least.json")
    sampler._states = [
        make_state(lane1_hi, 100.0),
        make_state(lane0_lo, 1.0),
    ]
    sampler._id_hash_lane_sample_counts = {
        "id_lane_0": 0,
        "id_lane_1": 10,
        "id_lane_2": 10,
        "id_lane_3": 10,
    }
    assert [state.id for state in sampler.sample_states(1)] == [lane0_lo]

    sampler = make_sampler(tmp_path, enabled=True, name="available-only.json")
    sampler._states = [
        make_state(lane1_hi, 100.0),
        make_state(lane2_lo, 1.0),
    ]
    sampler._id_hash_lane_sample_counts = {
        "id_lane_0": 0,
        "id_lane_1": 5,
        "id_lane_2": 0,
        "id_lane_3": 0,
    }
    assert [state.id for state in sampler.sample_states(1)] == [lane2_lo]

    sampler = make_sampler(tmp_path, enabled=True, name="fixed-order.json")
    sampler._states = [
        make_state(lane1_hi, 100.0),
        make_state(lane0_lo, 1.0),
    ]
    assert [state.id for state in sampler.sample_states(1)] == [lane0_lo]

    sampler = make_sampler(tmp_path, enabled=True, name="within.json")
    sampler._states = [
        make_state(lane0_lo, 1.0),
        make_state(lane0_hi, 100.0),
    ]
    assert [state.id for state in sampler.sample_states(1)] == [lane0_hi]


def test_multi_parent_updates_counts_and_preserves_lineage_blocking(tmp_path: Path) -> None:
    lane0 = state_id_for_lane("id_lane_0", "multi-lane0")
    blocked_lane1 = state_id_for_lane("id_lane_1", "multi-blocked-lane1")
    lane1 = state_id_for_lane("id_lane_1", "multi-lane1")
    lane2 = state_id_for_lane("id_lane_2", "multi-lane2")

    sampler = make_sampler(tmp_path, enabled=True, name="multi.json")
    sampler._states = [
        make_state(blocked_lane1, 100.0, parents=[{"id": lane0, "timestep": 0}]),
        make_state(lane1, 50.0),
        make_state(lane2, 40.0),
        make_state(lane0, 1.0),
    ]

    picked = sampler.sample_states(3)
    assert [state.id for state in picked] == [lane0, lane1, lane2]
    assert blocked_lane1 not in [state.id for state in picked]
    assert sampler._id_hash_lane_sample_counts["id_lane_0"] == 1
    assert sampler._id_hash_lane_sample_counts["id_lane_1"] == 1
    assert sampler._id_hash_lane_sample_counts["id_lane_2"] == 1
    assert sampler._last_id_hash_lane_rows == [
        ("id_lane_0", "id_lane_0", 0, 1, False),
        ("id_lane_1", "id_lane_1", 0, 1, False),
        ("id_lane_2", "id_lane_2", 0, 1, False),
    ]


def test_fallback_increments_actual_lane_and_respects_blocking(tmp_path: Path) -> None:
    lane1 = state_id_for_lane("id_lane_1", "fallback-lane1")
    blocked_lane2 = state_id_for_lane("id_lane_2", "fallback-blocked-lane2")
    lane2 = state_id_for_lane("id_lane_2", "fallback-lane2")

    sampler = make_sampler(tmp_path, enabled=True, name="fallback.json")
    sampler._states = [
        make_state(lane1, 100.0),
        make_state(blocked_lane2, 90.0, parents=[{"id": lane1, "timestep": 0}]),
        make_state(lane2, 80.0),
    ]
    sampler._choose_id_hash_lane_surface = lambda grouped: None  # type: ignore[method-assign]

    picked = sampler.sample_states(2)
    assert [state.id for state in picked] == [lane1, lane2]
    assert sampler._id_hash_lane_sample_counts["id_lane_1"] == 1
    assert sampler._id_hash_lane_sample_counts["id_lane_2"] == 1
    assert sampler._last_id_hash_lane_fallback_count == 2
    assert [row[4] for row in sampler._last_id_hash_lane_rows] == [True, True]


def test_save_resume_restores_counts_and_sanitizes_missing_bad_values(tmp_path: Path) -> None:
    lane0 = state_id_for_lane("id_lane_0", "resume-lane0")
    sampler = make_sampler(tmp_path, enabled=True, name="resume.json")
    sampler._states = [make_state(lane0, 1.0)]
    sampler.sample_states(1)
    sampler.flush(step=3)

    resumed = make_sampler(tmp_path, enabled=True, name="resume.json", resume_step=3)
    assert resumed._id_hash_lane_sample_counts["id_lane_0"] == 1

    missing_base = tmp_path / "missing.json"
    sampler_step_path(missing_base, 4).write_text(
        json.dumps(
            {
                "step": 4,
                "states": [],
                "initial_states": [],
                "puct_n": {},
                "puct_m": {},
                "puct_T": 0,
            }
        )
    )
    missing = make_sampler(tmp_path, enabled=True, name="missing.json", resume_step=4)
    assert missing._id_hash_lane_sample_counts == {surface: 0 for surface in ID_HASH_LANE_SURFACES}

    bad_base = tmp_path / "bad.json"
    sampler_step_path(bad_base, 5).write_text(
        json.dumps(
            {
                "step": 5,
                "states": [],
                "initial_states": [],
                "puct_n": {},
                "puct_m": {},
                "puct_T": 0,
                PUCT_ID_HASH_LANE_SAMPLE_COUNTS_KEY: {
                    "id_lane_0": 7,
                    "id_lane_1": True,
                    "id_lane_2": -1,
                    "id_lane_3": 2.5,
                    "unknown": 10,
                },
            }
        )
    )
    bad = make_sampler(tmp_path, enabled=True, name="bad.json", resume_step=5)
    assert bad._id_hash_lane_sample_counts == {
        "id_lane_0": 7,
        "id_lane_1": 0,
        "id_lane_2": 0,
        "id_lane_3": 0,
    }


def test_enabled_only_table_and_metrics(tmp_path: Path) -> None:
    lane0 = state_id_for_lane("id_lane_0", "metrics-lane0")
    sampler = make_sampler(tmp_path, enabled=True, name="metrics.json")
    sampler._states = [make_state(lane0, 1.0)]
    sampler.sample_states(1)

    metrics = sampler.get_sample_stats()
    assert metrics["puct/id_hash_lane/buffer/id_lane_0"] == 1
    assert metrics["puct/id_hash_lane/sampled/id_lane_0"] == 1
    assert metrics["puct/id_hash_lane/total/id_lane_0"] == 1
    assert metrics["puct/id_hash_lane/fallback_count"] == 0

    columns, rows = sampler.get_sample_table()
    assert columns[-5:] == [
        "surface",
        "surface_target",
        "surface_count_before",
        "surface_count_after",
        "surface_fallback",
    ]
    assert rows[0][-5:] == ("id_lane_0", "id_lane_0", 0, 1, False)


def run_dry_run(*args: str) -> dict:
    proc = subprocess.run(
        [sys.executable, "repro/run_discovery.py", *args],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return json.loads(proc.stdout)


def test_cli_and_config_plumbing_defaults_overrides_and_invalid_cli(tmp_path: Path) -> None:
    cfg = CodexNoFinetuneConfig(env_type=DummyEnv, log_path=str(tmp_path))
    assert cfg.parent_balance_surface == "disabled"
    enabled_cfg = CodexNoFinetuneConfig(
        env_type=DummyEnv,
        log_path=str(tmp_path / "build"),
        parent_balance_surface="id_hash_lane",
    )
    built = _build_sampler(enabled_cfg, start_batch=0)
    assert isinstance(built, PUCTSampler)
    assert built.parent_balance_surface == "id_hash_lane"

    sampler = create_sampler(
        log_path=str(tmp_path / "factory"),
        env_type=DummyEnv,
        parent_balance_surface="id_hash_lane",
        batch_size=0,
    )
    assert isinstance(sampler, PUCTSampler)
    assert sampler.parent_balance_surface == "id_hash_lane"
    with pytest.raises(ValueError):
        create_sampler(
            log_path=str(tmp_path / "bad-factory"),
            env_type=DummyEnv,
            parent_balance_surface="bad",
            batch_size=0,
        )

    default_dry_run = run_dry_run("dry-run")
    assert default_dry_run["codex_parent_balance_surface"] == "disabled"
    assert default_dry_run["codex_model_name"] is None
    assert run_dry_run("dry-run", "--codex-parent-balance-surface", "id_hash_lane")[
        "codex_parent_balance_surface"
    ] == "id_hash_lane"
    assert run_dry_run("--dry-run", "--codex-parent-balance-surface", "id_hash_lane")[
        "codex_parent_balance_surface"
    ] == "id_hash_lane"

    invalid = subprocess.run(
        [
            sys.executable,
            "repro/run_discovery.py",
            "dry-run",
            "--codex-parent-balance-surface",
            "bad",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )
    assert invalid.returncode != 0

    for rel_path in ("ttt_discover/codex_utils/discovery.py", "ttt_discover/discovery.py"):
        text = (REPO_ROOT / rel_path).read_text()
        assert "codex_parent_balance_surface" in text
        assert "parent_balance_surface=config.codex_parent_balance_surface" in text
````

## Diff Stat

### Worktree Tracked Diff Stat

````text
 ttt_discover/codex_utils/discovery.py |   2 +
 ttt_discover/codex_utils/sampler.py   | 280 +++++++++++++++++++++++++++++++++-
 ttt_discover/discovery.py             |   2 +
 ttt_discover/rl/codex_no_finetune.py  |   2 +
 4 files changed, 283 insertions(+), 3 deletions(-)
````
## Raw Diff

<details>
<summary>Tracked worktree patch</summary>

````diff
diff --git a/ttt_discover/codex_utils/discovery.py b/ttt_discover/codex_utils/discovery.py
index b21a80f..62d0462 100644
--- a/ttt_discover/codex_utils/discovery.py
+++ b/ttt_discover/codex_utils/discovery.py
@@ -50,6 +50,7 @@ class DiscoverConfig:
     codex_initial_program_paths: tuple[str, ...] = ()
     codex_initial_pool_paths: tuple[str, ...] = ()
     codex_autonomous: bool = False
+    codex_parent_balance_surface: Literal["disabled", "id_hash_lane"] = "disabled"
 
 
 def _run_codex_no_finetune(config: DiscoverConfig) -> None:
@@ -85,6 +86,7 @@ def _run_codex_no_finetune(config: DiscoverConfig) -> None:
         initial_program_paths=config.codex_initial_program_paths,
         initial_pool_paths=config.codex_initial_pool_paths,
         autonomous=config.codex_autonomous,
+        parent_balance_surface=config.codex_parent_balance_surface,
         wandb_project=config.wandb_project,
         wandb_name=experiment_name,
         log_path=log_path,
diff --git a/ttt_discover/codex_utils/sampler.py b/ttt_discover/codex_utils/sampler.py
index 5c5d4d1..f66af42 100644
--- a/ttt_discover/codex_utils/sampler.py
+++ b/ttt_discover/codex_utils/sampler.py
@@ -2,13 +2,15 @@
 from __future__ import annotations
 from abc import ABC, abstractmethod
 from contextlib import contextmanager
+import hashlib
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
 
@@ -16,6 +18,51 @@ from ttt_discover.codex_utils.runtime import State, state_from_dict, to_json_ser
 
 logger = logging.getLogger(__name__)
 
+ID_HASH_LANE_SURFACES = ("id_lane_0", "id_lane_1", "id_lane_2", "id_lane_3")
+PUCT_ID_HASH_LANE_SAMPLE_COUNTS_KEY = "puct_id_hash_lane_sample_counts"
+ParentBalanceSurface = Literal["disabled", "id_hash_lane"]
+_ID_HASH_LANE_SURFACE_ORDER = {
+    surface: idx for idx, surface in enumerate(ID_HASH_LANE_SURFACES)
+}
+
+
+def _validate_parent_balance_surface(value: str) -> ParentBalanceSurface:
+    if value not in ("disabled", "id_hash_lane"):
+        raise ValueError(
+            "parent_balance_surface must be 'disabled' or 'id_hash_lane', "
+            f"got {value!r}"
+        )
+    return value
+
+
+def id_hash_lane_surface_for_state(state: State) -> str:
+    digest = hashlib.blake2b(
+        str(state.id).encode("utf-8"),
+        digest_size=8,
+        person=b"puct_id_lane",
+    ).digest()
+    return ID_HASH_LANE_SURFACES[int.from_bytes(digest, "big") % 4]
+
+
+id_hash_lane_surface = id_hash_lane_surface_for_state
+
+
+def _zero_id_hash_lane_counts() -> dict[str, int]:
+    return {surface: 0 for surface in ID_HASH_LANE_SURFACES}
+
+
+def _sanitize_id_hash_lane_counts(raw: Any) -> dict[str, int]:
+    counts = _zero_id_hash_lane_counts()
+    if not isinstance(raw, dict):
+        return counts
+    for surface in ID_HASH_LANE_SURFACES:
+        value = raw.get(surface, 0)
+        if isinstance(value, bool) or not isinstance(value, Integral):
+            counts[surface] = 0
+        else:
+            counts[surface] = max(0, int(value))
+    return counts
+
 
 @contextmanager
 def _file_lock(lock_path: str, *, poll_s: float = 0.05, stale_s: float = 600.0):
@@ -353,6 +400,7 @@ class PUCTSampler(StateSampler):
         resume_step: int | None = None,
         puct_c: float = 1.0,
         topk_children: int = 2,
+        parent_balance_surface: str = "disabled",
     ):
         self.file_path = file_path
         self.env_type = env_type
@@ -361,6 +409,7 @@ class PUCTSampler(StateSampler):
         self.batch_size = batch_size
         self.topk_children = topk_children
         self.puct_c = float(puct_c)
+        self.parent_balance_surface = _validate_parent_balance_surface(parent_balance_surface)
         
         self._states: list[State] = []
         self._initial_states: list[State] = []
@@ -375,6 +424,9 @@ class PUCTSampler(StateSampler):
         self._T: int = 0
         self._last_scale: float = 1.0
         self._last_puct_stats: list[tuple[int, float, float, float, float]] = []
+        self._id_hash_lane_sample_counts: dict[str, int] = _zero_id_hash_lane_counts()
+        self._last_id_hash_lane_rows: list[tuple[str, str | None, int, int, bool]] = []
+        self._last_id_hash_lane_fallback_count = 0
         
         if resume_step is not None:
             self._load(resume_step)
@@ -399,6 +451,10 @@ class PUCTSampler(StateSampler):
         self._n = store.get("puct_n", {}) or {}
         self._m = store.get("puct_m", {}) or {}
         self._T = int(store.get("puct_T", 0) or 0)
+        if self.parent_balance_surface == "id_hash_lane":
+            self._id_hash_lane_sample_counts = _sanitize_id_hash_lane_counts(
+                store.get(PUCT_ID_HASH_LANE_SAMPLE_COUNTS_KEY, {})
+            )
 
     def _save(self, step: int):
         save_path = _sampler_file_for_step(self.file_path, step)
@@ -411,6 +467,10 @@ class PUCTSampler(StateSampler):
             "puct_m": self._m,
             "puct_T": self._T,
         }
+        if self.parent_balance_surface == "id_hash_lane":
+            store[PUCT_ID_HASH_LANE_SAMPLE_COUNTS_KEY] = (
+                _sanitize_id_hash_lane_counts(self._id_hash_lane_sample_counts)
+            )
         with _file_lock(f"{save_path}.lock"):
             _atomic_write_json(save_path, store)
 
@@ -489,7 +549,180 @@ class PUCTSampler(StateSampler):
                     queue.append(child_id)
         return lineage
 
+    def _reset_id_hash_lane_tracking(self) -> None:
+        self._last_id_hash_lane_rows = []
+        self._last_id_hash_lane_fallback_count = 0
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
+    def _group_id_hash_lane_entries(
+        self,
+        scores: list[tuple],
+        *,
+        picked_ids: set[str],
+        blocked_ids: set[str],
+        block_lineage: bool,
+    ) -> dict[str, list[tuple]]:
+        grouped: dict[str, list[tuple]] = {
+            surface: [] for surface in ID_HASH_LANE_SURFACES
+        }
+        for entry in self._iter_available_entries(
+            scores,
+            picked_ids=picked_ids,
+            blocked_ids=blocked_ids,
+            block_lineage=block_lineage,
+        ):
+            grouped[id_hash_lane_surface_for_state(entry[2])].append(entry)
+        return grouped
+
+    def _choose_id_hash_lane_surface(
+        self,
+        grouped: dict[str, list[tuple]],
+    ) -> str | None:
+        available_surfaces = [
+            surface for surface in ID_HASH_LANE_SURFACES if grouped.get(surface)
+        ]
+        if not available_surfaces:
+            return None
+        counts = _sanitize_id_hash_lane_counts(self._id_hash_lane_sample_counts)
+        return min(
+            available_surfaces,
+            key=lambda surface: (
+                counts[surface],
+                _ID_HASH_LANE_SURFACE_ORDER[surface],
+            ),
+        )
+
+    def _record_id_hash_lane_selection(
+        self,
+        state: State,
+        *,
+        target_surface: str | None,
+        fallback: bool,
+    ) -> None:
+        surface = id_hash_lane_surface_for_state(state)
+        before = _sanitize_id_hash_lane_counts(self._id_hash_lane_sample_counts)[surface]
+        after = before + 1
+        self._id_hash_lane_sample_counts[surface] = after
+        self._last_id_hash_lane_rows.append(
+            (surface, target_surface, before, after, fallback)
+        )
+        if fallback:
+            self._last_id_hash_lane_fallback_count += 1
+
+    def _sample_states_id_hash_lane(self, num_states: int) -> list[State]:
+        initial_ids = {s.id for s in self._initial_states}
+        candidates = list(self._states)
+        self._reset_id_hash_lane_tracking()
+
+        if not candidates:
+            picked = [
+                create_initial_state(self.env_type, self.problem_type)
+                for _ in range(num_states)
+            ]
+            self._last_sampled_states = picked
+            self._last_sampled_indices = []
+            self._last_puct_stats = [(0, 0.0, 0.0, 0.0, 0.0) for _ in picked]
+            for state in picked:
+                self._record_id_hash_lane_selection(
+                    state,
+                    target_surface=None,
+                    fallback=True,
+                )
+            return picked
+
+        vals = np.array([float(s.value if s.value is not None else float("-inf")) for s in candidates])
+        non_initial_mask = np.array([s.id not in initial_ids for s in candidates])
+        scale = self._compute_scale(vals, non_initial_mask if non_initial_mask.any() else None)
+        self._last_scale = scale
+        P = self._compute_prior(vals, scale)
+        sqrtT = np.sqrt(1.0 + self._T)
+
+        scores = []
+        for i, s in enumerate(candidates):
+            n = self._n.get(s.id, 0)
+            m = self._m.get(s.id, vals[i])
+            Q = m if n > 0 else vals[i]
+            bonus = self.puct_c * scale * P[i] * sqrtT / (1.0 + n)
+            score = Q + bonus
+            scores.append((score, vals[i], s, n, Q, P[i], bonus))
+
+        scores.sort(key=lambda x: (x[0], x[1]), reverse=True)
+
+        block_lineage = num_states > 1
+        children_map = self._build_children_map() if block_lineage else {}
+        picked: list[State] = []
+        top_scores = []
+        picked_ids = set()
+        blocked_ids = set()
+
+        while len(picked) < num_states:
+            grouped = self._group_id_hash_lane_entries(
+                scores,
+                picked_ids=picked_ids,
+                blocked_ids=blocked_ids,
+                block_lineage=block_lineage,
+            )
+            target_surface = self._choose_id_hash_lane_surface(grouped)
+            fallback = False
+            if target_surface is not None and grouped.get(target_surface):
+                selected_entry = grouped[target_surface][0]
+            else:
+                selected_entry = next(
+                    self._iter_available_entries(
+                        scores,
+                        picked_ids=picked_ids,
+                        blocked_ids=blocked_ids,
+                        block_lineage=block_lineage,
+                    ),
+                    None,
+                )
+                if selected_entry is None:
+                    break
+                fallback = True
+
+            state = selected_entry[2]
+
+            picked.append(state)
+            top_scores.append(selected_entry)
+            picked_ids.add(state.id)
+            self._record_id_hash_lane_selection(
+                state,
+                target_surface=target_surface,
+                fallback=fallback,
+            )
+            if block_lineage:
+                blocked_ids.update(self._get_full_lineage(state, children_map))
+
+        state_id_to_idx = {s.id: i for i, s in enumerate(self._states)}
+        self._last_sampled_states = picked
+        self._last_sampled_indices = [state_id_to_idx.get(s.id, -1) for s in picked]
+        self._last_puct_stats = [(t[3], t[4], t[5], t[6], t[0]) for t in top_scores]
+
+        for s in picked:
+            if s.id in initial_ids:
+                self._refresh_random_construction(s)
+
+        return picked
+
     def sample_states(self, num_states: int) -> list[State]:
+        if self.parent_balance_surface == "id_hash_lane":
+            return self._sample_states_id_hash_lane(num_states)
+
         initial_ids = {s.id for s in self._initial_states}
         candidates = list(self._states)
 
@@ -650,6 +883,7 @@ class PUCTSampler(StateSampler):
         with self._lock:
             self._states = []
             self._initial_states = []
+            self._id_hash_lane_sample_counts = _zero_id_hash_lane_counts()
             self._current_step = step
             self._load(step)
             if not self._states:
@@ -731,21 +965,56 @@ class PUCTSampler(StateSampler):
         stats.update(_stats(sampled_values, "puct/sampled_value"))
         stats.update(_stats(sampled_timesteps, "puct/sampled_timestep"))
         stats.update(_stats(sampled_constr_lens, "puct/sampled_construction_len"))
+        if self.parent_balance_surface == "id_hash_lane":
+            buffer_surfaces = {surface: 0 for surface in ID_HASH_LANE_SURFACES}
+            sampled_surfaces = {surface: 0 for surface in ID_HASH_LANE_SURFACES}
+            for state in self._states:
+                buffer_surfaces[id_hash_lane_surface_for_state(state)] += 1
+            for state in self._last_sampled_states:
+                sampled_surfaces[id_hash_lane_surface_for_state(state)] += 1
+            total_surfaces = _sanitize_id_hash_lane_counts(
+                self._id_hash_lane_sample_counts
+            )
+            for surface in ID_HASH_LANE_SURFACES:
+                stats[f"puct/id_hash_lane/buffer/{surface}"] = buffer_surfaces[surface]
+                stats[f"puct/id_hash_lane/sampled/{surface}"] = sampled_surfaces[surface]
+                stats[f"puct/id_hash_lane/total/{surface}"] = (
+                    total_surfaces[surface]
+                )
+            stats["puct/id_hash_lane/fallback_count"] = (
+                self._last_id_hash_lane_fallback_count
+            )
         return stats
 
     def get_sample_table(self) -> tuple[list[str], list[tuple]]:
         columns = ["buffer_idx", "timestep", "value", "terminal_value", "parent_value", "construction_len", "observation_len", "n", "Q", "P", "bonus", "score"]
+        if self.parent_balance_surface == "id_hash_lane":
+            columns = columns + [
+                "surface",
+                "surface_target",
+                "surface_count_before",
+                "surface_count_after",
+                "surface_fallback",
+            ]
         rows = []
         if not self._last_sampled_states:
             return columns, rows
         indices = self._last_sampled_indices if len(self._last_sampled_indices) == len(self._last_sampled_states) else [-1] * len(self._last_sampled_states)
         stats = self._last_puct_stats if len(self._last_puct_stats) == len(self._last_sampled_states) else [(0, 0.0, 0.0, 0.0, 0.0)] * len(self._last_sampled_states)
-        for idx, state, (n, Q, P, bonus, score) in zip(indices, self._last_sampled_states, stats):
+        surface_rows = self._last_id_hash_lane_rows if len(self._last_id_hash_lane_rows) == len(self._last_sampled_states) else []
+        for row_idx, (idx, state, (n, Q, P, bonus, score)) in enumerate(zip(indices, self._last_sampled_states, stats)):
             parent_val = state.parent_values[0] if state.parent_values else None
             constr = getattr(state, 'construction', None)
             constr_len = len(constr) if constr is not None else 0
             obs_len = len(state.observation) if state.observation else 0
-            rows.append((idx, state.timestep, state.value, 0, parent_val, constr_len, obs_len, n, Q, P, bonus, score))
+            row = (idx, state.timestep, state.value, 0, parent_val, constr_len, obs_len, n, Q, P, bonus, score)
+            if self.parent_balance_surface == "id_hash_lane":
+                if surface_rows:
+                    row = row + surface_rows[row_idx]
+                else:
+                    surface = id_hash_lane_surface_for_state(state)
+                    row = row + (surface, surface, 0, 0, False)
+            rows.append(row)
         return columns, rows
 
 
@@ -756,10 +1025,12 @@ def create_sampler(
     batch_size: int = 1,
     resume_step: int | None = None,
     topk_children: int = 2,
+    parent_balance_surface: str = "disabled",
 ) -> StateSampler:
     """Factory function to create samplers. Pass the env type (from config.env_type)."""
     if not log_path:
         raise ValueError("log_path is required when using PUCT sampler")
+    parent_balance_surface = _validate_parent_balance_surface(parent_balance_surface)
     sampler_path = os.path.join(log_path, "puct_sampler.json")
     return PUCTSampler(
         file_path=sampler_path,
@@ -768,6 +1039,7 @@ def create_sampler(
         batch_size=batch_size,
         resume_step=resume_step,
         topk_children=topk_children,
+        parent_balance_surface=parent_balance_surface,
     )
 
 
@@ -778,6 +1050,7 @@ def get_or_create_sampler_with_default(
     batch_size: int = 1,
     resume_step: int | None = None,
     topk_children: int = 2,
+    parent_balance_surface: str = "disabled",
 ) -> StateSampler:
     """Get sampler. Initial experience is created via env_type.create_initial_state."""
     return create_sampler(
@@ -787,4 +1060,5 @@ def get_or_create_sampler_with_default(
         batch_size=batch_size,
         resume_step=resume_step,
         topk_children=topk_children,
+        parent_balance_surface=parent_balance_surface,
     )
diff --git a/ttt_discover/discovery.py b/ttt_discover/discovery.py
index 6cdb434..a9df8bc 100644
--- a/ttt_discover/discovery.py
+++ b/ttt_discover/discovery.py
@@ -63,6 +63,7 @@ class DiscoverConfig:
     codex_initial_program_paths: tuple[str, ...] = ()
     codex_initial_pool_paths: tuple[str, ...] = ()
     codex_autonomous: bool = False
+    codex_parent_balance_surface: Literal["disabled", "id_hash_lane"] = "disabled"
 
 
 def init_ray(num_cpus_per_task: int, env_type: str):
@@ -146,6 +147,7 @@ async def discover_impl(config: DiscoverConfig):
             initial_program_paths=config.codex_initial_program_paths,
             initial_pool_paths=config.codex_initial_pool_paths,
             autonomous=config.codex_autonomous,
+            parent_balance_surface=config.codex_parent_balance_surface,
             wandb_project=config.wandb_project,
             wandb_name=config.experiment_name,
             log_path=log_path,
diff --git a/ttt_discover/rl/codex_no_finetune.py b/ttt_discover/rl/codex_no_finetune.py
index 18a0e9e..ad094aa 100644
--- a/ttt_discover/rl/codex_no_finetune.py
+++ b/ttt_discover/rl/codex_no_finetune.py
@@ -137,6 +137,7 @@ class CodexNoFinetuneConfig:
     initial_program_paths: tuple[str, ...] = ()
     initial_pool_paths: tuple[str, ...] = ()
     topk_children: int = 16
+    parent_balance_surface: Literal["disabled", "id_hash_lane"] = "disabled"
 
     # False is TTT Discover: non-auto, many short samples. True is AutoEvolve:
     # one Codex deep-dive workspace per sample with file edits and shell access.
@@ -960,6 +961,7 @@ def _build_sampler(cfg: CodexNoFinetuneConfig, start_batch: int) -> StateSampler
         batch_size=cfg.groups_per_batch,
         resume_step=start_batch if start_batch > 0 else None,
         topk_children=cfg.topk_children,
+        parent_balance_surface=cfg.parent_balance_surface,
     )
 
 
````
</details>

