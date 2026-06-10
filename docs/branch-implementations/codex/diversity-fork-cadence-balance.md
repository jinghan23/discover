# codex/diversity-fork-cadence-balance

## Summary

按同一 birth timestep/父系 fork 的节奏分类 singleton_birth/paired_birth/burst_birth/seed，平衡不同 fork cadence 的父节点。

## Branch State

- Worktree: `/opt/tiger/discover-fork-cadence-balance`
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

- `codex_fork_cadence_balance`
- `cohort_sizes`
- `fork_cadence_balance`

### Constants

- `FORK_CADENCE_SURFACES`

### Classes

- None

### Functions

- `_empty_fork_cadence_counts`
- `_sanitize_fork_cadence_counts`
- `_fork_cadence_direct_parent_id`
- `_fork_cadence_cohort_sizes`
- `_classify_fork_cadence_surface`
- `_available_fork_cadence_entries`
- `_count_fork_cadence_available`
- `_choose_fork_cadence_surface`
- `_record_fork_cadence_pick`
- `_sample_fork_cadence_balanced`

## Diff Summary

- Worktree tracked shortstat: `4 files changed, 269 insertions(+), 3 deletions(-)`
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

- `repro/gpu_mode/run_0609_fork_cadence_balance.sh (1118 bytes)`
- `repro/run_discovery.py (3898 bytes)`
- `tests/test_codex_fork_cadence_balance.py (16001 bytes)`

### Detected Test Functions

- `tests/test_codex_fork_cadence_balance.py::test_disabled_default_and_explicit_false_match_baseline_outputs`
- `tests/test_codex_fork_cadence_balance.py::test_classifier_uses_only_parent_and_timestep_structure`
- `tests/test_codex_fork_cadence_balance.py::test_least_count_fixed_surface_order_and_within_surface_puct_order`
- `tests/test_codex_fork_cadence_balance.py::test_multi_parent_sampling_updates_counts_and_preserves_blocking`
- `tests/test_codex_fork_cadence_balance.py::test_fallback_increments_classified_counts_and_respects_blocking`
- `tests/test_codex_fork_cadence_balance.py::test_save_resume_restores_missing_and_bad_counts`
- `tests/test_codex_fork_cadence_balance.py::test_enabled_only_table_and_metrics`
- `tests/test_codex_fork_cadence_balance.py::test_cli_and_config_plumbing_source_text`

## Untracked File Snapshots

These files are not part of `git diff` yet, but they are part of the worktree implementation state recorded here.

### `repro/gpu_mode/run_0609_fork_cadence_balance.sh`

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

EXPERIMENT_NAME="${EXPERIMENT_NAME:-gpu-mode-0609-fork-cadence-balance}"
NUM_EPOCHS="${NUM_EPOCHS:-50}"
GROUP_SIZE="${GROUP_SIZE:-1}"
GROUPS_PER_BATCH="${GROUPS_PER_BATCH:-1}"
EVAL_TIMEOUT="${EVAL_TIMEOUT:-1200}"
CODEX_CLI_TIMEOUT="${CODEX_CLI_TIMEOUT:-600}"
CODEX_MAX_CONCURRENT_REQUESTS="${CODEX_MAX_CONCURRENT_REQUESTS:-1}"

export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"

python repro/run_discovery.py "${MODE}" \
    --experiment-name "${EXPERIMENT_NAME}" \
    --num-epochs "${NUM_EPOCHS}" \
    --group-size "${GROUP_SIZE}" \
    --groups-per-batch "${GROUPS_PER_BATCH}" \
    --eval-timeout "${EVAL_TIMEOUT}" \
    --codex-cli-timeout "${CODEX_CLI_TIMEOUT}" \
    --codex-max-concurrent-requests "${CODEX_MAX_CONCURRENT_REQUESTS}" \
    --codex-fork-cadence-balance \
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
        description="Run the Codex no-finetune GPUMode discovery repro."
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
        default="gpu-mode-0609-fork-cadence-balance",
    )
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--group-size", type=int, default=1)
    parser.add_argument("--groups-per-batch", type=int, default=1)
    parser.add_argument("--num-cpus-per-task", type=int, default=1)
    parser.add_argument("--eval-timeout", type=int, default=1200)
    parser.add_argument("--codex-model-name", default=None)
    parser.add_argument("--codex-cli-command", default="codex")
    parser.add_argument("--codex-cli-timeout", type=float, default=600.0)
    parser.add_argument("--codex-max-concurrent-requests", type=int, default=1)
    parser.add_argument(
        "--codex-fork-cadence-balance",
        action="store_true",
        default=False,
    )
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace, env_type: type):
    from ttt_discover.codex_utils.discovery import DiscoverConfig

    return DiscoverConfig(
        runner="codex_no_finetune",
        env_type=env_type,
        problem_type=args.problem_type,
        experiment_name=args.experiment_name,
        wandb_project=args.wandb_project,
        num_epochs=args.num_epochs,
        groups_per_batch=args.groups_per_batch,
        group_size=args.group_size,
        num_cpus_per_task=args.num_cpus_per_task,
        eval_timeout=args.eval_timeout,
        codex_backend="cli",
        codex_model_name=_none_if_empty(args.codex_model_name),
        codex_cli_command=args.codex_cli_command,
        codex_cli_sandbox="read-only",
        codex_cli_timeout=args.codex_cli_timeout,
        codex_max_concurrent_requests=args.codex_max_concurrent_requests,
        codex_autonomous=False,
        codex_fork_cadence_balance=args.codex_fork_cadence_balance,
    )


def _dry_run_summary(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "runner": "codex_no_finetune",
        "problem_type": args.problem_type,
        "experiment_name": args.experiment_name,
        "num_epochs": args.num_epochs,
        "groups_per_batch": args.groups_per_batch,
        "group_size": args.group_size,
        "num_cpus_per_task": args.num_cpus_per_task,
        "eval_timeout": args.eval_timeout,
        "codex_model_name": _none_if_empty(args.codex_model_name),
        "codex_cli_command": args.codex_cli_command,
        "codex_cli_timeout": args.codex_cli_timeout,
        "codex_max_concurrent_requests": args.codex_max_concurrent_requests,
        "codex_fork_cadence_balance": args.codex_fork_cadence_balance,
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.dry_run or args.command == "dry-run":
        print(json.dumps(_dry_run_summary(args), indent=2, sort_keys=True))
        return

    from examples.gpu_mode.env import GpuModeEnv
    from ttt_discover.codex_utils.discovery import discover

    discover(build_config(args, env_type=GpuModeEnv))


if __name__ == "__main__":
    main()
````

### `tests/test_codex_fork_cadence_balance.py`

````python
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ttt_discover.codex_utils.runtime import State
from ttt_discover.codex_utils.sampler import (
    FORK_CADENCE_SURFACES,
    PUCTSampler,
    _classify_fork_cadence_surface,
    _empty_fork_cadence_counts,
    _fork_cadence_cohort_sizes,
    _sampler_file_for_step,
)


class DummyEnv:
    state_type = State
    refresh_calls: list[tuple[str, str]] = []

    @staticmethod
    def create_initial_state(problem_type: str) -> State:
        return _state(f"initial-{problem_type}", 0.0)

    @staticmethod
    def refresh_initial_state(state: State, problem_type: str) -> None:
        DummyEnv.refresh_calls.append((state.id, problem_type))
        state.observation = f"refreshed:{problem_type}"


def _state(
    state_id: str,
    value: float,
    *,
    timestep: int = 0,
    parent_id: str | None = None,
    parents: list[dict] | None = None,
    code: str | None = None,
    construction: list | None = None,
    observation: str = "",
) -> State:
    if parents is None:
        parents = (
            [{"id": parent_id, "timestep": timestep - 1}]
            if parent_id is not None
            else []
        )
    return State(
        id=state_id,
        timestep=timestep,
        value=value,
        construction=construction if construction is not None else [state_id],
        code=code if code is not None else f"code-{state_id}",
        parent_values=[value - 1.0] if parents else [],
        parents=parents,
        observation=observation,
    )


def _clone_states(states: list[State]) -> list[State]:
    return [State.from_dict(state.to_dict()) for state in states]


def _sampler(
    tmp_path: Path,
    name: str,
    *,
    fork_cadence_balance: bool = False,
) -> PUCTSampler:
    return PUCTSampler(
        file_path=str(tmp_path / f"{name}.json"),
        env_type=DummyEnv,
        problem_type="p",
        batch_size=0,
        topk_children=0,
        fork_cadence_balance=fork_cadence_balance,
    )


def _sample_ids(sampler: PUCTSampler, num_states: int) -> list[str]:
    return [state.id for state in sampler.sample_states(num_states)]


@pytest.mark.parametrize(
    ("num_states", "expected_ids"),
    [
        (1, ["root"]),
        (2, ["root", "other"]),
    ],
)
def test_disabled_default_and_explicit_false_match_baseline_outputs(
    tmp_path: Path,
    num_states: int,
    expected_ids: list[str],
) -> None:
    states = [
        _state("root", 10.0),
        _state("child", 9.0, timestep=1, parent_id="root"),
        _state("other", 8.0),
    ]
    default_sampler = PUCTSampler(
        file_path=str(tmp_path / "default.json"),
        env_type=DummyEnv,
        problem_type="p",
        batch_size=0,
        topk_children=0,
    )
    explicit_false_sampler = _sampler(tmp_path, "explicit_false")
    default_sampler._states = _clone_states(states)
    explicit_false_sampler._states = _clone_states(states)
    default_sampler._initial_states = [default_sampler._states[0]]
    explicit_false_sampler._initial_states = [explicit_false_sampler._states[0]]

    DummyEnv.refresh_calls = []
    assert _sample_ids(default_sampler, num_states) == expected_ids
    assert _sample_ids(explicit_false_sampler, num_states) == expected_ids

    assert default_sampler._last_puct_stats == explicit_false_sampler._last_puct_stats
    assert default_sampler.get_sample_table() == explicit_false_sampler.get_sample_table()
    assert default_sampler.get_sample_stats() == explicit_false_sampler.get_sample_stats()
    assert "fork_cadence_sample_counts" not in default_sampler.get_sample_stats()
    assert all(
        not key.startswith("puct/fork_cadence/")
        for key in default_sampler.get_sample_stats()
    )
    assert DummyEnv.refresh_calls == [("root", "p"), ("root", "p")]

    columns, _rows = default_sampler.get_sample_table()
    assert "fork_cadence_surface" not in columns
    assert "fork_cadence_count_before" not in columns

    default_sampler.flush(step=num_states)
    explicit_false_sampler.flush(step=num_states)
    default_store = json.loads(
        Path(_sampler_file_for_step(default_sampler.file_path, num_states)).read_text()
    )
    explicit_store = json.loads(
        Path(
            _sampler_file_for_step(explicit_false_sampler.file_path, num_states)
        ).read_text()
    )
    assert "fork_cadence_sample_counts" not in default_store
    assert "fork_cadence_sample_counts" not in explicit_store
    assert default_store.keys() == explicit_store.keys()


def test_classifier_uses_only_parent_and_timestep_structure() -> None:
    no_parent = _state("seed-a", 0.0)
    malformed_missing = _state("seed-b", 0.0, parents=[{}])
    malformed_not_dict = _state("seed-c", 0.0, parents=[])
    malformed_not_dict.parents = [None]
    singleton = _state("singleton", 1.0, timestep=1, parent_id="p1")
    pair_a = _state("pair-a", 2.0, timestep=1, parent_id="p2")
    pair_b = _state("pair-b", 3.0, timestep=1, parent_id="p2")
    burst_a = _state("burst-a", 4.0, timestep=1, parent_id="p3")
    burst_b = _state("burst-b", 5.0, timestep=1, parent_id="p3")
    burst_c = _state("burst-c", 6.0, timestep=1, parent_id="p3")
    diff_timestep_a = _state("diff-a", 7.0, timestep=1, parent_id="p4")
    diff_timestep_b = _state("diff-b", 8.0, timestep=2, parent_id="p4")
    states = [
        no_parent,
        malformed_missing,
        malformed_not_dict,
        singleton,
        pair_a,
        pair_b,
        burst_a,
        burst_b,
        burst_c,
        diff_timestep_a,
        diff_timestep_b,
    ]

    cohort_sizes = _fork_cadence_cohort_sizes(states)

    assert _classify_fork_cadence_surface(no_parent, cohort_sizes) == "seed"
    assert _classify_fork_cadence_surface(malformed_missing, cohort_sizes) == "seed"
    assert _classify_fork_cadence_surface(malformed_not_dict, cohort_sizes) == "seed"
    assert _classify_fork_cadence_surface(singleton, cohort_sizes) == "singleton_birth"
    assert _classify_fork_cadence_surface(pair_a, cohort_sizes) == "paired_birth"
    assert _classify_fork_cadence_surface(pair_b, cohort_sizes) == "paired_birth"
    assert _classify_fork_cadence_surface(burst_a, cohort_sizes) == "burst_birth"
    assert _classify_fork_cadence_surface(burst_b, cohort_sizes) == "burst_birth"
    assert _classify_fork_cadence_surface(burst_c, cohort_sizes) == "burst_birth"
    assert (
        _classify_fork_cadence_surface(diff_timestep_a, cohort_sizes)
        == "singleton_birth"
    )
    assert (
        _classify_fork_cadence_surface(diff_timestep_b, cohort_sizes)
        == "singleton_birth"
    )

    plain = _state(
        "content",
        1.0,
        timestep=7,
        parent_id="content-parent",
        code="code",
        construction=["construction"],
        observation="observation",
    )
    forbidden_changed = _state(
        "content",
        999.0,
        timestep=7,
        parent_id="content-parent",
        code="different code",
        construction=["different construction"],
        observation="different logs",
    )
    assert _fork_cadence_cohort_sizes([plain]) == _fork_cadence_cohort_sizes(
        [forbidden_changed]
    )
    assert _classify_fork_cadence_surface(
        plain,
        _fork_cadence_cohort_sizes([plain]),
    ) == _classify_fork_cadence_surface(
        forbidden_changed,
        _fork_cadence_cohort_sizes([forbidden_changed]),
    )


def test_least_count_fixed_surface_order_and_within_surface_puct_order(
    tmp_path: Path,
) -> None:
    states = [
        _state("seed-top", 100.0),
        _state("singleton", 10.0, timestep=1, parent_id="p-single"),
        _state("pair-high", 90.0, timestep=1, parent_id="p-pair"),
        _state("pair-low", 80.0, timestep=1, parent_id="p-pair"),
    ]

    equal_counts = _sampler(tmp_path, "equal_counts", fork_cadence_balance=True)
    equal_counts._states = _clone_states(states)
    assert _sample_ids(equal_counts, 1) == ["singleton"]
    assert equal_counts._last_fork_cadence_surfaces == ["singleton_birth"]

    least_seed = _sampler(tmp_path, "least_seed", fork_cadence_balance=True)
    least_seed._states = _clone_states(states)
    least_seed._fork_cadence_sample_counts = {
        "singleton_birth": 4,
        "paired_birth": 4,
        "burst_birth": 4,
        "seed": 0,
    }
    assert _sample_ids(least_seed, 1) == ["seed-top"]
    assert least_seed._last_fork_cadence_surfaces == ["seed"]

    paired_tie = _sampler(tmp_path, "paired_tie", fork_cadence_balance=True)
    paired_tie._states = _clone_states(states)
    paired_tie._fork_cadence_sample_counts = {
        "singleton_birth": 5,
        "paired_birth": 0,
        "burst_birth": 0,
        "seed": 0,
    }
    assert _sample_ids(paired_tie, 1) == ["pair-high"]
    assert paired_tie._last_fork_cadence_surfaces == ["paired_birth"]


def test_multi_parent_sampling_updates_counts_and_preserves_blocking(
    tmp_path: Path,
) -> None:
    sampler = _sampler(tmp_path, "multi", fork_cadence_balance=True)
    sampler._states = [
        _state("singleton", 10.0, timestep=1, parent_id="p-single"),
        _state("pair-high", 100.0, timestep=1, parent_id="p-pair"),
        _state("pair-low", 90.0, timestep=1, parent_id="p-pair"),
        _state("seed", 80.0),
    ]

    assert _sample_ids(sampler, 3) == ["singleton", "pair-high", "seed"]
    assert sampler._last_fork_cadence_surfaces == [
        "singleton_birth",
        "paired_birth",
        "seed",
    ]
    assert sampler._last_fork_cadence_count_before == [0, 0, 0]
    assert sampler._fork_cadence_sample_counts["singleton_birth"] == 1
    assert sampler._fork_cadence_sample_counts["paired_birth"] == 1
    assert sampler._fork_cadence_sample_counts["seed"] == 1

    blocking = _sampler(tmp_path, "blocking", fork_cadence_balance=True)
    blocking._states = [
        _state("root", 100.0),
        _state("child", 90.0, timestep=1, parent_id="root"),
        _state("other-seed", 1.0),
    ]
    blocking._fork_cadence_sample_counts = {
        "singleton_birth": 5,
        "paired_birth": 5,
        "burst_birth": 5,
        "seed": 0,
    }

    assert _sample_ids(blocking, 2) == ["root", "other-seed"]
    assert blocking._last_fork_cadence_surfaces == ["seed", "seed"]
    assert blocking._last_fork_cadence_count_before == [0, 1]


def test_fallback_increments_classified_counts_and_respects_blocking(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sampler = _sampler(tmp_path, "fallback", fork_cadence_balance=True)
    sampler._states = [
        _state("root", 100.0),
        _state("child", 90.0, timestep=1, parent_id="root"),
        _state("singleton", 80.0, timestep=1, parent_id="p-single"),
    ]
    monkeypatch.setattr(sampler, "_choose_fork_cadence_surface", lambda _groups: None)

    assert _sample_ids(sampler, 2) == ["root", "singleton"]
    assert sampler._last_fork_cadence_surfaces == ["seed", "singleton_birth"]
    assert sampler._last_fork_cadence_count_before == [0, 0]
    assert sampler._fork_cadence_sample_counts["seed"] == 1
    assert sampler._fork_cadence_sample_counts["singleton_birth"] == 1


def test_save_resume_restores_missing_and_bad_counts(tmp_path: Path) -> None:
    enabled = _sampler(tmp_path, "enabled", fork_cadence_balance=True)
    enabled._states = [_state("seed", 1.0)]
    assert _sample_ids(enabled, 1) == ["seed"]
    enabled.flush(step=2)

    restored = PUCTSampler(
        file_path=enabled.file_path,
        env_type=DummyEnv,
        problem_type="p",
        batch_size=0,
        resume_step=2,
        topk_children=0,
        fork_cadence_balance=True,
    )
    assert restored._fork_cadence_sample_counts["seed"] == 1

    missing = _sampler(tmp_path, "missing", fork_cadence_balance=False)
    missing._states = [_state("seed", 1.0)]
    missing.flush(step=3)
    missing_restored = PUCTSampler(
        file_path=missing.file_path,
        env_type=DummyEnv,
        problem_type="p",
        batch_size=0,
        resume_step=3,
        topk_children=0,
        fork_cadence_balance=True,
    )
    assert missing_restored._fork_cadence_sample_counts == _empty_fork_cadence_counts()

    bad_path = Path(_sampler_file_for_step(enabled.file_path, 2))
    store = json.loads(bad_path.read_text())
    store["fork_cadence_sample_counts"] = {
        "singleton_birth": True,
        "paired_birth": "4",
        "burst_birth": -1,
        "seed": 2,
        "unknown": 999,
    }
    bad_path.write_text(json.dumps(store))
    bad_restored = PUCTSampler(
        file_path=enabled.file_path,
        env_type=DummyEnv,
        problem_type="p",
        batch_size=0,
        resume_step=2,
        topk_children=0,
        fork_cadence_balance=True,
    )
    assert bad_restored._fork_cadence_sample_counts == {
        "singleton_birth": 0,
        "paired_birth": 0,
        "burst_birth": 0,
        "seed": 2,
    }


def test_enabled_only_table_and_metrics(tmp_path: Path) -> None:
    enabled = _sampler(tmp_path, "enabled_obs", fork_cadence_balance=True)
    enabled._states = [_state("singleton", 1.0, timestep=1, parent_id="p")]
    assert _sample_ids(enabled, 1) == ["singleton"]

    columns, rows = enabled.get_sample_table()
    assert columns[-2:] == ["fork_cadence_surface", "fork_cadence_count_before"]
    assert rows[0][-2:] == ("singleton_birth", 0)

    metrics = enabled.get_sample_stats()
    assert metrics["puct/fork_cadence/enabled"] == 1
    assert metrics["puct/fork_cadence/count/singleton_birth"] == 1
    assert metrics["puct/fork_cadence/available/singleton_birth"] == 1
    for surface in FORK_CADENCE_SURFACES:
        assert f"puct/fork_cadence/count/{surface}" in metrics
        assert f"puct/fork_cadence/available/{surface}" in metrics

    disabled = _sampler(tmp_path, "disabled_obs", fork_cadence_balance=False)
    disabled._states = [_state("singleton", 1.0, timestep=1, parent_id="p")]
    disabled.sample_states(1)
    disabled_columns, _disabled_rows = disabled.get_sample_table()
    disabled_metrics = disabled.get_sample_stats()
    assert "fork_cadence_surface" not in disabled_columns
    assert "fork_cadence_count_before" not in disabled_columns
    assert all(
        not key.startswith("puct/fork_cadence/")
        for key in disabled_metrics
    )


def test_cli_and_config_plumbing_source_text() -> None:
    from repro.run_discovery import _dry_run_summary, parse_args

    root = Path(__file__).resolve().parents[1]
    run_discovery = (root / "repro/run_discovery.py").read_text()
    sampler = (root / "ttt_discover/codex_utils/sampler.py").read_text()
    codex_runner = (root / "ttt_discover/rl/codex_no_finetune.py").read_text()
    codex_wrapper = (root / "ttt_discover/codex_utils/discovery.py").read_text()
    discovery_wrapper = (root / "ttt_discover/discovery.py").read_text()

    assert "--codex-fork-cadence-balance" in run_discovery
    assert "default=False" in run_discovery
    assert "codex_fork_cadence_balance" in run_discovery
    assert "codex_fork_cadence_balance: bool = False" in codex_wrapper
    assert "codex_fork_cadence_balance: bool = False" in discovery_wrapper
    assert "fork_cadence_balance: bool = False" in codex_runner
    assert "fork_cadence_balance: bool = False" in sampler
    assert "fork_cadence_balance=cfg.fork_cadence_balance" in codex_runner
    assert (
        "fork_cadence_balance=config.codex_fork_cadence_balance"
        in codex_wrapper
    )
    assert (
        "fork_cadence_balance=config.codex_fork_cadence_balance"
        in discovery_wrapper
    )

    default_args = parse_args(["dry-run"])
    enabled_args = parse_args(["--dry-run", "--codex-fork-cadence-balance"])
    default_summary = _dry_run_summary(default_args)
    enabled_summary = _dry_run_summary(enabled_args)
    assert default_summary["codex_model_name"] is None
    assert default_summary["codex_fork_cadence_balance"] is False
    assert enabled_summary["codex_fork_cadence_balance"] is True
````

## Diff Stat

### Worktree Tracked Diff Stat

````text
 ttt_discover/codex_utils/discovery.py |   2 +
 ttt_discover/codex_utils/sampler.py   | 266 +++++++++++++++++++++++++++++++++-
 ttt_discover/discovery.py             |   2 +
 ttt_discover/rl/codex_no_finetune.py  |   2 +
 4 files changed, 269 insertions(+), 3 deletions(-)
````
## Raw Diff

<details>
<summary>Tracked worktree patch</summary>

````diff
diff --git a/ttt_discover/codex_utils/discovery.py b/ttt_discover/codex_utils/discovery.py
index b21a80f..e5c94b3 100644
--- a/ttt_discover/codex_utils/discovery.py
+++ b/ttt_discover/codex_utils/discovery.py
@@ -50,6 +50,7 @@ class DiscoverConfig:
     codex_initial_program_paths: tuple[str, ...] = ()
     codex_initial_pool_paths: tuple[str, ...] = ()
     codex_autonomous: bool = False
+    codex_fork_cadence_balance: bool = False
 
 
 def _run_codex_no_finetune(config: DiscoverConfig) -> None:
@@ -85,6 +86,7 @@ def _run_codex_no_finetune(config: DiscoverConfig) -> None:
         initial_program_paths=config.codex_initial_program_paths,
         initial_pool_paths=config.codex_initial_pool_paths,
         autonomous=config.codex_autonomous,
+        fork_cadence_balance=config.codex_fork_cadence_balance,
         wandb_project=config.wandb_project,
         wandb_name=experiment_name,
         log_path=log_path,
diff --git a/ttt_discover/codex_utils/sampler.py b/ttt_discover/codex_utils/sampler.py
index 5c5d4d1..2796f51 100644
--- a/ttt_discover/codex_utils/sampler.py
+++ b/ttt_discover/codex_utils/sampler.py
@@ -16,6 +16,13 @@ from ttt_discover.codex_utils.runtime import State, state_from_dict, to_json_ser
 
 logger = logging.getLogger(__name__)
 
+FORK_CADENCE_SURFACES = (
+    "singleton_birth",
+    "paired_birth",
+    "burst_birth",
+    "seed",
+)
+
 
 @contextmanager
 def _file_lock(lock_path: str, *, poll_s: float = 0.05, stale_s: float = 600.0):
@@ -120,6 +127,60 @@ def _sampler_file_for_step(base_path: str, step: int) -> str:
     return f"{base_name}_step_{step:06d}.json"
 
 
+def _empty_fork_cadence_counts() -> dict[str, int]:
+    return {surface: 0 for surface in FORK_CADENCE_SURFACES}
+
+
+def _sanitize_fork_cadence_counts(raw_counts: Any) -> dict[str, int]:
+    counts = _empty_fork_cadence_counts()
+    if not isinstance(raw_counts, dict):
+        return counts
+    for surface in FORK_CADENCE_SURFACES:
+        value = raw_counts.get(surface, 0)
+        if type(value) is int and value >= 0:
+            counts[surface] = value
+    return counts
+
+
+def _fork_cadence_direct_parent_id(state: State) -> str | None:
+    parents = getattr(state, "parents", None)
+    if not isinstance(parents, list) or not parents:
+        return None
+    parent_edge = parents[0]
+    if not isinstance(parent_edge, dict):
+        return None
+    parent_id = parent_edge.get("id")
+    if not isinstance(parent_id, str) or not parent_id:
+        return None
+    return parent_id
+
+
+def _fork_cadence_cohort_sizes(states: list[State]) -> dict[tuple[str, int], int]:
+    cohort_sizes: dict[tuple[str, int], int] = {}
+    for state in states:
+        parent_id = _fork_cadence_direct_parent_id(state)
+        if parent_id is None:
+            continue
+        key = (parent_id, state.timestep)
+        cohort_sizes[key] = cohort_sizes.get(key, 0) + 1
+    return cohort_sizes
+
+
+def _classify_fork_cadence_surface(
+    state: State,
+    cohort_sizes: dict[tuple[str, int], int],
+) -> str:
+    parent_id = _fork_cadence_direct_parent_id(state)
+    if parent_id is None:
+        return "seed"
+    cohort_size = cohort_sizes.get((parent_id, state.timestep), 1)
+    if cohort_size == 1:
+        return "singleton_birth"
+    if cohort_size == 2:
+        return "paired_birth"
+    return "burst_birth"
+
+
 def create_initial_state(env_type: type, problem_type: str) -> State:
     """Create initial state by delegating to the env type. Custom envs implement create_initial_state on their class."""
     name = getattr(env_type, "env_name", env_type.__name__)
@@ -353,6 +414,7 @@ class PUCTSampler(StateSampler):
         resume_step: int | None = None,
         puct_c: float = 1.0,
         topk_children: int = 2,
+        fork_cadence_balance: bool = False,
     ):
         self.file_path = file_path
         self.env_type = env_type
@@ -360,6 +422,7 @@ class PUCTSampler(StateSampler):
         self.max_buffer_size = max_buffer_size
         self.batch_size = batch_size
         self.topk_children = topk_children
+        self.fork_cadence_balance = bool(fork_cadence_balance)
         self.puct_c = float(puct_c)
         
         self._states: list[State] = []
@@ -375,6 +438,10 @@ class PUCTSampler(StateSampler):
         self._T: int = 0
         self._last_scale: float = 1.0
         self._last_puct_stats: list[tuple[int, float, float, float, float]] = []
+        self._fork_cadence_sample_counts: dict[str, int] = _empty_fork_cadence_counts()
+        self._last_fork_cadence_surfaces: list[str] = []
+        self._last_fork_cadence_count_before: list[int] = []
+        self._last_fork_cadence_available_counts: dict[str, int] = _empty_fork_cadence_counts()
         
         if resume_step is not None:
             self._load(resume_step)
@@ -399,6 +466,10 @@ class PUCTSampler(StateSampler):
         self._n = store.get("puct_n", {}) or {}
         self._m = store.get("puct_m", {}) or {}
         self._T = int(store.get("puct_T", 0) or 0)
+        if self.fork_cadence_balance:
+            self._fork_cadence_sample_counts = _sanitize_fork_cadence_counts(
+                store.get("fork_cadence_sample_counts", {})
+            )
 
     def _save(self, step: int):
         save_path = _sampler_file_for_step(self.file_path, step)
@@ -411,6 +482,8 @@ class PUCTSampler(StateSampler):
             "puct_m": self._m,
             "puct_T": self._T,
         }
+        if self.fork_cadence_balance:
+            store["fork_cadence_sample_counts"] = self._fork_cadence_sample_counts
         with _file_lock(f"{save_path}.lock"):
             _atomic_write_json(save_path, store)
 
@@ -489,6 +562,153 @@ class PUCTSampler(StateSampler):
                     queue.append(child_id)
         return lineage
 
+    def _available_fork_cadence_entries(
+        self,
+        scores: list[tuple],
+        picked_ids: set[str],
+        blocked_ids: set[str],
+        use_lineage_blocking: bool,
+    ) -> list[tuple]:
+        available = []
+        for entry in scores:
+            state = entry[2]
+            if state.id in picked_ids:
+                continue
+            if use_lineage_blocking and state.id in blocked_ids:
+                continue
+            available.append(entry)
+        return available
+
+    def _count_fork_cadence_available(
+        self,
+        scores: list[tuple],
+        cohort_sizes: dict[tuple[str, int], int],
+        picked_ids: set[str],
+        blocked_ids: set[str],
+        use_lineage_blocking: bool,
+    ) -> dict[str, int]:
+        counts = _empty_fork_cadence_counts()
+        for entry in self._available_fork_cadence_entries(
+            scores,
+            picked_ids,
+            blocked_ids,
+            use_lineage_blocking,
+        ):
+            surface = _classify_fork_cadence_surface(entry[2], cohort_sizes)
+            counts[surface] += 1
+        return counts
+
+    def _choose_fork_cadence_surface(self, by_surface: dict[str, list[tuple]]) -> str | None:
+        available_surfaces = [
+            surface for surface in FORK_CADENCE_SURFACES if by_surface.get(surface)
+        ]
+        if not available_surfaces:
+            return None
+        return min(
+            available_surfaces,
+            key=lambda surface: (
+                self._fork_cadence_sample_counts.get(surface, 0),
+                FORK_CADENCE_SURFACES.index(surface),
+            ),
+        )
+
+    def _record_fork_cadence_pick(
+        self,
+        entry: tuple,
+        cohort_sizes: dict[tuple[str, int], int],
+        picked: list[State],
+        top_scores: list[tuple],
+        picked_ids: set[str],
+        blocked_ids: set[str],
+        children_map: dict[str, set[str]],
+        use_lineage_blocking: bool,
+    ) -> None:
+        state = entry[2]
+        surface = _classify_fork_cadence_surface(state, cohort_sizes)
+        count_before = self._fork_cadence_sample_counts.get(surface, 0)
+        self._last_fork_cadence_surfaces.append(surface)
+        self._last_fork_cadence_count_before.append(count_before)
+        self._fork_cadence_sample_counts[surface] = count_before + 1
+        picked.append(state)
+        top_scores.append(entry)
+        picked_ids.add(state.id)
+        if use_lineage_blocking:
+            blocked_ids.update(self._get_full_lineage(state, children_map))
+
+    def _sample_fork_cadence_balanced(
+        self,
+        scores: list[tuple],
+        num_states: int,
+    ) -> tuple[list[State], list[tuple]]:
+        cohort_sizes = _fork_cadence_cohort_sizes(self._states)
+        use_lineage_blocking = num_states > 1
+        children_map = self._build_children_map() if use_lineage_blocking else {}
+        picked: list[State] = []
+        top_scores: list[tuple] = []
+        picked_ids: set[str] = set()
+        blocked_ids: set[str] = set()
+        self._last_fork_cadence_surfaces = []
+        self._last_fork_cadence_count_before = []
+        self._last_fork_cadence_available_counts = self._count_fork_cadence_available(
+            scores,
+            cohort_sizes,
+            picked_ids,
+            blocked_ids,
+            use_lineage_blocking,
+        )
+
+        while len(picked) < num_states:
+            available = self._available_fork_cadence_entries(
+                scores,
+                picked_ids,
+                blocked_ids,
+                use_lineage_blocking,
+            )
+            if not available:
+                break
+
+            by_surface = {surface: [] for surface in FORK_CADENCE_SURFACES}
+            for entry in available:
+                surface = _classify_fork_cadence_surface(entry[2], cohort_sizes)
+                by_surface[surface].append(entry)
+
+            chosen_surface = self._choose_fork_cadence_surface(by_surface)
+            if chosen_surface is None:
+                break
+
+            self._record_fork_cadence_pick(
+                by_surface[chosen_surface][0],
+                cohort_sizes,
+                picked,
+                top_scores,
+                picked_ids,
+                blocked_ids,
+                children_map,
+                use_lineage_blocking,
+            )
+
+        if len(picked) < num_states:
+            for entry in scores:
+                if len(picked) >= num_states:
+                    break
+                state = entry[2]
+                if state.id in picked_ids:
+                    continue
+                if use_lineage_blocking and state.id in blocked_ids:
+                    continue
+                self._record_fork_cadence_pick(
+                    entry,
+                    cohort_sizes,
+                    picked,
+                    top_scores,
+                    picked_ids,
+                    blocked_ids,
+                    children_map,
+                    use_lineage_blocking,
+                )
+
+        return picked, top_scores
+
     def sample_states(self, num_states: int) -> list[State]:
         initial_ids = {s.id for s in self._initial_states}
         candidates = list(self._states)
@@ -501,6 +721,16 @@ class PUCTSampler(StateSampler):
             self._last_sampled_states = picked
             self._last_sampled_indices = []
             self._last_puct_stats = [(0, 0.0, 0.0, 0.0, 0.0) for _ in picked]
+            if self.fork_cadence_balance:
+                self._last_fork_cadence_surfaces = []
+                self._last_fork_cadence_count_before = []
+                self._last_fork_cadence_available_counts = _empty_fork_cadence_counts()
+                for state in picked:
+                    surface = "seed"
+                    count_before = self._fork_cadence_sample_counts[surface]
+                    self._last_fork_cadence_surfaces.append(surface)
+                    self._last_fork_cadence_count_before.append(count_before)
+                    self._fork_cadence_sample_counts[surface] = count_before + 1
             return picked
 
         vals = np.array([float(s.value if s.value is not None else float("-inf")) for s in candidates])
@@ -521,7 +751,9 @@ class PUCTSampler(StateSampler):
 
         scores.sort(key=lambda x: (x[0], x[1]), reverse=True)
 
-        if num_states > 1:
+        if self.fork_cadence_balance:
+            picked, top_scores = self._sample_fork_cadence_balanced(scores, num_states)
+        elif num_states > 1:
             children_map = self._build_children_map()
             picked, top_scores, blocked_ids = [], [], set()
             for entry in scores:
@@ -731,21 +963,45 @@ class PUCTSampler(StateSampler):
         stats.update(_stats(sampled_values, "puct/sampled_value"))
         stats.update(_stats(sampled_timesteps, "puct/sampled_timestep"))
         stats.update(_stats(sampled_constr_lens, "puct/sampled_construction_len"))
+        if self.fork_cadence_balance:
+            stats["puct/fork_cadence/enabled"] = 1
+            for surface in FORK_CADENCE_SURFACES:
+                stats[f"puct/fork_cadence/count/{surface}"] = (
+                    self._fork_cadence_sample_counts.get(surface, 0)
+                )
+                stats[f"puct/fork_cadence/available/{surface}"] = (
+                    self._last_fork_cadence_available_counts.get(surface, 0)
+                )
         return stats
 
     def get_sample_table(self) -> tuple[list[str], list[tuple]]:
         columns = ["buffer_idx", "timestep", "value", "terminal_value", "parent_value", "construction_len", "observation_len", "n", "Q", "P", "bonus", "score"]
+        if self.fork_cadence_balance:
+            columns.extend(["fork_cadence_surface", "fork_cadence_count_before"])
         rows = []
         if not self._last_sampled_states:
             return columns, rows
         indices = self._last_sampled_indices if len(self._last_sampled_indices) == len(self._last_sampled_states) else [-1] * len(self._last_sampled_states)
         stats = self._last_puct_stats if len(self._last_puct_stats) == len(self._last_sampled_states) else [(0, 0.0, 0.0, 0.0, 0.0)] * len(self._last_sampled_states)
-        for idx, state, (n, Q, P, bonus, score) in zip(indices, self._last_sampled_states, stats):
+        surfaces = (
+            self._last_fork_cadence_surfaces
+            if len(self._last_fork_cadence_surfaces) == len(self._last_sampled_states)
+            else ["seed"] * len(self._last_sampled_states)
+        )
+        counts_before = (
+            self._last_fork_cadence_count_before
+            if len(self._last_fork_cadence_count_before) == len(self._last_sampled_states)
+            else [0] * len(self._last_sampled_states)
+        )
+        for row_idx, (idx, state, (n, Q, P, bonus, score)) in enumerate(zip(indices, self._last_sampled_states, stats)):
             parent_val = state.parent_values[0] if state.parent_values else None
             constr = getattr(state, 'construction', None)
             constr_len = len(constr) if constr is not None else 0
             obs_len = len(state.observation) if state.observation else 0
-            rows.append((idx, state.timestep, state.value, 0, parent_val, constr_len, obs_len, n, Q, P, bonus, score))
+            row = (idx, state.timestep, state.value, 0, parent_val, constr_len, obs_len, n, Q, P, bonus, score)
+            if self.fork_cadence_balance:
+                row = row + (surfaces[row_idx], counts_before[row_idx])
+            rows.append(row)
         return columns, rows
 
 
@@ -756,6 +1012,7 @@ def create_sampler(
     batch_size: int = 1,
     resume_step: int | None = None,
     topk_children: int = 2,
+    fork_cadence_balance: bool = False,
 ) -> StateSampler:
     """Factory function to create samplers. Pass the env type (from config.env_type)."""
     if not log_path:
@@ -768,6 +1025,7 @@ def create_sampler(
         batch_size=batch_size,
         resume_step=resume_step,
         topk_children=topk_children,
+        fork_cadence_balance=fork_cadence_balance,
     )
 
 
@@ -778,6 +1036,7 @@ def get_or_create_sampler_with_default(
     batch_size: int = 1,
     resume_step: int | None = None,
     topk_children: int = 2,
+    fork_cadence_balance: bool = False,
 ) -> StateSampler:
     """Get sampler. Initial experience is created via env_type.create_initial_state."""
     return create_sampler(
@@ -787,4 +1046,5 @@ def get_or_create_sampler_with_default(
         batch_size=batch_size,
         resume_step=resume_step,
         topk_children=topk_children,
+        fork_cadence_balance=fork_cadence_balance,
     )
diff --git a/ttt_discover/discovery.py b/ttt_discover/discovery.py
index 6cdb434..4a98ac7 100644
--- a/ttt_discover/discovery.py
+++ b/ttt_discover/discovery.py
@@ -63,6 +63,7 @@ class DiscoverConfig:
     codex_initial_program_paths: tuple[str, ...] = ()
     codex_initial_pool_paths: tuple[str, ...] = ()
     codex_autonomous: bool = False
+    codex_fork_cadence_balance: bool = False
 
 
 def init_ray(num_cpus_per_task: int, env_type: str):
@@ -146,6 +147,7 @@ async def discover_impl(config: DiscoverConfig):
             initial_program_paths=config.codex_initial_program_paths,
             initial_pool_paths=config.codex_initial_pool_paths,
             autonomous=config.codex_autonomous,
+            fork_cadence_balance=config.codex_fork_cadence_balance,
             wandb_project=config.wandb_project,
             wandb_name=config.experiment_name,
             log_path=log_path,
diff --git a/ttt_discover/rl/codex_no_finetune.py b/ttt_discover/rl/codex_no_finetune.py
index 18a0e9e..475d1c7 100644
--- a/ttt_discover/rl/codex_no_finetune.py
+++ b/ttt_discover/rl/codex_no_finetune.py
@@ -137,6 +137,7 @@ class CodexNoFinetuneConfig:
     initial_program_paths: tuple[str, ...] = ()
     initial_pool_paths: tuple[str, ...] = ()
     topk_children: int = 16
+    fork_cadence_balance: bool = False
 
     # False is TTT Discover: non-auto, many short samples. True is AutoEvolve:
     # one Codex deep-dive workspace per sample with file edits and shell access.
@@ -960,6 +961,7 @@ def _build_sampler(cfg: CodexNoFinetuneConfig, start_batch: int) -> StateSampler
         batch_size=cfg.groups_per_batch,
         resume_step=start_batch if start_batch > 0 else None,
         topk_children=cfg.topk_children,
+        fork_cadence_balance=cfg.fork_cadence_balance,
     )
 
 
````
</details>

