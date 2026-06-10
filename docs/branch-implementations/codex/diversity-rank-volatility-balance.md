# codex/diversity-rank-volatility-balance

## Summary

持久记录上一轮 PUCT rank，比较当前 rank 得到 rank:new/up/down/flat，平衡不同 rank volatility 的候选。

## Branch State

- Worktree: `/opt/tiger/discover-rank-volatility-balance`
- HEAD: `2234282`
- Base used for comparison: `2234282`
- Commits ahead of base: `0`
- Commits behind base: `0`
- Group: `completed_19`
- Implementation location: `uncommitted worktree diff`

## Implemented Behavior

- The summary above is based on reading the changed Python code, added config fields, and sampler/prompt hooks in this branch.
- For sampler branches, the common pattern is to keep baseline PUCT ordering inside a chosen surface/bucket, while changing which surface/bucket gets the next parent pick.
- Tests detected: `13` test functions across untracked test files.

## Added Markers

### Config fields

- `codex_rank_volatility_balance`
- `codex_rank_volatility_delta`
- `clean`
- `counts`
- `rank_volatility_balance`
- `rank_volatility_delta`

### Constants

- `RANK_VOLATILITY_SURFACES`
- `_RANK_VOLATILITY_PREV_RANKS_KEY`
- `_RANK_VOLATILITY_SELECTED_COUNTS_KEY`

### Classes

- None

### Functions

- `_sanitize_rank_volatility_delta`
- `_sanitize_rank_volatility_prev_ranks`
- `_sanitize_rank_volatility_selected_counts`
- `_rank_volatility_metric_suffix`
- `classify_rank_volatility_surface`
- `_sanitize_rank_volatility_state`
- `_rank_volatility_current_ranks`
- `_rank_volatility_surface`
- `_rank_volatility_row`
- `_rank_volatility_available_entries`
- `_select_rank_volatility_entry`
- `_sample_states_rank_volatility_balanced`

## Diff Summary

- Worktree tracked shortstat: `4 files changed, 309 insertions(+), 3 deletions(-)`
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

- `repro/gpu_mode/run_0609_rank_volatility_balance.sh (692 bytes)`
- `repro/run_discovery.py (6462 bytes)`
- `tests/test_codex_rank_volatility_balance.py (13465 bytes)`

### Detected Test Functions

- `tests/test_codex_rank_volatility_balance.py::test_disabled_parity_no_rank_volatility_observability_or_persistence`
- `tests/test_codex_rank_volatility_balance.py::test_first_enabled_call_all_new_and_stores_live_rank_snapshot`
- `tests/test_codex_rank_volatility_balance.py::test_classifier_surfaces_malformed_previous_and_delta_clamping`
- `tests/test_codex_rank_volatility_balance.py::test_equal_count_tie_uses_earliest_puct_surface`
- `tests/test_codex_rank_volatility_balance.py::test_unequal_counts_pick_under_selected_surface_over_global_top`
- `tests/test_codex_rank_volatility_balance.py::test_within_chosen_surface_highest_puct_wins`
- `tests/test_codex_rank_volatility_balance.py::test_batch_lineage_blocking_recomputes_availability`
- `tests/test_codex_rank_volatility_balance.py::test_enabled_persistence_resume_and_save_sanitization`
- `tests/test_codex_rank_volatility_balance.py::test_record_failed_rollout_only_updates_puct_visits`
- `tests/test_codex_rank_volatility_balance.py::test_build_sampler_passes_rank_volatility_args`
- `tests/test_codex_rank_volatility_balance.py::test_run_discovery_dry_run_cli_defaults_enabled_override_and_validation`
- `tests/test_codex_rank_volatility_balance.py::test_gpu_mode_wrapper_dry_run_first_and_override_path`
- `tests/test_codex_rank_volatility_balance.py::test_classifier_helpers_only_read_state_id`

## Untracked File Snapshots

These files are not part of `git diff` yet, but they are part of the worktree implementation state recorded here.

### `repro/gpu_mode/run_0609_rank_volatility_balance.sh`

````bash
#!/usr/bin/env bash
set -euo pipefail

MODE="run"
if [[ $# -gt 0 && ( "$1" == "run" || "$1" == "dry-run" ) ]]; then
  MODE="$1"
  shift
fi

python -m repro.run_discovery "$MODE" \
  --task gpu_mode \
  --problem-type trimul \
  --experiment-name trimul_0609_rank_volatility_balance_gpu2 \
  --runner codex_no_finetune \
  --num-epochs 10 \
  --groups-per-batch 1 \
  --group-size 1 \
  --num-cpus-per-task 1 \
  --eval-timeout 530 \
  --codex-backend cli \
  --codex-model-name "" \
  --codex-cli-command codex \
  --codex-cli-sandbox read-only \
  --codex-cli-timeout 600 \
  --codex-max-concurrent-requests 1 \
  --codex-rank-volatility-balance \
  --codex-rank-volatility-delta 5 \
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or dry-run a Discover repro config.")
    parser.add_argument(
        "mode",
        nargs="?",
        choices=("run", "dry-run"),
        default="run",
        help="Use 'dry-run' to print the config payload without imports or execution.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the config payload without importing ttt_discover or running discovery.",
    )
    parser.add_argument("--task", choices=("gpu_mode",), default="gpu_mode")
    parser.add_argument("--problem-type", default="trimul")
    parser.add_argument(
        "--experiment-name",
        default="gpu-mode-0609-rank-volatility-balance",
    )
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument(
        "--runner",
        choices=("tinker_rl", "codex_no_finetune"),
        default="codex_no_finetune",
    )
    parser.add_argument(
        "--model-name",
        choices=("openai/gpt-oss-120b", "openai/gpt-oss-20b"),
        default="openai/gpt-oss-120b",
    )
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--group-size", type=int, default=1)
    parser.add_argument("--groups-per-batch", type=int, default=1)
    parser.add_argument("--num-cpus-per-task", type=int, default=1)
    parser.add_argument("--eval-timeout", type=int, default=530)
    parser.add_argument("--learning-rate", type=float, default=4e-5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--kl-penalty-coef", type=float, default=0.1)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--save-every", type=int, default=2)
    parser.add_argument("--phase1-max-tokens", type=int, default=26000)
    parser.add_argument("--codex-backend", choices=("cli", "responses"), default="cli")
    parser.add_argument("--codex-model-name", default=None)
    parser.add_argument("--codex-max-output-tokens", type=int, default=8192)
    parser.add_argument("--codex-temperature", type=float, default=None)
    parser.add_argument("--codex-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--codex-base-url", default=None)
    parser.add_argument("--codex-cli-command", default="codex")
    parser.add_argument(
        "--codex-cli-sandbox",
        choices=("read-only", "workspace-write", "danger-full-access"),
        default="read-only",
    )
    parser.add_argument("--codex-cli-timeout", type=float, default=None)
    parser.add_argument("--codex-max-concurrent-requests", type=int, default=1)
    parser.add_argument("--codex-initial-program", action="append", default=[])
    parser.add_argument("--codex-initial-pool", action="append", default=[])
    parser.add_argument("--codex-autonomous", action="store_true")
    parser.add_argument(
        "--codex-rank-volatility-balance",
        dest="codex_rank_volatility_balance",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--no-codex-rank-volatility-balance",
        dest="codex_rank_volatility_balance",
        action="store_false",
    )
    parser.add_argument("--codex-rank-volatility-delta", type=int, default=5)
    return parser.parse_args(argv)


def _normalize_optional_str(value: str | None) -> str | None:
    return value if value not in ("", None) else None


def _env_type_name(task: str) -> str:
    if task == "gpu_mode":
        return "examples.gpu_mode.env.GpuModeEnv"
    raise ValueError(f"Unknown task: {task}")


def _build_config_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "env_type": _env_type_name(args.task),
        "problem_type": args.problem_type,
        "model_name": args.model_name,
        "runner": args.runner,
        "lora_rank": args.lora_rank,
        "group_size": args.group_size,
        "groups_per_batch": args.groups_per_batch,
        "learning_rate": args.learning_rate,
        "num_epochs": args.num_epochs,
        "temperature": args.temperature,
        "kl_penalty_coef": args.kl_penalty_coef,
        "phase1_max_tokens": args.phase1_max_tokens,
        "save_every": args.save_every,
        "num_cpus_per_task": args.num_cpus_per_task,
        "eval_timeout": args.eval_timeout,
        "experiment_name": args.experiment_name,
        "wandb_project": _normalize_optional_str(args.wandb_project),
        "codex_model_name": _normalize_optional_str(args.codex_model_name),
        "codex_backend": args.codex_backend,
        "codex_max_output_tokens": args.codex_max_output_tokens,
        "codex_temperature": args.codex_temperature,
        "codex_api_key_env": args.codex_api_key_env,
        "codex_base_url": _normalize_optional_str(args.codex_base_url),
        "codex_cli_command": args.codex_cli_command,
        "codex_cli_sandbox": args.codex_cli_sandbox,
        "codex_cli_timeout": args.codex_cli_timeout,
        "codex_max_concurrent_requests": args.codex_max_concurrent_requests,
        "codex_initial_program_paths": tuple(args.codex_initial_program),
        "codex_initial_pool_paths": tuple(args.codex_initial_pool),
        "codex_autonomous": args.codex_autonomous,
        "codex_rank_volatility_balance": args.codex_rank_volatility_balance,
        "codex_rank_volatility_delta": args.codex_rank_volatility_delta,
    }


def _resolve_env_type(task: str) -> type:
    if task == "gpu_mode":
        from examples.gpu_mode.env import GpuModeEnv

        return GpuModeEnv
    raise ValueError(f"Unknown task: {task}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    payload = _build_config_payload(args)
    dry_run = args.dry_run or args.mode == "dry-run"
    if dry_run:
        print(
            json.dumps(
                {"mode": "dry-run", "discover_config": payload},
                indent=2,
                sort_keys=True,
            )
        )
        return

    from ttt_discover import DiscoverConfig, discover

    payload["env_type"] = _resolve_env_type(args.task)
    discover(DiscoverConfig(**payload))


if __name__ == "__main__":
    main()
````

### `tests/test_codex_rank_volatility_balance.py`

````python
from __future__ import annotations

import ast
import inspect
import json
import subprocess
import sys
import textwrap
from pathlib import Path

from ttt_discover.codex_utils.runtime import State
import ttt_discover.codex_utils.sampler as sampler_mod
from ttt_discover.codex_utils.sampler import PUCTSampler, RANK_VOLATILITY_SURFACES
from ttt_discover.rl import codex_no_finetune
from ttt_discover.rl.codex_no_finetune import CodexNoFinetuneConfig


ROOT = Path(__file__).resolve().parents[1]


class DummyEnv:
    state_type = State

    @staticmethod
    def create_initial_state(problem_type: str) -> State:
        return State(
            timestep=0,
            construction=["init", problem_type],
            code="init",
            value=0.0,
            id="init",
        )


def make_state(
    state_id: str,
    value: float,
    *,
    parents: list[dict] | None = None,
    timestep: int = 0,
) -> State:
    return State(
        timestep=timestep,
        construction=[state_id],
        code=f"code-{state_id}",
        value=value,
        parents=parents or [],
        id=state_id,
    )


def make_sampler(
    tmp_path: Path,
    *,
    enabled: bool = False,
    delta: int = 2,
) -> PUCTSampler:
    return PUCTSampler(
        file_path=str(tmp_path / "puct_sampler.json"),
        env_type=DummyEnv,
        batch_size=0,
        puct_c=0.0,
        topk_children=0,
        rank_volatility_balance=enabled,
        rank_volatility_delta=delta,
    )


def saved_store(tmp_path: Path, step: int) -> dict:
    path = tmp_path / f"puct_sampler_step_{step:06d}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def ids(states: list[State]) -> list[str]:
    return [state.id for state in states]


def test_disabled_parity_no_rank_volatility_observability_or_persistence(
    tmp_path: Path,
) -> None:
    sampler = make_sampler(tmp_path, enabled=False)
    sampler._states = [
        make_state("a", 1.0),
        make_state("b", 3.0),
        make_state("c", 2.0),
    ]

    picked = sampler.sample_states(2)

    assert ids(picked) == ["b", "c"]
    assert sampler._n == {}
    assert sampler._T == 0
    assert sampler._last_puct_stats == [
        (0, 3.0, 0.5, 0.0, 3.0),
        (0, 2.0, 1 / 3, 0.0, 2.0),
    ]
    columns, rows = sampler.get_sample_table()
    assert len(rows) == 2
    assert "rank_volatility_surface" not in columns
    stats = sampler.get_sample_stats()
    assert not any(key.startswith("puct/rank_volatility") for key in stats)

    sampler.flush(step=1)
    store = saved_store(tmp_path, 1)
    assert "puct_rank_volatility_prev_ranks" not in store
    assert "puct_rank_volatility_selected_counts" not in store


def test_first_enabled_call_all_new_and_stores_live_rank_snapshot(
    tmp_path: Path,
) -> None:
    sampler = make_sampler(tmp_path, enabled=True)
    sampler._states = [
        make_state("a", 1.0),
        make_state("b", 3.0),
        make_state("c", 2.0),
    ]

    picked = sampler.sample_states(2)

    assert ids(picked) == ["b", "c"]
    assert sampler._rank_volatility_selected_counts["rank:new"] == 2
    assert sampler._rank_volatility_prev_ranks == {"b": 1, "c": 2, "a": 3}
    columns, rows = sampler.get_sample_table()
    assert columns[-4:] == [
        "puct_rank",
        "rank_volatility_surface",
        "rank_volatility_prev_rank",
        "rank_volatility_delta",
    ]
    assert rows[0][-4:] == (1, "rank:new", None, None)
    assert rows[1][-4:] == (2, "rank:new", None, None)
    stats = sampler.get_sample_stats()
    assert stats["puct/rank_volatility_enabled"] == 1
    assert stats["puct/rank_volatility_delta"] == 2
    assert stats["puct/rank_volatility_selected/rank_new"] == 2


def test_classifier_surfaces_malformed_previous_and_delta_clamping(
    tmp_path: Path,
) -> None:
    sampler = make_sampler(tmp_path, enabled=True, delta=3)
    state = make_state("s", 0.0)

    assert sampler._rank_volatility_surface(state, {"s": 5}, {}) == "rank:new"
    assert (
        sampler._rank_volatility_surface(state, {"s": 5}, {"s": True})
        == "rank:new"
    )
    assert (
        sampler._rank_volatility_surface(state, {"s": 5}, {"s": 9})
        == "rank:up"
    )
    assert (
        sampler._rank_volatility_surface(state, {"s": 5}, {"s": 1})
        == "rank:down"
    )
    assert (
        sampler._rank_volatility_surface(state, {"s": 5}, {"s": 7})
        == "rank:flat"
    )

    sampler.rank_volatility_delta = sampler_mod._sanitize_rank_volatility_delta(-2)
    assert sampler.rank_volatility_delta == 1
    assert (
        sampler._rank_volatility_surface(state, {"s": 5}, {"s": 6})
        == "rank:up"
    )
    assert (
        sampler._rank_volatility_surface(state, {"s": 5}, {"s": 4})
        == "rank:down"
    )

    none_id_state = make_state("placeholder", 0.0)
    none_id_state.id = None
    assert (
        sampler._rank_volatility_surface(none_id_state, {"None": 1}, {"None": 2})
        == "rank:up"
    )


def test_equal_count_tie_uses_earliest_puct_surface(tmp_path: Path) -> None:
    sampler = make_sampler(tmp_path, enabled=True, delta=1)
    sampler._states = [make_state("up", 100.0), make_state("down", 90.0)]
    sampler._rank_volatility_prev_ranks = {
        "up": 2,
        "down": 1,
    }

    picked = sampler.sample_states(1)

    assert RANK_VOLATILITY_SURFACES == (
        "rank:new",
        "rank:up",
        "rank:down",
        "rank:flat",
    )
    assert ids(picked) == ["up"]
    assert sampler._rank_volatility_selected_counts["rank:up"] == 1


def test_unequal_counts_pick_under_selected_surface_over_global_top(
    tmp_path: Path,
) -> None:
    sampler = make_sampler(tmp_path, enabled=True, delta=1)
    sampler._states = [make_state("up", 100.0), make_state("down", 90.0)]
    sampler._rank_volatility_prev_ranks = {
        "up": 2,
        "down": 1,
    }
    sampler._rank_volatility_selected_counts = {
        "rank:new": 0,
        "rank:up": 5,
        "rank:down": 0,
        "rank:flat": 0,
    }

    picked = sampler.sample_states(1)

    assert ids(picked) == ["down"]
    assert sampler._rank_volatility_selected_counts["rank:down"] == 1


def test_within_chosen_surface_highest_puct_wins(tmp_path: Path) -> None:
    sampler = make_sampler(tmp_path, enabled=True, delta=1)
    sampler._states = [
        make_state("up", 100.0),
        make_state("down_high", 90.0),
        make_state("down_low", 80.0),
    ]
    sampler._rank_volatility_prev_ranks = {
        "up": 3,
        "down_high": 1,
        "down_low": 1,
    }
    sampler._rank_volatility_selected_counts = {
        "rank:new": 0,
        "rank:up": 5,
        "rank:down": 0,
        "rank:flat": 0,
    }

    picked = sampler.sample_states(1)

    assert ids(picked) == ["down_high"]
    assert sampler.get_sample_table()[1][0][-4:] == (2, "rank:down", 1, -1)


def test_batch_lineage_blocking_recomputes_availability(tmp_path: Path) -> None:
    sampler = make_sampler(tmp_path, enabled=True, delta=1)
    root = make_state("root", 100.0)
    child = make_state("child", 99.0, parents=[{"id": "root", "timestep": 0}])
    other = make_state("other", 80.0)
    sampler._states = [root, child, other]

    picked = sampler.sample_states(2)

    assert ids(picked) == ["root", "other"]
    assert "child" not in ids(picked)
    assert sampler._rank_volatility_selected_counts["rank:new"] == 2
    assert sampler._rank_volatility_prev_ranks == {
        "root": 1,
        "child": 2,
        "other": 3,
    }


def test_enabled_persistence_resume_and_save_sanitization(tmp_path: Path) -> None:
    sampler = make_sampler(tmp_path, enabled=True, delta=4)
    sampler._states = [make_state("a", 1.0), make_state("c", 2.0)]
    sampler._rank_volatility_prev_ranks = {
        "a": 1,
        "b": 2,
        "c": True,
        7: 3,
    }
    sampler._rank_volatility_selected_counts = {
        "rank:new": True,
        "rank:up": 2,
        "rank:down": -1,
        "rank:flat": 4,
        "rank:bad": 99,
    }
    sampler.flush(step=1)

    store = saved_store(tmp_path, 1)
    assert store["puct_rank_volatility_prev_ranks"] == {"a": 1}
    assert store["puct_rank_volatility_selected_counts"] == {
        "rank:new": 0,
        "rank:up": 2,
        "rank:down": 0,
        "rank:flat": 4,
    }

    resumed = PUCTSampler(
        file_path=str(tmp_path / "puct_sampler.json"),
        env_type=DummyEnv,
        batch_size=0,
        resume_step=1,
        rank_volatility_balance=True,
        rank_volatility_delta=4,
    )
    assert resumed._rank_volatility_prev_ranks == {"a": 1}
    assert resumed._rank_volatility_selected_counts["rank:flat"] == 4

    disabled = PUCTSampler(
        file_path=str(tmp_path / "puct_sampler.json"),
        env_type=DummyEnv,
        batch_size=0,
        resume_step=1,
        rank_volatility_balance=False,
    )
    disabled.flush(step=2)
    disabled_store = saved_store(tmp_path, 2)
    assert "puct_rank_volatility_prev_ranks" not in disabled_store
    assert "puct_rank_volatility_selected_counts" not in disabled_store


def test_record_failed_rollout_only_updates_puct_visits(tmp_path: Path) -> None:
    sampler = make_sampler(tmp_path, enabled=True)
    parent = make_state("parent", 10.0, parents=[{"id": "ancestor", "timestep": 0}])
    sampler._states = [parent]

    assert ids(sampler.sample_states(1)) == ["parent"]
    assert sampler._rank_volatility_selected_counts["rank:new"] == 1
    assert sampler._rank_volatility_prev_ranks == {"parent": 1}

    sampler.record_failed_rollout(parent)

    assert sampler._rank_volatility_selected_counts["rank:new"] == 1
    assert sampler._rank_volatility_prev_ranks == {"parent": 1}
    assert sampler._T == 1
    assert sampler._n["parent"] == 1
    assert sampler._n["ancestor"] == 1


def test_build_sampler_passes_rank_volatility_args(monkeypatch, tmp_path: Path) -> None:
    captured: dict = {}

    def fake_create_sampler(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(codex_no_finetune, "create_sampler", fake_create_sampler)
    cfg = CodexNoFinetuneConfig(
        env_type=DummyEnv,
        log_path=str(tmp_path),
        groups_per_batch=3,
        rank_volatility_balance=True,
        rank_volatility_delta=7,
    )

    assert codex_no_finetune._build_sampler(cfg, start_batch=4) is not None
    assert captured["batch_size"] == 3
    assert captured["resume_step"] == 4
    assert captured["rank_volatility_balance"] is True
    assert captured["rank_volatility_delta"] == 7


def run_dry_run(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "repro.run_discovery", *args],
        cwd=ROOT,
        check=check,
        text=True,
        capture_output=True,
    )


def dry_payload(*args: str) -> dict:
    return json.loads(run_dry_run(*args).stdout)["discover_config"]


def test_run_discovery_dry_run_cli_defaults_enabled_override_and_validation() -> None:
    default = dry_payload("dry-run")
    assert default["codex_rank_volatility_balance"] is False
    assert default["codex_rank_volatility_delta"] == 5
    assert default["codex_model_name"] is None
    assert default["num_cpus_per_task"] == 1
    assert default["wandb_project"] is None

    enabled = dry_payload(
        "--dry-run",
        "--codex-rank-volatility-balance",
        "--codex-rank-volatility-delta",
        "8",
        "--num-cpus-per-task",
        "3",
    )
    assert enabled["codex_rank_volatility_balance"] is True
    assert enabled["codex_rank_volatility_delta"] == 8
    assert enabled["num_cpus_per_task"] == 3

    disabled = dry_payload(
        "dry-run",
        "--codex-rank-volatility-balance",
        "--no-codex-rank-volatility-balance",
        "--codex-rank-volatility-delta",
        "4",
    )
    assert disabled["codex_rank_volatility_balance"] is False
    assert disabled["codex_rank_volatility_delta"] == 4

    invalid = run_dry_run(
        "dry-run",
        "--codex-rank-volatility-delta",
        "not-an-int",
        check=False,
    )
    assert invalid.returncode == 2


def test_gpu_mode_wrapper_dry_run_first_and_override_path() -> None:
    result = subprocess.run(
        [
            "repro/gpu_mode/run_0609_rank_volatility_balance.sh",
            "--dry-run",
            "--no-codex-rank-volatility-balance",
            "--codex-rank-volatility-delta",
            "4",
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)["discover_config"]
    assert payload["experiment_name"] == "trimul_0609_rank_volatility_balance_gpu2"
    assert payload["wandb_project"] is None
    assert payload["codex_model_name"] is None
    assert payload["codex_rank_volatility_balance"] is False
    assert payload["codex_rank_volatility_delta"] == 4


def test_classifier_helpers_only_read_state_id() -> None:
    source = "\n".join(
        textwrap.dedent(inspect.getsource(func))
        for func in (
            sampler_mod.classify_rank_volatility_surface,
            PUCTSampler._rank_volatility_surface,
        )
    )
    tree = ast.parse(source)
    forbidden = [
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "state"
        and node.attr != "id"
    ]
    assert forbidden == []
````

## Diff Stat

### Worktree Tracked Diff Stat

````text
 ttt_discover/codex_utils/discovery.py |   4 +
 ttt_discover/codex_utils/sampler.py   | 300 +++++++++++++++++++++++++++++++++-
 ttt_discover/discovery.py             |   4 +
 ttt_discover/rl/codex_no_finetune.py  |   4 +
 4 files changed, 309 insertions(+), 3 deletions(-)
````
## Raw Diff

<details>
<summary>Tracked worktree patch</summary>

````diff
diff --git a/ttt_discover/codex_utils/discovery.py b/ttt_discover/codex_utils/discovery.py
index b21a80f..b8664a8 100644
--- a/ttt_discover/codex_utils/discovery.py
+++ b/ttt_discover/codex_utils/discovery.py
@@ -50,6 +50,8 @@ class DiscoverConfig:
     codex_initial_program_paths: tuple[str, ...] = ()
     codex_initial_pool_paths: tuple[str, ...] = ()
     codex_autonomous: bool = False
+    codex_rank_volatility_balance: bool = False
+    codex_rank_volatility_delta: int = 5
 
 
 def _run_codex_no_finetune(config: DiscoverConfig) -> None:
@@ -85,6 +87,8 @@ def _run_codex_no_finetune(config: DiscoverConfig) -> None:
         initial_program_paths=config.codex_initial_program_paths,
         initial_pool_paths=config.codex_initial_pool_paths,
         autonomous=config.codex_autonomous,
+        rank_volatility_balance=config.codex_rank_volatility_balance,
+        rank_volatility_delta=config.codex_rank_volatility_delta,
         wandb_project=config.wandb_project,
         wandb_name=experiment_name,
         log_path=log_path,
diff --git a/ttt_discover/codex_utils/sampler.py b/ttt_discover/codex_utils/sampler.py
index 5c5d4d1..35b9301 100644
--- a/ttt_discover/codex_utils/sampler.py
+++ b/ttt_discover/codex_utils/sampler.py
@@ -16,6 +16,77 @@ from ttt_discover.codex_utils.runtime import State, state_from_dict, to_json_ser
 
 logger = logging.getLogger(__name__)
 
+RANK_VOLATILITY_SURFACES = ("rank:new", "rank:up", "rank:down", "rank:flat")
+_RANK_VOLATILITY_PREV_RANKS_KEY = "puct_rank_volatility_prev_ranks"
+_RANK_VOLATILITY_SELECTED_COUNTS_KEY = "puct_rank_volatility_selected_counts"
+_PuctEntry = tuple[float, float, State, int, float, float, float]
+
+
+def _sanitize_rank_volatility_delta(raw_delta: Any) -> int:
+    if isinstance(raw_delta, (bool, np.bool_)):
+        return 1
+    if isinstance(raw_delta, (int, np.integer)) and int(raw_delta) > 0:
+        return int(raw_delta)
+    return 1
+
+
+def _sanitize_rank_volatility_prev_ranks(raw: Any) -> dict[str, int]:
+    if not isinstance(raw, dict):
+        return {}
+    clean: dict[str, int] = {}
+    for state_id, rank in raw.items():
+        if not isinstance(state_id, str):
+            continue
+        if isinstance(rank, (bool, np.bool_)):
+            continue
+        if isinstance(rank, (int, np.integer)) and int(rank) > 0:
+            clean[state_id] = int(rank)
+    return clean
+
+
+def _sanitize_rank_volatility_selected_counts(raw: Any) -> dict[str, int]:
+    raw_counts = raw if isinstance(raw, dict) else {}
+    counts: dict[str, int] = {}
+    for surface in RANK_VOLATILITY_SURFACES:
+        value = raw_counts.get(surface, 0)
+        if isinstance(value, (bool, np.bool_)):
+            counts[surface] = 0
+        elif isinstance(value, (int, np.integer)) and int(value) >= 0:
+            counts[surface] = int(value)
+        else:
+            counts[surface] = 0
+    return counts
+
+
+def _rank_volatility_metric_suffix(surface: str) -> str:
+    return surface.replace(":", "_")
+
+
+def classify_rank_volatility_surface(
+    state: Any,
+    current_ranks: dict[str, int],
+    previous_ranks: dict[str, int],
+    rank_volatility_delta: int,
+) -> str:
+    state_id = str(state.id)
+    cur_rank = current_ranks.get(state_id)
+    prev_rank = previous_ranks.get(state_id)
+    if (
+        isinstance(cur_rank, (bool, np.bool_))
+        or isinstance(prev_rank, (bool, np.bool_))
+        or not isinstance(cur_rank, int)
+        or cur_rank <= 0
+        or not isinstance(prev_rank, int)
+        or prev_rank <= 0
+    ):
+        return "rank:new"
+    delta = _sanitize_rank_volatility_delta(rank_volatility_delta)
+    if prev_rank - cur_rank >= delta:
+        return "rank:up"
+    if cur_rank - prev_rank >= delta:
+        return "rank:down"
+    return "rank:flat"
+
 
 @contextmanager
 def _file_lock(lock_path: str, *, poll_s: float = 0.05, stale_s: float = 600.0):
@@ -353,6 +424,8 @@ class PUCTSampler(StateSampler):
         resume_step: int | None = None,
         puct_c: float = 1.0,
         topk_children: int = 2,
+        rank_volatility_balance: bool = False,
+        rank_volatility_delta: int = 5,
     ):
         self.file_path = file_path
         self.env_type = env_type
@@ -361,6 +434,10 @@ class PUCTSampler(StateSampler):
         self.batch_size = batch_size
         self.topk_children = topk_children
         self.puct_c = float(puct_c)
+        self.rank_volatility_balance = bool(rank_volatility_balance)
+        self.rank_volatility_delta = _sanitize_rank_volatility_delta(
+            rank_volatility_delta
+        )
         
         self._states: list[State] = []
         self._initial_states: list[State] = []
@@ -375,6 +452,11 @@ class PUCTSampler(StateSampler):
         self._T: int = 0
         self._last_scale: float = 1.0
         self._last_puct_stats: list[tuple[int, float, float, float, float]] = []
+        self._rank_volatility_prev_ranks: dict[str, int] = {}
+        self._rank_volatility_selected_counts: dict[str, int] = (
+            _sanitize_rank_volatility_selected_counts({})
+        )
+        self._last_rank_volatility_rows: list[tuple[int, str, int | None, int | None]] = []
         
         if resume_step is not None:
             self._load(resume_step)
@@ -399,6 +481,15 @@ class PUCTSampler(StateSampler):
         self._n = store.get("puct_n", {}) or {}
         self._m = store.get("puct_m", {}) or {}
         self._T = int(store.get("puct_T", 0) or 0)
+        if self.rank_volatility_balance:
+            self._rank_volatility_prev_ranks = _sanitize_rank_volatility_prev_ranks(
+                store.get(_RANK_VOLATILITY_PREV_RANKS_KEY, {})
+            )
+            self._rank_volatility_selected_counts = (
+                _sanitize_rank_volatility_selected_counts(
+                    store.get(_RANK_VOLATILITY_SELECTED_COUNTS_KEY, {})
+                )
+            )
 
     def _save(self, step: int):
         save_path = _sampler_file_for_step(self.file_path, step)
@@ -411,6 +502,12 @@ class PUCTSampler(StateSampler):
             "puct_m": self._m,
             "puct_T": self._T,
         }
+        if self.rank_volatility_balance:
+            self._sanitize_rank_volatility_state()
+            store[_RANK_VOLATILITY_PREV_RANKS_KEY] = self._rank_volatility_prev_ranks
+            store[_RANK_VOLATILITY_SELECTED_COUNTS_KEY] = (
+                self._rank_volatility_selected_counts
+            )
         with _file_lock(f"{save_path}.lock"):
             _atomic_write_json(save_path, store)
 
@@ -489,6 +586,155 @@ class PUCTSampler(StateSampler):
                     queue.append(child_id)
         return lineage
 
+    def _sanitize_rank_volatility_state(self) -> None:
+        if not self.rank_volatility_balance:
+            return
+        live_ids = {str(state.id) for state in self._states}
+        ranks = _sanitize_rank_volatility_prev_ranks(self._rank_volatility_prev_ranks)
+        self._rank_volatility_prev_ranks = {
+            state_id: rank
+            for state_id, rank in ranks.items()
+            if state_id in live_ids
+        }
+        self._rank_volatility_selected_counts = (
+            _sanitize_rank_volatility_selected_counts(
+                self._rank_volatility_selected_counts
+            )
+        )
+
+    def _rank_volatility_current_ranks(
+        self,
+        scores: list[_PuctEntry],
+    ) -> dict[str, int]:
+        return {str(entry[2].id): rank for rank, entry in enumerate(scores, 1)}
+
+    def _rank_volatility_surface(
+        self,
+        state: State,
+        current_ranks: dict[str, int],
+        previous_ranks: dict[str, int],
+    ) -> str:
+        return classify_rank_volatility_surface(
+            state,
+            current_ranks,
+            previous_ranks,
+            self.rank_volatility_delta,
+        )
+
+    def _rank_volatility_row(
+        self,
+        state: State,
+        current_ranks: dict[str, int],
+        previous_ranks: dict[str, int],
+    ) -> tuple[int, str, int | None, int | None]:
+        state_id = str(state.id)
+        current_rank = current_ranks[state_id]
+        prev_rank = previous_ranks.get(state_id)
+        surface = self._rank_volatility_surface(
+            state,
+            current_ranks,
+            previous_ranks,
+        )
+        rank_delta = prev_rank - current_rank if prev_rank is not None else None
+        return (current_rank, surface, prev_rank, rank_delta)
+
+    def _rank_volatility_available_entries(
+        self,
+        entries: list[_PuctEntry],
+        *,
+        picked_ids: set[str],
+        blocked_ids: set[str],
+        use_lineage_blocking: bool,
+    ) -> list[_PuctEntry]:
+        available: list[_PuctEntry] = []
+        for entry in entries:
+            state_id = str(entry[2].id)
+            if state_id in picked_ids:
+                continue
+            if use_lineage_blocking and state_id in blocked_ids:
+                continue
+            available.append(entry)
+        return available
+
+    def _select_rank_volatility_entry(
+        self,
+        available: list[_PuctEntry],
+        planned_counts: dict[str, int],
+        current_ranks: dict[str, int],
+        previous_ranks: dict[str, int],
+    ) -> _PuctEntry | None:
+        selected: _PuctEntry | None = None
+        selected_count: int | None = None
+        for entry in available:
+            surface = self._rank_volatility_surface(
+                entry[2],
+                current_ranks,
+                previous_ranks,
+            )
+            count = planned_counts[surface]
+            if selected_count is None or count < selected_count:
+                selected = entry
+                selected_count = count
+        return selected
+
+    def _sample_states_rank_volatility_balanced(
+        self,
+        scores: list[_PuctEntry],
+        num_states: int,
+    ) -> tuple[list[State], list[_PuctEntry]]:
+        self._sanitize_rank_volatility_state()
+        self._last_rank_volatility_rows = []
+        current_ranks = self._rank_volatility_current_ranks(scores)
+        previous_ranks = dict(self._rank_volatility_prev_ranks)
+        planned_counts = dict(self._rank_volatility_selected_counts)
+        use_lineage_blocking = num_states > 1
+        children_map = self._build_children_map() if use_lineage_blocking else {}
+
+        picked: list[State] = []
+        top_scores: list[_PuctEntry] = []
+        picked_ids: set[str] = set()
+        blocked_ids: set[str] = set()
+
+        while len(picked) < num_states:
+            available = self._rank_volatility_available_entries(
+                scores,
+                picked_ids=picked_ids,
+                blocked_ids=blocked_ids,
+                use_lineage_blocking=use_lineage_blocking,
+            )
+            entry = self._select_rank_volatility_entry(
+                available,
+                planned_counts,
+                current_ranks,
+                previous_ranks,
+            )
+            if entry is None:
+                break
+            state = entry[2]
+            surface = self._rank_volatility_surface(
+                state,
+                current_ranks,
+                previous_ranks,
+            )
+            picked.append(state)
+            top_scores.append(entry)
+            picked_ids.add(str(state.id))
+            self._last_rank_volatility_rows.append(
+                self._rank_volatility_row(state, current_ranks, previous_ranks)
+            )
+            planned_counts[surface] += 1
+            if use_lineage_blocking:
+                blocked_ids.update(
+                    str(state_id)
+                    for state_id in self._get_full_lineage(state, children_map)
+                )
+
+        self._rank_volatility_selected_counts = (
+            _sanitize_rank_volatility_selected_counts(planned_counts)
+        )
+        self._rank_volatility_prev_ranks = current_ranks
+        return picked, top_scores
+
     def sample_states(self, num_states: int) -> list[State]:
         initial_ids = {s.id for s in self._initial_states}
         candidates = list(self._states)
@@ -501,6 +747,9 @@ class PUCTSampler(StateSampler):
             self._last_sampled_states = picked
             self._last_sampled_indices = []
             self._last_puct_stats = [(0, 0.0, 0.0, 0.0, 0.0) for _ in picked]
+            if self.rank_volatility_balance:
+                self._last_rank_volatility_rows = []
+                self._rank_volatility_prev_ranks = {}
             return picked
 
         vals = np.array([float(s.value if s.value is not None else float("-inf")) for s in candidates])
@@ -521,7 +770,12 @@ class PUCTSampler(StateSampler):
 
         scores.sort(key=lambda x: (x[0], x[1]), reverse=True)
 
-        if num_states > 1:
+        if self.rank_volatility_balance:
+            picked, top_scores = self._sample_states_rank_volatility_balanced(
+                scores,
+                num_states,
+            )
+        elif num_states > 1:
             children_map = self._build_children_map()
             picked, top_scores, blocked_ids = [], [], set()
             for entry in scores:
@@ -731,21 +985,53 @@ class PUCTSampler(StateSampler):
         stats.update(_stats(sampled_values, "puct/sampled_value"))
         stats.update(_stats(sampled_timesteps, "puct/sampled_timestep"))
         stats.update(_stats(sampled_constr_lens, "puct/sampled_construction_len"))
+        if self.rank_volatility_balance:
+            self._sanitize_rank_volatility_state()
+            stats["puct/rank_volatility_enabled"] = 1
+            stats["puct/rank_volatility_delta"] = self.rank_volatility_delta
+            for surface in RANK_VOLATILITY_SURFACES:
+                suffix = _rank_volatility_metric_suffix(surface)
+                stats[f"puct/rank_volatility_selected/{suffix}"] = (
+                    self._rank_volatility_selected_counts[surface]
+                )
         return stats
 
     def get_sample_table(self) -> tuple[list[str], list[tuple]]:
         columns = ["buffer_idx", "timestep", "value", "terminal_value", "parent_value", "construction_len", "observation_len", "n", "Q", "P", "bonus", "score"]
+        if self.rank_volatility_balance:
+            columns.extend(
+                [
+                    "puct_rank",
+                    "rank_volatility_surface",
+                    "rank_volatility_prev_rank",
+                    "rank_volatility_delta",
+                ]
+            )
         rows = []
         if not self._last_sampled_states:
             return columns, rows
         indices = self._last_sampled_indices if len(self._last_sampled_indices) == len(self._last_sampled_states) else [-1] * len(self._last_sampled_states)
         stats = self._last_puct_stats if len(self._last_puct_stats) == len(self._last_sampled_states) else [(0, 0.0, 0.0, 0.0, 0.0)] * len(self._last_sampled_states)
-        for idx, state, (n, Q, P, bonus, score) in zip(indices, self._last_sampled_states, stats):
+        rank_rows = (
+            self._last_rank_volatility_rows
+            if self.rank_volatility_balance
+            and len(self._last_rank_volatility_rows) == len(self._last_sampled_states)
+            else [(None, None, None, None)] * len(self._last_sampled_states)
+        )
+        for idx, state, (n, Q, P, bonus, score), rank_row in zip(
+            indices,
+            self._last_sampled_states,
+            stats,
+            rank_rows,
+        ):
             parent_val = state.parent_values[0] if state.parent_values else None
             constr = getattr(state, 'construction', None)
             constr_len = len(constr) if constr is not None else 0
             obs_len = len(state.observation) if state.observation else 0
-            rows.append((idx, state.timestep, state.value, 0, parent_val, constr_len, obs_len, n, Q, P, bonus, score))
+            row = (idx, state.timestep, state.value, 0, parent_val, constr_len, obs_len, n, Q, P, bonus, score)
+            if self.rank_volatility_balance:
+                row = row + rank_row
+            rows.append(row)
         return columns, rows
 
 
@@ -756,6 +1042,8 @@ def create_sampler(
     batch_size: int = 1,
     resume_step: int | None = None,
     topk_children: int = 2,
+    rank_volatility_balance: bool = False,
+    rank_volatility_delta: int = 5,
 ) -> StateSampler:
     """Factory function to create samplers. Pass the env type (from config.env_type)."""
     if not log_path:
@@ -768,6 +1056,8 @@ def create_sampler(
         batch_size=batch_size,
         resume_step=resume_step,
         topk_children=topk_children,
+        rank_volatility_balance=rank_volatility_balance,
+        rank_volatility_delta=rank_volatility_delta,
     )
 
 
@@ -778,6 +1068,8 @@ def get_or_create_sampler_with_default(
     batch_size: int = 1,
     resume_step: int | None = None,
     topk_children: int = 2,
+    rank_volatility_balance: bool = False,
+    rank_volatility_delta: int = 5,
 ) -> StateSampler:
     """Get sampler. Initial experience is created via env_type.create_initial_state."""
     return create_sampler(
@@ -787,4 +1079,6 @@ def get_or_create_sampler_with_default(
         batch_size=batch_size,
         resume_step=resume_step,
         topk_children=topk_children,
+        rank_volatility_balance=rank_volatility_balance,
+        rank_volatility_delta=rank_volatility_delta,
     )
diff --git a/ttt_discover/discovery.py b/ttt_discover/discovery.py
index 6cdb434..21eb45b 100644
--- a/ttt_discover/discovery.py
+++ b/ttt_discover/discovery.py
@@ -63,6 +63,8 @@ class DiscoverConfig:
     codex_initial_program_paths: tuple[str, ...] = ()
     codex_initial_pool_paths: tuple[str, ...] = ()
     codex_autonomous: bool = False
+    codex_rank_volatility_balance: bool = False
+    codex_rank_volatility_delta: int = 5
 
 
 def init_ray(num_cpus_per_task: int, env_type: str):
@@ -146,6 +148,8 @@ async def discover_impl(config: DiscoverConfig):
             initial_program_paths=config.codex_initial_program_paths,
             initial_pool_paths=config.codex_initial_pool_paths,
             autonomous=config.codex_autonomous,
+            rank_volatility_balance=config.codex_rank_volatility_balance,
+            rank_volatility_delta=config.codex_rank_volatility_delta,
             wandb_project=config.wandb_project,
             wandb_name=config.experiment_name,
             log_path=log_path,
diff --git a/ttt_discover/rl/codex_no_finetune.py b/ttt_discover/rl/codex_no_finetune.py
index 18a0e9e..4c99329 100644
--- a/ttt_discover/rl/codex_no_finetune.py
+++ b/ttt_discover/rl/codex_no_finetune.py
@@ -137,6 +137,8 @@ class CodexNoFinetuneConfig:
     initial_program_paths: tuple[str, ...] = ()
     initial_pool_paths: tuple[str, ...] = ()
     topk_children: int = 16
+    rank_volatility_balance: bool = False
+    rank_volatility_delta: int = 5
 
     # False is TTT Discover: non-auto, many short samples. True is AutoEvolve:
     # one Codex deep-dive workspace per sample with file edits and shell access.
@@ -960,6 +962,8 @@ def _build_sampler(cfg: CodexNoFinetuneConfig, start_batch: int) -> StateSampler
         batch_size=cfg.groups_per_batch,
         resume_step=start_batch if start_batch > 0 else None,
         topk_children=cfg.topk_children,
+        rank_volatility_balance=cfg.rank_volatility_balance,
+        rank_volatility_delta=cfg.rank_volatility_delta,
     )
 
 
````
</details>

