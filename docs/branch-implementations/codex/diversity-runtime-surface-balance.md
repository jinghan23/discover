# codex/diversity-runtime-surface-balance

## Summary

在状态 metadata 中记录 birth_eval_wall_s，按运行时分为 unknown/short/medium/long，并平衡不同 runtime surface；支持短/长分位阈值和 timeout。

## Branch State

- Worktree: `/opt/tiger/discover-runtime-surface-balance`
- HEAD: `2234282`
- Base used for comparison: `2234282`
- Commits ahead of base: `0`
- Commits behind base: `0`
- Group: `completed_19`
- Implementation location: `uncommitted worktree diff`

## Implemented Behavior

- The summary above is based on reading the changed Python code, added config fields, and sampler/prompt hooks in this branch.
- For sampler branches, the common pattern is to keep baseline PUCT ordering inside a chosen surface/bucket, while changing which surface/bucket gets the next parent pick.
- Tests detected: `15` test functions across untracked test files.

## Added Markers

### Config fields

- `codex_runtime_surface_balance`
- `codex_runtime_surface_short_frac`
- `codex_runtime_surface_long_frac`
- `clean`
- `runtime_surface_balance`
- `runtime_surface_short_frac`
- `runtime_surface_long_frac`
- `runtime_surface_timeout_s`
- `birth_eval_wall_s`

### Constants

- `RUNTIME_SURFACES`
- `_RUNTIME_SURFACE_BIRTH_EVAL_WALL_S_KEY`
- `_RUNTIME_SURFACE_SAMPLE_COUNTS_KEY`

### Classes

- None

### Functions

- `_zero_runtime_surface_counts`
- `_valid_runtime_wall_s`
- `_sanitize_runtime_surface_birth_eval_wall_s`
- `_sanitize_runtime_surface_sample_counts`
- `_clamp_runtime_surface_fractions`
- `_coerce`
- `classify_runtime_surface`
- `update_states`
- `_runtime_surface`
- `_available_puct_entries`
- `_runtime_surface_available_counts`
- `_select_runtime_surface_entry`
- `_record_runtime_surface_selection`
- `_sample_states_runtime_surface_balanced`

## Diff Summary

- Worktree tracked shortstat: `4 files changed, 397 insertions(+), 8 deletions(-)`
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

- `repro/gpu_mode/run_0609_runtime_surface_balance.sh (656 bytes)`
- `repro/run_discovery.py (6542 bytes)`
- `tests/test_codex_runtime_surface_balance.py (19760 bytes)`

### Detected Test Functions

- `tests/test_codex_runtime_surface_balance.py::test_disabled_parity_no_runtime_observability_or_persistence`
- `tests/test_codex_runtime_surface_balance.py::test_classifier_surfaces_boundaries_invalids_and_fraction_clamping`
- `tests/test_codex_runtime_surface_balance.py::test_least_count_surface_wins_and_preserves_puct_within_surface`
- `tests/test_codex_runtime_surface_balance.py::test_surface_tie_uses_earliest_puct_surface`
- `tests/test_codex_runtime_surface_balance.py::test_batch_lineage_blocking_recomputes_available_surfaces`
- `tests/test_codex_runtime_surface_balance.py::test_enabled_persistence_resume_and_save_sanitization`
- `tests/test_codex_runtime_surface_balance.py::test_missing_bad_and_disabled_resume_persistence_edges`
- `tests/test_codex_runtime_surface_balance.py::test_update_states_records_metadata_only_when_enabled`
- `tests/test_codex_runtime_surface_balance.py::test_record_failed_rollout_only_updates_puct_visits`
- `tests/test_codex_runtime_surface_balance.py::test_update_sampler_from_results_passes_birth_eval_wall_metadata`
- `tests/test_codex_runtime_surface_balance.py::test_run_candidate_measures_safe_grade_wall_time`
- `tests/test_codex_runtime_surface_balance.py::test_build_sampler_passes_runtime_surface_args`
- `tests/test_codex_runtime_surface_balance.py::test_run_discovery_dry_run_cli_defaults_enabled_override_and_validation`
- `tests/test_codex_runtime_surface_balance.py::test_gpu_mode_wrapper_dry_run_first_and_override_path`
- `tests/test_codex_runtime_surface_balance.py::test_classifier_helpers_only_read_state_id`

## Untracked File Snapshots

These files are not part of `git diff` yet, but they are part of the worktree implementation state recorded here.

### `repro/gpu_mode/run_0609_runtime_surface_balance.sh`

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
  --experiment-name trimul_0609_runtime_surface_balance_gpu2 \
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
  --codex-runtime-surface-balance \
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
    parser.add_argument("--experiment-name", default="gpu-mode-0609-runtime-surface-balance")
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
        "--codex-runtime-surface-balance",
        dest="codex_runtime_surface_balance",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--no-codex-runtime-surface-balance",
        dest="codex_runtime_surface_balance",
        action="store_false",
    )
    parser.add_argument("--codex-runtime-surface-short-frac", type=float, default=0.25)
    parser.add_argument("--codex-runtime-surface-long-frac", type=float, default=0.75)
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
        "codex_runtime_surface_balance": args.codex_runtime_surface_balance,
        "codex_runtime_surface_short_frac": args.codex_runtime_surface_short_frac,
        "codex_runtime_surface_long_frac": args.codex_runtime_surface_long_frac,
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
        print(json.dumps({"mode": "dry-run", "discover_config": payload}, indent=2, sort_keys=True))
        return

    from ttt_discover import DiscoverConfig, discover

    payload["env_type"] = _resolve_env_type(args.task)
    discover(DiscoverConfig(**payload))


if __name__ == "__main__":
    main()
````

### `tests/test_codex_runtime_surface_balance.py`

````python
from __future__ import annotations

import ast
import asyncio
import inspect
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from ttt_discover.codex_utils.runtime import State
import ttt_discover.codex_utils.sampler as sampler_mod
from ttt_discover.codex_utils.sampler import PUCTSampler, RUNTIME_SURFACES
from ttt_discover.rl import codex_no_finetune
from ttt_discover.rl.codex_no_finetune import (
    CandidateResult,
    CodexNoFinetuneConfig,
    VerifyResult,
)


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


class CandidateEnv(DummyEnv):
    def get_question(self) -> str:
        return "question"

    def check_format(self, parsed_code: str) -> bool:
        return bool(parsed_code)

    def _create_next_state(
        self,
        step_idx: int,
        parsed_code: str,
        outs: VerifyResult,
    ) -> State:
        return make_state("child", outs.raw_score, timestep=step_idx)


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
    short_frac: float = 0.25,
    long_frac: float = 0.75,
    timeout_s: float | None = 100.0,
) -> PUCTSampler:
    return PUCTSampler(
        file_path=str(tmp_path / "puct_sampler.json"),
        env_type=DummyEnv,
        batch_size=0,
        puct_c=0.0,
        topk_children=0,
        runtime_surface_balance=enabled,
        runtime_surface_short_frac=short_frac,
        runtime_surface_long_frac=long_frac,
        runtime_surface_timeout_s=timeout_s,
    )


def saved_store(tmp_path: Path, step: int) -> dict:
    path = tmp_path / f"puct_sampler_step_{step:06d}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def ids(states: list[State]) -> list[str]:
    return [state.id for state in states]


def test_disabled_parity_no_runtime_observability_or_persistence(tmp_path: Path) -> None:
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
    assert "runtime_surface" not in columns
    assert "birth_eval_wall_s" not in columns
    stats = sampler.get_sample_stats()
    assert not any(key.startswith("puct/runtime_surface") for key in stats)

    sampler.update_states(
        [make_state("child", 4.0)],
        [sampler._states[0]],
        save=False,
        state_metadata=[{"birth_eval_wall_s": 2.0}],
    )
    sampler.flush(step=1)
    store = saved_store(tmp_path, 1)
    assert "runtime_surface_birth_eval_wall_s" not in store
    assert "runtime_surface_sample_counts" not in store


def test_classifier_surfaces_boundaries_invalids_and_fraction_clamping(tmp_path: Path) -> None:
    sampler = make_sampler(tmp_path, enabled=True, timeout_s=100.0)
    state = make_state("s", 0.0)

    assert sampler._runtime_surface(state) == "runtime:unknown"
    for invalid in (True, -1.0, float("nan"), float("inf"), "9"):
        sampler._runtime_surface_birth_eval_wall_s = {"s": invalid}
        assert sampler._runtime_surface(state) == "runtime:unknown"

    sampler._runtime_surface_birth_eval_wall_s = {"s": 24.999}
    assert sampler._runtime_surface(state) == "runtime:short"
    sampler._runtime_surface_birth_eval_wall_s = {"s": 25.0}
    assert sampler._runtime_surface(state) == "runtime:medium"
    sampler._runtime_surface_birth_eval_wall_s = {"s": 74.999}
    assert sampler._runtime_surface(state) == "runtime:medium"
    sampler._runtime_surface_birth_eval_wall_s = {"s": 75.0}
    assert sampler._runtime_surface(state) == "runtime:long"

    sampler.runtime_surface_timeout_s = 0
    assert sampler._runtime_surface(state) == "runtime:unknown"

    sampler.runtime_surface_timeout_s = 100
    sampler.runtime_surface_short_frac = -2
    sampler.runtime_surface_long_frac = 2
    sampler._runtime_surface_birth_eval_wall_s = {"s": 0.1}
    assert sampler._runtime_surface(state) == "runtime:medium"

    sampler.runtime_surface_short_frac = 0.8
    sampler.runtime_surface_long_frac = 0.2
    sampler._runtime_surface_birth_eval_wall_s = {"s": 79.9}
    assert sampler._runtime_surface(state) == "runtime:short"
    sampler._runtime_surface_birth_eval_wall_s = {"s": 80.0}
    assert sampler._runtime_surface(state) == "runtime:long"

    none_id_state = make_state("placeholder", 0.0)
    none_id_state.id = None
    sampler.runtime_surface_short_frac = 0.25
    sampler.runtime_surface_long_frac = 0.75
    sampler._runtime_surface_birth_eval_wall_s = {"None": 10.0}
    assert sampler._runtime_surface(none_id_state) == "runtime:short"


def test_least_count_surface_wins_and_preserves_puct_within_surface(tmp_path: Path) -> None:
    sampler = make_sampler(tmp_path, enabled=True)
    sampler._states = [
        make_state("short", 100.0),
        make_state("long_high", 90.0),
        make_state("long_low", 80.0),
    ]
    sampler._runtime_surface_birth_eval_wall_s = {
        "short": 10.0,
        "long_high": 90.0,
        "long_low": 95.0,
    }
    sampler._runtime_surface_sample_counts = {
        "runtime:unknown": 0,
        "runtime:short": 2,
        "runtime:medium": 0,
        "runtime:long": 0,
    }

    picked = sampler.sample_states(1)

    assert ids(picked) == ["long_high"]
    assert sampler._runtime_surface_sample_counts["runtime:long"] == 1
    columns, rows = sampler.get_sample_table()
    assert columns[-3:] == [
        "runtime_surface",
        "birth_eval_wall_s",
        "runtime_surface_sample_count",
    ]
    assert rows[0][-3:] == ("runtime:long", 90.0, 0)


def test_surface_tie_uses_earliest_puct_surface(tmp_path: Path) -> None:
    sampler = make_sampler(tmp_path, enabled=True)
    sampler._states = [
        make_state("short_high", 100.0),
        make_state("long", 90.0),
        make_state("short_low", 80.0),
    ]
    sampler._runtime_surface_birth_eval_wall_s = {
        "short_high": 10.0,
        "long": 90.0,
        "short_low": 12.0,
    }

    picked = sampler.sample_states(1)

    assert RUNTIME_SURFACES == (
        "runtime:unknown",
        "runtime:short",
        "runtime:medium",
        "runtime:long",
    )
    assert ids(picked) == ["short_high"]
    assert sampler._runtime_surface_sample_counts["runtime:short"] == 1


def test_batch_lineage_blocking_recomputes_available_surfaces(tmp_path: Path) -> None:
    sampler = make_sampler(tmp_path, enabled=True)
    root = make_state("root", 100.0)
    child = make_state("child", 99.0, parents=[{"id": "root", "timestep": 0}])
    long = make_state("long", 80.0)
    sampler._states = [root, child, long]
    sampler._runtime_surface_birth_eval_wall_s = {"long": 90.0}

    picked = sampler.sample_states(2)

    assert ids(picked) == ["root", "long"]
    assert "child" not in ids(picked)
    assert sampler._runtime_surface_sample_counts["runtime:unknown"] == 1
    assert sampler._runtime_surface_sample_counts["runtime:long"] == 1
    assert sampler._last_runtime_surface_selected["runtime:unknown"] == 1
    assert sampler._last_runtime_surface_selected["runtime:long"] == 1


def test_enabled_persistence_resume_and_save_sanitization(tmp_path: Path) -> None:
    sampler = make_sampler(tmp_path, enabled=True)
    sampler._states = [make_state("a", 1.0)]
    sampler._runtime_surface_birth_eval_wall_s = {"a": 1.5}
    sampler._runtime_surface_sample_counts = {
        "runtime:unknown": 1,
        "runtime:short": 2,
        "runtime:medium": 3,
        "runtime:long": 4,
    }
    sampler.flush(step=1)

    store = saved_store(tmp_path, 1)
    assert store["runtime_surface_birth_eval_wall_s"] == {"a": 1.5}
    assert store["runtime_surface_sample_counts"] == {
        "runtime:unknown": 1,
        "runtime:short": 2,
        "runtime:medium": 3,
        "runtime:long": 4,
    }

    resumed = PUCTSampler(
        file_path=str(tmp_path / "puct_sampler.json"),
        env_type=DummyEnv,
        batch_size=0,
        resume_step=1,
        runtime_surface_balance=True,
        runtime_surface_timeout_s=100.0,
    )
    assert resumed._runtime_surface_birth_eval_wall_s == {"a": 1.5}
    assert resumed._runtime_surface_sample_counts["runtime:long"] == 4

    resumed._runtime_surface_birth_eval_wall_s = {
        "a": True,
        "b": -1.0,
        "c": 2.5,
        7: 3.5,
    }
    resumed._runtime_surface_sample_counts = {
        "runtime:unknown": True,
        "runtime:short": -1,
        "runtime:medium": 4,
        "unknown": 99,
    }
    resumed._states = [make_state("c", 1.0)]
    resumed.flush(step=2)
    sanitized = saved_store(tmp_path, 2)
    assert sanitized["runtime_surface_birth_eval_wall_s"] == {"c": 2.5}
    assert sanitized["runtime_surface_sample_counts"] == {
        "runtime:unknown": 0,
        "runtime:short": 0,
        "runtime:medium": 4,
        "runtime:long": 0,
    }


def test_missing_bad_and_disabled_resume_persistence_edges(tmp_path: Path) -> None:
    state = make_state("a", 1.0).to_dict()
    missing_path = tmp_path / "missing" / "puct_sampler_step_000005.json"
    missing_path.parent.mkdir(parents=True)
    missing_path.write_text(
        json.dumps({"step": 5, "states": [state], "initial_states": []}),
        encoding="utf-8",
    )
    missing = PUCTSampler(
        file_path=str(tmp_path / "missing" / "puct_sampler.json"),
        env_type=DummyEnv,
        batch_size=0,
        resume_step=5,
        runtime_surface_balance=True,
        runtime_surface_timeout_s=100.0,
    )
    assert missing._runtime_surface_birth_eval_wall_s == {}
    assert missing._runtime_surface_sample_counts == {
        surface: 0 for surface in RUNTIME_SURFACES
    }

    bad_path = tmp_path / "bad" / "puct_sampler_step_000006.json"
    bad_path.parent.mkdir(parents=True)
    bad_path.write_text(
        json.dumps(
            {
                "step": 6,
                "states": [state],
                "initial_states": [],
                "runtime_surface_birth_eval_wall_s": {
                    "a": True,
                    "b": -1.0,
                    "c": 2.0,
                },
                "runtime_surface_sample_counts": {
                    "runtime:unknown": False,
                    "runtime:long": 5,
                    "runtime:short": -1,
                    "unknown": 3,
                },
            }
        ),
        encoding="utf-8",
    )
    bad = PUCTSampler(
        file_path=str(tmp_path / "bad" / "puct_sampler.json"),
        env_type=DummyEnv,
        batch_size=0,
        resume_step=6,
        runtime_surface_balance=True,
        runtime_surface_timeout_s=100.0,
    )
    assert bad._runtime_surface_birth_eval_wall_s == {"c": 2.0}
    assert bad._runtime_surface_sample_counts == {
        "runtime:unknown": 0,
        "runtime:short": 0,
        "runtime:medium": 0,
        "runtime:long": 5,
    }

    disabled = PUCTSampler(
        file_path=str(tmp_path / "bad" / "puct_sampler.json"),
        env_type=DummyEnv,
        batch_size=0,
        resume_step=6,
        runtime_surface_balance=False,
    )
    disabled.flush(step=7)
    disabled_store = json.loads(
        (tmp_path / "bad" / "puct_sampler_step_000007.json").read_text(encoding="utf-8")
    )
    assert "runtime_surface_birth_eval_wall_s" not in disabled_store
    assert "runtime_surface_sample_counts" not in disabled_store


def test_update_states_records_metadata_only_when_enabled(tmp_path: Path) -> None:
    parent = make_state("parent", 1.0)
    child = make_state("child", 2.0)

    enabled = make_sampler(tmp_path / "enabled", enabled=True)
    enabled._states = [parent]
    enabled.update_states(
        [child],
        [parent],
        save=False,
        state_metadata=[{"birth_eval_wall_s": 2.5}],
    )
    assert enabled._runtime_surface_birth_eval_wall_s == {"child": 2.5}

    disabled = make_sampler(tmp_path / "disabled", enabled=False)
    disabled._states = [parent]
    disabled.update_states(
        [make_state("child2", 2.0)],
        [parent],
        save=False,
        state_metadata=[{"birth_eval_wall_s": 2.5}],
    )
    assert disabled._runtime_surface_birth_eval_wall_s == {}


def test_record_failed_rollout_only_updates_puct_visits(tmp_path: Path) -> None:
    sampler = make_sampler(tmp_path, enabled=True)
    parent = make_state("parent", 10.0, parents=[{"id": "ancestor", "timestep": 0}])
    sampler._states = [parent]

    assert ids(sampler.sample_states(1)) == ["parent"]
    assert sampler._runtime_surface_sample_counts["runtime:unknown"] == 1

    sampler.record_failed_rollout(parent)

    assert sampler._runtime_surface_sample_counts["runtime:unknown"] == 1
    assert sampler._T == 1
    assert sampler._n["parent"] == 1
    assert sampler._n["ancestor"] == 1


def test_update_sampler_from_results_passes_birth_eval_wall_metadata() -> None:
    class FakeSampler:
        def __init__(self) -> None:
            self.metadata = None

        def has_state(self, state: State) -> bool:
            return False

        def update_states(self, states, parent_states, save=True, step=None, state_metadata=None):
            self.metadata = state_metadata

    result = CandidateResult(
        parent_state=make_state("parent", 1.0),
        group_idx=0,
        sample_idx=0,
        prompt="p",
        response="r",
        parsed_code="c",
        reward=1.0,
        correctness=1.0,
        raw_score=1.0,
        msg="",
        metrics={},
        birth_eval_wall_s=1.25,
        next_state=make_state("child", 2.0),
    )
    fake = FakeSampler()

    codex_no_finetune._update_sampler_from_results(fake, [result])

    assert fake.metadata == [{"birth_eval_wall_s": 1.25}]
    assert result.pool_status == "sampler_filtered"


def test_run_candidate_measures_safe_grade_wall_time(monkeypatch, tmp_path: Path) -> None:
    async def fake_complete(prompt: str) -> str:
        return "```python\npass\n```"

    async def fake_safe_grade(cfg, env, parsed_code, correct_format):
        return VerifyResult(
            reward=1.0,
            msg="ok",
            correctness=1.0,
            raw_score=2.0,
            result_construction=None,
            stdout="",
        )

    class FakeTime:
        def __init__(self) -> None:
            self.ticks = iter([100.0, 103.5])

        def monotonic(self) -> float:
            return next(self.ticks)

    monkeypatch.setattr(codex_no_finetune, "_make_completer", lambda *args, **kwargs: fake_complete)
    monkeypatch.setattr(codex_no_finetune, "_safe_grade", fake_safe_grade)
    monkeypatch.setattr(codex_no_finetune, "time", FakeTime())
    cfg = CodexNoFinetuneConfig(env_type=CandidateEnv, log_path=str(tmp_path))

    result = asyncio.run(
        codex_no_finetune._run_candidate(
            cfg,
            make_sampler(tmp_path, enabled=True),
            make_state("parent", 1.0),
            group_idx=0,
            sample_idx=0,
            step_idx=7,
            semaphore=None,
        )
    )

    assert result.next_state is not None
    assert result.birth_eval_wall_s == pytest.approx(3.5)


def test_build_sampler_passes_runtime_surface_args(monkeypatch, tmp_path: Path) -> None:
    captured: dict = {}

    def fake_create_sampler(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(codex_no_finetune, "create_sampler", fake_create_sampler)
    cfg = CodexNoFinetuneConfig(
        env_type=DummyEnv,
        log_path=str(tmp_path),
        groups_per_batch=3,
        eval_timeout=77,
        runtime_surface_balance=True,
        runtime_surface_short_frac=0.2,
        runtime_surface_long_frac=0.9,
    )

    assert codex_no_finetune._build_sampler(cfg, start_batch=4) is not None
    assert captured["batch_size"] == 3
    assert captured["resume_step"] == 4
    assert captured["runtime_surface_balance"] is True
    assert captured["runtime_surface_short_frac"] == 0.2
    assert captured["runtime_surface_long_frac"] == 0.9
    assert captured["runtime_surface_timeout_s"] == 77


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
    assert default["codex_runtime_surface_balance"] is False
    assert default["codex_model_name"] is None
    assert default["codex_runtime_surface_short_frac"] == 0.25
    assert default["codex_runtime_surface_long_frac"] == 0.75
    assert default["num_cpus_per_task"] == 1
    assert default["wandb_project"] is None

    enabled = dry_payload(
        "--dry-run",
        "--codex-runtime-surface-balance",
        "--num-cpus-per-task",
        "3",
    )
    assert enabled["codex_runtime_surface_balance"] is True
    assert enabled["num_cpus_per_task"] == 3

    disabled = dry_payload(
        "dry-run",
        "--codex-runtime-surface-balance",
        "--no-codex-runtime-surface-balance",
        "--codex-runtime-surface-short-frac",
        "0.2",
        "--codex-runtime-surface-long-frac",
        "0.9",
    )
    assert disabled["codex_runtime_surface_balance"] is False
    assert disabled["codex_runtime_surface_short_frac"] == 0.2
    assert disabled["codex_runtime_surface_long_frac"] == 0.9

    invalid = run_dry_run(
        "dry-run",
        "--codex-runtime-surface-short-frac",
        "not-a-float",
        check=False,
    )
    assert invalid.returncode == 2


def test_gpu_mode_wrapper_dry_run_first_and_override_path() -> None:
    result = subprocess.run(
        [
            "repro/gpu_mode/run_0609_runtime_surface_balance.sh",
            "--dry-run",
            "--no-codex-runtime-surface-balance",
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)["discover_config"]
    assert payload["experiment_name"] == "trimul_0609_runtime_surface_balance_gpu2"
    assert payload["wandb_project"] is None
    assert payload["codex_model_name"] is None
    assert payload["codex_runtime_surface_balance"] is False


def test_classifier_helpers_only_read_state_id() -> None:
    source = "\n".join(
        textwrap.dedent(inspect.getsource(func))
        for func in (
            sampler_mod.classify_runtime_surface,
            PUCTSampler._runtime_surface,
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
 ttt_discover/codex_utils/discovery.py |   6 +
 ttt_discover/codex_utils/sampler.py   | 375 +++++++++++++++++++++++++++++++++-
 ttt_discover/discovery.py             |   6 +
 ttt_discover/rl/codex_no_finetune.py  |  18 +-
 4 files changed, 397 insertions(+), 8 deletions(-)
````
## Raw Diff

<details>
<summary>Tracked worktree patch</summary>

````diff
diff --git a/ttt_discover/codex_utils/discovery.py b/ttt_discover/codex_utils/discovery.py
index b21a80f..d08e53f 100644
--- a/ttt_discover/codex_utils/discovery.py
+++ b/ttt_discover/codex_utils/discovery.py
@@ -50,6 +50,9 @@ class DiscoverConfig:
     codex_initial_program_paths: tuple[str, ...] = ()
     codex_initial_pool_paths: tuple[str, ...] = ()
     codex_autonomous: bool = False
+    codex_runtime_surface_balance: bool = False
+    codex_runtime_surface_short_frac: float = 0.25
+    codex_runtime_surface_long_frac: float = 0.75
 
 
 def _run_codex_no_finetune(config: DiscoverConfig) -> None:
@@ -85,6 +88,9 @@ def _run_codex_no_finetune(config: DiscoverConfig) -> None:
         initial_program_paths=config.codex_initial_program_paths,
         initial_pool_paths=config.codex_initial_pool_paths,
         autonomous=config.codex_autonomous,
+        runtime_surface_balance=config.codex_runtime_surface_balance,
+        runtime_surface_short_frac=config.codex_runtime_surface_short_frac,
+        runtime_surface_long_frac=config.codex_runtime_surface_long_frac,
         wandb_project=config.wandb_project,
         wandb_name=experiment_name,
         log_path=log_path,
diff --git a/ttt_discover/codex_utils/sampler.py b/ttt_discover/codex_utils/sampler.py
index 5c5d4d1..45a3b00 100644
--- a/ttt_discover/codex_utils/sampler.py
+++ b/ttt_discover/codex_utils/sampler.py
@@ -16,6 +16,99 @@ from ttt_discover.codex_utils.runtime import State, state_from_dict, to_json_ser
 
 logger = logging.getLogger(__name__)
 
+RUNTIME_SURFACES = ("runtime:unknown", "runtime:short", "runtime:medium", "runtime:long")
+_RUNTIME_SURFACE_BIRTH_EVAL_WALL_S_KEY = "runtime_surface_birth_eval_wall_s"
+_RUNTIME_SURFACE_SAMPLE_COUNTS_KEY = "runtime_surface_sample_counts"
+
+
+def _zero_runtime_surface_counts() -> dict[str, int]:
+    return {surface: 0 for surface in RUNTIME_SURFACES}
+
+
+def _valid_runtime_wall_s(value: Any) -> float | None:
+    if isinstance(value, (bool, np.bool_)):
+        return None
+    if not isinstance(value, (int, float, np.integer, np.floating)):
+        return None
+    wall_s = float(value)
+    if not np.isfinite(wall_s) or wall_s < 0.0:
+        return None
+    return wall_s
+
+
+def _sanitize_runtime_surface_birth_eval_wall_s(raw: Any) -> dict[str, float]:
+    if not isinstance(raw, dict):
+        return {}
+    clean: dict[str, float] = {}
+    for state_id, wall_s in raw.items():
+        if not isinstance(state_id, str):
+            continue
+        valid_wall_s = _valid_runtime_wall_s(wall_s)
+        if valid_wall_s is not None:
+            clean[state_id] = valid_wall_s
+    return clean
+
+
+def _sanitize_runtime_surface_sample_counts(raw: Any) -> dict[str, int]:
+    counts = _zero_runtime_surface_counts()
+    if not isinstance(raw, dict):
+        return counts
+    for surface in RUNTIME_SURFACES:
+        value = raw.get(surface, 0)
+        if isinstance(value, (bool, np.bool_)):
+            continue
+        if isinstance(value, (int, np.integer)) and int(value) >= 0:
+            counts[surface] = int(value)
+    return counts
+
+
+def _clamp_runtime_surface_fractions(
+    short_frac: float,
+    long_frac: float,
+) -> tuple[float, float]:
+    def _coerce(value: Any, default: float) -> float:
+        if isinstance(value, (bool, np.bool_)):
+            return default
+        try:
+            out = float(value)
+        except (TypeError, ValueError):
+            return default
+        if not np.isfinite(out):
+            return default
+        return min(max(out, 0.0), 1.0)
+
+    short = _coerce(short_frac, 0.25)
+    long = _coerce(long_frac, 0.75)
+    if long < short:
+        long = short
+    return short, long
+
+
+def classify_runtime_surface(
+    state: Any,
+    birth_eval_wall_s: dict[str, float],
+    runtime_surface_timeout_s: float | None,
+    runtime_surface_short_frac: float,
+    runtime_surface_long_frac: float,
+) -> str:
+    state_id = str(state.id)
+    wall_s = _valid_runtime_wall_s(birth_eval_wall_s.get(state_id))
+    timeout_s = _valid_runtime_wall_s(runtime_surface_timeout_s)
+    if wall_s is None or timeout_s is None or timeout_s <= 0.0:
+        return "runtime:unknown"
+
+    short_frac, long_frac = _clamp_runtime_surface_fractions(
+        runtime_surface_short_frac,
+        runtime_surface_long_frac,
+    )
+    short_threshold = short_frac * timeout_s
+    long_threshold = long_frac * timeout_s
+    if wall_s < short_threshold:
+        return "runtime:short"
+    if wall_s < long_threshold:
+        return "runtime:medium"
+    return "runtime:long"
+
 
 @contextmanager
 def _file_lock(lock_path: str, *, poll_s: float = 0.05, stale_s: float = 600.0):
@@ -75,7 +168,14 @@ class StateSampler(ABC):
         pass
 
     @abstractmethod
-    def update_states(self, states: list[State], parent_states: list[State], save: bool = True, step: int | None = None):
+    def update_states(
+        self,
+        states: list[State],
+        parent_states: list[State],
+        save: bool = True,
+        step: int | None = None,
+        state_metadata: list[dict[str, Any]] | None = None,
+    ):
         """Update internal storage with new states. Sets parent info automatically."""
         pass
 
@@ -353,6 +453,10 @@ class PUCTSampler(StateSampler):
         resume_step: int | None = None,
         puct_c: float = 1.0,
         topk_children: int = 2,
+        runtime_surface_balance: bool = False,
+        runtime_surface_short_frac: float = 0.25,
+        runtime_surface_long_frac: float = 0.75,
+        runtime_surface_timeout_s: float | None = None,
     ):
         self.file_path = file_path
         self.env_type = env_type
@@ -361,6 +465,10 @@ class PUCTSampler(StateSampler):
         self.batch_size = batch_size
         self.topk_children = topk_children
         self.puct_c = float(puct_c)
+        self.runtime_surface_balance = bool(runtime_surface_balance)
+        self.runtime_surface_short_frac = runtime_surface_short_frac
+        self.runtime_surface_long_frac = runtime_surface_long_frac
+        self.runtime_surface_timeout_s = runtime_surface_timeout_s
         
         self._states: list[State] = []
         self._initial_states: list[State] = []
@@ -375,6 +483,12 @@ class PUCTSampler(StateSampler):
         self._T: int = 0
         self._last_scale: float = 1.0
         self._last_puct_stats: list[tuple[int, float, float, float, float]] = []
+        self._runtime_surface_birth_eval_wall_s: dict[str, float] = {}
+        self._runtime_surface_sample_counts: dict[str, int] = _zero_runtime_surface_counts()
+        self._last_runtime_surface_rows: list[tuple[str, float | None, int]] = []
+        self._last_runtime_surface_selected: dict[str, int] = _zero_runtime_surface_counts()
+        self._last_runtime_surface_available: dict[str, int] = _zero_runtime_surface_counts()
+        self._last_runtime_surface_last_selected: str | None = None
         
         if resume_step is not None:
             self._load(resume_step)
@@ -399,6 +513,17 @@ class PUCTSampler(StateSampler):
         self._n = store.get("puct_n", {}) or {}
         self._m = store.get("puct_m", {}) or {}
         self._T = int(store.get("puct_T", 0) or 0)
+        if self.runtime_surface_balance:
+            self._runtime_surface_birth_eval_wall_s = (
+                _sanitize_runtime_surface_birth_eval_wall_s(
+                    store.get(_RUNTIME_SURFACE_BIRTH_EVAL_WALL_S_KEY, {})
+                )
+            )
+            self._runtime_surface_sample_counts = (
+                _sanitize_runtime_surface_sample_counts(
+                    store.get(_RUNTIME_SURFACE_SAMPLE_COUNTS_KEY, {})
+                )
+            )
 
     def _save(self, step: int):
         save_path = _sampler_file_for_step(self.file_path, step)
@@ -411,6 +536,17 @@ class PUCTSampler(StateSampler):
             "puct_m": self._m,
             "puct_T": self._T,
         }
+        if self.runtime_surface_balance:
+            store[_RUNTIME_SURFACE_BIRTH_EVAL_WALL_S_KEY] = (
+                _sanitize_runtime_surface_birth_eval_wall_s(
+                    self._runtime_surface_birth_eval_wall_s
+                )
+            )
+            store[_RUNTIME_SURFACE_SAMPLE_COUNTS_KEY] = (
+                _sanitize_runtime_surface_sample_counts(
+                    self._runtime_surface_sample_counts
+                )
+            )
         with _file_lock(f"{save_path}.lock"):
             _atomic_write_json(save_path, store)
 
@@ -489,6 +625,114 @@ class PUCTSampler(StateSampler):
                     queue.append(child_id)
         return lineage
 
+    def _runtime_surface(self, state: State) -> str:
+        return classify_runtime_surface(
+            state,
+            self._runtime_surface_birth_eval_wall_s,
+            self.runtime_surface_timeout_s,
+            self.runtime_surface_short_frac,
+            self.runtime_surface_long_frac,
+        )
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
+    def _runtime_surface_available_counts(
+        self,
+        entries: list[tuple[float, float, State, int, float, float, float]],
+    ) -> dict[str, int]:
+        counts = _zero_runtime_surface_counts()
+        for entry in entries:
+            counts[self._runtime_surface(entry[2])] += 1
+        return counts
+
+    def _select_runtime_surface_entry(
+        self,
+        available: list[tuple[float, float, State, int, float, float, float]],
+    ) -> tuple[float, float, State, int, float, float, float] | None:
+        best_entry: tuple[float, float, State, int, float, float, float] | None = None
+        best_count: int | None = None
+        for entry in available:
+            surface = self._runtime_surface(entry[2])
+            count = self._runtime_surface_sample_counts[surface]
+            if best_count is None or count < best_count:
+                best_count = count
+                best_entry = entry
+        return best_entry
+
+    def _record_runtime_surface_selection(
+        self,
+        entry: tuple[float, float, State, int, float, float, float],
+    ) -> None:
+        state = entry[2]
+        state_id = str(state.id)
+        surface = self._runtime_surface(state)
+        before = self._runtime_surface_sample_counts[surface]
+        self._runtime_surface_sample_counts[surface] = before + 1
+        self._last_runtime_surface_selected[surface] += 1
+        self._last_runtime_surface_last_selected = surface
+        self._last_runtime_surface_rows.append(
+            (surface, self._runtime_surface_birth_eval_wall_s.get(state_id), before)
+        )
+
+    def _sample_states_runtime_surface_balanced(
+        self,
+        scores: list[tuple[float, float, State, int, float, float, float]],
+        num_states: int,
+    ) -> tuple[
+        list[State],
+        list[tuple[float, float, State, int, float, float, float]],
+    ]:
+        self._last_runtime_surface_rows = []
+        self._last_runtime_surface_selected = _zero_runtime_surface_counts()
+        self._last_runtime_surface_available = _zero_runtime_surface_counts()
+        self._last_runtime_surface_last_selected = None
+
+        use_lineage_blocking = num_states > 1
+        children_map = self._build_children_map() if use_lineage_blocking else {}
+        picked: list[State] = []
+        top_scores: list[tuple[float, float, State, int, float, float, float]] = []
+        picked_ids: set[str] = set()
+        blocked_ids: set[str] = set()
+
+        while len(picked) < num_states:
+            available = self._available_puct_entries(
+                scores,
+                picked_ids=picked_ids,
+                blocked_ids=blocked_ids,
+                use_lineage_blocking=use_lineage_blocking,
+            )
+            self._last_runtime_surface_available = (
+                self._runtime_surface_available_counts(available)
+            )
+            entry = self._select_runtime_surface_entry(available)
+            if entry is None:
+                break
+            state = entry[2]
+            picked.append(state)
+            top_scores.append(entry)
+            picked_ids.add(state.id)
+            self._record_runtime_surface_selection(entry)
+            if use_lineage_blocking:
+                blocked_ids.update(self._get_full_lineage(state, children_map))
+
+        return picked, top_scores
+
     def sample_states(self, num_states: int) -> list[State]:
         initial_ids = {s.id for s in self._initial_states}
         candidates = list(self._states)
@@ -501,6 +745,11 @@ class PUCTSampler(StateSampler):
             self._last_sampled_states = picked
             self._last_sampled_indices = []
             self._last_puct_stats = [(0, 0.0, 0.0, 0.0, 0.0) for _ in picked]
+            if self.runtime_surface_balance:
+                self._last_runtime_surface_rows = []
+                self._last_runtime_surface_selected = _zero_runtime_surface_counts()
+                self._last_runtime_surface_available = _zero_runtime_surface_counts()
+                self._last_runtime_surface_last_selected = None
             return picked
 
         vals = np.array([float(s.value if s.value is not None else float("-inf")) for s in candidates])
@@ -521,7 +770,12 @@ class PUCTSampler(StateSampler):
 
         scores.sort(key=lambda x: (x[0], x[1]), reverse=True)
 
-        if num_states > 1:
+        if self.runtime_surface_balance:
+            picked, top_scores = self._sample_states_runtime_surface_balanced(
+                scores,
+                num_states,
+            )
+        elif num_states > 1:
             children_map = self._build_children_map()
             picked, top_scores, blocked_ids = [], [], set()
             for entry in scores:
@@ -548,10 +802,20 @@ class PUCTSampler(StateSampler):
 
         return picked
 
-    def update_states(self, states: list[State], parent_states: list[State], save: bool = True, step: int | None = None):
+    def update_states(
+        self,
+        states: list[State],
+        parent_states: list[State],
+        save: bool = True,
+        step: int | None = None,
+        state_metadata: list[dict[str, Any]] | None = None,
+    ):
         if not states:
             return
         assert len(states) == len(parent_states)
+        if state_metadata is not None:
+            assert len(state_metadata) == len(states)
+        metadata = state_metadata or [{} for _ in states]
 
         # Update PUCT stats for ALL states
         parent_max: dict[str, float] = {}
@@ -575,12 +839,32 @@ class PUCTSampler(StateSampler):
             return
 
         # Apply topk filter and dedup
-        states, parent_states = self._filter_topk_per_parent(states, parent_states, self.topk_children)
+        if self.topk_children > 0:
+            parent_to_children: dict[str, list[tuple[State, State, dict[str, Any]]]] = {}
+            for child, parent, item_metadata in zip(states, parent_states, metadata):
+                pid = parent.id
+                if pid not in parent_to_children:
+                    parent_to_children[pid] = []
+                parent_to_children[pid].append((child, parent, item_metadata))
+            filtered: list[tuple[State, State, dict[str, Any]]] = []
+            for children_and_parents in parent_to_children.values():
+                sorted_items = sorted(
+                    children_and_parents,
+                    key=lambda x: (
+                        x[0].value if x[0].value is not None else float("-inf")
+                    ),
+                    reverse=True,
+                )
+                filtered.extend(sorted_items[: self.topk_children])
+            states = [item[0] for item in filtered]
+            parent_states = [item[1] for item in filtered]
+            metadata = [item[2] for item in filtered]
         existing = {self._get_construction_key(s) for s in self._states}
         existing.discard(None)
         
         new_states = []
-        for child, parent in zip(states, parent_states):
+        new_metadata: list[dict[str, Any]] = []
+        for child, parent, item_metadata in zip(states, parent_states, metadata):
             if child.value is None:
                 continue
             limits = getattr(self.env_type, "construction_length_limits", None)
@@ -596,6 +880,7 @@ class PUCTSampler(StateSampler):
                 continue
             self._set_parent_info(child, parent)
             new_states.append(child)
+            new_metadata.append(item_metadata)
             if key is not None:
                 existing.add(key)
 
@@ -603,6 +888,13 @@ class PUCTSampler(StateSampler):
             return
         with self._lock:
             self._states.extend(new_states)
+            if self.runtime_surface_balance:
+                for child, item_metadata in zip(new_states, new_metadata):
+                    wall_s = _valid_runtime_wall_s(
+                        item_metadata.get("birth_eval_wall_s")
+                    )
+                    if wall_s is not None:
+                        self._runtime_surface_birth_eval_wall_s[str(child.id)] = wall_s
             if save:
                 self._finalize_and_save(step)
 
@@ -618,6 +910,20 @@ class PUCTSampler(StateSampler):
                     break
                 keep.add(i)
             self._states = [self._states[i] for i in sorted(keep)]
+        if self.runtime_surface_balance:
+            live_ids = {str(s.id) for s in self._states}
+            self._runtime_surface_birth_eval_wall_s = {
+                state_id: wall_s
+                for state_id, wall_s in _sanitize_runtime_surface_birth_eval_wall_s(
+                    self._runtime_surface_birth_eval_wall_s
+                ).items()
+                if state_id in live_ids
+            }
+            self._runtime_surface_sample_counts = (
+                _sanitize_runtime_surface_sample_counts(
+                    self._runtime_surface_sample_counts
+                )
+            )
         if step is not None:
             self._current_step = step
         self._save(self._current_step)
@@ -731,21 +1037,60 @@ class PUCTSampler(StateSampler):
         stats.update(_stats(sampled_values, "puct/sampled_value"))
         stats.update(_stats(sampled_timesteps, "puct/sampled_timestep"))
         stats.update(_stats(sampled_constr_lens, "puct/sampled_construction_len"))
+        if self.runtime_surface_balance:
+            stats["puct/runtime_surface_balance_enabled"] = 1
+            for surface in RUNTIME_SURFACES:
+                stats[f"puct/runtime_surface_selected_count/{surface}"] = (
+                    self._last_runtime_surface_selected[surface]
+                )
+                stats[f"puct/runtime_surface_available_count/{surface}"] = (
+                    self._last_runtime_surface_available[surface]
+                )
+                stats[f"puct/runtime_surface_last_selected/{surface}"] = int(
+                    self._last_runtime_surface_last_selected == surface
+                )
         return stats
 
     def get_sample_table(self) -> tuple[list[str], list[tuple]]:
         columns = ["buffer_idx", "timestep", "value", "terminal_value", "parent_value", "construction_len", "observation_len", "n", "Q", "P", "bonus", "score"]
+        if self.runtime_surface_balance:
+            columns.extend(
+                [
+                    "runtime_surface",
+                    "birth_eval_wall_s",
+                    "runtime_surface_sample_count",
+                ]
+            )
         rows = []
         if not self._last_sampled_states:
             return columns, rows
         indices = self._last_sampled_indices if len(self._last_sampled_indices) == len(self._last_sampled_states) else [-1] * len(self._last_sampled_states)
         stats = self._last_puct_stats if len(self._last_puct_stats) == len(self._last_sampled_states) else [(0, 0.0, 0.0, 0.0, 0.0)] * len(self._last_sampled_states)
-        for idx, state, (n, Q, P, bonus, score) in zip(indices, self._last_sampled_states, stats):
+        if self.runtime_surface_balance:
+            runtime_rows = (
+                self._last_runtime_surface_rows
+                if len(self._last_runtime_surface_rows) == len(self._last_sampled_states)
+                else [
+                    (self._runtime_surface(state), None, 0)
+                    for state in self._last_sampled_states
+                ]
+            )
+        else:
+            runtime_rows = [()] * len(self._last_sampled_states)
+        for idx, state, (n, Q, P, bonus, score), runtime_row in zip(
+            indices,
+            self._last_sampled_states,
+            stats,
+            runtime_rows,
+        ):
             parent_val = state.parent_values[0] if state.parent_values else None
             constr = getattr(state, 'construction', None)
             constr_len = len(constr) if constr is not None else 0
             obs_len = len(state.observation) if state.observation else 0
-            rows.append((idx, state.timestep, state.value, 0, parent_val, constr_len, obs_len, n, Q, P, bonus, score))
+            row = (idx, state.timestep, state.value, 0, parent_val, constr_len, obs_len, n, Q, P, bonus, score)
+            if self.runtime_surface_balance:
+                row = row + runtime_row
+            rows.append(row)
         return columns, rows
 
 
@@ -756,6 +1101,10 @@ def create_sampler(
     batch_size: int = 1,
     resume_step: int | None = None,
     topk_children: int = 2,
+    runtime_surface_balance: bool = False,
+    runtime_surface_short_frac: float = 0.25,
+    runtime_surface_long_frac: float = 0.75,
+    runtime_surface_timeout_s: float | None = None,
 ) -> StateSampler:
     """Factory function to create samplers. Pass the env type (from config.env_type)."""
     if not log_path:
@@ -768,6 +1117,10 @@ def create_sampler(
         batch_size=batch_size,
         resume_step=resume_step,
         topk_children=topk_children,
+        runtime_surface_balance=runtime_surface_balance,
+        runtime_surface_short_frac=runtime_surface_short_frac,
+        runtime_surface_long_frac=runtime_surface_long_frac,
+        runtime_surface_timeout_s=runtime_surface_timeout_s,
     )
 
 
@@ -778,6 +1131,10 @@ def get_or_create_sampler_with_default(
     batch_size: int = 1,
     resume_step: int | None = None,
     topk_children: int = 2,
+    runtime_surface_balance: bool = False,
+    runtime_surface_short_frac: float = 0.25,
+    runtime_surface_long_frac: float = 0.75,
+    runtime_surface_timeout_s: float | None = None,
 ) -> StateSampler:
     """Get sampler. Initial experience is created via env_type.create_initial_state."""
     return create_sampler(
@@ -787,4 +1144,8 @@ def get_or_create_sampler_with_default(
         batch_size=batch_size,
         resume_step=resume_step,
         topk_children=topk_children,
+        runtime_surface_balance=runtime_surface_balance,
+        runtime_surface_short_frac=runtime_surface_short_frac,
+        runtime_surface_long_frac=runtime_surface_long_frac,
+        runtime_surface_timeout_s=runtime_surface_timeout_s,
     )
diff --git a/ttt_discover/discovery.py b/ttt_discover/discovery.py
index 6cdb434..b187cfb 100644
--- a/ttt_discover/discovery.py
+++ b/ttt_discover/discovery.py
@@ -63,6 +63,9 @@ class DiscoverConfig:
     codex_initial_program_paths: tuple[str, ...] = ()
     codex_initial_pool_paths: tuple[str, ...] = ()
     codex_autonomous: bool = False
+    codex_runtime_surface_balance: bool = False
+    codex_runtime_surface_short_frac: float = 0.25
+    codex_runtime_surface_long_frac: float = 0.75
 
 
 def init_ray(num_cpus_per_task: int, env_type: str):
@@ -146,6 +149,9 @@ async def discover_impl(config: DiscoverConfig):
             initial_program_paths=config.codex_initial_program_paths,
             initial_pool_paths=config.codex_initial_pool_paths,
             autonomous=config.codex_autonomous,
+            runtime_surface_balance=config.codex_runtime_surface_balance,
+            runtime_surface_short_frac=config.codex_runtime_surface_short_frac,
+            runtime_surface_long_frac=config.codex_runtime_surface_long_frac,
             wandb_project=config.wandb_project,
             wandb_name=config.experiment_name,
             log_path=log_path,
diff --git a/ttt_discover/rl/codex_no_finetune.py b/ttt_discover/rl/codex_no_finetune.py
index 18a0e9e..1d45dd3 100644
--- a/ttt_discover/rl/codex_no_finetune.py
+++ b/ttt_discover/rl/codex_no_finetune.py
@@ -106,6 +106,7 @@ class CandidateResult:
     raw_score: float | None
     msg: str
     metrics: dict[str, Any]
+    birth_eval_wall_s: float | None = None
     next_state: Any | None = None
     error: str | None = None
     kept: bool = True
@@ -137,6 +138,9 @@ class CodexNoFinetuneConfig:
     initial_program_paths: tuple[str, ...] = ()
     initial_pool_paths: tuple[str, ...] = ()
     topk_children: int = 16
+    runtime_surface_balance: bool = False
+    runtime_surface_short_frac: float = 0.25
+    runtime_surface_long_frac: float = 0.75
 
     # False is TTT Discover: non-auto, many short samples. True is AutoEvolve:
     # one Codex deep-dive workspace per sample with file edits and shell access.
@@ -673,7 +677,9 @@ async def _run_candidate(
                 parsed_code = autonomous_submission
                 parsed_code_source = "workspace_submission.py"
         correct_format = _check_candidate_format(env, parsed_code)
+        grade_start = time.monotonic()
         outs = await _safe_grade(cfg, env, parsed_code, correct_format)
+        birth_eval_wall_s = time.monotonic() - grade_start
         metrics = _build_metrics(env, outs, response, parsed_code, correct_format)
         metrics["codex/parsed_code_source"] = parsed_code_source
         if autonomous_submission_path is not None:
@@ -691,6 +697,7 @@ async def _run_candidate(
             raw_score=outs.raw_score,
             msg=outs.msg,
             metrics=metrics,
+            birth_eval_wall_s=birth_eval_wall_s,
             next_state=next_state,
         )
     except Exception as exc:
@@ -730,7 +737,12 @@ def _update_sampler_from_results(sampler: StateSampler, results: list[CandidateR
                 except Exception as exc:
                     logger.warning("Failed to check sampler state before update: %s", exc)
             try:
-                sampler.update_states([result.next_state], [result.parent_state], save=False)
+                sampler.update_states(
+                    [result.next_state],
+                    [result.parent_state],
+                    save=False,
+                    state_metadata=[{"birth_eval_wall_s": result.birth_eval_wall_s}],
+                )
                 if was_present:
                     result.pool_status = "duplicate"
                 elif has_state is not None:
@@ -960,6 +972,10 @@ def _build_sampler(cfg: CodexNoFinetuneConfig, start_batch: int) -> StateSampler
         batch_size=cfg.groups_per_batch,
         resume_step=start_batch if start_batch > 0 else None,
         topk_children=cfg.topk_children,
+        runtime_surface_balance=cfg.runtime_surface_balance,
+        runtime_surface_short_frac=cfg.runtime_surface_short_frac,
+        runtime_surface_long_frac=cfg.runtime_surface_long_frac,
+        runtime_surface_timeout_s=cfg.eval_timeout,
     )
 
 
````
</details>

