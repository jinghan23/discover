# codex/diversity-parent-selection-lag-balance

## Summary

记录候选上次被选择的 timestep，分类 never_selected/stale/ready/cooldown，优先选择历史上采样较少或滞后的父节点。

## Branch State

- Worktree: `/opt/tiger/discover-parent-selection-lag-balance`
- HEAD: `2234282`
- Base used for comparison: `2234282`
- Commits ahead of base: `0`
- Commits behind base: `0`
- Group: `completed_19`
- Implementation location: `uncommitted worktree diff`

## Implemented Behavior

- The summary above is based on reading the changed Python code, added config fields, and sampler/prompt hooks in this branch.
- For sampler branches, the common pattern is to keep baseline PUCT ordering inside a chosen surface/bucket, while changing which surface/bucket gets the next parent pick.
- Tests detected: `14` test functions across untracked test files.

## Added Markers

### Config fields

- `codex_puct_selection_lag_balance`
- `codex_puct_selection_lag_cooldown_rounds`
- `codex_puct_selection_lag_stale_rounds`
- `out`
- `puct_selection_lag_balance`
- `puct_selection_lag_cooldown_rounds`
- `puct_selection_lag_stale_rounds`

### Constants

- `SELECTION_LAG_SURFACES`

### Classes

- None

### Functions

- `_zero_selection_lag_counts`
- `_is_nonnegative_int`
- `_sanitize_nonnegative_int`
- `_clamp_config_rounds`
- `_sanitize_selection_lag_last_clock`
- `_sanitize_selection_lag_surface_counts`
- `_selection_lag_state_id`
- `_sanitize_selection_lag_runtime`
- `_selection_lag_thresholds`
- `_classify_selection_lag`
- `_sample_states_with_selection_lag`

## Diff Summary

- Worktree tracked shortstat: `4 files changed, 270 insertions(+), 3 deletions(-)`
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

- `repro/gpu_mode/run_0609_selection_lag_balance.sh (654 bytes)`
- `repro/run_discovery.py (6590 bytes)`
- `tests/test_codex_selection_lag_balance.py (17354 bytes)`

### Detected Test Functions

- `tests/test_codex_selection_lag_balance.py::test_disabled_parity_no_selection_lag_observability_or_persistence`
- `tests/test_codex_selection_lag_balance.py::test_classifier_surfaces_invalid_values_and_threshold_clamping`
- `tests/test_codex_selection_lag_balance.py::test_least_count_available_non_cooldown_surface_wins`
- `tests/test_codex_selection_lag_balance.py::test_unavailable_lower_count_surfaces_are_ignored`
- `tests/test_codex_selection_lag_balance.py::test_cooldown_skip_and_all_cooldown_fallback`
- `tests/test_codex_selection_lag_balance.py::test_surface_order_tie_break_and_within_surface_puct_order`
- `tests/test_codex_selection_lag_balance.py::test_batch_lineage_blocking_recomputes_availability_and_updates_clock`
- `tests/test_codex_selection_lag_balance.py::test_enabled_persistence_resume_and_save_sanitization`
- `tests/test_codex_selection_lag_balance.py::test_missing_bad_and_disabled_resume_persistence_edges`
- `tests/test_codex_selection_lag_balance.py::test_record_failed_rollout_only_updates_puct_visits`
- `tests/test_codex_selection_lag_balance.py::test_build_sampler_passes_selection_lag_args`
- `tests/test_codex_selection_lag_balance.py::test_run_discovery_dry_run_cli_defaults_enabled_and_override`
- `tests/test_codex_selection_lag_balance.py::test_gpu_mode_wrapper_dry_run_and_override_path`
- `tests/test_codex_selection_lag_balance.py::test_classifier_helpers_only_read_state_id`

## Untracked File Snapshots

These files are not part of `git diff` yet, but they are part of the worktree implementation state recorded here.

### `repro/gpu_mode/run_0609_selection_lag_balance.sh`

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
  --experiment-name gpu-mode-0609-selection-lag-balance \
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
  --codex-puct-selection-lag-balance \
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
    parser.add_argument("--experiment-name", default="gpu-mode-0609-selection-lag-balance")
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
        "--codex-puct-selection-lag-balance",
        dest="codex_puct_selection_lag_balance",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--no-codex-puct-selection-lag-balance",
        dest="codex_puct_selection_lag_balance",
        action="store_false",
    )
    parser.add_argument("--codex-puct-selection-lag-cooldown-rounds", type=int, default=1)
    parser.add_argument("--codex-puct-selection-lag-stale-rounds", type=int, default=8)
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
        "codex_puct_selection_lag_balance": args.codex_puct_selection_lag_balance,
        "codex_puct_selection_lag_cooldown_rounds": args.codex_puct_selection_lag_cooldown_rounds,
        "codex_puct_selection_lag_stale_rounds": args.codex_puct_selection_lag_stale_rounds,
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

### `tests/test_codex_selection_lag_balance.py`

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
from ttt_discover.codex_utils.sampler import PUCTSampler, SELECTION_LAG_SURFACES
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
    cooldown: int = 1,
    stale: int = 8,
) -> PUCTSampler:
    return PUCTSampler(
        file_path=str(tmp_path / "puct_sampler.json"),
        env_type=DummyEnv,
        batch_size=0,
        puct_c=0.0,
        topk_children=0,
        puct_selection_lag_balance=enabled,
        puct_selection_lag_cooldown_rounds=cooldown,
        puct_selection_lag_stale_rounds=stale,
    )


def saved_sampler_path(tmp_path: Path, step: int) -> Path:
    return tmp_path / f"puct_sampler_step_{step:06d}.json"


def saved_store(tmp_path: Path, step: int) -> dict:
    return json.loads(saved_sampler_path(tmp_path, step).read_text(encoding="utf-8"))


def ids(states: list[State]) -> list[str]:
    return [state.id for state in states]


def test_disabled_parity_no_selection_lag_observability_or_persistence(tmp_path: Path) -> None:
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
    assert "selection_lag_surface" not in columns
    assert "selection_lag_clock" not in columns
    stats = sampler.get_sample_stats()
    assert not any(key.startswith("puct/selection_lag_balance/") for key in stats)

    sampler.flush(step=1)
    store = saved_store(tmp_path, 1)
    assert "puct_selection_lag_clock" not in store
    assert "puct_selection_lag_last_clock" not in store
    assert "puct_selection_lag_surface_counts" not in store


def test_classifier_surfaces_invalid_values_and_threshold_clamping(tmp_path: Path) -> None:
    sampler = make_sampler(tmp_path, enabled=True, cooldown=1, stale=8)
    state = make_state("s", 0.0)
    sampler._puct_selection_lag_clock = 10

    assert sampler._classify_selection_lag(state) == ("never_selected", None, None)
    for invalid in (True, "9", -1):
        sampler._puct_selection_lag_last_clock = {"s": invalid}
        assert sampler._classify_selection_lag(state) == ("never_selected", None, None)

    sampler._puct_selection_lag_last_clock = {"s": 9}
    assert sampler._classify_selection_lag(state) == ("cooldown", 9, 1)
    sampler._puct_selection_lag_last_clock = {"s": 8}
    assert sampler._classify_selection_lag(state) == ("ready", 8, 2)
    sampler._puct_selection_lag_last_clock = {"s": 1}
    assert sampler._classify_selection_lag(state) == ("stale", 1, 9)

    none_id_state = make_state("placeholder", 0.0)
    none_id_state.id = None
    sampler._puct_selection_lag_last_clock = {"None": 10}
    assert sampler._classify_selection_lag(none_id_state) == ("cooldown", 10, 0)

    sampler.puct_selection_lag_cooldown_rounds = -2
    sampler.puct_selection_lag_stale_rounds = -1
    sampler._puct_selection_lag_clock = 4
    sampler._puct_selection_lag_last_clock = {"s": 4}
    assert sampler._classify_selection_lag(state) == ("cooldown", 4, 0)
    sampler._puct_selection_lag_last_clock = {"s": 3}
    assert sampler._classify_selection_lag(state) == ("stale", 3, 1)

    sampler.puct_selection_lag_cooldown_rounds = 5
    sampler.puct_selection_lag_stale_rounds = 2
    sampler._puct_selection_lag_clock = 10
    sampler._puct_selection_lag_last_clock = {"s": 5}
    assert sampler._classify_selection_lag(state) == ("cooldown", 5, 5)
    sampler._puct_selection_lag_last_clock = {"s": 4}
    assert sampler._classify_selection_lag(state) == ("stale", 4, 6)


def test_least_count_available_non_cooldown_surface_wins(tmp_path: Path) -> None:
    sampler = make_sampler(tmp_path, enabled=True)
    sampler._states = [make_state("ready", 100.0), make_state("stale", 90.0)]
    sampler._puct_selection_lag_clock = 10
    sampler._puct_selection_lag_last_clock = {"ready": 8, "stale": 0}
    sampler._puct_selection_lag_surface_counts = {
        "never_selected": 0,
        "stale": 0,
        "ready": 5,
        "cooldown": 0,
    }

    picked = sampler.sample_states(1)

    assert ids(picked) == ["stale"]
    assert sampler._puct_selection_lag_surface_counts["stale"] == 1
    assert sampler._puct_selection_lag_last_clock["stale"] == 10
    assert sampler._puct_selection_lag_clock == 11


def test_unavailable_lower_count_surfaces_are_ignored(tmp_path: Path) -> None:
    sampler = make_sampler(tmp_path, enabled=True)
    sampler._states = [make_state("a", 100.0), make_state("b", 90.0)]
    sampler._puct_selection_lag_clock = 10
    sampler._puct_selection_lag_last_clock = {"a": 8, "b": 7}
    sampler._puct_selection_lag_surface_counts = {
        "never_selected": 0,
        "stale": 0,
        "ready": 9,
        "cooldown": 0,
    }

    picked = sampler.sample_states(1)

    assert ids(picked) == ["a"]
    assert sampler._puct_selection_lag_surface_counts["ready"] == 10


def test_cooldown_skip_and_all_cooldown_fallback(tmp_path: Path) -> None:
    sampler = make_sampler(tmp_path / "skip", enabled=True)
    sampler._states = [make_state("cool", 100.0), make_state("ready", 90.0)]
    sampler._puct_selection_lag_clock = 10
    sampler._puct_selection_lag_last_clock = {"cool": 10, "ready": 8}

    picked = sampler.sample_states(1)

    assert ids(picked) == ["ready"]
    assert sampler._puct_selection_lag_surface_counts["ready"] == 1
    assert sampler._puct_selection_lag_surface_counts["cooldown"] == 0

    fallback = make_sampler(tmp_path / "fallback", enabled=True)
    fallback._states = [make_state("cool_high", 100.0), make_state("cool_low", 90.0)]
    fallback._puct_selection_lag_clock = 10
    fallback._puct_selection_lag_last_clock = {"cool_high": 10, "cool_low": 9}

    picked = fallback.sample_states(1)

    assert ids(picked) == ["cool_high"]
    assert fallback._puct_selection_lag_surface_counts["cooldown"] == 1
    assert fallback._last_selection_lag_stats == [("cooldown", 10, 10, 0)]


def test_surface_order_tie_break_and_within_surface_puct_order(tmp_path: Path) -> None:
    sampler = make_sampler(tmp_path, enabled=True)
    sampler._states = [
        make_state("never_low", 70.0),
        make_state("never_high", 90.0),
        make_state("stale_highest", 100.0),
    ]
    sampler._puct_selection_lag_clock = 10
    sampler._puct_selection_lag_last_clock = {"stale_highest": 0}

    picked = sampler.sample_states(1)

    assert SELECTION_LAG_SURFACES == ("never_selected", "stale", "ready", "cooldown")
    assert ids(picked) == ["never_high"]
    assert sampler._puct_selection_lag_surface_counts["never_selected"] == 1


def test_batch_lineage_blocking_recomputes_availability_and_updates_clock(tmp_path: Path) -> None:
    sampler = make_sampler(tmp_path, enabled=True)
    root = make_state("root", 100.0)
    child = make_state("child", 99.0, parents=[{"id": "root", "timestep": 0}])
    never2 = make_state("never2", 90.0)
    stale = make_state("stale", 80.0)
    sampler._states = [root, child, never2, stale]
    sampler._puct_selection_lag_clock = 10
    sampler._puct_selection_lag_last_clock = {"stale": 0}

    picked = sampler.sample_states(2)

    assert ids(picked) == ["root", "stale"]
    assert "child" not in ids(picked)
    assert sampler._puct_selection_lag_clock == 12
    assert sampler._puct_selection_lag_surface_counts["never_selected"] == 1
    assert sampler._puct_selection_lag_surface_counts["stale"] == 1
    assert sampler._last_selection_lag_stats == [
        ("never_selected", None, 10, None),
        ("stale", 0, 11, 11),
    ]


def test_enabled_persistence_resume_and_save_sanitization(tmp_path: Path) -> None:
    sampler = make_sampler(tmp_path, enabled=True)
    sampler._states = [make_state("a", 1.0)]
    sampler._puct_selection_lag_clock = 3
    sampler._puct_selection_lag_last_clock = {"a": 1}
    sampler._puct_selection_lag_surface_counts = {
        "never_selected": 1,
        "stale": 2,
        "ready": 3,
        "cooldown": 4,
    }
    sampler.flush(step=1)

    store = saved_store(tmp_path, 1)
    assert store["puct_selection_lag_clock"] == 3
    assert store["puct_selection_lag_last_clock"] == {"a": 1}
    assert store["puct_selection_lag_surface_counts"] == {
        "cooldown": 4,
        "never_selected": 1,
        "ready": 3,
        "stale": 2,
    }

    resumed = PUCTSampler(
        file_path=str(tmp_path / "puct_sampler.json"),
        env_type=DummyEnv,
        batch_size=0,
        resume_step=1,
        puct_selection_lag_balance=True,
    )
    assert resumed._puct_selection_lag_clock == 3
    assert resumed._puct_selection_lag_last_clock == {"a": 1}
    assert resumed._puct_selection_lag_surface_counts["cooldown"] == 4

    resumed._puct_selection_lag_clock = -9
    resumed._puct_selection_lag_last_clock = {"a": True, "b": -1, "c": 2}
    resumed._puct_selection_lag_surface_counts = {
        "never_selected": True,
        "stale": -1,
        "ready": 4,
        "unknown": 99,
    }
    resumed.flush(step=2)
    sanitized = saved_store(tmp_path, 2)
    assert sanitized["puct_selection_lag_clock"] == 0
    assert sanitized["puct_selection_lag_last_clock"] == {"c": 2}
    assert sanitized["puct_selection_lag_surface_counts"] == {
        "cooldown": 0,
        "never_selected": 0,
        "ready": 4,
        "stale": 0,
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
        puct_selection_lag_balance=True,
    )
    assert missing._puct_selection_lag_clock == 0
    assert missing._puct_selection_lag_last_clock == {}
    assert missing._puct_selection_lag_surface_counts == {
        surface: 0 for surface in SELECTION_LAG_SURFACES
    }

    bad_path = tmp_path / "bad" / "puct_sampler_step_000006.json"
    bad_path.parent.mkdir(parents=True)
    bad_path.write_text(
        json.dumps(
            {
                "step": 6,
                "states": [state],
                "initial_states": [],
                "puct_selection_lag_clock": True,
                "puct_selection_lag_last_clock": {"a": True, "b": -1, "c": 2},
                "puct_selection_lag_surface_counts": {
                    "never_selected": False,
                    "cooldown": 5,
                    "ready": -1,
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
        puct_selection_lag_balance=True,
    )
    assert bad._puct_selection_lag_clock == 0
    assert bad._puct_selection_lag_last_clock == {"c": 2}
    assert bad._puct_selection_lag_surface_counts == {
        "never_selected": 0,
        "stale": 0,
        "ready": 0,
        "cooldown": 5,
    }

    disabled = PUCTSampler(
        file_path=str(tmp_path / "bad" / "puct_sampler.json"),
        env_type=DummyEnv,
        batch_size=0,
        resume_step=6,
        puct_selection_lag_balance=False,
    )
    disabled.flush(step=7)
    disabled_store = json.loads(
        (tmp_path / "bad" / "puct_sampler_step_000007.json").read_text(encoding="utf-8")
    )
    assert "puct_selection_lag_clock" not in disabled_store
    assert "puct_selection_lag_last_clock" not in disabled_store
    assert "puct_selection_lag_surface_counts" not in disabled_store


def test_record_failed_rollout_only_updates_puct_visits(tmp_path: Path) -> None:
    sampler = make_sampler(tmp_path, enabled=True)
    parent = make_state("parent", 10.0, parents=[{"id": "ancestor", "timestep": 0}])
    sampler._states = [parent]

    assert ids(sampler.sample_states(1)) == ["parent"]
    assert sampler._puct_selection_lag_clock == 1
    assert sampler._puct_selection_lag_surface_counts["never_selected"] == 1

    sampler.record_failed_rollout(parent)

    assert sampler._puct_selection_lag_clock == 1
    assert sampler._puct_selection_lag_surface_counts["never_selected"] == 1
    assert sampler._T == 1
    assert sampler._n["parent"] == 1
    assert sampler._n["ancestor"] == 1


def test_build_sampler_passes_selection_lag_args(monkeypatch, tmp_path: Path) -> None:
    captured: dict = {}

    def fake_create_sampler(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(codex_no_finetune, "create_sampler", fake_create_sampler)
    cfg = CodexNoFinetuneConfig(
        env_type=DummyEnv,
        log_path=str(tmp_path),
        groups_per_batch=3,
        puct_selection_lag_balance=True,
        puct_selection_lag_cooldown_rounds=2,
        puct_selection_lag_stale_rounds=5,
    )

    assert codex_no_finetune._build_sampler(cfg, start_batch=4) is not None
    assert captured["batch_size"] == 3
    assert captured["resume_step"] == 4
    assert captured["puct_selection_lag_balance"] is True
    assert captured["puct_selection_lag_cooldown_rounds"] == 2
    assert captured["puct_selection_lag_stale_rounds"] == 5


def run_dry_run(*args: str) -> dict:
    result = subprocess.run(
        [sys.executable, "-m", "repro.run_discovery", *args],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return json.loads(result.stdout)


def test_run_discovery_dry_run_cli_defaults_enabled_and_override() -> None:
    default = run_dry_run("dry-run")
    payload = default["discover_config"]
    assert payload["codex_puct_selection_lag_balance"] is False
    assert payload["codex_model_name"] is None
    assert payload["codex_puct_selection_lag_cooldown_rounds"] == 1
    assert payload["codex_puct_selection_lag_stale_rounds"] == 8
    assert payload["num_cpus_per_task"] == 1

    enabled = run_dry_run("--dry-run", "--codex-puct-selection-lag-balance")
    assert enabled["discover_config"]["codex_puct_selection_lag_balance"] is True

    disabled = run_dry_run(
        "dry-run",
        "--codex-puct-selection-lag-balance",
        "--no-codex-puct-selection-lag-balance",
    )
    assert disabled["discover_config"]["codex_puct_selection_lag_balance"] is False

    thresholds = run_dry_run(
        "dry-run",
        "--codex-puct-selection-lag-cooldown-rounds",
        "-3",
        "--codex-puct-selection-lag-stale-rounds",
        "0",
    )
    assert thresholds["discover_config"]["codex_puct_selection_lag_cooldown_rounds"] == -3
    assert thresholds["discover_config"]["codex_puct_selection_lag_stale_rounds"] == 0


def test_gpu_mode_wrapper_dry_run_and_override_path() -> None:
    result = subprocess.run(
        [
            "repro/gpu_mode/run_0609_selection_lag_balance.sh",
            "--dry-run",
            "--no-codex-puct-selection-lag-balance",
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)["discover_config"]
    assert payload["experiment_name"] == "gpu-mode-0609-selection-lag-balance"
    assert payload["wandb_project"] is None
    assert payload["codex_model_name"] is None
    assert payload["codex_puct_selection_lag_balance"] is False


def test_classifier_helpers_only_read_state_id() -> None:
    source = "\n".join(
        textwrap.dedent(inspect.getsource(func))
        for func in (
            sampler_mod._selection_lag_state_id,
            PUCTSampler._classify_selection_lag,
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
 ttt_discover/codex_utils/sampler.py   | 255 +++++++++++++++++++++++++++++++++-
 ttt_discover/discovery.py             |   6 +
 ttt_discover/rl/codex_no_finetune.py  |   6 +
 4 files changed, 270 insertions(+), 3 deletions(-)
````
## Raw Diff

<details>
<summary>Tracked worktree patch</summary>

````diff
diff --git a/ttt_discover/codex_utils/discovery.py b/ttt_discover/codex_utils/discovery.py
index b21a80f..8784c62 100644
--- a/ttt_discover/codex_utils/discovery.py
+++ b/ttt_discover/codex_utils/discovery.py
@@ -50,6 +50,9 @@ class DiscoverConfig:
     codex_initial_program_paths: tuple[str, ...] = ()
     codex_initial_pool_paths: tuple[str, ...] = ()
     codex_autonomous: bool = False
+    codex_puct_selection_lag_balance: bool = False
+    codex_puct_selection_lag_cooldown_rounds: int = 1
+    codex_puct_selection_lag_stale_rounds: int = 8
 
 
 def _run_codex_no_finetune(config: DiscoverConfig) -> None:
@@ -85,6 +88,9 @@ def _run_codex_no_finetune(config: DiscoverConfig) -> None:
         initial_program_paths=config.codex_initial_program_paths,
         initial_pool_paths=config.codex_initial_pool_paths,
         autonomous=config.codex_autonomous,
+        puct_selection_lag_balance=config.codex_puct_selection_lag_balance,
+        puct_selection_lag_cooldown_rounds=config.codex_puct_selection_lag_cooldown_rounds,
+        puct_selection_lag_stale_rounds=config.codex_puct_selection_lag_stale_rounds,
         wandb_project=config.wandb_project,
         wandb_name=experiment_name,
         log_path=log_path,
diff --git a/ttt_discover/codex_utils/sampler.py b/ttt_discover/codex_utils/sampler.py
index 5c5d4d1..31e0a5a 100644
--- a/ttt_discover/codex_utils/sampler.py
+++ b/ttt_discover/codex_utils/sampler.py
@@ -16,6 +16,54 @@ from ttt_discover.codex_utils.runtime import State, state_from_dict, to_json_ser
 
 logger = logging.getLogger(__name__)
 
+SELECTION_LAG_SURFACES = ("never_selected", "stale", "ready", "cooldown")
+
+
+def _zero_selection_lag_counts() -> dict[str, int]:
+    return {surface: 0 for surface in SELECTION_LAG_SURFACES}
+
+
+def _is_nonnegative_int(value: Any) -> bool:
+    return isinstance(value, int) and not isinstance(value, bool) and value >= 0
+
+
+def _sanitize_nonnegative_int(value: Any, *, default: int = 0) -> int:
+    if _is_nonnegative_int(value):
+        return int(value)
+    return max(0, int(default))
+
+
+def _clamp_config_rounds(value: Any, *, default: int) -> int:
+    try:
+        return max(0, int(value))
+    except (TypeError, ValueError):
+        return max(0, int(default))
+
+
+def _sanitize_selection_lag_last_clock(raw: Any) -> dict[str, int]:
+    if not isinstance(raw, dict):
+        return {}
+    out: dict[str, int] = {}
+    for key, value in raw.items():
+        if _is_nonnegative_int(value):
+            out[str(key)] = int(value)
+    return out
+
+
+def _sanitize_selection_lag_surface_counts(raw: Any) -> dict[str, int]:
+    counts = _zero_selection_lag_counts()
+    if not isinstance(raw, dict):
+        return counts
+    for surface in SELECTION_LAG_SURFACES:
+        value = raw.get(surface)
+        if _is_nonnegative_int(value):
+            counts[surface] = int(value)
+    return counts
+
+
+def _selection_lag_state_id(state: State) -> str:
+    return str(state.id)
+
 
 @contextmanager
 def _file_lock(lock_path: str, *, poll_s: float = 0.05, stale_s: float = 600.0):
@@ -353,6 +401,9 @@ class PUCTSampler(StateSampler):
         resume_step: int | None = None,
         puct_c: float = 1.0,
         topk_children: int = 2,
+        puct_selection_lag_balance: bool = False,
+        puct_selection_lag_cooldown_rounds: int = 1,
+        puct_selection_lag_stale_rounds: int = 8,
     ):
         self.file_path = file_path
         self.env_type = env_type
@@ -361,6 +412,18 @@ class PUCTSampler(StateSampler):
         self.batch_size = batch_size
         self.topk_children = topk_children
         self.puct_c = float(puct_c)
+        self.puct_selection_lag_balance = bool(puct_selection_lag_balance)
+        self.puct_selection_lag_cooldown_rounds = _clamp_config_rounds(
+            puct_selection_lag_cooldown_rounds,
+            default=1,
+        )
+        self.puct_selection_lag_stale_rounds = max(
+            self.puct_selection_lag_cooldown_rounds,
+            _clamp_config_rounds(
+                puct_selection_lag_stale_rounds,
+                default=8,
+            ),
+        )
         
         self._states: list[State] = []
         self._initial_states: list[State] = []
@@ -375,6 +438,13 @@ class PUCTSampler(StateSampler):
         self._T: int = 0
         self._last_scale: float = 1.0
         self._last_puct_stats: list[tuple[int, float, float, float, float]] = []
+        self._puct_selection_lag_clock: int = 0
+        self._puct_selection_lag_last_clock: dict[str, int] = {}
+        self._puct_selection_lag_surface_counts: dict[str, int] = _zero_selection_lag_counts()
+        self._last_selection_lag_available_counts: dict[str, int] = _zero_selection_lag_counts()
+        self._last_selection_lag_sampled_counts: dict[str, int] = _zero_selection_lag_counts()
+        self._last_selection_lag_stats: list[tuple[str, int | None, int | None, int | None]] = []
+        self._last_selection_lag_count_befores: list[int] = []
         
         if resume_step is not None:
             self._load(resume_step)
@@ -399,6 +469,16 @@ class PUCTSampler(StateSampler):
         self._n = store.get("puct_n", {}) or {}
         self._m = store.get("puct_m", {}) or {}
         self._T = int(store.get("puct_T", 0) or 0)
+        if self.puct_selection_lag_balance:
+            self._puct_selection_lag_clock = _sanitize_nonnegative_int(
+                store.get("puct_selection_lag_clock", 0),
+            )
+            self._puct_selection_lag_last_clock = _sanitize_selection_lag_last_clock(
+                store.get("puct_selection_lag_last_clock", {}),
+            )
+            self._puct_selection_lag_surface_counts = _sanitize_selection_lag_surface_counts(
+                store.get("puct_selection_lag_surface_counts", {}),
+            )
 
     def _save(self, step: int):
         save_path = _sampler_file_for_step(self.file_path, step)
@@ -411,9 +491,127 @@ class PUCTSampler(StateSampler):
             "puct_m": self._m,
             "puct_T": self._T,
         }
+        if self.puct_selection_lag_balance:
+            self._sanitize_selection_lag_runtime()
+            store.update(
+                {
+                    "puct_selection_lag_clock": self._puct_selection_lag_clock,
+                    "puct_selection_lag_last_clock": self._puct_selection_lag_last_clock,
+                    "puct_selection_lag_surface_counts": self._puct_selection_lag_surface_counts,
+                }
+            )
         with _file_lock(f"{save_path}.lock"):
             _atomic_write_json(save_path, store)
 
+    def _sanitize_selection_lag_runtime(self) -> None:
+        self._puct_selection_lag_clock = _sanitize_nonnegative_int(
+            self._puct_selection_lag_clock,
+        )
+        self._puct_selection_lag_last_clock = _sanitize_selection_lag_last_clock(
+            self._puct_selection_lag_last_clock,
+        )
+        self._puct_selection_lag_surface_counts = _sanitize_selection_lag_surface_counts(
+            self._puct_selection_lag_surface_counts,
+        )
+        self._last_selection_lag_available_counts = _sanitize_selection_lag_surface_counts(
+            self._last_selection_lag_available_counts,
+        )
+        self._last_selection_lag_sampled_counts = _sanitize_selection_lag_surface_counts(
+            self._last_selection_lag_sampled_counts,
+        )
+
+    def _selection_lag_thresholds(self) -> tuple[int, int]:
+        cooldown_rounds = _clamp_config_rounds(
+            self.puct_selection_lag_cooldown_rounds,
+            default=1,
+        )
+        stale_rounds = _clamp_config_rounds(
+            self.puct_selection_lag_stale_rounds,
+            default=8,
+        )
+        if stale_rounds < cooldown_rounds:
+            stale_rounds = cooldown_rounds
+        self.puct_selection_lag_cooldown_rounds = cooldown_rounds
+        self.puct_selection_lag_stale_rounds = stale_rounds
+        return cooldown_rounds, stale_rounds
+
+    def _classify_selection_lag(self, state: State) -> tuple[str, int | None, int | None]:
+        state_id = _selection_lag_state_id(state)
+        current_clock = _sanitize_nonnegative_int(self._puct_selection_lag_clock)
+        last_clock = self._puct_selection_lag_last_clock.get(state_id)
+        if not _is_nonnegative_int(last_clock):
+            return "never_selected", None, None
+
+        lag = max(0, current_clock - int(last_clock))
+        cooldown_rounds, stale_rounds = self._selection_lag_thresholds()
+        if lag <= cooldown_rounds:
+            return "cooldown", int(last_clock), lag
+        if lag <= stale_rounds:
+            return "ready", int(last_clock), lag
+        return "stale", int(last_clock), lag
+
+    def _sample_states_with_selection_lag(
+        self,
+        num_states: int,
+        scores: list[tuple[float, float, State, int, float, float, float]],
+    ) -> tuple[list[State], list[tuple[float, float, State, int, float, float, float]]]:
+        self._sanitize_selection_lag_runtime()
+        self._last_selection_lag_available_counts = _zero_selection_lag_counts()
+        self._last_selection_lag_sampled_counts = _zero_selection_lag_counts()
+        self._last_selection_lag_stats = []
+        self._last_selection_lag_count_befores = []
+
+        picked: list[State] = []
+        top_scores: list[tuple[float, float, State, int, float, float, float]] = []
+        blocked_ids: set[str] = set()
+        children_map = self._build_children_map() if num_states > 1 else {}
+
+        while len(picked) < num_states:
+            if num_states > 1:
+                selectable = [entry for entry in scores if entry[2].id not in blocked_ids]
+            else:
+                selectable = scores
+            if not selectable:
+                break
+
+            classified = []
+            available_counts = _zero_selection_lag_counts()
+            for entry in selectable:
+                surface, last_clock, lag = self._classify_selection_lag(entry[2])
+                available_counts[surface] += 1
+                classified.append((entry, surface, last_clock, lag))
+            self._last_selection_lag_available_counts = available_counts
+
+            non_cooldown = [item for item in classified if item[1] != "cooldown"]
+            if non_cooldown:
+                available_surfaces = {item[1] for item in non_cooldown}
+                target_surface = min(
+                    (surface for surface in SELECTION_LAG_SURFACES if surface in available_surfaces),
+                    key=lambda surface: self._puct_selection_lag_surface_counts[surface],
+                )
+                selected_entry, surface, last_clock, lag = next(
+                    item for item in classified if item[1] == target_surface
+                )
+            else:
+                selected_entry, surface, last_clock, lag = classified[0]
+
+            state = selected_entry[2]
+            clock_before = _sanitize_nonnegative_int(self._puct_selection_lag_clock)
+            count_before = self._puct_selection_lag_surface_counts[surface]
+            self._last_selection_lag_stats.append((surface, last_clock, clock_before, lag))
+            self._last_selection_lag_count_befores.append(count_before)
+            self._last_selection_lag_sampled_counts[surface] += 1
+            self._puct_selection_lag_surface_counts[surface] = count_before + 1
+            self._puct_selection_lag_last_clock[_selection_lag_state_id(state)] = clock_before
+            self._puct_selection_lag_clock = clock_before + 1
+
+            picked.append(state)
+            top_scores.append(selected_entry)
+            if num_states > 1:
+                blocked_ids.update(self._get_full_lineage(state, children_map))
+
+        return picked, top_scores
+
     def _refresh_random_construction(self, state: State) -> None:
         """Let task-specific envs refresh seed states without coupling sampler to a task."""
         refresh = getattr(self.env_type, "refresh_initial_state", None)
@@ -501,6 +699,13 @@ class PUCTSampler(StateSampler):
             self._last_sampled_states = picked
             self._last_sampled_indices = []
             self._last_puct_stats = [(0, 0.0, 0.0, 0.0, 0.0) for _ in picked]
+            if self.puct_selection_lag_balance:
+                self._last_selection_lag_available_counts = _zero_selection_lag_counts()
+                self._last_selection_lag_sampled_counts = _zero_selection_lag_counts()
+                self._last_selection_lag_stats = [
+                    ("never_selected", None, None, None) for _ in picked
+                ]
+                self._last_selection_lag_count_befores = [0 for _ in picked]
             return picked
 
         vals = np.array([float(s.value if s.value is not None else float("-inf")) for s in candidates])
@@ -521,7 +726,9 @@ class PUCTSampler(StateSampler):
 
         scores.sort(key=lambda x: (x[0], x[1]), reverse=True)
 
-        if num_states > 1:
+        if self.puct_selection_lag_balance:
+            picked, top_scores = self._sample_states_with_selection_lag(num_states, scores)
+        elif num_states > 1:
             children_map = self._build_children_map()
             picked, top_scores, blocked_ids = [], [], set()
             for entry in scores:
@@ -731,21 +938,51 @@ class PUCTSampler(StateSampler):
         stats.update(_stats(sampled_values, "puct/sampled_value"))
         stats.update(_stats(sampled_timesteps, "puct/sampled_timestep"))
         stats.update(_stats(sampled_constr_lens, "puct/sampled_construction_len"))
+        if self.puct_selection_lag_balance:
+            self._sanitize_selection_lag_runtime()
+            stats["puct/selection_lag_balance/enabled"] = 1
+            stats["puct/selection_lag_balance/clock"] = self._puct_selection_lag_clock
+            for surface in SELECTION_LAG_SURFACES:
+                stats[f"puct/selection_lag_balance/count/{surface}"] = (
+                    self._puct_selection_lag_surface_counts[surface]
+                )
+                stats[f"puct/selection_lag_balance/available/{surface}"] = (
+                    self._last_selection_lag_available_counts[surface]
+                )
+                stats[f"puct/selection_lag_balance/sampled/{surface}"] = (
+                    self._last_selection_lag_sampled_counts[surface]
+                )
         return stats
 
     def get_sample_table(self) -> tuple[list[str], list[tuple]]:
         columns = ["buffer_idx", "timestep", "value", "terminal_value", "parent_value", "construction_len", "observation_len", "n", "Q", "P", "bonus", "score"]
+        if self.puct_selection_lag_balance:
+            columns.extend(
+                [
+                    "selection_lag_surface",
+                    "selection_lag_last_clock",
+                    "selection_lag_clock",
+                    "selection_lag",
+                ]
+            )
         rows = []
         if not self._last_sampled_states:
             return columns, rows
         indices = self._last_sampled_indices if len(self._last_sampled_indices) == len(self._last_sampled_states) else [-1] * len(self._last_sampled_states)
         stats = self._last_puct_stats if len(self._last_puct_stats) == len(self._last_sampled_states) else [(0, 0.0, 0.0, 0.0, 0.0)] * len(self._last_sampled_states)
-        for idx, state, (n, Q, P, bonus, score) in zip(indices, self._last_sampled_states, stats):
+        if self.puct_selection_lag_balance and len(self._last_selection_lag_stats) == len(self._last_sampled_states):
+            lag_stats = self._last_selection_lag_stats
+        else:
+            lag_stats = [("never_selected", None, None, None)] * len(self._last_sampled_states)
+        for idx, state, (n, Q, P, bonus, score), lag_stat in zip(indices, self._last_sampled_states, stats, lag_stats):
             parent_val = state.parent_values[0] if state.parent_values else None
             constr = getattr(state, 'construction', None)
             constr_len = len(constr) if constr is not None else 0
             obs_len = len(state.observation) if state.observation else 0
-            rows.append((idx, state.timestep, state.value, 0, parent_val, constr_len, obs_len, n, Q, P, bonus, score))
+            row = (idx, state.timestep, state.value, 0, parent_val, constr_len, obs_len, n, Q, P, bonus, score)
+            if self.puct_selection_lag_balance:
+                row = row + lag_stat
+            rows.append(row)
         return columns, rows
 
 
@@ -756,6 +993,9 @@ def create_sampler(
     batch_size: int = 1,
     resume_step: int | None = None,
     topk_children: int = 2,
+    puct_selection_lag_balance: bool = False,
+    puct_selection_lag_cooldown_rounds: int = 1,
+    puct_selection_lag_stale_rounds: int = 8,
 ) -> StateSampler:
     """Factory function to create samplers. Pass the env type (from config.env_type)."""
     if not log_path:
@@ -768,6 +1008,9 @@ def create_sampler(
         batch_size=batch_size,
         resume_step=resume_step,
         topk_children=topk_children,
+        puct_selection_lag_balance=puct_selection_lag_balance,
+        puct_selection_lag_cooldown_rounds=puct_selection_lag_cooldown_rounds,
+        puct_selection_lag_stale_rounds=puct_selection_lag_stale_rounds,
     )
 
 
@@ -778,6 +1021,9 @@ def get_or_create_sampler_with_default(
     batch_size: int = 1,
     resume_step: int | None = None,
     topk_children: int = 2,
+    puct_selection_lag_balance: bool = False,
+    puct_selection_lag_cooldown_rounds: int = 1,
+    puct_selection_lag_stale_rounds: int = 8,
 ) -> StateSampler:
     """Get sampler. Initial experience is created via env_type.create_initial_state."""
     return create_sampler(
@@ -787,4 +1033,7 @@ def get_or_create_sampler_with_default(
         batch_size=batch_size,
         resume_step=resume_step,
         topk_children=topk_children,
+        puct_selection_lag_balance=puct_selection_lag_balance,
+        puct_selection_lag_cooldown_rounds=puct_selection_lag_cooldown_rounds,
+        puct_selection_lag_stale_rounds=puct_selection_lag_stale_rounds,
     )
diff --git a/ttt_discover/discovery.py b/ttt_discover/discovery.py
index 6cdb434..436c223 100644
--- a/ttt_discover/discovery.py
+++ b/ttt_discover/discovery.py
@@ -63,6 +63,9 @@ class DiscoverConfig:
     codex_initial_program_paths: tuple[str, ...] = ()
     codex_initial_pool_paths: tuple[str, ...] = ()
     codex_autonomous: bool = False
+    codex_puct_selection_lag_balance: bool = False
+    codex_puct_selection_lag_cooldown_rounds: int = 1
+    codex_puct_selection_lag_stale_rounds: int = 8
 
 
 def init_ray(num_cpus_per_task: int, env_type: str):
@@ -146,6 +149,9 @@ async def discover_impl(config: DiscoverConfig):
             initial_program_paths=config.codex_initial_program_paths,
             initial_pool_paths=config.codex_initial_pool_paths,
             autonomous=config.codex_autonomous,
+            puct_selection_lag_balance=config.codex_puct_selection_lag_balance,
+            puct_selection_lag_cooldown_rounds=config.codex_puct_selection_lag_cooldown_rounds,
+            puct_selection_lag_stale_rounds=config.codex_puct_selection_lag_stale_rounds,
             wandb_project=config.wandb_project,
             wandb_name=config.experiment_name,
             log_path=log_path,
diff --git a/ttt_discover/rl/codex_no_finetune.py b/ttt_discover/rl/codex_no_finetune.py
index 18a0e9e..7304738 100644
--- a/ttt_discover/rl/codex_no_finetune.py
+++ b/ttt_discover/rl/codex_no_finetune.py
@@ -137,6 +137,9 @@ class CodexNoFinetuneConfig:
     initial_program_paths: tuple[str, ...] = ()
     initial_pool_paths: tuple[str, ...] = ()
     topk_children: int = 16
+    puct_selection_lag_balance: bool = False
+    puct_selection_lag_cooldown_rounds: int = 1
+    puct_selection_lag_stale_rounds: int = 8
 
     # False is TTT Discover: non-auto, many short samples. True is AutoEvolve:
     # one Codex deep-dive workspace per sample with file edits and shell access.
@@ -960,6 +963,9 @@ def _build_sampler(cfg: CodexNoFinetuneConfig, start_batch: int) -> StateSampler
         batch_size=cfg.groups_per_batch,
         resume_step=start_batch if start_batch > 0 else None,
         topk_children=cfg.topk_children,
+        puct_selection_lag_balance=cfg.puct_selection_lag_balance,
+        puct_selection_lag_cooldown_rounds=cfg.puct_selection_lag_cooldown_rounds,
+        puct_selection_lag_stale_rounds=cfg.puct_selection_lag_stale_rounds,
     )
 
 
````
</details>

