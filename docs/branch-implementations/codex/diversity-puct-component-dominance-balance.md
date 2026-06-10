# codex/diversity-puct-component-dominance-balance

## Summary

拆分 PUCT score 中 exploitation(Q/P/value) 与 exploration bonus 的相对主导性，分类 unvisited/explore/exploit/mixed 后平衡。

## Branch State

- Worktree: `/opt/tiger/discover-puct-component-dominance-balance`
- HEAD: `2234282`
- Base used for comparison: `2234282`
- Commits ahead of base: `0`
- Commits behind base: `0`
- Group: `completed_19`
- Implementation location: `uncommitted worktree diff`

## Implemented Behavior

- The summary above is based on reading the changed Python code, added config fields, and sampler/prompt hooks in this branch.
- For sampler branches, the common pattern is to keep baseline PUCT ordering inside a chosen surface/bucket, while changing which surface/bucket gets the next parent pick.
- Tests detected: `12` test functions across untracked test files.

## Added Markers

### Config fields

- `codex_puct_component_balance`
- `codex_puct_component_dominance_ratio`
- `counts`
- `puct_component_balance`
- `puct_component_dominance_ratio`

### Constants

- `PUCT_COMPONENT_SURFACES`
- `_PUCT_COMPONENT_SELECTED_COUNTS_KEY`

### Classes

- None

### Functions

- `_sanitize_puct_component_ratio`
- `_sanitize_puct_component_counts`
- `_puct_component_metric_suffix`
- `puct_component_q_center`
- `puct_component_eps`
- `puct_component_q_edge`
- `classify_puct_component_surface`
- `_puct_component_q_center`
- `_puct_component_surface`
- `_puct_component_entry_details`
- `_puct_component_available_entries`
- `_puct_component_available_counts`
- `_select_puct_component_entry`
- `_sample_states_puct_component_balanced`

## Diff Summary

- Worktree tracked shortstat: `4 files changed, 317 insertions(+), 3 deletions(-)`
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

- `repro/gpu_mode/run_0609_puct_component_balance.sh (701 bytes)`
- `repro/run_discovery.py (6510 bytes)`
- `tests/test_codex_puct_component_balance.py (13079 bytes)`

### Detected Test Functions

- `tests/test_codex_puct_component_balance.py::test_disabled_parity_no_component_observability_or_persistence`
- `tests/test_codex_puct_component_balance.py::test_classifier_surfaces_and_reward_offset_invariance`
- `tests/test_codex_puct_component_balance.py::test_surface_count_balance_and_within_surface_puct_order`
- `tests/test_codex_puct_component_balance.py::test_equal_count_tie_uses_earliest_puct_surface`
- `tests/test_codex_puct_component_balance.py::test_batch_lineage_blocking_recomputes_availability`
- `tests/test_codex_puct_component_balance.py::test_enabled_persistence_resume_and_disabled_omits_keys`
- `tests/test_codex_puct_component_balance.py::test_stats_available_selected_and_table_values`
- `tests/test_codex_puct_component_balance.py::test_record_failed_rollout_only_updates_puct_visits`
- `tests/test_codex_puct_component_balance.py::test_build_sampler_passes_component_args`
- `tests/test_codex_puct_component_balance.py::test_run_discovery_dry_run_cli_defaults_enabled_override_and_validation`
- `tests/test_codex_puct_component_balance.py::test_gpu_mode_wrapper_dry_run_first_and_override_path`
- `tests/test_codex_puct_component_balance.py::test_classifier_helpers_only_read_state_id`

## Untracked File Snapshots

These files are not part of `git diff` yet, but they are part of the worktree implementation state recorded here.

### `repro/gpu_mode/run_0609_puct_component_balance.sh`

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
  --experiment-name trimul_0609_puct_component_balance_gpu2 \
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
  --codex-puct-component-balance \
  --codex-puct-component-dominance-ratio 2.0 \
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
        default="gpu-mode-0609-puct-component-balance",
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
        "--codex-puct-component-balance",
        dest="codex_puct_component_balance",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--no-codex-puct-component-balance",
        dest="codex_puct_component_balance",
        action="store_false",
    )
    parser.add_argument("--codex-puct-component-dominance-ratio", type=float, default=2.0)
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
        "codex_puct_component_balance": args.codex_puct_component_balance,
        "codex_puct_component_dominance_ratio": (
            args.codex_puct_component_dominance_ratio
        ),
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

### `tests/test_codex_puct_component_balance.py`

````python
from __future__ import annotations

import ast
import inspect
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from ttt_discover.codex_utils.runtime import State
import ttt_discover.codex_utils.sampler as sampler_mod
from ttt_discover.codex_utils.sampler import PUCTSampler, PUCT_COMPONENT_SURFACES
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
    ratio: float = 2.0,
) -> PUCTSampler:
    return PUCTSampler(
        file_path=str(tmp_path / "puct_sampler.json"),
        env_type=DummyEnv,
        batch_size=0,
        puct_c=0.0,
        topk_children=0,
        puct_component_balance=enabled,
        puct_component_dominance_ratio=ratio,
    )


def saved_store(tmp_path: Path, step: int) -> dict:
    path = tmp_path / f"puct_sampler_step_{step:06d}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def ids(states: list[State]) -> list[str]:
    return [state.id for state in states]


def test_disabled_parity_no_component_observability_or_persistence(
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
    assert "component_surface" not in columns
    stats = sampler.get_sample_stats()
    assert not any(key.startswith("puct_component_balance/") for key in stats)

    sampler.flush(step=1)
    store = saved_store(tmp_path, 1)
    assert "puct_component_balance_selected_counts" not in store


def test_classifier_surfaces_and_reward_offset_invariance(tmp_path: Path) -> None:
    sampler = make_sampler(tmp_path, enabled=True, ratio=2.0)
    state = make_state("s", 0.0)

    assert (
        sampler_mod.classify_puct_component_surface(
            state,
            n=0,
            q_value=100.0,
            bonus=100.0,
            q_center=0.0,
            scale=1.0,
            dominance_ratio=2.0,
        )
        == "component:unvisited"
    )
    assert (
        sampler_mod.classify_puct_component_surface(
            state,
            n=1,
            q_value=10.0,
            bonus=5.0,
            q_center=9.0,
            scale=10.0,
            dominance_ratio=2.0,
        )
        == "component:explore"
    )
    assert (
        sampler_mod.classify_puct_component_surface(
            state,
            n=1,
            q_value=10.0,
            bonus=0.5,
            q_center=9.0,
            scale=10.0,
            dominance_ratio=2.0,
        )
        == "component:exploit"
    )
    assert (
        sampler_mod.classify_puct_component_surface(
            state,
            n=1,
            q_value=10.0,
            bonus=0.75,
            q_center=9.0,
            scale=10.0,
            dominance_ratio=2.0,
        )
        == "component:mixed"
    )

    base = sampler_mod.classify_puct_component_surface(
        state,
        n=1,
        q_value=10.0,
        bonus=0.5,
        q_center=9.0,
        scale=10.0,
        dominance_ratio=2.0,
    )
    shifted = sampler_mod.classify_puct_component_surface(
        state,
        n=1,
        q_value=1010.0,
        bonus=0.5,
        q_center=1009.0,
        scale=10.0,
        dominance_ratio=2.0,
    )
    assert shifted == base
    assert sampler.puct_component_dominance_ratio == 2.0
    assert sampler_mod._sanitize_puct_component_ratio(-1.0) == 2.0


def test_surface_count_balance_and_within_surface_puct_order(tmp_path: Path) -> None:
    sampler = make_sampler(tmp_path, enabled=True, ratio=2.0)
    sampler._states = [
        make_state("unvisited_top", 100.0),
        make_state("explore_high", 90.0),
        make_state("explore_low", 80.0),
    ]
    sampler._n = {
        "explore_high": 1,
        "explore_low": 1,
    }
    sampler._puct_component_selected_counts = {
        "component:unvisited": 5,
        "component:explore": 0,
        "component:exploit": 5,
        "component:mixed": 5,
    }
    sampler.puct_c = 10.0

    picked = sampler.sample_states(1)

    assert ids(picked) == ["explore_high"]
    assert sampler._puct_component_selected_counts["component:explore"] == 1
    columns, rows = sampler.get_sample_table()
    assert columns[-4:] == [
        "component_surface",
        "component_count_before",
        "component_q_edge",
        "component_bonus_to_q_edge",
    ]
    assert rows[0][-4] == "component:explore"
    assert rows[0][-3] == 0


def test_equal_count_tie_uses_earliest_puct_surface(tmp_path: Path) -> None:
    sampler = make_sampler(tmp_path, enabled=True, ratio=2.0)
    sampler._states = [
        make_state("unvisited", 100.0),
        make_state("explore", 90.0),
    ]
    sampler._n = {"explore": 1}
    sampler.puct_c = 10.0

    picked = sampler.sample_states(1)

    assert PUCT_COMPONENT_SURFACES == (
        "component:unvisited",
        "component:explore",
        "component:exploit",
        "component:mixed",
    )
    assert ids(picked) == ["unvisited"]
    assert sampler._puct_component_selected_counts["component:unvisited"] == 1


def test_batch_lineage_blocking_recomputes_availability(tmp_path: Path) -> None:
    sampler = make_sampler(tmp_path, enabled=True)
    root = make_state("root", 100.0)
    child = make_state("child", 99.0, parents=[{"id": "root", "timestep": 0}])
    other = make_state("other", 80.0)
    sampler._states = [root, child, other]

    picked = sampler.sample_states(2)

    assert ids(picked) == ["root", "other"]
    assert "child" not in ids(picked)
    assert sampler._puct_component_selected_counts["component:unvisited"] == 2


def test_enabled_persistence_resume_and_disabled_omits_keys(tmp_path: Path) -> None:
    sampler = make_sampler(tmp_path, enabled=True)
    sampler._states = [make_state("a", 1.0)]
    sampler._puct_component_selected_counts = {
        "component:unvisited": True,
        "component:explore": 2,
        "component:exploit": -1,
        "component:mixed": 4,
        "bad": 99,
    }
    sampler.flush(step=1)

    store = saved_store(tmp_path, 1)
    assert store["puct_component_balance_selected_counts"] == {
        "component:unvisited": 0,
        "component:explore": 2,
        "component:exploit": 0,
        "component:mixed": 4,
    }

    resumed = PUCTSampler(
        file_path=str(tmp_path / "puct_sampler.json"),
        env_type=DummyEnv,
        batch_size=0,
        resume_step=1,
        puct_component_balance=True,
    )
    assert resumed._puct_component_selected_counts["component:mixed"] == 4

    disabled = PUCTSampler(
        file_path=str(tmp_path / "puct_sampler.json"),
        env_type=DummyEnv,
        batch_size=0,
        resume_step=1,
        puct_component_balance=False,
    )
    disabled.flush(step=2)
    disabled_store = saved_store(tmp_path, 2)
    assert "puct_component_balance_selected_counts" not in disabled_store


def test_stats_available_selected_and_table_values(tmp_path: Path) -> None:
    sampler = make_sampler(tmp_path, enabled=True)
    sampler._states = [make_state("a", 2.0), make_state("b", 1.0)]

    picked = sampler.sample_states(1)
    stats = sampler.get_sample_stats()

    assert ids(picked) == ["a"]
    assert stats["puct_component_balance/last_selected/unvisited"] == 1
    assert stats["puct_component_balance/available/unvisited"] == 2
    assert stats["puct_component_balance/selected_total/unvisited"] == 1
    row = sampler.get_sample_table()[1][0]
    assert row[-4] == "component:unvisited"
    assert row[-3] == 0
    assert row[-2] == pytest.approx(0.5)
    assert row[-1] == pytest.approx(0.0)


def test_record_failed_rollout_only_updates_puct_visits(tmp_path: Path) -> None:
    sampler = make_sampler(tmp_path, enabled=True)
    parent = make_state("parent", 10.0, parents=[{"id": "ancestor", "timestep": 0}])
    sampler._states = [parent]

    assert ids(sampler.sample_states(1)) == ["parent"]
    assert sampler._puct_component_selected_counts["component:unvisited"] == 1

    sampler.record_failed_rollout(parent)

    assert sampler._puct_component_selected_counts["component:unvisited"] == 1
    assert sampler._T == 1
    assert sampler._n["parent"] == 1
    assert sampler._n["ancestor"] == 1


def test_build_sampler_passes_component_args(monkeypatch, tmp_path: Path) -> None:
    captured: dict = {}

    def fake_create_sampler(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(codex_no_finetune, "create_sampler", fake_create_sampler)
    cfg = CodexNoFinetuneConfig(
        env_type=DummyEnv,
        log_path=str(tmp_path),
        groups_per_batch=3,
        puct_component_balance=True,
        puct_component_dominance_ratio=3.5,
    )

    assert codex_no_finetune._build_sampler(cfg, start_batch=4) is not None
    assert captured["batch_size"] == 3
    assert captured["resume_step"] == 4
    assert captured["puct_component_balance"] is True
    assert captured["puct_component_dominance_ratio"] == 3.5


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
    assert default["codex_puct_component_balance"] is False
    assert default["codex_puct_component_dominance_ratio"] == 2.0
    assert default["codex_model_name"] is None
    assert default["num_cpus_per_task"] == 1
    assert default["wandb_project"] is None

    enabled = dry_payload(
        "--dry-run",
        "--codex-puct-component-balance",
        "--codex-puct-component-dominance-ratio",
        "3.5",
        "--num-cpus-per-task",
        "3",
    )
    assert enabled["codex_puct_component_balance"] is True
    assert enabled["codex_puct_component_dominance_ratio"] == 3.5
    assert enabled["num_cpus_per_task"] == 3

    disabled = dry_payload(
        "dry-run",
        "--codex-puct-component-balance",
        "--no-codex-puct-component-balance",
        "--codex-puct-component-dominance-ratio",
        "4",
    )
    assert disabled["codex_puct_component_balance"] is False
    assert disabled["codex_puct_component_dominance_ratio"] == 4.0

    invalid = run_dry_run(
        "dry-run",
        "--codex-puct-component-dominance-ratio",
        "not-a-float",
        check=False,
    )
    assert invalid.returncode == 2


def test_gpu_mode_wrapper_dry_run_first_and_override_path() -> None:
    result = subprocess.run(
        [
            "repro/gpu_mode/run_0609_puct_component_balance.sh",
            "--dry-run",
            "--no-codex-puct-component-balance",
            "--codex-puct-component-dominance-ratio",
            "4",
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)["discover_config"]
    assert payload["experiment_name"] == "trimul_0609_puct_component_balance_gpu2"
    assert payload["wandb_project"] is None
    assert payload["codex_model_name"] is None
    assert payload["codex_puct_component_balance"] is False
    assert payload["codex_puct_component_dominance_ratio"] == 4.0


def test_classifier_helpers_only_read_state_id() -> None:
    source = "\n".join(
        textwrap.dedent(inspect.getsource(func))
        for func in (
            sampler_mod.classify_puct_component_surface,
            PUCTSampler._puct_component_surface,
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
 ttt_discover/codex_utils/sampler.py   | 308 +++++++++++++++++++++++++++++++++-
 ttt_discover/discovery.py             |   4 +
 ttt_discover/rl/codex_no_finetune.py  |   4 +
 4 files changed, 317 insertions(+), 3 deletions(-)
````
## Raw Diff

<details>
<summary>Tracked worktree patch</summary>

````diff
diff --git a/ttt_discover/codex_utils/discovery.py b/ttt_discover/codex_utils/discovery.py
index b21a80f..967845b 100644
--- a/ttt_discover/codex_utils/discovery.py
+++ b/ttt_discover/codex_utils/discovery.py
@@ -50,6 +50,8 @@ class DiscoverConfig:
     codex_initial_program_paths: tuple[str, ...] = ()
     codex_initial_pool_paths: tuple[str, ...] = ()
     codex_autonomous: bool = False
+    codex_puct_component_balance: bool = False
+    codex_puct_component_dominance_ratio: float = 2.0
 
 
 def _run_codex_no_finetune(config: DiscoverConfig) -> None:
@@ -85,6 +87,8 @@ def _run_codex_no_finetune(config: DiscoverConfig) -> None:
         initial_program_paths=config.codex_initial_program_paths,
         initial_pool_paths=config.codex_initial_pool_paths,
         autonomous=config.codex_autonomous,
+        puct_component_balance=config.codex_puct_component_balance,
+        puct_component_dominance_ratio=config.codex_puct_component_dominance_ratio,
         wandb_project=config.wandb_project,
         wandb_name=experiment_name,
         log_path=log_path,
diff --git a/ttt_discover/codex_utils/sampler.py b/ttt_discover/codex_utils/sampler.py
index 5c5d4d1..432c26f 100644
--- a/ttt_discover/codex_utils/sampler.py
+++ b/ttt_discover/codex_utils/sampler.py
@@ -16,6 +16,87 @@ from ttt_discover.codex_utils.runtime import State, state_from_dict, to_json_ser
 
 logger = logging.getLogger(__name__)
 
+PUCT_COMPONENT_SURFACES = (
+    "component:unvisited",
+    "component:explore",
+    "component:exploit",
+    "component:mixed",
+)
+_PUCT_COMPONENT_SELECTED_COUNTS_KEY = "puct_component_balance_selected_counts"
+_PuctEntry = tuple[float, float, State, int, float, float, float]
+
+
+def _sanitize_puct_component_ratio(raw_ratio: Any) -> float:
+    if isinstance(raw_ratio, (bool, np.bool_)):
+        return 2.0
+    if isinstance(raw_ratio, (int, float, np.integer, np.floating)):
+        value = float(raw_ratio)
+        if np.isfinite(value) and value > 0.0:
+            return value
+    return 2.0
+
+
+def _sanitize_puct_component_counts(raw: Any) -> dict[str, int]:
+    raw_counts = raw if isinstance(raw, dict) else {}
+    counts: dict[str, int] = {}
+    for surface in PUCT_COMPONENT_SURFACES:
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
+def _puct_component_metric_suffix(surface: str) -> str:
+    return surface.split(":", 1)[1]
+
+
+def puct_component_q_center(q_values: list[float]) -> float:
+    finite = [float(value) for value in q_values if np.isfinite(float(value))]
+    if not finite:
+        return 0.0
+    return float(np.median(np.array(finite, dtype=np.float64)))
+
+
+def puct_component_eps(scale: float) -> float:
+    safe_scale = float(scale) if np.isfinite(float(scale)) else 1.0
+    return max(1e-12, 1e-9 * max(1.0, safe_scale))
+
+
+def puct_component_q_edge(q_value: float, q_center: float) -> float:
+    q = float(q_value)
+    if not np.isfinite(q):
+        return 0.0
+    center = float(q_center) if np.isfinite(float(q_center)) else 0.0
+    return max(0.0, q - center)
+
+
+def classify_puct_component_surface(
+    state: Any,
+    *,
+    n: int,
+    q_value: float,
+    bonus: float,
+    q_center: float,
+    scale: float,
+    dominance_ratio: float,
+) -> str:
+    _ = str(state.id)
+    if isinstance(n, (bool, np.bool_)) or int(n) <= 0:
+        return "component:unvisited"
+    ratio = _sanitize_puct_component_ratio(dominance_ratio)
+    eps = puct_component_eps(scale)
+    q_edge = puct_component_q_edge(q_value, q_center)
+    bonus_mag = abs(float(bonus)) if np.isfinite(float(bonus)) else 0.0
+    if bonus_mag >= ratio * max(q_edge, eps):
+        return "component:explore"
+    if q_edge >= ratio * max(bonus_mag, eps):
+        return "component:exploit"
+    return "component:mixed"
+
 
 @contextmanager
 def _file_lock(lock_path: str, *, poll_s: float = 0.05, stale_s: float = 600.0):
@@ -353,6 +434,8 @@ class PUCTSampler(StateSampler):
         resume_step: int | None = None,
         puct_c: float = 1.0,
         topk_children: int = 2,
+        puct_component_balance: bool = False,
+        puct_component_dominance_ratio: float = 2.0,
     ):
         self.file_path = file_path
         self.env_type = env_type
@@ -361,6 +444,10 @@ class PUCTSampler(StateSampler):
         self.batch_size = batch_size
         self.topk_children = topk_children
         self.puct_c = float(puct_c)
+        self.puct_component_balance = bool(puct_component_balance)
+        self.puct_component_dominance_ratio = _sanitize_puct_component_ratio(
+            puct_component_dominance_ratio
+        )
         
         self._states: list[State] = []
         self._initial_states: list[State] = []
@@ -375,6 +462,16 @@ class PUCTSampler(StateSampler):
         self._T: int = 0
         self._last_scale: float = 1.0
         self._last_puct_stats: list[tuple[int, float, float, float, float]] = []
+        self._puct_component_selected_counts: dict[str, int] = (
+            _sanitize_puct_component_counts({})
+        )
+        self._last_puct_component_rows: list[tuple[str, int, float, float]] = []
+        self._last_puct_component_available_counts: dict[str, int] = (
+            _sanitize_puct_component_counts({})
+        )
+        self._last_puct_component_selected_counts: dict[str, int] = (
+            _sanitize_puct_component_counts({})
+        )
         
         if resume_step is not None:
             self._load(resume_step)
@@ -399,6 +496,10 @@ class PUCTSampler(StateSampler):
         self._n = store.get("puct_n", {}) or {}
         self._m = store.get("puct_m", {}) or {}
         self._T = int(store.get("puct_T", 0) or 0)
+        if self.puct_component_balance:
+            self._puct_component_selected_counts = _sanitize_puct_component_counts(
+                store.get(_PUCT_COMPONENT_SELECTED_COUNTS_KEY, {})
+            )
 
     def _save(self, step: int):
         save_path = _sampler_file_for_step(self.file_path, step)
@@ -411,6 +512,13 @@ class PUCTSampler(StateSampler):
             "puct_m": self._m,
             "puct_T": self._T,
         }
+        if self.puct_component_balance:
+            self._puct_component_selected_counts = _sanitize_puct_component_counts(
+                self._puct_component_selected_counts
+            )
+            store[_PUCT_COMPONENT_SELECTED_COUNTS_KEY] = (
+                self._puct_component_selected_counts
+            )
         with _file_lock(f"{save_path}.lock"):
             _atomic_write_json(save_path, store)
 
@@ -489,6 +597,145 @@ class PUCTSampler(StateSampler):
                     queue.append(child_id)
         return lineage
 
+    def _puct_component_q_center(self, scores: list[_PuctEntry]) -> float:
+        return puct_component_q_center([entry[4] for entry in scores])
+
+    def _puct_component_surface(
+        self,
+        entry: _PuctEntry,
+        q_center: float,
+    ) -> str:
+        score, _value, state, n, q_value, _prior, bonus = entry
+        _ = score
+        return classify_puct_component_surface(
+            state,
+            n=n,
+            q_value=q_value,
+            bonus=bonus,
+            q_center=q_center,
+            scale=self._last_scale,
+            dominance_ratio=self.puct_component_dominance_ratio,
+        )
+
+    def _puct_component_entry_details(
+        self,
+        entry: _PuctEntry,
+        q_center: float,
+    ) -> tuple[float, float]:
+        q_edge = puct_component_q_edge(entry[4], q_center)
+        eps = puct_component_eps(self._last_scale)
+        bonus_mag = abs(float(entry[6])) if np.isfinite(float(entry[6])) else 0.0
+        ratio = bonus_mag / max(q_edge, eps)
+        return q_edge, ratio
+
+    def _puct_component_available_entries(
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
+    def _puct_component_available_counts(
+        self,
+        entries: list[_PuctEntry],
+        q_center: float,
+    ) -> dict[str, int]:
+        counts = _sanitize_puct_component_counts({})
+        for entry in entries:
+            surface = self._puct_component_surface(entry, q_center)
+            counts[surface] += 1
+        return counts
+
+    def _select_puct_component_entry(
+        self,
+        available: list[_PuctEntry],
+        planned_counts: dict[str, int],
+        q_center: float,
+    ) -> _PuctEntry | None:
+        selected: _PuctEntry | None = None
+        selected_count: int | None = None
+        for entry in available:
+            surface = self._puct_component_surface(entry, q_center)
+            count = planned_counts[surface]
+            if selected_count is None or count < selected_count:
+                selected = entry
+                selected_count = count
+        return selected
+
+    def _sample_states_puct_component_balanced(
+        self,
+        scores: list[_PuctEntry],
+        num_states: int,
+    ) -> tuple[list[State], list[_PuctEntry]]:
+        q_center = self._puct_component_q_center(scores)
+        planned_counts = _sanitize_puct_component_counts(
+            self._puct_component_selected_counts
+        )
+        use_lineage_blocking = num_states > 1
+        children_map = self._build_children_map() if use_lineage_blocking else {}
+
+        picked: list[State] = []
+        top_scores: list[_PuctEntry] = []
+        picked_ids: set[str] = set()
+        blocked_ids: set[str] = set()
+        self._last_puct_component_rows = []
+        self._last_puct_component_selected_counts = _sanitize_puct_component_counts({})
+        self._last_puct_component_available_counts = self._puct_component_available_counts(
+            scores,
+            q_center,
+        )
+
+        while len(picked) < num_states:
+            available = self._puct_component_available_entries(
+                scores,
+                picked_ids=picked_ids,
+                blocked_ids=blocked_ids,
+                use_lineage_blocking=use_lineage_blocking,
+            )
+            entry = self._select_puct_component_entry(
+                available,
+                planned_counts,
+                q_center,
+            )
+            if entry is None:
+                break
+            state = entry[2]
+            surface = self._puct_component_surface(entry, q_center)
+            q_edge, bonus_to_q_edge = self._puct_component_entry_details(
+                entry,
+                q_center,
+            )
+            count_before = planned_counts[surface]
+            picked.append(state)
+            top_scores.append(entry)
+            picked_ids.add(str(state.id))
+            self._last_puct_component_rows.append(
+                (surface, count_before, q_edge, bonus_to_q_edge)
+            )
+            planned_counts[surface] = count_before + 1
+            self._last_puct_component_selected_counts[surface] += 1
+            if use_lineage_blocking:
+                blocked_ids.update(
+                    str(state_id)
+                    for state_id in self._get_full_lineage(state, children_map)
+                )
+
+        self._puct_component_selected_counts = _sanitize_puct_component_counts(
+            planned_counts
+        )
+        return picked, top_scores
+
     def sample_states(self, num_states: int) -> list[State]:
         initial_ids = {s.id for s in self._initial_states}
         candidates = list(self._states)
@@ -501,6 +748,10 @@ class PUCTSampler(StateSampler):
             self._last_sampled_states = picked
             self._last_sampled_indices = []
             self._last_puct_stats = [(0, 0.0, 0.0, 0.0, 0.0) for _ in picked]
+            if self.puct_component_balance:
+                self._last_puct_component_rows = []
+                self._last_puct_component_available_counts = _sanitize_puct_component_counts({})
+                self._last_puct_component_selected_counts = _sanitize_puct_component_counts({})
             return picked
 
         vals = np.array([float(s.value if s.value is not None else float("-inf")) for s in candidates])
@@ -521,7 +772,12 @@ class PUCTSampler(StateSampler):
 
         scores.sort(key=lambda x: (x[0], x[1]), reverse=True)
 
-        if num_states > 1:
+        if self.puct_component_balance:
+            picked, top_scores = self._sample_states_puct_component_balanced(
+                scores,
+                num_states,
+            )
+        elif num_states > 1:
             children_map = self._build_children_map()
             picked, top_scores, blocked_ids = [], [], set()
             for entry in scores:
@@ -731,21 +987,59 @@ class PUCTSampler(StateSampler):
         stats.update(_stats(sampled_values, "puct/sampled_value"))
         stats.update(_stats(sampled_timesteps, "puct/sampled_timestep"))
         stats.update(_stats(sampled_constr_lens, "puct/sampled_construction_len"))
+        if self.puct_component_balance:
+            self._puct_component_selected_counts = _sanitize_puct_component_counts(
+                self._puct_component_selected_counts
+            )
+            for surface in PUCT_COMPONENT_SURFACES:
+                suffix = _puct_component_metric_suffix(surface)
+                stats[f"puct_component_balance/last_selected/{suffix}"] = (
+                    self._last_puct_component_selected_counts[surface]
+                )
+                stats[f"puct_component_balance/available/{suffix}"] = (
+                    self._last_puct_component_available_counts[surface]
+                )
+                stats[f"puct_component_balance/selected_total/{suffix}"] = (
+                    self._puct_component_selected_counts[surface]
+                )
         return stats
 
     def get_sample_table(self) -> tuple[list[str], list[tuple]]:
         columns = ["buffer_idx", "timestep", "value", "terminal_value", "parent_value", "construction_len", "observation_len", "n", "Q", "P", "bonus", "score"]
+        if self.puct_component_balance:
+            columns.extend(
+                [
+                    "component_surface",
+                    "component_count_before",
+                    "component_q_edge",
+                    "component_bonus_to_q_edge",
+                ]
+            )
         rows = []
         if not self._last_sampled_states:
             return columns, rows
         indices = self._last_sampled_indices if len(self._last_sampled_indices) == len(self._last_sampled_states) else [-1] * len(self._last_sampled_states)
         stats = self._last_puct_stats if len(self._last_puct_stats) == len(self._last_sampled_states) else [(0, 0.0, 0.0, 0.0, 0.0)] * len(self._last_sampled_states)
-        for idx, state, (n, Q, P, bonus, score) in zip(indices, self._last_sampled_states, stats):
+        component_rows = (
+            self._last_puct_component_rows
+            if self.puct_component_balance
+            and len(self._last_puct_component_rows) == len(self._last_sampled_states)
+            else [(None, None, None, None)] * len(self._last_sampled_states)
+        )
+        for idx, state, (n, Q, P, bonus, score), component_row in zip(
+            indices,
+            self._last_sampled_states,
+            stats,
+            component_rows,
+        ):
             parent_val = state.parent_values[0] if state.parent_values else None
             constr = getattr(state, 'construction', None)
             constr_len = len(constr) if constr is not None else 0
             obs_len = len(state.observation) if state.observation else 0
-            rows.append((idx, state.timestep, state.value, 0, parent_val, constr_len, obs_len, n, Q, P, bonus, score))
+            row = (idx, state.timestep, state.value, 0, parent_val, constr_len, obs_len, n, Q, P, bonus, score)
+            if self.puct_component_balance:
+                row = row + component_row
+            rows.append(row)
         return columns, rows
 
 
@@ -756,6 +1050,8 @@ def create_sampler(
     batch_size: int = 1,
     resume_step: int | None = None,
     topk_children: int = 2,
+    puct_component_balance: bool = False,
+    puct_component_dominance_ratio: float = 2.0,
 ) -> StateSampler:
     """Factory function to create samplers. Pass the env type (from config.env_type)."""
     if not log_path:
@@ -768,6 +1064,8 @@ def create_sampler(
         batch_size=batch_size,
         resume_step=resume_step,
         topk_children=topk_children,
+        puct_component_balance=puct_component_balance,
+        puct_component_dominance_ratio=puct_component_dominance_ratio,
     )
 
 
@@ -778,6 +1076,8 @@ def get_or_create_sampler_with_default(
     batch_size: int = 1,
     resume_step: int | None = None,
     topk_children: int = 2,
+    puct_component_balance: bool = False,
+    puct_component_dominance_ratio: float = 2.0,
 ) -> StateSampler:
     """Get sampler. Initial experience is created via env_type.create_initial_state."""
     return create_sampler(
@@ -787,4 +1087,6 @@ def get_or_create_sampler_with_default(
         batch_size=batch_size,
         resume_step=resume_step,
         topk_children=topk_children,
+        puct_component_balance=puct_component_balance,
+        puct_component_dominance_ratio=puct_component_dominance_ratio,
     )
diff --git a/ttt_discover/discovery.py b/ttt_discover/discovery.py
index 6cdb434..01daf92 100644
--- a/ttt_discover/discovery.py
+++ b/ttt_discover/discovery.py
@@ -63,6 +63,8 @@ class DiscoverConfig:
     codex_initial_program_paths: tuple[str, ...] = ()
     codex_initial_pool_paths: tuple[str, ...] = ()
     codex_autonomous: bool = False
+    codex_puct_component_balance: bool = False
+    codex_puct_component_dominance_ratio: float = 2.0
 
 
 def init_ray(num_cpus_per_task: int, env_type: str):
@@ -146,6 +148,8 @@ async def discover_impl(config: DiscoverConfig):
             initial_program_paths=config.codex_initial_program_paths,
             initial_pool_paths=config.codex_initial_pool_paths,
             autonomous=config.codex_autonomous,
+            puct_component_balance=config.codex_puct_component_balance,
+            puct_component_dominance_ratio=config.codex_puct_component_dominance_ratio,
             wandb_project=config.wandb_project,
             wandb_name=config.experiment_name,
             log_path=log_path,
diff --git a/ttt_discover/rl/codex_no_finetune.py b/ttt_discover/rl/codex_no_finetune.py
index 18a0e9e..baf28d5 100644
--- a/ttt_discover/rl/codex_no_finetune.py
+++ b/ttt_discover/rl/codex_no_finetune.py
@@ -137,6 +137,8 @@ class CodexNoFinetuneConfig:
     initial_program_paths: tuple[str, ...] = ()
     initial_pool_paths: tuple[str, ...] = ()
     topk_children: int = 16
+    puct_component_balance: bool = False
+    puct_component_dominance_ratio: float = 2.0
 
     # False is TTT Discover: non-auto, many short samples. True is AutoEvolve:
     # one Codex deep-dive workspace per sample with file edits and shell access.
@@ -960,6 +962,8 @@ def _build_sampler(cfg: CodexNoFinetuneConfig, start_batch: int) -> StateSampler
         batch_size=cfg.groups_per_batch,
         resume_step=start_batch if start_batch > 0 else None,
         topk_children=cfg.topk_children,
+        puct_component_balance=cfg.puct_component_balance,
+        puct_component_dominance_ratio=cfg.puct_component_dominance_ratio,
     )
 
 
````
</details>

