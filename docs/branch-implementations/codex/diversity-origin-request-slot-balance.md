# codex/diversity-origin-request-slot-balance

## Summary

为每个 state 记录来源 request/sample slot（seed/unknown/slot_i），update_states 传入 metadata 后持久化，并按 origin slot 平衡父选择。

## Branch State

- Worktree: `/opt/tiger/discover-origin-request-slot-balance`
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

- `codex_origin_request_slot_balance`
- `codex_origin_request_slot_count`
- `clean`
- `counts`
- `origin_request_slot_balance`
- `origin_request_slot_count`

### Constants

- `ORIGIN_REQUEST_SLOT_SEED`
- `ORIGIN_REQUEST_SLOT_UNKNOWN`
- `_ORIGIN_REQUEST_SLOT_BY_STATE_KEY`
- `_ORIGIN_REQUEST_SLOT_SELECT_COUNTS_KEY`
- `_ORIGIN_REQUEST_SLOT_COUNT_KEY`

### Classes

- None

### Functions

- `_sanitize_origin_request_slot_count`
- `origin_request_slot_surfaces`
- `_valid_origin_request_slot_surface`
- `_sanitize_origin_request_slot_by_state`
- `_sanitize_origin_request_slot_select_counts`
- `_origin_request_surface_from_sample_idx`
- `classify_origin_request_slot_surface`
- `update_states`
- `_origin_request_slot_initial_state_ids`
- `_register_origin_request_slot_seed_state`
- `mark_origin_request_slot_seed_states`
- `_sanitize_origin_request_slot_state`
- `_origin_request_slot_surface`
- `_origin_request_slot_archive_counts`
- `_origin_request_slot_available_entries`
- `_select_origin_request_slot_entry`
- `_record_origin_request_slot_selection`
- `_sample_states_origin_request_slot_balanced`

## Diff Summary

- Worktree tracked shortstat: `4 files changed, 487 insertions(+), 9 deletions(-)`
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

- `repro/gpu_mode/run_0609_origin_request_slot_balance.sh (704 bytes)`
- `repro/run_discovery.py (6502 bytes)`
- `tests/test_codex_origin_request_slot_balance.py (15458 bytes)`

### Detected Test Functions

- `tests/test_codex_origin_request_slot_balance.py::test_disabled_parity_no_origin_observability_or_persistence`
- `tests/test_codex_origin_request_slot_balance.py::test_classifier_seed_unknown_slot_modulo_and_none_id`
- `tests/test_codex_origin_request_slot_balance.py::test_least_count_surface_wins_and_preserves_puct_within_surface`
- `tests/test_codex_origin_request_slot_balance.py::test_surface_tie_uses_earliest_puct_surface`
- `tests/test_codex_origin_request_slot_balance.py::test_batch_lineage_blocking_recomputes_available_surfaces`
- `tests/test_codex_origin_request_slot_balance.py::test_enabled_persistence_resume_and_save_sanitization`
- `tests/test_codex_origin_request_slot_balance.py::test_add_initial_states_marks_seed_surface`
- `tests/test_codex_origin_request_slot_balance.py::test_update_states_records_modulo_sample_idx_after_topk_only_when_enabled`
- `tests/test_codex_origin_request_slot_balance.py::test_record_failed_rollout_only_updates_puct_visits`
- `tests/test_codex_origin_request_slot_balance.py::test_update_sampler_from_results_passes_sample_idx_metadata`
- `tests/test_codex_origin_request_slot_balance.py::test_build_sampler_passes_origin_request_slot_args`
- `tests/test_codex_origin_request_slot_balance.py::test_run_discovery_dry_run_cli_defaults_enabled_override_and_validation`
- `tests/test_codex_origin_request_slot_balance.py::test_gpu_mode_wrapper_dry_run_first_and_override_path`
- `tests/test_codex_origin_request_slot_balance.py::test_classifier_helpers_only_read_state_id`

## Untracked File Snapshots

These files are not part of `git diff` yet, but they are part of the worktree implementation state recorded here.

### `repro/gpu_mode/run_0609_origin_request_slot_balance.sh`

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
  --experiment-name trimul_0609_origin_request_slot_balance_gpu2 \
  --runner codex_no_finetune \
  --num-epochs 10 \
  --groups-per-batch 1 \
  --group-size 8 \
  --num-cpus-per-task 1 \
  --eval-timeout 530 \
  --codex-backend cli \
  --codex-model-name "" \
  --codex-cli-command codex \
  --codex-cli-sandbox read-only \
  --codex-cli-timeout 600 \
  --codex-max-concurrent-requests 1 \
  --codex-origin-request-slot-balance \
  --codex-origin-request-slot-count 8 \
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
        default="gpu-mode-0609-origin-request-slot-balance",
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
        "--codex-origin-request-slot-balance",
        dest="codex_origin_request_slot_balance",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--no-codex-origin-request-slot-balance",
        dest="codex_origin_request_slot_balance",
        action="store_false",
    )
    parser.add_argument("--codex-origin-request-slot-count", type=int, default=0)
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
        "codex_origin_request_slot_balance": args.codex_origin_request_slot_balance,
        "codex_origin_request_slot_count": args.codex_origin_request_slot_count,
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

### `tests/test_codex_origin_request_slot_balance.py`

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
from ttt_discover.codex_utils.sampler import PUCTSampler, origin_request_slot_surfaces
from ttt_discover.rl import codex_no_finetune
from ttt_discover.rl.codex_no_finetune import (
    CandidateResult,
    CodexNoFinetuneConfig,
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
    count: int = 3,
    topk_children: int = 0,
) -> PUCTSampler:
    return PUCTSampler(
        file_path=str(tmp_path / "puct_sampler.json"),
        env_type=DummyEnv,
        batch_size=0,
        puct_c=0.0,
        topk_children=topk_children,
        origin_request_slot_balance=enabled,
        origin_request_slot_count=count,
    )


def saved_store(tmp_path: Path, step: int) -> dict:
    path = tmp_path / f"puct_sampler_step_{step:06d}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def ids(states: list[State]) -> list[str]:
    return [state.id for state in states]


def test_disabled_parity_no_origin_observability_or_persistence(tmp_path: Path) -> None:
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
    assert "origin_request_slot_surface" not in columns
    stats = sampler.get_sample_stats()
    assert not any(key.startswith("puct/origin_request_slot/") for key in stats)

    sampler.update_states(
        [make_state("child", 4.0)],
        [sampler._states[0]],
        save=False,
        state_metadata=[{"sample_idx": 1}],
    )
    sampler.flush(step=1)
    store = saved_store(tmp_path, 1)
    assert "origin_request_slot_by_state" not in store
    assert "origin_request_slot_select_counts" not in store
    assert "origin_request_slot_count" not in store


def test_classifier_seed_unknown_slot_modulo_and_none_id(tmp_path: Path) -> None:
    sampler = make_sampler(tmp_path, enabled=True, count=8)
    seed = make_state("seed_state", 0.0)
    sampler._states = [seed, make_state("unknown_state", 1.0)]
    sampler._initial_states = [seed]

    assert sampler._origin_request_slot_surface(seed) == "seed"
    assert sampler._origin_request_slot_surface(sampler._states[1]) == "unknown"
    assert sampler_mod._origin_request_surface_from_sample_idx(0, 8) == "slot_0"
    assert sampler_mod._origin_request_surface_from_sample_idx(1, 8) == "slot_1"
    assert sampler_mod._origin_request_surface_from_sample_idx(8, 8) == "slot_0"
    assert sampler_mod._origin_request_surface_from_sample_idx(True, 8) is None
    assert sampler_mod._origin_request_surface_from_sample_idx("1", 8) is None

    none_id_state = make_state("placeholder", 0.0)
    none_id_state.id = None
    sampler._origin_request_slot_by_state = {"None": "slot_2"}
    assert sampler._origin_request_slot_surface(none_id_state) == "slot_2"


def test_least_count_surface_wins_and_preserves_puct_within_surface(tmp_path: Path) -> None:
    sampler = make_sampler(tmp_path, enabled=True, count=2)
    sampler._states = [
        make_state("slot0_high", 100.0),
        make_state("slot1_high", 90.0),
        make_state("slot1_low", 80.0),
    ]
    sampler._origin_request_slot_by_state = {
        "slot0_high": "slot_0",
        "slot1_high": "slot_1",
        "slot1_low": "slot_1",
    }
    sampler._origin_request_slot_select_counts = {
        "seed": 0,
        "unknown": 0,
        "slot_0": 5,
        "slot_1": 0,
    }

    picked = sampler.sample_states(1)

    assert ids(picked) == ["slot1_high"]
    assert sampler._origin_request_slot_select_counts["slot_1"] == 1
    columns, rows = sampler.get_sample_table()
    assert columns[-3:] == [
        "origin_request_slot_surface",
        "origin_request_slot_select_count",
        "origin_request_slot_archive_count",
    ]
    assert rows[0][-3:] == ("slot_1", 0, 2)


def test_surface_tie_uses_earliest_puct_surface(tmp_path: Path) -> None:
    sampler = make_sampler(tmp_path, enabled=True, count=2)
    sampler._states = [
        make_state("slot1_top", 100.0),
        make_state("slot0_second", 90.0),
        make_state("slot1_low", 80.0),
    ]
    sampler._origin_request_slot_by_state = {
        "slot1_top": "slot_1",
        "slot0_second": "slot_0",
        "slot1_low": "slot_1",
    }

    picked = sampler.sample_states(1)

    assert origin_request_slot_surfaces(2) == ("seed", "unknown", "slot_0", "slot_1")
    assert ids(picked) == ["slot1_top"]
    assert sampler._origin_request_slot_select_counts["slot_1"] == 1


def test_batch_lineage_blocking_recomputes_available_surfaces(tmp_path: Path) -> None:
    sampler = make_sampler(tmp_path, enabled=True, count=2)
    root = make_state("root", 100.0)
    child = make_state("child", 99.0, parents=[{"id": "root", "timestep": 0}])
    slot1 = make_state("slot1", 80.0)
    sampler._states = [root, child, slot1]
    sampler._origin_request_slot_by_state = {
        "root": "slot_0",
        "child": "slot_0",
        "slot1": "slot_1",
    }

    picked = sampler.sample_states(2)

    assert ids(picked) == ["root", "slot1"]
    assert "child" not in ids(picked)
    assert sampler._origin_request_slot_select_counts["slot_0"] == 1
    assert sampler._origin_request_slot_select_counts["slot_1"] == 1


def test_enabled_persistence_resume_and_save_sanitization(tmp_path: Path) -> None:
    sampler = make_sampler(tmp_path, enabled=True, count=3)
    sampler._states = [make_state("a", 1.0), make_state("c", 2.0)]
    sampler._initial_states = [sampler._states[0]]
    sampler._origin_request_slot_by_state = {
        "a": "seed",
        "b": "slot_1",
        "c": "slot_9",
        "d": "bad",
        7: "slot_1",
    }
    sampler._origin_request_slot_select_counts = {
        "seed": True,
        "unknown": 1,
        "slot_0": 2,
        "slot_1": -1,
        "slot_2": 4,
        "slot_9": 99,
    }
    sampler.flush(step=1)

    store = saved_store(tmp_path, 1)
    assert store["origin_request_slot_by_state"] == {"a": "seed"}
    assert store["origin_request_slot_select_counts"] == {
        "seed": 0,
        "unknown": 1,
        "slot_0": 2,
        "slot_1": 0,
        "slot_2": 4,
    }
    assert store["origin_request_slot_count"] == 3

    resumed = PUCTSampler(
        file_path=str(tmp_path / "puct_sampler.json"),
        env_type=DummyEnv,
        batch_size=0,
        resume_step=1,
        origin_request_slot_balance=True,
        origin_request_slot_count=3,
    )
    assert resumed._origin_request_slot_by_state == {"a": "seed"}
    assert resumed._origin_request_slot_select_counts["slot_2"] == 4

    disabled = PUCTSampler(
        file_path=str(tmp_path / "puct_sampler.json"),
        env_type=DummyEnv,
        batch_size=0,
        resume_step=1,
        origin_request_slot_balance=False,
    )
    disabled.flush(step=2)
    disabled_store = saved_store(tmp_path, 2)
    assert "origin_request_slot_by_state" not in disabled_store
    assert "origin_request_slot_select_counts" not in disabled_store
    assert "origin_request_slot_count" not in disabled_store


def test_add_initial_states_marks_seed_surface(tmp_path: Path) -> None:
    sampler = make_sampler(tmp_path, enabled=True, count=2)
    pooled = make_state("pooled", 5.0)

    assert sampler.add_initial_states([pooled], save=False) == 1

    assert sampler._origin_request_slot_surface(pooled) == "seed"


def test_update_states_records_modulo_sample_idx_after_topk_only_when_enabled(
    tmp_path: Path,
) -> None:
    parent = make_state("parent", 1.0)
    children = [
        make_state("child0", 1.0),
        make_state("child1", 3.0),
        make_state("child8", 2.0),
    ]
    metadata = [{"sample_idx": 0}, {"sample_idx": 1}, {"sample_idx": 8}]

    enabled = make_sampler(tmp_path / "enabled", enabled=True, count=8, topk_children=2)
    enabled._states = [parent]
    enabled.update_states(
        children,
        [parent, parent, parent],
        save=False,
        state_metadata=metadata,
    )
    assert ids(enabled._states) == ["parent", "child1", "child8"]
    assert enabled._origin_request_slot_by_state == {
        "child1": "slot_1",
        "child8": "slot_0",
    }

    disabled = make_sampler(tmp_path / "disabled", enabled=False, count=8)
    disabled._states = [parent]
    disabled.update_states(
        [make_state("child2", 2.0)],
        [parent],
        save=False,
        state_metadata=[{"sample_idx": 2}],
    )
    assert disabled._origin_request_slot_by_state == {}


def test_record_failed_rollout_only_updates_puct_visits(tmp_path: Path) -> None:
    sampler = make_sampler(tmp_path, enabled=True, count=2)
    parent = make_state("parent", 10.0, parents=[{"id": "ancestor", "timestep": 0}])
    sampler._states = [parent]

    assert ids(sampler.sample_states(1)) == ["parent"]
    assert sampler._origin_request_slot_select_counts["unknown"] == 1

    sampler.record_failed_rollout(parent)

    assert sampler._origin_request_slot_select_counts["unknown"] == 1
    assert sampler._T == 1
    assert sampler._n["parent"] == 1
    assert sampler._n["ancestor"] == 1


def test_update_sampler_from_results_passes_sample_idx_metadata() -> None:
    class FakeSampler:
        def __init__(self) -> None:
            self.metadata = None

        def has_state(self, state: State) -> bool:
            return False

        def update_states(
            self,
            states,
            parent_states,
            save=True,
            step=None,
            state_metadata=None,
        ):
            self.metadata = state_metadata

    result = CandidateResult(
        parent_state=make_state("parent", 1.0),
        group_idx=0,
        sample_idx=7,
        prompt="p",
        response="r",
        parsed_code="c",
        reward=1.0,
        correctness=1.0,
        raw_score=1.0,
        msg="",
        metrics={},
        next_state=make_state("child", 2.0),
    )
    fake = FakeSampler()

    codex_no_finetune._update_sampler_from_results(fake, [result])

    assert fake.metadata == [{"sample_idx": 7}]
    assert result.pool_status == "sampler_filtered"


def test_build_sampler_passes_origin_request_slot_args(monkeypatch, tmp_path: Path) -> None:
    captured: dict = {}

    def fake_create_sampler(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(codex_no_finetune, "create_sampler", fake_create_sampler)
    cfg = CodexNoFinetuneConfig(
        env_type=DummyEnv,
        log_path=str(tmp_path),
        groups_per_batch=3,
        group_size=8,
        origin_request_slot_balance=True,
        origin_request_slot_count=0,
    )

    assert codex_no_finetune._build_sampler(cfg, start_batch=4) is not None
    assert captured["batch_size"] == 3
    assert captured["resume_step"] == 4
    assert captured["origin_request_slot_balance"] is True
    assert captured["origin_request_slot_count"] == 8

    cfg = CodexNoFinetuneConfig(
        env_type=DummyEnv,
        log_path=str(tmp_path),
        group_size=8,
        origin_request_slot_balance=True,
        origin_request_slot_count=5,
    )
    codex_no_finetune._build_sampler(cfg, start_batch=0)
    assert captured["resume_step"] is None
    assert captured["origin_request_slot_count"] == 5


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
    assert default["codex_origin_request_slot_balance"] is False
    assert default["codex_origin_request_slot_count"] == 0
    assert default["codex_model_name"] is None
    assert default["num_cpus_per_task"] == 1
    assert default["wandb_project"] is None

    enabled = dry_payload(
        "--dry-run",
        "--codex-origin-request-slot-balance",
        "--codex-origin-request-slot-count",
        "8",
        "--num-cpus-per-task",
        "3",
    )
    assert enabled["codex_origin_request_slot_balance"] is True
    assert enabled["codex_origin_request_slot_count"] == 8
    assert enabled["num_cpus_per_task"] == 3

    disabled = dry_payload(
        "dry-run",
        "--codex-origin-request-slot-balance",
        "--no-codex-origin-request-slot-balance",
        "--codex-origin-request-slot-count",
        "4",
    )
    assert disabled["codex_origin_request_slot_balance"] is False
    assert disabled["codex_origin_request_slot_count"] == 4

    invalid = run_dry_run(
        "dry-run",
        "--codex-origin-request-slot-count",
        "not-an-int",
        check=False,
    )
    assert invalid.returncode == 2


def test_gpu_mode_wrapper_dry_run_first_and_override_path() -> None:
    result = subprocess.run(
        [
            "repro/gpu_mode/run_0609_origin_request_slot_balance.sh",
            "--dry-run",
            "--no-codex-origin-request-slot-balance",
            "--codex-origin-request-slot-count",
            "4",
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)["discover_config"]
    assert payload["experiment_name"] == "trimul_0609_origin_request_slot_balance_gpu2"
    assert payload["group_size"] == 8
    assert payload["wandb_project"] is None
    assert payload["codex_model_name"] is None
    assert payload["codex_origin_request_slot_balance"] is False
    assert payload["codex_origin_request_slot_count"] == 4


def test_classifier_helpers_only_read_state_id() -> None:
    source = "\n".join(
        textwrap.dedent(inspect.getsource(func))
        for func in (
            sampler_mod.classify_origin_request_slot_surface,
            PUCTSampler._origin_request_slot_surface,
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
 ttt_discover/codex_utils/sampler.py   | 472 +++++++++++++++++++++++++++++++++-
 ttt_discover/discovery.py             |   4 +
 ttt_discover/rl/codex_no_finetune.py  |  16 +-
 4 files changed, 487 insertions(+), 9 deletions(-)
````
## Raw Diff

<details>
<summary>Tracked worktree patch</summary>

````diff
diff --git a/ttt_discover/codex_utils/discovery.py b/ttt_discover/codex_utils/discovery.py
index b21a80f..18c635b 100644
--- a/ttt_discover/codex_utils/discovery.py
+++ b/ttt_discover/codex_utils/discovery.py
@@ -50,6 +50,8 @@ class DiscoverConfig:
     codex_initial_program_paths: tuple[str, ...] = ()
     codex_initial_pool_paths: tuple[str, ...] = ()
     codex_autonomous: bool = False
+    codex_origin_request_slot_balance: bool = False
+    codex_origin_request_slot_count: int = 0
 
 
 def _run_codex_no_finetune(config: DiscoverConfig) -> None:
@@ -85,6 +87,8 @@ def _run_codex_no_finetune(config: DiscoverConfig) -> None:
         initial_program_paths=config.codex_initial_program_paths,
         initial_pool_paths=config.codex_initial_pool_paths,
         autonomous=config.codex_autonomous,
+        origin_request_slot_balance=config.codex_origin_request_slot_balance,
+        origin_request_slot_count=config.codex_origin_request_slot_count,
         wandb_project=config.wandb_project,
         wandb_name=experiment_name,
         log_path=log_path,
diff --git a/ttt_discover/codex_utils/sampler.py b/ttt_discover/codex_utils/sampler.py
index 5c5d4d1..49ffbb8 100644
--- a/ttt_discover/codex_utils/sampler.py
+++ b/ttt_discover/codex_utils/sampler.py
@@ -16,6 +16,118 @@ from ttt_discover.codex_utils.runtime import State, state_from_dict, to_json_ser
 
 logger = logging.getLogger(__name__)
 
+ORIGIN_REQUEST_SLOT_SEED = "seed"
+ORIGIN_REQUEST_SLOT_UNKNOWN = "unknown"
+_ORIGIN_REQUEST_SLOT_BY_STATE_KEY = "origin_request_slot_by_state"
+_ORIGIN_REQUEST_SLOT_SELECT_COUNTS_KEY = "origin_request_slot_select_counts"
+_ORIGIN_REQUEST_SLOT_COUNT_KEY = "origin_request_slot_count"
+_PuctEntry = tuple[float, float, State, int, float, float, float]
+
+
+def _sanitize_origin_request_slot_count(raw_count: Any) -> int:
+    if isinstance(raw_count, (bool, np.bool_)):
+        return 1
+    if isinstance(raw_count, (int, np.integer)) and int(raw_count) > 0:
+        return int(raw_count)
+    return 1
+
+
+def origin_request_slot_surfaces(origin_request_slot_count: int) -> tuple[str, ...]:
+    count = _sanitize_origin_request_slot_count(origin_request_slot_count)
+    return (
+        ORIGIN_REQUEST_SLOT_SEED,
+        ORIGIN_REQUEST_SLOT_UNKNOWN,
+        *(f"slot_{idx}" for idx in range(count)),
+    )
+
+
+def _valid_origin_request_slot_surface(
+    surface: Any,
+    origin_request_slot_count: int,
+) -> str | None:
+    if not isinstance(surface, str):
+        return None
+    if surface in (ORIGIN_REQUEST_SLOT_SEED, ORIGIN_REQUEST_SLOT_UNKNOWN):
+        return surface
+    prefix = "slot_"
+    if not surface.startswith(prefix):
+        return None
+    suffix = surface[len(prefix):]
+    if not suffix.isdecimal():
+        return None
+    slot_idx = int(suffix)
+    if surface != f"slot_{slot_idx}":
+        return None
+    if 0 <= slot_idx < _sanitize_origin_request_slot_count(origin_request_slot_count):
+        return surface
+    return None
+
+
+def _sanitize_origin_request_slot_by_state(
+    raw: Any,
+    origin_request_slot_count: int,
+) -> dict[str, str]:
+    if not isinstance(raw, dict):
+        return {}
+    clean: dict[str, str] = {}
+    for state_id, surface in raw.items():
+        if not isinstance(state_id, str):
+            continue
+        valid_surface = _valid_origin_request_slot_surface(
+            surface,
+            origin_request_slot_count,
+        )
+        if valid_surface is not None:
+            clean[state_id] = valid_surface
+    return clean
+
+
+def _sanitize_origin_request_slot_select_counts(
+    raw: Any,
+    origin_request_slot_count: int,
+) -> dict[str, int]:
+    raw_counts = raw if isinstance(raw, dict) else {}
+    counts: dict[str, int] = {}
+    for surface in origin_request_slot_surfaces(origin_request_slot_count):
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
+def _origin_request_surface_from_sample_idx(
+    sample_idx: Any,
+    origin_request_slot_count: int,
+) -> str | None:
+    if isinstance(sample_idx, (bool, np.bool_)):
+        return None
+    if not isinstance(sample_idx, (int, np.integer)):
+        return None
+    count = _sanitize_origin_request_slot_count(origin_request_slot_count)
+    return f"slot_{int(sample_idx) % count}"
+
+
+def classify_origin_request_slot_surface(
+    state: Any,
+    origin_request_slot_by_state: dict[str, str],
+    initial_state_ids: set[str],
+    origin_request_slot_count: int,
+) -> str:
+    state_id = str(state.id)
+    surface = _valid_origin_request_slot_surface(
+        origin_request_slot_by_state.get(state_id),
+        origin_request_slot_count,
+    )
+    if surface is not None:
+        return surface
+    if state_id in initial_state_ids:
+        return ORIGIN_REQUEST_SLOT_SEED
+    return ORIGIN_REQUEST_SLOT_UNKNOWN
+
 
 @contextmanager
 def _file_lock(lock_path: str, *, poll_s: float = 0.05, stale_s: float = 600.0):
@@ -75,7 +187,14 @@ class StateSampler(ABC):
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
 
@@ -326,6 +445,15 @@ def seed_initial_program_paths(
 
     if seed_states:
         sampler.update_states(seed_states, seed_parents, save=save)
+        mark_seed_states = getattr(
+            sampler,
+            "mark_origin_request_slot_seed_states",
+            None,
+        )
+        if mark_seed_states is not None:
+            mark_seed_states(seed_states)
+            if save:
+                sampler.flush()
         if not save:
             sampler.flush()
     return len(seed_states)
@@ -353,6 +481,8 @@ class PUCTSampler(StateSampler):
         resume_step: int | None = None,
         puct_c: float = 1.0,
         topk_children: int = 2,
+        origin_request_slot_balance: bool = False,
+        origin_request_slot_count: int = 1,
     ):
         self.file_path = file_path
         self.env_type = env_type
@@ -361,6 +491,10 @@ class PUCTSampler(StateSampler):
         self.batch_size = batch_size
         self.topk_children = topk_children
         self.puct_c = float(puct_c)
+        self.origin_request_slot_balance = bool(origin_request_slot_balance)
+        self.origin_request_slot_count = _sanitize_origin_request_slot_count(
+            origin_request_slot_count
+        )
         
         self._states: list[State] = []
         self._initial_states: list[State] = []
@@ -375,6 +509,15 @@ class PUCTSampler(StateSampler):
         self._T: int = 0
         self._last_scale: float = 1.0
         self._last_puct_stats: list[tuple[int, float, float, float, float]] = []
+        self._origin_request_slot_by_state: dict[str, str] = {}
+        self._origin_request_slot_select_counts: dict[str, int] = (
+            _sanitize_origin_request_slot_select_counts(
+                {},
+                self.origin_request_slot_count,
+            )
+        )
+        self._last_origin_request_slot_rows: list[tuple[str, int, int]] = []
+        self._last_origin_request_slot_last_selected: str | None = None
         
         if resume_step is not None:
             self._load(resume_step)
@@ -383,6 +526,7 @@ class PUCTSampler(StateSampler):
                 state = create_initial_state(self.env_type, self.problem_type)
                 self._initial_states.append(state)
                 self._states.append(state)
+                self._register_origin_request_slot_seed_state(state)
             self._save(self._current_step)
 
     def _load(self, step: int):
@@ -399,6 +543,19 @@ class PUCTSampler(StateSampler):
         self._n = store.get("puct_n", {}) or {}
         self._m = store.get("puct_m", {}) or {}
         self._T = int(store.get("puct_T", 0) or 0)
+        if self.origin_request_slot_balance:
+            self._origin_request_slot_by_state = (
+                _sanitize_origin_request_slot_by_state(
+                    store.get(_ORIGIN_REQUEST_SLOT_BY_STATE_KEY, {}),
+                    self.origin_request_slot_count,
+                )
+            )
+            self._origin_request_slot_select_counts = (
+                _sanitize_origin_request_slot_select_counts(
+                    store.get(_ORIGIN_REQUEST_SLOT_SELECT_COUNTS_KEY, {}),
+                    self.origin_request_slot_count,
+                )
+            )
 
     def _save(self, step: int):
         save_path = _sampler_file_for_step(self.file_path, step)
@@ -411,6 +568,13 @@ class PUCTSampler(StateSampler):
             "puct_m": self._m,
             "puct_T": self._T,
         }
+        if self.origin_request_slot_balance:
+            self._sanitize_origin_request_slot_state()
+            store[_ORIGIN_REQUEST_SLOT_BY_STATE_KEY] = self._origin_request_slot_by_state
+            store[_ORIGIN_REQUEST_SLOT_SELECT_COUNTS_KEY] = (
+                self._origin_request_slot_select_counts
+            )
+            store[_ORIGIN_REQUEST_SLOT_COUNT_KEY] = self.origin_request_slot_count
         with _file_lock(f"{save_path}.lock"):
             _atomic_write_json(save_path, store)
 
@@ -489,6 +653,172 @@ class PUCTSampler(StateSampler):
                     queue.append(child_id)
         return lineage
 
+    def _origin_request_slot_initial_state_ids(self) -> set[str]:
+        return {str(state.id) for state in self._initial_states}
+
+    def _register_origin_request_slot_seed_state(self, state: State) -> None:
+        if self.origin_request_slot_balance:
+            self._origin_request_slot_by_state[str(state.id)] = (
+                ORIGIN_REQUEST_SLOT_SEED
+            )
+
+    def mark_origin_request_slot_seed_states(self, states: list[State]) -> None:
+        if not self.origin_request_slot_balance:
+            return
+        live_ids = {str(state.id) for state in self._states}
+        for state in states:
+            state_id = str(state.id)
+            if state_id in live_ids:
+                self._origin_request_slot_by_state[state_id] = (
+                    ORIGIN_REQUEST_SLOT_SEED
+                )
+
+    def _sanitize_origin_request_slot_state(self) -> None:
+        if not self.origin_request_slot_balance:
+            return
+        live_ids = {str(state.id) for state in self._states}
+        by_state = _sanitize_origin_request_slot_by_state(
+            self._origin_request_slot_by_state,
+            self.origin_request_slot_count,
+        )
+        self._origin_request_slot_by_state = {
+            state_id: surface
+            for state_id, surface in by_state.items()
+            if state_id in live_ids
+        }
+        for state_id in self._origin_request_slot_initial_state_ids():
+            if state_id in live_ids:
+                self._origin_request_slot_by_state.setdefault(
+                    state_id,
+                    ORIGIN_REQUEST_SLOT_SEED,
+                )
+        self._origin_request_slot_select_counts = (
+            _sanitize_origin_request_slot_select_counts(
+                self._origin_request_slot_select_counts,
+                self.origin_request_slot_count,
+            )
+        )
+
+    def _origin_request_slot_surface(self, state: State) -> str:
+        return classify_origin_request_slot_surface(
+            state,
+            self._origin_request_slot_by_state,
+            self._origin_request_slot_initial_state_ids(),
+            self.origin_request_slot_count,
+        )
+
+    def _origin_request_slot_archive_counts(self) -> dict[str, int]:
+        counts = {
+            surface: 0
+            for surface in origin_request_slot_surfaces(self.origin_request_slot_count)
+        }
+        for state in self._states:
+            counts[self._origin_request_slot_surface(state)] += 1
+        return counts
+
+    def _origin_request_slot_available_entries(
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
+    def _select_origin_request_slot_entry(
+        self,
+        available: list[_PuctEntry],
+        planned_counts: dict[str, int],
+    ) -> _PuctEntry | None:
+        selected: _PuctEntry | None = None
+        selected_count: int | None = None
+        for entry in available:
+            surface = self._origin_request_slot_surface(entry[2])
+            count = planned_counts[surface]
+            if selected_count is None or count < selected_count:
+                selected = entry
+                selected_count = count
+        return selected
+
+    def _record_origin_request_slot_selection(
+        self,
+        entry: _PuctEntry,
+        planned_counts: dict[str, int],
+        archive_counts: dict[str, int],
+    ) -> None:
+        state = entry[2]
+        surface = self._origin_request_slot_surface(state)
+        before = planned_counts[surface]
+        planned_counts[surface] = before + 1
+        self._origin_request_slot_select_counts[surface] = planned_counts[surface]
+        self._last_origin_request_slot_last_selected = surface
+        self._last_origin_request_slot_rows.append(
+            (surface, before, archive_counts[surface])
+        )
+
+    def _sample_states_origin_request_slot_balanced(
+        self,
+        scores: list[_PuctEntry],
+        num_states: int,
+    ) -> tuple[list[State], list[_PuctEntry]]:
+        self._sanitize_origin_request_slot_state()
+        self._last_origin_request_slot_rows = []
+        self._last_origin_request_slot_last_selected = None
+        planned_counts = dict(self._origin_request_slot_select_counts)
+        archive_counts = self._origin_request_slot_archive_counts()
+        use_lineage_blocking = num_states > 1
+        children_map = self._build_children_map() if use_lineage_blocking else {}
+
+        picked: list[State] = []
+        top_scores: list[_PuctEntry] = []
+        picked_ids: set[str] = set()
+        blocked_ids: set[str] = set()
+
+        while len(picked) < num_states:
+            available = self._origin_request_slot_available_entries(
+                scores,
+                picked_ids=picked_ids,
+                blocked_ids=blocked_ids,
+                use_lineage_blocking=use_lineage_blocking,
+            )
+            entry = self._select_origin_request_slot_entry(
+                available,
+                planned_counts,
+            )
+            if entry is None:
+                break
+            state = entry[2]
+            picked.append(state)
+            top_scores.append(entry)
+            picked_ids.add(str(state.id))
+            self._record_origin_request_slot_selection(
+                entry,
+                planned_counts,
+                archive_counts,
+            )
+            if use_lineage_blocking:
+                blocked_ids.update(
+                    str(state_id)
+                    for state_id in self._get_full_lineage(state, children_map)
+                )
+
+        self._origin_request_slot_select_counts = (
+            _sanitize_origin_request_slot_select_counts(
+                planned_counts,
+                self.origin_request_slot_count,
+            )
+        )
+        return picked, top_scores
+
     def sample_states(self, num_states: int) -> list[State]:
         initial_ids = {s.id for s in self._initial_states}
         candidates = list(self._states)
@@ -501,6 +831,9 @@ class PUCTSampler(StateSampler):
             self._last_sampled_states = picked
             self._last_sampled_indices = []
             self._last_puct_stats = [(0, 0.0, 0.0, 0.0, 0.0) for _ in picked]
+            if self.origin_request_slot_balance:
+                self._last_origin_request_slot_rows = []
+                self._last_origin_request_slot_last_selected = None
             return picked
 
         vals = np.array([float(s.value if s.value is not None else float("-inf")) for s in candidates])
@@ -510,7 +843,7 @@ class PUCTSampler(StateSampler):
         P = self._compute_prior(vals, scale)
         sqrtT = np.sqrt(1.0 + self._T)
 
-        scores = []
+        scores: list[_PuctEntry] = []
         for i, s in enumerate(candidates):
             n = self._n.get(s.id, 0)
             m = self._m.get(s.id, vals[i])
@@ -521,7 +854,12 @@ class PUCTSampler(StateSampler):
 
         scores.sort(key=lambda x: (x[0], x[1]), reverse=True)
 
-        if num_states > 1:
+        if self.origin_request_slot_balance:
+            picked, top_scores = self._sample_states_origin_request_slot_balanced(
+                scores,
+                num_states,
+            )
+        elif num_states > 1:
             children_map = self._build_children_map()
             picked, top_scores, blocked_ids = [], [], set()
             for entry in scores:
@@ -548,10 +886,19 @@ class PUCTSampler(StateSampler):
 
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
 
         # Update PUCT stats for ALL states
         parent_max: dict[str, float] = {}
@@ -575,12 +922,48 @@ class PUCTSampler(StateSampler):
             return
 
         # Apply topk filter and dedup
-        states, parent_states = self._filter_topk_per_parent(states, parent_states, self.topk_children)
+        if state_metadata is None:
+            states, parent_states = self._filter_topk_per_parent(
+                states,
+                parent_states,
+                self.topk_children,
+            )
+            metadata = [{} for _ in states]
+        elif self.topk_children > 0:
+            parent_to_children: dict[
+                str,
+                list[tuple[State, State, dict[str, Any]]],
+            ] = {}
+            for child, parent, item_metadata in zip(
+                states,
+                parent_states,
+                state_metadata,
+            ):
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
+        else:
+            metadata = state_metadata
         existing = {self._get_construction_key(s) for s in self._states}
         existing.discard(None)
         
         new_states = []
-        for child, parent in zip(states, parent_states):
+        new_metadata: list[dict[str, Any]] = []
+        for child, parent, item_metadata in zip(states, parent_states, metadata):
             if child.value is None:
                 continue
             limits = getattr(self.env_type, "construction_length_limits", None)
@@ -596,6 +979,7 @@ class PUCTSampler(StateSampler):
                 continue
             self._set_parent_info(child, parent)
             new_states.append(child)
+            new_metadata.append(item_metadata)
             if key is not None:
                 existing.add(key)
 
@@ -603,6 +987,14 @@ class PUCTSampler(StateSampler):
             return
         with self._lock:
             self._states.extend(new_states)
+            if self.origin_request_slot_balance:
+                for child, item_metadata in zip(new_states, new_metadata):
+                    surface = _origin_request_surface_from_sample_idx(
+                        item_metadata.get("sample_idx"),
+                        self.origin_request_slot_count,
+                    )
+                    if surface is not None:
+                        self._origin_request_slot_by_state[str(child.id)] = surface
             if save:
                 self._finalize_and_save(step)
 
@@ -657,6 +1049,7 @@ class PUCTSampler(StateSampler):
                     state = create_initial_state(self.env_type, self.problem_type)
                     self._initial_states.append(state)
                     self._states.append(state)
+                    self._register_origin_request_slot_seed_state(state)
 
     def get_initial_states(self) -> list[State]:
         return list(self._initial_states)
@@ -698,6 +1091,8 @@ class PUCTSampler(StateSampler):
         with self._lock:
             self._states.extend(new_states)
             self._initial_states.extend(new_states)
+            for state in new_states:
+                self._register_origin_request_slot_seed_state(state)
             if save:
                 self._finalize_and_save(step)
         return len(new_states)
@@ -731,21 +1126,74 @@ class PUCTSampler(StateSampler):
         stats.update(_stats(sampled_values, "puct/sampled_value"))
         stats.update(_stats(sampled_timesteps, "puct/sampled_timestep"))
         stats.update(_stats(sampled_constr_lens, "puct/sampled_construction_len"))
+        if self.origin_request_slot_balance:
+            self._sanitize_origin_request_slot_state()
+            archive_counts = self._origin_request_slot_archive_counts()
+            stats["puct/origin_request_slot/enabled"] = 1
+            stats["puct/origin_request_slot/count"] = self.origin_request_slot_count
+            for surface in origin_request_slot_surfaces(self.origin_request_slot_count):
+                stats[f"puct/origin_request_slot/last_selected/{surface}"] = int(
+                    self._last_origin_request_slot_last_selected == surface
+                )
+                stats[f"puct/origin_request_slot/archive/{surface}"] = (
+                    archive_counts[surface]
+                )
+                stats[f"puct/origin_request_slot/select_count/{surface}"] = (
+                    self._origin_request_slot_select_counts[surface]
+                )
         return stats
 
     def get_sample_table(self) -> tuple[list[str], list[tuple]]:
         columns = ["buffer_idx", "timestep", "value", "terminal_value", "parent_value", "construction_len", "observation_len", "n", "Q", "P", "bonus", "score"]
+        if self.origin_request_slot_balance:
+            columns.extend(
+                [
+                    "origin_request_slot_surface",
+                    "origin_request_slot_select_count",
+                    "origin_request_slot_archive_count",
+                ]
+            )
         rows = []
         if not self._last_sampled_states:
             return columns, rows
         indices = self._last_sampled_indices if len(self._last_sampled_indices) == len(self._last_sampled_states) else [-1] * len(self._last_sampled_states)
         stats = self._last_puct_stats if len(self._last_puct_stats) == len(self._last_sampled_states) else [(0, 0.0, 0.0, 0.0, 0.0)] * len(self._last_sampled_states)
-        for idx, state, (n, Q, P, bonus, score) in zip(indices, self._last_sampled_states, stats):
+        if self.origin_request_slot_balance:
+            origin_request_slot_rows = (
+                self._last_origin_request_slot_rows
+                if len(self._last_origin_request_slot_rows)
+                == len(self._last_sampled_states)
+                else [
+                    (
+                        self._origin_request_slot_surface(state),
+                        self._origin_request_slot_select_counts.get(
+                            self._origin_request_slot_surface(state),
+                            0,
+                        ),
+                        self._origin_request_slot_archive_counts().get(
+                            self._origin_request_slot_surface(state),
+                            0,
+                        ),
+                    )
+                    for state in self._last_sampled_states
+                ]
+            )
+        else:
+            origin_request_slot_rows = [()] * len(self._last_sampled_states)
+        for idx, state, (n, Q, P, bonus, score), origin_request_slot_row in zip(
+            indices,
+            self._last_sampled_states,
+            stats,
+            origin_request_slot_rows,
+        ):
             parent_val = state.parent_values[0] if state.parent_values else None
             constr = getattr(state, 'construction', None)
             constr_len = len(constr) if constr is not None else 0
             obs_len = len(state.observation) if state.observation else 0
-            rows.append((idx, state.timestep, state.value, 0, parent_val, constr_len, obs_len, n, Q, P, bonus, score))
+            row = (idx, state.timestep, state.value, 0, parent_val, constr_len, obs_len, n, Q, P, bonus, score)
+            if self.origin_request_slot_balance:
+                row = row + origin_request_slot_row
+            rows.append(row)
         return columns, rows
 
 
@@ -756,6 +1204,8 @@ def create_sampler(
     batch_size: int = 1,
     resume_step: int | None = None,
     topk_children: int = 2,
+    origin_request_slot_balance: bool = False,
+    origin_request_slot_count: int = 1,
 ) -> StateSampler:
     """Factory function to create samplers. Pass the env type (from config.env_type)."""
     if not log_path:
@@ -768,6 +1218,8 @@ def create_sampler(
         batch_size=batch_size,
         resume_step=resume_step,
         topk_children=topk_children,
+        origin_request_slot_balance=origin_request_slot_balance,
+        origin_request_slot_count=origin_request_slot_count,
     )
 
 
@@ -778,6 +1230,8 @@ def get_or_create_sampler_with_default(
     batch_size: int = 1,
     resume_step: int | None = None,
     topk_children: int = 2,
+    origin_request_slot_balance: bool = False,
+    origin_request_slot_count: int = 1,
 ) -> StateSampler:
     """Get sampler. Initial experience is created via env_type.create_initial_state."""
     return create_sampler(
@@ -787,4 +1241,6 @@ def get_or_create_sampler_with_default(
         batch_size=batch_size,
         resume_step=resume_step,
         topk_children=topk_children,
+        origin_request_slot_balance=origin_request_slot_balance,
+        origin_request_slot_count=origin_request_slot_count,
     )
diff --git a/ttt_discover/discovery.py b/ttt_discover/discovery.py
index 6cdb434..611093b 100644
--- a/ttt_discover/discovery.py
+++ b/ttt_discover/discovery.py
@@ -63,6 +63,8 @@ class DiscoverConfig:
     codex_initial_program_paths: tuple[str, ...] = ()
     codex_initial_pool_paths: tuple[str, ...] = ()
     codex_autonomous: bool = False
+    codex_origin_request_slot_balance: bool = False
+    codex_origin_request_slot_count: int = 0
 
 
 def init_ray(num_cpus_per_task: int, env_type: str):
@@ -146,6 +148,8 @@ async def discover_impl(config: DiscoverConfig):
             initial_program_paths=config.codex_initial_program_paths,
             initial_pool_paths=config.codex_initial_pool_paths,
             autonomous=config.codex_autonomous,
+            origin_request_slot_balance=config.codex_origin_request_slot_balance,
+            origin_request_slot_count=config.codex_origin_request_slot_count,
             wandb_project=config.wandb_project,
             wandb_name=config.experiment_name,
             log_path=log_path,
diff --git a/ttt_discover/rl/codex_no_finetune.py b/ttt_discover/rl/codex_no_finetune.py
index 18a0e9e..71eb394 100644
--- a/ttt_discover/rl/codex_no_finetune.py
+++ b/ttt_discover/rl/codex_no_finetune.py
@@ -137,6 +137,8 @@ class CodexNoFinetuneConfig:
     initial_program_paths: tuple[str, ...] = ()
     initial_pool_paths: tuple[str, ...] = ()
     topk_children: int = 16
+    origin_request_slot_balance: bool = False
+    origin_request_slot_count: int = 0
 
     # False is TTT Discover: non-auto, many short samples. True is AutoEvolve:
     # one Codex deep-dive workspace per sample with file edits and shell access.
@@ -730,7 +732,12 @@ def _update_sampler_from_results(sampler: StateSampler, results: list[CandidateR
                 except Exception as exc:
                     logger.warning("Failed to check sampler state before update: %s", exc)
             try:
-                sampler.update_states([result.next_state], [result.parent_state], save=False)
+                sampler.update_states(
+                    [result.next_state],
+                    [result.parent_state],
+                    save=False,
+                    state_metadata=[{"sample_idx": result.sample_idx}],
+                )
                 if was_present:
                     result.pool_status = "duplicate"
                 elif has_state is not None:
@@ -953,6 +960,11 @@ async def do_sampling_only(
 
 
 def _build_sampler(cfg: CodexNoFinetuneConfig, start_batch: int) -> StateSampler:
+    effective_origin_request_slot_count = (
+        cfg.origin_request_slot_count
+        if cfg.origin_request_slot_count > 0
+        else max(1, cfg.group_size)
+    )
     return create_sampler(
         log_path=cfg.log_path,
         env_type=cfg.env_type,
@@ -960,6 +972,8 @@ def _build_sampler(cfg: CodexNoFinetuneConfig, start_batch: int) -> StateSampler
         batch_size=cfg.groups_per_batch,
         resume_step=start_batch if start_batch > 0 else None,
         topk_children=cfg.topk_children,
+        origin_request_slot_balance=cfg.origin_request_slot_balance,
+        origin_request_slot_count=effective_origin_request_slot_count,
     )
 
 
````
</details>

