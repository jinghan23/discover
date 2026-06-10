# codex/diversity-puct-backup-lift-surface-balance

## Summary

比较 PUCT backup 值 Q/m 与 state 当前 value 的 lift，分类 backup:unvisited/lifted/self/depressed，并平衡这些 PUCT 反馈表面。

## Branch State

- Worktree: `/opt/tiger/discover-puct-backup-lift-surface-balance`
- HEAD: `2234282`
- Base used for comparison: `2234282`
- Commits ahead of base: `0`
- Commits behind base: `0`
- Group: `diversity_extra`
- Implementation location: `uncommitted worktree diff`

## Implemented Behavior

- The summary above is based on reading the changed Python code, added config fields, and sampler/prompt hooks in this branch.
- For sampler branches, the common pattern is to keep baseline PUCT ordering inside a chosen surface/bucket, while changing which surface/bucket gets the next parent pick.
- Tests detected: `13` test functions across untracked test files.

## Added Markers

### Config fields

- `codex_puct_backup_lift_surface_balance`
- `codex_puct_backup_lift_eps`
- `codex_puct_backup_lift_span_frac`
- `puct_backup_lift_surface_balance`
- `puct_backup_lift_eps`
- `puct_backup_lift_span_frac`

### Constants

- `BACKUP_LIFT_SURFACES`
- `_BACKUP_LIFT_METRIC_SUFFIX`

### Classes

- `BackupLiftSurfaceInfo`

### Functions

- `_zero_backup_lift_counts`
- `_sanitize_backup_lift_counts`
- `_finite_float_or_none`
- `_sanitize_backup_lift_eps`
- `_sanitize_backup_lift_span_frac`
- `_backup_lift_tol`
- `_classify_backup_lift_surface`
- `_select_with_backup_lift_balance`

## Diff Summary

- Worktree tracked shortstat: `4 files changed, 277 insertions(+), 3 deletions(-)`
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

- `repro/gpu_mode/run_0609_puct_backup_lift_surface_balance.sh (1095 bytes)`
- `repro/run_discovery.py (5580 bytes)`
- `tests/test_codex_puct_backup_lift_surface_balance.py (14639 bytes)`

### Detected Test Functions

- `tests/test_codex_puct_backup_lift_surface_balance.py::test_disabled_parity_no_persistence_table_or_stats`
- `tests/test_codex_puct_backup_lift_surface_balance.py::test_classifier_surfaces_malformed_nonfinite_and_tolerance_sanitization`
- `tests/test_codex_puct_backup_lift_surface_balance.py::test_classifier_ignores_non_numeric_state_fields_and_child_relationships`
- `tests/test_codex_puct_backup_lift_surface_balance.py::test_equal_count_tie_uses_earliest_puct_surface`
- `tests/test_codex_puct_backup_lift_surface_balance.py::test_unequal_counts_pick_under_selected_surface_over_global_top`
- `tests/test_codex_puct_backup_lift_surface_balance.py::test_within_chosen_surface_highest_puct_wins`
- `tests/test_codex_puct_backup_lift_surface_balance.py::test_batch_lineage_blocking_recomputes_availability`
- `tests/test_codex_puct_backup_lift_surface_balance.py::test_save_load_enabled_counts_and_disabled_omits_key`
- `tests/test_codex_puct_backup_lift_surface_balance.py::test_record_failed_rollout_does_not_mutate_backup_lift_counts`
- `tests/test_codex_puct_backup_lift_surface_balance.py::test_build_sampler_passes_backup_lift_args`
- `tests/test_codex_puct_backup_lift_surface_balance.py::test_direct_dry_run_defaults_enabled_override_and_invalid_float`
- `tests/test_codex_puct_backup_lift_surface_balance.py::test_wrapper_dry_run_first_and_override`
- `tests/test_codex_puct_backup_lift_surface_balance.py::test_classifier_static_guard_only_reads_state_id_and_value`

## Untracked File Snapshots

These files are not part of `git diff` yet, but they are part of the worktree implementation state recorded here.

### `repro/gpu_mode/run_0609_puct_backup_lift_surface_balance.sh`

````bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

mode="run"
if [[ "${1:-}" == "--dry-run" ]]; then
  mode="dry-run"
  shift
elif [[ "${1:-}" == "run" || "${1:-}" == "dry-run" ]]; then
  mode="$1"
  shift
fi

cd "${REPO_ROOT}"

export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"

dry_run_args=()
if [[ "${mode}" == "dry-run" ]]; then
  dry_run_args=(--dry-run)
fi

"${PYTHON:-python}" repro/run_discovery.py \
  "${dry_run_args[@]}" \
  --experiment-name trimul_0609_puct_backup_lift_surface_balance_gpu2 \
  --problem-type trimul \
  --runner codex_no_finetune \
  --codex-backend cli \
  --codex-max-concurrent-requests 1 \
  --num-epochs 50 \
  --group-size 1 \
  --groups-per-batch 1 \
  --num-cpus-per-task 1 \
  --eval-timeout 1200 \
  --codex-puct-backup-lift-surface-balance \
  --codex-puct-backup-lift-eps 1e-9 \
  --codex-puct-backup-lift-span-frac 1e-6 \
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


def _optional_str(value: str | None) -> str | None:
    if value == "":
        return None
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch the GPUMode Codex discovery repro."
    )
    parser.add_argument("mode", nargs="?", choices=("run", "dry-run"), default="run")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--problem-type", default="trimul")
    parser.add_argument(
        "--experiment-name",
        default="trimul_0609_puct_backup_lift_surface_balance_gpu2",
    )
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument(
        "--runner",
        choices=("tinker_rl", "codex_no_finetune"),
        default="codex_no_finetune",
    )
    parser.add_argument("--model-name", default="openai/gpt-oss-120b")
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
    parser.add_argument("--codex-cli-timeout", type=float, default=None)
    parser.add_argument("--codex-max-concurrent-requests", type=int, default=1)
    parser.add_argument("--codex-autonomous", action="store_true")
    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--group-size", type=int, default=1)
    parser.add_argument("--groups-per-batch", type=int, default=1)
    parser.add_argument("--num-cpus-per-task", type=int, default=1)
    parser.add_argument("--eval-timeout", type=int, default=1200)
    parser.add_argument(
        "--codex-puct-backup-lift-surface-balance",
        dest="codex_puct_backup_lift_surface_balance",
        action="store_true",
    )
    parser.add_argument(
        "--no-codex-puct-backup-lift-surface-balance",
        dest="codex_puct_backup_lift_surface_balance",
        action="store_false",
    )
    parser.set_defaults(codex_puct_backup_lift_surface_balance=False)
    parser.add_argument("--codex-puct-backup-lift-eps", type=float, default=1e-9)
    parser.add_argument(
        "--codex-puct-backup-lift-span-frac",
        type=float,
        default=1e-6,
    )
    return parser.parse_args(argv)


def build_discover_config(args: argparse.Namespace, *, env_type: Any) -> dict[str, Any]:
    codex_cli_sandbox = args.codex_cli_sandbox
    if args.codex_autonomous and codex_cli_sandbox == "read-only":
        codex_cli_sandbox = "workspace-write"

    return {
        "env_type": env_type,
        "problem_type": args.problem_type,
        "model_name": args.model_name,
        "runner": args.runner,
        "group_size": args.group_size,
        "groups_per_batch": args.groups_per_batch,
        "num_epochs": args.num_epochs,
        "num_cpus_per_task": args.num_cpus_per_task,
        "eval_timeout": args.eval_timeout,
        "experiment_name": args.experiment_name,
        "wandb_project": _optional_str(args.wandb_project),
        "codex_backend": args.codex_backend,
        "codex_model_name": _optional_str(args.codex_model_name),
        "codex_max_output_tokens": args.codex_max_output_tokens,
        "codex_temperature": args.codex_temperature,
        "codex_cli_command": args.codex_cli_command,
        "codex_cli_sandbox": codex_cli_sandbox,
        "codex_cli_timeout": args.codex_cli_timeout,
        "codex_max_concurrent_requests": args.codex_max_concurrent_requests,
        "codex_autonomous": args.codex_autonomous,
        "codex_puct_backup_lift_surface_balance": (
            args.codex_puct_backup_lift_surface_balance
        ),
        "codex_puct_backup_lift_eps": args.codex_puct_backup_lift_eps,
        "codex_puct_backup_lift_span_frac": args.codex_puct_backup_lift_span_frac,
    }


def _json_ready_config(config: dict[str, Any]) -> dict[str, Any]:
    out = dict(config)
    env_type = out.get("env_type")
    if not isinstance(env_type, str):
        out["env_type"] = f"{env_type.__module__}.{env_type.__name__}"
    return out


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    dry_run = args.dry_run or args.mode == "dry-run"
    dry_config = build_discover_config(
        args,
        env_type="examples.gpu_mode.env.GpuModeEnv",
    )
    if dry_run:
        json.dump(
            {"mode": "dry-run", "discover_config": dry_config},
            sys.stdout,
            indent=2,
            sort_keys=True,
        )
        sys.stdout.write("\n")
        return

    from examples.gpu_mode.env import GpuModeEnv
    from ttt_discover import DiscoverConfig, discover

    config_dict = build_discover_config(args, env_type=GpuModeEnv)
    config = DiscoverConfig(**config_dict)
    json.dump(
        {"mode": "run", "discover_config": _json_ready_config(config_dict)},
        sys.stdout,
        indent=2,
        sort_keys=True,
    )
    sys.stdout.write("\n")
    discover(config)


if __name__ == "__main__":
    main()
````

### `tests/test_codex_puct_backup_lift_surface_balance.py`

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
from ttt_discover.codex_utils.sampler import (
    BACKUP_LIFT_SURFACES,
    PUCTSampler,
    _sampler_file_for_step,
)
from ttt_discover.rl.codex_no_finetune import CodexNoFinetuneConfig, _build_sampler


REPO_ROOT = Path(__file__).resolve().parents[1]


class DummyEnv:
    state_type = State
    _counter = 0

    @classmethod
    def create_initial_state(cls, problem_type: str) -> State:
        del problem_type
        cls._counter += 1
        return make_state(f"initial-{cls._counter}", 0.0)


def make_state(
    state_id: str,
    value,
    *,
    parents: list[dict] | None = None,
    code: str = "",
    construction: list | None = None,
    observation: str = "",
    timestep: int = 0,
) -> State:
    return State(
        timestep=timestep,
        construction=[] if construction is None else construction,
        code=code,
        value=value,
        parents=[] if parents is None else parents,
        id=state_id,
        observation=observation,
    )


def make_sampler(
    tmp_path: Path,
    *,
    balance: bool = False,
    eps: float = 1e-9,
    span_frac: float = 1e-6,
) -> PUCTSampler:
    return PUCTSampler(
        file_path=str(tmp_path / "puct_sampler.json"),
        env_type=DummyEnv,
        batch_size=0,
        puct_c=0.0,
        topk_children=0,
        puct_backup_lift_surface_balance=balance,
        puct_backup_lift_eps=eps,
        puct_backup_lift_span_frac=span_frac,
    )


def zero_counts() -> dict[str, int]:
    return {surface: 0 for surface in BACKUP_LIFT_SURFACES}


def run_cmd(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def test_disabled_parity_no_persistence_table_or_stats(tmp_path: Path) -> None:
    sampler = make_sampler(tmp_path, balance=False)
    sampler._states = [
        make_state("a", 3.0),
        make_state("b", 1.0),
        make_state("c", 2.0),
    ]

    picked = sampler.sample_states(2)

    assert [state.id for state in picked] == ["a", "c"]
    stats = sampler.get_sample_stats()
    assert not any(key.startswith("puct_backup_lift/") for key in stats)
    columns, rows = sampler.get_sample_table()
    assert "backup_lift_surface" not in columns
    assert len(rows) == 2

    sampler.flush(step=1)
    payload = json.loads(Path(_sampler_file_for_step(sampler.file_path, 1)).read_text())
    assert "puct_backup_lift_surface_counts" not in payload


def test_classifier_surfaces_malformed_nonfinite_and_tolerance_sanitization(
    tmp_path: Path,
) -> None:
    sampler = make_sampler(
        tmp_path,
        balance=True,
        eps=-1.0,
        span_frac=float("nan"),
    )
    assert sampler.puct_backup_lift_eps == pytest.approx(1e-9)
    assert sampler.puct_backup_lift_span_frac == pytest.approx(1e-6)
    assert sampler._backup_lift_tol(2.0) == pytest.approx(2e-6)

    sampler._n.update(
        {
            "lifted": 1,
            "self": 1,
            "depressed": 1,
            "bad-value": 1,
            "inf-value": 1,
            "bad-m": 1,
            "inf-m": 1,
        }
    )
    sampler._m.update(
        {
            "lifted": 1.1,
            "self": 1.0 + 1e-7,
            "depressed": 0.0,
            "bad-value": 1.0,
            "inf-value": 1.0,
            "bad-m": "not-a-number",
            "inf-m": float("inf"),
        }
    )

    cases = [
        ("unvisited", make_state("unvisited", "bad"), "backup:unvisited"),
        ("lifted", make_state("lifted", 1.0), "backup:lifted"),
        ("self", make_state("self", 1.0), "backup:self"),
        ("depressed", make_state("depressed", 1.0), "backup:depressed"),
        ("bad-value", make_state("bad-value", "bad"), "backup:depressed"),
        ("inf-value", make_state("inf-value", float("inf")), "backup:depressed"),
        ("bad-m", make_state("bad-m", 1.0), "backup:depressed"),
        ("inf-m", make_state("inf-m", 1.0), "backup:depressed"),
    ]

    for _, state, expected_surface in cases:
        assert sampler._classify_backup_lift_surface(state, 2.0).surface == expected_surface


def test_classifier_ignores_non_numeric_state_fields_and_child_relationships(
    tmp_path: Path,
) -> None:
    sampler = make_sampler(tmp_path, balance=True)
    sampler._n["same"] = 1
    sampler._m["same"] = 2.0
    sampler._states = [
        make_state("same", 1.0),
        make_state("child", 0.0, parents=[{"id": "same", "timestep": 0}]),
    ]
    baseline = sampler._classify_backup_lift_surface(make_state("same", 1.0), 1.0)

    variants = [
        make_state("same", 1.0, code="different code"),
        make_state("same", 1.0, construction=[{"shape": [1, 2, 3]}]),
        make_state("same", 1.0, observation="different output"),
        make_state("same", 1.0, timestep=999),
        make_state(
            "same",
            1.0,
            parents=[
                {"id": "parent", "timestep": 1},
                {"id": "grandparent", "timestep": 0},
            ],
        ),
    ]

    for state in variants:
        assert sampler._classify_backup_lift_surface(state, 1.0) == baseline


def test_equal_count_tie_uses_earliest_puct_surface(tmp_path: Path) -> None:
    sampler = make_sampler(tmp_path, balance=True)
    sampler._states = [
        make_state("top-depressed", 10.0),
        make_state("later-lifted", 0.0),
    ]
    sampler._n.update({"top-depressed": 1, "later-lifted": 1})
    sampler._m.update({"top-depressed": 9.0, "later-lifted": 2.0})

    picked = sampler.sample_states(1)

    assert [state.id for state in picked] == ["top-depressed"]
    assert sampler._backup_lift_surface_counts["backup:depressed"] == 1
    assert sampler.get_sample_table()[1][0][-4:] == (
        "backup:depressed",
        -1.0,
        pytest.approx(1e-5),
        0,
    )


def test_unequal_counts_pick_under_selected_surface_over_global_top(
    tmp_path: Path,
) -> None:
    sampler = make_sampler(tmp_path, balance=True)
    sampler._states = [
        make_state("top-depressed", 200.0),
        make_state("under-lifted", 0.0),
    ]
    sampler._n.update({"top-depressed": 1, "under-lifted": 1})
    sampler._m.update({"top-depressed": 100.0, "under-lifted": 1.0})
    sampler._backup_lift_surface_counts = zero_counts()
    sampler._backup_lift_surface_counts["backup:depressed"] = 5

    picked = sampler.sample_states(1)

    assert [state.id for state in picked] == ["under-lifted"]
    assert sampler._backup_lift_surface_counts["backup:lifted"] == 1


def test_within_chosen_surface_highest_puct_wins(tmp_path: Path) -> None:
    sampler = make_sampler(tmp_path, balance=True)
    sampler._states = [
        make_state("top-depressed", 200.0),
        make_state("lower-lifted", 0.0),
        make_state("higher-lifted", 0.0),
    ]
    sampler._n.update(
        {
            "top-depressed": 1,
            "lower-lifted": 1,
            "higher-lifted": 1,
        }
    )
    sampler._m.update(
        {
            "top-depressed": 100.0,
            "lower-lifted": 1.0,
            "higher-lifted": 2.0,
        }
    )
    sampler._backup_lift_surface_counts = zero_counts()
    sampler._backup_lift_surface_counts["backup:depressed"] = 5

    picked = sampler.sample_states(1)

    assert [state.id for state in picked] == ["higher-lifted"]


def test_batch_lineage_blocking_recomputes_availability(tmp_path: Path) -> None:
    sampler = make_sampler(tmp_path, balance=True)
    sampler._states = [
        make_state("root", 101.0),
        make_state("blocked-child", 0.0, parents=[{"id": "root", "timestep": 0}]),
        make_state("available-self", 80.0),
    ]
    sampler._n.update({"root": 1, "blocked-child": 1, "available-self": 1})
    sampler._m.update({"root": 100.0, "blocked-child": 90.0, "available-self": 80.0})

    picked = sampler.sample_states(2)

    assert [state.id for state in picked] == ["root", "available-self"]
    assert sampler._last_backup_lift_sampled_counts["backup:depressed"] == 1
    assert sampler._last_backup_lift_sampled_counts["backup:self"] == 1


def test_save_load_enabled_counts_and_disabled_omits_key(tmp_path: Path) -> None:
    sampler = make_sampler(tmp_path, balance=True)
    sampler._states = [make_state("a", 1.0)]
    sampler.sample_states(1)
    sampler.flush(step=1)

    step_path = Path(_sampler_file_for_step(sampler.file_path, 1))
    payload = json.loads(step_path.read_text())
    assert payload["puct_backup_lift_surface_counts"] == {
        "backup:unvisited": 1,
        "backup:lifted": 0,
        "backup:self": 0,
        "backup:depressed": 0,
    }

    loaded = PUCTSampler(
        file_path=sampler.file_path,
        env_type=DummyEnv,
        batch_size=0,
        resume_step=1,
        puct_backup_lift_surface_balance=True,
    )
    assert loaded._backup_lift_surface_counts == payload[
        "puct_backup_lift_surface_counts"
    ]

    payload["puct_backup_lift_surface_counts"] = {
        "backup:unvisited": -1,
        "backup:lifted": True,
        "backup:self": 3.2,
        "backup:depressed": 4,
    }
    step_path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = PUCTSampler(
        file_path=sampler.file_path,
        env_type=DummyEnv,
        batch_size=0,
        resume_step=1,
        puct_backup_lift_surface_balance=True,
    )
    assert loaded._backup_lift_surface_counts == {
        "backup:unvisited": 0,
        "backup:lifted": 0,
        "backup:self": 0,
        "backup:depressed": 4,
    }

    disabled = PUCTSampler(
        file_path=sampler.file_path,
        env_type=DummyEnv,
        batch_size=0,
        resume_step=1,
        puct_backup_lift_surface_balance=False,
    )
    assert disabled._backup_lift_surface_counts == zero_counts()
    disabled.flush(step=2)
    disabled_payload = json.loads(
        Path(_sampler_file_for_step(sampler.file_path, 2)).read_text()
    )
    assert "puct_backup_lift_surface_counts" not in disabled_payload


def test_record_failed_rollout_does_not_mutate_backup_lift_counts(
    tmp_path: Path,
) -> None:
    sampler = make_sampler(tmp_path, balance=True)
    sampler._backup_lift_surface_counts = {
        "backup:unvisited": 1,
        "backup:lifted": 2,
        "backup:self": 3,
        "backup:depressed": 4,
    }
    before = dict(sampler._backup_lift_surface_counts)

    sampler.record_failed_rollout(
        make_state("parent", 1.0, parents=[{"id": "ancestor", "timestep": 0}])
    )

    assert sampler._backup_lift_surface_counts == before
    assert sampler._n["parent"] == 1
    assert sampler._n["ancestor"] == 1


def test_build_sampler_passes_backup_lift_args(tmp_path: Path) -> None:
    cfg = CodexNoFinetuneConfig(
        env_type=DummyEnv,
        groups_per_batch=0,
        log_path=str(tmp_path),
        topk_children=7,
        puct_backup_lift_surface_balance=True,
        puct_backup_lift_eps=-1.0,
        puct_backup_lift_span_frac=float("inf"),
    )

    sampler = _build_sampler(cfg, start_batch=0)

    assert isinstance(sampler, PUCTSampler)
    assert sampler.topk_children == 7
    assert sampler.puct_backup_lift_surface_balance is True
    assert sampler.puct_backup_lift_eps == pytest.approx(1e-9)
    assert sampler.puct_backup_lift_span_frac == pytest.approx(1e-6)


def test_direct_dry_run_defaults_enabled_override_and_invalid_float() -> None:
    default = run_cmd([sys.executable, "repro/run_discovery.py", "--dry-run"])
    assert default.returncode == 0, default.stderr
    payload = json.loads(default.stdout)
    cfg = payload["discover_config"]
    assert cfg["codex_model_name"] is None
    assert cfg["wandb_project"] is None
    assert cfg["num_cpus_per_task"] == 1
    assert cfg["codex_puct_backup_lift_surface_balance"] is False

    enabled = run_cmd(
        [
            sys.executable,
            "repro/run_discovery.py",
            "--dry-run",
            "--codex-puct-backup-lift-surface-balance",
            "--codex-puct-backup-lift-eps",
            "5e-8",
            "--codex-puct-backup-lift-span-frac",
            "2e-4",
            "--num-cpus-per-task",
            "6",
        ]
    )
    assert enabled.returncode == 0, enabled.stderr
    cfg = json.loads(enabled.stdout)["discover_config"]
    assert cfg["codex_puct_backup_lift_surface_balance"] is True
    assert cfg["codex_puct_backup_lift_eps"] == pytest.approx(5e-8)
    assert cfg["codex_puct_backup_lift_span_frac"] == pytest.approx(2e-4)
    assert cfg["num_cpus_per_task"] == 6

    invalid = run_cmd(
        [
            sys.executable,
            "repro/run_discovery.py",
            "--dry-run",
            "--codex-puct-backup-lift-eps",
            "invalid",
        ]
    )
    assert invalid.returncode != 0
    assert "invalid float" in invalid.stderr


def test_wrapper_dry_run_first_and_override() -> None:
    proc = run_cmd(
        [
            "bash",
            "repro/gpu_mode/run_0609_puct_backup_lift_surface_balance.sh",
            "--dry-run",
            "--no-codex-puct-backup-lift-surface-balance",
            "--codex-puct-backup-lift-eps",
            "2e-9",
            "--num-cpus-per-task",
            "3",
        ]
    )
    assert proc.returncode == 0, proc.stderr
    cfg = json.loads(proc.stdout)["discover_config"]
    assert cfg["experiment_name"] == "trimul_0609_puct_backup_lift_surface_balance_gpu2"
    assert cfg["codex_model_name"] is None
    assert cfg["wandb_project"] is None
    assert cfg["codex_puct_backup_lift_surface_balance"] is False
    assert cfg["codex_puct_backup_lift_eps"] == pytest.approx(2e-9)
    assert cfg["codex_puct_backup_lift_span_frac"] == pytest.approx(1e-6)
    assert cfg["num_cpus_per_task"] == 3


def test_classifier_static_guard_only_reads_state_id_and_value() -> None:
    source = textwrap.dedent(
        inspect.getsource(PUCTSampler._classify_backup_lift_surface)
    )
    tree = ast.parse(source)
    attrs: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "state"
        ):
            attrs.add(node.attr)

    assert attrs == {"id", "value"}

    run_discovery = (REPO_ROOT / "repro/run_discovery.py").read_text()
    assert run_discovery.index("if dry_run:") < run_discovery.index(
        "from ttt_discover"
    )
````

## Diff Stat

### Worktree Tracked Diff Stat

````text
 ttt_discover/codex_utils/discovery.py |   8 ++
 ttt_discover/codex_utils/sampler.py   | 258 +++++++++++++++++++++++++++++++++-
 ttt_discover/discovery.py             |   8 ++
 ttt_discover/rl/codex_no_finetune.py  |   6 +
 4 files changed, 277 insertions(+), 3 deletions(-)
````
## Raw Diff

<details>
<summary>Tracked worktree patch</summary>

````diff
diff --git a/ttt_discover/codex_utils/discovery.py b/ttt_discover/codex_utils/discovery.py
index b21a80f..5257303 100644
--- a/ttt_discover/codex_utils/discovery.py
+++ b/ttt_discover/codex_utils/discovery.py
@@ -50,6 +50,9 @@ class DiscoverConfig:
     codex_initial_program_paths: tuple[str, ...] = ()
     codex_initial_pool_paths: tuple[str, ...] = ()
     codex_autonomous: bool = False
+    codex_puct_backup_lift_surface_balance: bool = False
+    codex_puct_backup_lift_eps: float = 1e-9
+    codex_puct_backup_lift_span_frac: float = 1e-6
 
 
 def _run_codex_no_finetune(config: DiscoverConfig) -> None:
@@ -85,6 +88,11 @@ def _run_codex_no_finetune(config: DiscoverConfig) -> None:
         initial_program_paths=config.codex_initial_program_paths,
         initial_pool_paths=config.codex_initial_pool_paths,
         autonomous=config.codex_autonomous,
+        puct_backup_lift_surface_balance=(
+            config.codex_puct_backup_lift_surface_balance
+        ),
+        puct_backup_lift_eps=config.codex_puct_backup_lift_eps,
+        puct_backup_lift_span_frac=config.codex_puct_backup_lift_span_frac,
         wandb_project=config.wandb_project,
         wandb_name=experiment_name,
         log_path=log_path,
diff --git a/ttt_discover/codex_utils/sampler.py b/ttt_discover/codex_utils/sampler.py
index 5c5d4d1..655d303 100644
--- a/ttt_discover/codex_utils/sampler.py
+++ b/ttt_discover/codex_utils/sampler.py
@@ -2,8 +2,10 @@
 from __future__ import annotations
 from abc import ABC, abstractmethod
 from contextlib import contextmanager
+from dataclasses import dataclass
 import json
 import logging
+import math
 import os
 from pathlib import Path
 import threading
@@ -16,6 +18,76 @@ from ttt_discover.codex_utils.runtime import State, state_from_dict, to_json_ser
 
 logger = logging.getLogger(__name__)
 
+BACKUP_LIFT_SURFACES = (
+    "backup:unvisited",
+    "backup:lifted",
+    "backup:self",
+    "backup:depressed",
+)
+
+_BACKUP_LIFT_METRIC_SUFFIX = {
+    "backup:unvisited": "backup_unvisited",
+    "backup:lifted": "backup_lifted",
+    "backup:self": "backup_self",
+    "backup:depressed": "backup_depressed",
+}
+
+
+@dataclass(frozen=True)
+class BackupLiftSurfaceInfo:
+    surface: str
+    lift: float | None
+    tol: float
+
+
+def _zero_backup_lift_counts() -> dict[str, int]:
+    return {surface: 0 for surface in BACKUP_LIFT_SURFACES}
+
+
+def _sanitize_backup_lift_counts(raw: Any) -> dict[str, int]:
+    counts = _zero_backup_lift_counts()
+    if not isinstance(raw, dict):
+        return counts
+    for surface in BACKUP_LIFT_SURFACES:
+        value = raw.get(surface, 0)
+        if (
+            isinstance(value, (bool, np.bool_))
+            or not isinstance(value, (int, np.integer))
+            or int(value) < 0
+        ):
+            counts[surface] = 0
+        else:
+            counts[surface] = int(value)
+    return counts
+
+
+def _finite_float_or_none(value: Any) -> float | None:
+    try:
+        out = float(value)
+    except (TypeError, ValueError):
+        return None
+    if not math.isfinite(out):
+        return None
+    return out
+
+
+def _sanitize_backup_lift_eps(value: Any) -> float:
+    try:
+        eps = float(value)
+    except (TypeError, ValueError):
+        return 1e-9
+    return eps if math.isfinite(eps) and eps > 0 else 1e-9
+
+
+def _sanitize_backup_lift_span_frac(value: Any) -> float:
+    try:
+        span_frac = float(value)
+    except (TypeError, ValueError):
+        return 1e-6
+    if span_frac < 0 or not math.isfinite(span_frac):
+        return 1e-6
+    return span_frac
+
 
 @contextmanager
 def _file_lock(lock_path: str, *, poll_s: float = 0.05, stale_s: float = 600.0):
@@ -353,6 +425,9 @@ class PUCTSampler(StateSampler):
         resume_step: int | None = None,
         puct_c: float = 1.0,
         topk_children: int = 2,
+        puct_backup_lift_surface_balance: bool = False,
+        puct_backup_lift_eps: float = 1e-9,
+        puct_backup_lift_span_frac: float = 1e-6,
     ):
         self.file_path = file_path
         self.env_type = env_type
@@ -361,6 +436,11 @@ class PUCTSampler(StateSampler):
         self.batch_size = batch_size
         self.topk_children = topk_children
         self.puct_c = float(puct_c)
+        self.puct_backup_lift_surface_balance = bool(puct_backup_lift_surface_balance)
+        self.puct_backup_lift_eps = _sanitize_backup_lift_eps(puct_backup_lift_eps)
+        self.puct_backup_lift_span_frac = _sanitize_backup_lift_span_frac(
+            puct_backup_lift_span_frac
+        )
         
         self._states: list[State] = []
         self._initial_states: list[State] = []
@@ -375,6 +455,9 @@ class PUCTSampler(StateSampler):
         self._T: int = 0
         self._last_scale: float = 1.0
         self._last_puct_stats: list[tuple[int, float, float, float, float]] = []
+        self._backup_lift_surface_counts: dict[str, int] = _zero_backup_lift_counts()
+        self._last_backup_lift_sampled_counts: dict[str, int] = _zero_backup_lift_counts()
+        self._last_backup_lift_stats: list[tuple[str, float | None, float, int]] = []
         
         if resume_step is not None:
             self._load(resume_step)
@@ -399,6 +482,12 @@ class PUCTSampler(StateSampler):
         self._n = store.get("puct_n", {}) or {}
         self._m = store.get("puct_m", {}) or {}
         self._T = int(store.get("puct_T", 0) or 0)
+        if self.puct_backup_lift_surface_balance:
+            self._backup_lift_surface_counts = _sanitize_backup_lift_counts(
+                store.get("puct_backup_lift_surface_counts")
+            )
+        else:
+            self._backup_lift_surface_counts = _zero_backup_lift_counts()
 
     def _save(self, step: int):
         save_path = _sampler_file_for_step(self.file_path, step)
@@ -411,6 +500,12 @@ class PUCTSampler(StateSampler):
             "puct_m": self._m,
             "puct_T": self._T,
         }
+        if self.puct_backup_lift_surface_balance:
+            counts = _sanitize_backup_lift_counts(self._backup_lift_surface_counts)
+            store["puct_backup_lift_surface_counts"] = {
+                surface: int(counts.get(surface, 0))
+                for surface in BACKUP_LIFT_SURFACES
+            }
         with _file_lock(f"{save_path}.lock"):
             _atomic_write_json(save_path, store)
 
@@ -489,6 +584,111 @@ class PUCTSampler(StateSampler):
                     queue.append(child_id)
         return lineage
 
+    def _backup_lift_tol(self, scale: float) -> float:
+        try:
+            scale_value = float(scale)
+        except (TypeError, ValueError):
+            scale_value = 1.0
+        if not math.isfinite(scale_value):
+            scale_value = 1.0
+        return max(
+            self.puct_backup_lift_eps,
+            self.puct_backup_lift_span_frac * scale_value,
+        )
+
+    def _classify_backup_lift_surface(
+        self,
+        state: State,
+        scale: float,
+    ) -> BackupLiftSurfaceInfo:
+        raw_state_id = state.id
+        state_id = str(raw_state_id)
+        n = self._n.get(state_id, self._n.get(raw_state_id, 0))
+        tol = self._backup_lift_tol(scale)
+        if n == 0:
+            return BackupLiftSurfaceInfo("backup:unvisited", None, tol)
+
+        value = _finite_float_or_none(state.value)
+        m = _finite_float_or_none(self._m.get(state_id, self._m.get(raw_state_id, value)))
+        if value is None or m is None:
+            return BackupLiftSurfaceInfo("backup:depressed", None, tol)
+
+        lift = m - value
+        if lift > tol:
+            return BackupLiftSurfaceInfo("backup:lifted", lift, tol)
+        if lift < -tol:
+            return BackupLiftSurfaceInfo("backup:depressed", lift, tol)
+        return BackupLiftSurfaceInfo("backup:self", lift, tol)
+
+    def _select_with_backup_lift_balance(
+        self,
+        scores: list[tuple[float, float, State, int, float, float, float]],
+        num_states: int,
+        scale: float,
+    ) -> tuple[list[State], list[tuple[float, float, State, int, float, float, float]]]:
+        use_lineage_blocking = num_states > 1
+        children_map = self._build_children_map() if use_lineage_blocking else {}
+        picked: list[State] = []
+        top_scores: list[tuple[float, float, State, int, float, float, float]] = []
+        picked_ids: set[str] = set()
+        blocked_ids: set[str] = set()
+        planned_counts = _sanitize_backup_lift_counts(self._backup_lift_surface_counts)
+        sampled_counts = _zero_backup_lift_counts()
+        backup_lift_stats: list[tuple[str, float | None, float, int]] = []
+
+        while len(picked) < num_states:
+            eligible_by_surface: dict[
+                str,
+                list[tuple[int, tuple[float, float, State, int, float, float, float], BackupLiftSurfaceInfo]],
+            ] = {surface: [] for surface in BACKUP_LIFT_SURFACES}
+            for baseline_idx, entry in enumerate(scores):
+                state = entry[2]
+                state_id = str(state.id)
+                if state_id in picked_ids or state_id in blocked_ids:
+                    continue
+                surface_info = self._classify_backup_lift_surface(state, scale)
+                eligible_by_surface[surface_info.surface].append(
+                    (baseline_idx, entry, surface_info)
+                )
+
+            available_surfaces = [
+                surface
+                for surface in BACKUP_LIFT_SURFACES
+                if eligible_by_surface[surface]
+            ]
+            if not available_surfaces:
+                break
+
+            chosen_surface = min(
+                available_surfaces,
+                key=lambda surface: (
+                    planned_counts.get(surface, 0),
+                    eligible_by_surface[surface][0][0],
+                ),
+            )
+            _, chosen_entry, surface_info = eligible_by_surface[chosen_surface][0]
+            state = chosen_entry[2]
+            count_before = planned_counts.get(chosen_surface, 0)
+
+            picked.append(state)
+            top_scores.append(chosen_entry)
+            picked_ids.add(str(state.id))
+            planned_counts[chosen_surface] = count_before + 1
+            sampled_counts[chosen_surface] += 1
+            backup_lift_stats.append(
+                (surface_info.surface, surface_info.lift, surface_info.tol, count_before)
+            )
+            if use_lineage_blocking:
+                blocked_ids.update(
+                    str(state_id)
+                    for state_id in self._get_full_lineage(state, children_map)
+                )
+
+        self._backup_lift_surface_counts = _sanitize_backup_lift_counts(planned_counts)
+        self._last_backup_lift_sampled_counts = sampled_counts
+        self._last_backup_lift_stats = backup_lift_stats
+        return picked, top_scores
+
     def sample_states(self, num_states: int) -> list[State]:
         initial_ids = {s.id for s in self._initial_states}
         candidates = list(self._states)
@@ -501,6 +701,8 @@ class PUCTSampler(StateSampler):
             self._last_sampled_states = picked
             self._last_sampled_indices = []
             self._last_puct_stats = [(0, 0.0, 0.0, 0.0, 0.0) for _ in picked]
+            self._last_backup_lift_sampled_counts = _zero_backup_lift_counts()
+            self._last_backup_lift_stats = []
             return picked
 
         vals = np.array([float(s.value if s.value is not None else float("-inf")) for s in candidates])
@@ -521,7 +723,13 @@ class PUCTSampler(StateSampler):
 
         scores.sort(key=lambda x: (x[0], x[1]), reverse=True)
 
-        if num_states > 1:
+        if self.puct_backup_lift_surface_balance:
+            picked, top_scores = self._select_with_backup_lift_balance(
+                scores,
+                num_states,
+                scale,
+            )
+        elif num_states > 1:
             children_map = self._build_children_map()
             picked, top_scores, blocked_ids = [], [], set()
             for entry in scores:
@@ -537,6 +745,10 @@ class PUCTSampler(StateSampler):
             top_scores = scores[:num_states]
             picked = [t[2] for t in top_scores]
 
+        if not self.puct_backup_lift_surface_balance:
+            self._last_backup_lift_sampled_counts = _zero_backup_lift_counts()
+            self._last_backup_lift_stats = []
+
         state_id_to_idx = {s.id: i for i, s in enumerate(self._states)}
         self._last_sampled_states = picked
         self._last_sampled_indices = [state_id_to_idx.get(s.id, -1) for s in picked]
@@ -731,21 +943,49 @@ class PUCTSampler(StateSampler):
         stats.update(_stats(sampled_values, "puct/sampled_value"))
         stats.update(_stats(sampled_timesteps, "puct/sampled_timestep"))
         stats.update(_stats(sampled_constr_lens, "puct/sampled_construction_len"))
+        if self.puct_backup_lift_surface_balance:
+            stats["puct_backup_lift/enabled"] = 1
+            for surface in BACKUP_LIFT_SURFACES:
+                suffix = _BACKUP_LIFT_METRIC_SUFFIX[surface]
+                stats[f"puct_backup_lift/count_{suffix}"] = int(
+                    self._backup_lift_surface_counts.get(surface, 0)
+                )
+                stats[f"puct_backup_lift/sampled_{suffix}"] = int(
+                    self._last_backup_lift_sampled_counts.get(surface, 0)
+                )
         return stats
 
     def get_sample_table(self) -> tuple[list[str], list[tuple]]:
         columns = ["buffer_idx", "timestep", "value", "terminal_value", "parent_value", "construction_len", "observation_len", "n", "Q", "P", "bonus", "score"]
+        if self.puct_backup_lift_surface_balance:
+            columns.extend(
+                [
+                    "backup_lift_surface",
+                    "backup_lift",
+                    "backup_lift_tol",
+                    "backup_lift_surface_count_before",
+                ]
+            )
         rows = []
         if not self._last_sampled_states:
             return columns, rows
         indices = self._last_sampled_indices if len(self._last_sampled_indices) == len(self._last_sampled_states) else [-1] * len(self._last_sampled_states)
         stats = self._last_puct_stats if len(self._last_puct_stats) == len(self._last_sampled_states) else [(0, 0.0, 0.0, 0.0, 0.0)] * len(self._last_sampled_states)
-        for idx, state, (n, Q, P, bonus, score) in zip(indices, self._last_sampled_states, stats):
+        backup_lift_stats = self._last_backup_lift_stats
+        if len(backup_lift_stats) != len(self._last_sampled_states):
+            backup_lift_stats = [
+                ("", None, self._backup_lift_tol(self._last_scale), 0)
+                for _ in self._last_sampled_states
+            ]
+        for row_idx, (idx, state, (n, Q, P, bonus, score)) in enumerate(zip(indices, self._last_sampled_states, stats)):
             parent_val = state.parent_values[0] if state.parent_values else None
             constr = getattr(state, 'construction', None)
             constr_len = len(constr) if constr is not None else 0
             obs_len = len(state.observation) if state.observation else 0
-            rows.append((idx, state.timestep, state.value, 0, parent_val, constr_len, obs_len, n, Q, P, bonus, score))
+            row = (idx, state.timestep, state.value, 0, parent_val, constr_len, obs_len, n, Q, P, bonus, score)
+            if self.puct_backup_lift_surface_balance:
+                row = row + backup_lift_stats[row_idx]
+            rows.append(row)
         return columns, rows
 
 
@@ -756,6 +996,9 @@ def create_sampler(
     batch_size: int = 1,
     resume_step: int | None = None,
     topk_children: int = 2,
+    puct_backup_lift_surface_balance: bool = False,
+    puct_backup_lift_eps: float = 1e-9,
+    puct_backup_lift_span_frac: float = 1e-6,
 ) -> StateSampler:
     """Factory function to create samplers. Pass the env type (from config.env_type)."""
     if not log_path:
@@ -768,6 +1011,9 @@ def create_sampler(
         batch_size=batch_size,
         resume_step=resume_step,
         topk_children=topk_children,
+        puct_backup_lift_surface_balance=puct_backup_lift_surface_balance,
+        puct_backup_lift_eps=puct_backup_lift_eps,
+        puct_backup_lift_span_frac=puct_backup_lift_span_frac,
     )
 
 
@@ -778,6 +1024,9 @@ def get_or_create_sampler_with_default(
     batch_size: int = 1,
     resume_step: int | None = None,
     topk_children: int = 2,
+    puct_backup_lift_surface_balance: bool = False,
+    puct_backup_lift_eps: float = 1e-9,
+    puct_backup_lift_span_frac: float = 1e-6,
 ) -> StateSampler:
     """Get sampler. Initial experience is created via env_type.create_initial_state."""
     return create_sampler(
@@ -787,4 +1036,7 @@ def get_or_create_sampler_with_default(
         batch_size=batch_size,
         resume_step=resume_step,
         topk_children=topk_children,
+        puct_backup_lift_surface_balance=puct_backup_lift_surface_balance,
+        puct_backup_lift_eps=puct_backup_lift_eps,
+        puct_backup_lift_span_frac=puct_backup_lift_span_frac,
     )
diff --git a/ttt_discover/discovery.py b/ttt_discover/discovery.py
index 6cdb434..924b635 100644
--- a/ttt_discover/discovery.py
+++ b/ttt_discover/discovery.py
@@ -63,6 +63,9 @@ class DiscoverConfig:
     codex_initial_program_paths: tuple[str, ...] = ()
     codex_initial_pool_paths: tuple[str, ...] = ()
     codex_autonomous: bool = False
+    codex_puct_backup_lift_surface_balance: bool = False
+    codex_puct_backup_lift_eps: float = 1e-9
+    codex_puct_backup_lift_span_frac: float = 1e-6
 
 
 def init_ray(num_cpus_per_task: int, env_type: str):
@@ -146,6 +149,11 @@ async def discover_impl(config: DiscoverConfig):
             initial_program_paths=config.codex_initial_program_paths,
             initial_pool_paths=config.codex_initial_pool_paths,
             autonomous=config.codex_autonomous,
+            puct_backup_lift_surface_balance=(
+                config.codex_puct_backup_lift_surface_balance
+            ),
+            puct_backup_lift_eps=config.codex_puct_backup_lift_eps,
+            puct_backup_lift_span_frac=config.codex_puct_backup_lift_span_frac,
             wandb_project=config.wandb_project,
             wandb_name=config.experiment_name,
             log_path=log_path,
diff --git a/ttt_discover/rl/codex_no_finetune.py b/ttt_discover/rl/codex_no_finetune.py
index 18a0e9e..d6abfe9 100644
--- a/ttt_discover/rl/codex_no_finetune.py
+++ b/ttt_discover/rl/codex_no_finetune.py
@@ -137,6 +137,9 @@ class CodexNoFinetuneConfig:
     initial_program_paths: tuple[str, ...] = ()
     initial_pool_paths: tuple[str, ...] = ()
     topk_children: int = 16
+    puct_backup_lift_surface_balance: bool = False
+    puct_backup_lift_eps: float = 1e-9
+    puct_backup_lift_span_frac: float = 1e-6
 
     # False is TTT Discover: non-auto, many short samples. True is AutoEvolve:
     # one Codex deep-dive workspace per sample with file edits and shell access.
@@ -960,6 +963,9 @@ def _build_sampler(cfg: CodexNoFinetuneConfig, start_batch: int) -> StateSampler
         batch_size=cfg.groups_per_batch,
         resume_step=start_batch if start_batch > 0 else None,
         topk_children=cfg.topk_children,
+        puct_backup_lift_surface_balance=cfg.puct_backup_lift_surface_balance,
+        puct_backup_lift_eps=cfg.puct_backup_lift_eps,
+        puct_backup_lift_span_frac=cfg.puct_backup_lift_span_frac,
     )
 
 
````
</details>

