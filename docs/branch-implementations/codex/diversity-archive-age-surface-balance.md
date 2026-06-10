# codex/diversity-archive-age-surface-balance

## Summary

新增 puct_surface_balance=archive_age，把 archive 中状态按 birth timestep 划成 seed/fresh/warm/cold，并用持久计数轮转选择低采样 surface。

## Branch State

- Worktree: `/opt/tiger/discover-archive-age-surface-balance`
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

- `codex_puct_surface_balance`
- `puct_surface_balance`

### Constants

- `ARCHIVE_AGE_SURFACES`
- `_PUCT_SURFACE_BALANCE_MODES`

### Classes

- None

### Functions

- `_validate_puct_surface_balance`
- `_archive_birth_step`
- `_archive_max_birth`
- `_archive_age_surface_for_state`
- `_sanitize_archive_age_sample_counts`
- `_choose_archive_age_surface`
- `_sample_states_archive_age`

## Diff Summary

- Worktree tracked shortstat: `4 files changed, 188 insertions(+), 2 deletions(-)`
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

- `repro/gpu_mode/run_0609_archive_age_balance.sh (461 bytes)`
- `repro/run_discovery.py (4825 bytes)`
- `tests/test_codex_archive_age_balance.py (13512 bytes)`

### Detected Test Functions

- `tests/test_codex_archive_age_balance.py::test_disabled_parity_has_no_archive_age_surface_observability_or_persistence`
- `tests/test_codex_archive_age_balance.py::test_archive_age_classifier_boundaries_and_invalid_timesteps`
- `tests/test_codex_archive_age_balance.py::test_least_count_available_surface_beats_higher_puct_oversampled_surface`
- `tests/test_codex_archive_age_balance.py::test_unavailable_lower_count_surfaces_are_ignored`
- `tests/test_codex_archive_age_balance.py::test_rotated_tie_order_uses_total_sample_count_modulo_surface_count`
- `tests/test_codex_archive_age_balance.py::test_within_selected_surface_highest_baseline_puct_candidate_wins`
- `tests/test_codex_archive_age_balance.py::test_batch_lineage_blocking_recomputes_surface_availability`
- `tests/test_codex_archive_age_balance.py::test_enabled_persistence_resume_sanitization_and_disabled_resume_ignore`
- `tests/test_codex_archive_age_balance.py::test_record_failed_rollout_updates_puct_visits_without_archive_age_double_count`
- `tests/test_codex_archive_age_balance.py::test_cli_dry_run_surface_balance_defaults_enabled_invalid_and_wrapper_override`
- `tests/test_codex_archive_age_balance.py::test_build_sampler_passes_surface_balance`
- `tests/test_codex_archive_age_balance.py::test_archive_age_classifier_helpers_only_read_timestep`

## Untracked File Snapshots

These files are not part of `git diff` yet, but they are part of the worktree implementation state recorded here.

### `repro/gpu_mode/run_0609_archive_age_balance.sh`

````bash
#!/usr/bin/env bash
set -euo pipefail

MODE="run"
if [[ "${1:-}" == "run" || "${1:-}" == "dry-run" ]]; then
  MODE="$1"
  shift
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

python "${REPO_ROOT}/repro/run_discovery.py" "${MODE}" \
  --experiment-name gpu-mode-0609-archive-age-balance \
  --problem-type trimul \
  --runner codex_no_finetune \
  --codex-puct-surface-balance archive_age \
  "$@"
````

### `repro/run_discovery.py`

````python
#!/usr/bin/env python
"""Repro entrypoint for Codex discovery runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run or dry-run a discovery experiment.")
    parser.add_argument("mode", nargs="?", choices=("run", "dry-run"), default="run")
    parser.add_argument("--dry-run", action="store_true", help="Print config and exit.")
    parser.add_argument("--runner", choices=("codex_no_finetune", "tinker_rl"), default="codex_no_finetune")
    parser.add_argument("--problem-type", default="trimul")
    parser.add_argument("--experiment-name", default="gpu-mode-0609")
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--groups-per-batch", type=int, default=1)
    parser.add_argument("--group-size", type=int, default=1)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--num-cpus-per-task", type=int, default=1)
    parser.add_argument("--eval-timeout", type=int, default=530)
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
    parser.add_argument("--codex-max-concurrent-requests", type=int, default=4)
    parser.add_argument(
        "--codex-puct-surface-balance",
        choices=("none", "archive_age"),
        default="none",
    )
    return parser


def _config_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "env_type": "examples.gpu_mode.env.GpuModeEnv",
        "runner": args.runner,
        "problem_type": args.problem_type,
        "experiment_name": args.experiment_name,
        "wandb_project": args.wandb_project,
        "groups_per_batch": args.groups_per_batch,
        "group_size": args.group_size,
        "num_epochs": args.num_epochs,
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
        "codex_puct_surface_balance": args.codex_puct_surface_balance,
    }


def _run_discovery(config_payload: dict[str, Any]) -> None:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    from examples.gpu_mode.env import GpuModeEnv
    from ttt_discover import DiscoverConfig, discover

    config = DiscoverConfig(
        env_type=GpuModeEnv,
        runner=config_payload["runner"],
        problem_type=config_payload["problem_type"],
        experiment_name=config_payload["experiment_name"],
        wandb_project=config_payload["wandb_project"],
        groups_per_batch=config_payload["groups_per_batch"],
        group_size=config_payload["group_size"],
        num_epochs=config_payload["num_epochs"],
        num_cpus_per_task=config_payload["num_cpus_per_task"],
        eval_timeout=config_payload["eval_timeout"],
        codex_backend=config_payload["codex_backend"],
        codex_model_name=config_payload["codex_model_name"],
        codex_max_output_tokens=config_payload["codex_max_output_tokens"],
        codex_temperature=config_payload["codex_temperature"],
        codex_cli_command=config_payload["codex_cli_command"],
        codex_cli_sandbox=config_payload["codex_cli_sandbox"],
        codex_cli_timeout=config_payload["codex_cli_timeout"],
        codex_max_concurrent_requests=config_payload["codex_max_concurrent_requests"],
        codex_puct_surface_balance=config_payload["codex_puct_surface_balance"],
    )
    discover(config)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    config_payload = _config_payload(args)
    if args.dry_run or args.mode == "dry-run":
        print(json.dumps(config_payload, indent=2, sort_keys=True))
        return
    _run_discovery(config_payload)


if __name__ == "__main__":
    main()
````

### `tests/test_codex_archive_age_balance.py`

````python
from __future__ import annotations

import ast
import inspect
import json
import subprocess
import sys
from pathlib import Path

import pytest

from ttt_discover.codex_utils import sampler as sampler_mod
from ttt_discover.codex_utils.runtime import State
from ttt_discover.codex_utils.sampler import (
    ARCHIVE_AGE_SURFACES,
    PUCTSampler,
    _archive_age_surface_for_state,
    _archive_birth_step,
    _archive_max_birth,
)


ROOT = Path(__file__).resolve().parents[1]


class DummyEnv:
    state_type = State

    @staticmethod
    def create_initial_state(problem_type: str) -> State:
        return State(
            timestep=-1,
            construction=[],
            code="",
            value=0.0,
            id=f"initial-{problem_type or 'default'}",
        )


def make_state(
    state_id: str,
    timestep: object,
    value: float,
    *,
    parents: list[dict] | None = None,
) -> State:
    state = State(
        timestep=-1,
        construction=[state_id],
        code=f"code-{state_id}",
        value=value,
        parents=parents or [],
        id=state_id,
    )
    state.timestep = timestep
    return state


def make_sampler(tmp_path: Path, *, mode: str = "none", puct_c: float = 0.0) -> PUCTSampler:
    return PUCTSampler(
        file_path=str(tmp_path / "puct_sampler.json"),
        env_type=DummyEnv,
        batch_size=0,
        puct_c=puct_c,
        topk_children=0,
        puct_surface_balance=mode,
    )


def counts(**overrides: int) -> dict[str, int]:
    out = {surface: 0 for surface in ARCHIVE_AGE_SURFACES}
    out.update(overrides)
    return out


def step_path(tmp_path: Path, step: int) -> Path:
    return Path(sampler_mod._sampler_file_for_step(str(tmp_path / "puct_sampler.json"), step))


def write_store(tmp_path: Path, step: int, **overrides: object) -> Path:
    store = {
        "step": step,
        "states": [],
        "initial_states": [],
        "puct_n": {},
        "puct_m": {},
        "puct_T": 0,
    }
    store.update(overrides)
    path = step_path(tmp_path, step)
    path.write_text(json.dumps(store), encoding="utf-8")
    return path


def test_disabled_parity_has_no_archive_age_surface_observability_or_persistence(tmp_path: Path) -> None:
    sampler = make_sampler(tmp_path, mode="none")
    sampler._states = [
        make_state("low", 0, 1.0),
        make_state("mid", 1, 2.0),
        make_state("high", 2, 3.0),
    ]

    picked = sampler.sample_states(2)

    assert [state.id for state in picked] == ["high", "mid"]
    assert [(n, q, bonus, score) for n, q, _p, bonus, score in sampler._last_puct_stats] == [
        (0, pytest.approx(3.0), pytest.approx(0.0), pytest.approx(3.0)),
        (0, pytest.approx(2.0), pytest.approx(0.0), pytest.approx(2.0)),
    ]
    columns, _rows = sampler.get_sample_table()
    assert "age_surface" not in columns
    assert "archive_age" not in columns
    assert not any(key.startswith("puct/archive_age/") for key in sampler.get_sample_stats())

    sampler.flush(step=1)
    store = json.loads(step_path(tmp_path, 1).read_text(encoding="utf-8"))
    assert "puct_surface_balance" not in store
    assert "puct_archive_age_sample_counts" not in store


def test_archive_age_classifier_boundaries_and_invalid_timesteps() -> None:
    states = [
        make_state("seed", -1, 0.0),
        make_state("newest", 10, 0.0),
        make_state("previous", 9, 0.0),
        make_state("age2", 8, 0.0),
        make_state("age4", 6, 0.0),
        make_state("age5", 5, 0.0),
    ]
    assert _archive_max_birth(states) == 10

    assert _archive_age_surface_for_state(states[0], 10) == ("seed", None)
    assert _archive_age_surface_for_state(states[1], 10) == ("fresh", 0)
    assert _archive_age_surface_for_state(states[2], 10) == ("fresh", 1)
    assert _archive_age_surface_for_state(states[3], 10) == ("warm", 2)
    assert _archive_age_surface_for_state(states[4], 10) == ("warm", 4)
    assert _archive_age_surface_for_state(states[5], 10) == ("cold", 5)

    invalid = make_state("invalid", "10", 0.0)
    truthy_bool = make_state("bool", True, 0.0)
    missing = object()
    assert _archive_birth_step(invalid) == -1
    assert _archive_birth_step(truthy_bool) == -1
    assert _archive_birth_step(missing) == -1
    assert _archive_age_surface_for_state(invalid, 10) == ("seed", None)
    assert _archive_age_surface_for_state(truthy_bool, 10) == ("seed", None)


def test_least_count_available_surface_beats_higher_puct_oversampled_surface(tmp_path: Path) -> None:
    sampler = make_sampler(tmp_path, mode="archive_age")
    sampler._states = [
        make_state("fresh-high", 10, 100.0),
        make_state("cold-low", 0, 1.0),
    ]
    sampler._archive_age_sample_counts = counts(seed=10, fresh=99, warm=10, cold=0)

    picked = sampler.sample_states(1)

    assert [state.id for state in picked] == ["cold-low"]
    assert sampler._archive_age_sample_counts["cold"] == 1


def test_unavailable_lower_count_surfaces_are_ignored(tmp_path: Path) -> None:
    sampler = make_sampler(tmp_path, mode="archive_age")
    sampler._states = [
        make_state("fresh-high", 10, 10.0),
        make_state("fresh-low", 9, 1.0),
    ]
    sampler._archive_age_sample_counts = counts(seed=0, fresh=5, warm=0, cold=0)

    picked = sampler.sample_states(1)

    assert [state.id for state in picked] == ["fresh-high"]
    assert sampler._archive_age_sample_counts["fresh"] == 6


def test_rotated_tie_order_uses_total_sample_count_modulo_surface_count(tmp_path: Path) -> None:
    sampler = make_sampler(tmp_path, mode="archive_age")
    sampler._states = [
        make_state("fresh-high", 10, 10.0),
        make_state("cold-low", 0, 1.0),
    ]
    sampler._archive_age_sample_counts = counts(seed=0, fresh=1, warm=0, cold=1)

    picked = sampler.sample_states(1)

    assert sum(counts(seed=0, fresh=1, warm=0, cold=1).values()) % len(ARCHIVE_AGE_SURFACES) == 2
    assert [state.id for state in picked] == ["cold-low"]


def test_within_selected_surface_highest_baseline_puct_candidate_wins(tmp_path: Path) -> None:
    sampler = make_sampler(tmp_path, mode="archive_age")
    sampler._states = [
        make_state("fresh", 10, 100.0),
        make_state("warm-high", 7, 8.0),
        make_state("warm-low", 6, 5.0),
    ]
    sampler._archive_age_sample_counts = counts(seed=9, fresh=9, warm=0, cold=9)

    picked = sampler.sample_states(1)

    assert [state.id for state in picked] == ["warm-high"]


def test_batch_lineage_blocking_recomputes_surface_availability(tmp_path: Path) -> None:
    sampler = make_sampler(tmp_path, mode="archive_age")
    sampler._states = [
        make_state("ancestor-cold", 0, 100.0),
        make_state(
            "descendant-cold",
            0,
            90.0,
            parents=[{"id": "ancestor-cold", "timestep": 0}],
        ),
        make_state("fresh", 10, 80.0),
        make_state("warm", 7, 70.0),
    ]
    sampler._archive_age_sample_counts = counts(seed=5, fresh=5, warm=5, cold=0)

    picked = sampler.sample_states(2)

    assert [state.id for state in picked] == ["ancestor-cold", "fresh"]
    assert "descendant-cold" not in {state.id for state in picked}
    stats = sampler.get_sample_stats()
    assert stats["puct/archive_age/available_surfaces_last"] == 2


def test_enabled_persistence_resume_sanitization_and_disabled_resume_ignore(tmp_path: Path) -> None:
    sampler = make_sampler(tmp_path, mode="archive_age")
    sampler._archive_age_sample_counts = counts(seed=1, fresh=2, warm=3, cold=4)
    sampler.flush(step=1)
    store = json.loads(step_path(tmp_path, 1).read_text(encoding="utf-8"))
    assert store["puct_surface_balance"] == "archive_age"
    assert store["puct_archive_age_sample_counts"] == counts(seed=1, fresh=2, warm=3, cold=4)

    reloaded = PUCTSampler(
        file_path=str(tmp_path / "puct_sampler.json"),
        env_type=DummyEnv,
        batch_size=0,
        resume_step=1,
        puct_surface_balance="archive_age",
    )
    assert reloaded._archive_age_sample_counts == counts(seed=1, fresh=2, warm=3, cold=4)

    write_store(tmp_path, 2)
    missing = PUCTSampler(
        file_path=str(tmp_path / "puct_sampler.json"),
        env_type=DummyEnv,
        batch_size=0,
        resume_step=2,
        puct_surface_balance="archive_age",
    )
    assert missing._archive_age_sample_counts == counts()

    write_store(
        tmp_path,
        3,
        puct_surface_balance="archive_age",
        puct_archive_age_sample_counts={
            "seed": True,
            "fresh": -2,
            "warm": 3,
            "cold": "7",
            "unknown": 99,
        },
    )
    bad = PUCTSampler(
        file_path=str(tmp_path / "puct_sampler.json"),
        env_type=DummyEnv,
        batch_size=0,
        resume_step=3,
        puct_surface_balance="archive_age",
    )
    assert bad._archive_age_sample_counts == counts(seed=0, fresh=0, warm=3, cold=0)
    bad.flush(step=4)
    saved = json.loads(step_path(tmp_path, 4).read_text(encoding="utf-8"))
    assert saved["puct_archive_age_sample_counts"] == counts(seed=0, fresh=0, warm=3, cold=0)

    disabled = PUCTSampler(
        file_path=str(tmp_path / "puct_sampler.json"),
        env_type=DummyEnv,
        batch_size=0,
        resume_step=3,
        puct_surface_balance="none",
    )
    disabled.flush(step=5)
    disabled_store = json.loads(step_path(tmp_path, 5).read_text(encoding="utf-8"))
    assert "puct_surface_balance" not in disabled_store
    assert "puct_archive_age_sample_counts" not in disabled_store


def test_record_failed_rollout_updates_puct_visits_without_archive_age_double_count(tmp_path: Path) -> None:
    sampler = make_sampler(tmp_path, mode="archive_age")
    sampler._states = [make_state("parent", 10, 1.0)]

    parent = sampler.sample_states(1)[0]
    assert sampler._archive_age_sample_counts["fresh"] == 1

    sampler.record_failed_rollout(parent)

    assert sampler._archive_age_sample_counts["fresh"] == 1
    assert sampler._n[parent.id] == 1
    assert sampler._T == 1


def test_cli_dry_run_surface_balance_defaults_enabled_invalid_and_wrapper_override() -> None:
    default = subprocess.run(
        [sys.executable, "repro/run_discovery.py", "dry-run"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    default_config = json.loads(default.stdout)
    assert default_config["codex_puct_surface_balance"] == "none"
    assert default_config["codex_model_name"] is None

    enabled = subprocess.run(
        [
            sys.executable,
            "repro/run_discovery.py",
            "--dry-run",
            "--codex-puct-surface-balance",
            "archive_age",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert json.loads(enabled.stdout)["codex_puct_surface_balance"] == "archive_age"

    invalid = subprocess.run(
        [
            sys.executable,
            "repro/run_discovery.py",
            "--dry-run",
            "--codex-puct-surface-balance",
            "invalid",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert invalid.returncode != 0

    wrapper = subprocess.run(
        ["bash", "repro/gpu_mode/run_0609_archive_age_balance.sh", "dry-run"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    wrapper_config = json.loads(wrapper.stdout)
    assert wrapper_config["experiment_name"] == "gpu-mode-0609-archive-age-balance"
    assert wrapper_config["codex_puct_surface_balance"] == "archive_age"

    override = subprocess.run(
        [
            "bash",
            "repro/gpu_mode/run_0609_archive_age_balance.sh",
            "dry-run",
            "--codex-puct-surface-balance",
            "none",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert json.loads(override.stdout)["codex_puct_surface_balance"] == "none"


def test_build_sampler_passes_surface_balance(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from ttt_discover.rl import codex_no_finetune

    captured = {}

    def fake_create_sampler(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(codex_no_finetune, "create_sampler", fake_create_sampler)
    cfg = codex_no_finetune.CodexNoFinetuneConfig(
        env_type=DummyEnv,
        log_path=str(tmp_path),
        puct_surface_balance="archive_age",
    )

    assert codex_no_finetune._build_sampler(cfg, start_batch=0) is not None
    assert captured["puct_surface_balance"] == "archive_age"


def test_archive_age_classifier_helpers_only_read_timestep() -> None:
    source = "\n".join(
        inspect.getsource(fn)
        for fn in (
            sampler_mod._archive_birth_step,
            sampler_mod._archive_max_birth,
            sampler_mod._archive_age_surface_for_state,
        )
    )
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id == "state":
                assert node.attr == "timestep"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "getattr" and isinstance(node.args[0], ast.Name):
                if node.args[0].id == "state":
                    assert isinstance(node.args[1], ast.Constant)
                    assert node.args[1].value == "timestep"
````

## Diff Stat

### Worktree Tracked Diff Stat

````text
 ttt_discover/codex_utils/discovery.py |   2 +
 ttt_discover/codex_utils/sampler.py   | 184 +++++++++++++++++++++++++++++++++-
 ttt_discover/discovery.py             |   2 +
 ttt_discover/rl/codex_no_finetune.py  |   2 +
 4 files changed, 188 insertions(+), 2 deletions(-)
````
## Raw Diff

<details>
<summary>Tracked worktree patch</summary>

````diff
diff --git a/ttt_discover/codex_utils/discovery.py b/ttt_discover/codex_utils/discovery.py
index b21a80f..5930012 100644
--- a/ttt_discover/codex_utils/discovery.py
+++ b/ttt_discover/codex_utils/discovery.py
@@ -50,6 +50,7 @@ class DiscoverConfig:
     codex_initial_program_paths: tuple[str, ...] = ()
     codex_initial_pool_paths: tuple[str, ...] = ()
     codex_autonomous: bool = False
+    codex_puct_surface_balance: Literal["none", "archive_age"] = "none"
 
 
 def _run_codex_no_finetune(config: DiscoverConfig) -> None:
@@ -85,6 +86,7 @@ def _run_codex_no_finetune(config: DiscoverConfig) -> None:
         initial_program_paths=config.codex_initial_program_paths,
         initial_pool_paths=config.codex_initial_pool_paths,
         autonomous=config.codex_autonomous,
+        puct_surface_balance=config.codex_puct_surface_balance,
         wandb_project=config.wandb_project,
         wandb_name=experiment_name,
         log_path=log_path,
diff --git a/ttt_discover/codex_utils/sampler.py b/ttt_discover/codex_utils/sampler.py
index 5c5d4d1..71de81e 100644
--- a/ttt_discover/codex_utils/sampler.py
+++ b/ttt_discover/codex_utils/sampler.py
@@ -16,6 +16,54 @@ from ttt_discover.codex_utils.runtime import State, state_from_dict, to_json_ser
 
 logger = logging.getLogger(__name__)
 
+ARCHIVE_AGE_SURFACES = ("seed", "fresh", "warm", "cold")
+_PUCT_SURFACE_BALANCE_MODES = ("none", "archive_age")
+
+
+def _validate_puct_surface_balance(mode: str) -> str:
+    if mode not in _PUCT_SURFACE_BALANCE_MODES:
+        raise ValueError(
+            f"Unsupported puct_surface_balance={mode!r}; "
+            f"expected one of {_PUCT_SURFACE_BALANCE_MODES}"
+        )
+    return mode
+
+
+def _archive_birth_step(state: Any) -> int:
+    raw = getattr(state, "timestep", None)
+    if isinstance(raw, (bool, np.bool_)) or not isinstance(raw, (int, np.integer)):
+        return -1
+    birth = int(raw)
+    return birth if birth >= 0 else -1
+
+
+def _archive_max_birth(states: list[Any]) -> int:
+    return max((_archive_birth_step(state) for state in states), default=-1)
+
+
+def _archive_age_surface_for_state(state: Any, max_birth: int) -> tuple[str, int | None]:
+    birth = _archive_birth_step(state)
+    if birth < 0 or max_birth < 0:
+        return "seed", None
+    age = max(0, max_birth - birth)
+    if age <= 1:
+        return "fresh", age
+    if age <= 4:
+        return "warm", age
+    return "cold", age
+
+
+def _sanitize_archive_age_sample_counts(raw: Any) -> dict[str, int]:
+    counts = {surface: 0 for surface in ARCHIVE_AGE_SURFACES}
+    if not isinstance(raw, dict):
+        return counts
+    for surface in ARCHIVE_AGE_SURFACES:
+        value = raw.get(surface, 0)
+        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
+            continue
+        counts[surface] = max(0, int(value))
+    return counts
+
 
 @contextmanager
 def _file_lock(lock_path: str, *, poll_s: float = 0.05, stale_s: float = 600.0):
@@ -353,6 +401,7 @@ class PUCTSampler(StateSampler):
         resume_step: int | None = None,
         puct_c: float = 1.0,
         topk_children: int = 2,
+        puct_surface_balance: str = "none",
     ):
         self.file_path = file_path
         self.env_type = env_type
@@ -361,6 +410,7 @@ class PUCTSampler(StateSampler):
         self.batch_size = batch_size
         self.topk_children = topk_children
         self.puct_c = float(puct_c)
+        self.puct_surface_balance = _validate_puct_surface_balance(puct_surface_balance)
         
         self._states: list[State] = []
         self._initial_states: list[State] = []
@@ -375,6 +425,9 @@ class PUCTSampler(StateSampler):
         self._T: int = 0
         self._last_scale: float = 1.0
         self._last_puct_stats: list[tuple[int, float, float, float, float]] = []
+        self._archive_age_sample_counts = _sanitize_archive_age_sample_counts(None)
+        self._last_archive_age_sample_rows: list[tuple[str, int | None, int]] = []
+        self._last_archive_age_available_surfaces = 0
         
         if resume_step is not None:
             self._load(resume_step)
@@ -399,6 +452,10 @@ class PUCTSampler(StateSampler):
         self._n = store.get("puct_n", {}) or {}
         self._m = store.get("puct_m", {}) or {}
         self._T = int(store.get("puct_T", 0) or 0)
+        if self.puct_surface_balance == "archive_age":
+            self._archive_age_sample_counts = _sanitize_archive_age_sample_counts(
+                store.get("puct_archive_age_sample_counts", {})
+            )
 
     def _save(self, step: int):
         save_path = _sampler_file_for_step(self.file_path, step)
@@ -411,6 +468,12 @@ class PUCTSampler(StateSampler):
             "puct_m": self._m,
             "puct_T": self._T,
         }
+        if self.puct_surface_balance == "archive_age":
+            self._archive_age_sample_counts = _sanitize_archive_age_sample_counts(
+                self._archive_age_sample_counts
+            )
+            store["puct_surface_balance"] = "archive_age"
+            store["puct_archive_age_sample_counts"] = self._archive_age_sample_counts
         with _file_lock(f"{save_path}.lock"):
             _atomic_write_json(save_path, store)
 
@@ -489,6 +552,83 @@ class PUCTSampler(StateSampler):
                     queue.append(child_id)
         return lineage
 
+    def _choose_archive_age_surface(self, available_surfaces: set[str]) -> str | None:
+        if not available_surfaces:
+            return None
+        min_count = min(self._archive_age_sample_counts[s] for s in available_surfaces)
+        total_samples = sum(self._archive_age_sample_counts.values())
+        start = total_samples % len(ARCHIVE_AGE_SURFACES)
+        rotated = ARCHIVE_AGE_SURFACES[start:] + ARCHIVE_AGE_SURFACES[:start]
+        for surface in rotated:
+            if surface in available_surfaces and self._archive_age_sample_counts[surface] == min_count:
+                return surface
+        return None
+
+    def _sample_states_archive_age(
+        self,
+        num_states: int,
+        candidates: list[State],
+        scores: list[tuple[float, float, State, int, float, float, float]],
+        initial_ids: set[str],
+    ) -> list[State]:
+        max_birth = _archive_max_birth(candidates)
+        children_map = self._build_children_map() if num_states > 1 else {}
+        picked: list[State] = []
+        top_scores: list[tuple[float, float, State, int, float, float, float]] = []
+        blocked_ids: set[str] = set()
+        archive_rows: list[tuple[str, int | None, int]] = []
+
+        while len(picked) < num_states:
+            unblocked = [entry for entry in scores if entry[2].id not in blocked_ids]
+            if not unblocked:
+                self._last_archive_age_available_surfaces = 0
+                break
+
+            classified = [
+                (entry, *_archive_age_surface_for_state(entry[2], max_birth))
+                for entry in unblocked
+            ]
+            available_surfaces = {surface for _, surface, _age in classified}
+            self._last_archive_age_available_surfaces = len(available_surfaces)
+
+            target_surface = self._choose_archive_age_surface(available_surfaces)
+            selected = None
+            if target_surface is not None:
+                for entry, surface, age in classified:
+                    if surface == target_surface:
+                        selected = (entry, surface, age)
+                        break
+
+            if selected is None:
+                entry = unblocked[0]
+                surface, age = _archive_age_surface_for_state(entry[2], max_birth)
+                selected = (entry, surface, age)
+
+            entry, surface, age = selected
+            if surface not in ARCHIVE_AGE_SURFACES:
+                surface, age = "seed", None
+            count_before = self._archive_age_sample_counts[surface]
+            self._archive_age_sample_counts[surface] = count_before + 1
+
+            state = entry[2]
+            picked.append(state)
+            top_scores.append(entry)
+            archive_rows.append((surface, age, count_before))
+            if num_states > 1:
+                blocked_ids.update(self._get_full_lineage(state, children_map))
+
+        state_id_to_idx = {s.id: i for i, s in enumerate(self._states)}
+        self._last_sampled_states = picked
+        self._last_sampled_indices = [state_id_to_idx.get(s.id, -1) for s in picked]
+        self._last_puct_stats = [(t[3], t[4], t[5], t[6], t[0]) for t in top_scores]
+        self._last_archive_age_sample_rows = archive_rows
+
+        for s in picked:
+            if s.id in initial_ids:
+                self._refresh_random_construction(s)
+
+        return picked
+
     def sample_states(self, num_states: int) -> list[State]:
         initial_ids = {s.id for s in self._initial_states}
         candidates = list(self._states)
@@ -501,6 +641,8 @@ class PUCTSampler(StateSampler):
             self._last_sampled_states = picked
             self._last_sampled_indices = []
             self._last_puct_stats = [(0, 0.0, 0.0, 0.0, 0.0) for _ in picked]
+            self._last_archive_age_sample_rows = []
+            self._last_archive_age_available_surfaces = 0
             return picked
 
         vals = np.array([float(s.value if s.value is not None else float("-inf")) for s in candidates])
@@ -521,6 +663,14 @@ class PUCTSampler(StateSampler):
 
         scores.sort(key=lambda x: (x[0], x[1]), reverse=True)
 
+        if self.puct_surface_balance == "archive_age":
+            return self._sample_states_archive_age(
+                num_states=num_states,
+                candidates=candidates,
+                scores=scores,
+                initial_ids=initial_ids,
+            )
+
         if num_states > 1:
             children_map = self._build_children_map()
             picked, top_scores, blocked_ids = [], [], set()
@@ -731,21 +881,47 @@ class PUCTSampler(StateSampler):
         stats.update(_stats(sampled_values, "puct/sampled_value"))
         stats.update(_stats(sampled_timesteps, "puct/sampled_timestep"))
         stats.update(_stats(sampled_constr_lens, "puct/sampled_construction_len"))
+        if self.puct_surface_balance == "archive_age":
+            counts = _sanitize_archive_age_sample_counts(self._archive_age_sample_counts)
+            for surface in ARCHIVE_AGE_SURFACES:
+                stats[f"puct/archive_age/{surface}_samples"] = counts[surface]
+            values = list(counts.values())
+            stats["puct/archive_age/available_surfaces_last"] = int(
+                self._last_archive_age_available_surfaces
+            )
+            stats["puct/archive_age/min_surface_samples"] = min(values)
+            stats["puct/archive_age/max_surface_samples"] = max(values)
         return stats
 
     def get_sample_table(self) -> tuple[list[str], list[tuple]]:
         columns = ["buffer_idx", "timestep", "value", "terminal_value", "parent_value", "construction_len", "observation_len", "n", "Q", "P", "bonus", "score"]
+        if self.puct_surface_balance == "archive_age":
+            columns.extend(["age_surface", "archive_age", "age_surface_count_before"])
         rows = []
         if not self._last_sampled_states:
             return columns, rows
         indices = self._last_sampled_indices if len(self._last_sampled_indices) == len(self._last_sampled_states) else [-1] * len(self._last_sampled_states)
         stats = self._last_puct_stats if len(self._last_puct_stats) == len(self._last_sampled_states) else [(0, 0.0, 0.0, 0.0, 0.0)] * len(self._last_sampled_states)
-        for idx, state, (n, Q, P, bonus, score) in zip(indices, self._last_sampled_states, stats):
+        max_birth = _archive_max_birth(self._states)
+        archive_rows = (
+            self._last_archive_age_sample_rows
+            if len(self._last_archive_age_sample_rows) == len(self._last_sampled_states)
+            else []
+        )
+        for row_idx, (idx, state, (n, Q, P, bonus, score)) in enumerate(zip(indices, self._last_sampled_states, stats)):
             parent_val = state.parent_values[0] if state.parent_values else None
             constr = getattr(state, 'construction', None)
             constr_len = len(constr) if constr is not None else 0
             obs_len = len(state.observation) if state.observation else 0
-            rows.append((idx, state.timestep, state.value, 0, parent_val, constr_len, obs_len, n, Q, P, bonus, score))
+            row = (idx, state.timestep, state.value, 0, parent_val, constr_len, obs_len, n, Q, P, bonus, score)
+            if self.puct_surface_balance == "archive_age":
+                if archive_rows:
+                    surface, age, count_before = archive_rows[row_idx]
+                else:
+                    surface, age = _archive_age_surface_for_state(state, max_birth)
+                    count_before = self._archive_age_sample_counts.get(surface, 0)
+                row = row + (surface, age, count_before)
+            rows.append(row)
         return columns, rows
 
 
@@ -756,6 +932,7 @@ def create_sampler(
     batch_size: int = 1,
     resume_step: int | None = None,
     topk_children: int = 2,
+    puct_surface_balance: str = "none",
 ) -> StateSampler:
     """Factory function to create samplers. Pass the env type (from config.env_type)."""
     if not log_path:
@@ -768,6 +945,7 @@ def create_sampler(
         batch_size=batch_size,
         resume_step=resume_step,
         topk_children=topk_children,
+        puct_surface_balance=puct_surface_balance,
     )
 
 
@@ -778,6 +956,7 @@ def get_or_create_sampler_with_default(
     batch_size: int = 1,
     resume_step: int | None = None,
     topk_children: int = 2,
+    puct_surface_balance: str = "none",
 ) -> StateSampler:
     """Get sampler. Initial experience is created via env_type.create_initial_state."""
     return create_sampler(
@@ -787,4 +966,5 @@ def get_or_create_sampler_with_default(
         batch_size=batch_size,
         resume_step=resume_step,
         topk_children=topk_children,
+        puct_surface_balance=puct_surface_balance,
     )
diff --git a/ttt_discover/discovery.py b/ttt_discover/discovery.py
index 6cdb434..1924a77 100644
--- a/ttt_discover/discovery.py
+++ b/ttt_discover/discovery.py
@@ -63,6 +63,7 @@ class DiscoverConfig:
     codex_initial_program_paths: tuple[str, ...] = ()
     codex_initial_pool_paths: tuple[str, ...] = ()
     codex_autonomous: bool = False
+    codex_puct_surface_balance: Literal["none", "archive_age"] = "none"
 
 
 def init_ray(num_cpus_per_task: int, env_type: str):
@@ -146,6 +147,7 @@ async def discover_impl(config: DiscoverConfig):
             initial_program_paths=config.codex_initial_program_paths,
             initial_pool_paths=config.codex_initial_pool_paths,
             autonomous=config.codex_autonomous,
+            puct_surface_balance=config.codex_puct_surface_balance,
             wandb_project=config.wandb_project,
             wandb_name=config.experiment_name,
             log_path=log_path,
diff --git a/ttt_discover/rl/codex_no_finetune.py b/ttt_discover/rl/codex_no_finetune.py
index 18a0e9e..c9b2cd9 100644
--- a/ttt_discover/rl/codex_no_finetune.py
+++ b/ttt_discover/rl/codex_no_finetune.py
@@ -137,6 +137,7 @@ class CodexNoFinetuneConfig:
     initial_program_paths: tuple[str, ...] = ()
     initial_pool_paths: tuple[str, ...] = ()
     topk_children: int = 16
+    puct_surface_balance: Literal["none", "archive_age"] = "none"
 
     # False is TTT Discover: non-auto, many short samples. True is AutoEvolve:
     # one Codex deep-dive workspace per sample with file edits and shell access.
@@ -960,6 +961,7 @@ def _build_sampler(cfg: CodexNoFinetuneConfig, start_batch: int) -> StateSampler
         batch_size=cfg.groups_per_batch,
         resume_step=start_batch if start_batch > 0 else None,
         topk_children=cfg.topk_children,
+        puct_surface_balance=cfg.puct_surface_balance,
     )
 
 
````
</details>

