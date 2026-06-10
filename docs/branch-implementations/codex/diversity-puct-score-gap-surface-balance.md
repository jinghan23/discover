# codex/diversity-puct-score-gap-surface-balance

## Summary

根据候选 PUCT score 与 leader 的 gap 分类 gap:leader/near/middle/far，用 close/far frac 阈值做 surface 平衡。

## Branch State

- Worktree: `/opt/tiger/discover-puct-score-gap-surface-balance`
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

- `codex_puct_score_gap_surface_balance`
- `codex_puct_score_gap_close_frac`
- `codex_puct_score_gap_far_frac`
- `codex_puct_score_gap_span_eps`
- `surfaces`
- `prev_score`
- `puct_score_gap_surface_balance`
- `puct_score_gap_close_frac`
- `puct_score_gap_far_frac`
- `puct_score_gap_span_eps`

### Constants

- `PUCT_SCORE_GAP_SURFACES`
- `PUCT_SCORE_GAP_DEFAULT_CLOSE_FRAC`
- `PUCT_SCORE_GAP_DEFAULT_FAR_FRAC`
- `PUCT_SCORE_GAP_DEFAULT_SPAN_EPS`

### Classes

- None

### Functions

- `_empty_puct_score_gap_surface_counts`
- `_sanitize_puct_score_gap_thresholds`
- `_sanitize_puct_score_gap_surface_counts`
- `_puct_score_gap_surface_suffix`
- `_classify_puct_score_gap_surfaces`

## Diff Summary

- Worktree tracked shortstat: `4 files changed, 273 insertions(+), 3 deletions(-)`
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

- `repro/gpu_mode/run_0609_puct_score_gap_surface_balance.sh (847 bytes)`
- `repro/run_discovery.py (4929 bytes)`
- `tests/test_codex_puct_score_gap_surface_balance.py (13899 bytes)`

### Detected Test Functions

- `tests/test_codex_puct_score_gap_surface_balance.py::test_disabled_parity_no_persistence_columns_or_stats`
- `tests/test_codex_puct_score_gap_surface_balance.py::test_boundary_classification_leader_near_middle_far`
- `tests/test_codex_puct_score_gap_surface_balance.py::test_non_finite_score_or_predecessor_is_middle_with_none_gap_fields`
- `tests/test_codex_puct_score_gap_surface_balance.py::test_equal_count_tie_uses_earliest_puct_surface`
- `tests/test_codex_puct_score_gap_surface_balance.py::test_unequal_counts_pick_under_selected_surface_over_global_top`
- `tests/test_codex_puct_score_gap_surface_balance.py::test_within_chosen_surface_highest_puct_wins`
- `tests/test_codex_puct_score_gap_surface_balance.py::test_batch_lineage_blocking_recomputes_availability`
- `tests/test_codex_puct_score_gap_surface_balance.py::test_save_load_enabled_sanitizes_and_disabled_omits_key`
- `tests/test_codex_puct_score_gap_surface_balance.py::test_record_failed_rollout_does_not_mutate_score_gap_counts`
- `tests/test_codex_puct_score_gap_surface_balance.py::test_build_sampler_passes_score_gap_config`
- `tests/test_codex_puct_score_gap_surface_balance.py::test_dry_run_defaults_do_not_import_ttt_discover`
- `tests/test_codex_puct_score_gap_surface_balance.py::test_direct_dry_run_enabled_override_all_thresholds`
- `tests/test_codex_puct_score_gap_surface_balance.py::test_direct_dry_run_invalid_float_fails`
- `tests/test_codex_puct_score_gap_surface_balance.py::test_wrapper_dry_run_first_and_override`
- `tests/test_codex_puct_score_gap_surface_balance.py::test_classifier_static_guard_only_reads_state_id`

## Untracked File Snapshots

These files are not part of `git diff` yet, but they are part of the worktree implementation state recorded here.

### `repro/gpu_mode/run_0609_puct_score_gap_surface_balance.sh`

````bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODE="run"
if [[ "${1:-}" == "--dry-run" ]]; then
  MODE="dry-run"
  shift
elif [[ "${1:-}" == "run" || "${1:-}" == "dry-run" ]]; then
  MODE="$1"
  shift
fi

export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"

python "${REPO_ROOT}/repro/run_discovery.py" "${MODE}" \
  --experiment-name "trimul_0609_puct_score_gap_surface_balance_gpu2" \
  --problem-type "trimul" \
  --runner "codex_no_finetune" \
  --codex-puct-score-gap-surface-balance \
  --codex-puct-score-gap-close-frac 0.02 \
  --codex-puct-score-gap-far-frac 0.15 \
  --codex-puct-score-gap-span-eps 1e-6 \
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
    parser = argparse.ArgumentParser(
        description="Launch a Codex no-finetune GPUMode discovery run."
    )
    parser.add_argument("mode", nargs="?", choices=("run", "dry-run"), default="run")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--problem-type", default="trimul")
    parser.add_argument(
        "--experiment-name",
        default="trimul_0609_puct_score_gap_surface_balance_gpu2",
    )
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument(
        "--runner",
        choices=("tinker_rl", "codex_no_finetune"),
        default="codex_no_finetune",
    )
    parser.add_argument("--model-name", default="openai/gpt-oss-120b")
    parser.add_argument("--codex-model-name", default=None)
    parser.add_argument(
        "--codex-backend",
        choices=("cli", "responses"),
        default="cli",
    )
    parser.add_argument("--codex-max-output-tokens", type=int, default=8192)
    parser.add_argument("--codex-temperature", type=float, default=None)
    parser.add_argument(
        "--codex-cli-sandbox",
        choices=("read-only", "workspace-write", "danger-full-access"),
        default="read-only",
    )
    parser.add_argument("--codex-cli-timeout", type=float, default=None)
    parser.add_argument("--codex-max-concurrent-requests", type=int, default=1)
    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--group-size", type=int, default=1)
    parser.add_argument("--groups-per-batch", type=int, default=1)
    parser.add_argument("--num-cpus-per-task", type=int, default=1)
    parser.add_argument("--eval-timeout", type=int, default=1200)

    parser.add_argument(
        "--codex-puct-score-gap-surface-balance",
        dest="codex_puct_score_gap_surface_balance",
        action="store_true",
    )
    parser.add_argument(
        "--no-codex-puct-score-gap-surface-balance",
        dest="codex_puct_score_gap_surface_balance",
        action="store_false",
    )
    parser.set_defaults(codex_puct_score_gap_surface_balance=False)
    parser.add_argument("--codex-puct-score-gap-close-frac", type=float, default=0.02)
    parser.add_argument("--codex-puct-score-gap-far-frac", type=float, default=0.15)
    parser.add_argument("--codex-puct-score-gap-span-eps", type=float, default=1e-6)
    return parser.parse_args(argv)


def _none_if_empty(value: str | None) -> str | None:
    if value == "":
        return None
    return value


def build_discover_config(args: argparse.Namespace, env_type: Any) -> dict[str, Any]:
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
        "wandb_project": _none_if_empty(args.wandb_project),
        "codex_model_name": _none_if_empty(args.codex_model_name),
        "codex_backend": args.codex_backend,
        "codex_max_output_tokens": args.codex_max_output_tokens,
        "codex_temperature": args.codex_temperature,
        "codex_cli_sandbox": args.codex_cli_sandbox,
        "codex_cli_timeout": args.codex_cli_timeout,
        "codex_max_concurrent_requests": args.codex_max_concurrent_requests,
        "codex_puct_score_gap_surface_balance": args.codex_puct_score_gap_surface_balance,
        "codex_puct_score_gap_close_frac": args.codex_puct_score_gap_close_frac,
        "codex_puct_score_gap_far_frac": args.codex_puct_score_gap_far_frac,
        "codex_puct_score_gap_span_eps": args.codex_puct_score_gap_span_eps,
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    dry_run = args.dry_run or args.mode == "dry-run"

    if dry_run:
        discover_config = build_discover_config(args, "examples.gpu_mode.env:GpuModeEnv")
        print(
            json.dumps(
                {
                    "codex_model_name": discover_config["codex_model_name"],
                    "discover_config": discover_config,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    from examples.gpu_mode.env import GpuModeEnv
    from ttt_discover import DiscoverConfig, discover

    discover_config = build_discover_config(args, GpuModeEnv)
    discover(DiscoverConfig(**discover_config))


if __name__ == "__main__":
    main()
````

### `tests/test_codex_puct_score_gap_surface_balance.py`

````python
from __future__ import annotations

import ast
import builtins
import importlib.util
import inspect
import json
import subprocess
import sys
from pathlib import Path

import pytest

from ttt_discover.codex_utils.runtime import State
from ttt_discover.codex_utils.sampler import (
    PUCTSampler,
    _classify_puct_score_gap_surfaces,
)
from ttt_discover.rl.codex_no_finetune import (
    CodexNoFinetuneConfig,
    _build_sampler,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class DummyEnv:
    state_type = State

    @staticmethod
    def create_initial_state(problem_type: str) -> State:
        return make_state("initial", 0.0)


def make_state(
    state_id: str,
    value: float,
    *,
    parents: list[dict] | None = None,
) -> State:
    return State(
        timestep=0,
        construction=[state_id],
        code=f"code-{state_id}",
        value=value,
        parents=parents or [],
        id=state_id,
    )


def make_sampler(
    tmp_path: Path,
    states: list[State],
    *,
    enabled: bool,
    close_frac: float = 0.02,
    far_frac: float = 0.15,
    span_eps: float = 1e-6,
) -> PUCTSampler:
    sampler = PUCTSampler(
        file_path=str(tmp_path / "puct_sampler.json"),
        env_type=DummyEnv,
        batch_size=0,
        topk_children=0,
        puct_c=0.0,
        puct_score_gap_surface_balance=enabled,
        puct_score_gap_close_frac=close_frac,
        puct_score_gap_far_frac=far_frac,
        puct_score_gap_span_eps=span_eps,
    )
    sampler._states = list(states)
    sampler._initial_states = []
    return sampler


def row_values(sampler: PUCTSampler, column: str) -> list:
    columns, rows = sampler.get_sample_table()
    idx = columns.index(column)
    return [row[idx] for row in rows]


def read_step(path: Path, step: int) -> dict:
    return json.loads(path.with_name(f"{path.stem}_step_{step:06d}.json").read_text())


def run_json(args: list[str]) -> dict:
    proc = subprocess.run(
        [sys.executable, *args],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return json.loads(proc.stdout)


def test_disabled_parity_no_persistence_columns_or_stats(tmp_path: Path) -> None:
    states = [make_state("a", 1.0), make_state("b", 3.0), make_state("c", 2.0)]
    sampler = make_sampler(tmp_path, states, enabled=False)

    picked = sampler.sample_states(2)

    assert [state.id for state in picked] == ["b", "c"]
    stats = sampler.get_sample_stats()
    assert not any(key.startswith("puct_score_gap_surface_balance/") for key in stats)
    columns, _rows = sampler.get_sample_table()
    assert "puct_rank" not in columns
    assert "score_gap_surface" not in columns

    sampler.flush(step=1)
    assert "puct_score_gap_surface_counts" not in read_step(tmp_path / "puct_sampler.json", 1)


def test_boundary_classification_leader_near_middle_far(tmp_path: Path) -> None:
    states = [
        make_state("leader", 100.0),
        make_state("near", 99.0),
        make_state("middle", 90.0),
        make_state("far", 80.0),
    ]
    sampler = make_sampler(
        tmp_path,
        states,
        enabled=True,
        close_frac=0.05,
        far_frac=0.5,
    )

    picked = sampler.sample_states(4)

    assert [state.id for state in picked] == ["leader", "near", "middle", "far"]
    assert row_values(sampler, "puct_rank") == [0, 1, 2, 3]
    assert row_values(sampler, "score_gap_surface") == [
        "gap:leader",
        "gap:near",
        "gap:middle",
        "gap:far",
    ]
    assert row_values(sampler, "score_gap_raw") == [None, 1.0, 9.0, 10.0]
    assert row_values(sampler, "score_gap_norm") == [
        None,
        pytest.approx(0.05),
        pytest.approx(0.45),
        pytest.approx(0.5),
    ]


def test_non_finite_score_or_predecessor_is_middle_with_none_gap_fields(
    tmp_path: Path,
) -> None:
    states = [
        make_state("inf", 3.0),
        make_state("finite", 2.0),
        make_state("nan", 1.0),
    ]
    direct = _classify_puct_score_gap_surfaces(
        [
            (float("inf"), 3.0, states[0], 0, 0.0, 0.0, 0.0),
            (1.0, 2.0, states[1], 0, 0.0, 0.0, 0.0),
            (float("nan"), 1.0, states[2], 0, 0.0, 0.0, 0.0),
        ],
        close_frac=0.02,
        far_frac=0.15,
        span_eps=1e-6,
    )
    assert direct["finite"] == ("gap:middle", None, None, 1)
    assert direct["nan"] == ("gap:middle", None, None, 2)

    sampler = make_sampler(tmp_path, states, enabled=True)
    sampler._n.update({"inf": 1, "finite": 1, "nan": 1})
    sampler._m.update({"inf": float("inf"), "finite": 1.0, "nan": 0.0})

    sampler.sample_states(2)

    assert row_values(sampler, "score_gap_surface") == ["gap:leader", "gap:middle"]
    assert row_values(sampler, "score_gap_raw") == [None, None]
    assert row_values(sampler, "score_gap_norm") == [None, None]


def test_equal_count_tie_uses_earliest_puct_surface(tmp_path: Path) -> None:
    sampler = make_sampler(
        tmp_path,
        [
            make_state("leader", 10.0),
            make_state("near", 9.9),
            make_state("far", 0.0),
        ],
        enabled=True,
        close_frac=0.5,
        far_frac=0.9,
    )

    picked = sampler.sample_states(1)

    assert [state.id for state in picked] == ["leader"]
    assert row_values(sampler, "score_gap_surface") == ["gap:leader"]


def test_unequal_counts_pick_under_selected_surface_over_global_top(
    tmp_path: Path,
) -> None:
    sampler = make_sampler(
        tmp_path,
        [
            make_state("leader", 10.0),
            make_state("near", 9.9),
            make_state("far", 0.0),
        ],
        enabled=True,
        close_frac=0.5,
        far_frac=0.9,
    )
    sampler._puct_score_gap_surface_counts["gap:leader"] = 5

    picked = sampler.sample_states(1)

    assert [state.id for state in picked] == ["near"]
    assert row_values(sampler, "score_gap_surface") == ["gap:near"]
    assert row_values(sampler, "score_gap_surface_count_before") == [0]


def test_within_chosen_surface_highest_puct_wins(tmp_path: Path) -> None:
    sampler = make_sampler(
        tmp_path,
        [
            make_state("leader", 10.0),
            make_state("near_best", 9.9),
            make_state("near_next", 9.8),
        ],
        enabled=True,
        close_frac=0.6,
        far_frac=0.9,
    )
    sampler._puct_score_gap_surface_counts["gap:leader"] = 5

    picked = sampler.sample_states(1)

    assert [state.id for state in picked] == ["near_best"]


def test_batch_lineage_blocking_recomputes_availability(tmp_path: Path) -> None:
    leader = make_state("leader", 100.0)
    child = make_state("child", 99.0, parents=[{"id": "leader", "timestep": 0}])
    far = make_state("far", 50.0)
    sampler = make_sampler(
        tmp_path,
        [leader, child, far],
        enabled=True,
        close_frac=0.05,
        far_frac=0.15,
    )

    picked = sampler.sample_states(2)

    assert [state.id for state in picked] == ["leader", "far"]
    assert row_values(sampler, "score_gap_surface") == ["gap:leader", "gap:far"]


def test_save_load_enabled_sanitizes_and_disabled_omits_key(tmp_path: Path) -> None:
    sampler_path = tmp_path / "puct_sampler.json"
    sampler = make_sampler(tmp_path, [make_state("a", 1.0)], enabled=True)
    sampler._puct_score_gap_surface_counts = {
        "gap:leader": 3,
        "gap:near": 2,
        "gap:middle": 1,
        "gap:far": 4,
    }
    sampler.flush(step=1)
    store_path = sampler_path.with_name("puct_sampler_step_000001.json")
    store = json.loads(store_path.read_text())
    store["puct_score_gap_surface_counts"] = {
        "gap:leader": 3,
        "gap:near": -1,
        "gap:middle": True,
        "gap:far": "4",
    }
    store_path.write_text(json.dumps(store), encoding="utf-8")

    loaded = PUCTSampler(
        file_path=str(sampler_path),
        env_type=DummyEnv,
        batch_size=0,
        resume_step=1,
        puct_score_gap_surface_balance=True,
    )
    assert loaded._puct_score_gap_surface_counts == {
        "gap:leader": 3,
        "gap:near": 0,
        "gap:middle": 0,
        "gap:far": 0,
    }

    disabled = PUCTSampler(
        file_path=str(sampler_path),
        env_type=DummyEnv,
        batch_size=0,
        resume_step=1,
        puct_score_gap_surface_balance=False,
    )
    disabled.flush(step=2)
    assert "puct_score_gap_surface_counts" not in read_step(sampler_path, 2)


def test_record_failed_rollout_does_not_mutate_score_gap_counts(tmp_path: Path) -> None:
    parent = make_state("parent", 1.0)
    sampler = make_sampler(tmp_path, [parent], enabled=True)
    sampler._puct_score_gap_surface_counts["gap:leader"] = 7

    sampler.record_failed_rollout(parent)

    assert sampler._puct_score_gap_surface_counts["gap:leader"] == 7


def test_build_sampler_passes_score_gap_config(monkeypatch, tmp_path: Path) -> None:
    captured = {}

    def fake_create_sampler(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        "ttt_discover.rl.codex_no_finetune.create_sampler",
        fake_create_sampler,
    )
    cfg = CodexNoFinetuneConfig(
        env_type=DummyEnv,
        log_path=str(tmp_path),
        groups_per_batch=3,
        topk_children=4,
        puct_score_gap_surface_balance=True,
        puct_score_gap_close_frac=0.03,
        puct_score_gap_far_frac=0.4,
        puct_score_gap_span_eps=1e-5,
    )

    _build_sampler(cfg, start_batch=2)

    assert captured["batch_size"] == 3
    assert captured["resume_step"] == 2
    assert captured["topk_children"] == 4
    assert captured["puct_score_gap_surface_balance"] is True
    assert captured["puct_score_gap_close_frac"] == 0.03
    assert captured["puct_score_gap_far_frac"] == 0.4
    assert captured["puct_score_gap_span_eps"] == 1e-5


def test_dry_run_defaults_do_not_import_ttt_discover(monkeypatch, capsys) -> None:
    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name.startswith("ttt_discover"):
            raise AssertionError("dry-run imported ttt_discover")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    spec = importlib.util.spec_from_file_location(
        "run_discovery_under_test",
        REPO_ROOT / "repro/run_discovery.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    module.main(["dry-run"])

    payload = json.loads(capsys.readouterr().out)
    cfg = payload["discover_config"]
    assert payload["codex_model_name"] is None
    assert cfg["codex_model_name"] is None
    assert cfg["wandb_project"] is None
    assert cfg["num_cpus_per_task"] == 1
    assert cfg["codex_puct_score_gap_surface_balance"] is False
    assert cfg["codex_puct_score_gap_close_frac"] == 0.02
    assert cfg["codex_puct_score_gap_far_frac"] == 0.15
    assert cfg["codex_puct_score_gap_span_eps"] == 1e-6


def test_direct_dry_run_enabled_override_all_thresholds() -> None:
    payload = run_json(
        [
            "repro/run_discovery.py",
            "dry-run",
            "--codex-puct-score-gap-surface-balance",
            "--codex-puct-score-gap-close-frac",
            "0.03",
            "--codex-puct-score-gap-far-frac",
            "0.4",
            "--codex-puct-score-gap-span-eps",
            "0.0002",
            "--codex-model-name",
            "gpt-test",
            "--num-cpus-per-task",
            "7",
            "--experiment-name",
            "override-exp",
        ]
    )
    cfg = payload["discover_config"]
    assert payload["codex_model_name"] == "gpt-test"
    assert cfg["codex_puct_score_gap_surface_balance"] is True
    assert cfg["codex_puct_score_gap_close_frac"] == 0.03
    assert cfg["codex_puct_score_gap_far_frac"] == 0.4
    assert cfg["codex_puct_score_gap_span_eps"] == 0.0002
    assert cfg["num_cpus_per_task"] == 7
    assert cfg["experiment_name"] == "override-exp"


def test_direct_dry_run_invalid_float_fails() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "repro/run_discovery.py",
            "dry-run",
            "--codex-puct-score-gap-close-frac",
            "not-a-float",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )

    assert proc.returncode != 0
    assert "invalid float value" in proc.stderr


def test_wrapper_dry_run_first_and_override() -> None:
    proc = subprocess.run(
        [
            "bash",
            "repro/gpu_mode/run_0609_puct_score_gap_surface_balance.sh",
            "--dry-run",
            "--no-codex-puct-score-gap-surface-balance",
            "--codex-puct-score-gap-close-frac",
            "0.04",
            "--num-epochs",
            "2",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    cfg = payload["discover_config"]
    assert payload["codex_model_name"] is None
    assert cfg["wandb_project"] is None
    assert cfg["experiment_name"] == "trimul_0609_puct_score_gap_surface_balance_gpu2"
    assert cfg["codex_puct_score_gap_surface_balance"] is False
    assert cfg["codex_puct_score_gap_close_frac"] == 0.04
    assert cfg["codex_puct_score_gap_far_frac"] == 0.15
    assert cfg["codex_puct_score_gap_span_eps"] == 1e-6
    assert cfg["num_epochs"] == 2


def test_classifier_static_guard_only_reads_state_id() -> None:
    tree = ast.parse(inspect.getsource(_classify_puct_score_gap_surfaces))
    attrs = sorted(
        {
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "state"
        }
    )
    assert attrs == ["id"]
````

## Diff Stat

### Worktree Tracked Diff Stat

````text
 ttt_discover/codex_utils/discovery.py |   8 ++
 ttt_discover/codex_utils/sampler.py   | 252 +++++++++++++++++++++++++++++++++-
 ttt_discover/discovery.py             |   8 ++
 ttt_discover/rl/codex_no_finetune.py  |   8 ++
 4 files changed, 273 insertions(+), 3 deletions(-)
````
## Raw Diff

<details>
<summary>Tracked worktree patch</summary>

````diff
diff --git a/ttt_discover/codex_utils/discovery.py b/ttt_discover/codex_utils/discovery.py
index b21a80f..82847ef 100644
--- a/ttt_discover/codex_utils/discovery.py
+++ b/ttt_discover/codex_utils/discovery.py
@@ -50,6 +50,10 @@ class DiscoverConfig:
     codex_initial_program_paths: tuple[str, ...] = ()
     codex_initial_pool_paths: tuple[str, ...] = ()
     codex_autonomous: bool = False
+    codex_puct_score_gap_surface_balance: bool = False
+    codex_puct_score_gap_close_frac: float = 0.02
+    codex_puct_score_gap_far_frac: float = 0.15
+    codex_puct_score_gap_span_eps: float = 1e-6
 
 
 def _run_codex_no_finetune(config: DiscoverConfig) -> None:
@@ -85,6 +89,10 @@ def _run_codex_no_finetune(config: DiscoverConfig) -> None:
         initial_program_paths=config.codex_initial_program_paths,
         initial_pool_paths=config.codex_initial_pool_paths,
         autonomous=config.codex_autonomous,
+        puct_score_gap_surface_balance=config.codex_puct_score_gap_surface_balance,
+        puct_score_gap_close_frac=config.codex_puct_score_gap_close_frac,
+        puct_score_gap_far_frac=config.codex_puct_score_gap_far_frac,
+        puct_score_gap_span_eps=config.codex_puct_score_gap_span_eps,
         wandb_project=config.wandb_project,
         wandb_name=experiment_name,
         log_path=log_path,
diff --git a/ttt_discover/codex_utils/sampler.py b/ttt_discover/codex_utils/sampler.py
index 5c5d4d1..e20d328 100644
--- a/ttt_discover/codex_utils/sampler.py
+++ b/ttt_discover/codex_utils/sampler.py
@@ -4,6 +4,7 @@ from abc import ABC, abstractmethod
 from contextlib import contextmanager
 import json
 import logging
+import math
 import os
 from pathlib import Path
 import threading
@@ -127,6 +128,99 @@ def create_initial_state(env_type: type, problem_type: str) -> State:
     return env_type.create_initial_state(problem_type)
 
 
+PUCT_SCORE_GAP_SURFACES = ("gap:leader", "gap:near", "gap:middle", "gap:far")
+PUCT_SCORE_GAP_DEFAULT_CLOSE_FRAC = 0.02
+PUCT_SCORE_GAP_DEFAULT_FAR_FRAC = 0.15
+PUCT_SCORE_GAP_DEFAULT_SPAN_EPS = 1e-6
+
+
+def _empty_puct_score_gap_surface_counts() -> dict[str, int]:
+    return {surface: 0 for surface in PUCT_SCORE_GAP_SURFACES}
+
+
+def _sanitize_puct_score_gap_thresholds(
+    close_frac: float,
+    far_frac: float,
+    span_eps: float,
+) -> tuple[float, float, float]:
+    try:
+        close = float(close_frac)
+        far = float(far_frac)
+    except (TypeError, ValueError):
+        close = PUCT_SCORE_GAP_DEFAULT_CLOSE_FRAC
+        far = PUCT_SCORE_GAP_DEFAULT_FAR_FRAC
+    if not (
+        math.isfinite(close)
+        and math.isfinite(far)
+        and 0.0 <= close < far <= 1.0
+    ):
+        close = PUCT_SCORE_GAP_DEFAULT_CLOSE_FRAC
+        far = PUCT_SCORE_GAP_DEFAULT_FAR_FRAC
+
+    try:
+        eps = float(span_eps)
+    except (TypeError, ValueError):
+        eps = PUCT_SCORE_GAP_DEFAULT_SPAN_EPS
+    if not math.isfinite(eps) or eps <= 0.0:
+        eps = PUCT_SCORE_GAP_DEFAULT_SPAN_EPS
+    return close, far, eps
+
+
+def _sanitize_puct_score_gap_surface_counts(raw_counts: Any) -> dict[str, int]:
+    counts = _empty_puct_score_gap_surface_counts()
+    if not isinstance(raw_counts, dict):
+        return counts
+    for surface in PUCT_SCORE_GAP_SURFACES:
+        value = raw_counts.get(surface, 0)
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
+def _puct_score_gap_surface_suffix(surface: str) -> str:
+    return surface.split(":", 1)[1]
+
+
+def _classify_puct_score_gap_surfaces(
+    sorted_scores: list[tuple[float, float, State, int, float, float, float]],
+    *,
+    close_frac: float,
+    far_frac: float,
+    span_eps: float,
+) -> dict[str, tuple[str, float | None, float | None, int]]:
+    finite_scores = [score for score, *_ in sorted_scores if math.isfinite(score)]
+    if finite_scores:
+        span = max(max(finite_scores) - min(finite_scores), span_eps)
+    else:
+        span = span_eps
+
+    surfaces: dict[str, tuple[str, float | None, float | None, int]] = {}
+    prev_score: float | None = None
+    for rank, (score, _value, state, _n, _Q, _P, _bonus) in enumerate(sorted_scores):
+        if rank == 0:
+            surfaces[str(state.id)] = ("gap:leader", None, None, rank)
+        elif prev_score is None or not (math.isfinite(score) and math.isfinite(prev_score)):
+            surfaces[str(state.id)] = ("gap:middle", None, None, rank)
+        else:
+            raw_gap = max(prev_score - score, 0.0)
+            norm_gap = raw_gap / span
+            if norm_gap <= close_frac:
+                surface = "gap:near"
+            elif norm_gap >= far_frac:
+                surface = "gap:far"
+            else:
+                surface = "gap:middle"
+            surfaces[str(state.id)] = (surface, raw_gap, norm_gap, rank)
+        prev_score = score
+    return surfaces
+
+
 def _resolve_path(raw_path: str | os.PathLike) -> Path:
     path = Path(os.path.expanduser(str(raw_path)))
     if not path.is_absolute():
@@ -353,6 +447,10 @@ class PUCTSampler(StateSampler):
         resume_step: int | None = None,
         puct_c: float = 1.0,
         topk_children: int = 2,
+        puct_score_gap_surface_balance: bool = False,
+        puct_score_gap_close_frac: float = PUCT_SCORE_GAP_DEFAULT_CLOSE_FRAC,
+        puct_score_gap_far_frac: float = PUCT_SCORE_GAP_DEFAULT_FAR_FRAC,
+        puct_score_gap_span_eps: float = PUCT_SCORE_GAP_DEFAULT_SPAN_EPS,
     ):
         self.file_path = file_path
         self.env_type = env_type
@@ -361,6 +459,16 @@ class PUCTSampler(StateSampler):
         self.batch_size = batch_size
         self.topk_children = topk_children
         self.puct_c = float(puct_c)
+        (
+            self.puct_score_gap_close_frac,
+            self.puct_score_gap_far_frac,
+            self.puct_score_gap_span_eps,
+        ) = _sanitize_puct_score_gap_thresholds(
+            puct_score_gap_close_frac,
+            puct_score_gap_far_frac,
+            puct_score_gap_span_eps,
+        )
+        self.puct_score_gap_surface_balance = bool(puct_score_gap_surface_balance)
         
         self._states: list[State] = []
         self._initial_states: list[State] = []
@@ -375,6 +483,10 @@ class PUCTSampler(StateSampler):
         self._T: int = 0
         self._last_scale: float = 1.0
         self._last_puct_stats: list[tuple[int, float, float, float, float]] = []
+        self._puct_score_gap_surface_counts = _empty_puct_score_gap_surface_counts()
+        self._last_puct_score_gap_available_counts = _empty_puct_score_gap_surface_counts()
+        self._last_puct_score_gap_sampled_counts = _empty_puct_score_gap_surface_counts()
+        self._last_puct_score_gap_table: list[tuple[int, str, float | None, float | None, int]] = []
         
         if resume_step is not None:
             self._load(resume_step)
@@ -399,6 +511,10 @@ class PUCTSampler(StateSampler):
         self._n = store.get("puct_n", {}) or {}
         self._m = store.get("puct_m", {}) or {}
         self._T = int(store.get("puct_T", 0) or 0)
+        if self.puct_score_gap_surface_balance:
+            self._puct_score_gap_surface_counts = _sanitize_puct_score_gap_surface_counts(
+                store.get("puct_score_gap_surface_counts")
+            )
 
     def _save(self, step: int):
         save_path = _sampler_file_for_step(self.file_path, step)
@@ -411,6 +527,10 @@ class PUCTSampler(StateSampler):
             "puct_m": self._m,
             "puct_T": self._T,
         }
+        if self.puct_score_gap_surface_balance:
+            store["puct_score_gap_surface_counts"] = _sanitize_puct_score_gap_surface_counts(
+                self._puct_score_gap_surface_counts
+            )
         with _file_lock(f"{save_path}.lock"):
             _atomic_write_json(save_path, store)
 
@@ -501,6 +621,10 @@ class PUCTSampler(StateSampler):
             self._last_sampled_states = picked
             self._last_sampled_indices = []
             self._last_puct_stats = [(0, 0.0, 0.0, 0.0, 0.0) for _ in picked]
+            if self.puct_score_gap_surface_balance:
+                self._last_puct_score_gap_available_counts = _empty_puct_score_gap_surface_counts()
+                self._last_puct_score_gap_sampled_counts = _empty_puct_score_gap_surface_counts()
+                self._last_puct_score_gap_table = []
             return picked
 
         vals = np.array([float(s.value if s.value is not None else float("-inf")) for s in candidates])
@@ -521,7 +645,81 @@ class PUCTSampler(StateSampler):
 
         scores.sort(key=lambda x: (x[0], x[1]), reverse=True)
 
-        if num_states > 1:
+        if self.puct_score_gap_surface_balance:
+            surface_info = _classify_puct_score_gap_surfaces(
+                scores,
+                close_frac=self.puct_score_gap_close_frac,
+                far_frac=self.puct_score_gap_far_frac,
+                span_eps=self.puct_score_gap_span_eps,
+            )
+            use_lineage_blocking = num_states > 1
+            children_map = self._build_children_map() if use_lineage_blocking else {}
+            picked, top_scores = [], []
+            picked_ids: set[str] = set()
+            blocked_ids: set[str] = set()
+            planned_counts = _sanitize_puct_score_gap_surface_counts(
+                self._puct_score_gap_surface_counts
+            )
+            sampled_counts = _empty_puct_score_gap_surface_counts()
+            initial_available_counts: dict[str, int] | None = None
+            self._last_puct_score_gap_table = []
+
+            while len(picked) < num_states:
+                available_counts = _empty_puct_score_gap_surface_counts()
+                surface_entries: dict[str, list[tuple[float, float, State, int, float, float, float]]] = {}
+                surface_first_rank: dict[str, int] = {}
+                for entry in scores:
+                    s = entry[2]
+                    state_id = str(s.id)
+                    if state_id in picked_ids or state_id in blocked_ids:
+                        continue
+                    surface, _raw_gap, _norm_gap, rank = surface_info[state_id]
+                    available_counts[surface] += 1
+                    surface_entries.setdefault(surface, []).append(entry)
+                    surface_first_rank.setdefault(surface, rank)
+
+                if initial_available_counts is None:
+                    initial_available_counts = available_counts
+                if not surface_entries:
+                    break
+
+                selected_surface = min(
+                    surface_entries,
+                    key=lambda surface: (
+                        planned_counts.get(surface, 0),
+                        surface_first_rank[surface],
+                    ),
+                )
+                entry = surface_entries[selected_surface][0]
+                s = entry[2]
+                state_id = str(s.id)
+                count_before = planned_counts.get(selected_surface, 0)
+                surface, raw_gap, norm_gap, rank = surface_info[state_id]
+
+                picked.append(s)
+                top_scores.append(entry)
+                picked_ids.add(state_id)
+                sampled_counts[selected_surface] += 1
+                planned_counts[selected_surface] = count_before + 1
+                self._last_puct_score_gap_table.append(
+                    (rank, surface, raw_gap, norm_gap, count_before)
+                )
+                if use_lineage_blocking:
+                    blocked_ids.update(
+                        str(state_id)
+                        for state_id in self._get_full_lineage(s, children_map)
+                    )
+
+            self._puct_score_gap_surface_counts = _sanitize_puct_score_gap_surface_counts(
+                planned_counts
+            )
+            self._last_puct_score_gap_available_counts = (
+                initial_available_counts
+                if initial_available_counts is not None
+                else _empty_puct_score_gap_surface_counts()
+            )
+            self._last_puct_score_gap_sampled_counts = sampled_counts
+        elif num_states > 1:
             children_map = self._build_children_map()
             picked, top_scores, blocked_ids = [], [], set()
             for entry in scores:
@@ -731,21 +929,53 @@ class PUCTSampler(StateSampler):
         stats.update(_stats(sampled_values, "puct/sampled_value"))
         stats.update(_stats(sampled_timesteps, "puct/sampled_timestep"))
         stats.update(_stats(sampled_constr_lens, "puct/sampled_construction_len"))
+        if self.puct_score_gap_surface_balance:
+            prefix = "puct_score_gap_surface_balance"
+            for surface in PUCT_SCORE_GAP_SURFACES:
+                suffix = _puct_score_gap_surface_suffix(surface)
+                stats[f"{prefix}/selected_count/{suffix}"] = int(
+                    self._puct_score_gap_surface_counts.get(surface, 0)
+                )
+                stats[f"{prefix}/available/{suffix}"] = int(
+                    self._last_puct_score_gap_available_counts.get(surface, 0)
+                )
+                stats[f"{prefix}/sampled/{suffix}"] = int(
+                    self._last_puct_score_gap_sampled_counts.get(surface, 0)
+                )
         return stats
 
     def get_sample_table(self) -> tuple[list[str], list[tuple]]:
         columns = ["buffer_idx", "timestep", "value", "terminal_value", "parent_value", "construction_len", "observation_len", "n", "Q", "P", "bonus", "score"]
+        if self.puct_score_gap_surface_balance:
+            columns.extend(
+                [
+                    "puct_rank",
+                    "score_gap_surface",
+                    "score_gap_raw",
+                    "score_gap_norm",
+                    "score_gap_surface_count_before",
+                ]
+            )
         rows = []
         if not self._last_sampled_states:
             return columns, rows
         indices = self._last_sampled_indices if len(self._last_sampled_indices) == len(self._last_sampled_states) else [-1] * len(self._last_sampled_states)
         stats = self._last_puct_stats if len(self._last_puct_stats) == len(self._last_sampled_states) else [(0, 0.0, 0.0, 0.0, 0.0)] * len(self._last_sampled_states)
-        for idx, state, (n, Q, P, bonus, score) in zip(indices, self._last_sampled_states, stats):
+        gap_rows = (
+            self._last_puct_score_gap_table
+            if self.puct_score_gap_surface_balance
+            and len(self._last_puct_score_gap_table) == len(self._last_sampled_states)
+            else [(None, None, None, None, None)] * len(self._last_sampled_states)
+        )
+        for idx, state, (n, Q, P, bonus, score), gap_row in zip(indices, self._last_sampled_states, stats, gap_rows):
             parent_val = state.parent_values[0] if state.parent_values else None
             constr = getattr(state, 'construction', None)
             constr_len = len(constr) if constr is not None else 0
             obs_len = len(state.observation) if state.observation else 0
-            rows.append((idx, state.timestep, state.value, 0, parent_val, constr_len, obs_len, n, Q, P, bonus, score))
+            row = (idx, state.timestep, state.value, 0, parent_val, constr_len, obs_len, n, Q, P, bonus, score)
+            if self.puct_score_gap_surface_balance:
+                row = row + gap_row
+            rows.append(row)
         return columns, rows
 
 
@@ -756,6 +986,10 @@ def create_sampler(
     batch_size: int = 1,
     resume_step: int | None = None,
     topk_children: int = 2,
+    puct_score_gap_surface_balance: bool = False,
+    puct_score_gap_close_frac: float = PUCT_SCORE_GAP_DEFAULT_CLOSE_FRAC,
+    puct_score_gap_far_frac: float = PUCT_SCORE_GAP_DEFAULT_FAR_FRAC,
+    puct_score_gap_span_eps: float = PUCT_SCORE_GAP_DEFAULT_SPAN_EPS,
 ) -> StateSampler:
     """Factory function to create samplers. Pass the env type (from config.env_type)."""
     if not log_path:
@@ -768,6 +1002,10 @@ def create_sampler(
         batch_size=batch_size,
         resume_step=resume_step,
         topk_children=topk_children,
+        puct_score_gap_surface_balance=puct_score_gap_surface_balance,
+        puct_score_gap_close_frac=puct_score_gap_close_frac,
+        puct_score_gap_far_frac=puct_score_gap_far_frac,
+        puct_score_gap_span_eps=puct_score_gap_span_eps,
     )
 
 
@@ -778,6 +1016,10 @@ def get_or_create_sampler_with_default(
     batch_size: int = 1,
     resume_step: int | None = None,
     topk_children: int = 2,
+    puct_score_gap_surface_balance: bool = False,
+    puct_score_gap_close_frac: float = PUCT_SCORE_GAP_DEFAULT_CLOSE_FRAC,
+    puct_score_gap_far_frac: float = PUCT_SCORE_GAP_DEFAULT_FAR_FRAC,
+    puct_score_gap_span_eps: float = PUCT_SCORE_GAP_DEFAULT_SPAN_EPS,
 ) -> StateSampler:
     """Get sampler. Initial experience is created via env_type.create_initial_state."""
     return create_sampler(
@@ -787,4 +1029,8 @@ def get_or_create_sampler_with_default(
         batch_size=batch_size,
         resume_step=resume_step,
         topk_children=topk_children,
+        puct_score_gap_surface_balance=puct_score_gap_surface_balance,
+        puct_score_gap_close_frac=puct_score_gap_close_frac,
+        puct_score_gap_far_frac=puct_score_gap_far_frac,
+        puct_score_gap_span_eps=puct_score_gap_span_eps,
     )
diff --git a/ttt_discover/discovery.py b/ttt_discover/discovery.py
index 6cdb434..b33fa7c 100644
--- a/ttt_discover/discovery.py
+++ b/ttt_discover/discovery.py
@@ -63,6 +63,10 @@ class DiscoverConfig:
     codex_initial_program_paths: tuple[str, ...] = ()
     codex_initial_pool_paths: tuple[str, ...] = ()
     codex_autonomous: bool = False
+    codex_puct_score_gap_surface_balance: bool = False
+    codex_puct_score_gap_close_frac: float = 0.02
+    codex_puct_score_gap_far_frac: float = 0.15
+    codex_puct_score_gap_span_eps: float = 1e-6
 
 
 def init_ray(num_cpus_per_task: int, env_type: str):
@@ -146,6 +150,10 @@ async def discover_impl(config: DiscoverConfig):
             initial_program_paths=config.codex_initial_program_paths,
             initial_pool_paths=config.codex_initial_pool_paths,
             autonomous=config.codex_autonomous,
+            puct_score_gap_surface_balance=config.codex_puct_score_gap_surface_balance,
+            puct_score_gap_close_frac=config.codex_puct_score_gap_close_frac,
+            puct_score_gap_far_frac=config.codex_puct_score_gap_far_frac,
+            puct_score_gap_span_eps=config.codex_puct_score_gap_span_eps,
             wandb_project=config.wandb_project,
             wandb_name=config.experiment_name,
             log_path=log_path,
diff --git a/ttt_discover/rl/codex_no_finetune.py b/ttt_discover/rl/codex_no_finetune.py
index 18a0e9e..6fba4bd 100644
--- a/ttt_discover/rl/codex_no_finetune.py
+++ b/ttt_discover/rl/codex_no_finetune.py
@@ -137,6 +137,10 @@ class CodexNoFinetuneConfig:
     initial_program_paths: tuple[str, ...] = ()
     initial_pool_paths: tuple[str, ...] = ()
     topk_children: int = 16
+    puct_score_gap_surface_balance: bool = False
+    puct_score_gap_close_frac: float = 0.02
+    puct_score_gap_far_frac: float = 0.15
+    puct_score_gap_span_eps: float = 1e-6
 
     # False is TTT Discover: non-auto, many short samples. True is AutoEvolve:
     # one Codex deep-dive workspace per sample with file edits and shell access.
@@ -960,6 +964,10 @@ def _build_sampler(cfg: CodexNoFinetuneConfig, start_batch: int) -> StateSampler
         batch_size=cfg.groups_per_batch,
         resume_step=start_batch if start_batch > 0 else None,
         topk_children=cfg.topk_children,
+        puct_score_gap_surface_balance=cfg.puct_score_gap_surface_balance,
+        puct_score_gap_close_frac=cfg.puct_score_gap_close_frac,
+        puct_score_gap_far_frac=cfg.puct_score_gap_far_frac,
+        puct_score_gap_span_eps=cfg.puct_score_gap_span_eps,
     )
 
 
````
</details>

