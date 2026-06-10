# codex/diversity-recent-ancestry-trace-balance

## Summary

检查最近若干 parent trace 是否仍完整保留在 archive 中，分类 root/trace_intact/prefix_gap/skip/lost，并按 trace surface 平衡。

## Branch State

- Worktree: `/opt/tiger/discover-recent-ancestry-trace-balance`
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

- `codex_sampler_surface_balance`
- `surface_balance_mode`
- `sampler_surface_balance`

### Constants

- `RECENT_ANCESTRY_TRACE_SURFACES`
- `RECENT_ANCESTRY_TRACE_WINDOW`
- `_SURFACE_BALANCE_MODES`

### Classes

- None

### Functions

- `_validate_surface_balance_mode`
- `_empty_recent_ancestry_trace_counts`
- `_sanitize_recent_ancestry_trace_counts`
- `classify_recent_ancestry_trace_surface`
- `_recent_ancestry_trace_counts_for_states`
- `_choose_recent_ancestry_trace_surface`
- `_sample_states_recent_ancestry_trace`

## Diff Summary

- Worktree tracked shortstat: `4 files changed, 259 insertions(+), 2 deletions(-)`
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

- `repro/gpu_mode/run_0609_recent_ancestry_trace_balance.sh (711 bytes)`
- `repro/run_discovery.py (5600 bytes)`
- `tests/test_codex_recent_ancestry_trace_balance.py (18998 bytes)`

### Detected Test Functions

- `tests/test_codex_recent_ancestry_trace_balance.py::test_classifier_covers_surfaces_malformed_refs_and_fixed_window`
- `tests/test_codex_recent_ancestry_trace_balance.py::test_classifier_static_check_uses_only_state_parents`
- `tests/test_codex_recent_ancestry_trace_balance.py::test_disabled_mode_matches_default_and_has_no_trace_observability_or_persistence`
- `tests/test_codex_recent_ancestry_trace_balance.py::test_enabled_sampler_rejects_unknown_mode`
- `tests/test_codex_recent_ancestry_trace_balance.py::test_least_count_available_surface_wins_over_higher_puct`
- `tests/test_codex_recent_ancestry_trace_balance.py::test_unavailable_zero_count_surfaces_are_ignored`
- `tests/test_codex_recent_ancestry_trace_balance.py::test_surface_count_ties_break_by_bucket_head_puct_then_fixed_order`
- `tests/test_codex_recent_ancestry_trace_balance.py::test_within_selected_surface_highest_baseline_puct_wins`
- `tests/test_codex_recent_ancestry_trace_balance.py::test_batch_counts_affect_later_picks_and_lineage_blocking_recomputes_availability`
- `tests/test_codex_recent_ancestry_trace_balance.py::test_enabled_metrics_and_table_columns_are_present`
- `tests/test_codex_recent_ancestry_trace_balance.py::test_persistence_resume_sanitizes_and_disabled_resume_ignores_trace_keys`
- `tests/test_codex_recent_ancestry_trace_balance.py::test_record_failed_rollout_updates_puct_visits_without_double_counting_trace`
- `tests/test_codex_recent_ancestry_trace_balance.py::test_public_factories_accept_surface_balance_mode`
- `tests/test_codex_recent_ancestry_trace_balance.py::test_repro_cli_dry_run_enabled_invalid_and_build_sampler_pass_through`

## Untracked File Snapshots

These files are not part of `git diff` yet, but they are part of the worktree implementation state recorded here.

### `repro/gpu_mode/run_0609_recent_ancestry_trace_balance.sh`

````bash
#!/usr/bin/env bash
set -euo pipefail

mode="run"
if [[ "${1:-}" == "run" || "${1:-}" == "dry-run" ]]; then
  mode="$1"
  shift
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
cd "$repo_root"

exec "${PYTHON:-python}" repro/run_discovery.py "$mode" \
  --problem-type trimul \
  --experiment-name gpu-mode-trimul-0609-recent-ancestry-trace-balance \
  --wandb-project "" \
  --runner codex_no_finetune \
  --num-epochs 50 \
  --groups-per-batch 1 \
  --group-size 1 \
  --num-cpus-per-task 1 \
  --eval-timeout 1200 \
  --codex-cli-timeout 600 \
  --codex-max-concurrent-requests 1 \
  --codex-sampler-surface-balance recent_ancestry_trace \
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
SURFACE_BALANCE_CHOICES = ("disabled", "recent_ancestry_trace")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch or dry-run the GPUMode Codex discovery task."
    )
    parser.add_argument(
        "mode",
        nargs="?",
        choices=("run", "dry-run"),
        default="run",
        help="Use 'dry-run' to print the config without importing ttt_discover.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the config without importing ttt_discover.",
    )
    parser.add_argument(
        "--problem-type",
        choices=("trimul", "mla_decode_nvidia"),
        default="trimul",
    )
    parser.add_argument(
        "--experiment-name",
        default="gpu-mode-trimul-codex",
    )
    parser.add_argument(
        "--wandb-project",
        default=None,
        help="WANDB project. Pass '' to disable.",
    )
    parser.add_argument(
        "--runner",
        choices=("codex_no_finetune",),
        default="codex_no_finetune",
    )
    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--group-size", type=int, default=1)
    parser.add_argument("--groups-per-batch", type=int, default=1)
    parser.add_argument("--num-cpus-per-task", type=int, default=1)
    parser.add_argument("--eval-timeout", type=int, default=1200)
    parser.add_argument(
        "--codex-backend",
        choices=("cli", "responses"),
        default="cli",
    )
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
    parser.add_argument(
        "--codex-autonomous",
        action="store_true",
        help="Let Codex inspect and edit an isolated workspace before returning.",
    )
    parser.add_argument(
        "--codex-sampler-surface-balance",
        choices=SURFACE_BALANCE_CHOICES,
        default="disabled",
    )
    return parser.parse_args(argv)


def _none_if_empty(value: str | None) -> str | None:
    if value == "":
        return None
    return value


def build_config_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "env_type": "examples.gpu_mode.env.GpuModeEnv",
        "problem_type": args.problem_type,
        "runner": args.runner,
        "num_epochs": args.num_epochs,
        "group_size": args.group_size,
        "groups_per_batch": args.groups_per_batch,
        "num_cpus_per_task": args.num_cpus_per_task,
        "eval_timeout": args.eval_timeout,
        "experiment_name": args.experiment_name,
        "wandb_project": _none_if_empty(args.wandb_project),
        "codex_backend": args.codex_backend,
        "codex_model_name": args.codex_model_name,
        "codex_max_output_tokens": args.codex_max_output_tokens,
        "codex_temperature": args.codex_temperature,
        "codex_cli_command": args.codex_cli_command,
        "codex_cli_sandbox": args.codex_cli_sandbox,
        "codex_cli_timeout": args.codex_cli_timeout,
        "codex_max_concurrent_requests": args.codex_max_concurrent_requests,
        "codex_autonomous": args.codex_autonomous,
        "codex_sampler_surface_balance": args.codex_sampler_surface_balance,
    }


def _print_dry_run(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = build_config_payload(args)
    if args.dry_run or args.mode == "dry-run":
        _print_dry_run(payload)
        return 0

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    from examples.gpu_mode.env import GpuModeEnv
    from ttt_discover import DiscoverConfig, discover

    config = DiscoverConfig(
        env_type=GpuModeEnv,
        problem_type=payload["problem_type"],
        runner=payload["runner"],
        num_epochs=payload["num_epochs"],
        group_size=payload["group_size"],
        groups_per_batch=payload["groups_per_batch"],
        num_cpus_per_task=payload["num_cpus_per_task"],
        eval_timeout=payload["eval_timeout"],
        experiment_name=payload["experiment_name"],
        wandb_project=payload["wandb_project"],
        codex_backend=payload["codex_backend"],
        codex_model_name=payload["codex_model_name"],
        codex_max_output_tokens=payload["codex_max_output_tokens"],
        codex_temperature=payload["codex_temperature"],
        codex_cli_command=payload["codex_cli_command"],
        codex_cli_sandbox=payload["codex_cli_sandbox"],
        codex_cli_timeout=payload["codex_cli_timeout"],
        codex_max_concurrent_requests=payload["codex_max_concurrent_requests"],
        codex_autonomous=payload["codex_autonomous"],
        codex_sampler_surface_balance=payload["codex_sampler_surface_balance"],
    )
    discover(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
````

### `tests/test_codex_recent_ancestry_trace_balance.py`

````python
from __future__ import annotations

import ast
import inspect
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from ttt_discover.codex_utils.runtime import State
from ttt_discover.codex_utils.sampler import (
    PUCTSampler,
    RECENT_ANCESTRY_TRACE_SURFACES,
    RECENT_ANCESTRY_TRACE_WINDOW,
    _sampler_file_for_step,
    classify_recent_ancestry_trace_surface,
    create_sampler,
    get_or_create_sampler_with_default,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class DummyEnv:
    state_type = State

    @classmethod
    def create_initial_state(cls, problem_type: str) -> State:
        return State(
            timestep=-1,
            construction=[problem_type, "initial"],
            code="",
            value=0.0,
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
        parents=parents if parents is not None else [],
        id=state_id,
    )


def make_sampler(
    tmp_path: Path,
    *,
    mode: str = "disabled",
    states: list[State] | None = None,
    puct_c: float = 0.0,
) -> PUCTSampler:
    sampler = PUCTSampler(
        file_path=str(tmp_path / "puct_sampler.json"),
        env_type=DummyEnv,
        problem_type="dummy",
        batch_size=1,
        topk_children=0,
        puct_c=puct_c,
        surface_balance_mode=mode,
    )
    if states is not None:
        sampler._states = states
        sampler._initial_states = []
    return sampler


def write_sampler_store(file_path: Path, step: int, *, extra: dict | None = None) -> Path:
    state = make_state("root", 1.0)
    store = {
        "step": step,
        "states": [state.to_dict()],
        "initial_states": [],
        "puct_n": {},
        "puct_m": {},
        "puct_T": 0,
    }
    if extra:
        store.update(extra)
    step_path = Path(_sampler_file_for_step(str(file_path), step))
    step_path.parent.mkdir(parents=True, exist_ok=True)
    step_path.write_text(json.dumps(store), encoding="utf-8")
    return step_path


def surface_rows_by_column(sampler: PUCTSampler) -> list[tuple[str, int, int]]:
    columns, rows = sampler.get_sample_table()
    surface_idx = columns.index("recent_ancestry_trace_surface")
    before_idx = columns.index("surface_count_before")
    after_idx = columns.index("surface_count_after")
    return [(row[surface_idx], row[before_idx], row[after_idx]) for row in rows]


def test_classifier_covers_surfaces_malformed_refs_and_fixed_window() -> None:
    assert RECENT_ANCESTRY_TRACE_WINDOW == 3
    assert RECENT_ANCESTRY_TRACE_SURFACES == (
        "root",
        "trace_intact",
        "trace_prefix_gap",
        "trace_skip",
        "trace_lost",
    )

    assert classify_recent_ancestry_trace_surface(make_state("s", 0.0), set()) == "root"
    non_list = make_state("s", 0.0)
    non_list.parents = None
    assert classify_recent_ancestry_trace_surface(non_list, set()) == "root"

    assert (
        classify_recent_ancestry_trace_surface(
            make_state("s", 0.0, parents=[{"id": "a"}, {"id": "b"}, {"id": "c"}]),
            {"a", "b", "c"},
        )
        == "trace_intact"
    )
    assert (
        classify_recent_ancestry_trace_surface(
            make_state("s", 0.0, parents=[{"id": 1}, {"id": "2"}, {"id": 3}]),
            {"1", 2, "3"},
        )
        == "trace_intact"
    )
    assert (
        classify_recent_ancestry_trace_surface(
            make_state("s", 0.0, parents=[{"id": "a"}, {"id": "b"}, {"id": "c"}]),
            {"a"},
        )
        == "trace_prefix_gap"
    )
    assert (
        classify_recent_ancestry_trace_surface(
            make_state("s", 0.0, parents=[{"id": "a"}, {"id": "b"}, {"id": "c"}]),
            {"b"},
        )
        == "trace_skip"
    )
    assert (
        classify_recent_ancestry_trace_surface(
            make_state("s", 0.0, parents=[{"id": "a"}, {"id": "b"}]),
            set(),
        )
        == "trace_lost"
    )
    assert (
        classify_recent_ancestry_trace_surface(
            make_state("s", 0.0, parents=["bad", {"missing": "id"}, {"id": ""}]),
            {"bad"},
        )
        == "trace_lost"
    )
    assert (
        classify_recent_ancestry_trace_surface(
            make_state("s", 0.0, parents=["bad", {"id": "a"}]),
            {"a"},
        )
        == "trace_intact"
    )
    assert (
        classify_recent_ancestry_trace_surface(
            make_state("s", 0.0, parents=["bad", {}, {"id": ""}, {"id": "d"}]),
            {"d"},
        )
        == "trace_lost"
    )
    assert (
        classify_recent_ancestry_trace_surface(
            make_state(
                "s",
                0.0,
                parents=[{"id": "a"}, {"id": "b"}, {"id": "c"}, {"id": "d"}],
            ),
            {"a", "b", "c"},
        )
        == "trace_intact"
    )
    assert (
        classify_recent_ancestry_trace_surface(
            make_state(
                "s",
                0.0,
                parents=[{"id": "a"}, {"id": "b"}, {"id": "c"}, {"id": "d"}],
            ),
            {"d"},
        )
        == "trace_lost"
    )


def test_classifier_static_check_uses_only_state_parents() -> None:
    source = inspect.getsource(classify_recent_ancestry_trace_surface)
    tree = ast.parse(source)
    state_attrs = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "state"
    }
    assert state_attrs == {"parents"}


def test_disabled_mode_matches_default_and_has_no_trace_observability_or_persistence(tmp_path: Path) -> None:
    states = [
        make_state("a", 10.0),
        make_state("b", 5.0),
        make_state("c", 1.0),
    ]
    explicit = make_sampler(tmp_path / "explicit", mode="disabled", states=list(states))
    default = make_sampler(tmp_path / "default", states=list(states))

    assert [s.id for s in explicit.sample_states(2)] == [s.id for s in default.sample_states(2)]
    assert explicit._last_puct_stats == default._last_puct_stats

    columns, _rows = explicit.get_sample_table()
    assert "recent_ancestry_trace_surface" not in columns
    assert not any(key.startswith("puct/recent_ancestry_trace/") for key in explicit.get_sample_stats())

    explicit.flush(step=1)
    saved = json.loads(
        Path(_sampler_file_for_step(explicit.file_path, 1)).read_text(encoding="utf-8")
    )
    assert "puct_surface_balance_mode" not in saved
    assert "puct_recent_ancestry_trace_counts" not in saved


def test_enabled_sampler_rejects_unknown_mode(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported surface_balance_mode"):
        make_sampler(tmp_path, mode="not-a-mode")


def test_least_count_available_surface_wins_over_higher_puct(tmp_path: Path) -> None:
    root = make_state("root", 1.0)
    intact = make_state("intact", 100.0, parents=[{"id": "root"}])
    sampler = make_sampler(
        tmp_path,
        mode="recent_ancestry_trace",
        states=[root, intact],
    )
    sampler._recent_ancestry_trace_counts.update({"root": 0, "trace_intact": 5})

    picked = sampler.sample_states(1)

    assert [state.id for state in picked] == ["root"]
    assert surface_rows_by_column(sampler) == [("root", 0, 1)]


def test_unavailable_zero_count_surfaces_are_ignored(tmp_path: Path) -> None:
    root = make_state("root", 1.0)
    intact = make_state("intact", 100.0, parents=[{"id": "root"}])
    sampler = make_sampler(
        tmp_path,
        mode="recent_ancestry_trace",
        states=[root, intact],
    )
    sampler._recent_ancestry_trace_counts.update(
        {
            "root": 9,
            "trace_intact": 8,
            "trace_prefix_gap": 0,
            "trace_skip": 0,
            "trace_lost": 0,
        }
    )

    picked = sampler.sample_states(1)

    assert [state.id for state in picked] == ["intact"]
    assert surface_rows_by_column(sampler) == [("trace_intact", 8, 9)]


def test_surface_count_ties_break_by_bucket_head_puct_then_fixed_order(tmp_path: Path) -> None:
    root = make_state("root", 1.0)
    intact = make_state("intact", 3.0, parents=[{"id": "root"}])
    sampler = make_sampler(
        tmp_path / "puct",
        mode="recent_ancestry_trace",
        states=[root, intact],
    )

    assert [state.id for state in sampler.sample_states(1)] == ["intact"]

    root_tie = make_state("root", 1.0)
    intact_tie = make_state("intact", 1.0, parents=[{"id": "root"}])
    tie_sampler = make_sampler(
        tmp_path / "fixed",
        mode="recent_ancestry_trace",
        states=[intact_tie, root_tie],
    )

    assert [state.id for state in tie_sampler.sample_states(1)] == ["root"]
    assert surface_rows_by_column(tie_sampler) == [("root", 0, 1)]


def test_within_selected_surface_highest_baseline_puct_wins(tmp_path: Path) -> None:
    root = make_state("root", 1.0)
    lost_best = make_state("lost-best", 10.0, parents=[{"id": "missing"}])
    lost_other = make_state("lost-other", 5.0, parents=[{"id": "missing"}])
    sampler = make_sampler(
        tmp_path,
        mode="recent_ancestry_trace",
        states=[root, lost_other, lost_best],
    )
    sampler._recent_ancestry_trace_counts.update({"root": 4, "trace_lost": 0})

    assert [state.id for state in sampler.sample_states(1)] == ["lost-best"]
    assert surface_rows_by_column(sampler) == [("trace_lost", 0, 1)]


def test_batch_counts_affect_later_picks_and_lineage_blocking_recomputes_availability(tmp_path: Path) -> None:
    ancestor = make_state("ancestor", 90.0)
    child = make_state("child", 100.0, parents=[{"id": "ancestor"}])
    sibling = make_state("sibling", 80.0, parents=[{"id": "ancestor"}])
    unrelated = make_state("unrelated", 1.0)
    sampler = make_sampler(
        tmp_path,
        mode="recent_ancestry_trace",
        states=[ancestor, child, sibling, unrelated],
    )

    picked = sampler.sample_states(2)

    assert [state.id for state in picked] == ["child", "unrelated"]
    assert surface_rows_by_column(sampler) == [
        ("trace_intact", 0, 1),
        ("root", 0, 1),
    ]


def test_enabled_metrics_and_table_columns_are_present(tmp_path: Path) -> None:
    root = make_state("root", 1.0)
    lost = make_state("lost", 2.0, parents=[{"id": "missing"}])
    sampler = make_sampler(
        tmp_path,
        mode="recent_ancestry_trace",
        states=[root, lost],
    )

    sampler.sample_states(1)
    columns, rows = sampler.get_sample_table()
    stats = sampler.get_sample_stats()

    assert columns[-3:] == [
        "recent_ancestry_trace_surface",
        "surface_count_before",
        "surface_count_after",
    ]
    assert len(rows) == 1
    for surface in RECENT_ANCESTRY_TRACE_SURFACES:
        assert f"puct/recent_ancestry_trace/count/{surface}" in stats
        assert f"puct/recent_ancestry_trace/buffer/{surface}" in stats
        assert f"puct/recent_ancestry_trace/sampled/{surface}" in stats
    assert "puct/recent_ancestry_trace/available_surfaces_last" in stats


def test_persistence_resume_sanitizes_and_disabled_resume_ignores_trace_keys(tmp_path: Path) -> None:
    sampler = make_sampler(tmp_path / "enabled", mode="recent_ancestry_trace")
    sampler._recent_ancestry_trace_counts.update({"root": 2, "trace_lost": 1})
    sampler.flush(step=1)
    saved = json.loads(
        Path(_sampler_file_for_step(sampler.file_path, 1)).read_text(encoding="utf-8")
    )
    assert saved["puct_surface_balance_mode"] == "recent_ancestry_trace"
    assert set(saved["puct_recent_ancestry_trace_counts"]) == set(RECENT_ANCESTRY_TRACE_SURFACES)

    reloaded = PUCTSampler(
        file_path=sampler.file_path,
        env_type=DummyEnv,
        problem_type="dummy",
        resume_step=1,
        topk_children=0,
        surface_balance_mode="recent_ancestry_trace",
    )
    assert reloaded._recent_ancestry_trace_counts["root"] == 2
    assert reloaded._recent_ancestry_trace_counts["trace_lost"] == 1

    bad_file = tmp_path / "bad" / "puct_sampler.json"
    write_sampler_store(
        bad_file,
        1,
        extra={
            "puct_surface_balance_mode": "recent_ancestry_trace",
            "puct_recent_ancestry_trace_counts": {
                "root": 3,
                "trace_intact": True,
                "trace_prefix_gap": -2,
                "trace_skip": "4",
                "trace_lost": 5,
                "unknown": 99,
            },
        },
    )
    bad_reloaded = PUCTSampler(
        file_path=str(bad_file),
        env_type=DummyEnv,
        problem_type="dummy",
        resume_step=1,
        topk_children=0,
        surface_balance_mode="recent_ancestry_trace",
    )
    assert bad_reloaded._recent_ancestry_trace_counts == {
        "root": 3,
        "trace_intact": 0,
        "trace_prefix_gap": 0,
        "trace_skip": 0,
        "trace_lost": 5,
    }

    bad_reloaded._recent_ancestry_trace_counts = {
        "root": 1,
        "trace_intact": False,
        "trace_prefix_gap": -1,
        "trace_skip": 2.5,
        "trace_lost": 4,
        "unknown": 10,
    }
    bad_reloaded.flush(step=2)
    sanitized = json.loads(
        Path(_sampler_file_for_step(str(bad_file), 2)).read_text(encoding="utf-8")
    )["puct_recent_ancestry_trace_counts"]
    assert sanitized == {
        "root": 1,
        "trace_intact": 0,
        "trace_prefix_gap": 0,
        "trace_skip": 0,
        "trace_lost": 4,
    }

    missing_file = tmp_path / "missing" / "puct_sampler.json"
    write_sampler_store(missing_file, 1)
    missing_reloaded = PUCTSampler(
        file_path=str(missing_file),
        env_type=DummyEnv,
        problem_type="dummy",
        resume_step=1,
        topk_children=0,
        surface_balance_mode="recent_ancestry_trace",
    )
    assert missing_reloaded._recent_ancestry_trace_counts == {
        surface: 0 for surface in RECENT_ANCESTRY_TRACE_SURFACES
    }

    disabled_file = tmp_path / "disabled" / "puct_sampler.json"
    write_sampler_store(
        disabled_file,
        1,
        extra={
            "puct_surface_balance_mode": "recent_ancestry_trace",
            "puct_recent_ancestry_trace_counts": {"root": 9},
        },
    )
    disabled_reloaded = PUCTSampler(
        file_path=str(disabled_file),
        env_type=DummyEnv,
        problem_type="dummy",
        resume_step=1,
        topk_children=0,
        surface_balance_mode="disabled",
    )
    disabled_reloaded.flush(step=2)
    disabled_saved = json.loads(
        Path(_sampler_file_for_step(str(disabled_file), 2)).read_text(encoding="utf-8")
    )
    assert "puct_surface_balance_mode" not in disabled_saved
    assert "puct_recent_ancestry_trace_counts" not in disabled_saved


def test_record_failed_rollout_updates_puct_visits_without_double_counting_trace(tmp_path: Path) -> None:
    root = make_state("root", 1.0)
    sampler = make_sampler(
        tmp_path,
        mode="recent_ancestry_trace",
        states=[root],
    )

    parent = sampler.sample_states(1)[0]
    assert sampler._recent_ancestry_trace_counts["root"] == 1
    before_visits = sampler._n.get(parent.id, 0)
    before_T = sampler._T

    sampler.record_failed_rollout(parent)

    assert sampler._n[parent.id] == before_visits + 1
    assert sampler._T == before_T + 1
    assert sampler._recent_ancestry_trace_counts["root"] == 1


def test_public_factories_accept_surface_balance_mode(tmp_path: Path) -> None:
    sampler = create_sampler(
        log_path=str(tmp_path / "create"),
        env_type=DummyEnv,
        problem_type="dummy",
        surface_balance_mode="recent_ancestry_trace",
    )
    assert isinstance(sampler, PUCTSampler)
    assert sampler.surface_balance_mode == "recent_ancestry_trace"

    sampler = get_or_create_sampler_with_default(
        log_path=str(tmp_path / "get"),
        env_type=DummyEnv,
        problem_type="dummy",
        surface_balance_mode="recent_ancestry_trace",
    )
    assert isinstance(sampler, PUCTSampler)
    assert sampler.surface_balance_mode == "recent_ancestry_trace"


def test_repro_cli_dry_run_enabled_invalid_and_build_sampler_pass_through(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    default = subprocess.run(
        [sys.executable, "repro/run_discovery.py", "dry-run"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    default_payload = json.loads(default.stdout)
    assert default_payload["codex_sampler_surface_balance"] == "disabled"
    assert default_payload["codex_model_name"] is None
    assert default_payload["num_cpus_per_task"] == 1

    enabled = subprocess.run(
        [
            sys.executable,
            "repro/run_discovery.py",
            "--dry-run",
            "--codex-sampler-surface-balance",
            "recent_ancestry_trace",
            "--num-cpus-per-task",
            "3",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    enabled_payload = json.loads(enabled.stdout)
    assert enabled_payload["codex_sampler_surface_balance"] == "recent_ancestry_trace"
    assert enabled_payload["num_cpus_per_task"] == 3

    invalid = subprocess.run(
        [
            sys.executable,
            "repro/run_discovery.py",
            "dry-run",
            "--codex-sampler-surface-balance",
            "bad",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )
    assert invalid.returncode != 0

    wrapper = subprocess.run(
        [
            "bash",
            "repro/gpu_mode/run_0609_recent_ancestry_trace_balance.sh",
            "dry-run",
            "--codex-sampler-surface-balance",
            "disabled",
            "--num-epochs",
            "2",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    wrapper_payload = json.loads(wrapper.stdout)
    assert wrapper_payload["codex_sampler_surface_balance"] == "disabled"
    assert wrapper_payload["num_epochs"] == 2
    assert wrapper_payload["wandb_project"] is None

    import ttt_discover.rl.codex_no_finetune as codex_no_finetune

    captured: dict[str, object] = {}

    def fake_create_sampler(**kwargs: object) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(codex_no_finetune, "create_sampler", fake_create_sampler)
    cfg = SimpleNamespace(
        log_path=str(tmp_path),
        env_type=DummyEnv,
        problem_type="dummy",
        groups_per_batch=1,
        topk_children=16,
        sampler_surface_balance="recent_ancestry_trace",
    )

    codex_no_finetune._build_sampler(cfg, start_batch=0)

    assert captured["surface_balance_mode"] == "recent_ancestry_trace"
````

## Diff Stat

### Worktree Tracked Diff Stat

````text
 ttt_discover/codex_utils/discovery.py |   2 +
 ttt_discover/codex_utils/sampler.py   | 255 +++++++++++++++++++++++++++++++++-
 ttt_discover/discovery.py             |   2 +
 ttt_discover/rl/codex_no_finetune.py  |   2 +
 4 files changed, 259 insertions(+), 2 deletions(-)
````
## Raw Diff

<details>
<summary>Tracked worktree patch</summary>

````diff
diff --git a/ttt_discover/codex_utils/discovery.py b/ttt_discover/codex_utils/discovery.py
index b21a80f..ad9ad7d 100644
--- a/ttt_discover/codex_utils/discovery.py
+++ b/ttt_discover/codex_utils/discovery.py
@@ -50,6 +50,7 @@ class DiscoverConfig:
     codex_initial_program_paths: tuple[str, ...] = ()
     codex_initial_pool_paths: tuple[str, ...] = ()
     codex_autonomous: bool = False
+    codex_sampler_surface_balance: Literal["disabled", "recent_ancestry_trace"] = "disabled"
 
 
 def _run_codex_no_finetune(config: DiscoverConfig) -> None:
@@ -85,6 +86,7 @@ def _run_codex_no_finetune(config: DiscoverConfig) -> None:
         initial_program_paths=config.codex_initial_program_paths,
         initial_pool_paths=config.codex_initial_pool_paths,
         autonomous=config.codex_autonomous,
+        sampler_surface_balance=config.codex_sampler_surface_balance,
         wandb_project=config.wandb_project,
         wandb_name=experiment_name,
         log_path=log_path,
diff --git a/ttt_discover/codex_utils/sampler.py b/ttt_discover/codex_utils/sampler.py
index 5c5d4d1..ff40bff 100644
--- a/ttt_discover/codex_utils/sampler.py
+++ b/ttt_discover/codex_utils/sampler.py
@@ -16,6 +16,70 @@ from ttt_discover.codex_utils.runtime import State, state_from_dict, to_json_ser
 
 logger = logging.getLogger(__name__)
 
+RECENT_ANCESTRY_TRACE_SURFACES = (
+    "root",
+    "trace_intact",
+    "trace_prefix_gap",
+    "trace_skip",
+    "trace_lost",
+)
+RECENT_ANCESTRY_TRACE_WINDOW = 3
+_SURFACE_BALANCE_MODES = ("disabled", "recent_ancestry_trace")
+
+
+def _validate_surface_balance_mode(surface_balance_mode: str) -> str:
+    if surface_balance_mode not in _SURFACE_BALANCE_MODES:
+        raise ValueError(
+            "Unsupported surface_balance_mode: "
+            f"{surface_balance_mode!r}. Expected one of {_SURFACE_BALANCE_MODES}."
+        )
+    return surface_balance_mode
+
+
+def _empty_recent_ancestry_trace_counts() -> dict[str, int]:
+    return {surface: 0 for surface in RECENT_ANCESTRY_TRACE_SURFACES}
+
+
+def _sanitize_recent_ancestry_trace_counts(raw_counts: Any) -> dict[str, int]:
+    counts = _empty_recent_ancestry_trace_counts()
+    if not isinstance(raw_counts, dict):
+        return counts
+    for surface, value in raw_counts.items():
+        if surface not in counts:
+            continue
+        if isinstance(value, bool) or not isinstance(value, int):
+            counts[surface] = 0
+        else:
+            counts[surface] = max(0, value)
+    return counts
+
+
+def classify_recent_ancestry_trace_surface(state: State, retained_ids: set[Any]) -> str:
+    parents = state.parents
+    if not isinstance(parents, list) or not parents:
+        return "root"
+
+    retained = {str(retained_id) for retained_id in retained_ids}
+    window_ids = []
+    for ref in parents[:RECENT_ANCESTRY_TRACE_WINDOW]:
+        if not isinstance(ref, dict):
+            continue
+        ref_id = ref.get("id")
+        if ref_id:
+            window_ids.append(str(ref_id))
+
+    if not window_ids:
+        return "trace_lost"
+
+    retained_flags = [ref_id in retained for ref_id in window_ids]
+    if all(retained_flags):
+        return "trace_intact"
+    if retained_flags[0]:
+        return "trace_prefix_gap"
+    if any(retained_flags[1:]):
+        return "trace_skip"
+    return "trace_lost"
+
 
 @contextmanager
 def _file_lock(lock_path: str, *, poll_s: float = 0.05, stale_s: float = 600.0):
@@ -353,6 +417,7 @@ class PUCTSampler(StateSampler):
         resume_step: int | None = None,
         puct_c: float = 1.0,
         topk_children: int = 2,
+        surface_balance_mode: str = "disabled",
     ):
         self.file_path = file_path
         self.env_type = env_type
@@ -361,6 +426,7 @@ class PUCTSampler(StateSampler):
         self.batch_size = batch_size
         self.topk_children = topk_children
         self.puct_c = float(puct_c)
+        self.surface_balance_mode = _validate_surface_balance_mode(surface_balance_mode)
         
         self._states: list[State] = []
         self._initial_states: list[State] = []
@@ -375,6 +441,9 @@ class PUCTSampler(StateSampler):
         self._T: int = 0
         self._last_scale: float = 1.0
         self._last_puct_stats: list[tuple[int, float, float, float, float]] = []
+        self._recent_ancestry_trace_counts: dict[str, int] = _empty_recent_ancestry_trace_counts()
+        self._last_recent_ancestry_trace_rows: list[tuple[str, int, int]] = []
+        self._last_recent_ancestry_trace_available_surfaces: int = 0
         
         if resume_step is not None:
             self._load(resume_step)
@@ -399,6 +468,10 @@ class PUCTSampler(StateSampler):
         self._n = store.get("puct_n", {}) or {}
         self._m = store.get("puct_m", {}) or {}
         self._T = int(store.get("puct_T", 0) or 0)
+        if self.surface_balance_mode == "recent_ancestry_trace":
+            self._recent_ancestry_trace_counts = _sanitize_recent_ancestry_trace_counts(
+                store.get("puct_recent_ancestry_trace_counts", {})
+            )
 
     def _save(self, step: int):
         save_path = _sampler_file_for_step(self.file_path, step)
@@ -411,6 +484,12 @@ class PUCTSampler(StateSampler):
             "puct_m": self._m,
             "puct_T": self._T,
         }
+        if self.surface_balance_mode == "recent_ancestry_trace":
+            self._recent_ancestry_trace_counts = _sanitize_recent_ancestry_trace_counts(
+                self._recent_ancestry_trace_counts
+            )
+            store["puct_surface_balance_mode"] = "recent_ancestry_trace"
+            store["puct_recent_ancestry_trace_counts"] = self._recent_ancestry_trace_counts
         with _file_lock(f"{save_path}.lock"):
             _atomic_write_json(save_path, store)
 
@@ -476,6 +555,46 @@ class PUCTSampler(StateSampler):
                     children.setdefault(str(pid), set()).add(s.id)
         return children
 
+    def _sanitize_recent_ancestry_trace_counts(self) -> dict[str, int]:
+        self._recent_ancestry_trace_counts = _sanitize_recent_ancestry_trace_counts(
+            self._recent_ancestry_trace_counts
+        )
+        return self._recent_ancestry_trace_counts
+
+    def _recent_ancestry_trace_counts_for_states(
+        self,
+        states: list[State],
+        retained_ids: set[Any] | None = None,
+    ) -> dict[str, int]:
+        counts = _empty_recent_ancestry_trace_counts()
+        retained = (
+            {str(retained_id) for retained_id in retained_ids}
+            if retained_ids is not None
+            else {str(s.id) for s in states}
+        )
+        for state in states:
+            surface = classify_recent_ancestry_trace_surface(state, retained)
+            counts[surface] += 1
+        return counts
+
+    def _choose_recent_ancestry_trace_surface(
+        self,
+        buckets: dict[str, list[tuple[float, float, State, int, float, float, float]]],
+    ) -> str | None:
+        counts = self._sanitize_recent_ancestry_trace_counts()
+        available = [surface for surface in RECENT_ANCESTRY_TRACE_SURFACES if buckets.get(surface)]
+        self._last_recent_ancestry_trace_available_surfaces = len(available)
+        if not available:
+            return None
+
+        lowest_count = min(counts[surface] for surface in available)
+        count_tied = [surface for surface in available if counts[surface] == lowest_count]
+        best_head = max((buckets[surface][0][0], buckets[surface][0][1]) for surface in count_tied)
+        for surface in RECENT_ANCESTRY_TRACE_SURFACES:
+            if surface in count_tied and (buckets[surface][0][0], buckets[surface][0][1]) == best_head:
+                return surface
+        return None
+
     def _get_full_lineage(self, state: State, children_map: dict[str, set[str]]) -> set[str]:
         lineage = self._get_lineage(state)
         queue = [state.id]
@@ -489,7 +608,103 @@ class PUCTSampler(StateSampler):
                     queue.append(child_id)
         return lineage
 
+    def _sample_states_recent_ancestry_trace(self, num_states: int) -> list[State]:
+        self._sanitize_recent_ancestry_trace_counts()
+        self._last_recent_ancestry_trace_rows = []
+        self._last_recent_ancestry_trace_available_surfaces = 0
+        initial_ids = {s.id for s in self._initial_states}
+        candidates = list(self._states)
+
+        if not candidates:
+            picked = [
+                create_initial_state(self.env_type, self.problem_type)
+                for _ in range(num_states)
+            ]
+            rows = []
+            for _state in picked:
+                before = self._recent_ancestry_trace_counts["root"]
+                after = before + 1
+                self._recent_ancestry_trace_counts["root"] = after
+                rows.append(("root", before, after))
+            self._last_sampled_states = picked
+            self._last_sampled_indices = []
+            self._last_puct_stats = [(0, 0.0, 0.0, 0.0, 0.0) for _ in picked]
+            self._last_recent_ancestry_trace_rows = rows
+            self._last_recent_ancestry_trace_available_surfaces = 1 if picked else 0
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
+        retained_ids = {str(s.id) for s in candidates}
+        children_map = self._build_children_map() if num_states > 1 else {}
+        picked, top_scores, blocked_ids = [], [], set()
+        surface_rows: list[tuple[str, int, int]] = []
+
+        while len(picked) < num_states:
+            selectable = scores
+            if num_states > 1:
+                selectable = [entry for entry in scores if entry[2].id not in blocked_ids]
+            if not selectable:
+                self._last_recent_ancestry_trace_available_surfaces = 0
+                break
+
+            buckets: dict[str, list[tuple[float, float, State, int, float, float, float]]] = {
+                surface: [] for surface in RECENT_ANCESTRY_TRACE_SURFACES
+            }
+            for entry in selectable:
+                surface = classify_recent_ancestry_trace_surface(entry[2], retained_ids)
+                buckets[surface].append(entry)
+
+            selected_surface = self._choose_recent_ancestry_trace_surface(buckets)
+            if selected_surface is None or not buckets.get(selected_surface):
+                chosen = selectable[0]
+                selected_surface = classify_recent_ancestry_trace_surface(chosen[2], retained_ids)
+            else:
+                chosen = buckets[selected_surface][0]
+
+            before = self._recent_ancestry_trace_counts[selected_surface]
+            after = before + 1
+            self._recent_ancestry_trace_counts[selected_surface] = after
+            surface_rows.append((selected_surface, before, after))
+
+            selected_state = chosen[2]
+            picked.append(selected_state)
+            top_scores.append(chosen)
+            if num_states > 1:
+                blocked_ids.update(self._get_full_lineage(selected_state, children_map))
+
+        state_id_to_idx = {s.id: i for i, s in enumerate(self._states)}
+        self._last_sampled_states = picked
+        self._last_sampled_indices = [state_id_to_idx.get(s.id, -1) for s in picked]
+        self._last_puct_stats = [(t[3], t[4], t[5], t[6], t[0]) for t in top_scores]
+        self._last_recent_ancestry_trace_rows = surface_rows
+
+        for s in picked:
+            if s.id in initial_ids:
+                self._refresh_random_construction(s)
+
+        return picked
+
     def sample_states(self, num_states: int) -> list[State]:
+        if self.surface_balance_mode == "recent_ancestry_trace":
+            return self._sample_states_recent_ancestry_trace(num_states)
+
         initial_ids = {s.id for s in self._initial_states}
         candidates = list(self._states)
 
@@ -731,21 +946,53 @@ class PUCTSampler(StateSampler):
         stats.update(_stats(sampled_values, "puct/sampled_value"))
         stats.update(_stats(sampled_timesteps, "puct/sampled_timestep"))
         stats.update(_stats(sampled_constr_lens, "puct/sampled_construction_len"))
+        if self.surface_balance_mode == "recent_ancestry_trace":
+            trace_counts = self._sanitize_recent_ancestry_trace_counts()
+            retained_ids = {str(s.id) for s in self._states}
+            buffer_counts = self._recent_ancestry_trace_counts_for_states(
+                self._states,
+                retained_ids,
+            )
+            sampled_counts = self._recent_ancestry_trace_counts_for_states(
+                self._last_sampled_states,
+                retained_ids,
+            )
+            for surface in RECENT_ANCESTRY_TRACE_SURFACES:
+                stats[f"puct/recent_ancestry_trace/count/{surface}"] = trace_counts[surface]
+                stats[f"puct/recent_ancestry_trace/buffer/{surface}"] = buffer_counts[surface]
+                stats[f"puct/recent_ancestry_trace/sampled/{surface}"] = sampled_counts[surface]
+            stats["puct/recent_ancestry_trace/available_surfaces_last"] = (
+                self._last_recent_ancestry_trace_available_surfaces
+            )
         return stats
 
     def get_sample_table(self) -> tuple[list[str], list[tuple]]:
         columns = ["buffer_idx", "timestep", "value", "terminal_value", "parent_value", "construction_len", "observation_len", "n", "Q", "P", "bonus", "score"]
+        if self.surface_balance_mode == "recent_ancestry_trace":
+            columns = columns + [
+                "recent_ancestry_trace_surface",
+                "surface_count_before",
+                "surface_count_after",
+            ]
         rows = []
         if not self._last_sampled_states:
             return columns, rows
         indices = self._last_sampled_indices if len(self._last_sampled_indices) == len(self._last_sampled_states) else [-1] * len(self._last_sampled_states)
         stats = self._last_puct_stats if len(self._last_puct_stats) == len(self._last_sampled_states) else [(0, 0.0, 0.0, 0.0, 0.0)] * len(self._last_sampled_states)
-        for idx, state, (n, Q, P, bonus, score) in zip(indices, self._last_sampled_states, stats):
+        trace_rows = (
+            self._last_recent_ancestry_trace_rows
+            if len(self._last_recent_ancestry_trace_rows) == len(self._last_sampled_states)
+            else [("", 0, 0)] * len(self._last_sampled_states)
+        )
+        for row_idx, (idx, state, (n, Q, P, bonus, score)) in enumerate(zip(indices, self._last_sampled_states, stats)):
             parent_val = state.parent_values[0] if state.parent_values else None
             constr = getattr(state, 'construction', None)
             constr_len = len(constr) if constr is not None else 0
             obs_len = len(state.observation) if state.observation else 0
-            rows.append((idx, state.timestep, state.value, 0, parent_val, constr_len, obs_len, n, Q, P, bonus, score))
+            row = (idx, state.timestep, state.value, 0, parent_val, constr_len, obs_len, n, Q, P, bonus, score)
+            if self.surface_balance_mode == "recent_ancestry_trace":
+                row = row + trace_rows[row_idx]
+            rows.append(row)
         return columns, rows
 
 
@@ -756,6 +1003,7 @@ def create_sampler(
     batch_size: int = 1,
     resume_step: int | None = None,
     topk_children: int = 2,
+    surface_balance_mode: str = "disabled",
 ) -> StateSampler:
     """Factory function to create samplers. Pass the env type (from config.env_type)."""
     if not log_path:
@@ -768,6 +1016,7 @@ def create_sampler(
         batch_size=batch_size,
         resume_step=resume_step,
         topk_children=topk_children,
+        surface_balance_mode=surface_balance_mode,
     )
 
 
@@ -778,6 +1027,7 @@ def get_or_create_sampler_with_default(
     batch_size: int = 1,
     resume_step: int | None = None,
     topk_children: int = 2,
+    surface_balance_mode: str = "disabled",
 ) -> StateSampler:
     """Get sampler. Initial experience is created via env_type.create_initial_state."""
     return create_sampler(
@@ -787,4 +1037,5 @@ def get_or_create_sampler_with_default(
         batch_size=batch_size,
         resume_step=resume_step,
         topk_children=topk_children,
+        surface_balance_mode=surface_balance_mode,
     )
diff --git a/ttt_discover/discovery.py b/ttt_discover/discovery.py
index 6cdb434..31946a3 100644
--- a/ttt_discover/discovery.py
+++ b/ttt_discover/discovery.py
@@ -63,6 +63,7 @@ class DiscoverConfig:
     codex_initial_program_paths: tuple[str, ...] = ()
     codex_initial_pool_paths: tuple[str, ...] = ()
     codex_autonomous: bool = False
+    codex_sampler_surface_balance: Literal["disabled", "recent_ancestry_trace"] = "disabled"
 
 
 def init_ray(num_cpus_per_task: int, env_type: str):
@@ -146,6 +147,7 @@ async def discover_impl(config: DiscoverConfig):
             initial_program_paths=config.codex_initial_program_paths,
             initial_pool_paths=config.codex_initial_pool_paths,
             autonomous=config.codex_autonomous,
+            sampler_surface_balance=config.codex_sampler_surface_balance,
             wandb_project=config.wandb_project,
             wandb_name=config.experiment_name,
             log_path=log_path,
diff --git a/ttt_discover/rl/codex_no_finetune.py b/ttt_discover/rl/codex_no_finetune.py
index 18a0e9e..6903420 100644
--- a/ttt_discover/rl/codex_no_finetune.py
+++ b/ttt_discover/rl/codex_no_finetune.py
@@ -137,6 +137,7 @@ class CodexNoFinetuneConfig:
     initial_program_paths: tuple[str, ...] = ()
     initial_pool_paths: tuple[str, ...] = ()
     topk_children: int = 16
+    sampler_surface_balance: Literal["disabled", "recent_ancestry_trace"] = "disabled"
 
     # False is TTT Discover: non-auto, many short samples. True is AutoEvolve:
     # one Codex deep-dive workspace per sample with file edits and shell access.
@@ -960,6 +961,7 @@ def _build_sampler(cfg: CodexNoFinetuneConfig, start_batch: int) -> StateSampler
         batch_size=cfg.groups_per_batch,
         resume_step=start_batch if start_batch > 0 else None,
         topk_children=cfg.topk_children,
+        surface_balance_mode=cfg.sampler_surface_balance,
     )
 
 
````
</details>

