# codex/diversity-subtree-frontier-balance

## Summary

构造 retained archive 的 children map，计算每个候选子树 frontier leaf count，分类 leaf/chain/branch_2_3/branch_4_plus 并平衡。

## Branch State

- Worktree: `/opt/tiger/discover-subtree-frontier-balance`
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

- `codex_sampler_diversity_method`
- `memo`
- `active`
- `sampler_diversity_method`

### Constants

- `SUBTREE_FRONTIER_SURFACES`
- `SUBTREE_FRONTIER_BALANCE_METHOD`
- `SAMPLER_DIVERSITY_METHODS`

### Classes

- None

### Functions

- `_validate_sampler_diversity_method`
- `_first_parent_id`
- `build_subtree_frontier_children_map`
- `subtree_frontier_leaf_count`
- `_count`
- `classify_subtree_frontier_surface`
- `_zero_subtree_frontier_counts`
- `_sanitize_subtree_frontier_sample_counts`
- `_score_candidates`
- `_select_subtree_frontier_surface`
- `_sample_states_subtree_frontier`
- `_surface_for`
- `sample_states`

## Diff Summary

- Worktree tracked shortstat: `4 files changed, 312 insertions(+), 16 deletions(-)`
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

- `repro/gpu_mode/run_0609_subtree_frontier_balance.sh (901 bytes)`
- `repro/run_discovery.py (7135 bytes)`
- `tests/test_codex_subtree_frontier_balance.py (14748 bytes)`

### Detected Test Functions

- `tests/test_codex_subtree_frontier_balance.py::test_classifier_surfaces_and_cycle_guard`
- `tests/test_codex_subtree_frontier_balance.py::test_classifier_helpers_only_read_id_and_parents`
- `tests/test_codex_subtree_frontier_balance.py::test_disabled_mode_preserves_baseline_shapes`
- `tests/test_codex_subtree_frontier_balance.py::test_least_count_available_surface_wins_over_puct`
- `tests/test_codex_subtree_frontier_balance.py::test_unavailable_zero_count_surfaces_are_ignored`
- `tests/test_codex_subtree_frontier_balance.py::test_surface_ties_use_puct_head_then_surface_order`
- `tests/test_codex_subtree_frontier_balance.py::test_within_selected_surface_highest_puct_wins`
- `tests/test_codex_subtree_frontier_balance.py::test_batch_lineage_blocking_recomputes_availability`
- `tests/test_codex_subtree_frontier_balance.py::test_enabled_persistence_resume_sanitize_and_disabled_ignore`
- `tests/test_codex_subtree_frontier_balance.py::test_record_failed_rollout_does_not_double_count_surface`
- `tests/test_codex_subtree_frontier_balance.py::test_enabled_observability_is_gated_and_populated`
- `tests/test_codex_subtree_frontier_balance.py::test_invalid_sampler_method_rejected`
- `tests/test_codex_subtree_frontier_balance.py::test_cli_dry_run_and_build_sampler_pass_through`
- `tests/test_codex_subtree_frontier_balance.py::test_gpu_mode_wrapper_dry_run_and_override`

## Untracked File Snapshots

These files are not part of `git diff` yet, but they are part of the worktree implementation state recorded here.

### `repro/gpu_mode/run_0609_subtree_frontier_balance.sh`

````bash
#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

MODE="run"
if [[ $# -gt 0 ]]; then
  case "$1" in
    run|dry-run)
      MODE="$1"
      shift
      ;;
  esac
fi

CMD=(
  python repro/run_discovery.py "${MODE}"
  --env gpu_mode \
  --runner codex_no_finetune \
  --problem-type trimul \
  --experiment-name gpu-mode-trimul-0609-subtree-frontier-balance \
  --num-epochs 50 \
  --group-size 1 \
  --groups-per-batch 1 \
  --num-cpus-per-task 1 \
  --eval-timeout 530 \
  --codex-backend cli \
  --codex-cli-command "${CODEX_CLI_COMMAND:-codex}" \
  --codex-cli-sandbox read-only \
  --codex-cli-timeout 600 \
  --codex-max-concurrent-requests 1 \
  --codex-sampler-diversity-method subtree_frontier_balance
)

if [[ -n "${CODEX_MODEL_NAME:-}" ]]; then
  CMD+=(--codex-model-name "${CODEX_MODEL_NAME}")
fi

exec "${CMD[@]}" "$@"
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
GPU_MODE_ENV = "examples.gpu_mode.env.GpuModeEnv"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or dry-run a discovery repro.")
    parser.add_argument("mode", nargs="?", choices=("run", "dry-run"), default="run")
    parser.add_argument("--dry-run", action="store_true", help="Print config and exit.")
    parser.add_argument("--env", choices=("gpu_mode",), default="gpu_mode")
    parser.add_argument("--problem-type", default="trimul")
    parser.add_argument("--experiment-name", default="discover-repro")
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument(
        "--model-name",
        choices=("openai/gpt-oss-120b", "openai/gpt-oss-20b"),
        default="openai/gpt-oss-120b",
    )
    parser.add_argument(
        "--runner",
        choices=("tinker_rl", "codex_no_finetune"),
        default="codex_no_finetune",
    )
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--group-size", type=int, default=1)
    parser.add_argument("--groups-per-batch", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=4e-5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--kl-penalty-coef", type=float, default=0.1)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--save-every", type=int, default=2)
    parser.add_argument("--phase1-max-tokens", type=int, default=26000)
    parser.add_argument("--num-cpus-per-task", type=int, default=1)
    parser.add_argument("--eval-timeout", type=int, default=530)
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
    parser.add_argument(
        "--codex-sampler-diversity-method",
        choices=("none", "subtree_frontier_balance"),
        default="none",
    )
    parser.add_argument("--codex-autonomous", action="store_true")
    return parser.parse_args(argv)


def _config_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "env_type": GPU_MODE_ENV if args.env == "gpu_mode" else args.env,
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
        "wandb_project": args.wandb_project,
        "codex_backend": args.codex_backend,
        "codex_model_name": args.codex_model_name,
        "codex_max_output_tokens": args.codex_max_output_tokens,
        "codex_temperature": args.codex_temperature,
        "codex_api_key_env": args.codex_api_key_env,
        "codex_base_url": args.codex_base_url,
        "codex_cli_command": args.codex_cli_command,
        "codex_cli_sandbox": args.codex_cli_sandbox,
        "codex_cli_timeout": args.codex_cli_timeout,
        "codex_max_concurrent_requests": args.codex_max_concurrent_requests,
        "codex_initial_program_paths": list(args.codex_initial_program),
        "codex_initial_pool_paths": list(args.codex_initial_pool),
        "codex_sampler_diversity_method": args.codex_sampler_diversity_method,
        "codex_autonomous": args.codex_autonomous,
    }


def _resolve_env(env_name: str) -> type:
    if env_name == "gpu_mode":
        from examples.gpu_mode.env import GpuModeEnv

        return GpuModeEnv
    raise ValueError(f"Unsupported env: {env_name}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    is_dry_run = args.dry_run or args.mode == "dry-run"
    payload = _config_payload(args)
    if is_dry_run:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    from ttt_discover import DiscoverConfig, discover

    config = DiscoverConfig(
        env_type=_resolve_env(args.env),
        problem_type=payload["problem_type"],
        model_name=payload["model_name"],
        runner=payload["runner"],
        lora_rank=payload["lora_rank"],
        group_size=payload["group_size"],
        groups_per_batch=payload["groups_per_batch"],
        learning_rate=payload["learning_rate"],
        num_epochs=payload["num_epochs"],
        temperature=payload["temperature"],
        kl_penalty_coef=payload["kl_penalty_coef"],
        phase1_max_tokens=payload["phase1_max_tokens"],
        save_every=payload["save_every"],
        num_cpus_per_task=payload["num_cpus_per_task"],
        eval_timeout=payload["eval_timeout"],
        experiment_name=payload["experiment_name"],
        wandb_project=payload["wandb_project"],
        codex_backend=payload["codex_backend"],
        codex_model_name=payload["codex_model_name"],
        codex_max_output_tokens=payload["codex_max_output_tokens"],
        codex_temperature=payload["codex_temperature"],
        codex_api_key_env=payload["codex_api_key_env"],
        codex_base_url=payload["codex_base_url"],
        codex_cli_command=payload["codex_cli_command"],
        codex_cli_sandbox=payload["codex_cli_sandbox"],
        codex_cli_timeout=payload["codex_cli_timeout"],
        codex_max_concurrent_requests=payload["codex_max_concurrent_requests"],
        codex_initial_program_paths=tuple(payload["codex_initial_program_paths"]),
        codex_initial_pool_paths=tuple(payload["codex_initial_pool_paths"]),
        codex_sampler_diversity_method=payload["codex_sampler_diversity_method"],
        codex_autonomous=payload["codex_autonomous"],
    )
    discover(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
````

### `tests/test_codex_subtree_frontier_balance.py`

````python
from __future__ import annotations

import inspect
import json
import subprocess
import sys
from pathlib import Path

import pytest

from ttt_discover.codex_utils.runtime import State
from ttt_discover.codex_utils.sampler import (
    PUCTSampler,
    SUBTREE_FRONTIER_SURFACES,
    _first_parent_id,
    _sampler_file_for_step,
    build_subtree_frontier_children_map,
    classify_subtree_frontier_surface,
)


ROOT = Path(__file__).resolve().parents[1]


class DummyEnv:
    state_type = State
    _next_id = 0

    @classmethod
    def create_initial_state(cls, problem_type: str) -> State:
        cls._next_id += 1
        return state(f"init-{cls._next_id}", value=0.0)


def state(sid: str, *, value: float = 0.0, parent: str | None = None) -> State:
    parents = [] if parent is None else [{"id": parent, "timestep": 0}]
    return State(
        timestep=0,
        construction=[sid],
        code=f"# {sid}",
        value=value,
        parents=parents,
        id=sid,
        observation="",
    )


def counts(**overrides: int) -> dict[str, int]:
    out = {surface: 0 for surface in SUBTREE_FRONTIER_SURFACES}
    out.update(overrides)
    return out


def make_sampler(
    tmp_path: Path,
    states: list[State],
    *,
    method: str = "subtree_frontier_balance",
    sample_counts: dict[str, int] | None = None,
) -> PUCTSampler:
    sampler = PUCTSampler(
        file_path=str(tmp_path / "puct_sampler.json"),
        env_type=DummyEnv,
        batch_size=0,
        puct_c=0.0,
        topk_children=0,
        sampler_diversity_method=method,
    )
    sampler._states = states
    sampler._initial_states = []
    sampler._n = {}
    sampler._m = {}
    sampler._T = 0
    if sample_counts is not None:
        sampler._subtree_frontier_sample_counts = sample_counts.copy()
    return sampler


def sampler_json(path: Path, step: int) -> dict:
    return json.loads(Path(_sampler_file_for_step(str(path / "puct_sampler.json"), step)).read_text())


def test_classifier_surfaces_and_cycle_guard() -> None:
    states = [
        state("leaf"),
        state("chain"),
        state("chain_leaf", parent="chain"),
        state("one_child_branch"),
        state("branch_mid", parent="one_child_branch"),
        state("branch_leaf_1", parent="branch_mid"),
        state("branch_leaf_2", parent="branch_mid"),
        state("wide"),
        state("wide_leaf_1", parent="wide"),
        state("wide_leaf_2", parent="wide"),
        state("wide_leaf_3", parent="wide"),
        state("wide_leaf_4", parent="wide"),
    ]
    children = build_subtree_frontier_children_map(states)

    assert SUBTREE_FRONTIER_SURFACES == ("leaf", "chain", "branch_2_3", "branch_4_plus")
    assert classify_subtree_frontier_surface("leaf", children) == ("leaf", 1)
    assert classify_subtree_frontier_surface("chain", children) == ("chain", 1)
    assert classify_subtree_frontier_surface("one_child_branch", children) == ("branch_2_3", 2)
    assert classify_subtree_frontier_surface("wide", children) == ("branch_4_plus", 4)

    cycle = [
        state("cycle_a", parent="cycle_c"),
        state("cycle_b", parent="cycle_a"),
        state("cycle_c", parent="cycle_b"),
    ]
    surface, leaves = classify_subtree_frontier_surface(
        "cycle_a",
        build_subtree_frontier_children_map(cycle),
    )
    assert surface in SUBTREE_FRONTIER_SURFACES
    assert leaves == 0


def test_classifier_helpers_only_read_id_and_parents() -> None:
    source = "\n".join(
        [
            inspect.getsource(_first_parent_id),
            inspect.getsource(build_subtree_frontier_children_map),
            inspect.getsource(classify_subtree_frontier_surface),
        ]
    )
    for forbidden in (
        "construction",
        "observation",
        "value",
        "timestep",
        "parent_values",
        "code",
    ):
        assert forbidden not in source


def test_disabled_mode_preserves_baseline_shapes(tmp_path: Path) -> None:
    sampler = make_sampler(
        tmp_path,
        [state("a", value=3.0), state("b", value=4.0), state("c", value=2.0)],
        method="none",
    )

    picked = sampler.sample_states(2)
    assert [s.id for s in picked] == ["b", "a"]
    assert [(n, q, bonus, score) for n, q, _p, bonus, score in sampler._last_puct_stats] == [
        (0, 4.0, 0.0, 4.0),
        (0, 3.0, 0.0, 3.0),
    ]

    columns, rows = sampler.get_sample_table()
    assert "subtree_frontier_surface" not in columns
    assert len(columns) == 12
    assert len(rows[0]) == 12
    assert not any(key.startswith("puct/subtree_frontier/") for key in sampler.get_sample_stats())

    sampler.flush(step=1)
    saved = sampler_json(tmp_path, 1)
    assert "sampler_diversity_method" not in saved
    assert "subtree_frontier_sample_counts" not in saved


def test_least_count_available_surface_wins_over_puct(tmp_path: Path) -> None:
    sampler = make_sampler(
        tmp_path,
        [
            state("leaf_hi", value=100.0),
            state("chain_parent", value=10.0),
            state("chain_child", value=1.0, parent="chain_parent"),
        ],
        sample_counts=counts(leaf=10, chain=0),
    )

    assert [s.id for s in sampler.sample_states(1)] == ["chain_parent"]


def test_unavailable_zero_count_surfaces_are_ignored(tmp_path: Path) -> None:
    sampler = make_sampler(
        tmp_path,
        [
            state("leaf", value=100.0),
            state("chain_parent", value=90.0),
            state("chain_child", value=80.0, parent="chain_parent"),
        ],
        sample_counts=counts(leaf=2, chain=3, branch_2_3=0, branch_4_plus=0),
    )

    assert [s.id for s in sampler.sample_states(2)] == ["leaf", "chain_parent"]


def test_surface_ties_use_puct_head_then_surface_order(tmp_path: Path) -> None:
    by_puct = make_sampler(
        tmp_path / "puct",
        [
            state("leaf", value=100.0),
            state("chain_parent", value=90.0),
            state("chain_child", value=1.0, parent="chain_parent"),
        ],
        sample_counts=counts(leaf=0, chain=0),
    )
    assert [s.id for s in by_puct.sample_states(1)] == ["leaf"]

    by_order = make_sampler(
        tmp_path / "order",
        [
            state("chain_parent", value=10.0),
            state("chain_child", value=0.0, parent="chain_parent"),
            state("leaf", value=10.0),
        ],
        sample_counts=counts(leaf=0, chain=0),
    )
    assert [s.id for s in by_order.sample_states(1)] == ["leaf"]


def test_within_selected_surface_highest_puct_wins(tmp_path: Path) -> None:
    sampler = make_sampler(
        tmp_path,
        [
            state("leaf", value=100.0),
            state("chain_low", value=5.0),
            state("chain_low_child", value=0.0, parent="chain_low"),
            state("chain_high", value=7.0),
            state("chain_high_child", value=0.0, parent="chain_high"),
        ],
        sample_counts=counts(leaf=10, chain=0),
    )

    assert [s.id for s in sampler.sample_states(1)] == ["chain_high"]


def test_batch_lineage_blocking_recomputes_availability(tmp_path: Path) -> None:
    sampler = make_sampler(
        tmp_path,
        [
            state("ancestor", value=100.0),
            state("descendant", value=90.0, parent="ancestor"),
            state("other_leaf", value=80.0),
        ],
        sample_counts=counts(),
    )

    assert [s.id for s in sampler.sample_states(2)] == ["ancestor", "other_leaf"]


def test_enabled_persistence_resume_sanitize_and_disabled_ignore(tmp_path: Path) -> None:
    states = [state("leaf", value=1.0)]
    sampler = make_sampler(
        tmp_path,
        states,
        sample_counts=counts(leaf=2, chain=3, branch_2_3=4, branch_4_plus=5),
    )
    sampler.flush(step=1)

    saved = sampler_json(tmp_path, 1)
    assert saved["sampler_diversity_method"] == "subtree_frontier_balance"
    assert saved["subtree_frontier_sample_counts"] == counts(
        leaf=2,
        chain=3,
        branch_2_3=4,
        branch_4_plus=5,
    )

    reloaded = PUCTSampler(
        file_path=str(tmp_path / "puct_sampler.json"),
        env_type=DummyEnv,
        batch_size=0,
        resume_step=1,
        sampler_diversity_method="subtree_frontier_balance",
    )
    assert reloaded._subtree_frontier_sample_counts == saved["subtree_frontier_sample_counts"]

    bad = saved.copy()
    bad["step"] = 2
    bad["subtree_frontier_sample_counts"] = {
        "leaf": True,
        "chain": -2,
        "branch_2_3": 1.5,
        "branch_4_plus": 7,
        "unknown": 99,
    }
    Path(_sampler_file_for_step(str(tmp_path / "puct_sampler.json"), 2)).write_text(
        json.dumps(bad),
        encoding="utf-8",
    )
    sanitized = PUCTSampler(
        file_path=str(tmp_path / "puct_sampler.json"),
        env_type=DummyEnv,
        batch_size=0,
        resume_step=2,
        sampler_diversity_method="subtree_frontier_balance",
    )
    assert sanitized._subtree_frontier_sample_counts == counts(branch_4_plus=7)
    sanitized._subtree_frontier_sample_counts.update(
        {
            "leaf": True,
            "chain": -2,
            "branch_2_3": 1.5,
            "branch_4_plus": 7,
            "unknown": 99,
        }
    )
    sanitized.flush(step=4)
    sanitized_saved = sampler_json(tmp_path, 4)
    assert sanitized_saved["subtree_frontier_sample_counts"] == counts(branch_4_plus=7)

    missing = saved.copy()
    missing["step"] = 3
    missing.pop("subtree_frontier_sample_counts")
    Path(_sampler_file_for_step(str(tmp_path / "puct_sampler.json"), 3)).write_text(
        json.dumps(missing),
        encoding="utf-8",
    )
    missing_reloaded = PUCTSampler(
        file_path=str(tmp_path / "puct_sampler.json"),
        env_type=DummyEnv,
        batch_size=0,
        resume_step=3,
        sampler_diversity_method="subtree_frontier_balance",
    )
    assert missing_reloaded._subtree_frontier_sample_counts == counts()

    disabled = PUCTSampler(
        file_path=str(tmp_path / "puct_sampler.json"),
        env_type=DummyEnv,
        batch_size=0,
        resume_step=1,
        sampler_diversity_method="none",
    )
    disabled.flush(step=5)
    disabled_saved = sampler_json(tmp_path, 5)
    assert "sampler_diversity_method" not in disabled_saved
    assert "subtree_frontier_sample_counts" not in disabled_saved


def test_record_failed_rollout_does_not_double_count_surface(tmp_path: Path) -> None:
    parent = state("leaf", value=1.0)
    sampler = make_sampler(tmp_path, [parent], sample_counts=counts())

    assert [s.id for s in sampler.sample_states(1)] == ["leaf"]
    assert sampler._subtree_frontier_sample_counts["leaf"] == 1
    sampler.record_failed_rollout(parent)
    assert sampler._subtree_frontier_sample_counts["leaf"] == 1
    assert sampler._n["leaf"] == 1
    assert sampler._T == 1


def test_enabled_observability_is_gated_and_populated(tmp_path: Path) -> None:
    sampler = make_sampler(tmp_path, [state("leaf", value=1.0)], sample_counts=counts())

    sampler.sample_states(1)
    stats = sampler.get_sample_stats()
    for surface in SUBTREE_FRONTIER_SURFACES:
        assert f"puct/subtree_frontier/sample_count/{surface}" in stats
        assert f"puct/subtree_frontier/buffer_count/{surface}" in stats
        assert f"puct/subtree_frontier/last_selected_count/{surface}" in stats
    assert stats["puct/subtree_frontier/available_surfaces"] == 1
    assert stats["puct/subtree_frontier/fallback_last"] == 0

    columns, rows = sampler.get_sample_table()
    assert columns[-4:] == [
        "subtree_frontier_surface",
        "subtree_frontier_leaves",
        "subtree_frontier_sample_count",
        "subtree_frontier_selected_by",
    ]
    assert rows[0][-4:] == ("leaf", 1, 1, "surface")


def test_invalid_sampler_method_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        make_sampler(tmp_path, [], method="bad")


def test_cli_dry_run_and_build_sampler_pass_through(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    script = ROOT / "repro" / "run_discovery.py"
    default = subprocess.run(
        [sys.executable, str(script), "dry-run"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    default_payload = json.loads(default.stdout)
    assert default_payload["codex_sampler_diversity_method"] == "none"
    assert default_payload["codex_model_name"] is None
    assert default_payload["num_cpus_per_task"] == 1

    enabled = subprocess.run(
        [
            sys.executable,
            str(script),
            "--dry-run",
            "--codex-sampler-diversity-method",
            "subtree_frontier_balance",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert json.loads(enabled.stdout)["codex_sampler_diversity_method"] == "subtree_frontier_balance"

    cpus = subprocess.run(
        [sys.executable, str(script), "dry-run", "--num-cpus-per-task", "3"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert json.loads(cpus.stdout)["num_cpus_per_task"] == 3

    invalid = subprocess.run(
        [sys.executable, str(script), "dry-run", "--codex-sampler-diversity-method", "bad"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert invalid.returncode != 0

    from ttt_discover.rl import codex_no_finetune

    captured = {}

    def fake_create_sampler(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(codex_no_finetune, "create_sampler", fake_create_sampler)
    cfg = codex_no_finetune.CodexNoFinetuneConfig(
        env_type=DummyEnv,
        log_path=str(tmp_path),
        sampler_diversity_method="subtree_frontier_balance",
    )
    codex_no_finetune._build_sampler(cfg, start_batch=0)
    assert captured["sampler_diversity_method"] == "subtree_frontier_balance"


def test_gpu_mode_wrapper_dry_run_and_override() -> None:
    wrapper = ROOT / "repro" / "gpu_mode" / "run_0609_subtree_frontier_balance.sh"
    enabled = subprocess.run(
        ["bash", str(wrapper), "dry-run"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert json.loads(enabled.stdout)["codex_sampler_diversity_method"] == "subtree_frontier_balance"
    assert json.loads(enabled.stdout)["codex_model_name"] is None
    assert json.loads(enabled.stdout)["wandb_project"] is None

    override = subprocess.run(
        ["bash", str(wrapper), "dry-run", "--codex-sampler-diversity-method", "none"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert json.loads(override.stdout)["codex_sampler_diversity_method"] == "none"
````

## Diff Stat

### Worktree Tracked Diff Stat

````text
 ttt_discover/codex_utils/discovery.py |   2 +
 ttt_discover/codex_utils/sampler.py   | 322 ++++++++++++++++++++++++++++++++--
 ttt_discover/discovery.py             |   2 +
 ttt_discover/rl/codex_no_finetune.py  |   2 +
 4 files changed, 312 insertions(+), 16 deletions(-)
````
## Raw Diff

<details>
<summary>Tracked worktree patch</summary>

````diff
diff --git a/ttt_discover/codex_utils/discovery.py b/ttt_discover/codex_utils/discovery.py
index b21a80f..6d29c26 100644
--- a/ttt_discover/codex_utils/discovery.py
+++ b/ttt_discover/codex_utils/discovery.py
@@ -49,6 +49,7 @@ class DiscoverConfig:
     codex_max_concurrent_requests: int | None = 4
     codex_initial_program_paths: tuple[str, ...] = ()
     codex_initial_pool_paths: tuple[str, ...] = ()
+    codex_sampler_diversity_method: Literal["none", "subtree_frontier_balance"] = "none"
     codex_autonomous: bool = False
 
 
@@ -84,6 +85,7 @@ def _run_codex_no_finetune(config: DiscoverConfig) -> None:
         max_concurrent_requests=config.codex_max_concurrent_requests,
         initial_program_paths=config.codex_initial_program_paths,
         initial_pool_paths=config.codex_initial_pool_paths,
+        sampler_diversity_method=config.codex_sampler_diversity_method,
         autonomous=config.codex_autonomous,
         wandb_project=config.wandb_project,
         wandb_name=experiment_name,
diff --git a/ttt_discover/codex_utils/sampler.py b/ttt_discover/codex_utils/sampler.py
index 5c5d4d1..23c8048 100644
--- a/ttt_discover/codex_utils/sampler.py
+++ b/ttt_discover/codex_utils/sampler.py
@@ -16,6 +16,103 @@ from ttt_discover.codex_utils.runtime import State, state_from_dict, to_json_ser
 
 logger = logging.getLogger(__name__)
 
+SUBTREE_FRONTIER_SURFACES = ("leaf", "chain", "branch_2_3", "branch_4_plus")
+SUBTREE_FRONTIER_BALANCE_METHOD = "subtree_frontier_balance"
+SAMPLER_DIVERSITY_METHODS = ("none", SUBTREE_FRONTIER_BALANCE_METHOD)
+
+
+def _validate_sampler_diversity_method(method: str) -> str:
+    if method not in SAMPLER_DIVERSITY_METHODS:
+        valid = ", ".join(SAMPLER_DIVERSITY_METHODS)
+        raise ValueError(f"Unsupported sampler_diversity_method={method!r}; expected one of: {valid}")
+    return method
+
+
+def _first_parent_id(parents: Any) -> str | None:
+    if isinstance(parents, list):
+        if not parents:
+            return None
+        first_parent = parents[0]
+    elif isinstance(parents, dict):
+        first_parent = parents
+    else:
+        return None
+
+    if isinstance(first_parent, dict):
+        parent_id = first_parent.get("id")
+    else:
+        parent_id = first_parent
+    return str(parent_id) if parent_id else None
+
+
+def build_subtree_frontier_children_map(states: list[State]) -> dict[str, list[str]]:
+    retained = {str(s.id) for s in states if s.id is not None}
+    children = {str(s.id): [] for s in states if s.id is not None}
+    for child in states:
+        child_id = str(child.id)
+        parent_id = _first_parent_id(child.parents)
+        if child_id in retained and parent_id in retained and parent_id != child_id:
+            children[parent_id].append(child_id)
+    return children
+
+
+def subtree_frontier_leaf_count(state_id: str, children: dict[str, list[str]]) -> int:
+    memo: dict[str, int] = {}
+    active: set[str] = set()
+
+    def _count(sid: str) -> int:
+        if sid in active:
+            return 0
+        if sid in memo:
+            return memo[sid]
+        child_ids = children.get(sid, [])
+        if not child_ids:
+            memo[sid] = 1
+            return 1
+        active.add(sid)
+        total = 0
+        for child_id in child_ids:
+            total += _count(child_id)
+        active.remove(sid)
+        memo[sid] = total
+        return total
+
+    return _count(str(state_id))
+
+
+def classify_subtree_frontier_surface(
+    state_id: str,
+    children: dict[str, list[str]],
+) -> tuple[str, int]:
+    sid = str(state_id)
+    leaves = subtree_frontier_leaf_count(sid, children)
+    if not children.get(sid):
+        surface = "leaf"
+    elif leaves == 1:
+        surface = "chain"
+    elif leaves <= 3:
+        surface = "branch_2_3"
+    else:
+        surface = "branch_4_plus"
+    return surface, leaves
+
+
+def _zero_subtree_frontier_counts() -> dict[str, int]:
+    return {surface: 0 for surface in SUBTREE_FRONTIER_SURFACES}
+
+
+def _sanitize_subtree_frontier_sample_counts(raw_counts: Any) -> dict[str, int]:
+    counts = _zero_subtree_frontier_counts()
+    if not isinstance(raw_counts, dict):
+        return counts
+    for surface in SUBTREE_FRONTIER_SURFACES:
+        value = raw_counts.get(surface, 0)
+        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
+            counts[surface] = 0
+        else:
+            counts[surface] = value
+    return counts
+
 
 @contextmanager
 def _file_lock(lock_path: str, *, poll_s: float = 0.05, stale_s: float = 600.0):
@@ -353,6 +450,7 @@ class PUCTSampler(StateSampler):
         resume_step: int | None = None,
         puct_c: float = 1.0,
         topk_children: int = 2,
+        sampler_diversity_method: str = "none",
     ):
         self.file_path = file_path
         self.env_type = env_type
@@ -361,6 +459,10 @@ class PUCTSampler(StateSampler):
         self.batch_size = batch_size
         self.topk_children = topk_children
         self.puct_c = float(puct_c)
+        self.sampler_diversity_method = _validate_sampler_diversity_method(sampler_diversity_method)
+        self._subtree_frontier_enabled = (
+            self.sampler_diversity_method == SUBTREE_FRONTIER_BALANCE_METHOD
+        )
         
         self._states: list[State] = []
         self._initial_states: list[State] = []
@@ -375,6 +477,11 @@ class PUCTSampler(StateSampler):
         self._T: int = 0
         self._last_scale: float = 1.0
         self._last_puct_stats: list[tuple[int, float, float, float, float]] = []
+        self._subtree_frontier_sample_counts: dict[str, int] = _zero_subtree_frontier_counts()
+        self._last_subtree_frontier_metadata: list[tuple[str, int, int, str]] = []
+        self._last_subtree_frontier_selected_counts: dict[str, int] = _zero_subtree_frontier_counts()
+        self._last_subtree_frontier_available_surfaces: int = 0
+        self._last_subtree_frontier_fallback: bool = False
         
         if resume_step is not None:
             self._load(resume_step)
@@ -399,6 +506,10 @@ class PUCTSampler(StateSampler):
         self._n = store.get("puct_n", {}) or {}
         self._m = store.get("puct_m", {}) or {}
         self._T = int(store.get("puct_T", 0) or 0)
+        if self._subtree_frontier_enabled:
+            self._subtree_frontier_sample_counts = _sanitize_subtree_frontier_sample_counts(
+                store.get("subtree_frontier_sample_counts", {})
+            )
 
     def _save(self, step: int):
         save_path = _sampler_file_for_step(self.file_path, step)
@@ -411,6 +522,15 @@ class PUCTSampler(StateSampler):
             "puct_m": self._m,
             "puct_T": self._T,
         }
+        if self._subtree_frontier_enabled:
+            self._subtree_frontier_sample_counts = _sanitize_subtree_frontier_sample_counts(
+                self._subtree_frontier_sample_counts
+            )
+            store["sampler_diversity_method"] = SUBTREE_FRONTIER_BALANCE_METHOD
+            store["subtree_frontier_sample_counts"] = {
+                surface: self._subtree_frontier_sample_counts.get(surface, 0)
+                for surface in SUBTREE_FRONTIER_SURFACES
+            }
         with _file_lock(f"{save_path}.lock"):
             _atomic_write_json(save_path, store)
 
@@ -489,20 +609,11 @@ class PUCTSampler(StateSampler):
                     queue.append(child_id)
         return lineage
 
-    def sample_states(self, num_states: int) -> list[State]:
-        initial_ids = {s.id for s in self._initial_states}
-        candidates = list(self._states)
-
-        if not candidates:
-            picked = [
-                create_initial_state(self.env_type, self.problem_type)
-                for _ in range(num_states)
-            ]
-            self._last_sampled_states = picked
-            self._last_sampled_indices = []
-            self._last_puct_stats = [(0, 0.0, 0.0, 0.0, 0.0) for _ in picked]
-            return picked
-
+    def _score_candidates(
+        self,
+        candidates: list[State],
+        initial_ids: set[str],
+    ) -> list[tuple[float, float, State, int, float, float, float]]:
         vals = np.array([float(s.value if s.value is not None else float("-inf")) for s in candidates])
         non_initial_mask = np.array([s.id not in initial_ids for s in candidates])
         scale = self._compute_scale(vals, non_initial_mask if non_initial_mask.any() else None)
@@ -520,6 +631,134 @@ class PUCTSampler(StateSampler):
             scores.append((score, vals[i], s, n, Q, P[i], bonus))
 
         scores.sort(key=lambda x: (x[0], x[1]), reverse=True)
+        return scores
+
+    def _select_subtree_frontier_surface(
+        self,
+        buckets: dict[str, list[tuple[float, float, State, int, float, float, float]]],
+    ) -> str | None:
+        available = [surface for surface in SUBTREE_FRONTIER_SURFACES if buckets.get(surface)]
+        self._last_subtree_frontier_available_surfaces = len(available)
+        if not available:
+            return None
+
+        min_count = min(self._subtree_frontier_sample_counts.get(surface, 0) for surface in available)
+        tied = [
+            surface
+            for surface in SUBTREE_FRONTIER_SURFACES
+            if surface in available
+            and self._subtree_frontier_sample_counts.get(surface, 0) == min_count
+        ]
+        best_score = max(buckets[surface][0][0] for surface in tied)
+        for surface in SUBTREE_FRONTIER_SURFACES:
+            if surface in tied and buckets[surface][0][0] == best_score:
+                return surface
+        return None
+
+    def _sample_states_subtree_frontier(
+        self,
+        num_states: int,
+        candidates: list[State],
+        initial_ids: set[str],
+    ) -> list[State]:
+        scores = self._score_candidates(candidates, initial_ids)
+        children = build_subtree_frontier_children_map(self._states)
+        surface_cache: dict[str, tuple[str, int]] = {}
+
+        def _surface_for(state: State) -> tuple[str, int]:
+            sid = str(state.id)
+            if sid not in surface_cache:
+                surface_cache[sid] = classify_subtree_frontier_surface(sid, children)
+            return surface_cache[sid]
+
+        picked: list[State] = []
+        top_scores: list[tuple[float, float, State, int, float, float, float]] = []
+        selected_metadata: list[tuple[str, int, int, str]] = []
+        self._last_subtree_frontier_selected_counts = _zero_subtree_frontier_counts()
+        self._last_subtree_frontier_available_surfaces = 0
+        self._last_subtree_frontier_fallback = False
+
+        blocked_ids: set[str] = set()
+        children_map = self._build_children_map() if num_states > 1 else {}
+
+        while len(picked) < num_states:
+            selectable = [
+                entry
+                for entry in scores
+                if num_states <= 1 or entry[2].id not in blocked_ids
+            ]
+            if not selectable:
+                self._last_subtree_frontier_available_surfaces = 0
+                break
+
+            buckets: dict[str, list[tuple[float, float, State, int, float, float, float]]] = {
+                surface: [] for surface in SUBTREE_FRONTIER_SURFACES
+            }
+            for entry in selectable:
+                surface, _leaves = _surface_for(entry[2])
+                buckets[surface].append(entry)
+
+            selected_by = "surface"
+            selected_surface = self._select_subtree_frontier_surface(buckets)
+            if selected_surface is None:
+                selected_entry = selectable[0]
+                selected_surface, leaves = _surface_for(selected_entry[2])
+                selected_by = "fallback"
+            else:
+                selected_entry = buckets[selected_surface][0]
+                _surface, leaves = _surface_for(selected_entry[2])
+
+            selected_state = selected_entry[2]
+            picked.append(selected_state)
+            top_scores.append(selected_entry)
+            self._subtree_frontier_sample_counts[selected_surface] += 1
+            self._last_subtree_frontier_selected_counts[selected_surface] += 1
+            self._last_subtree_frontier_fallback = selected_by == "fallback"
+            selected_metadata.append(
+                (
+                    selected_surface,
+                    leaves,
+                    self._subtree_frontier_sample_counts[selected_surface],
+                    selected_by,
+                )
+            )
+
+            if num_states > 1:
+                blocked_ids.update(self._get_full_lineage(selected_state, children_map))
+            else:
+                break
+
+        state_id_to_idx = {s.id: i for i, s in enumerate(self._states)}
+        self._last_sampled_states = picked
+        self._last_sampled_indices = [state_id_to_idx.get(s.id, -1) for s in picked]
+        self._last_puct_stats = [(t[3], t[4], t[5], t[6], t[0]) for t in top_scores]
+        self._last_subtree_frontier_metadata = selected_metadata
+
+        for s in picked:
+            if s.id in initial_ids:
+                self._refresh_random_construction(s)
+
+        return picked
+
+    def sample_states(self, num_states: int) -> list[State]:
+        initial_ids = {s.id for s in self._initial_states}
+        candidates = list(self._states)
+
+        if not candidates:
+            picked = [
+                create_initial_state(self.env_type, self.problem_type)
+                for _ in range(num_states)
+            ]
+            self._last_sampled_states = picked
+            self._last_sampled_indices = []
+            self._last_puct_stats = [(0, 0.0, 0.0, 0.0, 0.0) for _ in picked]
+            self._last_subtree_frontier_metadata = []
+            return picked
+
+        if self._subtree_frontier_enabled:
+            return self._sample_states_subtree_frontier(num_states, candidates, initial_ids)
+
+        scores = self._score_candidates(candidates, initial_ids)
 
         if num_states > 1:
             children_map = self._build_children_map()
@@ -541,6 +780,7 @@ class PUCTSampler(StateSampler):
         self._last_sampled_states = picked
         self._last_sampled_indices = [state_id_to_idx.get(s.id, -1) for s in picked]
         self._last_puct_stats = [(t[3], t[4], t[5], t[6], t[0]) for t in top_scores]
+        self._last_subtree_frontier_metadata = []
 
         for s in picked:
             if s.id in initial_ids:
@@ -731,21 +971,67 @@ class PUCTSampler(StateSampler):
         stats.update(_stats(sampled_values, "puct/sampled_value"))
         stats.update(_stats(sampled_timesteps, "puct/sampled_timestep"))
         stats.update(_stats(sampled_constr_lens, "puct/sampled_construction_len"))
+        if self._subtree_frontier_enabled:
+            children = build_subtree_frontier_children_map(self._states)
+            buffer_counts = _zero_subtree_frontier_counts()
+            for state in self._states:
+                if state.id is None:
+                    continue
+                surface, _leaves = classify_subtree_frontier_surface(str(state.id), children)
+                buffer_counts[surface] += 1
+            stats["puct/subtree_frontier/available_surfaces"] = (
+                self._last_subtree_frontier_available_surfaces
+            )
+            stats["puct/subtree_frontier/fallback_last"] = int(self._last_subtree_frontier_fallback)
+            for surface in SUBTREE_FRONTIER_SURFACES:
+                stats[f"puct/subtree_frontier/sample_count/{surface}"] = (
+                    self._subtree_frontier_sample_counts.get(surface, 0)
+                )
+                stats[f"puct/subtree_frontier/buffer_count/{surface}"] = buffer_counts[surface]
+                stats[f"puct/subtree_frontier/last_selected_count/{surface}"] = (
+                    self._last_subtree_frontier_selected_counts.get(surface, 0)
+                )
         return stats
 
     def get_sample_table(self) -> tuple[list[str], list[tuple]]:
         columns = ["buffer_idx", "timestep", "value", "terminal_value", "parent_value", "construction_len", "observation_len", "n", "Q", "P", "bonus", "score"]
+        if self._subtree_frontier_enabled:
+            columns = columns + [
+                "subtree_frontier_surface",
+                "subtree_frontier_leaves",
+                "subtree_frontier_sample_count",
+                "subtree_frontier_selected_by",
+            ]
         rows = []
         if not self._last_sampled_states:
             return columns, rows
         indices = self._last_sampled_indices if len(self._last_sampled_indices) == len(self._last_sampled_states) else [-1] * len(self._last_sampled_states)
         stats = self._last_puct_stats if len(self._last_puct_stats) == len(self._last_sampled_states) else [(0, 0.0, 0.0, 0.0, 0.0)] * len(self._last_sampled_states)
-        for idx, state, (n, Q, P, bonus, score) in zip(indices, self._last_sampled_states, stats):
+        metadata = self._last_subtree_frontier_metadata
+        if self._subtree_frontier_enabled and len(metadata) != len(self._last_sampled_states):
+            children = build_subtree_frontier_children_map(self._states)
+            metadata = []
+            for state in self._last_sampled_states:
+                surface, leaves = classify_subtree_frontier_surface(str(state.id), children)
+                metadata.append(
+                    (
+                        surface,
+                        leaves,
+                        self._subtree_frontier_sample_counts.get(surface, 0),
+                        "surface",
+                    )
+                )
+        elif not self._subtree_frontier_enabled:
+            metadata = []
+        for row_idx, (idx, state, (n, Q, P, bonus, score)) in enumerate(zip(indices, self._last_sampled_states, stats)):
             parent_val = state.parent_values[0] if state.parent_values else None
             constr = getattr(state, 'construction', None)
             constr_len = len(constr) if constr is not None else 0
             obs_len = len(state.observation) if state.observation else 0
-            rows.append((idx, state.timestep, state.value, 0, parent_val, constr_len, obs_len, n, Q, P, bonus, score))
+            row = (idx, state.timestep, state.value, 0, parent_val, constr_len, obs_len, n, Q, P, bonus, score)
+            if self._subtree_frontier_enabled:
+                row = row + metadata[row_idx]
+            rows.append(row)
         return columns, rows
 
 
@@ -756,6 +1042,7 @@ def create_sampler(
     batch_size: int = 1,
     resume_step: int | None = None,
     topk_children: int = 2,
+    sampler_diversity_method: str = "none",
 ) -> StateSampler:
     """Factory function to create samplers. Pass the env type (from config.env_type)."""
     if not log_path:
@@ -768,6 +1055,7 @@ def create_sampler(
         batch_size=batch_size,
         resume_step=resume_step,
         topk_children=topk_children,
+        sampler_diversity_method=sampler_diversity_method,
     )
 
 
@@ -778,6 +1066,7 @@ def get_or_create_sampler_with_default(
     batch_size: int = 1,
     resume_step: int | None = None,
     topk_children: int = 2,
+    sampler_diversity_method: str = "none",
 ) -> StateSampler:
     """Get sampler. Initial experience is created via env_type.create_initial_state."""
     return create_sampler(
@@ -787,4 +1076,5 @@ def get_or_create_sampler_with_default(
         batch_size=batch_size,
         resume_step=resume_step,
         topk_children=topk_children,
+        sampler_diversity_method=sampler_diversity_method,
     )
diff --git a/ttt_discover/discovery.py b/ttt_discover/discovery.py
index 6cdb434..5e1455c 100644
--- a/ttt_discover/discovery.py
+++ b/ttt_discover/discovery.py
@@ -62,6 +62,7 @@ class DiscoverConfig:
     codex_max_concurrent_requests: int | None = 4
     codex_initial_program_paths: tuple[str, ...] = ()
     codex_initial_pool_paths: tuple[str, ...] = ()
+    codex_sampler_diversity_method: Literal["none", "subtree_frontier_balance"] = "none"
     codex_autonomous: bool = False
 
 
@@ -145,6 +146,7 @@ async def discover_impl(config: DiscoverConfig):
             max_concurrent_requests=config.codex_max_concurrent_requests,
             initial_program_paths=config.codex_initial_program_paths,
             initial_pool_paths=config.codex_initial_pool_paths,
+            sampler_diversity_method=config.codex_sampler_diversity_method,
             autonomous=config.codex_autonomous,
             wandb_project=config.wandb_project,
             wandb_name=config.experiment_name,
diff --git a/ttt_discover/rl/codex_no_finetune.py b/ttt_discover/rl/codex_no_finetune.py
index 18a0e9e..c4382b9 100644
--- a/ttt_discover/rl/codex_no_finetune.py
+++ b/ttt_discover/rl/codex_no_finetune.py
@@ -137,6 +137,7 @@ class CodexNoFinetuneConfig:
     initial_program_paths: tuple[str, ...] = ()
     initial_pool_paths: tuple[str, ...] = ()
     topk_children: int = 16
+    sampler_diversity_method: Literal["none", "subtree_frontier_balance"] = "none"
 
     # False is TTT Discover: non-auto, many short samples. True is AutoEvolve:
     # one Codex deep-dive workspace per sample with file edits and shell access.
@@ -960,6 +961,7 @@ def _build_sampler(cfg: CodexNoFinetuneConfig, start_batch: int) -> StateSampler
         batch_size=cfg.groups_per_batch,
         resume_step=start_batch if start_batch > 0 else None,
         topk_children=cfg.topk_children,
+        sampler_diversity_method=cfg.sampler_diversity_method,
     )
 
 
````
</details>

