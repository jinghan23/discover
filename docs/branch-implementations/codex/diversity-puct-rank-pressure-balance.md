# codex/diversity-puct-rank-pressure-balance

## Summary

新增独立 rank-pressure helper，根据候选在 PUCT 排序中的压力/相邻 rank 情况分类 surface；平衡 surface 时仍在每个 surface 内按 PUCT 顺序取样。

## Branch State

- Worktree: `/opt/tiger/discover-puct-rank-pressure-balance`
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

- `codex_puct_rank_pressure_balance`
- `codex_puct_rank_pressure_rank_slack`
- `codex_puct_rank_pressure_min_candidates`
- `rank_pressure_balance`
- `rank_pressure_rank_slack`
- `rank_pressure_min_candidates`
- `puct_rank_pressure_balance`
- `puct_rank_pressure_rank_slack`
- `puct_rank_pressure_min_candidates`

### Constants

- None

### Classes

- None

### Functions

- `_select_baseline_entries`
- `_select_rank_pressure_entries`

## Diff Summary

- Worktree tracked shortstat: `4 files changed, 199 insertions(+), 17 deletions(-)`
- Untracked files: `4`

### Worktree Status

````text
 M ttt_discover/codex_utils/discovery.py
 M ttt_discover/codex_utils/sampler.py
 M ttt_discover/discovery.py
 M ttt_discover/rl/codex_no_finetune.py
?? repro/gpu_mode/
?? repro/run_discovery.py
?? tests/
?? ttt_discover/puct_rank_pressure.py
````
### Tracked Worktree Files

````text
M	ttt_discover/codex_utils/discovery.py
M	ttt_discover/codex_utils/sampler.py
M	ttt_discover/discovery.py
M	ttt_discover/rl/codex_no_finetune.py
````
### Untracked Files

- `repro/gpu_mode/run_0608_puct_rank_pressure_balance.sh (948 bytes)`
- `repro/run_discovery.py (7631 bytes)`
- `tests/test_puct_rank_pressure_balance.py (16875 bytes)`
- `ttt_discover/puct_rank_pressure.py (2998 bytes)`

### Detected Test Functions

- `tests/test_puct_rank_pressure_balance.py::test_disabled_mode_matches_baseline_and_does_not_increment_counts`
- `tests/test_puct_rank_pressure_balance.py::test_pure_classifier_surfaces_and_small_candidate_residual`
- `tests/test_puct_rank_pressure_balance.py::test_rank_ties_preserve_baseline_order`
- `tests/test_puct_rank_pressure_balance.py::test_least_sampled_surface_beats_higher_puct_oversampled_surface`
- `tests/test_puct_rank_pressure_balance.py::test_within_surface_baseline_puct_order_is_preserved`
- `tests/test_puct_rank_pressure_balance.py::test_full_lineage_blocking_prevents_related_pairs`
- `tests/test_puct_rank_pressure_balance.py::test_global_puct_order_fills_after_under_sampled_surface_exhausts`
- `tests/test_puct_rank_pressure_balance.py::test_edge_cases_for_requested_sample_count_and_empty_pool`
- `tests/test_puct_rank_pressure_balance.py::test_counts_persist_missing_key_loads_zero_and_bad_values_are_sanitized`
- `tests/test_puct_rank_pressure_balance.py::test_config_threading_and_cli_dry_run_values`
- `tests/test_puct_rank_pressure_balance.py::test_sample_table_and_metrics_include_pressure_diagnostics`
- `tests/test_puct_rank_pressure_balance.py::test_pressure_ranks_ignore_code_output_and_external_parent_metadata`

## Untracked File Snapshots

These files are not part of `git diff` yet, but they are part of the worktree implementation state recorded here.

### `repro/gpu_mode/run_0608_puct_rank_pressure_balance.sh`

````bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-gpu-mode-trimul-puct-rank-pressure-balance-0608}"
NUM_EPOCHS="${NUM_EPOCHS:-50}"
GROUP_SIZE="${GROUP_SIZE:-1}"
GROUPS_PER_BATCH="${GROUPS_PER_BATCH:-1}"
LOG_ROOT="${LOG_ROOT:-tinker_log}"
WANDB_PROJECT="${WANDB_PROJECT:-}"
GPU="${GPU:-}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"

args=(
  --task trimul
  --runner codex_no_finetune
  --experiment-name "${EXPERIMENT_NAME}"
  --log-root "${LOG_ROOT}"
  --num-epochs "${NUM_EPOCHS}"
  --group-size "${GROUP_SIZE}"
  --groups-per-batch "${GROUPS_PER_BATCH}"
  --wandb-project "${WANDB_PROJECT}"
  --torch-cuda-arch-list "${TORCH_CUDA_ARCH_LIST}"
  --codex-puct-rank-pressure-balance
)

if [[ -n "${GPU}" ]]; then
  args+=(--gpu "${GPU}")
fi

python repro/run_discovery.py "${args[@]}" "$@"
````

### `repro/run_discovery.py`

````python
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ttt_discover.puct_rank_pressure import clamp_nonnegative_int


@dataclass(frozen=True)
class TaskSpec:
    env_module: str
    env_name: str
    problem_type: str
    experiment_name: str
    eval_timeout: int
    num_epochs: int
    group_size: int
    groups_per_batch: int

    @property
    def env_path(self) -> str:
        return f"{self.env_module}.{self.env_name}"

    def load_env(self) -> type:
        module = importlib.import_module(self.env_module)
        return getattr(module, self.env_name)


def _task_spec(task: str) -> TaskSpec:
    if task in {"trimul", "gpu_mode:trimul"}:
        return TaskSpec(
            env_module="examples.gpu_mode.env",
            env_name="GpuModeEnv",
            problem_type="trimul",
            experiment_name="gpu-mode-trimul",
            eval_timeout=1200,
            num_epochs=50,
            group_size=1,
            groups_per_batch=1,
        )
    raise ValueError(f"Unknown task: {task}")


def _none_if_empty(value: str | None) -> str | None:
    if value is None or value == "":
        return None
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch a TTT-Discover task through a shared repro entrypoint."
    )
    parser.add_argument("command", nargs="?", choices=("dry-run",))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--task", default="trimul", choices=("trimul", "gpu_mode:trimul"))
    parser.add_argument("--runner", choices=("tinker_rl", "codex_no_finetune"), default="codex_no_finetune")
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--log-root", default="tinker_log")
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--group-size", type=int, default=None)
    parser.add_argument("--groups-per-batch", type=int, default=None)
    parser.add_argument("--num-cpus-per-task", type=int, default=1)
    parser.add_argument("--eval-timeout", type=int, default=None)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--torch-cuda-arch-list", default=None)

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
    parser.add_argument(
        "--codex-puct-rank-pressure-balance",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--codex-puct-rank-pressure-rank-slack",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--codex-puct-rank-pressure-min-candidates",
        type=int,
        default=4,
    )
    return parser.parse_args()


def _resolved(args: argparse.Namespace) -> dict:
    spec = _task_spec(args.task)
    experiment_name = args.experiment_name or spec.experiment_name
    log_path = str(Path(args.log_root) / experiment_name)
    return {
        "task": args.task,
        "env_type": spec.env_path,
        "problem_type": spec.problem_type,
        "runner": args.runner,
        "experiment_name": experiment_name,
        "log_path": log_path,
        "num_epochs": args.num_epochs if args.num_epochs is not None else spec.num_epochs,
        "group_size": args.group_size if args.group_size is not None else spec.group_size,
        "groups_per_batch": (
            args.groups_per_batch
            if args.groups_per_batch is not None
            else spec.groups_per_batch
        ),
        "num_cpus_per_task": args.num_cpus_per_task,
        "eval_timeout": args.eval_timeout if args.eval_timeout is not None else spec.eval_timeout,
        "wandb_project": _none_if_empty(args.wandb_project),
        "gpu": _none_if_empty(args.gpu),
        "torch_cuda_arch_list": _none_if_empty(args.torch_cuda_arch_list),
        "codex_backend": args.codex_backend,
        "codex_model_name": _none_if_empty(args.codex_model_name),
        "codex_max_output_tokens": args.codex_max_output_tokens,
        "codex_temperature": args.codex_temperature,
        "codex_cli_command": args.codex_cli_command,
        "codex_cli_sandbox": args.codex_cli_sandbox,
        "codex_cli_timeout": args.codex_cli_timeout,
        "codex_max_concurrent_requests": args.codex_max_concurrent_requests,
        "codex_autonomous": bool(args.codex_autonomous),
        "codex_puct_rank_pressure_balance": bool(
            args.codex_puct_rank_pressure_balance
        ),
        "codex_puct_rank_pressure_rank_slack": clamp_nonnegative_int(
            args.codex_puct_rank_pressure_rank_slack
        ),
        "codex_puct_rank_pressure_min_candidates": clamp_nonnegative_int(
            args.codex_puct_rank_pressure_min_candidates
        ),
    }


def main() -> None:
    args = parse_args()
    resolved = _resolved(args)
    if args.command == "dry-run" or args.dry_run:
        print(json.dumps(resolved, indent=2, sort_keys=True))
        return

    gpu = resolved["gpu"]
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    arch_list = resolved["torch_cuda_arch_list"]
    if arch_list is not None:
        os.environ["TORCH_CUDA_ARCH_LIST"] = arch_list

    from ttt_discover import DiscoverConfig, discover

    spec = _task_spec(args.task)
    config = DiscoverConfig(
        runner=resolved["runner"],
        env_type=spec.load_env(),
        problem_type=resolved["problem_type"],
        experiment_name=resolved["experiment_name"],
        wandb_project=resolved["wandb_project"],
        num_epochs=resolved["num_epochs"],
        group_size=resolved["group_size"],
        groups_per_batch=resolved["groups_per_batch"],
        num_cpus_per_task=resolved["num_cpus_per_task"],
        eval_timeout=resolved["eval_timeout"],
        codex_backend=resolved["codex_backend"],
        codex_model_name=resolved["codex_model_name"],
        codex_max_output_tokens=resolved["codex_max_output_tokens"],
        codex_temperature=resolved["codex_temperature"],
        codex_cli_command=resolved["codex_cli_command"],
        codex_cli_sandbox=resolved["codex_cli_sandbox"],
        codex_cli_timeout=resolved["codex_cli_timeout"],
        codex_max_concurrent_requests=resolved["codex_max_concurrent_requests"],
        codex_autonomous=resolved["codex_autonomous"],
        codex_puct_rank_pressure_balance=resolved[
            "codex_puct_rank_pressure_balance"
        ],
        codex_puct_rank_pressure_rank_slack=resolved[
            "codex_puct_rank_pressure_rank_slack"
        ],
        codex_puct_rank_pressure_min_candidates=resolved[
            "codex_puct_rank_pressure_min_candidates"
        ],
    )
    discover(config)


if __name__ == "__main__":
    main()
````

### `tests/test_puct_rank_pressure_balance.py`

````python
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from ttt_discover.codex_utils.runtime import State
from ttt_discover.codex_utils.sampler import PUCTSampler
from ttt_discover.puct_rank_pressure import (
    RANK_PRESSURE_SURFACE_ORDER,
    empty_rank_pressure_counts,
    rank_pressure_entry_stats,
    rank_pressure_surface,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class DummyEnv:
    env_name = "DummyEnv"
    state_type = State
    created = 0

    @classmethod
    def create_initial_state(cls, problem_type: str) -> State:
        cls.created += 1
        return State(
            timestep=-1,
            construction=[cls.created],
            code="",
            value=0.0,
            id=f"init-{cls.created}",
        )


def _state(
    state_id: str,
    value: float,
    *,
    parents: list[dict] | None = None,
    code: str = "",
    observation: str = "",
) -> State:
    return State(
        timestep=0,
        construction=[state_id],
        code=code,
        value=value,
        parents=parents or [],
        id=state_id,
        observation=observation,
    )


def _mix_states(*, external_metadata: bool = False) -> list[State]:
    states = [
        _state("A", 100),
        _state("B", 99),
        _state("C", 98),
        _state("D", 97),
        _state("E", 1),
    ]
    if external_metadata:
        for state in states:
            state.code = f"code for {state.id}"
            state.observation = f"output for {state.id}"
            state.parents = [{"id": f"external-{state.id}", "timestep": -2}]
    return states


def _make_sampler(
    tmpdir: str,
    *,
    balance: bool = False,
    states: list[State] | None = None,
    counts: dict[str, int] | None = None,
    slack: int = 0,
    min_candidates: int = 4,
) -> PUCTSampler:
    sampler = PUCTSampler(
        os.path.join(tmpdir, "puct_sampler.json"),
        DummyEnv,
        batch_size=0,
        puct_c=10,
        topk_children=0,
        rank_pressure_balance=balance,
        rank_pressure_rank_slack=slack,
        rank_pressure_min_candidates=min_candidates,
    )
    sampler._states = states if states is not None else _mix_states()
    sampler._initial_states = []
    sampler._n = {"A": 1, "B": 10, "C": 1, "D": 1, "E": 1}
    sampler._m = {"A": 0, "B": 100, "C": 80, "D": 75, "E": 90}
    if counts:
        sampler._rank_pressure_surface_sample_counts.update(counts)
    return sampler


def _ids(states: list[State]) -> list[str]:
    return [state.id for state in states]


def _surface_rows(sampler: PUCTSampler) -> list[tuple[str, str, int, int, int]]:
    columns, rows = sampler.get_sample_table()
    surface_idx = columns.index("rank_pressure_surface")
    puct_idx = columns.index("puct_rank")
    q_idx = columns.index("q_rank")
    bonus_idx = columns.index("bonus_rank")
    return [
        (
            sampler._states[row[0]].id,
            row[surface_idx],
            row[puct_idx],
            row[q_idx],
            row[bonus_idx],
        )
        for row in rows
    ]


class PuctRankPressureBalanceTests(unittest.TestCase):
    def test_disabled_mode_matches_baseline_and_does_not_increment_counts(self):
        with tempfile.TemporaryDirectory() as tmp_a, tempfile.TemporaryDirectory() as tmp_b:
            baseline = _make_sampler(tmp_a, balance=False)
            disabled = _make_sampler(tmp_b, balance=False)

            self.assertEqual(_ids(disabled.sample_states(3)), _ids(baseline.sample_states(3)))
            self.assertEqual(_ids(disabled._last_sampled_states), ["C", "A", "D"])
            self.assertEqual(
                disabled._rank_pressure_surface_sample_counts,
                empty_rank_pressure_counts(),
            )

    def test_pure_classifier_surfaces_and_small_candidate_residual(self):
        self.assertEqual(
            rank_pressure_surface(4, 1, 0, 3, 0, 4),
            "rank_pressure_exploit",
        )
        self.assertEqual(
            rank_pressure_surface(4, 1, 3, 0, 0, 4),
            "rank_pressure_explore",
        )
        self.assertEqual(
            rank_pressure_surface(4, 2, 2, 2, 0, 4),
            "rank_pressure_balanced",
        )
        self.assertEqual(
            rank_pressure_surface(4, 0, 1, 1, 0, 4),
            "rank_pressure_residual",
        )
        self.assertEqual(
            rank_pressure_surface(3, 0, 0, 0, 0, 4),
            "rank_pressure_residual",
        )

    def test_rank_ties_preserve_baseline_order(self):
        entries = [
            (10.0, 1.0, "A", 0, 5.0, 0.0, 2.0),
            (9.0, 1.0, "B", 0, 5.0, 0.0, 2.0),
            (8.0, 0.0, "C", 0, 1.0, 0.0, 1.0),
        ]
        stats = rank_pressure_entry_stats(entries, rank_slack=0, min_candidates=0)

        self.assertEqual(stats[id(entries[0])][1:], (0, 0, 0))
        self.assertEqual(stats[id(entries[1])][1:], (1, 1, 1))
        self.assertEqual(stats[id(entries[2])][1:], (2, 2, 2))

    def test_least_sampled_surface_beats_higher_puct_oversampled_surface(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sampler = _make_sampler(
                tmpdir,
                balance=True,
                counts={"rank_pressure_residual": 5},
            )

            picked = sampler.sample_states(1)

            self.assertEqual(_ids(picked), ["A"])
            self.assertEqual(_surface_rows(sampler)[0][:3], ("A", "rank_pressure_explore", 1))

    def test_within_surface_baseline_puct_order_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sampler = _make_sampler(
                tmpdir,
                balance=True,
                counts={
                    "rank_pressure_exploit": 10,
                    "rank_pressure_balanced": 10,
                    "rank_pressure_residual": 10,
                },
            )

            picked = sampler.sample_states(2)

            self.assertEqual(_ids(picked), ["A", "D"])
            self.assertEqual(
                [row[:3] for row in _surface_rows(sampler)],
                [
                    ("A", "rank_pressure_explore", 1),
                    ("D", "rank_pressure_explore", 2),
                ],
            )

    def test_full_lineage_blocking_prevents_related_pairs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            states = _mix_states()
            states[3].parents = [{"id": "A", "timestep": 0}]
            sampler = _make_sampler(
                tmpdir,
                balance=True,
                states=states,
                counts={
                    "rank_pressure_exploit": 10,
                    "rank_pressure_balanced": 10,
                    "rank_pressure_residual": 10,
                },
            )

            picked = sampler.sample_states(2)

            self.assertEqual(_ids(picked), ["A", "C"])
            self.assertNotIn("D", _ids(picked))

    def test_global_puct_order_fills_after_under_sampled_surface_exhausts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sampler = _make_sampler(
                tmpdir,
                balance=True,
                counts={
                    "rank_pressure_exploit": 10,
                    "rank_pressure_balanced": 10,
                    "rank_pressure_residual": 10,
                },
            )

            picked = sampler.sample_states(4)

            self.assertEqual(_ids(picked), ["A", "D", "C", "B"])
            self.assertEqual(
                sampler._rank_pressure_surface_sample_counts["rank_pressure_explore"],
                2,
            )
            self.assertEqual(
                sampler._rank_pressure_surface_sample_counts["rank_pressure_residual"],
                11,
            )
            self.assertEqual(
                sampler._rank_pressure_surface_sample_counts["rank_pressure_exploit"],
                11,
            )

    def test_edge_cases_for_requested_sample_count_and_empty_pool(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sampler = _make_sampler(tmpdir, balance=True)
            self.assertEqual(sampler.sample_states(0), [])
            self.assertEqual(sampler.sample_states(-2), [])
            self.assertEqual(
                sampler._rank_pressure_surface_sample_counts,
                empty_rank_pressure_counts(),
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            DummyEnv.created = 0
            sampler = _make_sampler(tmpdir, balance=True, states=[])
            picked = sampler.sample_states(2)
            self.assertEqual(_ids(picked), ["init-1", "init-2"])
            self.assertEqual(
                sampler._rank_pressure_surface_sample_counts,
                empty_rank_pressure_counts(),
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            sampler = _make_sampler(
                tmpdir,
                balance=True,
                counts={"rank_pressure_residual": 5},
            )
            self.assertEqual(_ids(sampler.sample_states(1)), ["A"])
            self.assertEqual(len(sampler._last_puct_stats), 1)

    def test_counts_persist_missing_key_loads_zero_and_bad_values_are_sanitized(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "puct_sampler.json")
            sampler = _make_sampler(tmpdir, balance=True)
            sampler.sample_states(2)
            sampler.flush(step=1)

            resumed = PUCTSampler(
                file_path,
                DummyEnv,
                batch_size=0,
                resume_step=1,
                rank_pressure_balance=True,
            )
            self.assertEqual(
                resumed._rank_pressure_surface_sample_counts["rank_pressure_residual"],
                1,
            )
            self.assertEqual(
                resumed._rank_pressure_surface_sample_counts["rank_pressure_explore"],
                1,
            )

            checkpoint = os.path.join(tmpdir, "puct_sampler_step_000001.json")
            with open(checkpoint, "r", encoding="utf-8") as f:
                store = json.load(f)
            store.pop("rank_pressure_surface_sample_counts", None)
            with open(checkpoint, "w", encoding="utf-8") as f:
                json.dump(store, f)

            old_resumed = PUCTSampler(
                file_path,
                DummyEnv,
                batch_size=0,
                resume_step=1,
                rank_pressure_balance=True,
            )
            self.assertEqual(
                old_resumed._rank_pressure_surface_sample_counts,
                empty_rank_pressure_counts(),
            )

            store["rank_pressure_surface_sample_counts"] = {
                "rank_pressure_exploit": -5,
                "rank_pressure_explore": True,
                "rank_pressure_balanced": "7",
                "rank_pressure_residual": 3,
                "rank_pressure_extra": 99,
            }
            with open(checkpoint, "w", encoding="utf-8") as f:
                json.dump(store, f)

            bad_resumed = PUCTSampler(
                file_path,
                DummyEnv,
                batch_size=0,
                resume_step=1,
                rank_pressure_balance=True,
            )
            self.assertEqual(
                bad_resumed._rank_pressure_surface_sample_counts,
                {
                    "rank_pressure_exploit": 0,
                    "rank_pressure_explore": 0,
                    "rank_pressure_balanced": 0,
                    "rank_pressure_residual": 3,
                },
            )

    def test_config_threading_and_cli_dry_run_values(self):
        from ttt_discover.codex_utils.discovery import DiscoverConfig as CodexDiscoverConfig
        from ttt_discover.rl.codex_no_finetune import CodexNoFinetuneConfig, _build_sampler

        codex_cfg = CodexNoFinetuneConfig(env_type=DummyEnv)
        self.assertFalse(codex_cfg.puct_rank_pressure_balance)
        self.assertEqual(codex_cfg.puct_rank_pressure_rank_slack, 1)
        self.assertEqual(codex_cfg.puct_rank_pressure_min_candidates, 4)

        codex_public_cfg = CodexDiscoverConfig(
            codex_puct_rank_pressure_balance=True,
            codex_puct_rank_pressure_rank_slack=2,
            codex_puct_rank_pressure_min_candidates=3,
        )
        self.assertTrue(codex_public_cfg.codex_puct_rank_pressure_balance)

        public_source = (REPO_ROOT / "ttt_discover/discovery.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("codex_puct_rank_pressure_balance: bool = False", public_source)
        self.assertIn("codex_puct_rank_pressure_rank_slack: int = 1", public_source)
        self.assertIn("codex_puct_rank_pressure_min_candidates: int = 4", public_source)
        self.assertIn(
            "puct_rank_pressure_balance=config.codex_puct_rank_pressure_balance",
            public_source,
        )
        self.assertIn(
            "puct_rank_pressure_rank_slack=config.codex_puct_rank_pressure_rank_slack",
            public_source,
        )
        self.assertIn(
            "puct_rank_pressure_min_candidates=config.codex_puct_rank_pressure_min_candidates",
            public_source,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            threaded_cfg = CodexNoFinetuneConfig(
                env_type=DummyEnv,
                log_path=tmpdir,
                puct_rank_pressure_balance=True,
                puct_rank_pressure_rank_slack=2,
                puct_rank_pressure_min_candidates=3,
            )
            sampler = _build_sampler(threaded_cfg, 0)
            self.assertTrue(sampler.rank_pressure_balance)
            self.assertEqual(sampler.rank_pressure_rank_slack, 2)
            self.assertEqual(sampler.rank_pressure_min_candidates, 3)

        default_out = subprocess.check_output(
            [sys.executable, "repro/run_discovery.py", "dry-run", "--task", "trimul"],
            cwd=REPO_ROOT,
            text=True,
        )
        default_resolved = json.loads(default_out)
        self.assertFalse(default_resolved["codex_puct_rank_pressure_balance"])
        self.assertEqual(default_resolved["codex_puct_rank_pressure_rank_slack"], 1)
        self.assertEqual(default_resolved["codex_puct_rank_pressure_min_candidates"], 4)

        custom_out = subprocess.check_output(
            [
                sys.executable,
                "repro/run_discovery.py",
                "dry-run",
                "--task",
                "trimul",
                "--codex-puct-rank-pressure-balance",
                "--codex-puct-rank-pressure-rank-slack",
                "2",
                "--codex-puct-rank-pressure-min-candidates",
                "3",
            ],
            cwd=REPO_ROOT,
            text=True,
        )
        custom_resolved = json.loads(custom_out)
        self.assertTrue(custom_resolved["codex_puct_rank_pressure_balance"])
        self.assertEqual(custom_resolved["codex_puct_rank_pressure_rank_slack"], 2)
        self.assertEqual(custom_resolved["codex_puct_rank_pressure_min_candidates"], 3)

    def test_sample_table_and_metrics_include_pressure_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sampler = _make_sampler(tmpdir, balance=True)
            sampler.sample_states(2)

            stats = sampler.get_sample_stats()
            for surface in RANK_PRESSURE_SURFACE_ORDER:
                self.assertIn(f"puct/rank_pressure_sample_count/{surface}", stats)

            columns, rows = sampler.get_sample_table()
            for column in ("rank_pressure_surface", "puct_rank", "q_rank", "bonus_rank"):
                self.assertIn(column, columns)
            self.assertEqual(len(rows), 2)

        with tempfile.TemporaryDirectory() as tmpdir:
            sampler = _make_sampler(tmpdir, balance=False)
            sampler._rank_pressure_surface_sample_counts["rank_pressure_explore"] = 1
            sampler.sample_states(1)
            columns, _ = sampler.get_sample_table()
            self.assertIn("rank_pressure_surface", columns)

    def test_pressure_ranks_ignore_code_output_and_external_parent_metadata(self):
        with tempfile.TemporaryDirectory() as tmp_a, tempfile.TemporaryDirectory() as tmp_b:
            sampler_a = _make_sampler(tmp_a, balance=True, states=_mix_states())
            sampler_b = _make_sampler(
                tmp_b,
                balance=True,
                states=_mix_states(external_metadata=True),
            )

            self.assertEqual(_ids(sampler_a.sample_states(5)), _ids(sampler_b.sample_states(5)))
            self.assertEqual(_surface_rows(sampler_a), _surface_rows(sampler_b))


if __name__ == "__main__":
    unittest.main()
````

### `ttt_discover/puct_rank_pressure.py`

````python
"""Rank-pressure helpers for PUCT sampling."""

from __future__ import annotations

from typing import Any


RANK_PRESSURE_SURFACE_ORDER = (
    "rank_pressure_exploit",
    "rank_pressure_explore",
    "rank_pressure_balanced",
    "rank_pressure_residual",
)


def clamp_nonnegative_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def empty_rank_pressure_counts() -> dict[str, int]:
    return {surface: 0 for surface in RANK_PRESSURE_SURFACE_ORDER}


def sanitize_rank_pressure_counts(raw: Any) -> dict[str, int]:
    counts = empty_rank_pressure_counts()
    if not isinstance(raw, dict):
        return counts
    for surface in RANK_PRESSURE_SURFACE_ORDER:
        value = raw.get(surface, 0)
        if isinstance(value, bool) or not isinstance(value, int):
            counts[surface] = 0
        else:
            counts[surface] = max(0, value)
    return counts


def rank_pressure_surface(
    candidate_count: int,
    puct_rank: int,
    q_rank: int,
    bonus_rank: int,
    rank_slack: int,
    min_candidates: int,
) -> str:
    rank_slack = clamp_nonnegative_int(rank_slack)
    min_candidates = clamp_nonnegative_int(min_candidates)
    if candidate_count < min_candidates:
        return "rank_pressure_residual"

    q_close = q_rank <= puct_rank + rank_slack
    bonus_close = bonus_rank <= puct_rank + rank_slack
    if q_close and bonus_close:
        return "rank_pressure_balanced"
    if q_close:
        return "rank_pressure_exploit"
    if bonus_close:
        return "rank_pressure_explore"
    return "rank_pressure_residual"


def rank_pressure_entry_stats(
    baseline_entries: list[Any],
    *,
    rank_slack: int,
    min_candidates: int,
) -> dict[int, tuple[str, int, int, int]]:
    """Return surface and ranks for baseline-ordered PUCT entries."""
    baseline_rank = {id(entry): rank for rank, entry in enumerate(baseline_entries)}
    q_order = sorted(
        baseline_entries,
        key=lambda entry: (-entry[4], -entry[1], baseline_rank[id(entry)]),
    )
    bonus_order = sorted(
        baseline_entries,
        key=lambda entry: (-entry[6], -entry[1], baseline_rank[id(entry)]),
    )
    q_rank = {id(entry): rank for rank, entry in enumerate(q_order)}
    bonus_rank = {id(entry): rank for rank, entry in enumerate(bonus_order)}
    candidate_count = len(baseline_entries)

    stats: dict[int, tuple[str, int, int, int]] = {}
    for entry in baseline_entries:
        entry_id = id(entry)
        puct_rank = baseline_rank[entry_id]
        qr = q_rank[entry_id]
        br = bonus_rank[entry_id]
        stats[entry_id] = (
            rank_pressure_surface(
                candidate_count,
                puct_rank,
                qr,
                br,
                rank_slack,
                min_candidates,
            ),
            puct_rank,
            qr,
            br,
        )
    return stats
````

## Diff Stat

### Worktree Tracked Diff Stat

````text
 ttt_discover/codex_utils/discovery.py |   6 ++
 ttt_discover/codex_utils/sampler.py   | 198 +++++++++++++++++++++++++++++++---
 ttt_discover/discovery.py             |   6 ++
 ttt_discover/rl/codex_no_finetune.py  |   6 ++
 4 files changed, 199 insertions(+), 17 deletions(-)
````
## Raw Diff

<details>
<summary>Tracked worktree patch</summary>

````diff
diff --git a/ttt_discover/codex_utils/discovery.py b/ttt_discover/codex_utils/discovery.py
index b21a80f..0b0ba61 100644
--- a/ttt_discover/codex_utils/discovery.py
+++ b/ttt_discover/codex_utils/discovery.py
@@ -50,6 +50,9 @@ class DiscoverConfig:
     codex_initial_program_paths: tuple[str, ...] = ()
     codex_initial_pool_paths: tuple[str, ...] = ()
     codex_autonomous: bool = False
+    codex_puct_rank_pressure_balance: bool = False
+    codex_puct_rank_pressure_rank_slack: int = 1
+    codex_puct_rank_pressure_min_candidates: int = 4
 
 
 def _run_codex_no_finetune(config: DiscoverConfig) -> None:
@@ -85,6 +88,9 @@ def _run_codex_no_finetune(config: DiscoverConfig) -> None:
         initial_program_paths=config.codex_initial_program_paths,
         initial_pool_paths=config.codex_initial_pool_paths,
         autonomous=config.codex_autonomous,
+        puct_rank_pressure_balance=config.codex_puct_rank_pressure_balance,
+        puct_rank_pressure_rank_slack=config.codex_puct_rank_pressure_rank_slack,
+        puct_rank_pressure_min_candidates=config.codex_puct_rank_pressure_min_candidates,
         wandb_project=config.wandb_project,
         wandb_name=experiment_name,
         log_path=log_path,
diff --git a/ttt_discover/codex_utils/sampler.py b/ttt_discover/codex_utils/sampler.py
index 5c5d4d1..224e2e3 100644
--- a/ttt_discover/codex_utils/sampler.py
+++ b/ttt_discover/codex_utils/sampler.py
@@ -12,6 +12,13 @@ from typing import Any, Callable
 
 import numpy as np
 
+from ttt_discover.puct_rank_pressure import (
+    RANK_PRESSURE_SURFACE_ORDER,
+    clamp_nonnegative_int,
+    empty_rank_pressure_counts,
+    rank_pressure_entry_stats,
+    sanitize_rank_pressure_counts,
+)
 from ttt_discover.codex_utils.runtime import State, state_from_dict, to_json_serializable
 
 logger = logging.getLogger(__name__)
@@ -353,6 +360,9 @@ class PUCTSampler(StateSampler):
         resume_step: int | None = None,
         puct_c: float = 1.0,
         topk_children: int = 2,
+        rank_pressure_balance: bool = False,
+        rank_pressure_rank_slack: int = 1,
+        rank_pressure_min_candidates: int = 4,
     ):
         self.file_path = file_path
         self.env_type = env_type
@@ -361,6 +371,9 @@ class PUCTSampler(StateSampler):
         self.batch_size = batch_size
         self.topk_children = topk_children
         self.puct_c = float(puct_c)
+        self.rank_pressure_balance = bool(rank_pressure_balance)
+        self.rank_pressure_rank_slack = clamp_nonnegative_int(rank_pressure_rank_slack)
+        self.rank_pressure_min_candidates = clamp_nonnegative_int(rank_pressure_min_candidates)
         
         self._states: list[State] = []
         self._initial_states: list[State] = []
@@ -375,6 +388,10 @@ class PUCTSampler(StateSampler):
         self._T: int = 0
         self._last_scale: float = 1.0
         self._last_puct_stats: list[tuple[int, float, float, float, float]] = []
+        self._last_rank_pressure_stats: list[tuple[str, int, int, int]] = []
+        self._rank_pressure_surface_sample_counts: dict[str, int] = (
+            empty_rank_pressure_counts()
+        )
         
         if resume_step is not None:
             self._load(resume_step)
@@ -399,6 +416,9 @@ class PUCTSampler(StateSampler):
         self._n = store.get("puct_n", {}) or {}
         self._m = store.get("puct_m", {}) or {}
         self._T = int(store.get("puct_T", 0) or 0)
+        self._rank_pressure_surface_sample_counts = sanitize_rank_pressure_counts(
+            store.get("rank_pressure_surface_sample_counts")
+        )
 
     def _save(self, step: int):
         save_path = _sampler_file_for_step(self.file_path, step)
@@ -410,6 +430,10 @@ class PUCTSampler(StateSampler):
             "puct_n": self._n,
             "puct_m": self._m,
             "puct_T": self._T,
+            "rank_pressure_surface_sample_counts": {
+                surface: self._rank_pressure_surface_sample_counts.get(surface, 0)
+                for surface in RANK_PRESSURE_SURFACE_ORDER
+            },
         }
         with _file_lock(f"{save_path}.lock"):
             _atomic_write_json(save_path, store)
@@ -489,7 +513,110 @@ class PUCTSampler(StateSampler):
                     queue.append(child_id)
         return lineage
 
+    def _select_baseline_entries(
+        self,
+        scores: list[tuple],
+        num_states: int,
+    ) -> list[tuple]:
+        if num_states <= 0:
+            return []
+        if num_states > 1:
+            children_map = self._build_children_map()
+            picked_entries: list[tuple] = []
+            blocked_ids: set[str] = set()
+            for entry in scores:
+                state = entry[2]
+                if state.id in blocked_ids:
+                    continue
+                picked_entries.append(entry)
+                blocked_ids.update(self._get_full_lineage(state, children_map))
+                if len(picked_entries) >= num_states:
+                    break
+            return picked_entries
+        return scores[:num_states]
+
+    def _select_rank_pressure_entries(
+        self,
+        scores: list[tuple],
+        rank_stats: dict[int, tuple[str, int, int, int]],
+        num_states: int,
+    ) -> list[tuple]:
+        if num_states <= 0:
+            return []
+
+        groups = {surface: [] for surface in RANK_PRESSURE_SURFACE_ORDER}
+        for entry in scores:
+            surface = rank_stats[id(entry)][0]
+            groups[surface].append(entry)
+
+        use_lineage_blocking = num_states > 1
+        children_map = self._build_children_map() if use_lineage_blocking else {}
+        picked_entries: list[tuple] = []
+        picked_ids: set[str] = set()
+        blocked_ids: set[str] = set()
+        local_counts = {
+            surface: self._rank_pressure_surface_sample_counts.get(surface, 0)
+            for surface in RANK_PRESSURE_SURFACE_ORDER
+        }
+
+        while len(picked_entries) < num_states:
+            choices = []
+            for surface in RANK_PRESSURE_SURFACE_ORDER:
+                first_available = None
+                for entry in groups[surface]:
+                    state = entry[2]
+                    if state.id in picked_ids:
+                        continue
+                    if use_lineage_blocking and state.id in blocked_ids:
+                        continue
+                    first_available = entry
+                    break
+                if first_available is None:
+                    continue
+                choices.append(
+                    (
+                        local_counts.get(surface, 0),
+                        rank_stats[id(first_available)][1],
+                        surface,
+                        first_available,
+                    )
+                )
+
+            if not choices:
+                break
+
+            _, _, surface, entry = min(choices, key=lambda item: (item[0], item[1]))
+            state = entry[2]
+            picked_entries.append(entry)
+            picked_ids.add(state.id)
+            local_counts[surface] = local_counts.get(surface, 0) + 1
+            if use_lineage_blocking:
+                blocked_ids.update(self._get_full_lineage(state, children_map))
+
+        if len(picked_entries) < num_states:
+            for entry in scores:
+                state = entry[2]
+                if state.id in picked_ids:
+                    continue
+                if use_lineage_blocking and state.id in blocked_ids:
+                    continue
+                picked_entries.append(entry)
+                picked_ids.add(state.id)
+                if use_lineage_blocking:
+                    blocked_ids.update(self._get_full_lineage(state, children_map))
+                if len(picked_entries) >= num_states:
+                    break
+
+        return picked_entries
+
     def sample_states(self, num_states: int) -> list[State]:
+        if num_states <= 0:
+            self._last_sampled_states = []
+            self._last_sampled_indices = []
+            self._last_puct_stats = []
+            self._last_rank_pressure_stats = []
+            return []
+
         initial_ids = {s.id for s in self._initial_states}
         candidates = list(self._states)
 
@@ -501,6 +628,9 @@ class PUCTSampler(StateSampler):
             self._last_sampled_states = picked
             self._last_sampled_indices = []
             self._last_puct_stats = [(0, 0.0, 0.0, 0.0, 0.0) for _ in picked]
+            self._last_rank_pressure_stats = [
+                ("rank_pressure_residual", -1, -1, -1) for _ in picked
+            ]
             return picked
 
         vals = np.array([float(s.value if s.value is not None else float("-inf")) for s in candidates])
@@ -520,27 +650,35 @@ class PUCTSampler(StateSampler):
             scores.append((score, vals[i], s, n, Q, P[i], bonus))
 
         scores.sort(key=lambda x: (x[0], x[1]), reverse=True)
-
-        if num_states > 1:
-            children_map = self._build_children_map()
-            picked, top_scores, blocked_ids = [], [], set()
-            for entry in scores:
-                s = entry[2]
-                if s.id in blocked_ids:
-                    continue
-                picked.append(s)
-                top_scores.append(entry)
-                blocked_ids.update(self._get_full_lineage(s, children_map))
-                if len(picked) >= num_states:
-                    break
+        rank_stats = rank_pressure_entry_stats(
+            scores,
+            rank_slack=self.rank_pressure_rank_slack,
+            min_candidates=self.rank_pressure_min_candidates,
+        )
+
+        if self.rank_pressure_balance:
+            top_scores = self._select_rank_pressure_entries(
+                scores,
+                rank_stats,
+                num_states,
+            )
+            for entry in top_scores:
+                surface = rank_stats[id(entry)][0]
+                self._rank_pressure_surface_sample_counts[surface] = (
+                    self._rank_pressure_surface_sample_counts.get(surface, 0) + 1
+                )
         else:
-            top_scores = scores[:num_states]
-            picked = [t[2] for t in top_scores]
+            top_scores = self._select_baseline_entries(scores, num_states)
+        picked = [t[2] for t in top_scores]
 
         state_id_to_idx = {s.id: i for i, s in enumerate(self._states)}
         self._last_sampled_states = picked
         self._last_sampled_indices = [state_id_to_idx.get(s.id, -1) for s in picked]
         self._last_puct_stats = [(t[3], t[4], t[5], t[6], t[0]) for t in top_scores]
+        self._last_rank_pressure_stats = [
+            rank_stats.get(id(t), ("rank_pressure_residual", -1, -1, -1))
+            for t in top_scores
+        ]
 
         for s in picked:
             if s.id in initial_ids:
@@ -725,6 +863,10 @@ class PUCTSampler(StateSampler):
             "puct/T": self._T,
             "puct/scale_last": float(self._last_scale),
         }
+        for surface in RANK_PRESSURE_SURFACE_ORDER:
+            stats[f"puct/rank_pressure_sample_count/{surface}"] = int(
+                self._rank_pressure_surface_sample_counts.get(surface, 0)
+            )
         stats.update(_stats(buffer_values, "puct/buffer_value"))
         stats.update(_stats(buffer_timesteps, "puct/buffer_timestep"))
         stats.update(_stats(buffer_constr_lens, "puct/buffer_construction_len"))
@@ -735,17 +877,27 @@ class PUCTSampler(StateSampler):
 
     def get_sample_table(self) -> tuple[list[str], list[tuple]]:
         columns = ["buffer_idx", "timestep", "value", "terminal_value", "parent_value", "construction_len", "observation_len", "n", "Q", "P", "bonus", "score"]
+        include_rank_pressure = self.rank_pressure_balance or any(
+            self._rank_pressure_surface_sample_counts.get(surface, 0) > 0
+            for surface in RANK_PRESSURE_SURFACE_ORDER
+        )
+        if include_rank_pressure:
+            columns.extend(["rank_pressure_surface", "puct_rank", "q_rank", "bonus_rank"])
         rows = []
         if not self._last_sampled_states:
             return columns, rows
         indices = self._last_sampled_indices if len(self._last_sampled_indices) == len(self._last_sampled_states) else [-1] * len(self._last_sampled_states)
         stats = self._last_puct_stats if len(self._last_puct_stats) == len(self._last_sampled_states) else [(0, 0.0, 0.0, 0.0, 0.0)] * len(self._last_sampled_states)
-        for idx, state, (n, Q, P, bonus, score) in zip(indices, self._last_sampled_states, stats):
+        rank_stats = self._last_rank_pressure_stats if len(self._last_rank_pressure_stats) == len(self._last_sampled_states) else [("rank_pressure_residual", -1, -1, -1)] * len(self._last_sampled_states)
+        for idx, state, (n, Q, P, bonus, score), (surface, puct_rank, q_rank, bonus_rank) in zip(indices, self._last_sampled_states, stats, rank_stats):
             parent_val = state.parent_values[0] if state.parent_values else None
             constr = getattr(state, 'construction', None)
             constr_len = len(constr) if constr is not None else 0
             obs_len = len(state.observation) if state.observation else 0
-            rows.append((idx, state.timestep, state.value, 0, parent_val, constr_len, obs_len, n, Q, P, bonus, score))
+            row = (idx, state.timestep, state.value, 0, parent_val, constr_len, obs_len, n, Q, P, bonus, score)
+            if include_rank_pressure:
+                row = row + (surface, puct_rank, q_rank, bonus_rank)
+            rows.append(row)
         return columns, rows
 
 
@@ -756,6 +908,9 @@ def create_sampler(
     batch_size: int = 1,
     resume_step: int | None = None,
     topk_children: int = 2,
+    rank_pressure_balance: bool = False,
+    rank_pressure_rank_slack: int = 1,
+    rank_pressure_min_candidates: int = 4,
 ) -> StateSampler:
     """Factory function to create samplers. Pass the env type (from config.env_type)."""
     if not log_path:
@@ -768,6 +923,9 @@ def create_sampler(
         batch_size=batch_size,
         resume_step=resume_step,
         topk_children=topk_children,
+        rank_pressure_balance=rank_pressure_balance,
+        rank_pressure_rank_slack=rank_pressure_rank_slack,
+        rank_pressure_min_candidates=rank_pressure_min_candidates,
     )
 
 
@@ -778,6 +936,9 @@ def get_or_create_sampler_with_default(
     batch_size: int = 1,
     resume_step: int | None = None,
     topk_children: int = 2,
+    rank_pressure_balance: bool = False,
+    rank_pressure_rank_slack: int = 1,
+    rank_pressure_min_candidates: int = 4,
 ) -> StateSampler:
     """Get sampler. Initial experience is created via env_type.create_initial_state."""
     return create_sampler(
@@ -787,4 +948,7 @@ def get_or_create_sampler_with_default(
         batch_size=batch_size,
         resume_step=resume_step,
         topk_children=topk_children,
+        rank_pressure_balance=rank_pressure_balance,
+        rank_pressure_rank_slack=rank_pressure_rank_slack,
+        rank_pressure_min_candidates=rank_pressure_min_candidates,
     )
diff --git a/ttt_discover/discovery.py b/ttt_discover/discovery.py
index 6cdb434..d5d6e47 100644
--- a/ttt_discover/discovery.py
+++ b/ttt_discover/discovery.py
@@ -63,6 +63,9 @@ class DiscoverConfig:
     codex_initial_program_paths: tuple[str, ...] = ()
     codex_initial_pool_paths: tuple[str, ...] = ()
     codex_autonomous: bool = False
+    codex_puct_rank_pressure_balance: bool = False
+    codex_puct_rank_pressure_rank_slack: int = 1
+    codex_puct_rank_pressure_min_candidates: int = 4
 
 
 def init_ray(num_cpus_per_task: int, env_type: str):
@@ -146,6 +149,9 @@ async def discover_impl(config: DiscoverConfig):
             initial_program_paths=config.codex_initial_program_paths,
             initial_pool_paths=config.codex_initial_pool_paths,
             autonomous=config.codex_autonomous,
+            puct_rank_pressure_balance=config.codex_puct_rank_pressure_balance,
+            puct_rank_pressure_rank_slack=config.codex_puct_rank_pressure_rank_slack,
+            puct_rank_pressure_min_candidates=config.codex_puct_rank_pressure_min_candidates,
             wandb_project=config.wandb_project,
             wandb_name=config.experiment_name,
             log_path=log_path,
diff --git a/ttt_discover/rl/codex_no_finetune.py b/ttt_discover/rl/codex_no_finetune.py
index 18a0e9e..94f12ef 100644
--- a/ttt_discover/rl/codex_no_finetune.py
+++ b/ttt_discover/rl/codex_no_finetune.py
@@ -137,6 +137,9 @@ class CodexNoFinetuneConfig:
     initial_program_paths: tuple[str, ...] = ()
     initial_pool_paths: tuple[str, ...] = ()
     topk_children: int = 16
+    puct_rank_pressure_balance: bool = False
+    puct_rank_pressure_rank_slack: int = 1
+    puct_rank_pressure_min_candidates: int = 4
 
     # False is TTT Discover: non-auto, many short samples. True is AutoEvolve:
     # one Codex deep-dive workspace per sample with file edits and shell access.
@@ -960,6 +963,9 @@ def _build_sampler(cfg: CodexNoFinetuneConfig, start_batch: int) -> StateSampler
         batch_size=cfg.groups_per_batch,
         resume_step=start_batch if start_batch > 0 else None,
         topk_children=cfg.topk_children,
+        rank_pressure_balance=cfg.puct_rank_pressure_balance,
+        rank_pressure_rank_slack=cfg.puct_rank_pressure_rank_slack,
+        rank_pressure_min_candidates=cfg.puct_rank_pressure_min_candidates,
     )
 
 
````
</details>

