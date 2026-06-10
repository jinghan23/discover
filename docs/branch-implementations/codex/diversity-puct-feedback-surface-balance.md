# codex/diversity-puct-feedback-surface-balance

## Summary

按 PUCT 访问/备份反馈分类 feedback_untried/improved/flat/declined，持久记录 surface sample counts 并做平衡选择。

## Branch State

- Worktree: `/opt/tiger/discover-puct-feedback-surface-balance`
- HEAD: `2234282`
- Base used for comparison: `2234282`
- Commits ahead of base: `0`
- Commits behind base: `0`
- Group: `diversity_extra`
- Implementation location: `uncommitted worktree diff`

## Implemented Behavior

- The summary above is based on reading the changed Python code, added config fields, and sampler/prompt hooks in this branch.
- For sampler branches, the common pattern is to keep baseline PUCT ordering inside a chosen surface/bucket, while changing which surface/bucket gets the next parent pick.
- Tests detected: `11` test functions across untracked test files.

## Added Markers

### Config fields

- `codex_puct_feedback_surface_balance`
- `codex_puct_feedback_surface_eps`
- `counts`
- `puct_feedback_surface_balance`
- `puct_feedback_surface_eps`

### Constants

- `FEEDBACK_SURFACE_ORDER`

### Classes

- None

### Functions

- `_sanitize_feedback_surface_sample_counts`
- `feedback_surface_for_state`
- `_rank_puct_candidates`
- `_sample_new_initial_states`
- `_finalize_sampled_entries`
- `_feedback_surface_for_entry`
- `_pick_baseline_puct`
- `_sample_states_feedback_surface_balanced`
- `is_unblocked`
- `first_unblocked`
- `record_pick`
- `sample_states`

## Diff Summary

- Worktree tracked shortstat: `5 files changed, 241 insertions(+), 25 deletions(-)`
- Untracked files: `3`

### Worktree Status

````text
 M ttt_discover/codex_utils/__init__.py
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
M	ttt_discover/codex_utils/__init__.py
M	ttt_discover/codex_utils/discovery.py
M	ttt_discover/codex_utils/sampler.py
M	ttt_discover/discovery.py
M	ttt_discover/rl/codex_no_finetune.py
````
### Untracked Files

- `repro/gpu_mode/run_0608_puct_feedback_surface_balance.sh (954 bytes)`
- `repro/run_discovery.py (14882 bytes)`
- `tests/test_puct_feedback_surface_balance.py (13979 bytes)`

### Detected Test Functions

- `tests/test_puct_feedback_surface_balance.py::test_disabled_mode_matches_baseline_and_does_not_increment_counts`
- `tests/test_puct_feedback_surface_balance.py::test_enabled_mode_preserves_baseline_for_nonpositive_requests`
- `tests/test_puct_feedback_surface_balance.py::test_classification_covers_all_feedback_surfaces`
- `tests/test_puct_feedback_surface_balance.py::test_balanced_mode_picks_least_sampled_surface`
- `tests/test_puct_feedback_surface_balance.py::test_within_surface_candidates_remain_puct_ordered`
- `tests/test_puct_feedback_surface_balance.py::test_full_lineage_blocking_prevents_ancestor_descendant_pairs`
- `tests/test_puct_feedback_surface_balance.py::test_fallback_fills_by_global_puct_when_target_surface_blocked`
- `tests/test_puct_feedback_surface_balance.py::test_counts_persist_and_old_or_bad_checkpoints_are_sanitized`
- `tests/test_puct_feedback_surface_balance.py::test_runner_wiring_sets_sampler_flags_and_epsilon`
- `tests/test_puct_feedback_surface_balance.py::test_cli_dry_run_prints_default_and_custom_feedback_values`
- `tests/test_puct_feedback_surface_balance.py::test_stats_and_table_include_feedback_surfaces`

## Untracked File Snapshots

These files are not part of `git diff` yet, but they are part of the worktree implementation state recorded here.

### `repro/gpu_mode/run_0608_puct_feedback_surface_balance.sh`

````bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-gpu-mode-trimul-puct-feedback-surface-balance-0608}"
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
  --codex-puct-feedback-surface-balance
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
import asyncio
import importlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass(frozen=True)
class TaskSpec:
    env_module: str
    env_name: str
    problem_type: str
    experiment_name: str
    wandb_project: str | None
    eval_timeout: int
    num_epochs: int
    group_size: int
    groups_per_batch: int
    initial_pools: tuple[Path, ...] = ()
    uses_gpu: bool = False

    @property
    def env_path(self) -> str:
        return f"{self.env_module}.{self.env_name}"

    def load_env(self) -> type:
        module = importlib.import_module(self.env_module)
        return getattr(module, self.env_name)


def _existing_paths(paths: tuple[Path, ...]) -> tuple[Path, ...]:
    return tuple(path for path in paths if path.exists())


def _task_spec(args: argparse.Namespace) -> TaskSpec:
    task = args.task
    if task == "trimul":
        task = "gpu_mode:trimul"
    elif task == "mla_decode_nvidia":
        task = "gpu_mode:mla_decode_nvidia"

    if task == "cap_set":
        dimension = args.dimension if args.dimension is not None else 8
        return TaskSpec(
            env_module="examples.cap_set_priority.env",
            env_name="CapSetPriorityEnv",
            problem_type=str(dimension),
            experiment_name=f"cap-set-priority-{dimension}",
            wandb_project="cap-set-priority",
            eval_timeout=45,
            num_epochs=10,
            group_size=1,
            groups_per_batch=1,
            initial_pools=_existing_paths(
                (REPO_ROOT / "repro/cap_set/initial_pool_400_to_512.json",)
            ),
        )

    if task == "erdos":
        return TaskSpec(
            env_module="examples.erdos_min_overlap.env",
            env_name="ErdosMinOverlapEnv",
            problem_type="",
            experiment_name="erdos-min-overlap",
            wandb_project="erdos-min-overlap",
            eval_timeout=1100,
            num_epochs=50,
            group_size=64,
            groups_per_batch=8,
            initial_pools=_existing_paths(
                (
                    REPO_ROOT
                    / "repro/erdos/initial_pool_reference_plus_codex_20260603.json",
                )
            ),
        )

    if task == "kakeya":
        dimension = args.dimension if args.dimension is not None else 3
        primes = args.primes or "5,7,13"
        return TaskSpec(
            env_module="examples.kakeya.env",
            env_name="KakeyaEnv",
            problem_type=f"{dimension}:{primes}",
            experiment_name="kakeya",
            wandb_project="kakeya",
            eval_timeout=120,
            num_epochs=10,
            group_size=1,
            groups_per_batch=1,
        )

    if task in {"ahc039", "ahc058"}:
        return TaskSpec(
            env_module="examples.ahc.env",
            env_name="AhcEnv",
            problem_type=task,
            experiment_name=f"{task}-codex",
            wandb_project=None,
            eval_timeout=530,
            num_epochs=10,
            group_size=1,
            groups_per_batch=1,
        )

    if task == "gpu_mode:trimul":
        return TaskSpec(
            env_module="examples.gpu_mode.env",
            env_name="GpuModeEnv",
            problem_type="trimul",
            experiment_name="gpu-mode-trimul",
            wandb_project=None,
            eval_timeout=1200,
            num_epochs=50,
            group_size=1,
            groups_per_batch=1,
            uses_gpu=True,
        )

    if task == "gpu_mode:mla_decode_nvidia":
        return TaskSpec(
            env_module="examples.gpu_mode.env",
            env_name="GpuModeEnv",
            problem_type="mla_decode_nvidia",
            experiment_name="gpu-mode-mla-decode-nvidia",
            wandb_project=None,
            eval_timeout=1200,
            num_epochs=50,
            group_size=1,
            groups_per_batch=1,
            uses_gpu=True,
        )

    raise ValueError(f"Unknown task: {args.task}")


def _none_if_empty(value: str | None) -> str | None:
    if value is None or value == "":
        return None
    return value


def _append_paths(
    defaults: tuple[Path, ...],
    extras: list[str] | None,
    *,
    use_defaults: bool,
) -> tuple[str, ...]:
    paths: list[str] = []
    if use_defaults:
        paths.extend(str(path) for path in defaults)
    paths.extend(extras or [])
    return tuple(paths)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch a TTT-Discover task through a shared repro entrypoint."
    )
    parser.add_argument("command", nargs="?", choices=("dry-run",))
    parser.add_argument(
        "--task",
        default="trimul",
        choices=(
            "cap_set",
            "erdos",
            "kakeya",
            "ahc039",
            "ahc058",
            "gpu_mode:trimul",
            "gpu_mode:mla_decode_nvidia",
            "trimul",
            "mla_decode_nvidia",
        ),
    )
    parser.add_argument("--problem-type", default=None)
    parser.add_argument("--dimension", type=int, default=None)
    parser.add_argument("--primes", default=None)
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--log-root", default="tinker_log")
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--no-default-initial-pool", action="store_true")
    parser.add_argument("--codex-initial-pool", action="append", default=None)
    parser.add_argument("--codex-initial-program", action="append", default=None)

    parser.add_argument(
        "--runner",
        choices=("tinker_rl", "codex_no_finetune"),
        default="codex_no_finetune",
    )
    parser.add_argument("--model-name", default="openai/gpt-oss-120b")
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--group-size", type=int, default=None)
    parser.add_argument("--groups-per-batch", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=4e-5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--kl-penalty-coef", type=float, default=0.1)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--save-every", type=int, default=2)
    parser.add_argument("--phase1-max-tokens", type=int, default=26000)
    parser.add_argument("--num-cpus-per-task", type=int, default=1)
    parser.add_argument("--eval-timeout", type=int, default=None)
    parser.add_argument(
        "--remove-constant-reward-groups",
        action=argparse.BooleanOptionalAction,
        default=None,
    )

    parser.add_argument("--codex-backend", choices=("cli", "responses"), default="cli")
    parser.add_argument("--codex-model-name", default=None)
    parser.add_argument("--codex-max-output-tokens", type=int, default=None)
    parser.add_argument("--codex-temperature", type=float, default=None)
    parser.add_argument("--codex-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--codex-base-url", default=None)
    parser.add_argument("--codex-cli-command", default="codex")
    parser.add_argument(
        "--codex-cli-sandbox",
        choices=("read-only", "workspace-write", "danger-full-access"),
        default=None,
    )
    parser.add_argument("--codex-cli-timeout", type=float, default=None)
    parser.add_argument("--codex-max-concurrent-requests", type=int, default=4)
    parser.add_argument("--codex-autonomous", action="store_true")
    parser.add_argument(
        "--codex-puct-feedback-surface-balance",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--codex-puct-feedback-surface-eps",
        type=float,
        default=1e-9,
    )

    parser.add_argument("--gpu", default=None)
    parser.add_argument("--cuda-device-order", default="PCI_BUS_ID")
    parser.add_argument("--torch-cuda-arch-list", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _configure_gpu_env(args: argparse.Namespace, spec: TaskSpec) -> None:
    if not spec.uses_gpu:
        return
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("CUDA_DEVICE_ORDER", args.cuda_device_order)
    if args.torch_cuda_arch_list:
        os.environ["TORCH_CUDA_ARCH_LIST"] = args.torch_cuda_arch_list
    os.environ.setdefault("PYTHONUNBUFFERED", "1")


def main() -> None:
    args = parse_args()
    spec = _task_spec(args)
    _configure_gpu_env(args, spec)

    experiment_name = args.experiment_name or spec.experiment_name
    problem_type = args.problem_type if args.problem_type is not None else spec.problem_type
    wandb_project = (
        _none_if_empty(args.wandb_project)
        if args.wandb_project is not None
        else spec.wandb_project
    )
    num_epochs = args.num_epochs if args.num_epochs is not None else spec.num_epochs
    group_size = args.group_size if args.group_size is not None else spec.group_size
    groups_per_batch = (
        args.groups_per_batch
        if args.groups_per_batch is not None
        else spec.groups_per_batch
    )
    eval_timeout = args.eval_timeout if args.eval_timeout is not None else spec.eval_timeout
    discovery_cpus = args.num_cpus_per_task
    if args.runner == "codex_no_finetune":
        discovery_cpus = 0

    codex_cli_sandbox = args.codex_cli_sandbox
    if codex_cli_sandbox is None:
        codex_cli_sandbox = "workspace-write" if args.codex_autonomous else "read-only"

    remove_constant_reward_groups = args.remove_constant_reward_groups
    if remove_constant_reward_groups is None:
        remove_constant_reward_groups = group_size > 1

    initial_pools = _append_paths(
        spec.initial_pools,
        args.codex_initial_pool,
        use_defaults=not args.no_default_initial_pool,
    )
    log_path = str(Path(args.log_root) / experiment_name)

    if args.dry_run or args.command == "dry-run":
        print(f"task={args.task}")
        print(f"env_type={spec.env_path}")
        print(f"problem_type={problem_type!r}")
        print(f"experiment_name={experiment_name!r}")
        print(f"log_root={args.log_root!r}")
        print(f"log_path={log_path!r}")
        print(f"runner={args.runner!r}")
        print(f"num_epochs={num_epochs}")
        print(f"group_size={group_size}")
        print(f"groups_per_batch={groups_per_batch}")
        print(f"eval_timeout={eval_timeout}")
        print(f"wandb_project={wandb_project!r}")
        print(f"codex_autonomous={args.codex_autonomous}")
        print(f"codex_cli_sandbox={codex_cli_sandbox!r}")
        print(f"codex_initial_pool_paths={initial_pools!r}")
        print(
            "codex_puct_feedback_surface_balance="
            f"{args.codex_puct_feedback_surface_balance}"
        )
        print(
            "codex_puct_feedback_surface_eps="
            f"{args.codex_puct_feedback_surface_eps}"
        )
        if spec.uses_gpu:
            print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}")
            print(f"CUDA_DEVICE_ORDER={os.environ.get('CUDA_DEVICE_ORDER')!r}")
            print(f"TORCH_CUDA_ARCH_LIST={os.environ.get('TORCH_CUDA_ARCH_LIST')!r}")
        return

    env_type = spec.load_env()

    if args.runner == "codex_no_finetune":
        from ttt_discover.rl.codex_no_finetune import (
            CodexNoFinetuneConfig,
            main as codex_no_finetune_main,
        )

        cfg = CodexNoFinetuneConfig(
            env_type=env_type,
            problem_type=problem_type,
            backend=args.codex_backend,
            model_name=args.codex_model_name,
            groups_per_batch=groups_per_batch,
            group_size=group_size,
            num_cpus_per_task=max(1, int(args.num_cpus_per_task)),
            eval_timeout=eval_timeout,
            num_epochs=num_epochs,
            max_output_tokens=args.codex_max_output_tokens,
            temperature=args.codex_temperature,
            api_key_env=args.codex_api_key_env,
            base_url=args.codex_base_url,
            cli_command=args.codex_cli_command,
            cli_sandbox=codex_cli_sandbox,
            cli_timeout=args.codex_cli_timeout,
            max_concurrent_requests=args.codex_max_concurrent_requests,
            initial_program_paths=tuple(args.codex_initial_program or ()),
            initial_pool_paths=initial_pools,
            autonomous=args.codex_autonomous,
            puct_feedback_surface_balance=(
                args.codex_puct_feedback_surface_balance
            ),
            puct_feedback_surface_eps=args.codex_puct_feedback_surface_eps,
            wandb_project=wandb_project,
            wandb_name=experiment_name,
            log_path=log_path,
            remove_constant_reward_groups=remove_constant_reward_groups,
        )
        asyncio.run(codex_no_finetune_main(cfg))
        return

    from ttt_discover import DiscoverConfig, discover

    config = DiscoverConfig(
        env_type=env_type,
        problem_type=problem_type,
        model_name=args.model_name,
        runner=args.runner,
        lora_rank=args.lora_rank,
        group_size=group_size,
        groups_per_batch=groups_per_batch,
        learning_rate=args.learning_rate,
        num_epochs=num_epochs,
        temperature=args.temperature,
        kl_penalty_coef=args.kl_penalty_coef,
        phase1_max_tokens=args.phase1_max_tokens,
        save_every=args.save_every,
        num_cpus_per_task=discovery_cpus,
        eval_timeout=eval_timeout,
        experiment_name=experiment_name,
        wandb_project=wandb_project,
        codex_model_name=args.codex_model_name,
        codex_backend=args.codex_backend,
        codex_max_output_tokens=args.codex_max_output_tokens,
        codex_temperature=args.codex_temperature,
        codex_api_key_env=args.codex_api_key_env,
        codex_base_url=args.codex_base_url,
        codex_cli_command=args.codex_cli_command,
        codex_cli_sandbox=codex_cli_sandbox,
        codex_cli_timeout=args.codex_cli_timeout,
        codex_max_concurrent_requests=args.codex_max_concurrent_requests,
        codex_initial_program_paths=tuple(args.codex_initial_program or ()),
        codex_initial_pool_paths=initial_pools,
        codex_autonomous=args.codex_autonomous,
        codex_puct_feedback_surface_balance=(
            args.codex_puct_feedback_surface_balance
        ),
        codex_puct_feedback_surface_eps=args.codex_puct_feedback_surface_eps,
        remove_constant_reward_groups=remove_constant_reward_groups,
    )
    discover(config)


if __name__ == "__main__":
    main()
````

### `tests/test_puct_feedback_surface_balance.py`

````python
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from ttt_discover.codex_utils.runtime import State
from ttt_discover.codex_utils.sampler import (
    FEEDBACK_SURFACE_ORDER,
    PUCTSampler,
    _sampler_file_for_step,
    create_sampler,
    feedback_surface_for_state,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class DummyEnv:
    env_name = "DummyEnv"
    state_type = State
    _counter = 0

    @classmethod
    def create_initial_state(cls, problem_type: str) -> State:
        cls._counter += 1
        return State(
            timestep=0,
            construction=[problem_type, cls._counter],
            code="initial",
            value=0.0,
            id=f"initial-{cls._counter}",
        )


def make_state(
    sid: str,
    value: float,
    *,
    parents: list[dict] | None = None,
) -> State:
    return State(
        timestep=0,
        construction=[sid],
        code=f"code-{sid}",
        value=value,
        parents=parents or [],
        id=sid,
    )


def make_sampler(tmpdir: str, **kwargs) -> PUCTSampler:
    return PUCTSampler(
        file_path=str(Path(tmpdir) / "puct_sampler.json"),
        env_type=DummyEnv,
        batch_size=0,
        puct_c=0.0,
        topk_children=0,
        **kwargs,
    )


def zero_counts() -> dict[str, int]:
    return {surface: 0 for surface in FEEDBACK_SURFACE_ORDER}


class PuctFeedbackSurfaceBalanceTest(unittest.TestCase):
    def test_disabled_mode_matches_baseline_and_does_not_increment_counts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sampler = make_sampler(tmpdir, feedback_surface_balance=False)
            sampler._states = [
                make_state("low", 1.0),
                make_state("high", 3.0),
                make_state("mid", 2.0),
            ]

            picked = sampler.sample_states(2)

            self.assertEqual([state.id for state in picked], ["high", "mid"])
            self.assertEqual(sampler.feedback_surface_sample_counts, zero_counts())

    def test_enabled_mode_preserves_baseline_for_nonpositive_requests(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sampler = make_sampler(tmpdir, feedback_surface_balance=True)
            sampler._states = [make_state("only", 1.0)]

            picked = sampler.sample_states(0)

            self.assertEqual(picked, [])
            self.assertEqual(sampler.feedback_surface_sample_counts, zero_counts())

    def test_classification_covers_all_feedback_surfaces(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sampler = make_sampler(tmpdir, feedback_surface_eps=1e-3)
            untried = make_state("untried", 10.0)
            missing = make_state("missing", 10.0)
            improved = make_state("improved", 10.0)
            flat = make_state("flat", 10.0)
            declined = make_state("declined", 10.0)
            sampler._n = {
                missing.id: 2,
                improved.id: 2,
                flat.id: 2,
                declined.id: 2,
            }
            sampler._m = {
                improved.id: 10.002,
                flat.id: 10.0005,
                declined.id: 9.998,
            }

            self.assertEqual(
                feedback_surface_for_state(untried, 0, 99.0, 1e-3),
                "feedback_untried",
            )
            self.assertEqual(
                sampler.feedback_surface_for_state(missing),
                "feedback_flat",
            )
            self.assertEqual(
                sampler.feedback_surface_for_state(improved),
                "feedback_improved",
            )
            self.assertEqual(
                sampler.feedback_surface_for_state(flat),
                "feedback_flat",
            )
            self.assertEqual(
                sampler.feedback_surface_for_state(declined),
                "feedback_declined",
            )

    def test_balanced_mode_picks_least_sampled_surface(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sampler = make_sampler(tmpdir, feedback_surface_balance=True)
            untried = make_state("untried", 100.0)
            improved = make_state("improved", 1.0)
            sampler._states = [untried, improved]
            sampler._n = {improved.id: 1}
            sampler._m = {improved.id: 2.0}
            sampler.feedback_surface_sample_counts = {
                "feedback_untried": 5,
                "feedback_improved": 0,
                "feedback_flat": 5,
                "feedback_declined": 5,
            }

            picked = sampler.sample_states(1)

            self.assertEqual([state.id for state in picked], ["improved"])
            self.assertEqual(
                sampler.feedback_surface_sample_counts["feedback_improved"],
                1,
            )

    def test_within_surface_candidates_remain_puct_ordered(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sampler = make_sampler(tmpdir, feedback_surface_balance=True)
            low = make_state("low", 1.0)
            high = make_state("high", 1.0)
            sampler._states = [low, high]
            sampler._n = {low.id: 1, high.id: 1}
            sampler._m = {low.id: 2.0, high.id: 4.0}
            sampler.feedback_surface_sample_counts = {
                "feedback_untried": 5,
                "feedback_improved": 0,
                "feedback_flat": 5,
                "feedback_declined": 5,
            }

            picked = sampler.sample_states(1)

            self.assertEqual([state.id for state in picked], ["high"])

    def test_full_lineage_blocking_prevents_ancestor_descendant_pairs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sampler = make_sampler(tmpdir, feedback_surface_balance=True)
            ancestor = make_state("ancestor", 10.0)
            child = make_state(
                "child",
                9.0,
                parents=[{"id": ancestor.id, "timestep": ancestor.timestep}],
            )
            unrelated = make_state("unrelated", 1.0)
            sampler._states = [ancestor, child, unrelated]
            sampler.feedback_surface_sample_counts = {
                "feedback_untried": 0,
                "feedback_improved": 10,
                "feedback_flat": 10,
                "feedback_declined": 10,
            }

            picked = sampler.sample_states(2)
            picked_ids = {state.id for state in picked}

            self.assertEqual(len(picked), 2)
            self.assertFalse({ancestor.id, child.id}.issubset(picked_ids))

    def test_fallback_fills_by_global_puct_when_target_surface_blocked(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sampler = make_sampler(tmpdir, feedback_surface_balance=True)
            ancestor = make_state("ancestor", 100.0)
            blocked_child = make_state(
                "blocked_child",
                90.0,
                parents=[{"id": ancestor.id, "timestep": ancestor.timestep}],
            )
            declined = make_state("declined", 80.0)
            flat = make_state("flat", 10.0)
            sampler._states = [ancestor, blocked_child, declined, flat]
            sampler._n = {declined.id: 1, flat.id: 1}
            sampler._m = {declined.id: 70.0, flat.id: 10.0}
            sampler.feedback_surface_sample_counts = {
                "feedback_untried": 0,
                "feedback_improved": 5,
                "feedback_flat": 5,
                "feedback_declined": 5,
            }

            picked = sampler.sample_states(2)

            self.assertEqual([state.id for state in picked], ["ancestor", "declined"])

    def test_counts_persist_and_old_or_bad_checkpoints_are_sanitized(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sampler = make_sampler(tmpdir, feedback_surface_balance=True)
            improved = make_state("improved", 1.0)
            sampler._states = [improved]
            sampler._n = {improved.id: 1}
            sampler._m = {improved.id: 2.0}
            sampler.sample_states(1)
            sampler.flush(step=3)

            loaded = PUCTSampler(
                file_path=str(Path(tmpdir) / "puct_sampler.json"),
                env_type=DummyEnv,
                batch_size=0,
                resume_step=3,
                feedback_surface_balance=True,
            )
            self.assertEqual(
                loaded.feedback_surface_sample_counts,
                sampler.feedback_surface_sample_counts,
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            old_base = str(Path(tmpdir) / "old_sampler.json")
            old_path = _sampler_file_for_step(old_base, 0)
            Path(old_path).write_text(
                json.dumps(
                    {
                        "step": 0,
                        "states": [make_state("old", 1.0).to_dict()],
                        "initial_states": [],
                        "puct_n": {},
                        "puct_m": {},
                        "puct_T": 0,
                    }
                ),
                encoding="utf-8",
            )
            old_loaded = PUCTSampler(
                file_path=old_base,
                env_type=DummyEnv,
                batch_size=0,
                resume_step=0,
            )
            self.assertEqual(old_loaded.feedback_surface_sample_counts, zero_counts())

        with tempfile.TemporaryDirectory() as tmpdir:
            bad_base = str(Path(tmpdir) / "bad_sampler.json")
            bad_path = _sampler_file_for_step(bad_base, 0)
            Path(bad_path).write_text(
                json.dumps(
                    {
                        "step": 0,
                        "states": [make_state("bad", 1.0).to_dict()],
                        "initial_states": [],
                        "puct_n": {},
                        "puct_m": {},
                        "puct_T": 0,
                        "feedback_surface_sample_counts": {
                            "feedback_untried": -3,
                            "feedback_improved": "4",
                            "feedback_flat": 2.5,
                            "feedback_declined": [1],
                            "extra": 99,
                        },
                    }
                ),
                encoding="utf-8",
            )
            bad_loaded = PUCTSampler(
                file_path=bad_base,
                env_type=DummyEnv,
                batch_size=0,
                resume_step=0,
            )
            self.assertEqual(bad_loaded.feedback_surface_sample_counts, zero_counts())

    def test_runner_wiring_sets_sampler_flags_and_epsilon(self):
        from ttt_discover.rl.codex_no_finetune import (
            CodexNoFinetuneConfig,
            _build_sampler,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = CodexNoFinetuneConfig(
                env_type=DummyEnv,
                groups_per_batch=0,
                log_path=tmpdir,
                puct_feedback_surface_balance=True,
                puct_feedback_surface_eps=1e-6,
            )
            sampler = _build_sampler(cfg, start_batch=0)

            self.assertTrue(sampler.feedback_surface_balance)
            self.assertEqual(sampler.feedback_surface_eps, 1e-6)

        with tempfile.TemporaryDirectory() as tmpdir:
            made = create_sampler(
                log_path=tmpdir,
                env_type=DummyEnv,
                batch_size=0,
                puct_feedback_surface_balance=True,
                puct_feedback_surface_eps=2e-6,
            )
            self.assertTrue(made.feedback_surface_balance)
            self.assertEqual(made.feedback_surface_eps, 2e-6)

    def test_cli_dry_run_prints_default_and_custom_feedback_values(self):
        default = subprocess.run(
            [
                sys.executable,
                "repro/run_discovery.py",
                "dry-run",
                "--task",
                "trimul",
            ],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
        self.assertIn("codex_puct_feedback_surface_balance=False", default.stdout)
        self.assertIn("codex_puct_feedback_surface_eps=1e-09", default.stdout)

        custom = subprocess.run(
            [
                sys.executable,
                "repro/run_discovery.py",
                "dry-run",
                "--task",
                "trimul",
                "--codex-puct-feedback-surface-balance",
                "--codex-puct-feedback-surface-eps",
                "1e-6",
            ],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
        self.assertIn("codex_puct_feedback_surface_balance=True", custom.stdout)
        self.assertIn("codex_puct_feedback_surface_eps=1e-06", custom.stdout)

    def test_stats_and_table_include_feedback_surfaces(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sampler = make_sampler(tmpdir, feedback_surface_balance=True)
            improved = make_state("improved", 1.0)
            sampler._states = [improved]
            sampler._n = {improved.id: 1}
            sampler._m = {improved.id: 2.0}

            sampler.sample_states(1)
            stats = sampler.get_sample_stats()
            columns, rows = sampler.get_sample_table()

            for surface in FEEDBACK_SURFACE_ORDER:
                self.assertIn(
                    f"puct/feedback_surface_sample_count/{surface}",
                    stats,
                )
            self.assertIn("feedback_surface", columns)
            surface_idx = columns.index("feedback_surface")
            self.assertEqual(rows[0][surface_idx], "feedback_improved")


if __name__ == "__main__":
    unittest.main()
````

## Diff Stat

### Worktree Tracked Diff Stat

````text
 ttt_discover/codex_utils/__init__.py  |   2 +
 ttt_discover/codex_utils/discovery.py |   4 +
 ttt_discover/codex_utils/sampler.py   | 252 ++++++++++++++++++++++++++++++----
 ttt_discover/discovery.py             |   4 +
 ttt_discover/rl/codex_no_finetune.py  |   4 +
 5 files changed, 241 insertions(+), 25 deletions(-)
````
## Raw Diff

<details>
<summary>Tracked worktree patch</summary>

````diff
diff --git a/ttt_discover/codex_utils/__init__.py b/ttt_discover/codex_utils/__init__.py
index 1a51326..0430674 100644
--- a/ttt_discover/codex_utils/__init__.py
+++ b/ttt_discover/codex_utils/__init__.py
@@ -10,6 +10,7 @@ from ttt_discover.codex_utils.sampler import (
     PUCTSampler,
     StateSampler,
     create_sampler,
+    feedback_surface_for_state,
     get_or_create_sampler_with_default,
 )
 
@@ -25,6 +26,7 @@ __all__ = [
     "VerifyResult",
     "create_sampler",
     "discover",
+    "feedback_surface_for_state",
     "get_or_create_sampler_with_default",
     "state_from_dict",
     "to_json_serializable",
diff --git a/ttt_discover/codex_utils/discovery.py b/ttt_discover/codex_utils/discovery.py
index b21a80f..4b04b3c 100644
--- a/ttt_discover/codex_utils/discovery.py
+++ b/ttt_discover/codex_utils/discovery.py
@@ -50,6 +50,8 @@ class DiscoverConfig:
     codex_initial_program_paths: tuple[str, ...] = ()
     codex_initial_pool_paths: tuple[str, ...] = ()
     codex_autonomous: bool = False
+    codex_puct_feedback_surface_balance: bool = False
+    codex_puct_feedback_surface_eps: float = 1e-9
 
 
 def _run_codex_no_finetune(config: DiscoverConfig) -> None:
@@ -85,6 +87,8 @@ def _run_codex_no_finetune(config: DiscoverConfig) -> None:
         initial_program_paths=config.codex_initial_program_paths,
         initial_pool_paths=config.codex_initial_pool_paths,
         autonomous=config.codex_autonomous,
+        puct_feedback_surface_balance=config.codex_puct_feedback_surface_balance,
+        puct_feedback_surface_eps=config.codex_puct_feedback_surface_eps,
         wandb_project=config.wandb_project,
         wandb_name=experiment_name,
         log_path=log_path,
diff --git a/ttt_discover/codex_utils/sampler.py b/ttt_discover/codex_utils/sampler.py
index 5c5d4d1..8c7494d 100644
--- a/ttt_discover/codex_utils/sampler.py
+++ b/ttt_discover/codex_utils/sampler.py
@@ -16,6 +16,45 @@ from ttt_discover.codex_utils.runtime import State, state_from_dict, to_json_ser
 
 logger = logging.getLogger(__name__)
 
+FEEDBACK_SURFACE_ORDER = (
+    "feedback_untried",
+    "feedback_improved",
+    "feedback_flat",
+    "feedback_declined",
+)
+_PuctEntry = tuple[float, float, State, int, float, float, float]
+
+
+def _sanitize_feedback_surface_sample_counts(raw: Any) -> dict[str, int]:
+    counts: dict[str, int] = {}
+    raw_counts = raw if isinstance(raw, dict) else {}
+    for surface in FEEDBACK_SURFACE_ORDER:
+        value = raw_counts.get(surface, 0)
+        if isinstance(value, bool) or not isinstance(value, int):
+            count = 0
+        else:
+            count = value
+        counts[surface] = max(0, count)
+    return counts
+
+
+def feedback_surface_for_state(state: State, n: int, m: float | None, eps: float) -> str:
+    value = float(state.value if state.value is not None else float("-inf"))
+    if n == 0:
+        return "feedback_untried"
+    if m is None:
+        return "feedback_flat"
+    m_value = float(m)
+    if not np.isfinite(value) or not np.isfinite(m_value):
+        if m_value == value:
+            return "feedback_flat"
+        return "feedback_improved" if m_value > value else "feedback_declined"
+    if m_value > value + eps:
+        return "feedback_improved"
+    if abs(m_value - value) <= eps:
+        return "feedback_flat"
+    return "feedback_declined"
+
 
 @contextmanager
 def _file_lock(lock_path: str, *, poll_s: float = 0.05, stale_s: float = 600.0):
@@ -353,6 +392,8 @@ class PUCTSampler(StateSampler):
         resume_step: int | None = None,
         puct_c: float = 1.0,
         topk_children: int = 2,
+        feedback_surface_balance: bool = False,
+        feedback_surface_eps: float = 1e-9,
     ):
         self.file_path = file_path
         self.env_type = env_type
@@ -361,6 +402,8 @@ class PUCTSampler(StateSampler):
         self.batch_size = batch_size
         self.topk_children = topk_children
         self.puct_c = float(puct_c)
+        self.feedback_surface_balance = bool(feedback_surface_balance)
+        self.feedback_surface_eps = float(feedback_surface_eps)
         
         self._states: list[State] = []
         self._initial_states: list[State] = []
@@ -375,6 +418,9 @@ class PUCTSampler(StateSampler):
         self._T: int = 0
         self._last_scale: float = 1.0
         self._last_puct_stats: list[tuple[int, float, float, float, float]] = []
+        self.feedback_surface_sample_counts: dict[str, int] = (
+            _sanitize_feedback_surface_sample_counts({})
+        )
         
         if resume_step is not None:
             self._load(resume_step)
@@ -399,10 +445,16 @@ class PUCTSampler(StateSampler):
         self._n = store.get("puct_n", {}) or {}
         self._m = store.get("puct_m", {}) or {}
         self._T = int(store.get("puct_T", 0) or 0)
+        self.feedback_surface_sample_counts = _sanitize_feedback_surface_sample_counts(
+            store.get("feedback_surface_sample_counts", {})
+        )
 
     def _save(self, step: int):
         save_path = _sampler_file_for_step(self.file_path, step)
         os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
+        self.feedback_surface_sample_counts = _sanitize_feedback_surface_sample_counts(
+            self.feedback_surface_sample_counts
+        )
         store = {
             "step": step,
             "states": [s.to_dict() for s in self._states],
@@ -410,6 +462,7 @@ class PUCTSampler(StateSampler):
             "puct_n": self._n,
             "puct_m": self._m,
             "puct_T": self._T,
+            "feedback_surface_sample_counts": self.feedback_surface_sample_counts,
         }
         with _file_lock(f"{save_path}.lock"):
             _atomic_write_json(save_path, store)
@@ -489,20 +542,11 @@ class PUCTSampler(StateSampler):
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
+    def _rank_puct_candidates(
+        self,
+        candidates: list[State],
+        initial_ids: set[str],
+    ) -> list[_PuctEntry]:
         vals = np.array([float(s.value if s.value is not None else float("-inf")) for s in candidates])
         non_initial_mask = np.array([s.id not in initial_ids for s in candidates])
         scale = self._compute_scale(vals, non_initial_mask if non_initial_mask.any() else None)
@@ -510,7 +554,7 @@ class PUCTSampler(StateSampler):
         P = self._compute_prior(vals, scale)
         sqrtT = np.sqrt(1.0 + self._T)
 
-        scores = []
+        scores: list[_PuctEntry] = []
         for i, s in enumerate(candidates):
             n = self._n.get(s.id, 0)
             m = self._m.get(s.id, vals[i])
@@ -520,7 +564,68 @@ class PUCTSampler(StateSampler):
             scores.append((score, vals[i], s, n, Q, P[i], bonus))
 
         scores.sort(key=lambda x: (x[0], x[1]), reverse=True)
+        return scores
 
+    def _sample_new_initial_states(self, num_states: int) -> list[State]:
+        picked = [
+            create_initial_state(self.env_type, self.problem_type)
+            for _ in range(num_states)
+        ]
+        self._last_sampled_states = picked
+        self._last_sampled_indices = []
+        self._last_puct_stats = [(0, 0.0, 0.0, 0.0, 0.0) for _ in picked]
+        if self.feedback_surface_balance:
+            counts = _sanitize_feedback_surface_sample_counts(
+                self.feedback_surface_sample_counts
+            )
+            counts["feedback_untried"] += len(picked)
+            self.feedback_surface_sample_counts = counts
+        return picked
+
+    def _finalize_sampled_entries(
+        self,
+        picked: list[State],
+        top_scores: list[_PuctEntry],
+        initial_ids: set[str],
+    ) -> list[State]:
+        state_id_to_idx = {s.id: i for i, s in enumerate(self._states)}
+        self._last_sampled_states = picked
+        self._last_sampled_indices = [state_id_to_idx.get(s.id, -1) for s in picked]
+        self._last_puct_stats = [(t[3], t[4], t[5], t[6], t[0]) for t in top_scores]
+
+        for s in picked:
+            if s.id in initial_ids:
+                self._refresh_random_construction(s)
+
+        return picked
+
+    def feedback_surface_for_state(
+        self,
+        state: State,
+        n: int | None = None,
+        m: float | None = None,
+    ) -> str:
+        resolved_n = self._n.get(state.id, 0) if n is None else n
+        if m is None:
+            m = self._m.get(state.id, state.value)
+        return feedback_surface_for_state(
+            state,
+            resolved_n,
+            m,
+            self.feedback_surface_eps,
+        )
+
+    def _feedback_surface_for_entry(self, entry: _PuctEntry) -> str:
+        state = entry[2]
+        n = entry[3]
+        m = self._m.get(state.id, state.value) if n > 0 else entry[4]
+        return feedback_surface_for_state(state, n, m, self.feedback_surface_eps)
+
+    def _pick_baseline_puct(
+        self,
+        scores: list[_PuctEntry],
+        num_states: int,
+    ) -> tuple[list[State], list[_PuctEntry]]:
         if num_states > 1:
             children_map = self._build_children_map()
             picked, top_scores, blocked_ids = [], [], set()
@@ -536,17 +641,94 @@ class PUCTSampler(StateSampler):
         else:
             top_scores = scores[:num_states]
             picked = [t[2] for t in top_scores]
+        return picked, top_scores
 
-        state_id_to_idx = {s.id: i for i, s in enumerate(self._states)}
-        self._last_sampled_states = picked
-        self._last_sampled_indices = [state_id_to_idx.get(s.id, -1) for s in picked]
-        self._last_puct_stats = [(t[3], t[4], t[5], t[6], t[0]) for t in top_scores]
+    def _sample_states_feedback_surface_balanced(
+        self,
+        num_states: int,
+        initial_ids: set[str],
+        scores: list[_PuctEntry],
+    ) -> tuple[list[State], list[_PuctEntry]]:
+        entries_by_surface: dict[str, list[_PuctEntry]] = {
+            surface: [] for surface in FEEDBACK_SURFACE_ORDER
+        }
+        for entry in scores:
+            entries_by_surface[self._feedback_surface_for_entry(entry)].append(entry)
+
+        use_lineage_blocking = num_states > 1
+        children_map = self._build_children_map() if use_lineage_blocking else {}
+        picked: list[State] = []
+        top_scores: list[_PuctEntry] = []
+        picked_ids: set[str] = set()
+        blocked_ids: set[str] = set()
+        counts = _sanitize_feedback_surface_sample_counts(
+            self.feedback_surface_sample_counts
+        )
+
+        def is_unblocked(entry: _PuctEntry) -> bool:
+            sid = entry[2].id
+            if sid in picked_ids:
+                return False
+            return not use_lineage_blocking or sid not in blocked_ids
+
+        def first_unblocked(surface: str) -> _PuctEntry | None:
+            for entry in entries_by_surface[surface]:
+                if is_unblocked(entry):
+                    return entry
+            return None
+
+        def record_pick(entry: _PuctEntry) -> None:
+            state = entry[2]
+            surface = self._feedback_surface_for_entry(entry)
+            picked.append(state)
+            top_scores.append(entry)
+            picked_ids.add(state.id)
+            counts[surface] += 1
+            if use_lineage_blocking:
+                blocked_ids.update(self._get_full_lineage(state, children_map))
+
+        while len(picked) < num_states:
+            min_count = min(counts[surface] for surface in FEEDBACK_SURFACE_ORDER)
+            selected_entry = None
+            for surface in FEEDBACK_SURFACE_ORDER:
+                if counts[surface] != min_count:
+                    continue
+                selected_entry = first_unblocked(surface)
+                if selected_entry is not None:
+                    break
+            if selected_entry is None:
+                break
+            record_pick(selected_entry)
 
-        for s in picked:
-            if s.id in initial_ids:
-                self._refresh_random_construction(s)
+        if len(picked) < num_states:
+            for entry in scores:
+                if not is_unblocked(entry):
+                    continue
+                record_pick(entry)
+                if len(picked) >= num_states:
+                    break
 
-        return picked
+        self.feedback_surface_sample_counts = counts
+        return picked, top_scores
+
+    def sample_states(self, num_states: int) -> list[State]:
+        initial_ids = {s.id for s in self._initial_states}
+        candidates = list(self._states)
+
+        if not candidates:
+            return self._sample_new_initial_states(num_states)
+
+        scores = self._rank_puct_candidates(candidates, initial_ids)
+        if self.feedback_surface_balance and num_states > 0:
+            picked, top_scores = self._sample_states_feedback_surface_balanced(
+                num_states,
+                initial_ids,
+                scores,
+            )
+        else:
+            picked, top_scores = self._pick_baseline_puct(scores, num_states)
+
+        return self._finalize_sampled_entries(picked, top_scores, initial_ids)
 
     def update_states(self, states: list[State], parent_states: list[State], save: bool = True, step: int | None = None):
         if not states:
@@ -725,6 +907,12 @@ class PUCTSampler(StateSampler):
             "puct/T": self._T,
             "puct/scale_last": float(self._last_scale),
         }
+        counts = _sanitize_feedback_surface_sample_counts(
+            self.feedback_surface_sample_counts
+        )
+        self.feedback_surface_sample_counts = counts
+        for surface in FEEDBACK_SURFACE_ORDER:
+            stats[f"puct/feedback_surface_sample_count/{surface}"] = counts[surface]
         stats.update(_stats(buffer_values, "puct/buffer_value"))
         stats.update(_stats(buffer_timesteps, "puct/buffer_timestep"))
         stats.update(_stats(buffer_constr_lens, "puct/buffer_construction_len"))
@@ -734,7 +922,7 @@ class PUCTSampler(StateSampler):
         return stats
 
     def get_sample_table(self) -> tuple[list[str], list[tuple]]:
-        columns = ["buffer_idx", "timestep", "value", "terminal_value", "parent_value", "construction_len", "observation_len", "n", "Q", "P", "bonus", "score"]
+        columns = ["buffer_idx", "timestep", "value", "terminal_value", "parent_value", "construction_len", "observation_len", "feedback_surface", "n", "Q", "P", "bonus", "score"]
         rows = []
         if not self._last_sampled_states:
             return columns, rows
@@ -745,7 +933,13 @@ class PUCTSampler(StateSampler):
             constr = getattr(state, 'construction', None)
             constr_len = len(constr) if constr is not None else 0
             obs_len = len(state.observation) if state.observation else 0
-            rows.append((idx, state.timestep, state.value, 0, parent_val, constr_len, obs_len, n, Q, P, bonus, score))
+            surface = feedback_surface_for_state(
+                state,
+                n,
+                self._m.get(state.id, state.value) if n > 0 else Q,
+                self.feedback_surface_eps,
+            )
+            rows.append((idx, state.timestep, state.value, 0, parent_val, constr_len, obs_len, surface, n, Q, P, bonus, score))
         return columns, rows
 
 
@@ -756,6 +950,8 @@ def create_sampler(
     batch_size: int = 1,
     resume_step: int | None = None,
     topk_children: int = 2,
+    puct_feedback_surface_balance: bool = False,
+    puct_feedback_surface_eps: float = 1e-9,
 ) -> StateSampler:
     """Factory function to create samplers. Pass the env type (from config.env_type)."""
     if not log_path:
@@ -768,6 +964,8 @@ def create_sampler(
         batch_size=batch_size,
         resume_step=resume_step,
         topk_children=topk_children,
+        feedback_surface_balance=puct_feedback_surface_balance,
+        feedback_surface_eps=puct_feedback_surface_eps,
     )
 
 
@@ -778,6 +976,8 @@ def get_or_create_sampler_with_default(
     batch_size: int = 1,
     resume_step: int | None = None,
     topk_children: int = 2,
+    puct_feedback_surface_balance: bool = False,
+    puct_feedback_surface_eps: float = 1e-9,
 ) -> StateSampler:
     """Get sampler. Initial experience is created via env_type.create_initial_state."""
     return create_sampler(
@@ -787,4 +987,6 @@ def get_or_create_sampler_with_default(
         batch_size=batch_size,
         resume_step=resume_step,
         topk_children=topk_children,
+        puct_feedback_surface_balance=puct_feedback_surface_balance,
+        puct_feedback_surface_eps=puct_feedback_surface_eps,
     )
diff --git a/ttt_discover/discovery.py b/ttt_discover/discovery.py
index 6cdb434..48f7dae 100644
--- a/ttt_discover/discovery.py
+++ b/ttt_discover/discovery.py
@@ -63,6 +63,8 @@ class DiscoverConfig:
     codex_initial_program_paths: tuple[str, ...] = ()
     codex_initial_pool_paths: tuple[str, ...] = ()
     codex_autonomous: bool = False
+    codex_puct_feedback_surface_balance: bool = False
+    codex_puct_feedback_surface_eps: float = 1e-9
 
 
 def init_ray(num_cpus_per_task: int, env_type: str):
@@ -146,6 +148,8 @@ async def discover_impl(config: DiscoverConfig):
             initial_program_paths=config.codex_initial_program_paths,
             initial_pool_paths=config.codex_initial_pool_paths,
             autonomous=config.codex_autonomous,
+            puct_feedback_surface_balance=config.codex_puct_feedback_surface_balance,
+            puct_feedback_surface_eps=config.codex_puct_feedback_surface_eps,
             wandb_project=config.wandb_project,
             wandb_name=config.experiment_name,
             log_path=log_path,
diff --git a/ttt_discover/rl/codex_no_finetune.py b/ttt_discover/rl/codex_no_finetune.py
index 18a0e9e..a9ed2fa 100644
--- a/ttt_discover/rl/codex_no_finetune.py
+++ b/ttt_discover/rl/codex_no_finetune.py
@@ -137,6 +137,8 @@ class CodexNoFinetuneConfig:
     initial_program_paths: tuple[str, ...] = ()
     initial_pool_paths: tuple[str, ...] = ()
     topk_children: int = 16
+    puct_feedback_surface_balance: bool = False
+    puct_feedback_surface_eps: float = 1e-9
 
     # False is TTT Discover: non-auto, many short samples. True is AutoEvolve:
     # one Codex deep-dive workspace per sample with file edits and shell access.
@@ -960,6 +962,8 @@ def _build_sampler(cfg: CodexNoFinetuneConfig, start_batch: int) -> StateSampler
         batch_size=cfg.groups_per_batch,
         resume_step=start_batch if start_batch > 0 else None,
         topk_children=cfg.topk_children,
+        puct_feedback_surface_balance=cfg.puct_feedback_surface_balance,
+        puct_feedback_surface_eps=cfg.puct_feedback_surface_eps,
     )
 
 
````
</details>

