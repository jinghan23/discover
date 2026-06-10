# codex/diversity-age-balance

## Summary

在 PUCT 排序基础上按非 seed 状态的年龄/归档位置分桶 early/middle/recent，优先覆盖不同年龄段；seed 单独标记，缺口时回退到原 PUCT 顺序。

## Branch State

- Worktree: `/opt/tiger/discover-age-balance`
- HEAD: `2234282`
- Base used for comparison: `2234282`
- Commits ahead of base: `0`
- Commits behind base: `0`
- Group: `diversity_extra`
- Implementation location: `uncommitted worktree diff`

## Implemented Behavior

- The summary above is based on reading the changed Python code, added config fields, and sampler/prompt hooks in this branch.
- For sampler branches, the common pattern is to keep baseline PUCT ordering inside a chosen surface/bucket, while changing which surface/bucket gets the next parent pick.
- Tests detected: `8` test functions across untracked test files.

## Added Markers

### Config fields

- `codex_age_balanced_sampling`
- `age_balanced_sampling`

### Constants

- `NON_SEED_AGE_BUCKETS`
- `INITIAL_OR_SEED_BUCKET`

### Classes

- None

### Functions

- `_ranked_puct_entries`
- `_is_initial_or_seed`
- `_state_timestep`
- `_age_key`
- `_bucket_for_age_rank`
- `_age_metadata_by_state_id`
- `_set_last_sampled`
- `sample_states`
- `sample_states_age_balanced`
- `_pick`

## Diff Summary

- Worktree tracked shortstat: `4 files changed, 257 insertions(+), 17 deletions(-)`
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

- `repro/gpu_mode/run_0608_age_balance.sh (962 bytes)`
- `repro/run_discovery.py (15783 bytes)`
- `tests/test_age_balanced_sampling.py (9915 bytes)`

### Detected Test Functions

- `tests/test_age_balanced_sampling.py::test_seed_bucket_is_not_hard_quota`
- `tests/test_age_balanced_sampling.py::test_age_bucket_assignment_variants`
- `tests/test_age_balanced_sampling.py::test_first_pass_uses_unique_non_seed_buckets`
- `tests/test_age_balanced_sampling.py::test_fallback_relaxes_only_bucket_uniqueness`
- `tests/test_age_balanced_sampling.py::test_full_lineage_blocking_is_preserved`
- `tests/test_age_balanced_sampling.py::test_num_states_one_matches_normal_puct_top_pick`
- `tests/test_age_balanced_sampling.py::test_sample_table_uses_sample_time_age_cache_after_update_and_flush`
- `tests/test_age_balanced_sampling.py::test_unsupported_sampler_api_raises_value_error`

## Untracked File Snapshots

These files are not part of `git diff` yet, but they are part of the worktree implementation state recorded here.

### `repro/gpu_mode/run_0608_age_balance.sh`

````bash
#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

# TTT Discover age-balanced parent sampling: non-auto, read-only Codex samples.
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}" \
CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}" \
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}" \
python repro/run_discovery.py \
  --task trimul \
  --runner codex_no_finetune \
  --experiment-name "${EXPERIMENT_NAME:-trimul_0608_age_balance_gpu2}" \
  --log-root codex_runs/trimul_exec_workspaces \
  --num-epochs "${NUM_EPOCHS:-50}" \
  --group-size 2 \
  --groups-per-batch 4 \
  --num-cpus-per-task 1 \
  --eval-timeout 1200 \
  --wandb-project "" \
  --codex-backend cli \
  --codex-model-name "${CODEX_MODEL_NAME:-gpt-5.5}" \
  --codex-cli-command "${CODEX_CLI_COMMAND:-codex}" \
  --codex-cli-sandbox read-only \
  --codex-cli-timeout "${CODEX_CLI_TIMEOUT:-600}" \
  --codex-max-concurrent-requests 4 \
  --codex-age-balanced-sampling \
  "$@"
````

### `repro/run_discovery.py`

````python
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


EnvLoader = Callable[[], type]


@dataclass(frozen=True)
class TaskSpec:
    env_loader: EnvLoader
    problem_type: str
    experiment_name: str
    wandb_project: str | None
    eval_timeout: int
    num_epochs: int
    group_size: int
    groups_per_batch: int
    initial_pools: tuple[Path, ...] = ()
    uses_gpu: bool = False


def _load_cap_set_env() -> type:
    from examples.cap_set_priority.env import CapSetPriorityEnv

    return CapSetPriorityEnv


def _load_erdos_env() -> type:
    from examples.erdos_min_overlap.env import ErdosMinOverlapEnv

    return ErdosMinOverlapEnv


def _load_gpu_mode_env() -> type:
    from examples.gpu_mode.env import GpuModeEnv

    return GpuModeEnv


def _load_ahc_env() -> type:
    from examples.ahc.env import AhcEnv

    return AhcEnv


def _load_kakeya_env() -> type:
    from examples.kakeya.env import KakeyaEnv

    return KakeyaEnv


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
            env_loader=_load_cap_set_env,
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
            env_loader=_load_erdos_env,
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
        if dimension != 3:
            raise ValueError("The Kakeya environment currently supports only d=3")
        primes = args.primes or "5,7,13"
        return TaskSpec(
            env_loader=_load_kakeya_env,
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
            env_loader=_load_ahc_env,
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
            env_loader=_load_gpu_mode_env,
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
            env_loader=_load_gpu_mode_env,
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
        description=(
            "Launch a TTT-Discover task through a small task registry. Existing "
            "task-specific repro runners remain supported; this is a shared "
            "entrypoint for new or routine runs."
        )
    )
    parser.add_argument(
        "--task",
        required=True,
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
    parser.add_argument(
        "--problem-type",
        default=None,
        help="Override the registry problem_type passed to the environment.",
    )
    parser.add_argument("--dimension", type=int, default=None)
    parser.add_argument(
        "--primes",
        default=None,
        help="Kakeya-only comma-separated prime list, e.g. '5,7,13'.",
    )
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--log-root", default="tinker_log")
    parser.add_argument(
        "--wandb-project",
        default=None,
        help="Override WANDB project. Pass '' to disable.",
    )
    parser.add_argument(
        "--no-default-initial-pool",
        action="store_true",
        help="Do not load default initial-pool files from the task registry.",
    )
    parser.add_argument(
        "--codex-initial-pool",
        action="append",
        default=None,
        help="Reusable Codex initial-pool state JSON. May be repeated.",
    )
    parser.add_argument(
        "--codex-initial-program",
        action="append",
        default=None,
        help="Seed program to verify and add to the Codex sampler pool.",
    )

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
        help="Defaults to read-only unless --codex-autonomous is set.",
    )
    parser.add_argument("--codex-cli-timeout", type=float, default=None)
    parser.add_argument("--codex-max-concurrent-requests", type=int, default=4)
    parser.add_argument(
        "--codex-autonomous",
        action="store_true",
        help="AutoEvolve mode: give Codex a writable workspace for deep dives.",
    )
    parser.add_argument(
        "--codex-age-balanced-sampling",
        action="store_true",
        help="Use age/cohort balanced PUCT parent sampling.",
    )

    parser.add_argument(
        "--gpu",
        default=None,
        help="Set CUDA_VISIBLE_DEVICES for GPU tasks.",
    )
    parser.add_argument("--cuda-device-order", default="PCI_BUS_ID")
    parser.add_argument(
        "--torch-cuda-arch-list",
        default=None,
        help="Optional TORCH_CUDA_ARCH_LIST override, e.g. 8.0 for A800.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved configuration without launching discovery.",
    )
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

    env_type = spec.env_loader()
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

    if args.dry_run:
        print(f"task={args.task}")
        print(f"env_type={env_type.__module__}.{env_type.__name__}")
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
        print(f"codex_age_balanced_sampling={args.codex_age_balanced_sampling}")
        if spec.uses_gpu:
            print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}")
            print(f"CUDA_DEVICE_ORDER={os.environ.get('CUDA_DEVICE_ORDER')!r}")
            print(f"TORCH_CUDA_ARCH_LIST={os.environ.get('TORCH_CUDA_ARCH_LIST')!r}")
        return

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
            age_balanced_sampling=args.codex_age_balanced_sampling,
            autonomous=args.codex_autonomous,
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
        codex_age_balanced_sampling=args.codex_age_balanced_sampling,
        codex_autonomous=args.codex_autonomous,
        remove_constant_reward_groups=remove_constant_reward_groups,
    )
    discover(config)


if __name__ == "__main__":
    main()
````

### `tests/test_age_balanced_sampling.py`

````python
from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

from ttt_discover.codex_utils.runtime import State
from ttt_discover.codex_utils.sampler import (
    INITIAL_OR_SEED_BUCKET,
    PUCTSampler,
)
from ttt_discover.rl.codex_no_finetune import (
    CodexNoFinetuneConfig,
    sample_batch,
)


class DummyEnv:
    state_type = State

    @staticmethod
    def create_initial_state(problem_type: str) -> State:
        return State(
            timestep=-1,
            construction=["initial"],
            code="initial",
            value=0.0,
            parents=[],
            id=f"initial-{problem_type}",
        )


def make_state(
    state_id: str,
    *,
    timestep: int | None,
    value: float,
    parents: list[dict] | None = None,
) -> State:
    state = State(
        timestep=0 if timestep is None else timestep,
        construction=[state_id],
        code=state_id,
        value=value,
        parents=[{"id": "root"}] if parents is None else parents,
        id=state_id,
    )
    if timestep is None:
        state.timestep = None
    return state


class SamplerTestCase(unittest.TestCase):
    def make_sampler(self, states: list[State]) -> tuple[tempfile.TemporaryDirectory, PUCTSampler]:
        tmp = tempfile.TemporaryDirectory()
        sampler = PUCTSampler(
            file_path=str(Path(tmp.name) / "sampler.json"),
            env_type=DummyEnv,
            problem_type="test",
            batch_size=0,
            puct_c=0.0,
            topk_children=0,
        )
        sampler._states = list(states)
        sampler._initial_states = [state for state in states if state.parents == []]
        return tmp, sampler

    def buckets(self, sampler: PUCTSampler) -> dict[str, str]:
        return {
            state_id: bucket
            for state_id, (_age_key, bucket) in sampler._age_metadata_by_state_id().items()
        }

    def test_seed_bucket_is_not_hard_quota(self) -> None:
        seed = make_state("seed", timestep=-1, value=0.0, parents=[])
        states = [
            seed,
            make_state("early_a", timestep=0, value=100.0),
            make_state("early_b", timestep=1, value=99.0),
            make_state("middle", timestep=2, value=98.0),
            make_state("recent", timestep=3, value=97.0),
        ]
        tmp, sampler = self.make_sampler(states)
        with tmp:
            picked = sampler.sample_states_age_balanced(4)
            self.assertEqual(
                [state.id for state in picked],
                ["early_a", "middle", "recent", "early_b"],
            )
            self.assertNotIn("seed", [state.id for state in picked])
            self.assertEqual(sampler.get_sample_stats()["puct/age_balance_fallback_count"], 1)

    def test_age_bucket_assignment_variants(self) -> None:
        normal_states = [
            make_state("seed", timestep=-1, value=0.0, parents=[]),
            make_state("old", timestep=0, value=1.0),
            make_state("mid", timestep=5, value=2.0),
            make_state("new", timestep=10, value=3.0),
        ]
        tmp, sampler = self.make_sampler(normal_states)
        with tmp:
            buckets = self.buckets(sampler)
            self.assertEqual(buckets["seed"], INITIAL_OR_SEED_BUCKET)
            self.assertEqual(buckets["old"], "early")
            self.assertEqual(buckets["mid"], "middle")
            self.assertEqual(buckets["new"], "recent")

        repeated_states = [
            make_state("a", timestep=1, value=1.0),
            make_state("b", timestep=1, value=2.0),
            make_state("c", timestep=2, value=3.0),
            make_state("d", timestep=2, value=4.0),
            make_state("e", timestep=3, value=5.0),
            make_state("f", timestep=3, value=6.0),
        ]
        tmp, sampler = self.make_sampler(repeated_states)
        with tmp:
            buckets = self.buckets(sampler)
            self.assertEqual([buckets[state.id] for state in repeated_states], [
                "early",
                "early",
                "middle",
                "middle",
                "recent",
                "recent",
            ])

        all_same_states = [
            make_state("idx0", timestep=7, value=10.0),
            make_state("idx1", timestep=7, value=9.0),
            make_state("idx2", timestep=7, value=8.0),
        ]
        tmp, sampler = self.make_sampler(all_same_states)
        with tmp:
            buckets = self.buckets(sampler)
            self.assertEqual([buckets[state.id] for state in all_same_states], [
                "early",
                "middle",
                "recent",
            ])

        missing_states = [
            make_state("idx0", timestep=10, value=10.0),
            make_state("idx1", timestep=None, value=9.0),
            make_state("idx2", timestep=0, value=8.0),
        ]
        tmp, sampler = self.make_sampler(missing_states)
        with tmp:
            buckets = self.buckets(sampler)
            self.assertEqual([buckets[state.id] for state in missing_states], [
                "early",
                "middle",
                "recent",
            ])

    def test_first_pass_uses_unique_non_seed_buckets(self) -> None:
        states = [
            make_state("early_a", timestep=0, value=100.0),
            make_state("early_b", timestep=1, value=99.0),
            make_state("middle", timestep=2, value=98.0),
            make_state("recent", timestep=3, value=97.0),
        ]
        tmp, sampler = self.make_sampler(states)
        with tmp:
            picked = sampler.sample_states_age_balanced(3)
            self.assertEqual(
                [state.id for state in picked],
                ["early_a", "middle", "recent"],
            )
            self.assertEqual(sampler._last_age_balance_fallback, [False, False, False])

    def test_fallback_relaxes_only_bucket_uniqueness(self) -> None:
        states = [
            make_state("early_a", timestep=0, value=100.0),
            make_state("early_b", timestep=1, value=99.0),
            make_state("middle", timestep=2, value=98.0),
            make_state("recent", timestep=3, value=97.0),
        ]
        tmp, sampler = self.make_sampler(states)
        with tmp:
            picked = sampler.sample_states_age_balanced(4)
            self.assertEqual(
                [state.id for state in picked],
                ["early_a", "middle", "recent", "early_b"],
            )
            self.assertEqual(sampler._last_age_balance_fallback, [False, False, False, True])
            self.assertEqual(sampler.get_sample_stats()["puct/age_balance_shortage_count"], 0)

    def test_full_lineage_blocking_is_preserved(self) -> None:
        parent = make_state("parent", timestep=0, value=99.0)
        child = make_state(
            "child",
            timestep=2,
            value=100.0,
            parents=[{"id": "parent"}, {"id": "root"}],
        )
        unrelated = make_state("unrelated", timestep=1, value=98.0)
        tmp, sampler = self.make_sampler([parent, unrelated, child])
        with tmp:
            picked = sampler.sample_states_age_balanced(2)
            self.assertEqual([state.id for state in picked], ["child", "unrelated"])

    def test_num_states_one_matches_normal_puct_top_pick(self) -> None:
        states = [
            make_state("seed", timestep=-1, value=100.0, parents=[]),
            make_state("nonseed", timestep=2, value=99.0),
        ]
        tmp, sampler = self.make_sampler(states)
        with tmp:
            normal = sampler.sample_states(1)
            balanced = sampler.sample_states_age_balanced(1)
            self.assertEqual([state.id for state in balanced], [state.id for state in normal])

    def test_sample_table_uses_sample_time_age_cache_after_update_and_flush(self) -> None:
        states = [
            make_state("old", timestep=0, value=8.0),
            make_state("mid", timestep=1, value=9.0),
            make_state("new", timestep=2, value=10.0),
        ]
        tmp, sampler = self.make_sampler(states)
        with tmp:
            picked = sampler.sample_states_age_balanced(2)
            columns, before_rows = sampler.get_sample_table()
            col = {name: idx for idx, name in enumerate(columns)}
            self.assertEqual(before_rows[0][col["age_key"]], (2, 2))
            self.assertEqual(before_rows[0][col["age_bucket"]], "recent")

            picked[0].timestep = -1
            picked[0].parents = []
            child = make_state("child", timestep=3, value=11.0, parents=[{"id": picked[0].id}])
            sampler.update_states([child], [picked[0]], save=False)
            sampler.flush(step=1)

            _columns, after_rows = sampler.get_sample_table()
            self.assertEqual(after_rows[0][col["age_key"]], (2, 2))
            self.assertEqual(after_rows[0][col["age_bucket"]], "recent")
            self.assertFalse(after_rows[0][col["age_balance_fallback"]])

    def test_unsupported_sampler_api_raises_value_error(self) -> None:
        class NoAgeBalancedSampler:
            def sample_states(self, num_states: int) -> list[State]:
                return []

            def update_states(
                self,
                states: list[State],
                parent_states: list[State],
                save: bool = True,
                step: int | None = None,
            ) -> None:
                return None

            def flush(self, step: int | None = None) -> None:
                return None

        cfg = CodexNoFinetuneConfig(
            env_type=DummyEnv,
            groups_per_batch=1,
            group_size=1,
            age_balanced_sampling=True,
        )
        with self.assertRaisesRegex(ValueError, "sample_states_age_balanced"):
            asyncio.run(sample_batch(cfg, NoAgeBalancedSampler(), 0))


if __name__ == "__main__":
    unittest.main()
````

## Diff Stat

### Worktree Tracked Diff Stat

````text
 ttt_discover/codex_utils/discovery.py |   2 +
 ttt_discover/codex_utils/sampler.py   | 259 +++++++++++++++++++++++++++++++---
 ttt_discover/discovery.py             |   2 +
 ttt_discover/rl/codex_no_finetune.py  |  11 +-
 4 files changed, 257 insertions(+), 17 deletions(-)
````
## Raw Diff

<details>
<summary>Tracked worktree patch</summary>

````diff
diff --git a/ttt_discover/codex_utils/discovery.py b/ttt_discover/codex_utils/discovery.py
index b21a80f..259a253 100644
--- a/ttt_discover/codex_utils/discovery.py
+++ b/ttt_discover/codex_utils/discovery.py
@@ -49,6 +49,7 @@ class DiscoverConfig:
     codex_max_concurrent_requests: int | None = 4
     codex_initial_program_paths: tuple[str, ...] = ()
     codex_initial_pool_paths: tuple[str, ...] = ()
+    codex_age_balanced_sampling: bool = False
     codex_autonomous: bool = False
 
 
@@ -84,6 +85,7 @@ def _run_codex_no_finetune(config: DiscoverConfig) -> None:
         max_concurrent_requests=config.codex_max_concurrent_requests,
         initial_program_paths=config.codex_initial_program_paths,
         initial_pool_paths=config.codex_initial_pool_paths,
+        age_balanced_sampling=config.codex_age_balanced_sampling,
         autonomous=config.codex_autonomous,
         wandb_project=config.wandb_project,
         wandb_name=experiment_name,
diff --git a/ttt_discover/codex_utils/sampler.py b/ttt_discover/codex_utils/sampler.py
index 5c5d4d1..4d5cc15 100644
--- a/ttt_discover/codex_utils/sampler.py
+++ b/ttt_discover/codex_utils/sampler.py
@@ -17,6 +17,10 @@ from ttt_discover.codex_utils.runtime import State, state_from_dict, to_json_ser
 logger = logging.getLogger(__name__)
 
 
+NON_SEED_AGE_BUCKETS = ("early", "middle", "recent")
+INITIAL_OR_SEED_BUCKET = "initial_or_seed"
+
+
 @contextmanager
 def _file_lock(lock_path: str, *, poll_s: float = 0.05, stale_s: float = 600.0):
     while True:
@@ -375,6 +379,10 @@ class PUCTSampler(StateSampler):
         self._T: int = 0
         self._last_scale: float = 1.0
         self._last_puct_stats: list[tuple[int, float, float, float, float]] = []
+        self._last_age_keys: list[tuple[Any, int] | None] = []
+        self._last_age_buckets: list[str] = []
+        self._last_age_balance_fallback: list[bool] = []
+        self._last_age_balance_shortage_count: int = 0
         
         if resume_step is not None:
             self._load(resume_step)
@@ -489,19 +497,12 @@ class PUCTSampler(StateSampler):
                     queue.append(child_id)
         return lineage
 
-    def sample_states(self, num_states: int) -> list[State]:
+    def _ranked_puct_entries(self) -> list[tuple[float, float, State, int, float, float, float]]:
         initial_ids = {s.id for s in self._initial_states}
         candidates = list(self._states)
 
         if not candidates:
-            picked = [
-                create_initial_state(self.env_type, self.problem_type)
-                for _ in range(num_states)
-            ]
-            self._last_sampled_states = picked
-            self._last_sampled_indices = []
-            self._last_puct_stats = [(0, 0.0, 0.0, 0.0, 0.0) for _ in picked]
-            return picked
+            return []
 
         vals = np.array([float(s.value if s.value is not None else float("-inf")) for s in candidates])
         non_initial_mask = np.array([s.id not in initial_ids for s in candidates])
@@ -520,6 +521,120 @@ class PUCTSampler(StateSampler):
             scores.append((score, vals[i], s, n, Q, P[i], bonus))
 
         scores.sort(key=lambda x: (x[0], x[1]), reverse=True)
+        return scores
+
+    def _is_initial_or_seed(self, state: State) -> bool:
+        timestep = getattr(state, "timestep", None)
+        return (
+            (isinstance(timestep, (int, float)) and timestep < 0)
+            or getattr(state, "parents", []) == []
+        )
+
+    def _state_timestep(self, state: State) -> int | None:
+        timestep = getattr(state, "timestep", None)
+        if isinstance(timestep, bool):
+            return int(timestep)
+        if isinstance(timestep, int):
+            return timestep
+        return None
+
+    def _age_key(self, state: State, archive_idx: int) -> tuple[Any, int]:
+        return (getattr(state, "timestep", None), archive_idx)
+
+    def _bucket_for_age_rank(self, rank: int, total: int) -> str:
+        if total <= 0:
+            return ""
+        bucket_idx = min(
+            (rank * len(NON_SEED_AGE_BUCKETS)) // total,
+            len(NON_SEED_AGE_BUCKETS) - 1,
+        )
+        return NON_SEED_AGE_BUCKETS[bucket_idx]
+
+    def _age_metadata_by_state_id(self) -> dict[str, tuple[tuple[Any, int], str]]:
+        metadata: dict[str, tuple[tuple[Any, int], str]] = {}
+        non_seed_records: list[tuple[int, State, int | None]] = []
+        has_missing_timestep = False
+        timesteps: list[int] = []
+
+        for archive_idx, state in enumerate(self._states):
+            age_key = self._age_key(state, archive_idx)
+            if self._is_initial_or_seed(state):
+                metadata[state.id] = (age_key, INITIAL_OR_SEED_BUCKET)
+                continue
+
+            timestep = self._state_timestep(state)
+            if timestep is None:
+                has_missing_timestep = True
+            else:
+                timesteps.append(timestep)
+            non_seed_records.append((archive_idx, state, timestep))
+            metadata[state.id] = (age_key, "")
+
+        use_timestep_order = (
+            not has_missing_timestep
+            and len(timesteps) == len(non_seed_records)
+            and len(set(timesteps)) > 1
+        )
+        if use_timestep_order:
+            ordered = sorted(non_seed_records, key=lambda item: (item[2], item[0]))
+        else:
+            ordered = sorted(non_seed_records, key=lambda item: item[0])
+
+        total = len(ordered)
+        for rank, (archive_idx, state, _timestep) in enumerate(ordered):
+            metadata[state.id] = (
+                self._age_key(state, archive_idx),
+                self._bucket_for_age_rank(rank, total),
+            )
+        return metadata
+
+    def _set_last_sampled(
+        self,
+        picked: list[State],
+        picked_entries: list[tuple[float, float, State, int, float, float, float]],
+        *,
+        age_balance_fallback: list[bool] | None = None,
+        age_balance_shortage_count: int = 0,
+    ) -> None:
+        state_id_to_idx = {s.id: i for i, s in enumerate(self._states)}
+        age_metadata = self._age_metadata_by_state_id()
+
+        self._last_sampled_states = picked
+        self._last_sampled_indices = [state_id_to_idx.get(s.id, -1) for s in picked]
+        if len(picked_entries) == len(picked):
+            self._last_puct_stats = [
+                (entry[3], entry[4], entry[5], entry[6], entry[0])
+                for entry in picked_entries
+            ]
+        else:
+            self._last_puct_stats = [(0, 0.0, 0.0, 0.0, 0.0) for _ in picked]
+
+        self._last_age_keys = []
+        self._last_age_buckets = []
+        for state in picked:
+            fallback_bucket = (
+                INITIAL_OR_SEED_BUCKET if self._is_initial_or_seed(state) else ""
+            )
+            age_key, age_bucket = age_metadata.get(state.id, (None, fallback_bucket))
+            self._last_age_keys.append(age_key)
+            self._last_age_buckets.append(age_bucket)
+
+        if age_balance_fallback is None:
+            age_balance_fallback = [False] * len(picked)
+        self._last_age_balance_fallback = list(age_balance_fallback)
+        self._last_age_balance_shortage_count = max(0, int(age_balance_shortage_count))
+
+    def sample_states(self, num_states: int) -> list[State]:
+        initial_ids = {s.id for s in self._initial_states}
+        scores = self._ranked_puct_entries()
+
+        if not scores:
+            picked = [
+                create_initial_state(self.env_type, self.problem_type)
+                for _ in range(num_states)
+            ]
+            self._set_last_sampled(picked, [])
+            return picked
 
         if num_states > 1:
             children_map = self._build_children_map()
@@ -537,10 +652,7 @@ class PUCTSampler(StateSampler):
             top_scores = scores[:num_states]
             picked = [t[2] for t in top_scores]
 
-        state_id_to_idx = {s.id: i for i, s in enumerate(self._states)}
-        self._last_sampled_states = picked
-        self._last_sampled_indices = [state_id_to_idx.get(s.id, -1) for s in picked]
-        self._last_puct_stats = [(t[3], t[4], t[5], t[6], t[0]) for t in top_scores]
+        self._set_last_sampled(picked, top_scores)
 
         for s in picked:
             if s.id in initial_ids:
@@ -548,6 +660,69 @@ class PUCTSampler(StateSampler):
 
         return picked
 
+    def sample_states_age_balanced(self, num_states: int) -> list[State]:
+        """Sample from the PUCT-ranked list while spreading non-seed age cohorts."""
+        if num_states <= 1:
+            return self.sample_states(num_states)
+
+        initial_ids = {s.id for s in self._initial_states}
+        scores = self._ranked_puct_entries()
+        if not scores:
+            return self.sample_states(num_states)
+
+        age_metadata = self._age_metadata_by_state_id()
+        children_map = self._build_children_map()
+        picked: list[State] = []
+        picked_entries: list[tuple[float, float, State, int, float, float, float]] = []
+        fallback_flags: list[bool] = []
+        picked_ids: set[str] = set()
+        blocked_ids: set[str] = set()
+        used_non_seed_buckets: set[str] = set()
+
+        def _pick(entry: tuple[float, float, State, int, float, float, float], *, fallback: bool) -> None:
+            state = entry[2]
+            picked.append(state)
+            picked_entries.append(entry)
+            fallback_flags.append(fallback)
+            picked_ids.add(state.id)
+            blocked_ids.update(self._get_full_lineage(state, children_map))
+
+        for entry in scores:
+            state = entry[2]
+            if state.id in blocked_ids:
+                continue
+            _age_key, age_bucket = age_metadata.get(state.id, ("", ""))
+            if age_bucket not in NON_SEED_AGE_BUCKETS:
+                continue
+            if age_bucket in used_non_seed_buckets:
+                continue
+            _pick(entry, fallback=False)
+            used_non_seed_buckets.add(age_bucket)
+            if len(picked) >= num_states:
+                break
+
+        if len(picked) < num_states:
+            for entry in scores:
+                state = entry[2]
+                if state.id in picked_ids or state.id in blocked_ids:
+                    continue
+                _pick(entry, fallback=True)
+                if len(picked) >= num_states:
+                    break
+
+        self._set_last_sampled(
+            picked,
+            picked_entries,
+            age_balance_fallback=fallback_flags,
+            age_balance_shortage_count=max(0, num_states - len(picked)),
+        )
+
+        for state in picked:
+            if state.id in initial_ids:
+                self._refresh_random_construction(state)
+
+        return picked
+
     def update_states(self, states: list[State], parent_states: list[State], save: bool = True, step: int | None = None):
         if not states:
             return
@@ -719,11 +894,21 @@ class PUCTSampler(StateSampler):
         sampled_values = [s.value for s in self._last_sampled_states]
         sampled_timesteps = [s.timestep for s in self._last_sampled_states]
         sampled_constr_lens = [len(s.construction) if hasattr(s, 'construction') and s.construction else 0 for s in self._last_sampled_states]
+        unique_non_seed_age_buckets = {
+            bucket
+            for bucket in self._last_age_buckets
+            if bucket in NON_SEED_AGE_BUCKETS
+        }
         stats = {
             "puct/buffer_size": len(self._states),
             "puct/sampled_size": len(self._last_sampled_states),
             "puct/T": self._T,
             "puct/scale_last": float(self._last_scale),
+            "puct/age_bucket_unique_count": len(unique_non_seed_age_buckets),
+            "puct/age_balance_fallback_count": sum(
+                1 for used_fallback in self._last_age_balance_fallback if used_fallback
+            ),
+            "puct/age_balance_shortage_count": self._last_age_balance_shortage_count,
         }
         stats.update(_stats(buffer_values, "puct/buffer_value"))
         stats.update(_stats(buffer_timesteps, "puct/buffer_timestep"))
@@ -734,18 +919,60 @@ class PUCTSampler(StateSampler):
         return stats
 
     def get_sample_table(self) -> tuple[list[str], list[tuple]]:
-        columns = ["buffer_idx", "timestep", "value", "terminal_value", "parent_value", "construction_len", "observation_len", "n", "Q", "P", "bonus", "score"]
+        columns = [
+            "buffer_idx",
+            "timestep",
+            "value",
+            "terminal_value",
+            "parent_value",
+            "construction_len",
+            "observation_len",
+            "n",
+            "Q",
+            "P",
+            "bonus",
+            "score",
+            "age_key",
+            "age_bucket",
+            "age_balance_fallback",
+        ]
         rows = []
         if not self._last_sampled_states:
             return columns, rows
         indices = self._last_sampled_indices if len(self._last_sampled_indices) == len(self._last_sampled_states) else [-1] * len(self._last_sampled_states)
         stats = self._last_puct_stats if len(self._last_puct_stats) == len(self._last_sampled_states) else [(0, 0.0, 0.0, 0.0, 0.0)] * len(self._last_sampled_states)
-        for idx, state, (n, Q, P, bonus, score) in zip(indices, self._last_sampled_states, stats):
+        age_keys = self._last_age_keys if len(self._last_age_keys) == len(self._last_sampled_states) else [None] * len(self._last_sampled_states)
+        age_buckets = self._last_age_buckets if len(self._last_age_buckets) == len(self._last_sampled_states) else [""] * len(self._last_sampled_states)
+        fallback_flags = self._last_age_balance_fallback if len(self._last_age_balance_fallback) == len(self._last_sampled_states) else [False] * len(self._last_sampled_states)
+        for idx, state, (n, Q, P, bonus, score), age_key, age_bucket, used_fallback in zip(
+            indices,
+            self._last_sampled_states,
+            stats,
+            age_keys,
+            age_buckets,
+            fallback_flags,
+        ):
             parent_val = state.parent_values[0] if state.parent_values else None
             constr = getattr(state, 'construction', None)
             constr_len = len(constr) if constr is not None else 0
             obs_len = len(state.observation) if state.observation else 0
-            rows.append((idx, state.timestep, state.value, 0, parent_val, constr_len, obs_len, n, Q, P, bonus, score))
+            rows.append((
+                idx,
+                state.timestep,
+                state.value,
+                0,
+                parent_val,
+                constr_len,
+                obs_len,
+                n,
+                Q,
+                P,
+                bonus,
+                score,
+                age_key,
+                age_bucket,
+                used_fallback,
+            ))
         return columns, rows
 
 
diff --git a/ttt_discover/discovery.py b/ttt_discover/discovery.py
index 6cdb434..e7a9eba 100644
--- a/ttt_discover/discovery.py
+++ b/ttt_discover/discovery.py
@@ -62,6 +62,7 @@ class DiscoverConfig:
     codex_max_concurrent_requests: int | None = 4
     codex_initial_program_paths: tuple[str, ...] = ()
     codex_initial_pool_paths: tuple[str, ...] = ()
+    codex_age_balanced_sampling: bool = False
     codex_autonomous: bool = False
 
 
@@ -145,6 +146,7 @@ async def discover_impl(config: DiscoverConfig):
             max_concurrent_requests=config.codex_max_concurrent_requests,
             initial_program_paths=config.codex_initial_program_paths,
             initial_pool_paths=config.codex_initial_pool_paths,
+            age_balanced_sampling=config.codex_age_balanced_sampling,
             autonomous=config.codex_autonomous,
             wandb_project=config.wandb_project,
             wandb_name=config.experiment_name,
diff --git a/ttt_discover/rl/codex_no_finetune.py b/ttt_discover/rl/codex_no_finetune.py
index 18a0e9e..cb2213f 100644
--- a/ttt_discover/rl/codex_no_finetune.py
+++ b/ttt_discover/rl/codex_no_finetune.py
@@ -137,6 +137,7 @@ class CodexNoFinetuneConfig:
     initial_program_paths: tuple[str, ...] = ()
     initial_pool_paths: tuple[str, ...] = ()
     topk_children: int = 16
+    age_balanced_sampling: bool = False
 
     # False is TTT Discover: non-auto, many short samples. True is AutoEvolve:
     # one Codex deep-dive workspace per sample with file edits and shell access.
@@ -820,7 +821,15 @@ async def sample_batch(
     i_batch: int,
 ) -> tuple[list[CandidateResult], dict[str, Any], list[CandidateResult]]:
     metrics: dict[str, Any] = {}
-    parent_states = sampler.sample_states(cfg.groups_per_batch)
+    if cfg.age_balanced_sampling:
+        sample_age_balanced = getattr(sampler, "sample_states_age_balanced", None)
+        if not callable(sample_age_balanced):
+            raise ValueError(
+                "age_balanced_sampling requires sampler.sample_states_age_balanced()"
+            )
+        parent_states = sample_age_balanced(cfg.groups_per_batch)
+    else:
+        parent_states = sampler.sample_states(cfg.groups_per_batch)
     semaphore = (
         asyncio.Semaphore(cfg.max_concurrent_requests)
         if cfg.max_concurrent_requests is not None
````
</details>

