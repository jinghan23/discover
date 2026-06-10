# codex/diversity-value-quantile-balance

## Summary

对有效 numeric value 做分位桶（另有 seed_or_invalid_value），在不同 value quantile 之间平衡父选择。

## Branch State

- Worktree: `/opt/tiger/discover-value-quantile-balance`
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

- `codex_value_quantile_balanced_sampling`
- `value_quantile_balanced_sampling`

### Constants

- `_INVALID_VALUE_BUCKET`
- `_VALUE_QUANTILE_BUCKETS`

### Classes

- None

### Functions

- `_finite_numeric_value`
- `_ranked_puct_entries`
- `_set_last_sampled`
- `_set_last_sampled_without_puct`
- `_refresh_sampled_initials`
- `_is_value_quantile_eligible`
- `_value_quantile_info`
- `sample_states_value_quantile_balanced`

## Diff Summary

- Worktree tracked shortstat: `4 files changed, 399 insertions(+), 35 deletions(-)`
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

- `repro/gpu_mode/run_0608_value_quantile_balance.sh (967 bytes)`
- `repro/run_discovery.py (15966 bytes)`
- `tests/test_value_quantile_balanced_sampling.py (10958 bytes)`

### Detected Test Functions

- `tests/test_value_quantile_balanced_sampling.py::test_num_states_le_one_matches_normal_puct`
- `tests/test_value_quantile_balanced_sampling.py::test_negative_finite_values_bucket_correctly`
- `tests/test_value_quantile_balanced_sampling.py::test_seed_and_invalid_values_are_invalid_bucket`
- `tests/test_value_quantile_balanced_sampling.py::test_all_equal_finite_values_collapse_to_uniform_bucket`
- `tests/test_value_quantile_balanced_sampling.py::test_two_unique_values_use_promising_and_elite_buckets`
- `tests/test_value_quantile_balanced_sampling.py::test_first_pass_unique_buckets_then_fallback_in_puct_order`
- `tests/test_value_quantile_balanced_sampling.py::test_seed_invalid_can_only_enter_on_fallback`
- `tests/test_value_quantile_balanced_sampling.py::test_full_lineage_blocking_is_preserved`
- `tests/test_value_quantile_balanced_sampling.py::test_sample_table_uses_sample_time_cache_after_update_and_flush`
- `tests/test_value_quantile_balanced_sampling.py::test_unsupported_sampler_api_raises_value_error`
- `tests/test_value_quantile_balanced_sampling.py::test_flag_off_uses_ordinary_sampler`

## Untracked File Snapshots

These files are not part of `git diff` yet, but they are part of the worktree implementation state recorded here.

### `repro/gpu_mode/run_0608_value_quantile_balance.sh`

````bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

# TTT Discover sampling mode: intentionally no --codex-autonomous.
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}" \
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}" \
python repro/run_discovery.py \
  --task gpu_mode:trimul \
  --runner codex_no_finetune \
  --experiment-name "${EXPERIMENT_NAME:-gpu-mode-trimul-value-quantile-balance-0608}" \
  --num-epochs "${NUM_EPOCHS:-50}" \
  --groups-per-batch 4 \
  --group-size 2 \
  --num-cpus-per-task 1 \
  --eval-timeout 1200 \
  --codex-backend cli \
  --codex-model-name "${CODEX_MODEL_NAME:-gpt-5.5}" \
  --codex-cli-command "${CODEX_CLI_COMMAND:-codex}" \
  --codex-cli-sandbox read-only \
  --codex-cli-timeout "${CODEX_CLI_TIMEOUT:-600}" \
  --codex-max-concurrent-requests 4 \
  --codex-value-quantile-balanced-sampling \
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
        "--codex-value-quantile-balanced-sampling",
        action="store_true",
        help="Use value-quantile balanced PUCT parent sampling.",
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
    if args.dry_run:
        print(f"task={args.task}")
        print(f"env_type={env_type.__module__}.{env_type.__name__}")
        print(f"problem_type={problem_type!r}")
        print(f"experiment_name={experiment_name!r}")
        print(f"log_root={args.log_root!r}")
        print(f"log_path={str(Path(args.log_root) / experiment_name)!r}")
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
            "codex_value_quantile_balanced_sampling="
            f"{args.codex_value_quantile_balanced_sampling}"
        )
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
            value_quantile_balanced_sampling=(
                args.codex_value_quantile_balanced_sampling
            ),
            autonomous=args.codex_autonomous,
            wandb_project=wandb_project,
            wandb_name=experiment_name,
            log_path=str(Path(args.log_root) / experiment_name),
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
        codex_value_quantile_balanced_sampling=(
            args.codex_value_quantile_balanced_sampling
        ),
        codex_autonomous=args.codex_autonomous,
        remove_constant_reward_groups=remove_constant_reward_groups,
    )
    discover(config)


if __name__ == "__main__":
    main()
````

### `tests/test_value_quantile_balanced_sampling.py`

````python
from __future__ import annotations

import asyncio
import os
import tempfile
import unittest

from ttt_discover.codex_utils.runtime import State
from ttt_discover.codex_utils.sampler import PUCTSampler
from ttt_discover.rl.codex_no_finetune import CodexNoFinetuneConfig, sample_batch


class DummyEnv:
    state_type = State

    @staticmethod
    def create_initial_state(problem_type: str) -> State:
        return State(
            timestep=0,
            construction=["initial", problem_type],
            code="initial",
            value=0.0,
            id=f"initial-{problem_type or 'default'}",
        )


def make_state(
    state_id: str,
    value,
    *,
    parent_ids: tuple[str, ...] | None = ("root",),
    timestep: int = 1,
) -> State:
    parents = (
        []
        if parent_ids is None
        else [{"id": parent_id, "timestep": 0} for parent_id in parent_ids]
    )
    return State(
        timestep=timestep,
        construction=[state_id],
        code=state_id,
        value=value,
        parents=parents,
        id=state_id,
    )


class ValueQuantileBalancedSamplingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self._sampler_idx = 0

    def make_sampler(
        self,
        states: list[State],
        *,
        initial_states: list[State] | None = None,
    ) -> PUCTSampler:
        self._sampler_idx += 1
        sampler = PUCTSampler(
            file_path=os.path.join(self.tmp.name, f"sampler_{self._sampler_idx}.json"),
            env_type=DummyEnv,
            batch_size=1,
            puct_c=0.0,
            topk_children=16,
        )
        initial_states = initial_states or []
        sampler._initial_states = list(initial_states)
        sampler._states = list(initial_states) + list(states)
        sampler._n = {}
        sampler._m = {}
        sampler._T = 0
        return sampler

    def value_rows(self, sampler: PUCTSampler) -> list[dict]:
        columns, rows = sampler.get_sample_table()
        return [dict(zip(columns, row, strict=False)) for row in rows]

    def test_num_states_le_one_matches_normal_puct(self) -> None:
        states = [
            make_state("low", 1.0),
            make_state("high", 3.0),
            make_state("mid", 2.0),
        ]
        normal = self.make_sampler(states)
        balanced = self.make_sampler(
            [make_state("low", 1.0), make_state("high", 3.0), make_state("mid", 2.0)]
        )

        self.assertEqual(
            [state.id for state in normal.sample_states(1)],
            [state.id for state in balanced.sample_states_value_quantile_balanced(1)],
        )
        self.assertEqual(
            [state.id for state in normal.sample_states(0)],
            [state.id for state in balanced.sample_states_value_quantile_balanced(0)],
        )

    def test_negative_finite_values_bucket_correctly(self) -> None:
        states = [
            make_state("promising", -3.0),
            make_state("strong", -2.0),
            make_state("elite", -1.0),
        ]
        sampler = self.make_sampler(states)

        picked = sampler.sample_states_value_quantile_balanced(3)
        rows = self.value_rows(sampler)
        buckets = {state.id: row["value_quantile_bucket"] for state, row in zip(picked, rows)}

        self.assertEqual([state.id for state in picked], ["elite", "strong", "promising"])
        self.assertEqual(buckets["promising"], "value_promising")
        self.assertEqual(buckets["strong"], "value_strong")
        self.assertEqual(buckets["elite"], "value_elite")

    def test_seed_and_invalid_values_are_invalid_bucket(self) -> None:
        seed = make_state("seed", 100.0, parent_ids=None, timestep=-1)
        invalid_states = [
            make_state("parentless", 5.0, parent_ids=None),
            make_state("negative_timestep", 4.0, timestep=-1),
            make_state("none", None),
            make_state("nan", float("nan")),
            make_state("inf", float("inf")),
            make_state("nonnumeric", "bad"),
        ]
        sampler = self.make_sampler(invalid_states, initial_states=[seed])

        sampler.sample_states_value_quantile_balanced(10)
        rows = self.value_rows(sampler)
        stats = sampler.get_sample_stats()

        self.assertTrue(rows)
        self.assertTrue(
            all(row["value_quantile_bucket"] == "seed_or_invalid_value" for row in rows)
        )
        self.assertTrue(all(row["value_quantile_balance_fallback"] for row in rows))
        self.assertEqual(stats["puct/value_quantile_bucket_unique_count"], 0)

    def test_all_equal_finite_values_collapse_to_uniform_bucket(self) -> None:
        states = [
            make_state("a", 1.0),
            make_state("b", 1.0),
            make_state("c", 1.0),
        ]
        sampler = self.make_sampler(states)

        sampler.sample_states_value_quantile_balanced(3)
        rows = self.value_rows(sampler)
        stats = sampler.get_sample_stats()

        self.assertEqual(
            [row["value_quantile_bucket"] for row in rows],
            ["uniform_value", "uniform_value", "uniform_value"],
        )
        self.assertEqual(
            [row["value_quantile_balance_fallback"] for row in rows],
            [False, True, True],
        )
        self.assertEqual(stats["puct/value_quantile_bucket_unique_count"], 1)
        self.assertEqual(stats["puct/value_quantile_balance_fallback_count"], 2)

    def test_two_unique_values_use_promising_and_elite_buckets(self) -> None:
        states = [
            make_state("low", 1.0),
            make_state("high", 2.0),
            make_state("high_tie", 2.0),
        ]
        sampler = self.make_sampler(states)

        picked = sampler.sample_states_value_quantile_balanced(3)
        rows = self.value_rows(sampler)
        rows_by_id = {state.id: row for state, row in zip(picked, rows)}

        self.assertEqual(rows_by_id["low"]["value_quantile_bucket"], "value_promising")
        self.assertEqual(rows_by_id["high"]["value_quantile_bucket"], "value_elite")
        self.assertEqual(rows_by_id["high_tie"]["value_quantile_bucket"], "value_elite")

    def test_first_pass_unique_buckets_then_fallback_in_puct_order(self) -> None:
        states = [
            make_state("v6", 6.0),
            make_state("v5", 5.0),
            make_state("v4", 4.0),
            make_state("v3", 3.0),
            make_state("v2a", 2.0),
            make_state("v2b", 2.0),
            make_state("v1", 1.0),
        ]
        sampler = self.make_sampler(states)

        picked = sampler.sample_states_value_quantile_balanced(7)
        rows = self.value_rows(sampler)
        rows_by_id = {state.id: row for state, row in zip(picked, rows)}

        self.assertEqual(
            [state.id for state in picked[:4]],
            ["v6", "v4", "v2a", "v5"],
        )
        self.assertEqual(
            [row["value_quantile_balance_fallback"] for row in rows[:4]],
            [False, False, False, True],
        )
        self.assertEqual(
            rows_by_id["v2a"]["value_quantile_bucket"],
            rows_by_id["v2b"]["value_quantile_bucket"],
        )
        self.assertEqual(rows_by_id["v2a"]["value_bucket_key"], 2.0)
        self.assertEqual(rows_by_id["v2b"]["value_bucket_key"], 2.0)

    def test_seed_invalid_can_only_enter_on_fallback(self) -> None:
        seed = make_state("seed", 100.0, parent_ids=None, timestep=-1)
        states = [
            make_state("low", 1.0),
            make_state("mid", 2.0),
            make_state("high", 3.0),
        ]
        sampler = self.make_sampler(states, initial_states=[seed])

        picked = sampler.sample_states_value_quantile_balanced(4)
        rows = self.value_rows(sampler)

        self.assertEqual([state.id for state in picked], ["high", "mid", "low", "seed"])
        self.assertEqual(rows[-1]["value_quantile_bucket"], "seed_or_invalid_value")
        self.assertTrue(rows[-1]["value_quantile_balance_fallback"])

    def test_full_lineage_blocking_is_preserved(self) -> None:
        parent = make_state("parent", 10.0, parent_ids=("root-parent",))
        child = make_state("child", 9.0, parent_ids=("parent",))
        unrelated = make_state("unrelated", 1.0, parent_ids=("root-unrelated",))
        sampler = self.make_sampler([parent, child, unrelated])

        picked = sampler.sample_states_value_quantile_balanced(3)
        stats = sampler.get_sample_stats()

        self.assertEqual([state.id for state in picked], ["parent", "unrelated"])
        self.assertNotIn("child", [state.id for state in picked])
        self.assertEqual(stats["puct/value_quantile_balance_shortage_count"], 1)

    def test_sample_table_uses_sample_time_cache_after_update_and_flush(self) -> None:
        states = [
            make_state("low", 1.0),
            make_state("mid", 2.0),
            make_state("high", 3.0),
        ]
        sampler = self.make_sampler(states)

        picked = sampler.sample_states_value_quantile_balanced(3)
        before_columns, before_rows = sampler.get_sample_table()
        child = make_state("new_child", 99.0, parent_ids=("high",), timestep=2)
        sampler.update_states([child], [picked[0]], save=False)
        sampler.flush(step=1)
        after_columns, after_rows = sampler.get_sample_table()

        self.assertEqual(before_columns, after_columns)
        self.assertEqual(before_rows, after_rows)


class SampleBatchFlagTests(unittest.TestCase):
    def test_unsupported_sampler_api_raises_value_error(self) -> None:
        cfg = CodexNoFinetuneConfig(
            env_type=DummyEnv,
            groups_per_batch=1,
            group_size=1,
            value_quantile_balanced_sampling=True,
        )

        class UnsupportedSampler:
            def sample_states(self, num_states: int) -> list[State]:
                return []

        with self.assertRaisesRegex(ValueError, "sample_states_value_quantile_balanced"):
            asyncio.run(sample_batch(cfg, UnsupportedSampler(), 0))

    def test_flag_off_uses_ordinary_sampler(self) -> None:
        cfg = CodexNoFinetuneConfig(
            env_type=DummyEnv,
            groups_per_batch=2,
            group_size=1,
            value_quantile_balanced_sampling=False,
        )

        class OrdinarySampler:
            def __init__(self) -> None:
                self.called_with: int | None = None

            def sample_states(self, num_states: int) -> list[State]:
                self.called_with = num_states
                return []

            def sample_states_value_quantile_balanced(self, num_states: int) -> list[State]:
                raise AssertionError("balanced sampler should not be called")

        sampler = OrdinarySampler()
        asyncio.run(sample_batch(cfg, sampler, 0))

        self.assertEqual(sampler.called_with, 2)


if __name__ == "__main__":
    unittest.main()
````

## Diff Stat

### Worktree Tracked Diff Stat

````text
 ttt_discover/codex_utils/discovery.py |   4 +
 ttt_discover/codex_utils/sampler.py   | 410 +++++++++++++++++++++++++++++++---
 ttt_discover/discovery.py             |   4 +
 ttt_discover/rl/codex_no_finetune.py  |  16 +-
 4 files changed, 399 insertions(+), 35 deletions(-)
````
## Raw Diff

<details>
<summary>Tracked worktree patch</summary>

````diff
diff --git a/ttt_discover/codex_utils/discovery.py b/ttt_discover/codex_utils/discovery.py
index b21a80f..6594658 100644
--- a/ttt_discover/codex_utils/discovery.py
+++ b/ttt_discover/codex_utils/discovery.py
@@ -50,6 +50,7 @@ class DiscoverConfig:
     codex_initial_program_paths: tuple[str, ...] = ()
     codex_initial_pool_paths: tuple[str, ...] = ()
     codex_autonomous: bool = False
+    codex_value_quantile_balanced_sampling: bool = False
 
 
 def _run_codex_no_finetune(config: DiscoverConfig) -> None:
@@ -84,6 +85,9 @@ def _run_codex_no_finetune(config: DiscoverConfig) -> None:
         max_concurrent_requests=config.codex_max_concurrent_requests,
         initial_program_paths=config.codex_initial_program_paths,
         initial_pool_paths=config.codex_initial_pool_paths,
+        value_quantile_balanced_sampling=(
+            config.codex_value_quantile_balanced_sampling
+        ),
         autonomous=config.codex_autonomous,
         wandb_project=config.wandb_project,
         wandb_name=experiment_name,
diff --git a/ttt_discover/codex_utils/sampler.py b/ttt_discover/codex_utils/sampler.py
index 5c5d4d1..33df33c 100644
--- a/ttt_discover/codex_utils/sampler.py
+++ b/ttt_discover/codex_utils/sampler.py
@@ -4,6 +4,7 @@ from abc import ABC, abstractmethod
 from contextlib import contextmanager
 import json
 import logging
+from numbers import Real
 import os
 from pathlib import Path
 import threading
@@ -66,6 +67,27 @@ def _read_json_or_default(path: str, default: Any) -> Any:
         return default
 
 
+_INVALID_VALUE_BUCKET = "seed_or_invalid_value"
+_VALUE_QUANTILE_BUCKETS = (
+    "value_promising",
+    "value_strong",
+    "value_elite",
+)
+
+
+def _finite_numeric_value(value: Any) -> float | None:
+    if value is None or isinstance(value, (bool, np.bool_)):
+        return None
+    if isinstance(value, np.generic):
+        value = value.item()
+    if not isinstance(value, Real):
+        return None
+    value = float(value)
+    if not np.isfinite(value):
+        return None
+    return value
+
+
 class StateSampler(ABC):
     """Abstract base class for sampling states."""
 
@@ -375,6 +397,12 @@ class PUCTSampler(StateSampler):
         self._T: int = 0
         self._last_scale: float = 1.0
         self._last_puct_stats: list[tuple[int, float, float, float, float]] = []
+        self._last_value_quantile_balance_active: bool = False
+        self._last_value_bucket_keys: list[float | str] = []
+        self._last_sampled_values: list[float | None] = []
+        self._last_value_quantile_buckets: list[str] = []
+        self._last_value_quantile_balance_fallback_flags: list[bool] = []
+        self._last_value_quantile_balance_shortage_count: int = 0
         
         if resume_step is not None:
             self._load(resume_step)
@@ -460,6 +488,130 @@ class PUCTSampler(StateSampler):
         weights = (N - ranks).astype(np.float64)
         return weights / weights.sum()
 
+    def _ranked_puct_entries(
+        self,
+        *,
+        robust_values: bool = False,
+    ) -> tuple[list[tuple[float, float, State, int, float, float, float]], set[str]]:
+        initial_ids = {s.id for s in self._initial_states}
+        candidates = list(self._states)
+        if not candidates:
+            return [], initial_ids
+
+        if robust_values:
+            vals = np.array(
+                [
+                    value if (value := _finite_numeric_value(s.value)) is not None else float("-inf")
+                    for s in candidates
+                ],
+                dtype=np.float64,
+            )
+        else:
+            vals = np.array([float(s.value if s.value is not None else float("-inf")) for s in candidates])
+
+        non_initial_mask = np.array([s.id not in initial_ids for s in candidates])
+        if robust_values:
+            finite_mask = np.isfinite(vals)
+            scale_mask = non_initial_mask & finite_mask
+            if not scale_mask.any():
+                scale_mask = finite_mask if finite_mask.any() else None
+        else:
+            scale_mask = non_initial_mask if non_initial_mask.any() else None
+        scale = 1.0 if robust_values and not np.isfinite(vals).any() else self._compute_scale(vals, scale_mask)
+        self._last_scale = scale
+        P = self._compute_prior(vals, scale)
+        sqrtT = np.sqrt(1.0 + self._T)
+
+        scores = []
+        for i, s in enumerate(candidates):
+            n = self._n.get(s.id, 0)
+            raw_m = self._m.get(s.id, vals[i])
+            if robust_values:
+                m_value = _finite_numeric_value(raw_m)
+                m = m_value if m_value is not None else vals[i]
+            else:
+                m = raw_m
+            Q = m if n > 0 else vals[i]
+            bonus = self.puct_c * scale * P[i] * sqrtT / (1.0 + n)
+            score = Q + bonus
+            scores.append((score, vals[i], s, n, Q, P[i], bonus))
+
+        scores.sort(key=lambda x: (x[0], x[1]), reverse=True)
+        return scores, initial_ids
+
+    def _set_last_sampled(
+        self,
+        picked: list[State],
+        top_scores: list[tuple[float, float, State, int, float, float, float]],
+        *,
+        value_info: dict[str, tuple[float | str, float | None, str]] | None = None,
+        fallback_flags: list[bool] | None = None,
+        shortage_count: int = 0,
+        value_quantile_active: bool = False,
+    ) -> None:
+        state_id_to_idx = {s.id: i for i, s in enumerate(self._states)}
+        self._last_sampled_states = picked
+        self._last_sampled_indices = [state_id_to_idx.get(s.id, -1) for s in picked]
+        self._last_puct_stats = [(t[3], t[4], t[5], t[6], t[0]) for t in top_scores]
+
+        self._last_value_quantile_balance_active = value_quantile_active
+        if not value_quantile_active:
+            self._last_value_bucket_keys = []
+            self._last_sampled_values = []
+            self._last_value_quantile_buckets = []
+            self._last_value_quantile_balance_fallback_flags = []
+            self._last_value_quantile_balance_shortage_count = 0
+            return
+
+        if fallback_flags is None or len(fallback_flags) != len(picked):
+            fallback_flags = [False] * len(picked)
+        if value_info is None:
+            value_info = {}
+        bucket_keys: list[float | str] = []
+        sampled_values: list[float | None] = []
+        buckets: list[str] = []
+        for state in picked:
+            bucket_key, sampled_value, bucket = value_info.get(
+                state.id,
+                (_INVALID_VALUE_BUCKET, _finite_numeric_value(state.value), _INVALID_VALUE_BUCKET),
+            )
+            bucket_keys.append(bucket_key)
+            sampled_values.append(sampled_value)
+            buckets.append(bucket)
+        self._last_value_bucket_keys = bucket_keys
+        self._last_sampled_values = sampled_values
+        self._last_value_quantile_buckets = buckets
+        self._last_value_quantile_balance_fallback_flags = list(fallback_flags)
+        self._last_value_quantile_balance_shortage_count = int(shortage_count)
+
+    def _set_last_sampled_without_puct(
+        self,
+        picked: list[State],
+        *,
+        value_quantile_active: bool = False,
+    ) -> None:
+        self._last_sampled_states = picked
+        self._last_sampled_indices = []
+        self._last_puct_stats = [(0, 0.0, 0.0, 0.0, 0.0) for _ in picked]
+        self._last_value_quantile_balance_active = value_quantile_active
+        if value_quantile_active:
+            self._last_value_bucket_keys = [_INVALID_VALUE_BUCKET] * len(picked)
+            self._last_sampled_values = [_finite_numeric_value(s.value) for s in picked]
+            self._last_value_quantile_buckets = [_INVALID_VALUE_BUCKET] * len(picked)
+            self._last_value_quantile_balance_fallback_flags = [False] * len(picked)
+            self._last_value_quantile_balance_shortage_count = 0
+        else:
+            self._last_value_bucket_keys = []
+            self._last_sampled_values = []
+            self._last_value_quantile_buckets = []
+            self._last_value_quantile_balance_fallback_flags = []
+            self._last_value_quantile_balance_shortage_count = 0
+
+    def _refresh_sampled_initials(self, picked: list[State], initial_ids: set[str]) -> None:
+        for s in picked:
+            if s.id in initial_ids:
+                self._refresh_random_construction(s)
+
     def _get_lineage(self, state: State) -> set[str]:
         lineage = {state.id}
         for p in (state.parents or []):
@@ -489,38 +641,87 @@ class PUCTSampler(StateSampler):
                     queue.append(child_id)
         return lineage
 
+    def _is_value_quantile_eligible(
+        self,
+        state: State,
+        *,
+        initial_ids: set[str],
+        sampled_value: float | None,
+    ) -> bool:
+        if state.id in initial_ids:
+            return False
+        if not (state.parents or []):
+            return False
+        try:
+            if state.timestep < 0:
+                return False
+        except TypeError:
+            return False
+        return sampled_value is not None
+
+    def _value_quantile_info(
+        self,
+        initial_ids: set[str],
+    ) -> dict[str, tuple[float | str, float | None, str]]:
+        sampled_values: dict[str, float | None] = {}
+        eligible_state_ids: set[str] = set()
+        eligible_values: list[float] = []
+
+        for state in self._states:
+            sampled_value = _finite_numeric_value(state.value)
+            sampled_values[state.id] = sampled_value
+            if self._is_value_quantile_eligible(
+                state,
+                initial_ids=initial_ids,
+                sampled_value=sampled_value,
+            ):
+                eligible_state_ids.add(state.id)
+                eligible_values.append(sampled_value)
+
+        unique_values = sorted(set(eligible_values))
+        value_to_bucket: dict[float, str] = {}
+        if len(unique_values) == 1:
+            value_to_bucket[unique_values[0]] = "uniform_value"
+        elif len(unique_values) == 2:
+            value_to_bucket[unique_values[0]] = "value_promising"
+            value_to_bucket[unique_values[1]] = "value_elite"
+        elif unique_values:
+            low_end = (len(unique_values) + 2) // 3
+            strong_end = (2 * len(unique_values) + 2) // 3
+            for idx, value in enumerate(unique_values):
+                if idx < low_end:
+                    bucket = _VALUE_QUANTILE_BUCKETS[0]
+                elif idx < strong_end:
+                    bucket = _VALUE_QUANTILE_BUCKETS[1]
+                else:
+                    bucket = _VALUE_QUANTILE_BUCKETS[2]
+                value_to_bucket[value] = bucket
+
+        value_info: dict[str, tuple[float | str, float | None, str]] = {}
+        for state in self._states:
+            sampled_value = sampled_values.get(state.id)
+            if state.id in eligible_state_ids and sampled_value in value_to_bucket:
+                bucket = value_to_bucket[sampled_value]
+                value_info[state.id] = (sampled_value, sampled_value, bucket)
+            else:
+                value_info[state.id] = (
+                    _INVALID_VALUE_BUCKET,
+                    sampled_value,
+                    _INVALID_VALUE_BUCKET,
+                )
+        return value_info
+
     def sample_states(self, num_states: int) -> list[State]:
-        initial_ids = {s.id for s in self._initial_states}
-        candidates = list(self._states)
+        scores, initial_ids = self._ranked_puct_entries()
 
-        if not candidates:
+        if not scores and not self._states:
             picked = [
                 create_initial_state(self.env_type, self.problem_type)
                 for _ in range(num_states)
             ]
-            self._last_sampled_states = picked
-            self._last_sampled_indices = []
-            self._last_puct_stats = [(0, 0.0, 0.0, 0.0, 0.0) for _ in picked]
+            self._set_last_sampled_without_puct(picked)
             return picked
 
-        vals = np.array([float(s.value if s.value is not None else float("-inf")) for s in candidates])
-        non_initial_mask = np.array([s.id not in initial_ids for s in candidates])
-        scale = self._compute_scale(vals, non_initial_mask if non_initial_mask.any() else None)
-        self._last_scale = scale
-        P = self._compute_prior(vals, scale)
-        sqrtT = np.sqrt(1.0 + self._T)
-
-        scores = []
-        for i, s in enumerate(candidates):
-            n = self._n.get(s.id, 0)
-            m = self._m.get(s.id, vals[i])
-            Q = m if n > 0 else vals[i]
-            bonus = self.puct_c * scale * P[i] * sqrtT / (1.0 + n)
-            score = Q + bonus
-            scores.append((score, vals[i], s, n, Q, P[i], bonus))
-
-        scores.sort(key=lambda x: (x[0], x[1]), reverse=True)
-
         if num_states > 1:
             children_map = self._build_children_map()
             picked, top_scores, blocked_ids = [], [], set()
@@ -537,14 +738,90 @@ class PUCTSampler(StateSampler):
             top_scores = scores[:num_states]
             picked = [t[2] for t in top_scores]
 
-        state_id_to_idx = {s.id: i for i, s in enumerate(self._states)}
-        self._last_sampled_states = picked
-        self._last_sampled_indices = [state_id_to_idx.get(s.id, -1) for s in picked]
-        self._last_puct_stats = [(t[3], t[4], t[5], t[6], t[0]) for t in top_scores]
+        self._set_last_sampled(picked, top_scores)
+        self._refresh_sampled_initials(picked, initial_ids)
 
-        for s in picked:
-            if s.id in initial_ids:
-                self._refresh_random_construction(s)
+        return picked
+
+    def sample_states_value_quantile_balanced(self, num_states: int) -> list[State]:
+        """Sample by PUCT rank while preferring distinct retained value buckets."""
+        scores, initial_ids = self._ranked_puct_entries(robust_values=True)
+        value_info = self._value_quantile_info(initial_ids)
+
+        if not scores and not self._states:
+            picked = [
+                create_initial_state(self.env_type, self.problem_type)
+                for _ in range(num_states)
+            ]
+            self._set_last_sampled_without_puct(
+                picked,
+                value_quantile_active=True,
+            )
+            return picked
+
+        if num_states <= 1:
+            top_scores = scores[:num_states]
+            picked = [t[2] for t in top_scores]
+            self._set_last_sampled(
+                picked,
+                top_scores,
+                value_info=value_info,
+                value_quantile_active=True,
+            )
+            self._refresh_sampled_initials(picked, initial_ids)
+            return picked
+
+        children_map = self._build_children_map()
+        picked: list[State] = []
+        top_scores: list[tuple[float, float, State, int, float, float, float]] = []
+        fallback_flags: list[bool] = []
+        blocked_ids: set[str] = set()
+        picked_ids: set[str] = set()
+        picked_buckets: set[str] = set()
+
+        for entry in scores:
+            if len(picked) >= num_states:
+                break
+            state = entry[2]
+            bucket = value_info.get(
+                state.id,
+                (_INVALID_VALUE_BUCKET, None, _INVALID_VALUE_BUCKET),
+            )[2]
+            if (
+                state.id in blocked_ids
+                or bucket == _INVALID_VALUE_BUCKET
+                or bucket in picked_buckets
+            ):
+                continue
+            picked.append(state)
+            top_scores.append(entry)
+            fallback_flags.append(False)
+            picked_ids.add(state.id)
+            picked_buckets.add(bucket)
+            blocked_ids.update(self._get_full_lineage(state, children_map))
+
+        for entry in scores:
+            if len(picked) >= num_states:
+                break
+            state = entry[2]
+            if state.id in blocked_ids or state.id in picked_ids:
+                continue
+            picked.append(state)
+            top_scores.append(entry)
+            fallback_flags.append(True)
+            picked_ids.add(state.id)
+            blocked_ids.update(self._get_full_lineage(state, children_map))
+
+        shortage_count = max(0, int(num_states) - len(picked))
+        self._set_last_sampled(
+            picked,
+            top_scores,
+            value_info=value_info,
+            fallback_flags=fallback_flags,
+            shortage_count=shortage_count,
+            value_quantile_active=True,
+        )
+        self._refresh_sampled_initials(picked, initial_ids)
 
         return picked
 
@@ -704,7 +981,16 @@ class PUCTSampler(StateSampler):
 
     def get_sample_stats(self) -> dict:
         def _stats(values, prefix):
-            arr = np.array([v for v in values if v is not None])
+            if self._last_value_quantile_balance_active:
+                arr = np.array(
+                    [
+                        numeric_value
+                        for v in values
+                        if (numeric_value := _finite_numeric_value(v)) is not None
+                    ]
+                )
+            else:
+                arr = np.array([v for v in values if v is not None])
             if len(arr) == 0:
                 return {}
             return {
@@ -731,21 +1017,77 @@ class PUCTSampler(StateSampler):
         stats.update(_stats(sampled_values, "puct/sampled_value"))
         stats.update(_stats(sampled_timesteps, "puct/sampled_timestep"))
         stats.update(_stats(sampled_constr_lens, "puct/sampled_construction_len"))
+        if self._last_value_quantile_balance_active:
+            diversity_buckets = {
+                bucket
+                for bucket in self._last_value_quantile_buckets
+                if bucket != _INVALID_VALUE_BUCKET
+            }
+            fallback_flags = (
+                self._last_value_quantile_balance_fallback_flags
+                if len(self._last_value_quantile_balance_fallback_flags) == len(self._last_sampled_states)
+                else []
+            )
+            stats.update(
+                {
+                    "puct/value_quantile_bucket_unique_count": len(diversity_buckets),
+                    "puct/value_quantile_balance_fallback_count": int(sum(fallback_flags)),
+                    "puct/value_quantile_balance_shortage_count": self._last_value_quantile_balance_shortage_count,
+                }
+            )
         return stats
 
     def get_sample_table(self) -> tuple[list[str], list[tuple]]:
         columns = ["buffer_idx", "timestep", "value", "terminal_value", "parent_value", "construction_len", "observation_len", "n", "Q", "P", "bonus", "score"]
+        if self._last_value_quantile_balance_active:
+            columns = columns + [
+                "value_bucket_key",
+                "sampled_value",
+                "value_quantile_bucket",
+                "value_quantile_balance_fallback",
+            ]
         rows = []
         if not self._last_sampled_states:
             return columns, rows
         indices = self._last_sampled_indices if len(self._last_sampled_indices) == len(self._last_sampled_states) else [-1] * len(self._last_sampled_states)
         stats = self._last_puct_stats if len(self._last_puct_stats) == len(self._last_sampled_states) else [(0, 0.0, 0.0, 0.0, 0.0)] * len(self._last_sampled_states)
-        for idx, state, (n, Q, P, bonus, score) in zip(indices, self._last_sampled_states, stats):
+        value_bucket_keys = (
+            self._last_value_bucket_keys
+            if len(self._last_value_bucket_keys) == len(self._last_sampled_states)
+            else [_INVALID_VALUE_BUCKET] * len(self._last_sampled_states)
+        )
+        sampled_values = (
+            self._last_sampled_values
+            if len(self._last_sampled_values) == len(self._last_sampled_states)
+            else [None] * len(self._last_sampled_states)
+        )
+        value_buckets = (
+            self._last_value_quantile_buckets
+            if len(self._last_value_quantile_buckets) == len(self._last_sampled_states)
+            else [_INVALID_VALUE_BUCKET] * len(self._last_sampled_states)
+        )
+        fallback_flags = (
+            self._last_value_quantile_balance_fallback_flags
+            if len(self._last_value_quantile_balance_fallback_flags) == len(self._last_sampled_states)
+            else [False] * len(self._last_sampled_states)
+        )
+        for idx, state, (n, Q, P, bonus, score), bucket_key, sampled_value, value_bucket, fallback in zip(
+            indices,
+            self._last_sampled_states,
+            stats,
+            value_bucket_keys,
+            sampled_values,
+            value_buckets,
+            fallback_flags,
+        ):
             parent_val = state.parent_values[0] if state.parent_values else None
             constr = getattr(state, 'construction', None)
             constr_len = len(constr) if constr is not None else 0
             obs_len = len(state.observation) if state.observation else 0
-            rows.append((idx, state.timestep, state.value, 0, parent_val, constr_len, obs_len, n, Q, P, bonus, score))
+            row = (idx, state.timestep, state.value, 0, parent_val, constr_len, obs_len, n, Q, P, bonus, score)
+            if self._last_value_quantile_balance_active:
+                row = row + (bucket_key, sampled_value, value_bucket, fallback)
+            rows.append(row)
         return columns, rows
 
 
diff --git a/ttt_discover/discovery.py b/ttt_discover/discovery.py
index 6cdb434..365a0c9 100644
--- a/ttt_discover/discovery.py
+++ b/ttt_discover/discovery.py
@@ -63,6 +63,7 @@ class DiscoverConfig:
     codex_initial_program_paths: tuple[str, ...] = ()
     codex_initial_pool_paths: tuple[str, ...] = ()
     codex_autonomous: bool = False
+    codex_value_quantile_balanced_sampling: bool = False
 
 
 def init_ray(num_cpus_per_task: int, env_type: str):
@@ -145,6 +146,9 @@ async def discover_impl(config: DiscoverConfig):
             max_concurrent_requests=config.codex_max_concurrent_requests,
             initial_program_paths=config.codex_initial_program_paths,
             initial_pool_paths=config.codex_initial_pool_paths,
+            value_quantile_balanced_sampling=(
+                config.codex_value_quantile_balanced_sampling
+            ),
             autonomous=config.codex_autonomous,
             wandb_project=config.wandb_project,
             wandb_name=config.experiment_name,
diff --git a/ttt_discover/rl/codex_no_finetune.py b/ttt_discover/rl/codex_no_finetune.py
index 18a0e9e..563b72e 100644
--- a/ttt_discover/rl/codex_no_finetune.py
+++ b/ttt_discover/rl/codex_no_finetune.py
@@ -137,6 +137,7 @@ class CodexNoFinetuneConfig:
     initial_program_paths: tuple[str, ...] = ()
     initial_pool_paths: tuple[str, ...] = ()
     topk_children: int = 16
+    value_quantile_balanced_sampling: bool = False
 
     # False is TTT Discover: non-auto, many short samples. True is AutoEvolve:
     # one Codex deep-dive workspace per sample with file edits and shell access.
@@ -820,7 +821,20 @@ async def sample_batch(
     i_batch: int,
 ) -> tuple[list[CandidateResult], dict[str, Any], list[CandidateResult]]:
     metrics: dict[str, Any] = {}
-    parent_states = sampler.sample_states(cfg.groups_per_batch)
+    if cfg.value_quantile_balanced_sampling:
+        sample_value_quantile_balanced = getattr(
+            sampler,
+            "sample_states_value_quantile_balanced",
+            None,
+        )
+        if not callable(sample_value_quantile_balanced):
+            raise ValueError(
+                "value_quantile_balanced_sampling=True requires sampler public API "
+                "sample_states_value_quantile_balanced(num_states)"
+            )
+        parent_states = sample_value_quantile_balanced(cfg.groups_per_batch)
+    else:
+        parent_states = sampler.sample_states(cfg.groups_per_batch)
     semaphore = (
         asyncio.Semaphore(cfg.max_concurrent_requests)
         if cfg.max_concurrent_requests is not None
````
</details>

