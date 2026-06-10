# codex/diversity-shape-profile-balance

## Summary

对 construction 的嵌套容器/ndarray/标量结构生成 shape profile，过滤退化 profile 后按结构形态平衡父选择。

## Branch State

- Worktree: `/opt/tiger/discover-shape-profile-balance`
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

- `codex_shape_profile_balanced_sampling`
- `depth`
- `seen`
- `hist`
- `shape_profile_balanced_sampling`

### Constants

- `_SHAPE_PROFILE_MAX_DEPTH`
- `_SHAPE_PROFILE_MAX_VISITED`
- `_SHAPE_PROFILE_KINDS`
- `_SHAPE_PROFILE_CONTAINER_KINDS`
- `_SHAPE_PROFILE_SCALAR_KINDS`

### Classes

- None

### Functions

- `_shape_count_bucket`
- `_shape_depth_bucket`
- `_shape_ndim_bucket`
- `_shape_kind`
- `_shape_len`
- `_shape_container_values`
- `_shape_dtype_kind`
- `_shape_rank_bucket`
- `_shape_structural_sort_key`
- `_shape_is_single_scalar`
- `construction_shape_profile_v1`
- `record_truncated_children`
- `visit`
- `is_degenerate_shape_profile`
- `sample_states_shape_profile_balanced`
- `_sample_parent_states`

## Diff Summary

- Worktree tracked shortstat: `4 files changed, 481 insertions(+), 1 deletion(-)`
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

- `repro/gpu_mode/run_0608_shape_profile_balance.sh (1150 bytes)`
- `repro/run_discovery.py (15442 bytes)`
- `tests/test_shape_profile_balanced_sampling.py (12422 bytes)`

### Detected Test Functions

- `tests/test_shape_profile_balanced_sampling.py::test_enabled_flag_requires_balanced_sampler_api`
- `tests/test_shape_profile_balanced_sampling.py::test_flag_off_matches_baseline_ids_stats_and_table_schema`
- `tests/test_shape_profile_balanced_sampling.py::test_num_states_one_uses_baseline`
- `tests/test_shape_profile_balanced_sampling.py::test_first_pass_selects_lower_puct_distinct_profile`
- `tests/test_shape_profile_balanced_sampling.py::test_fallback_fills_by_puct_when_profiles_run_out`
- `tests/test_shape_profile_balanced_sampling.py::test_degenerate_constructions_skip_first_pass_but_fill_fallback`
- `tests/test_shape_profile_balanced_sampling.py::test_full_lineage_blocking_in_first_pass_and_fallback`
- `tests/test_shape_profile_balanced_sampling.py::test_blocked_candidate_does_not_mark_profile_seen`
- `tests/test_shape_profile_balanced_sampling.py::test_dict_and_set_order_do_not_affect_profile`
- `tests/test_shape_profile_balanced_sampling.py::test_raw_values_keys_and_object_ids_do_not_change_profile`
- `tests/test_shape_profile_balanced_sampling.py::test_depth_fanout_count_and_ndarray_buckets`
- `tests/test_shape_profile_balanced_sampling.py::test_nested_cyclic_and_large_constructions_are_stable`
- `tests/test_shape_profile_balanced_sampling.py::test_persistence_schema_has_no_shape_profile_metadata`

## Untracked File Snapshots

These files are not part of `git diff` yet, but they are part of the worktree implementation state recorded here.

### `repro/gpu_mode/run_0608_shape_profile_balance.sh`

````bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-gpu-mode-trimul-shape-profile-balance-0608}"
LOG_ROOT="${LOG_ROOT:-tinker_log}"
NUM_EPOCHS="${NUM_EPOCHS:-50}"
GROUP_SIZE="${GROUP_SIZE:-1}"
GROUPS_PER_BATCH="${GROUPS_PER_BATCH:-4}"
EVAL_TIMEOUT="${EVAL_TIMEOUT:-1200}"
WANDB_PROJECT="${WANDB_PROJECT:-}"
CODEX_MAX_CONCURRENT_REQUESTS="${CODEX_MAX_CONCURRENT_REQUESTS:-4}"

args=(
  --task trimul
  --runner codex_no_finetune
  --experiment-name "${EXPERIMENT_NAME}"
  --log-root "${LOG_ROOT}"
  --num-epochs "${NUM_EPOCHS}"
  --group-size "${GROUP_SIZE}"
  --groups-per-batch "${GROUPS_PER_BATCH}"
  --eval-timeout "${EVAL_TIMEOUT}"
  --wandb-project "${WANDB_PROJECT}"
  --codex-max-concurrent-requests "${CODEX_MAX_CONCURRENT_REQUESTS}"
  --codex-shape-profile-balanced-sampling
)

if [[ -n "${GPU:-}" ]]; then
  args+=(--gpu "${GPU}")
fi

if [[ -n "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
  args+=(--torch-cuda-arch-list "${TORCH_CUDA_ARCH_LIST}")
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
        if dimension != 3:
            raise ValueError("The Kakeya environment currently supports only d=3")
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
    parser.add_argument("--problem-type", default=None)
    parser.add_argument("--dimension", type=int, default=None)
    parser.add_argument("--primes", default=None)
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
        "--codex-shape-profile-balanced-sampling",
        action="store_true",
        help=(
            "Use construction shape profile balanced PUCT parent sampling."
        ),
    )

    parser.add_argument("--gpu", default=None)
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
            "codex_shape_profile_balanced_sampling="
            f"{args.codex_shape_profile_balanced_sampling}"
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
            shape_profile_balanced_sampling=(
                args.codex_shape_profile_balanced_sampling
            ),
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
        codex_shape_profile_balanced_sampling=(
            args.codex_shape_profile_balanced_sampling
        ),
        codex_autonomous=args.codex_autonomous,
        remove_constant_reward_groups=remove_constant_reward_groups,
    )
    discover(config)


if __name__ == "__main__":
    main()
````

### `tests/test_shape_profile_balanced_sampling.py`

````python
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ttt_discover.codex_utils.runtime import State
from ttt_discover.codex_utils.sampler import (
    PUCTSampler,
    _sampler_file_for_step,
    construction_shape_profile_v1,
    is_degenerate_shape_profile,
)
from ttt_discover.rl.codex_no_finetune import _sample_parent_states


_DEFAULT_CONSTRUCTION = object()


class DummyEnv:
    state_type = State

    @staticmethod
    def create_initial_state(problem_type: str) -> State:
        return State(
            timestep=0,
            construction=["initial", "state"],
            code="",
            value=0.0,
            id=f"initial-{problem_type or 'default'}",
        )


def make_state(
    state_id: str,
    value: float,
    construction=_DEFAULT_CONSTRUCTION,
    *,
    parents: list[dict] | None = None,
    code: str = "",
) -> State:
    if construction is _DEFAULT_CONSTRUCTION:
        construction = [1, 2]
    return State(
        timestep=1,
        construction=construction,
        code=code,
        value=value,
        id=state_id,
        parents=parents or [],
    )


def make_sampler(tmpdir: str, states: list[State]) -> PUCTSampler:
    sampler = PUCTSampler(
        file_path=str(Path(tmpdir) / "puct_sampler.json"),
        env_type=DummyEnv,
        problem_type="test",
        batch_size=1,
        puct_c=0.0,
        topk_children=0,
    )
    sampler._states = list(states)
    sampler._initial_states = []
    sampler._n = {}
    sampler._m = {}
    sampler._T = 0
    sampler._last_sampled_states = []
    sampler._last_sampled_indices = []
    sampler._last_puct_stats = []
    return sampler


def profile_obj(value) -> dict:
    profile = construction_shape_profile_v1(value)
    if profile is None:
        raise AssertionError("expected non-degenerate shape profile")
    return json.loads(profile)


class ShapeProfileBalancedSamplingTest(unittest.TestCase):
    def test_enabled_flag_requires_balanced_sampler_api(self):
        cfg = SimpleNamespace(
            groups_per_batch=2,
            shape_profile_balanced_sampling=True,
        )
        sampler = SimpleNamespace(sample_states=lambda num_states: [])

        with self.assertRaisesRegex(
            ValueError,
            "sample_states_shape_profile_balanced",
        ):
            _sample_parent_states(cfg, sampler)

        sampler = SimpleNamespace(
            sample_states=lambda num_states: [],
            sample_states_shape_profile_balanced=None,
        )
        with self.assertRaisesRegex(
            ValueError,
            "sample_states_shape_profile_balanced",
        ):
            _sample_parent_states(cfg, sampler)

    def test_flag_off_matches_baseline_ids_stats_and_table_schema(self):
        states = [
            make_state("s0", 10.0, [1, 2]),
            make_state("s1", 9.0, [3, 4]),
            make_state("s2", 8.0, [5, [6, 7]]),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            baseline = make_sampler(tmpdir, states)
            expected = baseline.sample_states(3)
            expected_stats = baseline.get_sample_stats()
            expected_columns, _ = baseline.get_sample_table()

            off_sampler = make_sampler(tmpdir, states)
            cfg = SimpleNamespace(
                groups_per_batch=3,
                shape_profile_balanced_sampling=False,
            )
            actual = _sample_parent_states(cfg, off_sampler)
            actual_stats = off_sampler.get_sample_stats()
            actual_columns, rows = off_sampler.get_sample_table()

        self.assertEqual([s.id for s in actual], [s.id for s in expected])
        self.assertEqual(actual_stats, expected_stats)
        self.assertEqual(actual_columns, expected_columns)
        self.assertEqual(
            actual_columns,
            [
                "buffer_idx",
                "timestep",
                "value",
                "terminal_value",
                "parent_value",
                "construction_len",
                "observation_len",
                "n",
                "Q",
                "P",
                "bonus",
                "score",
            ],
        )
        self.assertEqual(len(rows), 3)

    def test_num_states_one_uses_baseline(self):
        states = [
            make_state("degenerate_high", 100.0, [1]),
            make_state("valid_low", 1.0, [1, 2]),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            sampler = make_sampler(tmpdir, states)
            picked = sampler.sample_states_shape_profile_balanced(1)

        self.assertEqual([s.id for s in picked], ["degenerate_high"])

    def test_first_pass_selects_lower_puct_distinct_profile(self):
        states = [
            make_state("same_top", 100.0, [1, 2]),
            make_state("same_next", 99.0, [3, 4]),
            make_state("different_lower", 10.0, [1, [2, 3]]),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            sampler = make_sampler(tmpdir, states)
            picked = sampler.sample_states_shape_profile_balanced(2)

        self.assertEqual([s.id for s in picked], ["same_top", "different_lower"])

    def test_fallback_fills_by_puct_when_profiles_run_out(self):
        states = [
            make_state("same_top", 100.0, [1, 2]),
            make_state("same_next", 99.0, [3, 4]),
            make_state("different", 98.0, [1, [2, 3]]),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            sampler = make_sampler(tmpdir, states)
            picked = sampler.sample_states_shape_profile_balanced(3)

        self.assertEqual([s.id for s in picked], ["same_top", "different", "same_next"])

    def test_degenerate_constructions_skip_first_pass_but_fill_fallback(self):
        missing = make_state("missing", 100.0, [1, 2])
        delattr(missing, "construction")
        states = [
            missing,
            make_state("none", 99.0, None),
            make_state("empty", 98.0, []),
            make_state("single_scalar", 97.0, [1]),
            make_state("object_only", 96.0, [object(), object()]),
            make_state("valid", 1.0, [1, 2]),
        ]
        self.assertIsNone(construction_shape_profile_v1(None))
        self.assertIsNone(construction_shape_profile_v1([]))
        self.assertIsNone(construction_shape_profile_v1([1]))
        self.assertIsNone(construction_shape_profile_v1([object(), object()]))

        with tempfile.TemporaryDirectory() as tmpdir:
            sampler = make_sampler(tmpdir, states)
            picked = sampler.sample_states_shape_profile_balanced(3)

        self.assertEqual([s.id for s in picked], ["valid", "missing", "none"])

    def test_full_lineage_blocking_in_first_pass_and_fallback(self):
        states = [
            make_state("parent", 100.0, [1, 2]),
            make_state("child", 99.0, [1, [2, 3]], parents=[{"id": "parent"}]),
            make_state("degenerate", 98.0, [1]),
            make_state("other", 97.0, [[1], [2]]),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            sampler = make_sampler(tmpdir, states)
            picked = sampler.sample_states_shape_profile_balanced(3)

        self.assertEqual([s.id for s in picked], ["parent", "other", "degenerate"])

    def test_blocked_candidate_does_not_mark_profile_seen(self):
        states = [
            make_state("parent", 100.0, [1, 2]),
            make_state("blocked", 99.0, [1, [2, 3]], parents=[{"id": "parent"}]),
            make_state("same_profile_available", 98.0, [4, [5, 6]]),
            make_state("other", 97.0, [[1], [2]]),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            sampler = make_sampler(tmpdir, states)
            picked = sampler.sample_states_shape_profile_balanced(2)

        self.assertEqual([s.id for s in picked], ["parent", "same_profile_available"])

    def test_dict_and_set_order_do_not_affect_profile(self):
        dict_a = {"first": [1, 2], "second": {"left": [3, 4]}}
        dict_b = {}
        dict_b["renamed_second"] = {"right": [9, 8]}
        dict_b["renamed_first"] = [7, 6]
        self.assertEqual(
            construction_shape_profile_v1(dict_a),
            construction_shape_profile_v1(dict_b),
        )

        set_a = set()
        set_a.add((1, "a"))
        set_a.add((2, "b"))
        set_b = set()
        set_b.add((9, "z"))
        set_b.add((8, "y"))
        self.assertEqual(
            construction_shape_profile_v1(set_a),
            construction_shape_profile_v1(set_b),
        )

    def test_raw_values_keys_and_object_ids_do_not_change_profile(self):
        self.assertEqual(
            construction_shape_profile_v1([1, 2, "alpha"]),
            construction_shape_profile_v1([999, -5, "omega"]),
        )
        self.assertEqual(
            construction_shape_profile_v1({"left": [1, 2], "right": [3, 4]}),
            construction_shape_profile_v1({"x": [9, 8], "y": [7, 6]}),
        )
        self.assertEqual(
            construction_shape_profile_v1([object(), [1, 2]]),
            construction_shape_profile_v1([object(), [9, 8]]),
        )

    def test_depth_fanout_count_and_ndarray_buckets(self):
        one = profile_obj([[1]])
        self.assertEqual(one["max_fanout"], "1")
        self.assertEqual(one["max_depth"], "2")
        self.assertEqual(one["counts"]["number"], "1")

        self.assertEqual(profile_obj([1, 2])["max_fanout"], "2-3")
        self.assertEqual(profile_obj(list(range(4)))["max_fanout"], "4-7")
        self.assertEqual(profile_obj(list(range(8)))["max_fanout"], "8-15")
        self.assertEqual(profile_obj(list(range(16)))["max_fanout"], "16-31")
        self.assertEqual(profile_obj(list(range(32)))["max_fanout"], "32+")

        deep = profile_obj([[[[1, 2]]]])
        self.assertEqual(deep["max_depth"], "3+")
        self.assertTrue(deep["flags"]["truncated"])

        vector = profile_obj(np.ones((2,)))
        matrix = profile_obj(np.ones((2, 2)))
        tensor = profile_obj(np.ones((2, 2, 2)))
        high = profile_obj(np.ones((2, 2, 2, 2)))
        self.assertEqual(vector["ndarray"]["ndim"], ["1"])
        self.assertEqual(vector["ndarray"]["size"], ["2-3"])
        self.assertEqual(vector["ndarray"]["rank"], ["vector"])
        self.assertEqual(matrix["ndarray"]["ndim"], ["2"])
        self.assertEqual(matrix["ndarray"]["rank"], ["matrix"])
        self.assertEqual(tensor["ndarray"]["ndim"], ["3"])
        self.assertEqual(tensor["ndarray"]["rank"], ["tensor"])
        self.assertEqual(high["ndarray"]["ndim"], ["4+"])
        self.assertEqual(high["ndarray"]["rank"], ["high"])

    def test_nested_cyclic_and_large_constructions_are_stable(self):
        cyclic = []
        cyclic.append(cyclic)
        profile = construction_shape_profile_v1(cyclic)
        self.assertFalse(is_degenerate_shape_profile(profile))
        parsed = json.loads(profile)
        self.assertTrue(parsed["flags"]["cycle"])

        large = [list(range(1000)), {"ignored_key": list(range(1000))}]
        first = construction_shape_profile_v1(large)
        second = construction_shape_profile_v1(large)
        self.assertEqual(first, second)
        parsed = json.loads(first)
        self.assertTrue(parsed["flags"]["truncated"])
        self.assertEqual(parsed["max_fanout"], "32+")

    def test_persistence_schema_has_no_shape_profile_metadata(self):
        states = [
            make_state("a", 2.0, [1, 2]),
            make_state("b", 1.0, [1, [2, 3]]),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            sampler = make_sampler(tmpdir, states)
            sampler.sample_states_shape_profile_balanced(2)
            sampler.flush(step=3)
            store_path = _sampler_file_for_step(sampler.file_path, 3)
            with open(store_path, "r", encoding="utf-8") as f:
                store = json.load(f)
            columns, _ = sampler.get_sample_table()

        self.assertEqual(
            set(store),
            {"step", "states", "initial_states", "puct_n", "puct_m", "puct_T"},
        )
        self.assertFalse(any("shape" in key or "profile" in key for key in store))
        self.assertFalse(
            any("shape" in column or "profile" in column for column in columns)
        )


if __name__ == "__main__":
    unittest.main()
````

## Diff Stat

### Worktree Tracked Diff Stat

````text
 ttt_discover/codex_utils/discovery.py |   4 +
 ttt_discover/codex_utils/sampler.py   | 456 ++++++++++++++++++++++++++++++++++
 ttt_discover/discovery.py             |   4 +
 ttt_discover/rl/codex_no_finetune.py  |  18 +-
 4 files changed, 481 insertions(+), 1 deletion(-)
````
## Raw Diff

<details>
<summary>Tracked worktree patch</summary>

````diff
diff --git a/ttt_discover/codex_utils/discovery.py b/ttt_discover/codex_utils/discovery.py
index b21a80f..3b947b4 100644
--- a/ttt_discover/codex_utils/discovery.py
+++ b/ttt_discover/codex_utils/discovery.py
@@ -50,6 +50,7 @@ class DiscoverConfig:
     codex_initial_program_paths: tuple[str, ...] = ()
     codex_initial_pool_paths: tuple[str, ...] = ()
     codex_autonomous: bool = False
+    codex_shape_profile_balanced_sampling: bool = False
 
 
 def _run_codex_no_finetune(config: DiscoverConfig) -> None:
@@ -84,6 +85,9 @@ def _run_codex_no_finetune(config: DiscoverConfig) -> None:
         max_concurrent_requests=config.codex_max_concurrent_requests,
         initial_program_paths=config.codex_initial_program_paths,
         initial_pool_paths=config.codex_initial_pool_paths,
+        shape_profile_balanced_sampling=(
+            config.codex_shape_profile_balanced_sampling
+        ),
         autonomous=config.codex_autonomous,
         wandb_project=config.wandb_project,
         wandb_name=experiment_name,
diff --git a/ttt_discover/codex_utils/sampler.py b/ttt_discover/codex_utils/sampler.py
index 5c5d4d1..5acedd0 100644
--- a/ttt_discover/codex_utils/sampler.py
+++ b/ttt_discover/codex_utils/sampler.py
@@ -66,6 +66,385 @@ def _read_json_or_default(path: str, default: Any) -> Any:
         return default
 
 
+_SHAPE_PROFILE_MAX_DEPTH = 3
+_SHAPE_PROFILE_MAX_VISITED = 64
+_SHAPE_PROFILE_KINDS = (
+    "none",
+    "scalar",
+    "string",
+    "number",
+    "bool",
+    "list",
+    "tuple",
+    "set",
+    "dict",
+    "ndarray",
+    "object",
+    "unknown",
+)
+_SHAPE_PROFILE_CONTAINER_KINDS = {"list", "tuple", "set", "dict"}
+_SHAPE_PROFILE_SCALAR_KINDS = {"none", "string", "number", "bool"}
+
+
+def _shape_count_bucket(n: int) -> str:
+    if n <= 0:
+        return "0"
+    if n == 1:
+        return "1"
+    if n <= 3:
+        return "2-3"
+    if n <= 7:
+        return "4-7"
+    if n <= 15:
+        return "8-15"
+    if n <= 31:
+        return "16-31"
+    return "32+"
+
+
+def _shape_depth_bucket(depth: int) -> str:
+    if depth <= 0:
+        return "0"
+    if depth == 1:
+        return "1"
+    if depth == 2:
+        return "2"
+    return "3+"
+
+
+def _shape_ndim_bucket(ndim: int) -> str:
+    if ndim <= 0:
+        return "0"
+    if ndim == 1:
+        return "1"
+    if ndim == 2:
+        return "2"
+    if ndim == 3:
+        return "3"
+    return "4+"
+
+
+def _shape_kind(value: Any) -> str:
+    try:
+        if value is None:
+            return "none"
+        if isinstance(value, bool) or isinstance(value, np.bool_):
+            return "bool"
+        if isinstance(value, str):
+            return "string"
+        if isinstance(value, (int, float, complex, np.number)):
+            return "number"
+        if isinstance(value, np.ndarray):
+            return "ndarray"
+        if isinstance(value, list):
+            return "list"
+        if isinstance(value, tuple):
+            return "tuple"
+        if isinstance(value, (set, frozenset)):
+            return "set"
+        if isinstance(value, dict):
+            return "dict"
+        return "object"
+    except Exception:
+        return "unknown"
+
+
+def _shape_len(value: Any) -> int | None:
+    try:
+        return len(value)
+    except Exception:
+        return None
+
+
+def _shape_container_values(value: Any, kind: str) -> list[Any]:
+    if kind == "dict":
+        return list(value.values())
+    if kind in {"list", "tuple", "set"}:
+        return list(value)
+    return []
+
+
+def _shape_dtype_kind(value: np.ndarray) -> str:
+    try:
+        dtype_kind = value.dtype.kind
+    except Exception:
+        return "unknown"
+    if dtype_kind in {"b"}:
+        return "bool"
+    if dtype_kind in {"i", "u", "f", "c"}:
+        return "number"
+    if dtype_kind in {"U", "S"}:
+        return "string"
+    if dtype_kind in {"O"}:
+        return "object"
+    return "unknown"
+
+
+def _shape_rank_bucket(value: np.ndarray) -> str:
+    try:
+        ndim = int(value.ndim)
+    except Exception:
+        return "unknown"
+    if ndim <= 0:
+        return "scalar"
+    if ndim == 1:
+        return "vector"
+    if ndim == 2:
+        return "matrix"
+    if ndim == 3:
+        return "tensor"
+    return "high"
+
+
+def _shape_structural_sort_key(
+    value: Any,
+    *,
+    depth: int = 0,
+    seen: set[int] | None = None,
+) -> tuple:
+    if seen is None:
+        seen = set()
+    kind = _shape_kind(value)
+    if kind == "ndarray":
+        try:
+            return (
+                kind,
+                _shape_ndim_bucket(int(value.ndim)),
+                _shape_count_bucket(int(value.size)),
+                _shape_dtype_kind(value),
+                _shape_rank_bucket(value),
+            )
+        except Exception:
+            return (kind, "unknown")
+    if kind not in _SHAPE_PROFILE_CONTAINER_KINDS or depth >= 2:
+        return (kind,)
+    oid = id(value)
+    if oid in seen:
+        return (kind, "cycle")
+    seen.add(oid)
+    length = _shape_len(value)
+    children = _shape_container_values(value, kind)
+    hist: dict[str, int] = {}
+    for child in children[:_SHAPE_PROFILE_MAX_VISITED]:
+        child_kind = _shape_kind(child)
+        hist[child_kind] = hist.get(child_kind, 0) + 1
+    child_hist = tuple(
+        (child_kind, _shape_count_bucket(count))
+        for child_kind, count in sorted(hist.items())
+    )
+    seen.remove(oid)
+    return (kind, _shape_count_bucket(length or 0), child_hist)
+
+
+def _shape_is_single_scalar(value: Any) -> bool:
+    kind = _shape_kind(value)
+    if kind in _SHAPE_PROFILE_SCALAR_KINDS:
+        return True
+    if kind == "ndarray":
+        try:
+            return int(value.size) <= 1
+        except Exception:
+            return True
+    if kind not in _SHAPE_PROFILE_CONTAINER_KINDS:
+        return False
+    length = _shape_len(value)
+    if length != 1:
+        return False
+    values = _shape_container_values(value, kind)
+    if len(values) != 1:
+        return False
+    child_kind = _shape_kind(values[0])
+    if child_kind in _SHAPE_PROFILE_SCALAR_KINDS:
+        return True
+    if child_kind == "ndarray":
+        try:
+            return int(values[0].size) <= 1
+        except Exception:
+            return True
+    return False
+
+
+def construction_shape_profile_v1(value: Any) -> str | None:
+    """Return a deterministic coarse shape profile for construction values."""
+    try:
+        root_kind = _shape_kind(value)
+        if root_kind in _SHAPE_PROFILE_SCALAR_KINDS:
+            return None
+        if root_kind == "ndarray":
+            if int(value.size) <= 1:
+                return None
+        elif root_kind in _SHAPE_PROFILE_CONTAINER_KINDS:
+            if (_shape_len(value) or 0) == 0:
+                return None
+            if _shape_is_single_scalar(value):
+                return None
+        else:
+            return None
+
+        counts: dict[str, int] = {kind: 0 for kind in _SHAPE_PROFILE_KINDS}
+        levels: dict[int, dict[str, int]] = {}
+        flags = {
+            "cycle": False,
+            "has_dict": False,
+            "has_list": False,
+            "has_ndarray": False,
+            "has_set": False,
+            "has_tuple": False,
+            "truncated": False,
+        }
+        ndarray_ndim: set[str] = set()
+        ndarray_size: set[str] = set()
+        ndarray_dtype: set[str] = set()
+        ndarray_rank: set[str] = set()
+        truncated_kind_hist: dict[str, int] = {}
+        max_depth_seen = 0
+        max_fanout = 0
+        visited = 0
+        seen: set[int] = set()
+
+        def record_truncated_children(children: list[Any]) -> None:
+            for child in children:
+                child_kind = _shape_kind(child)
+                truncated_kind_hist[child_kind] = (
+                    truncated_kind_hist.get(child_kind, 0) + 1
+                )
+
+        def visit(node: Any, depth: int) -> None:
+            nonlocal max_depth_seen, max_fanout, visited
+            if visited >= _SHAPE_PROFILE_MAX_VISITED:
+                flags["truncated"] = True
+                return
+            kind = _shape_kind(node)
+            visited += 1
+            counts[kind] = counts.get(kind, 0) + 1
+            levels.setdefault(depth, {})
+            levels[depth][kind] = levels[depth].get(kind, 0) + 1
+            max_depth_seen = max(max_depth_seen, depth)
+
+            if kind == "ndarray":
+                flags["has_ndarray"] = True
+                try:
+                    ndarray_ndim.add(_shape_ndim_bucket(int(node.ndim)))
+                    ndarray_size.add(_shape_count_bucket(int(node.size)))
+                    ndarray_dtype.add(_shape_dtype_kind(node))
+                    ndarray_rank.add(_shape_rank_bucket(node))
+                except Exception:
+                    ndarray_ndim.add("unknown")
+                    ndarray_size.add("unknown")
+                    ndarray_dtype.add("unknown")
+                    ndarray_rank.add("unknown")
+                return
+
+            if kind not in _SHAPE_PROFILE_CONTAINER_KINDS:
+                return
+
+            flags[f"has_{kind}"] = True
+            length = _shape_len(node)
+            if length is None:
+                flags["truncated"] = True
+                return
+            max_fanout = max(max_fanout, int(length))
+            oid = id(node)
+            if oid in seen:
+                flags["cycle"] = True
+                return
+            if depth >= _SHAPE_PROFILE_MAX_DEPTH:
+                if length > 0:
+                    flags["truncated"] = True
+                    record_truncated_children(_shape_container_values(node, kind))
+                return
+
+            children = _shape_container_values(node, kind)
+            if len(children) > _SHAPE_PROFILE_MAX_VISITED:
+                flags["truncated"] = True
+                record_truncated_children(children)
+                if kind in {"dict", "set"}:
+                    return
+                children = children[:_SHAPE_PROFILE_MAX_VISITED]
+            seen.add(oid)
+            children.sort(key=lambda child: _shape_structural_sort_key(child))
+            for child in children:
+                visit(child, depth + 1)
+            seen.remove(oid)
+
+        visit(value, 0)
+
+        scalar_count = (
+            counts.get("none", 0)
+            + counts.get("string", 0)
+            + counts.get("number", 0)
+            + counts.get("bool", 0)
+        )
+        object_count = counts.get("object", 0) + counts.get("unknown", 0)
+        payload_count = scalar_count + object_count + counts.get("ndarray", 0)
+        truncated_non_object = any(
+            kind not in {"object", "unknown"}
+            for kind, count in truncated_kind_hist.items()
+            if count
+        )
+        if payload_count == 0 and not flags["cycle"] and not truncated_non_object:
+            return None
+        if (
+            object_count == payload_count
+            and payload_count > 0
+            and not flags["cycle"]
+            and not truncated_non_object
+        ):
+            return None
+
+        profile = {
+            "v": 1,
+            "root": root_kind,
+            "flags": flags,
+            "max_depth": _shape_depth_bucket(max_depth_seen),
+            "max_fanout": _shape_count_bucket(max_fanout),
+            "visited": _shape_count_bucket(visited),
+            "counts": {
+                kind: _shape_count_bucket(count)
+                for kind, count in counts.items()
+                if count
+            },
+            "groups": {
+                "container": _shape_count_bucket(
+                    counts.get("list", 0)
+                    + counts.get("tuple", 0)
+                    + counts.get("set", 0)
+                    + counts.get("dict", 0)
+                    + counts.get("ndarray", 0)
+                ),
+                "scalar": _shape_count_bucket(scalar_count),
+                "string": _shape_count_bucket(counts.get("string", 0)),
+                "numeric": _shape_count_bucket(counts.get("number", 0)),
+                "bool": _shape_count_bucket(counts.get("bool", 0)),
+                "object": _shape_count_bucket(object_count),
+            },
+            "levels": {
+                str(level): {
+                    kind: _shape_count_bucket(count)
+                    for kind, count in sorted(hist.items())
+                }
+                for level, hist in sorted(levels.items())
+            },
+            "truncated_children": {
+                kind: _shape_count_bucket(count)
+                for kind, count in sorted(truncated_kind_hist.items())
+            },
+            "ndarray": {
+                "dtype": sorted(ndarray_dtype),
+                "ndim": sorted(ndarray_ndim),
+                "rank": sorted(ndarray_rank),
+                "size": sorted(ndarray_size),
+            },
+        }
+        return json.dumps(profile, sort_keys=True, separators=(",", ":"))
+    except Exception:
+        return None
+
+
+def is_degenerate_shape_profile(profile: str | None) -> bool:
+    return not profile or '"degenerate":true' in profile
+
+
 class StateSampler(ABC):
     """Abstract base class for sampling states."""
 
@@ -548,6 +927,83 @@ class PUCTSampler(StateSampler):
 
         return picked
 
+    def sample_states_shape_profile_balanced(self, num_states: int) -> list[State]:
+        if num_states <= 1:
+            return self.sample_states(num_states)
+
+        initial_ids = {s.id for s in self._initial_states}
+        candidates = list(self._states)
+
+        if not candidates:
+            return self.sample_states(num_states)
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
+        children_map = self._build_children_map()
+        picked: list[State] = []
+        top_scores = []
+        picked_ids: set[str] = set()
+        blocked_ids: set[str] = set()
+        seen_profiles: set[str] = set()
+
+        for entry in scores:
+            s = entry[2]
+            if s.id in blocked_ids:
+                continue
+            profile = construction_shape_profile_v1(
+                getattr(s, "construction", None)
+            )
+            if is_degenerate_shape_profile(profile):
+                continue
+            if profile in seen_profiles:
+                continue
+            picked.append(s)
+            top_scores.append(entry)
+            picked_ids.add(s.id)
+            seen_profiles.add(profile)
+            blocked_ids.update(self._get_full_lineage(s, children_map))
+            if len(picked) >= num_states:
+                break
+
+        if len(picked) < num_states:
+            for entry in scores:
+                s = entry[2]
+                if s.id in picked_ids or s.id in blocked_ids:
+                    continue
+                picked.append(s)
+                top_scores.append(entry)
+                picked_ids.add(s.id)
+                blocked_ids.update(self._get_full_lineage(s, children_map))
+                if len(picked) >= num_states:
+                    break
+
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
     def update_states(self, states: list[State], parent_states: list[State], save: bool = True, step: int | None = None):
         if not states:
             return
diff --git a/ttt_discover/discovery.py b/ttt_discover/discovery.py
index 6cdb434..bec19d3 100644
--- a/ttt_discover/discovery.py
+++ b/ttt_discover/discovery.py
@@ -63,6 +63,7 @@ class DiscoverConfig:
     codex_initial_program_paths: tuple[str, ...] = ()
     codex_initial_pool_paths: tuple[str, ...] = ()
     codex_autonomous: bool = False
+    codex_shape_profile_balanced_sampling: bool = False
 
 
 def init_ray(num_cpus_per_task: int, env_type: str):
@@ -145,6 +146,9 @@ async def discover_impl(config: DiscoverConfig):
             max_concurrent_requests=config.codex_max_concurrent_requests,
             initial_program_paths=config.codex_initial_program_paths,
             initial_pool_paths=config.codex_initial_pool_paths,
+            shape_profile_balanced_sampling=(
+                config.codex_shape_profile_balanced_sampling
+            ),
             autonomous=config.codex_autonomous,
             wandb_project=config.wandb_project,
             wandb_name=config.experiment_name,
diff --git a/ttt_discover/rl/codex_no_finetune.py b/ttt_discover/rl/codex_no_finetune.py
index 18a0e9e..017d80a 100644
--- a/ttt_discover/rl/codex_no_finetune.py
+++ b/ttt_discover/rl/codex_no_finetune.py
@@ -137,6 +137,7 @@ class CodexNoFinetuneConfig:
     initial_program_paths: tuple[str, ...] = ()
     initial_pool_paths: tuple[str, ...] = ()
     topk_children: int = 16
+    shape_profile_balanced_sampling: bool = False
 
     # False is TTT Discover: non-auto, many short samples. True is AutoEvolve:
     # one Codex deep-dive workspace per sample with file edits and shell access.
@@ -814,13 +815,28 @@ def _result_metrics(results: list[CandidateResult], kept_results: list[Candidate
     return metrics
 
 
+def _sample_parent_states(
+    cfg: CodexNoFinetuneConfig,
+    sampler: StateSampler,
+) -> list[Any]:
+    if cfg.shape_profile_balanced_sampling:
+        sample = getattr(sampler, "sample_states_shape_profile_balanced", None)
+        if not callable(sample):
+            raise ValueError(
+                "shape_profile_balanced_sampling requires "
+                "sample_states_shape_profile_balanced"
+            )
+        return sample(cfg.groups_per_batch)
+    return sampler.sample_states(cfg.groups_per_batch)
+
+
 async def sample_batch(
     cfg: CodexNoFinetuneConfig,
     sampler: StateSampler,
     i_batch: int,
 ) -> tuple[list[CandidateResult], dict[str, Any], list[CandidateResult]]:
     metrics: dict[str, Any] = {}
-    parent_states = sampler.sample_states(cfg.groups_per_batch)
+    parent_states = _sample_parent_states(cfg, sampler)
     semaphore = (
         asyncio.Semaphore(cfg.max_concurrent_requests)
         if cfg.max_concurrent_requests is not None
````
</details>

