# codex/diversity-edit-distance-balance

## Summary

比较 child 与 parent 的 token edit（新增/删除 token）并按 edit distance bucket 采样，鼓励不同修改幅度的父节点。

## Branch State

- Worktree: `/opt/tiger/discover-edit-distance-balance`
- HEAD: `2234282`
- Base used for comparison: `2234282`
- Commits ahead of base: `0`
- Commits behind base: `0`
- Group: `diversity_extra`
- Implementation location: `uncommitted worktree diff`

## Implemented Behavior

- The summary above is based on reading the changed Python code, added config fields, and sampler/prompt hooks in this branch.
- For sampler branches, the common pattern is to keep baseline PUCT ordering inside a chosen surface/bucket, while changing which surface/bucket gets the next parent pick.
- Tests detected: `12` test functions across untracked test files.

## Added Markers

### Config fields

- `codex_edit_distance_balanced_sampling`
- `edit_distance_balanced_sampling`

### Constants

- `_TOKEN_RE`

### Classes

- None

### Functions

- `_rank_puct_candidates`
- `_store_sample_result`
- `_tokens_from_value`
- `_edge_edit_tokens`
- `_edge_edit_distance_bucket`
- `sample_states`
- `sample_states_edit_distance_balanced`

## Diff Summary

- Worktree tracked shortstat: `4 files changed, 169 insertions(+), 17 deletions(-)`
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

- `repro/gpu_mode/run_0608_edit_distance_balance.sh (965 bytes)`
- `repro/run_discovery.py (15495 bytes)`
- `tests/test_edit_distance_balanced_sampling.py (15976 bytes)`

### Detected Test Functions

- `tests/test_edit_distance_balanced_sampling.py::test_num_states_le_one_matches_ordinary_puct`
- `tests/test_edit_distance_balanced_sampling.py::test_three_known_buckets_selected_in_puct_scan_order`
- `tests/test_edit_distance_balanced_sampling.py::test_bucket_boundaries`
- `tests/test_edit_distance_balanced_sampling.py::test_unknown_skipped_first_pass_and_all_unknown_falls_back`
- `tests/test_edit_distance_balanced_sampling.py::test_fallback_after_three_buckets_does_not_repeat_picks`
- `tests/test_edit_distance_balanced_sampling.py::test_full_lineage_blocking_applies_in_first_pass_and_fallback`
- `tests/test_edit_distance_balanced_sampling.py::test_missing_direct_parent_is_unknown_and_does_not_crash`
- `tests/test_edit_distance_balanced_sampling.py::test_construction_preferred_then_code_fallback_and_types_do_not_crash`
- `tests/test_edit_distance_balanced_sampling.py::test_no_instance_level_stale_cache_for_changed_parent_tokens`
- `tests/test_edit_distance_balanced_sampling.py::test_sampler_json_schema_and_old_checkpoint_load`
- `tests/test_edit_distance_balanced_sampling.py::test_sample_table_schema_unchanged_after_edit_distance_mode`
- `tests/test_edit_distance_balanced_sampling.py::test_sample_batch_api_selection`

## Untracked File Snapshots

These files are not part of `git diff` yet, but they are part of the worktree implementation state recorded here.

### `repro/gpu_mode/run_0608_edit_distance_balance.sh`

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
  --experiment-name "${EXPERIMENT_NAME:-gpu-mode-trimul-edit-distance-balance-0608}" \
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
  --codex-edit-distance-balanced-sampling \
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
        "--codex-edit-distance-balanced-sampling",
        action="store_true",
        help="Use edit-distance balanced PUCT parent sampling.",
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
            "codex_edit_distance_balanced_sampling="
            f"{args.codex_edit_distance_balanced_sampling}"
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
            edit_distance_balanced_sampling=args.codex_edit_distance_balanced_sampling,
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
        codex_edit_distance_balanced_sampling=(
            args.codex_edit_distance_balanced_sampling
        ),
        codex_autonomous=args.codex_autonomous,
        remove_constant_reward_groups=remove_constant_reward_groups,
    )
    discover(config)


if __name__ == "__main__":
    main()
````

### `tests/test_edit_distance_balanced_sampling.py`

````python
from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np

from ttt_discover.codex_utils.runtime import State
from ttt_discover.codex_utils.sampler import PUCTSampler
from ttt_discover.rl.codex_no_finetune import CodexNoFinetuneConfig, sample_batch


class DummyEnv:
    state_type = State

    @staticmethod
    def create_initial_state(problem_type: str = "") -> State:
        return State(
            timestep=-1,
            construction=["seed"],
            code="seed",
            value=0.0,
            id=f"seed-{problem_type}",
        )


def make_state(
    state_id: str,
    value: float,
    *,
    construction: Any = None,
    code: Any = "",
    parent: State | None = None,
) -> State:
    parents = []
    parent_values = []
    if parent is not None:
        parents = [{"id": parent.id, "timestep": parent.timestep}] + list(
            parent.parents or []
        )
        parent_values = [parent.value]
    return State(
        timestep=1,
        construction=construction,
        code=code,
        value=value,
        parent_values=parent_values,
        parents=parents,
        id=state_id,
    )


class OrdinaryOnlySampler:
    def __init__(self) -> None:
        self.sample_calls = 0

    def sample_states(self, num_states: int) -> list[State]:
        self.sample_calls += 1
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


class EditDistanceBalancedSamplingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self._sampler_idx = 0

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def sampler(
        self,
        states: list[State],
        *,
        initial_states: list[State] | None = None,
    ) -> PUCTSampler:
        self._sampler_idx += 1
        path = Path(self.tmp.name) / f"sampler_{self._sampler_idx}.json"
        sampler = PUCTSampler(
            file_path=str(path),
            env_type=DummyEnv,
            batch_size=0,
            puct_c=0.0,
            topk_children=0,
        )
        sampler._states = list(states)
        sampler._initial_states = list(initial_states or [])
        sampler._n = {}
        sampler._m = {}
        sampler._T = 0
        return sampler

    def bucket(self, sampler: PUCTSampler, state: State) -> str | None:
        return sampler._edge_edit_distance_bucket(
            state,
            state_by_id={str(s.id): s for s in sampler._states},
            initial_ids={s.id for s in sampler._initial_states},
        )

    def test_num_states_le_one_matches_ordinary_puct(self) -> None:
        root = make_state("root", 0.0, construction=["a", "b"], code="root")
        child = make_state(
            "child",
            10.0,
            construction=["a", "b", "c"],
            code="child",
            parent=root,
        )
        states = [root, child]

        ordinary_one = self.sampler(states).sample_states(1)
        balanced_one = self.sampler(states).sample_states_edit_distance_balanced(1)
        self.assertEqual([s.id for s in ordinary_one], [s.id for s in balanced_one])

        ordinary_zero = self.sampler(states).sample_states(0)
        balanced_zero = self.sampler(states).sample_states_edit_distance_balanced(0)
        self.assertEqual([s.id for s in ordinary_zero], [s.id for s in balanced_zero])

    def test_three_known_buckets_selected_in_puct_scan_order(self) -> None:
        p_small = make_state("p_small", 0.0, construction=["a", "b", "c", "d"])
        small = make_state(
            "small",
            30.0,
            construction=["a", "b", "c", "d", "e"],
            parent=p_small,
        )
        p_large = make_state("p_large", 0.0, construction=["l1", "l2", "l3", "l4"])
        large = make_state(
            "large",
            29.0,
            construction=["l1", "x", "y", "z"],
            parent=p_large,
        )
        p_medium = make_state(
            "p_medium",
            0.0,
            construction=["m1", "m2", "m3", "m4", "m5"],
        )
        medium = make_state(
            "medium",
            28.0,
            construction=["m1", "m2", "m3", "x", "y"],
            parent=p_medium,
        )
        sampler = self.sampler([p_small, p_large, p_medium, small, large, medium])

        picked = sampler.sample_states_edit_distance_balanced(3)

        self.assertEqual([s.id for s in picked], ["small", "large", "medium"])
        self.assertEqual([self.bucket(sampler, s) for s in picked], [
            "edit_small",
            "edit_large",
            "edit_medium",
        ])

    def test_bucket_boundaries(self) -> None:
        p_small = make_state("p_small", 0.0, construction=["a", "b", "c", "d"])
        small = make_state(
            "small",
            10.0,
            construction=["a", "b", "c"],
            parent=p_small,
        )
        p_medium = make_state("p_medium", 0.0, construction=["a", "b"])
        medium = make_state(
            "medium",
            9.0,
            construction=["a", "b", "c", "d", "e"],
            parent=p_medium,
        )
        p_large = make_state("p_large", 0.0, construction=["a", "b"])
        large = make_state(
            "large",
            8.0,
            construction=["a", "c", "d", "e", "f"],
            parent=p_large,
        )
        sampler = self.sampler([p_small, small, p_medium, medium, p_large, large])

        self.assertEqual(self.bucket(sampler, small), "edit_small")
        self.assertEqual(self.bucket(sampler, medium), "edit_medium")
        self.assertEqual(self.bucket(sampler, large), "edit_large")

    def test_unknown_skipped_first_pass_and_all_unknown_falls_back(self) -> None:
        parent = make_state("parent", 0.0, construction=["a", "b", "c", "d"])
        known = make_state(
            "known",
            20.0,
            construction=["a", "b", "c", "d", "e"],
            parent=parent,
        )
        unknown = make_state("unknown", 30.0, construction=["u"], code="u")
        sampler = self.sampler([parent, known, unknown])

        picked = sampler.sample_states_edit_distance_balanced(2)

        self.assertEqual([s.id for s in picked], ["known", "unknown"])

        unknown_a = make_state("unknown_a", 30.0, construction=["a"], code="a")
        unknown_b = make_state("unknown_b", 20.0, construction=["b"], code="b")
        fallback_sampler = self.sampler([unknown_a, unknown_b])

        fallback = fallback_sampler.sample_states_edit_distance_balanced(2)

        self.assertEqual([s.id for s in fallback], ["unknown_a", "unknown_b"])

    def test_fallback_after_three_buckets_does_not_repeat_picks(self) -> None:
        p_small = make_state("p_small", 0.0, construction=["a", "b", "c", "d"])
        small1 = make_state(
            "small1",
            100.0,
            construction=["a", "b", "c", "d", "e"],
            parent=p_small,
        )
        small2 = make_state(
            "small2",
            99.0,
            construction=["a", "b", "c", "d", "f"],
            parent=p_small,
        )
        p_medium = make_state("p_medium", 0.0, construction=["m1", "m2"])
        medium = make_state(
            "medium",
            98.0,
            construction=["m1", "m2", "m3", "m4", "m5"],
            parent=p_medium,
        )
        p_large = make_state("p_large", 0.0, construction=["l1", "l2"])
        large = make_state(
            "large",
            97.0,
            construction=["l1", "x", "y", "z"],
            parent=p_large,
        )
        sampler = self.sampler([p_small, p_medium, p_large, small1, small2, medium, large])

        picked = sampler.sample_states_edit_distance_balanced(4)

        self.assertEqual([s.id for s in picked], ["small1", "medium", "large", "small2"])
        self.assertEqual(len({s.id for s in picked}), 4)

    def test_full_lineage_blocking_applies_in_first_pass_and_fallback(self) -> None:
        root = make_state("root", 0.0, construction=["r", "s", "t", "u"])
        parent = make_state(
            "parent",
            100.0,
            construction=["r", "s", "t", "u", "v"],
            parent=root,
        )
        blocked_child = make_state(
            "blocked_child",
            99.0,
            construction=["r", "x", "y", "z"],
            parent=parent,
        )
        p_medium = make_state("p_medium", 0.0, construction=["m1", "m2"])
        medium = make_state(
            "medium",
            80.0,
            construction=["m1", "m2", "m3", "m4", "m5"],
            parent=p_medium,
        )
        p_large = make_state("p_large", 0.0, construction=["l1", "l2"])
        large = make_state(
            "large",
            70.0,
            construction=["l1", "x", "y", "z"],
            parent=p_large,
        )
        p_small2 = make_state("p_small2", 0.0, construction=["q", "w", "e", "r"])
        small2 = make_state(
            "small2",
            60.0,
            construction=["q", "w", "e", "r", "t"],
            parent=p_small2,
        )
        sampler = self.sampler(
            [
                root,
                parent,
                blocked_child,
                p_medium,
                medium,
                p_large,
                large,
                p_small2,
                small2,
            ]
        )

        picked = sampler.sample_states_edit_distance_balanced(4)

        ids = [s.id for s in picked]
        self.assertEqual(ids, ["parent", "medium", "large", "small2"])
        self.assertNotIn("blocked_child", ids)

    def test_missing_direct_parent_is_unknown_and_does_not_crash(self) -> None:
        child = State(
            timestep=1,
            construction=["a"],
            code="a",
            value=10.0,
            parents=[{"id": "pruned", "timestep": 0}],
            id="child",
        )
        sampler = self.sampler([child])

        self.assertIsNone(self.bucket(sampler, child))
        self.assertEqual(
            [s.id for s in sampler.sample_states_edit_distance_balanced(2)],
            ["child"],
        )

    def test_construction_preferred_then_code_fallback_and_types_do_not_crash(self) -> None:
        parent = make_state(
            "parent",
            0.0,
            construction=["same", "core"],
            code="parent code only",
        )
        child = make_state(
            "child",
            10.0,
            construction=("same", "core"),
            code="totally different code",
            parent=parent,
        )
        sampler = self.sampler([parent, child])
        self.assertEqual(self.bucket(sampler, child), "edit_small")

        fallback_parent = make_state(
            "fallback_parent",
            0.0,
            construction=None,
            code="shared code tokens",
        )
        fallback_child = make_state(
            "fallback_child",
            9.0,
            construction=[],
            code="shared code tokens",
            parent=fallback_parent,
        )
        fallback_sampler = self.sampler([fallback_parent, fallback_child])
        self.assertEqual(self.bucket(fallback_sampler, fallback_child), "edit_small")

        mixed_parent = make_state(
            "mixed_parent",
            0.0,
            construction={"arr": np.array(["one", np.int64(2)]), "none": None},
            code=b"bytes_token",
        )
        mixed_child = make_state(
            "mixed_child",
            8.0,
            construction=[{"nested": (b"one", None, np.float64(3.0))}],
            code=None,
            parent=mixed_parent,
        )
        mixed_sampler = self.sampler([mixed_parent, mixed_child])
        self.assertIn(
            self.bucket(mixed_sampler, mixed_child),
            {"edit_small", "edit_medium", "edit_large"},
        )

    def test_no_instance_level_stale_cache_for_changed_parent_tokens(self) -> None:
        parent = make_state("parent", 0.0, construction=["a", "b", "c", "d"])
        child = make_state(
            "child",
            10.0,
            construction=["a", "b", "c", "d"],
            parent=parent,
        )
        sampler = self.sampler([parent, child])

        self.assertEqual(self.bucket(sampler, child), "edit_small")
        parent.construction = ["w", "x", "y", "z"]
        self.assertEqual(self.bucket(sampler, child), "edit_large")

    def test_sampler_json_schema_and_old_checkpoint_load(self) -> None:
        state = make_state("state", 1.0, construction=["a"], code="a")
        sampler = self.sampler([state])
        sampler.flush(step=2)
        saved = json.loads(
            (Path(self.tmp.name) / "sampler_1_step_000002.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            set(saved.keys()),
            {"step", "states", "initial_states", "puct_n", "puct_m", "puct_T"},
        )

        old_path = Path(self.tmp.name) / "old_sampler.json"
        old_checkpoint = old_path.with_name("old_sampler_step_000001.json")
        old_checkpoint.write_text(
            json.dumps(
                {
                    "step": 1,
                    "states": [state.to_dict()],
                    "initial_states": [],
                    "puct_n": {},
                    "puct_m": {},
                    "puct_T": 0,
                }
            ),
            encoding="utf-8",
        )
        loaded = PUCTSampler(
            file_path=str(old_path),
            env_type=DummyEnv,
            batch_size=0,
            resume_step=1,
            puct_c=0.0,
        )
        self.assertEqual([s.id for s in loaded._states], ["state"])

    def test_sample_table_schema_unchanged_after_edit_distance_mode(self) -> None:
        parent = make_state("parent", 0.0, construction=["a", "b", "c", "d"])
        child = make_state(
            "child",
            10.0,
            construction=["a", "b", "c", "d", "e"],
            parent=parent,
        )
        sampler = self.sampler([parent, child])
        expected_columns = [
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
        ]

        sampler.sample_states_edit_distance_balanced(2)
        columns, rows = sampler.get_sample_table()

        self.assertEqual(columns, expected_columns)
        self.assertTrue(rows)
        self.assertTrue(all(len(row) == len(expected_columns) for row in rows))

    def test_sample_batch_api_selection(self) -> None:
        cfg = CodexNoFinetuneConfig(
            env_type=DummyEnv,
            groups_per_batch=2,
            group_size=1,
            log_path=self.tmp.name,
            edit_distance_balanced_sampling=True,
        )
        with self.assertRaisesRegex(ValueError, "sample_states_edit_distance_balanced"):
            asyncio.run(sample_batch(cfg, OrdinaryOnlySampler(), 0))

        ordinary_sampler = OrdinaryOnlySampler()
        ordinary_cfg = CodexNoFinetuneConfig(
            env_type=DummyEnv,
            groups_per_batch=2,
            group_size=1,
            log_path=self.tmp.name,
            edit_distance_balanced_sampling=False,
        )

        kept, metrics, all_results = asyncio.run(
            sample_batch(ordinary_cfg, ordinary_sampler, 0)
        )

        self.assertEqual(ordinary_sampler.sample_calls, 1)
        self.assertEqual(kept, [])
        self.assertEqual(all_results, [])
        self.assertEqual(metrics["codex/parent_states"], 0)


if __name__ == "__main__":
    unittest.main()
````

## Diff Stat

### Worktree Tracked Diff Stat

````text
 ttt_discover/codex_utils/discovery.py |   2 +
 ttt_discover/codex_utils/sampler.py   | 170 ++++++++++++++++++++++++++++++----
 ttt_discover/discovery.py             |   2 +
 ttt_discover/rl/codex_no_finetune.py  |  12 ++-
 4 files changed, 169 insertions(+), 17 deletions(-)
````
## Raw Diff

<details>
<summary>Tracked worktree patch</summary>

````diff
diff --git a/ttt_discover/codex_utils/discovery.py b/ttt_discover/codex_utils/discovery.py
index b21a80f..94e047c 100644
--- a/ttt_discover/codex_utils/discovery.py
+++ b/ttt_discover/codex_utils/discovery.py
@@ -49,6 +49,7 @@ class DiscoverConfig:
     codex_max_concurrent_requests: int | None = 4
     codex_initial_program_paths: tuple[str, ...] = ()
     codex_initial_pool_paths: tuple[str, ...] = ()
+    codex_edit_distance_balanced_sampling: bool = False
     codex_autonomous: bool = False
 
 
@@ -84,6 +85,7 @@ def _run_codex_no_finetune(config: DiscoverConfig) -> None:
         max_concurrent_requests=config.codex_max_concurrent_requests,
         initial_program_paths=config.codex_initial_program_paths,
         initial_pool_paths=config.codex_initial_pool_paths,
+        edit_distance_balanced_sampling=config.codex_edit_distance_balanced_sampling,
         autonomous=config.codex_autonomous,
         wandb_project=config.wandb_project,
         wandb_name=experiment_name,
diff --git a/ttt_discover/codex_utils/sampler.py b/ttt_discover/codex_utils/sampler.py
index 5c5d4d1..3e478cf 100644
--- a/ttt_discover/codex_utils/sampler.py
+++ b/ttt_discover/codex_utils/sampler.py
@@ -6,6 +6,7 @@ import json
 import logging
 import os
 from pathlib import Path
+import re
 import threading
 import time
 from typing import Any, Callable
@@ -16,6 +17,8 @@ from ttt_discover.codex_utils.runtime import State, state_from_dict, to_json_ser
 
 logger = logging.getLogger(__name__)
 
+_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|\d+(?:\.\d+)?")
+
 
 @contextmanager
 def _file_lock(lock_path: str, *, poll_s: float = 0.05, stale_s: float = 600.0):
@@ -489,19 +492,14 @@ class PUCTSampler(StateSampler):
                     queue.append(child_id)
         return lineage
 
-    def sample_states(self, num_states: int) -> list[State]:
+    def _rank_puct_candidates(
+        self,
+    ) -> tuple[set[str], list[State], list[tuple[float, float, State, int, float, float, float]]]:
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
+            return initial_ids, candidates, []
 
         vals = np.array([float(s.value if s.value is not None else float("-inf")) for s in candidates])
         non_initial_mask = np.array([s.id not in initial_ids for s in candidates])
@@ -520,6 +518,104 @@ class PUCTSampler(StateSampler):
             scores.append((score, vals[i], s, n, Q, P[i], bonus))
 
         scores.sort(key=lambda x: (x[0], x[1]), reverse=True)
+        return initial_ids, candidates, scores
+
+    def _store_sample_result(
+        self,
+        candidates: list[State],
+        initial_ids: set[str],
+        picked: list[State],
+        top_scores: list[tuple[float, float, State, int, float, float, float]],
+    ) -> None:
+        state_id_to_idx = {s.id: i for i, s in enumerate(candidates)}
+        self._last_sampled_states = picked
+        self._last_sampled_indices = [state_id_to_idx.get(s.id, -1) for s in picked]
+        self._last_puct_stats = [(t[3], t[4], t[5], t[6], t[0]) for t in top_scores]
+
+        for s in picked:
+            if s.id in initial_ids:
+                self._refresh_random_construction(s)
+
+    def _tokens_from_value(self, value: Any) -> set[str]:
+        if value is None:
+            return set()
+        if isinstance(value, np.ndarray):
+            return self._tokens_from_value(value.tolist())
+        if isinstance(value, np.generic):
+            return self._tokens_from_value(value.item())
+        if isinstance(value, bytes):
+            value = value.decode("utf-8", errors="replace")
+        if isinstance(value, str):
+            return set(_TOKEN_RE.findall(value.lower()))
+        if isinstance(value, dict):
+            tokens: set[str] = set()
+            for key, item in value.items():
+                tokens.update(self._tokens_from_value(key))
+                tokens.update(self._tokens_from_value(item))
+            return tokens
+        if isinstance(value, (list, tuple, set, frozenset)):
+            tokens: set[str] = set()
+            for item in value:
+                tokens.update(self._tokens_from_value(item))
+            return tokens
+        return set(_TOKEN_RE.findall(str(value).lower()))
+
+    def _edge_edit_tokens(self, child: State, parent: State) -> tuple[set[str], set[str]]:
+        child_construction_tokens = self._tokens_from_value(getattr(child, "construction", None))
+        parent_construction_tokens = self._tokens_from_value(getattr(parent, "construction", None))
+        if child_construction_tokens and parent_construction_tokens:
+            return child_construction_tokens, parent_construction_tokens
+        return (
+            self._tokens_from_value(getattr(child, "code", None)),
+            self._tokens_from_value(getattr(parent, "code", None)),
+        )
+
+    def _edge_edit_distance_bucket(
+        self,
+        state: State,
+        *,
+        state_by_id: dict[str, State],
+        initial_ids: set[str],
+    ) -> str | None:
+        if state.id in initial_ids:
+            return None
+        parent_entries = getattr(state, "parents", None) or []
+        if not parent_entries:
+            return None
+        direct_parent = parent_entries[0]
+        if not isinstance(direct_parent, dict):
+            return None
+        parent_id = direct_parent.get("id")
+        if not parent_id:
+            return None
+        parent = state_by_id.get(str(parent_id))
+        if parent is None:
+            return None
+
+        child_tokens, parent_tokens = self._edge_edit_tokens(state, parent)
+        union = child_tokens | parent_tokens
+        if not union:
+            return None
+
+        distance = 1.0 - (len(child_tokens & parent_tokens) / len(union))
+        if distance <= 0.25:
+            return "edit_small"
+        if distance <= 0.60:
+            return "edit_medium"
+        return "edit_large"
+
+    def sample_states(self, num_states: int) -> list[State]:
+        initial_ids, candidates, scores = self._rank_puct_candidates()
+
+        if not candidates:
+            picked = [
+                create_initial_state(self.env_type, self.problem_type)
+                for _ in range(num_states)
+            ]
+            self._last_sampled_states = picked
+            self._last_sampled_indices = []
+            self._last_puct_stats = [(0, 0.0, 0.0, 0.0, 0.0) for _ in picked]
+            return picked
 
         if num_states > 1:
             children_map = self._build_children_map()
@@ -537,15 +633,57 @@ class PUCTSampler(StateSampler):
             top_scores = scores[:num_states]
             picked = [t[2] for t in top_scores]
 
-        state_id_to_idx = {s.id: i for i, s in enumerate(self._states)}
-        self._last_sampled_states = picked
-        self._last_sampled_indices = [state_id_to_idx.get(s.id, -1) for s in picked]
-        self._last_puct_stats = [(t[3], t[4], t[5], t[6], t[0]) for t in top_scores]
+        self._store_sample_result(candidates, initial_ids, picked, top_scores)
+        return picked
 
-        for s in picked:
-            if s.id in initial_ids:
-                self._refresh_random_construction(s)
+    def sample_states_edit_distance_balanced(self, num_states: int) -> list[State]:
+        if num_states <= 1:
+            return self.sample_states(num_states)
+
+        initial_ids, candidates, scores = self._rank_puct_candidates()
+        if not candidates:
+            return self.sample_states(num_states)
+
+        state_by_id = {str(s.id): s for s in candidates}
+        children_map = self._build_children_map()
+        picked: list[State] = []
+        top_scores: list[tuple[float, float, State, int, float, float, float]] = []
+        picked_ids: set[str] = set()
+        blocked_ids: set[str] = set()
+        covered_buckets: set[str] = set()
+
+        for entry in scores:
+            state = entry[2]
+            if state.id in blocked_ids:
+                continue
+            bucket = self._edge_edit_distance_bucket(
+                state,
+                state_by_id=state_by_id,
+                initial_ids=initial_ids,
+            )
+            if bucket is None or bucket in covered_buckets:
+                continue
+            picked.append(state)
+            top_scores.append(entry)
+            picked_ids.add(state.id)
+            covered_buckets.add(bucket)
+            blocked_ids.update(self._get_full_lineage(state, children_map))
+            if len(picked) >= num_states:
+                break
+
+        if len(picked) < num_states:
+            for entry in scores:
+                state = entry[2]
+                if state.id in picked_ids or state.id in blocked_ids:
+                    continue
+                picked.append(state)
+                top_scores.append(entry)
+                picked_ids.add(state.id)
+                blocked_ids.update(self._get_full_lineage(state, children_map))
+                if len(picked) >= num_states:
+                    break
 
+        self._store_sample_result(candidates, initial_ids, picked, top_scores)
         return picked
 
     def update_states(self, states: list[State], parent_states: list[State], save: bool = True, step: int | None = None):
diff --git a/ttt_discover/discovery.py b/ttt_discover/discovery.py
index 6cdb434..37c1b8f 100644
--- a/ttt_discover/discovery.py
+++ b/ttt_discover/discovery.py
@@ -62,6 +62,7 @@ class DiscoverConfig:
     codex_max_concurrent_requests: int | None = 4
     codex_initial_program_paths: tuple[str, ...] = ()
     codex_initial_pool_paths: tuple[str, ...] = ()
+    codex_edit_distance_balanced_sampling: bool = False
     codex_autonomous: bool = False
 
 
@@ -145,6 +146,7 @@ async def discover_impl(config: DiscoverConfig):
             max_concurrent_requests=config.codex_max_concurrent_requests,
             initial_program_paths=config.codex_initial_program_paths,
             initial_pool_paths=config.codex_initial_pool_paths,
+            edit_distance_balanced_sampling=config.codex_edit_distance_balanced_sampling,
             autonomous=config.codex_autonomous,
             wandb_project=config.wandb_project,
             wandb_name=config.experiment_name,
diff --git a/ttt_discover/rl/codex_no_finetune.py b/ttt_discover/rl/codex_no_finetune.py
index 18a0e9e..46b5312 100644
--- a/ttt_discover/rl/codex_no_finetune.py
+++ b/ttt_discover/rl/codex_no_finetune.py
@@ -137,6 +137,7 @@ class CodexNoFinetuneConfig:
     initial_program_paths: tuple[str, ...] = ()
     initial_pool_paths: tuple[str, ...] = ()
     topk_children: int = 16
+    edit_distance_balanced_sampling: bool = False
 
     # False is TTT Discover: non-auto, many short samples. True is AutoEvolve:
     # one Codex deep-dive workspace per sample with file edits and shell access.
@@ -820,7 +821,16 @@ async def sample_batch(
     i_batch: int,
 ) -> tuple[list[CandidateResult], dict[str, Any], list[CandidateResult]]:
     metrics: dict[str, Any] = {}
-    parent_states = sampler.sample_states(cfg.groups_per_batch)
+    if cfg.edit_distance_balanced_sampling:
+        sample_balanced = getattr(sampler, "sample_states_edit_distance_balanced", None)
+        if sample_balanced is None or not callable(sample_balanced):
+            raise ValueError(
+                "edit_distance_balanced_sampling=True requires sampler "
+                "sample_states_edit_distance_balanced(num_states)"
+            )
+        parent_states = sample_balanced(cfg.groups_per_batch)
+    else:
+        parent_states = sampler.sample_states(cfg.groups_per_batch)
     semaphore = (
         asyncio.Semaphore(cfg.max_concurrent_requests)
         if cfg.max_concurrent_requests is not None
````
</details>

