# codex/diversity-root-drift-balance

## Summary

把候选与 lineage root 的 construction/code token 集合比较，按从 root 漂移程度分桶，鼓励不同 root-drift 水平。

## Branch State

- Worktree: `/opt/tiger/discover-root-drift-balance`
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

- `codex_root_drift_balanced_sampling`
- `root_drift_balanced_sampling`

### Constants

- None

### Classes

- None

### Functions

- `_root_drift_tokens`
- `_root_drift_tokens_inner`
- `_root_drift_bucket_from_tokens`
- `_root_drift_bucket`
- `sample_states_root_drift_balanced`

## Diff Summary

- Worktree tracked shortstat: `4 files changed, 179 insertions(+), 1 deletion(-)`
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

- `repro/gpu_mode/run_0608_root_drift_balance.sh (959 bytes)`
- `repro/run_discovery.py (15447 bytes)`
- `tests/test_root_drift_balanced_sampling.py (18032 bytes)`

### Detected Test Functions

- `tests/test_root_drift_balanced_sampling.py::test_num_states_one_or_less_matches_ordinary_puct`
- `tests/test_root_drift_balanced_sampling.py::test_root_selection_uses_earliest_visible_ancestor`
- `tests/test_root_drift_balanced_sampling.py::test_parent_order_scans_from_tail_toward_direct_parent`
- `tests/test_root_drift_balanced_sampling.py::test_token_modality_and_supported_token_inputs`
- `tests/test_root_drift_balanced_sampling.py::test_jaccard_bucket_boundaries`
- `tests/test_root_drift_balanced_sampling.py::test_first_pass_balances_known_buckets_by_puct_order`
- `tests/test_root_drift_balanced_sampling.py::test_unknowns_are_fallback_only_and_all_unknown_follows_puct`
- `tests/test_root_drift_balanced_sampling.py::test_initial_state_with_parent_metadata_is_unknown_until_fallback`
- `tests/test_root_drift_balanced_sampling.py::test_fallback_fills_past_three_without_duplicates`
- `tests/test_root_drift_balanced_sampling.py::test_full_lineage_blocking_applies_in_first_pass_and_fallback`
- `tests/test_root_drift_balanced_sampling.py::test_last_sampled_state_stats_and_sample_table_schema_align`
- `tests/test_root_drift_balanced_sampling.py::test_sampler_json_schema_has_no_root_drift_fields_and_old_checkpoint_loads`
- `tests/test_root_drift_balanced_sampling.py::test_sample_batch_requires_api_only_when_flag_enabled`

## Untracked File Snapshots

These files are not part of `git diff` yet, but they are part of the worktree implementation state recorded here.

### `repro/gpu_mode/run_0608_root_drift_balance.sh`

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
  --experiment-name "${EXPERIMENT_NAME:-gpu-mode-trimul-root-drift-balance-0608}" \
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
  --codex-root-drift-balanced-sampling \
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
        "--codex-root-drift-balanced-sampling",
        action="store_true",
        help="Use root-drift balanced PUCT parent sampling.",
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
            "codex_root_drift_balanced_sampling="
            f"{args.codex_root_drift_balanced_sampling}"
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
            root_drift_balanced_sampling=args.codex_root_drift_balanced_sampling,
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
        codex_root_drift_balanced_sampling=args.codex_root_drift_balanced_sampling,
        codex_autonomous=args.codex_autonomous,
        remove_constant_reward_groups=remove_constant_reward_groups,
    )
    discover(config)


if __name__ == "__main__":
    main()
````

### `tests/test_root_drift_balanced_sampling.py`

````python
from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from ttt_discover.codex_utils.runtime import State
from ttt_discover.codex_utils.sampler import (
    PUCTSampler,
    _sampler_file_for_step,
)
from ttt_discover.rl import codex_no_finetune
from ttt_discover.rl.codex_no_finetune import (
    CandidateResult,
    CodexNoFinetuneConfig,
)


class DummyEnv:
    state_type = State

    @classmethod
    def create_initial_state(cls, problem_type: str) -> State:
        return State(
            timestep=0,
            construction=["seed"],
            code="seed",
            value=0.0,
            id=f"seed-{problem_type}",
        )


def parent_ref(state_id: str) -> dict:
    return {"id": state_id, "timestep": 0}


def make_state(
    state_id: str,
    value: float,
    *,
    construction=None,
    code="",
    parents: list[dict] | None = None,
    timestep: int = 0,
) -> State:
    return State(
        timestep=timestep,
        construction=construction,
        code=code,
        value=value,
        parents=parents or [],
        id=state_id,
    )


class RootDriftBalancedSamplingTest(unittest.TestCase):
    def make_sampler(
        self,
        states: list[State],
        *,
        initial_states: list[State] | None = None,
    ) -> PUCTSampler:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        sampler = PUCTSampler(
            file_path=str(Path(tmp.name) / "puct_sampler.json"),
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

    def test_num_states_one_or_less_matches_ordinary_puct(self):
        states = [
            make_state("low", 1.0, construction="a"),
            make_state("high", 3.0, construction="b"),
            make_state("mid", 2.0, construction="c"),
        ]
        sampler = self.make_sampler(states)

        ordinary_zero = sampler.sample_states(0)
        balanced_zero = sampler.sample_states_root_drift_balanced(0)
        self.assertEqual([s.id for s in ordinary_zero], [s.id for s in balanced_zero])

        ordinary_one = sampler.sample_states(1)
        balanced_one = sampler.sample_states_root_drift_balanced(1)
        self.assertEqual([s.id for s in ordinary_one], [s.id for s in balanced_one])

    def test_root_selection_uses_earliest_visible_ancestor(self):
        root = make_state("root", 0.0, construction="r1 r2 r3")
        node_a = make_state(
            "A",
            1.0,
            construction="a1 a2 a3",
            parents=[parent_ref("root")],
        )
        node_b = make_state(
            "B",
            2.0,
            construction="a1 a2",
            parents=[parent_ref("A"), parent_ref("root")],
        )
        sampler = self.make_sampler([root, node_a, node_b])
        state_by_id = {s.id: s for s in sampler._states}

        self.assertEqual(sampler._root_drift_bucket(node_b, state_by_id), "root_far")

        sampler._states = [node_a, node_b]
        state_by_id = {s.id: s for s in sampler._states}
        self.assertEqual(sampler._root_drift_bucket(node_b, state_by_id), "root_near")

        sampler._states = [node_b]
        state_by_id = {s.id: s for s in sampler._states}
        self.assertIsNone(sampler._root_drift_bucket(node_b, state_by_id))

    def test_parent_order_scans_from_tail_toward_direct_parent(self):
        root = make_state("root", 0.0, construction="r1 r2 r3")
        direct = make_state(
            "direct",
            1.0,
            construction="b1 b2",
            parents=[parent_ref("root")],
        )
        current = make_state(
            "current",
            2.0,
            construction="b1 b2",
            parents=[parent_ref("direct"), parent_ref("root")],
        )
        sampler = self.make_sampler([root, direct, current])
        state_by_id = {s.id: s for s in sampler._states}

        self.assertEqual(sampler._root_drift_bucket(current, state_by_id), "root_far")

    def test_token_modality_and_supported_token_inputs(self):
        root = make_state("root", 0.0, construction="same tokens", code="root_code")
        current = make_state(
            "current",
            1.0,
            construction="same tokens",
            code="different_code",
            parents=[parent_ref("root")],
        )
        sampler = self.make_sampler([root, current])
        state_by_id = {s.id: s for s in sampler._states}
        self.assertEqual(sampler._root_drift_bucket(current, state_by_id), "root_near")

        root = make_state("root", 0.0, construction=["construction"], code="c1 c2 c3")
        current = make_state(
            "current",
            1.0,
            construction=None,
            code="c1 c2",
            parents=[parent_ref("root")],
        )
        sampler = self.make_sampler([root, current])
        state_by_id = {s.id: s for s in sampler._states}
        self.assertEqual(sampler._root_drift_bucket(current, state_by_id), "root_near")

        root = make_state("root", 0.0, construction=None, code="")
        current = make_state(
            "current",
            1.0,
            construction=None,
            code="",
            parents=[parent_ref("root")],
        )
        sampler = self.make_sampler([root, current])
        state_by_id = {s.id: s for s in sampler._states}
        self.assertIsNone(sampler._root_drift_bucket(current, state_by_id))

        root = make_state(
            "root",
            0.0,
            construction={"k": [b"alpha beta", np.array(["gamma"])]},
            code=None,
        )
        current = make_state(
            "current",
            1.0,
            construction=frozenset({b"alpha"}),
            code=np.array(["unused"]),
            parents=[parent_ref("root")],
        )
        sampler = self.make_sampler([root, current])
        state_by_id = {s.id: s for s in sampler._states}
        self.assertIsNotNone(sampler._root_drift_bucket(current, state_by_id))
        self.assertEqual(sampler._root_drift_tokens(np.int64(7)), {"7"})
        self.assertEqual(sampler._root_drift_tokens(None), set())

    def test_jaccard_bucket_boundaries(self):
        bucket = PUCTSampler._root_drift_bucket_from_tokens

        self.assertEqual(bucket({"a"}, {"a"}), "root_near")
        self.assertEqual(bucket({"a", "b"}, {"a", "b", "c"}), "root_near")
        self.assertEqual(bucket({"a"}, {"a", "b", "c"}), "root_mid")
        self.assertEqual(bucket({"a"}, {"b"}), "root_far")

    def test_first_pass_balances_known_buckets_by_puct_order(self):
        root = make_state("root", 0.0, construction="a b c")
        near_high = make_state(
            "near_high",
            100.0,
            construction="a b c",
            parents=[parent_ref("root")],
        )
        near_low = make_state(
            "near_low",
            95.0,
            construction="a b",
            parents=[parent_ref("root")],
        )
        mid = make_state(
            "mid",
            90.0,
            construction="a",
            parents=[parent_ref("root")],
        )
        far = make_state(
            "far",
            85.0,
            construction="z",
            parents=[parent_ref("root")],
        )
        sampler = self.make_sampler([root, near_high, near_low, mid, far])

        picked = sampler.sample_states_root_drift_balanced(3)

        self.assertEqual([s.id for s in picked], ["near_high", "mid", "far"])

    def test_unknowns_are_fallback_only_and_all_unknown_follows_puct(self):
        root = make_state("root", 0.0, construction="a b c")
        unknown = make_state("unknown", 100.0, construction="x")
        near = make_state(
            "near",
            90.0,
            construction="a b c",
            parents=[parent_ref("root")],
        )
        sampler = self.make_sampler([root, unknown, near])
        picked = sampler.sample_states_root_drift_balanced(2)
        self.assertEqual([s.id for s in picked], ["near", "unknown"])

        all_unknown = [
            make_state("u1", 30.0, construction="x"),
            make_state("u2", 20.0, construction="y"),
            make_state("u3", 10.0, construction="z"),
        ]
        sampler = self.make_sampler(all_unknown)
        picked = sampler.sample_states_root_drift_balanced(2)
        self.assertEqual([s.id for s in picked], ["u1", "u2"])

    def test_initial_state_with_parent_metadata_is_unknown_until_fallback(self):
        root = make_state("root", 0.0, construction="a b c")
        seed = make_state(
            "seed",
            100.0,
            construction="a b c",
            parents=[parent_ref("root")],
        )
        near = make_state(
            "near",
            90.0,
            construction="a b c",
            parents=[parent_ref("root")],
        )
        sampler = self.make_sampler([root, seed, near], initial_states=[seed])

        picked = sampler.sample_states_root_drift_balanced(2)

        self.assertEqual([s.id for s in picked], ["near", "seed"])

    def test_fallback_fills_past_three_without_duplicates(self):
        root = make_state("root", 0.0, construction="a b c")
        near_high = make_state(
            "near_high",
            100.0,
            construction="a b c",
            parents=[parent_ref("root")],
        )
        near_low = make_state(
            "near_low",
            95.0,
            construction="a b",
            parents=[parent_ref("root")],
        )
        mid = make_state(
            "mid",
            90.0,
            construction="a",
            parents=[parent_ref("root")],
        )
        far = make_state(
            "far",
            85.0,
            construction="z",
            parents=[parent_ref("root")],
        )
        unknown = make_state("unknown", 80.0, construction="q")
        sampler = self.make_sampler([root, near_high, near_low, mid, far, unknown])

        picked = sampler.sample_states_root_drift_balanced(5)

        picked_ids = [s.id for s in picked]
        self.assertEqual(picked_ids, ["near_high", "mid", "far", "near_low", "unknown"])
        self.assertEqual(len(picked_ids), len(set(picked_ids)))

    def test_full_lineage_blocking_applies_in_first_pass_and_fallback(self):
        root = make_state("root", 0.0, construction="a b c")
        ancestor = make_state(
            "ancestor",
            100.0,
            construction="a b c",
            parents=[parent_ref("root")],
        )
        descendant = make_state(
            "descendant",
            95.0,
            construction="z",
            parents=[parent_ref("ancestor"), parent_ref("root")],
        )
        mid = make_state(
            "mid",
            90.0,
            construction="a",
            parents=[parent_ref("root")],
        )
        unknown = make_state("unknown", 80.0, construction="q")
        sampler = self.make_sampler([root, ancestor, descendant, mid, unknown])

        picked = sampler.sample_states_root_drift_balanced(3)

        self.assertEqual([s.id for s in picked], ["ancestor", "mid", "unknown"])
        self.assertNotIn("descendant", [s.id for s in picked])

    def test_last_sampled_state_stats_and_sample_table_schema_align(self):
        root = make_state("root", 0.0, construction="a b c")
        near = make_state(
            "near",
            30.0,
            construction="a b c",
            parents=[parent_ref("root")],
        )
        mid = make_state(
            "mid",
            20.0,
            construction="a",
            parents=[parent_ref("root")],
        )
        far = make_state(
            "far",
            10.0,
            construction="z",
            parents=[parent_ref("root")],
        )
        sampler = self.make_sampler([root, near, mid, far])
        ordinary_columns, _ = sampler.get_sample_table()

        picked = sampler.sample_states_root_drift_balanced(3)
        columns, rows = sampler.get_sample_table()

        self.assertEqual(columns, ordinary_columns)
        self.assertEqual(
            columns,
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
        self.assertEqual(sampler._last_sampled_states, picked)
        self.assertEqual(len(sampler._last_sampled_indices), len(picked))
        self.assertEqual(len(sampler._last_puct_stats), len(picked))
        self.assertEqual(len(rows), len(picked))
        self.assertTrue(all(len(row) == len(columns) for row in rows))
        self.assertEqual(
            sampler._last_sampled_indices,
            [sampler._states.index(state) for state in picked],
        )

    def test_sampler_json_schema_has_no_root_drift_fields_and_old_checkpoint_loads(self):
        root = make_state("root", 0.0, construction="a b c")
        near = make_state(
            "near",
            1.0,
            construction="a b c",
            parents=[parent_ref("root")],
        )
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        file_path = str(Path(tmp.name) / "puct_sampler.json")
        sampler = PUCTSampler(
            file_path=file_path,
            env_type=DummyEnv,
            batch_size=0,
            puct_c=0.0,
            topk_children=0,
        )
        sampler._states = [root, near]
        sampler.sample_states_root_drift_balanced(2)
        sampler.flush(step=1)

        saved = json.loads(Path(_sampler_file_for_step(file_path, 1)).read_text())
        self.assertEqual(
            set(saved),
            {"step", "states", "initial_states", "puct_n", "puct_m", "puct_T"},
        )

        old_file_path = str(Path(tmp.name) / "old_sampler.json")
        old_store = {
            "step": 3,
            "states": [root.to_dict(), near.to_dict()],
            "initial_states": [root.to_dict()],
            "puct_n": {"root": 1},
            "puct_m": {"root": 1.0},
            "puct_T": 1,
        }
        Path(_sampler_file_for_step(old_file_path, 3)).write_text(
            json.dumps(old_store),
            encoding="utf-8",
        )

        loaded = PUCTSampler(
            file_path=old_file_path,
            env_type=DummyEnv,
            batch_size=0,
            resume_step=3,
            puct_c=0.0,
            topk_children=0,
        )
        self.assertEqual([s.id for s in loaded._states], ["root", "near"])
        self.assertEqual(loaded._n, {"root": 1})
        self.assertEqual(loaded._m, {"root": 1.0})
        self.assertEqual(loaded._T, 1)

    def test_sample_batch_requires_api_only_when_flag_enabled(self):
        parent = make_state("parent", 1.0, construction="p")

        class OrdinaryOnlySampler:
            def __init__(self) -> None:
                self.calls: list[int] = []
                self.failed = 0

            def sample_states(self, num_states: int) -> list[State]:
                self.calls.append(num_states)
                return [parent]

            def update_states(self, states, parent_states, save=True, step=None):
                return None

            def flush(self, step=None):
                return None

            def record_failed_rollout(self, parent_state):
                self.failed += 1

        cfg = CodexNoFinetuneConfig(
            env_type=DummyEnv,
            groups_per_batch=1,
            group_size=1,
            max_concurrent_requests=None,
            log_path="/tmp/root-drift-test",
            root_drift_balanced_sampling=True,
        )
        with self.assertRaisesRegex(ValueError, "sample_states_root_drift_balanced"):
            asyncio.run(codex_no_finetune.sample_batch(cfg, OrdinaryOnlySampler(), 0))

        class NonCallableRootDriftSampler(OrdinaryOnlySampler):
            sample_states_root_drift_balanced = object()

        with self.assertRaisesRegex(ValueError, "sample_states_root_drift_balanced"):
            asyncio.run(
                codex_no_finetune.sample_batch(cfg, NonCallableRootDriftSampler(), 0)
            )

        async def fake_run_candidate(
            cfg,
            sampler,
            parent_state,
            *,
            group_idx,
            sample_idx,
            step_idx,
            semaphore,
        ):
            return CandidateResult(
                parent_state=parent_state,
                group_idx=group_idx,
                sample_idx=sample_idx,
                prompt="",
                response="",
                parsed_code="",
                reward=0.0,
                correctness=0.0,
                raw_score=None,
                msg="",
                metrics={},
                next_state=None,
            )

        cfg = CodexNoFinetuneConfig(
            env_type=DummyEnv,
            groups_per_batch=1,
            group_size=1,
            max_concurrent_requests=None,
            log_path="/tmp/root-drift-test",
            root_drift_balanced_sampling=False,
        )
        sampler = OrdinaryOnlySampler()
        with mock.patch.object(codex_no_finetune, "_run_candidate", fake_run_candidate):
            kept_results, metrics, all_results = asyncio.run(
                codex_no_finetune.sample_batch(cfg, sampler, 0)
            )

        self.assertEqual(sampler.calls, [1])
        self.assertEqual(sampler.failed, 1)
        self.assertEqual(len(kept_results), 1)
        self.assertEqual(len(all_results), 1)
        self.assertEqual(metrics["codex/parent_states"], 1)


if __name__ == "__main__":
    unittest.main()
````

## Diff Stat

### Worktree Tracked Diff Stat

````text
 ttt_discover/codex_utils/discovery.py |   2 +
 ttt_discover/codex_utils/sampler.py   | 165 ++++++++++++++++++++++++++++++++++
 ttt_discover/discovery.py             |   2 +
 ttt_discover/rl/codex_no_finetune.py  |  11 ++-
 4 files changed, 179 insertions(+), 1 deletion(-)
````
## Raw Diff

<details>
<summary>Tracked worktree patch</summary>

````diff
diff --git a/ttt_discover/codex_utils/discovery.py b/ttt_discover/codex_utils/discovery.py
index b21a80f..addd7fd 100644
--- a/ttt_discover/codex_utils/discovery.py
+++ b/ttt_discover/codex_utils/discovery.py
@@ -50,6 +50,7 @@ class DiscoverConfig:
     codex_initial_program_paths: tuple[str, ...] = ()
     codex_initial_pool_paths: tuple[str, ...] = ()
     codex_autonomous: bool = False
+    codex_root_drift_balanced_sampling: bool = False
 
 
 def _run_codex_no_finetune(config: DiscoverConfig) -> None:
@@ -85,6 +86,7 @@ def _run_codex_no_finetune(config: DiscoverConfig) -> None:
         initial_program_paths=config.codex_initial_program_paths,
         initial_pool_paths=config.codex_initial_pool_paths,
         autonomous=config.codex_autonomous,
+        root_drift_balanced_sampling=config.codex_root_drift_balanced_sampling,
         wandb_project=config.wandb_project,
         wandb_name=experiment_name,
         log_path=log_path,
diff --git a/ttt_discover/codex_utils/sampler.py b/ttt_discover/codex_utils/sampler.py
index 5c5d4d1..75323a0 100644
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
@@ -489,6 +490,96 @@ class PUCTSampler(StateSampler):
                     queue.append(child_id)
         return lineage
 
+    def _root_drift_tokens(self, value: Any) -> set[str] | None:
+        try:
+            return self._root_drift_tokens_inner(value, set())
+        except Exception:
+            return None
+
+    def _root_drift_tokens_inner(self, value: Any, seen: set[int]) -> set[str]:
+        if value is None:
+            return set()
+        if isinstance(value, np.generic):
+            return self._root_drift_tokens_inner(value.item(), seen)
+        if isinstance(value, np.ndarray):
+            return self._root_drift_tokens_inner(value.tolist(), seen)
+        if isinstance(value, bytes):
+            text = value.decode("utf-8", errors="replace")
+            return set(re.findall(r"[A-Za-z0-9_]+", text.lower()))
+        if isinstance(value, str):
+            return set(re.findall(r"[A-Za-z0-9_]+", value.lower()))
+        if isinstance(value, dict):
+            obj_id = id(value)
+            if obj_id in seen:
+                return set()
+            seen.add(obj_id)
+            tokens: set[str] = set()
+            for key, item in value.items():
+                tokens.update(self._root_drift_tokens_inner(key, seen))
+                tokens.update(self._root_drift_tokens_inner(item, seen))
+            seen.remove(obj_id)
+            return tokens
+        if isinstance(value, (list, tuple, set, frozenset)):
+            obj_id = id(value)
+            if obj_id in seen:
+                return set()
+            seen.add(obj_id)
+            tokens: set[str] = set()
+            for item in value:
+                tokens.update(self._root_drift_tokens_inner(item, seen))
+            seen.remove(obj_id)
+            return tokens
+        return set(re.findall(r"[A-Za-z0-9_]+", str(value).lower()))
+
+    @staticmethod
+    def _root_drift_bucket_from_tokens(current: set[str], root: set[str]) -> str | None:
+        union_size = len(current | root)
+        if union_size == 0:
+            return None
+        drift_numer = union_size - len(current & root)
+        if 3 * drift_numer <= union_size:
+            return "root_near"
+        if 3 * drift_numer <= 2 * union_size:
+            return "root_mid"
+        return "root_far"
+
+    def _root_drift_bucket(
+        self,
+        state: State,
+        state_by_id: dict[str, State],
+        initial_ids: set[str] | None = None,
+    ) -> str | None:
+        if initial_ids is not None and state.id in initial_ids:
+            return None
+        if not state.parents:
+            return None
+
+        root_state = None
+        for parent in reversed(state.parents):
+            pid = parent.get("id") if isinstance(parent, dict) else None
+            if pid is None:
+                continue
+            root_state = state_by_id.get(str(pid))
+            if root_state is not None:
+                break
+        if root_state is None:
+            return None
+
+        current_construction = self._root_drift_tokens(getattr(state, "construction", None))
+        root_construction = self._root_drift_tokens(getattr(root_state, "construction", None))
+        if current_construction is None or root_construction is None:
+            return None
+        if current_construction and root_construction:
+            return self._root_drift_bucket_from_tokens(current_construction, root_construction)
+
+        current_code = self._root_drift_tokens(getattr(state, "code", None))
+        root_code = self._root_drift_tokens(getattr(root_state, "code", None))
+        if current_code is None or root_code is None:
+            return None
+        if not current_code or not root_code:
+            return None
+        return self._root_drift_bucket_from_tokens(current_code, root_code)
+
     def sample_states(self, num_states: int) -> list[State]:
         initial_ids = {s.id for s in self._initial_states}
         candidates = list(self._states)
@@ -548,6 +639,80 @@ class PUCTSampler(StateSampler):
 
         return picked
 
+    def sample_states_root_drift_balanced(self, num_states: int) -> list[State]:
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
+        state_by_id = {str(s.id): s for s in self._states}
+        children_map = self._build_children_map()
+        picked: list[State] = []
+        top_scores = []
+        picked_ids: set[str] = set()
+        blocked_ids: set[str] = set()
+        covered_buckets: set[str] = set()
+
+        for entry in scores:
+            s = entry[2]
+            if s.id in picked_ids or s.id in blocked_ids:
+                continue
+            bucket = self._root_drift_bucket(s, state_by_id, initial_ids)
+            if bucket is None or bucket in covered_buckets:
+                continue
+            picked.append(s)
+            top_scores.append(entry)
+            picked_ids.add(s.id)
+            covered_buckets.add(bucket)
+            blocked_ids.update(self._get_full_lineage(s, children_map))
+            if len(picked) >= num_states or len(covered_buckets) >= 3:
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
index 6cdb434..cb02689 100644
--- a/ttt_discover/discovery.py
+++ b/ttt_discover/discovery.py
@@ -63,6 +63,7 @@ class DiscoverConfig:
     codex_initial_program_paths: tuple[str, ...] = ()
     codex_initial_pool_paths: tuple[str, ...] = ()
     codex_autonomous: bool = False
+    codex_root_drift_balanced_sampling: bool = False
 
 
 def init_ray(num_cpus_per_task: int, env_type: str):
@@ -146,6 +147,7 @@ async def discover_impl(config: DiscoverConfig):
             initial_program_paths=config.codex_initial_program_paths,
             initial_pool_paths=config.codex_initial_pool_paths,
             autonomous=config.codex_autonomous,
+            root_drift_balanced_sampling=config.codex_root_drift_balanced_sampling,
             wandb_project=config.wandb_project,
             wandb_name=config.experiment_name,
             log_path=log_path,
diff --git a/ttt_discover/rl/codex_no_finetune.py b/ttt_discover/rl/codex_no_finetune.py
index 18a0e9e..301d8d5 100644
--- a/ttt_discover/rl/codex_no_finetune.py
+++ b/ttt_discover/rl/codex_no_finetune.py
@@ -137,6 +137,7 @@ class CodexNoFinetuneConfig:
     initial_program_paths: tuple[str, ...] = ()
     initial_pool_paths: tuple[str, ...] = ()
     topk_children: int = 16
+    root_drift_balanced_sampling: bool = False
 
     # False is TTT Discover: non-auto, many short samples. True is AutoEvolve:
     # one Codex deep-dive workspace per sample with file edits and shell access.
@@ -820,7 +821,15 @@ async def sample_batch(
     i_batch: int,
 ) -> tuple[list[CandidateResult], dict[str, Any], list[CandidateResult]]:
     metrics: dict[str, Any] = {}
-    parent_states = sampler.sample_states(cfg.groups_per_batch)
+    if cfg.root_drift_balanced_sampling:
+        sample_root_drift = getattr(sampler, "sample_states_root_drift_balanced", None)
+        if not callable(sample_root_drift):
+            raise ValueError(
+                "root_drift_balanced_sampling requires sampler.sample_states_root_drift_balanced"
+            )
+        parent_states = sample_root_drift(cfg.groups_per_batch)
+    else:
+        parent_states = sampler.sample_states(cfg.groups_per_batch)
     semaphore = (
         asyncio.Semaphore(cfg.max_concurrent_requests)
         if cfg.max_concurrent_requests is not None
````
</details>

