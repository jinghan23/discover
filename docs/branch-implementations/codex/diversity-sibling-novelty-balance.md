# codex/diversity-sibling-novelty-balance

## Summary

提取候选 token 并与同一直接 parent 的 siblings 比较，按 sibling novelty token pair/bucket 做平衡。

## Branch State

- Worktree: `/opt/tiger/discover-sibling-novelty-balance`
- HEAD: `2234282`
- Base used for comparison: `2234282`
- Commits ahead of base: `0`
- Commits behind base: `0`
- Group: `diversity_extra`
- Implementation location: `uncommitted worktree diff`

## Implemented Behavior

- The summary above is based on reading the changed Python code, added config fields, and sampler/prompt hooks in this branch.
- For sampler branches, the common pattern is to keep baseline PUCT ordering inside a chosen surface/bucket, while changing which surface/bucket gets the next parent pick.
- Tests detected: `10` test functions across untracked test files.

## Added Markers

### Config fields

- `codex_sibling_novelty_balanced_sampling`
- `sibling_novelty_balanced_sampling`

### Constants

- `_SIBLING_NOVELTY_TOKEN_RE`

### Classes

- None

### Functions

- `_puct_sorted_scores`
- `_sibling_novelty_tokens`
- `_direct_parent_id`
- `_sibling_novelty_token_pair`
- `_sibling_novelty_bucket`
- `sample_states_sibling_novelty_balanced`
- `pick`

## Diff Summary

- Worktree tracked shortstat: `4 files changed, 231 insertions(+), 1 deletion(-)`
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

- `repro/gpu_mode/run_0608_sibling_novelty_balance.sh (969 bytes)`
- `repro/run_discovery.py (15543 bytes)`
- `tests/test_sibling_novelty_balanced_sampling.py (15543 bytes)`

### Detected Test Functions

- `tests/test_sibling_novelty_balanced_sampling.py::test_sample_batch_uses_ordinary_sampler_unless_enabled`
- `tests/test_sibling_novelty_balanced_sampling.py::test_num_states_at_most_one_matches_ordinary_puct`
- `tests/test_sibling_novelty_balanced_sampling.py::test_first_pass_selects_one_per_known_bucket_in_puct_order`
- `tests/test_sibling_novelty_balanced_sampling.py::test_max_jaccard_and_bucket_boundaries`
- `tests/test_sibling_novelty_balanced_sampling.py::test_unknowns_skip_first_pass_and_fallback_uses_puct_order`
- `tests/test_sibling_novelty_balanced_sampling.py::test_fallback_skips_picked_and_full_lineage_blocked_states`
- `tests/test_sibling_novelty_balanced_sampling.py::test_direct_parent_semantics_use_only_parents_zero`
- `tests/test_sibling_novelty_balanced_sampling.py::test_token_modality_and_supported_token_input_types`
- `tests/test_sibling_novelty_balanced_sampling.py::test_last_sampled_state_metadata_and_table_schema_are_unchanged`
- `tests/test_sibling_novelty_balanced_sampling.py::test_sampler_json_schema_has_no_new_fields_and_old_checkpoint_loads`

## Untracked File Snapshots

These files are not part of `git diff` yet, but they are part of the worktree implementation state recorded here.

### `repro/gpu_mode/run_0608_sibling_novelty_balance.sh`

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
  --experiment-name "${EXPERIMENT_NAME:-gpu-mode-trimul-sibling-novelty-balance-0608}" \
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
  --codex-sibling-novelty-balanced-sampling \
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
        "--codex-sibling-novelty-balanced-sampling",
        action="store_true",
        help="Use sibling novelty balanced PUCT parent sampling.",
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
            "codex_sibling_novelty_balanced_sampling="
            f"{args.codex_sibling_novelty_balanced_sampling}"
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
            sibling_novelty_balanced_sampling=(
                args.codex_sibling_novelty_balanced_sampling
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
        codex_sibling_novelty_balanced_sampling=(
            args.codex_sibling_novelty_balanced_sampling
        ),
        codex_autonomous=args.codex_autonomous,
        remove_constant_reward_groups=remove_constant_reward_groups,
    )
    discover(config)


if __name__ == "__main__":
    main()
````

### `tests/test_sibling_novelty_balanced_sampling.py`

````python
from __future__ import annotations

import asyncio
import json
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

from ttt_discover.codex_utils.runtime import State
from ttt_discover.codex_utils.sampler import PUCTSampler
from ttt_discover.rl import codex_no_finetune as codex_runner


class DummyEnv:
    state_type = State

    @staticmethod
    def create_initial_state(problem_type: str) -> State:
        return State(
            timestep=0,
            construction=[],
            code="",
            value=0.0,
            id=f"initial-{problem_type or 'default'}",
        )


def state(
    state_id: str,
    value: float,
    *,
    parent: str | None = None,
    ancestors: tuple[str, ...] = (),
    construction=None,
    code: str | bytes | None = "",
) -> State:
    parents: list[dict] = []
    if parent is not None:
        parents.append({"id": parent, "timestep": 0})
    parents.extend({"id": ancestor, "timestep": 0} for ancestor in ancestors)
    return State(
        timestep=1,
        construction=[] if construction is None else construction,
        code="" if code is None else code,
        value=value,
        parent_values=[0.0] if parent is not None else [],
        parents=parents,
        id=state_id,
    )


def ids(states: list[State]) -> list[str]:
    return [item.id for item in states]


class SiblingNoveltyBalancedSamplingTest(unittest.TestCase):
    def make_sampler(
        self,
        states: list[State],
        *,
        initial_states: list[State] | None = None,
    ) -> PUCTSampler:
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        sampler = PUCTSampler(
            file_path=str(Path(tmpdir.name) / "puct_sampler.json"),
            env_type=DummyEnv,
            problem_type="test",
            batch_size=0,
            topk_children=0,
            puct_c=0.0,
        )
        sampler._states = list(states)
        sampler._initial_states = list(initial_states or [])
        sampler._n = {}
        sampler._m = {}
        sampler._T = 0
        return sampler

    def parent_groups(self, sampler: PUCTSampler) -> dict[str, list[State]]:
        groups: dict[str, list[State]] = {}
        for item in sampler._states:
            parent_id = sampler._direct_parent_id(item)
            if parent_id is not None:
                groups.setdefault(parent_id, []).append(item)
        return groups

    def test_sample_batch_uses_ordinary_sampler_unless_enabled(self):
        parent = state("parent", 1.0, construction="p", code="p")

        class TrackingSampler:
            def __init__(self):
                self.ordinary_calls: list[int] = []
                self.balanced_calls: list[int] = []

            def sample_states(self, num_states: int) -> list[State]:
                self.ordinary_calls.append(num_states)
                return [parent]

            def sample_states_sibling_novelty_balanced(self, num_states: int) -> list[State]:
                self.balanced_calls.append(num_states)
                return [parent]

        ordinary_cfg = types.SimpleNamespace(
            groups_per_batch=2,
            group_size=0,
            max_concurrent_requests=None,
            remove_constant_reward_groups=False,
            sibling_novelty_balanced_sampling=False,
        )
        ordinary_sampler = TrackingSampler()
        asyncio.run(codex_runner.sample_batch(ordinary_cfg, ordinary_sampler, 0))
        self.assertEqual(ordinary_sampler.ordinary_calls, [2])
        self.assertEqual(ordinary_sampler.balanced_calls, [])

        enabled_cfg = types.SimpleNamespace(
            groups_per_batch=3,
            group_size=0,
            max_concurrent_requests=None,
            remove_constant_reward_groups=False,
            sibling_novelty_balanced_sampling=True,
        )
        enabled_sampler = TrackingSampler()
        asyncio.run(codex_runner.sample_batch(enabled_cfg, enabled_sampler, 0))
        self.assertEqual(enabled_sampler.ordinary_calls, [])
        self.assertEqual(enabled_sampler.balanced_calls, [3])

        class MissingSampler:
            def sample_states(self, num_states: int) -> list[State]:
                return [parent]

        with self.assertRaisesRegex(
            ValueError,
            "sample_states_sibling_novelty_balanced",
        ):
            asyncio.run(codex_runner.sample_batch(enabled_cfg, MissingSampler(), 0))

        class NonCallableSampler(MissingSampler):
            sample_states_sibling_novelty_balanced = None

        with self.assertRaisesRegex(
            ValueError,
            "sample_states_sibling_novelty_balanced",
        ):
            asyncio.run(codex_runner.sample_batch(enabled_cfg, NonCallableSampler(), 0))

    def test_num_states_at_most_one_matches_ordinary_puct(self):
        archive = [
            state("best", 10.0, parent="p", construction="a", code="a"),
            state("second", 9.0, parent="p", construction="b", code="b"),
        ]
        ordinary = self.make_sampler(archive)
        balanced = self.make_sampler(archive)

        self.assertEqual(
            ids(balanced.sample_states_sibling_novelty_balanced(1)),
            ids(ordinary.sample_states(1)),
        )
        self.assertEqual(balanced.sample_states_sibling_novelty_balanced(0), [])

    def test_first_pass_selects_one_per_known_bucket_in_puct_order(self):
        archive = [
            state("redundant", 100.0, parent="p", construction="a b c d"),
            state("redundant-shadow", 99.0, parent="p", construction="a b c"),
            state("variant", 98.0, parent="p", construction="e f g"),
            state("variant-shadow", 97.0, parent="p", construction="e f h i"),
            state("distinct", 96.0, parent="p", construction="j k"),
        ]
        sampler = self.make_sampler(archive)

        picked = sampler.sample_states_sibling_novelty_balanced(3)

        self.assertEqual(ids(picked), ["redundant", "variant", "distinct"])

    def test_max_jaccard_and_bucket_boundaries(self):
        archive = [
            state("max-case", 100.0, parent="p1", construction="a b c d"),
            state("close", 1.0, parent="p1", construction="a b c"),
            state("far", 1.0, parent="p1", construction="x y z"),
            state("variant-boundary", 90.0, parent="p2", construction="m n o"),
            state("variant-peer", 1.0, parent="p2", construction="m n p q"),
            state("distinct-boundary", 80.0, parent="p3", construction="u v"),
            state("distinct-peer", 1.0, parent="p3", construction="u x y z q"),
        ]
        sampler = self.make_sampler(archive)
        groups = self.parent_groups(sampler)

        self.assertEqual(
            sampler._sibling_novelty_bucket(archive[0], groups, set()),
            "sibling_redundant",
        )
        self.assertEqual(
            sampler._sibling_novelty_bucket(archive[3], groups, set()),
            "sibling_variant",
        )
        self.assertEqual(
            sampler._sibling_novelty_bucket(archive[5], groups, set()),
            "sibling_distinct",
        )

    def test_unknowns_skip_first_pass_and_fallback_uses_puct_order(self):
        seed = state("seed", 100.0, parent="seed-parent", construction="s", code="s")
        archive = [
            seed,
            state("parentless", 99.0, construction="p", code="p"),
            state("solo", 98.0, parent="solo-parent", construction="q", code="q"),
            state("empty-a", 97.0, parent="empty-parent", construction=[], code=""),
            state("empty-b", 96.0, parent="empty-parent", construction=[], code=""),
        ]
        sampler = self.make_sampler(archive, initial_states=[seed])

        picked = sampler.sample_states_sibling_novelty_balanced(5)

        self.assertEqual(ids(picked), ["seed", "parentless", "solo", "empty-a", "empty-b"])

    def test_fallback_skips_picked_and_full_lineage_blocked_states(self):
        archive = [
            state("picked", 100.0, parent="root", construction="a", code="a"),
            state("descendant", 99.0, parent="picked", ancestors=("root",), construction="d", code="d"),
            state("fallback-sibling", 98.0, parent="root", construction="b", code="b"),
            state("root", 97.0, construction="root", code="root"),
            state("fallback-unknown", 96.0, parent="lonely", construction="u", code="u"),
        ]
        sampler = self.make_sampler(archive)

        picked = sampler.sample_states_sibling_novelty_balanced(3)

        picked_ids = ids(picked)
        self.assertEqual(picked_ids, ["picked", "fallback-sibling", "fallback-unknown"])
        self.assertEqual(len(picked_ids), len(set(picked_ids)))
        self.assertNotIn("descendant", picked_ids)
        self.assertNotIn("root", picked_ids)

    def test_direct_parent_semantics_use_only_parents_zero(self):
        archive = [
            state("direct-a", 10.0, parent="p1", ancestors=("shared-root",), construction="a"),
            state("different-direct", 9.0, parent="p2", ancestors=("shared-root",), construction="b"),
            state("direct-peer", 8.0, parent="p1", ancestors=("other-root",), construction="c"),
            state("later-ancestor-only", 7.0, parent="p3", ancestors=("p1",), construction="d"),
        ]
        sampler = self.make_sampler(archive)
        groups = self.parent_groups(sampler)

        self.assertEqual(
            sampler._sibling_novelty_bucket(archive[0], groups, set()),
            "sibling_distinct",
        )
        self.assertIsNone(sampler._sibling_novelty_bucket(archive[1], groups, set()))
        self.assertIsNone(sampler._sibling_novelty_bucket(archive[3], groups, set()))

    def test_token_modality_and_supported_token_input_types(self):
        construction_first = [
            state("construction-candidate", 10.0, parent="p1", construction="left", code="same"),
            state("construction-peer", 9.0, parent="p1", construction="right", code="same"),
        ]
        sampler = self.make_sampler(construction_first)
        groups = self.parent_groups(sampler)
        self.assertEqual(
            sampler._sibling_novelty_bucket(construction_first[0], groups, set()),
            "sibling_distinct",
        )

        code_fallback = [
            state("code-candidate", 10.0, parent="p2", construction=[], code="alpha beta"),
            state("code-peer", 9.0, parent="p2", construction=["ignored"], code="alpha beta"),
        ]
        sampler = self.make_sampler(code_fallback)
        groups = self.parent_groups(sampler)
        self.assertEqual(
            sampler._sibling_novelty_bucket(code_fallback[0], groups, set()),
            "sibling_redundant",
        )

        no_cross_modality = [
            state("construction-only", 10.0, parent="p3", construction="token", code=""),
            state("code-only", 9.0, parent="p3", construction=[], code="token"),
        ]
        sampler = self.make_sampler(no_cross_modality)
        groups = self.parent_groups(sampler)
        self.assertIsNone(
            sampler._sibling_novelty_bucket(no_cross_modality[0], groups, set())
        )

        empty_code = [
            state("empty-code-a", 10.0, parent="p4", construction=[], code=""),
            state("empty-code-b", 9.0, parent="p4", construction=None, code=None),
        ]
        sampler = self.make_sampler(empty_code)
        groups = self.parent_groups(sampler)
        self.assertIsNone(sampler._sibling_novelty_bucket(empty_code[0], groups, set()))

        weird_values = [
            state(
                "weird-a",
                10.0,
                parent="p5",
                construction=[
                    {"k": np.array(["Alpha", "Beta"]), "n": np.int64(7)},
                    frozenset({"Gamma"}),
                    b"bytes token",
                    None,
                ],
                code=b"unused bytes",
            ),
            state(
                "weird-b",
                9.0,
                parent="p5",
                construction=({"k": ("Alpha", "Delta")}, {np.float64(3.5)}),
                code="unused text",
            ),
            state("none-value", 8.0, parent="p5", construction=None, code=None),
        ]
        sampler = self.make_sampler(weird_values)
        picked = sampler.sample_states_sibling_novelty_balanced(2)
        self.assertEqual(len(picked), 2)

        circular: list = ["loop"]
        circular.append(circular)
        self.assertEqual(sampler._sibling_novelty_tokens(circular), {"loop"})

    def test_last_sampled_state_metadata_and_table_schema_are_unchanged(self):
        archive = [
            state("redundant", 100.0, parent="p", construction="a b c d"),
            state("redundant-shadow", 99.0, parent="p", construction="a b c"),
            state("variant", 98.0, parent="p", construction="e f g"),
            state("variant-shadow", 97.0, parent="p", construction="e f h i"),
            state("distinct", 96.0, parent="p", construction="j k"),
        ]
        sampler = self.make_sampler(archive)
        picked = sampler.sample_states_sibling_novelty_balanced(3)

        self.assertEqual(len(sampler._last_sampled_states), len(picked))
        self.assertEqual(len(sampler._last_sampled_indices), len(picked))
        self.assertEqual(len(sampler._last_puct_stats), len(picked))
        self.assertEqual(sampler._last_sampled_indices, [0, 2, 4])

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
        columns, rows = sampler.get_sample_table()
        self.assertEqual(columns, expected_columns)
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(len(row) == len(columns) for row in rows))

        ordinary = self.make_sampler(archive)
        ordinary.sample_states(3)
        ordinary_columns, _ = ordinary.get_sample_table()
        self.assertEqual(columns, ordinary_columns)

    def test_sampler_json_schema_has_no_new_fields_and_old_checkpoint_loads(self):
        archive = [
            state("a", 10.0, parent="p", construction="a", code="a"),
            state("b", 9.0, parent="p", construction="b", code="b"),
        ]
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        file_path = str(Path(tmpdir.name) / "puct_sampler.json")
        sampler = PUCTSampler(
            file_path=file_path,
            env_type=DummyEnv,
            problem_type="test",
            batch_size=0,
            topk_children=0,
            puct_c=0.0,
        )
        sampler._states = archive
        sampler._initial_states = []
        sampler.flush(step=1)

        checkpoint_path = Path(tmpdir.name) / "puct_sampler_step_000001.json"
        store = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        self.assertEqual(
            set(store),
            {"step", "states", "initial_states", "puct_n", "puct_m", "puct_T"},
        )

        loaded = PUCTSampler(
            file_path=file_path,
            env_type=DummyEnv,
            problem_type="test",
            batch_size=0,
            resume_step=1,
            topk_children=0,
            puct_c=0.0,
        )
        self.assertEqual(ids(loaded._states), ["a", "b"])


if __name__ == "__main__":
    unittest.main()
````

## Diff Stat

### Worktree Tracked Diff Stat

````text
 ttt_discover/codex_utils/discovery.py |   4 +
 ttt_discover/codex_utils/sampler.py   | 212 ++++++++++++++++++++++++++++++++++
 ttt_discover/discovery.py             |   4 +
 ttt_discover/rl/codex_no_finetune.py  |  12 +-
 4 files changed, 231 insertions(+), 1 deletion(-)
````
## Raw Diff

<details>
<summary>Tracked worktree patch</summary>

````diff
diff --git a/ttt_discover/codex_utils/discovery.py b/ttt_discover/codex_utils/discovery.py
index b21a80f..fbdfc01 100644
--- a/ttt_discover/codex_utils/discovery.py
+++ b/ttt_discover/codex_utils/discovery.py
@@ -50,6 +50,7 @@ class DiscoverConfig:
     codex_initial_program_paths: tuple[str, ...] = ()
     codex_initial_pool_paths: tuple[str, ...] = ()
     codex_autonomous: bool = False
+    codex_sibling_novelty_balanced_sampling: bool = False
 
 
 def _run_codex_no_finetune(config: DiscoverConfig) -> None:
@@ -85,6 +86,9 @@ def _run_codex_no_finetune(config: DiscoverConfig) -> None:
         initial_program_paths=config.codex_initial_program_paths,
         initial_pool_paths=config.codex_initial_pool_paths,
         autonomous=config.codex_autonomous,
+        sibling_novelty_balanced_sampling=(
+            config.codex_sibling_novelty_balanced_sampling
+        ),
         wandb_project=config.wandb_project,
         wandb_name=experiment_name,
         log_path=log_path,
diff --git a/ttt_discover/codex_utils/sampler.py b/ttt_discover/codex_utils/sampler.py
index 5c5d4d1..c178b01 100644
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
 
+_SIBLING_NOVELTY_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
+
 
 @contextmanager
 def _file_lock(lock_path: str, *, poll_s: float = 0.05, stale_s: float = 600.0):
@@ -489,6 +492,153 @@ class PUCTSampler(StateSampler):
                     queue.append(child_id)
         return lineage
 
+    def _puct_sorted_scores(self, candidates: list[State], initial_ids: set[str]) -> list[tuple]:
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
+        return scores
+
+    def _sibling_novelty_tokens(
+        self,
+        value: Any,
+        seen: set[int] | None = None,
+    ) -> set[str]:
+        try:
+            if seen is None:
+                seen = set()
+            if value is None:
+                return set()
+            if isinstance(value, np.ndarray):
+                return self._sibling_novelty_tokens(value.tolist(), seen)
+            if isinstance(value, np.generic):
+                return self._sibling_novelty_tokens(value.item(), seen)
+            if isinstance(value, bytes):
+                text = value.decode("utf-8", errors="replace")
+                return {token.lower() for token in _SIBLING_NOVELTY_TOKEN_RE.findall(text)}
+            if isinstance(value, str):
+                return {token.lower() for token in _SIBLING_NOVELTY_TOKEN_RE.findall(value)}
+            if isinstance(value, dict):
+                obj_id = id(value)
+                if obj_id in seen:
+                    return set()
+                seen.add(obj_id)
+                tokens: set[str] = set()
+                try:
+                    for key, item in value.items():
+                        tokens.update(self._sibling_novelty_tokens(key, seen))
+                        tokens.update(self._sibling_novelty_tokens(item, seen))
+                    return tokens
+                finally:
+                    seen.discard(obj_id)
+            if isinstance(value, (list, tuple, set, frozenset)):
+                obj_id = id(value)
+                if obj_id in seen:
+                    return set()
+                seen.add(obj_id)
+                tokens: set[str] = set()
+                try:
+                    for item in value:
+                        tokens.update(self._sibling_novelty_tokens(item, seen))
+                    return tokens
+                finally:
+                    seen.discard(obj_id)
+            text = str(value)
+            return {token.lower() for token in _SIBLING_NOVELTY_TOKEN_RE.findall(text)}
+        except Exception:
+            return set()
+
+    def _direct_parent_id(self, state: State) -> str | None:
+        parents = getattr(state, "parents", None) or []
+        if not parents:
+            return None
+        first_parent = parents[0]
+        if not isinstance(first_parent, dict):
+            return None
+        parent_id = first_parent.get("id")
+        if parent_id is None or parent_id == "":
+            return None
+        return str(parent_id)
+
+    def _sibling_novelty_token_pair(
+        self,
+        candidate: State,
+        sibling: State,
+    ) -> tuple[set[str], set[str]] | None:
+        candidate_construction = self._sibling_novelty_tokens(
+            getattr(candidate, "construction", None)
+        )
+        sibling_construction = self._sibling_novelty_tokens(
+            getattr(sibling, "construction", None)
+        )
+        if candidate_construction and sibling_construction:
+            return candidate_construction, sibling_construction
+
+        candidate_code = self._sibling_novelty_tokens(getattr(candidate, "code", None))
+        sibling_code = self._sibling_novelty_tokens(getattr(sibling, "code", None))
+        if candidate_code and sibling_code:
+            return candidate_code, sibling_code
+        return None
+
+    def _sibling_novelty_bucket(
+        self,
+        candidate: State,
+        parent_to_states: dict[str, list[State]],
+        initial_ids: set[str],
+    ) -> str | None:
+        if candidate.id in initial_ids:
+            return None
+        parent_id = self._direct_parent_id(candidate)
+        if parent_id is None:
+            return None
+
+        visible_siblings = [
+            sibling
+            for sibling in parent_to_states.get(parent_id, [])
+            if sibling.id != candidate.id
+        ]
+        if not visible_siblings:
+            return None
+
+        max_similarity: float | None = None
+        for sibling in visible_siblings:
+            token_pair = self._sibling_novelty_token_pair(candidate, sibling)
+            if token_pair is None:
+                continue
+            left, right = token_pair
+            union = left | right
+            if not union:
+                continue
+            similarity = len(left & right) / len(union)
+            max_similarity = (
+                similarity
+                if max_similarity is None
+                else max(max_similarity, similarity)
+            )
+
+        if max_similarity is None:
+            return None
+
+        distance = 1.0 - max_similarity
+        if distance <= 0.25 + 1e-12:
+            return "sibling_redundant"
+        if distance <= 0.60 + 1e-12:
+            return "sibling_variant"
+        return "sibling_distinct"
+
     def sample_states(self, num_states: int) -> list[State]:
         initial_ids = {s.id for s in self._initial_states}
         candidates = list(self._states)
@@ -548,6 +698,68 @@ class PUCTSampler(StateSampler):
 
         return picked
 
+    def sample_states_sibling_novelty_balanced(self, num_states: int) -> list[State]:
+        if num_states <= 1:
+            return self.sample_states(num_states)
+
+        initial_ids = {s.id for s in self._initial_states}
+        candidates = list(self._states)
+        if not candidates:
+            return self.sample_states(num_states)
+
+        scores = self._puct_sorted_scores(candidates, initial_ids)
+        children_map = self._build_children_map()
+        parent_to_states: dict[str, list[State]] = {}
+        for state in candidates:
+            parent_id = self._direct_parent_id(state)
+            if parent_id is not None:
+                parent_to_states.setdefault(parent_id, []).append(state)
+
+        picked: list[State] = []
+        top_scores: list[tuple] = []
+        picked_ids: set[str] = set()
+        blocked_ids: set[str] = set()
+        covered_buckets: set[str] = set()
+
+        def pick(entry: tuple) -> None:
+            state = entry[2]
+            picked.append(state)
+            top_scores.append(entry)
+            picked_ids.add(state.id)
+            blocked_ids.update(self._get_full_lineage(state, children_map))
+
+        for entry in scores:
+            state = entry[2]
+            if state.id in blocked_ids:
+                continue
+            bucket = self._sibling_novelty_bucket(state, parent_to_states, initial_ids)
+            if bucket is None or bucket in covered_buckets:
+                continue
+            pick(entry)
+            covered_buckets.add(bucket)
+            if len(picked) >= num_states:
+                break
+
+        if len(picked) < num_states:
+            for entry in scores:
+                state = entry[2]
+                if state.id in picked_ids or state.id in blocked_ids:
+                    continue
+                pick(entry)
+                if len(picked) >= num_states:
+                    break
+
+        state_id_to_idx = {s.id: i for i, s in enumerate(self._states)}
+        self._last_sampled_states = picked
+        self._last_sampled_indices = [state_id_to_idx.get(s.id, -1) for s in picked]
+        self._last_puct_stats = [(t[3], t[4], t[5], t[6], t[0]) for t in top_scores]
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
diff --git a/ttt_discover/discovery.py b/ttt_discover/discovery.py
index 6cdb434..63b6c90 100644
--- a/ttt_discover/discovery.py
+++ b/ttt_discover/discovery.py
@@ -63,6 +63,7 @@ class DiscoverConfig:
     codex_initial_program_paths: tuple[str, ...] = ()
     codex_initial_pool_paths: tuple[str, ...] = ()
     codex_autonomous: bool = False
+    codex_sibling_novelty_balanced_sampling: bool = False
 
 
 def init_ray(num_cpus_per_task: int, env_type: str):
@@ -146,6 +147,9 @@ async def discover_impl(config: DiscoverConfig):
             initial_program_paths=config.codex_initial_program_paths,
             initial_pool_paths=config.codex_initial_pool_paths,
             autonomous=config.codex_autonomous,
+            sibling_novelty_balanced_sampling=(
+                config.codex_sibling_novelty_balanced_sampling
+            ),
             wandb_project=config.wandb_project,
             wandb_name=config.experiment_name,
             log_path=log_path,
diff --git a/ttt_discover/rl/codex_no_finetune.py b/ttt_discover/rl/codex_no_finetune.py
index 18a0e9e..86686e4 100644
--- a/ttt_discover/rl/codex_no_finetune.py
+++ b/ttt_discover/rl/codex_no_finetune.py
@@ -137,6 +137,7 @@ class CodexNoFinetuneConfig:
     initial_program_paths: tuple[str, ...] = ()
     initial_pool_paths: tuple[str, ...] = ()
     topk_children: int = 16
+    sibling_novelty_balanced_sampling: bool = False
 
     # False is TTT Discover: non-auto, many short samples. True is AutoEvolve:
     # one Codex deep-dive workspace per sample with file edits and shell access.
@@ -820,7 +821,16 @@ async def sample_batch(
     i_batch: int,
 ) -> tuple[list[CandidateResult], dict[str, Any], list[CandidateResult]]:
     metrics: dict[str, Any] = {}
-    parent_states = sampler.sample_states(cfg.groups_per_batch)
+    if cfg.sibling_novelty_balanced_sampling:
+        sample_fn = getattr(sampler, "sample_states_sibling_novelty_balanced", None)
+        if not callable(sample_fn):
+            raise ValueError(
+                "sibling_novelty_balanced_sampling requires sampler."
+                "sample_states_sibling_novelty_balanced(num_states)"
+            )
+        parent_states = sample_fn(cfg.groups_per_batch)
+    else:
+        parent_states = sampler.sample_states(cfg.groups_per_batch)
     semaphore = (
         asyncio.Semaphore(cfg.max_concurrent_requests)
         if cfg.max_concurrent_requests is not None
````
</details>

