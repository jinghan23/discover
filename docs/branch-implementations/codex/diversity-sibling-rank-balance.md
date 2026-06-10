# codex/diversity-sibling-rank-balance

## Summary

在同一 sibling group 内按 value 排名，记录 group size、rank 和 rank bucket，并平衡不同 sibling rank 位置。

## Branch State

- Worktree: `/opt/tiger/discover-sibling-rank-balance`
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

- `codex_sibling_rank_balanced_sampling`
- `initial_ids`
- `groups`
- `metadata`
- `sibling_rank_balanced_sampling`

### Constants

- None

### Classes

- None

### Functions

- `_direct_parent_id`
- `_state_value_for_sibling_rank`
- `_has_seed_timestep`
- `_sibling_rank_metadata`
- `_ranked_puct_entries`
- `_set_last_sampled`
- `_set_last_sampled_without_puct`
- `_refresh_sampled_initials`
- `sample_states_sibling_rank_balanced`

## Diff Summary

- Worktree tracked shortstat: `4 files changed, 337 insertions(+), 35 deletions(-)`
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

- `repro/gpu_mode/run_0608_sibling_rank_balance.sh (908 bytes)`
- `repro/run_discovery.py (15894 bytes)`
- `tests/test_sibling_rank_balanced_sampling.py (12026 bytes)`

### Detected Test Functions

- `tests/test_sibling_rank_balanced_sampling.py::test_direct_parent_only_parentless_and_only_child_metadata`
- `tests/test_sibling_rank_balanced_sampling.py::test_leader_contender_tail_are_first_pass_unique_buckets`
- `tests/test_sibling_rank_balanced_sampling.py::test_all_tied_siblings_do_not_create_fake_diversity`
- `tests/test_sibling_rank_balanced_sampling.py::test_tied_top_and_bottom_share_rank_and_bucket`
- `tests/test_sibling_rank_balanced_sampling.py::test_none_value_is_bottom_sibling_rank`
- `tests/test_sibling_rank_balanced_sampling.py::test_parentless_is_not_first_pass_quota_but_can_fallback`
- `tests/test_sibling_rank_balanced_sampling.py::test_seed_timestep_with_direct_parent_is_not_first_pass_quota`
- `tests/test_sibling_rank_balanced_sampling.py::test_fallback_does_not_relax_full_lineage_blocking`
- `tests/test_sibling_rank_balanced_sampling.py::test_table_uses_sample_time_cache_after_archive_changes_and_flush`
- `tests/test_sibling_rank_balanced_sampling.py::test_num_states_less_than_or_equal_one_matches_normal_puct`
- `tests/test_sibling_rank_balanced_sampling.py::test_sample_batch_flag_on_requires_sampler_api`
- `tests/test_sibling_rank_balanced_sampling.py::test_sample_batch_flag_off_uses_regular_sampler`

## Untracked File Snapshots

These files are not part of `git diff` yet, but they are part of the worktree implementation state recorded here.

### `repro/gpu_mode/run_0608_sibling_rank_balance.sh`

````bash
#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

# TTT Discover sampling mode: intentionally no --codex-autonomous.
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}" \
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}" \
python repro/run_discovery.py \
  --task gpu_mode:trimul \
  --runner codex_no_finetune \
  --experiment-name "${EXPERIMENT_NAME:-gpu-mode-trimul-sibling-rank-balance-0608}" \
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
  --codex-sibling-rank-balanced-sampling \
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
        "--codex-sibling-rank-balanced-sampling",
        action="store_true",
        help="Use sibling-rank balanced PUCT parent sampling.",
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
            "codex_sibling_rank_balanced_sampling="
            f"{args.codex_sibling_rank_balanced_sampling}"
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
            sibling_rank_balanced_sampling=args.codex_sibling_rank_balanced_sampling,
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
        codex_sibling_rank_balanced_sampling=args.codex_sibling_rank_balanced_sampling,
        codex_autonomous=args.codex_autonomous,
        remove_constant_reward_groups=remove_constant_reward_groups,
    )
    discover(config)


if __name__ == "__main__":
    main()
````

### `tests/test_sibling_rank_balanced_sampling.py`

````python
from __future__ import annotations

import asyncio
import os
import tempfile
import unittest

from ttt_discover.codex_utils.runtime import State
from ttt_discover.codex_utils.sampler import PUCTSampler, _sibling_rank_metadata
from ttt_discover.rl.codex_no_finetune import CodexNoFinetuneConfig, sample_batch


class DummyEnv:
    state_type = State
    _seed_counter = 0

    @classmethod
    def create_initial_state(cls, problem_type: str) -> State:
        cls._seed_counter += 1
        return State(
            timestep=0,
            construction=[problem_type, cls._seed_counter],
            code="",
            value=0.0,
            id=f"seed-{cls._seed_counter}",
        )


def make_state(
    state_id: str,
    value: float | None,
    direct_parent_id: str | None = None,
    ancestors: tuple[str, ...] = (),
    parents: list[dict] | None = None,
    timestep: int = 1,
) -> State:
    if parents is None:
        parents = []
        if direct_parent_id is not None:
            parents.append({"id": direct_parent_id, "timestep": 0})
        parents.extend({"id": ancestor_id, "timestep": 0} for ancestor_id in ancestors)
    return State(
        timestep=timestep,
        construction=[state_id],
        code=state_id,
        value=value,
        parents=parents,
        id=state_id,
    )


class RegularOnlySampler:
    def __init__(self) -> None:
        self.sample_states_calls = 0

    def sample_states(self, num_states: int) -> list[State]:
        self.sample_states_calls += 1
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


class SiblingRankBalancedSamplingTests(unittest.TestCase):
    def make_sampler(self, states: list[State]) -> PUCTSampler:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        sampler = PUCTSampler(
            file_path=os.path.join(tmp.name, "puct_sampler.json"),
            env_type=DummyEnv,
            max_buffer_size=100,
            batch_size=1,
            puct_c=0.0,
            topk_children=0,
        )
        sampler._states = list(states)
        sampler._initial_states = []
        sampler._n = {}
        sampler._m = {}
        sampler._T = 0
        return sampler

    def sample_rows_by_id(self, sampler: PUCTSampler) -> dict[str, dict]:
        columns, rows = sampler.get_sample_table()
        return {
            state.id: dict(zip(columns, row))
            for state, row in zip(sampler._last_sampled_states, rows)
        }

    def test_direct_parent_only_parentless_and_only_child_metadata(self) -> None:
        states = [
            make_state("seed", 10.0),
            make_state("a", 3.0, "parent-a", ancestors=("shared-ancestor",)),
            make_state("b", 2.0, "parent-b", ancestors=("shared-ancestor",)),
            make_state("missing-parent-id", 1.0, parents=[{"timestep": 0}]),
        ]
        sampler = self.make_sampler(states)

        picked = sampler.sample_states_sibling_rank_balanced(4)
        rows = self.sample_rows_by_id(sampler)

        self.assertEqual([state.id for state in picked], ["a", "seed", "b", "missing-parent-id"])
        self.assertEqual(rows["a"]["sibling_group_id"], "parent-a")
        self.assertEqual(rows["b"]["sibling_group_id"], "parent-b")
        self.assertEqual(rows["a"]["sibling_group_size"], 1)
        self.assertEqual(rows["b"]["sibling_group_size"], 1)
        self.assertEqual(rows["a"]["sibling_rank_bucket"], "only_child")
        self.assertEqual(rows["b"]["sibling_rank_bucket"], "only_child")
        self.assertEqual(rows["seed"]["sibling_rank_bucket"], "parentless_or_seed")
        self.assertEqual(
            rows["missing-parent-id"]["sibling_rank_bucket"],
            "parentless_or_seed",
        )
        self.assertEqual(sampler.get_sample_stats()["puct/sibling_rank_bucket_unique_count"], 1)

    def test_leader_contender_tail_are_first_pass_unique_buckets(self) -> None:
        states = [
            make_state("leader", 30.0, "parent"),
            make_state("contender", 20.0, "parent"),
            make_state("tail", 10.0, "parent"),
        ]
        sampler = self.make_sampler(states)

        picked = sampler.sample_states_sibling_rank_balanced(3)
        rows = self.sample_rows_by_id(sampler)
        stats = sampler.get_sample_stats()

        self.assertEqual([state.id for state in picked], ["leader", "contender", "tail"])
        self.assertEqual(rows["leader"]["sibling_value_rank"], 1)
        self.assertEqual(rows["contender"]["sibling_value_rank"], 2)
        self.assertEqual(rows["tail"]["sibling_value_rank"], 3)
        self.assertEqual(rows["leader"]["sibling_rank_bucket"], "sibling_leader")
        self.assertEqual(rows["contender"]["sibling_rank_bucket"], "sibling_contender")
        self.assertEqual(rows["tail"]["sibling_rank_bucket"], "sibling_tail")
        self.assertFalse(any(row["sibling_rank_balance_fallback"] for row in rows.values()))
        self.assertEqual(stats["puct/sibling_rank_bucket_unique_count"], 3)
        self.assertEqual(stats["puct/sibling_rank_balance_fallback_count"], 0)
        self.assertEqual(stats["puct/sibling_rank_balance_shortage_count"], 0)

    def test_all_tied_siblings_do_not_create_fake_diversity(self) -> None:
        states = [
            make_state("a", 5.0, "parent"),
            make_state("b", 5.0, "parent"),
            make_state("c", 5.0, "parent"),
        ]
        sampler = self.make_sampler(states)

        sampler.sample_states_sibling_rank_balanced(3)
        rows = self.sample_rows_by_id(sampler)
        stats = sampler.get_sample_stats()

        self.assertEqual({row["sibling_value_rank"] for row in rows.values()}, {1})
        self.assertEqual(
            {row["sibling_rank_bucket"] for row in rows.values()},
            {"tied_siblings"},
        )
        self.assertEqual(stats["puct/sibling_rank_bucket_unique_count"], 1)
        self.assertEqual(stats["puct/sibling_rank_balance_fallback_count"], 2)

    def test_tied_top_and_bottom_share_rank_and_bucket(self) -> None:
        states = [
            make_state("top-a", 10.0, "parent"),
            make_state("top-b", 10.0, "parent"),
            make_state("bottom-a", 5.0, "parent"),
            make_state("bottom-b", 5.0, "parent"),
        ]
        sampler = self.make_sampler(states)

        sampler.sample_states_sibling_rank_balanced(4)
        rows = self.sample_rows_by_id(sampler)

        for state_id in ("top-a", "top-b"):
            self.assertEqual(rows[state_id]["sibling_value_rank"], 1)
            self.assertEqual(rows[state_id]["sibling_rank_bucket"], "sibling_leader")
        for state_id in ("bottom-a", "bottom-b"):
            self.assertEqual(rows[state_id]["sibling_value_rank"], 2)
            self.assertEqual(rows[state_id]["sibling_rank_bucket"], "sibling_tail")

    def test_none_value_is_bottom_sibling_rank(self) -> None:
        metadata = _sibling_rank_metadata(
            [
                make_state("top", 10.0, "parent"),
                make_state("none-value", None, "parent"),
            ]
        )

        self.assertEqual(metadata["top"], ("parent", 2, 1, "sibling_leader"))
        self.assertEqual(metadata["none-value"], ("parent", 2, 2, "sibling_tail"))

    def test_parentless_is_not_first_pass_quota_but_can_fallback(self) -> None:
        states = [
            make_state("seed", 100.0),
            make_state("only-child", 1.0, "parent"),
        ]
        sampler = self.make_sampler(states)

        picked = sampler.sample_states_sibling_rank_balanced(2)
        rows = self.sample_rows_by_id(sampler)

        self.assertEqual([state.id for state in picked], ["only-child", "seed"])
        self.assertFalse(rows["only-child"]["sibling_rank_balance_fallback"])
        self.assertTrue(rows["seed"]["sibling_rank_balance_fallback"])
        self.assertEqual(rows["seed"]["sibling_rank_bucket"], "parentless_or_seed")

    def test_seed_timestep_with_direct_parent_is_not_first_pass_quota(self) -> None:
        states = [
            make_state("seeded-program", 100.0, "parent", timestep=-1),
            make_state("leader", 90.0, "parent"),
            make_state("tail", 80.0, "parent"),
        ]
        sampler = self.make_sampler(states)

        picked = sampler.sample_states_sibling_rank_balanced(3)
        rows = self.sample_rows_by_id(sampler)

        self.assertEqual([state.id for state in picked], ["leader", "tail", "seeded-program"])
        self.assertEqual(rows["seeded-program"]["sibling_rank_bucket"], "parentless_or_seed")
        self.assertTrue(rows["seeded-program"]["sibling_rank_balance_fallback"])
        self.assertFalse(rows["leader"]["sibling_rank_balance_fallback"])
        self.assertFalse(rows["tail"]["sibling_rank_balance_fallback"])

    def test_fallback_does_not_relax_full_lineage_blocking(self) -> None:
        states = [
            make_state("parent", 100.0, "root"),
            make_state("child", 90.0, "parent"),
            make_state("other", 80.0, "other-root"),
        ]
        sampler = self.make_sampler(states)

        picked = sampler.sample_states_sibling_rank_balanced(2)
        rows = self.sample_rows_by_id(sampler)

        self.assertEqual([state.id for state in picked], ["parent", "other"])
        self.assertNotIn("child", rows)
        self.assertTrue(rows["other"]["sibling_rank_balance_fallback"])

    def test_table_uses_sample_time_cache_after_archive_changes_and_flush(self) -> None:
        state = make_state("sampled", 10.0, "parent")
        sampler = self.make_sampler([state])

        sampler.sample_states_sibling_rank_balanced(1)
        sampler._states.append(make_state("new-sibling", 9.0, "parent"))
        sampler.flush()
        rows = self.sample_rows_by_id(sampler)

        self.assertEqual(rows["sampled"]["sibling_group_size"], 1)
        self.assertEqual(rows["sampled"]["sibling_value_rank"], 1)
        self.assertEqual(rows["sampled"]["sibling_rank_bucket"], "only_child")

    def test_num_states_less_than_or_equal_one_matches_normal_puct(self) -> None:
        states = [
            make_state("seed", 100.0),
            make_state("only-child", 1.0, "parent"),
        ]
        normal_sampler = self.make_sampler(states)
        balanced_sampler = self.make_sampler(states)

        normal_one = normal_sampler.sample_states(1)
        balanced_one = balanced_sampler.sample_states_sibling_rank_balanced(1)
        normal_zero = normal_sampler.sample_states(0)
        balanced_zero = balanced_sampler.sample_states_sibling_rank_balanced(0)

        self.assertEqual([state.id for state in balanced_one], [state.id for state in normal_one])
        self.assertEqual(balanced_zero, normal_zero)

    def test_sample_batch_flag_on_requires_sampler_api(self) -> None:
        cfg = CodexNoFinetuneConfig(
            env_type=DummyEnv,
            groups_per_batch=1,
            group_size=0,
            log_path="unused",
            sibling_rank_balanced_sampling=True,
        )

        with self.assertRaisesRegex(ValueError, "sample_states_sibling_rank_balanced"):
            asyncio.run(sample_batch(cfg, RegularOnlySampler(), 0))

    def test_sample_batch_flag_off_uses_regular_sampler(self) -> None:
        cfg = CodexNoFinetuneConfig(
            env_type=DummyEnv,
            groups_per_batch=0,
            group_size=0,
            log_path="unused",
            sibling_rank_balanced_sampling=False,
        )
        sampler = RegularOnlySampler()

        kept_results, metrics, all_results = asyncio.run(sample_batch(cfg, sampler, 0))

        self.assertEqual(sampler.sample_states_calls, 1)
        self.assertEqual(kept_results, [])
        self.assertEqual(all_results, [])
        self.assertEqual(metrics["codex/parent_states"], 0)


if __name__ == "__main__":
    unittest.main()
````

## Diff Stat

### Worktree Tracked Diff Stat

````text
 ttt_discover/codex_utils/discovery.py |   2 +
 ttt_discover/codex_utils/sampler.py   | 352 ++++++++++++++++++++++++++++++----
 ttt_discover/discovery.py             |   2 +
 ttt_discover/rl/codex_no_finetune.py  |  16 +-
 4 files changed, 337 insertions(+), 35 deletions(-)
````
## Raw Diff

<details>
<summary>Tracked worktree patch</summary>

````diff
diff --git a/ttt_discover/codex_utils/discovery.py b/ttt_discover/codex_utils/discovery.py
index b21a80f..cd2f264 100644
--- a/ttt_discover/codex_utils/discovery.py
+++ b/ttt_discover/codex_utils/discovery.py
@@ -50,6 +50,7 @@ class DiscoverConfig:
     codex_initial_program_paths: tuple[str, ...] = ()
     codex_initial_pool_paths: tuple[str, ...] = ()
     codex_autonomous: bool = False
+    codex_sibling_rank_balanced_sampling: bool = False
 
 
 def _run_codex_no_finetune(config: DiscoverConfig) -> None:
@@ -85,6 +86,7 @@ def _run_codex_no_finetune(config: DiscoverConfig) -> None:
         initial_program_paths=config.codex_initial_program_paths,
         initial_pool_paths=config.codex_initial_pool_paths,
         autonomous=config.codex_autonomous,
+        sibling_rank_balanced_sampling=config.codex_sibling_rank_balanced_sampling,
         wandb_project=config.wandb_project,
         wandb_name=experiment_name,
         log_path=log_path,
diff --git a/ttt_discover/codex_utils/sampler.py b/ttt_discover/codex_utils/sampler.py
index 5c5d4d1..d66d9ce 100644
--- a/ttt_discover/codex_utils/sampler.py
+++ b/ttt_discover/codex_utils/sampler.py
@@ -17,6 +17,74 @@ from ttt_discover.codex_utils.runtime import State, state_from_dict, to_json_ser
 logger = logging.getLogger(__name__)
 
 
+def _direct_parent_id(state: State) -> str | None:
+    parents = getattr(state, "parents", None) or []
+    if not parents or not isinstance(parents[0], dict):
+        return None
+    parent_id = parents[0].get("id")
+    if parent_id is None or parent_id == "":
+        return None
+    return str(parent_id)
+
+
+def _state_value_for_sibling_rank(state: State) -> float:
+    if state.value is None:
+        return float("-inf")
+    value = float(state.value)
+    if np.isnan(value):
+        return float("-inf")
+    return value
+
+
+def _has_seed_timestep(state: State) -> bool:
+    timestep = getattr(state, "timestep", None)
+    return (
+        not isinstance(timestep, bool)
+        and isinstance(timestep, (int, float))
+        and timestep < 0
+    )
+
+
+def _sibling_rank_metadata(
+    states: list[State],
+    initial_ids: set[str] | None = None,
+) -> dict[str, tuple[str | None, int, int | None, str]]:
+    initial_ids = initial_ids or set()
+    groups: dict[str, list[State]] = {}
+    metadata: dict[str, tuple[str | None, int, int | None, str]] = {}
+    for state in states:
+        parent_id = _direct_parent_id(state)
+        if state.id in initial_ids or _has_seed_timestep(state) or parent_id is None:
+            metadata[state.id] = (None, 0, None, "parentless_or_seed")
+            continue
+        groups.setdefault(parent_id, []).append(state)
+
+    for parent_id, siblings in groups.items():
+        group_size = len(siblings)
+        values = [_state_value_for_sibling_rank(state) for state in siblings]
+        value_ranks = {
+            value: rank
+            for rank, value in enumerate(sorted(set(values), reverse=True), start=1)
+        }
+        num_value_groups = len(value_ranks)
+
+        for state, value in zip(siblings, values):
+            rank = value_ranks[value]
+            if group_size == 1:
+                bucket = "only_child"
+            elif num_value_groups == 1:
+                bucket = "tied_siblings"
+            elif rank == 1:
+                bucket = "sibling_leader"
+            elif rank == num_value_groups:
+                bucket = "sibling_tail"
+            else:
+                bucket = "sibling_contender"
+            metadata[state.id] = (parent_id, group_size, rank, bucket)
+
+    return metadata
+
+
 @contextmanager
 def _file_lock(lock_path: str, *, poll_s: float = 0.05, stale_s: float = 600.0):
     while True:
@@ -375,6 +443,12 @@ class PUCTSampler(StateSampler):
         self._T: int = 0
         self._last_scale: float = 1.0
         self._last_puct_stats: list[tuple[int, float, float, float, float]] = []
+        self._last_sampled_sibling_group_ids: list[str | None] = []
+        self._last_sampled_sibling_group_sizes: list[int] = []
+        self._last_sampled_sibling_value_ranks: list[int | None] = []
+        self._last_sampled_sibling_rank_buckets: list[str] = []
+        self._last_sibling_rank_balance_fallback_flags: list[bool] = []
+        self._last_sibling_rank_balance_shortage_count: int = 0
         
         if resume_step is not None:
             self._load(resume_step)
@@ -460,6 +534,84 @@ class PUCTSampler(StateSampler):
         weights = (N - ranks).astype(np.float64)
         return weights / weights.sum()
 
+    def _ranked_puct_entries(
+        self,
+    ) -> tuple[list[tuple[float, float, State, int, float, float, float]], set[str]]:
+        initial_ids = {s.id for s in self._initial_states}
+        candidates = list(self._states)
+        if not candidates:
+            return [], initial_ids
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
+        return scores, initial_ids
+
+    def _set_last_sampled(
+        self,
+        picked: list[State],
+        top_scores: list[tuple[float, float, State, int, float, float, float]],
+        *,
+        fallback_flags: list[bool] | None = None,
+        shortage_count: int = 0,
+    ) -> None:
+        state_id_to_idx = {s.id: i for i, s in enumerate(self._states)}
+        initial_ids = {s.id for s in self._initial_states}
+        sibling_metadata = _sibling_rank_metadata(self._states, initial_ids)
+
+        self._last_sampled_states = picked
+        self._last_sampled_indices = [state_id_to_idx.get(s.id, -1) for s in picked]
+        self._last_puct_stats = [(t[3], t[4], t[5], t[6], t[0]) for t in top_scores]
+        self._last_sampled_sibling_group_ids = []
+        self._last_sampled_sibling_group_sizes = []
+        self._last_sampled_sibling_value_ranks = []
+        self._last_sampled_sibling_rank_buckets = []
+        for state in picked:
+            group_id, group_size, value_rank, bucket = sibling_metadata.get(
+                state.id,
+                (None, 0, None, "parentless_or_seed"),
+            )
+            self._last_sampled_sibling_group_ids.append(group_id)
+            self._last_sampled_sibling_group_sizes.append(group_size)
+            self._last_sampled_sibling_value_ranks.append(value_rank)
+            self._last_sampled_sibling_rank_buckets.append(bucket)
+        if fallback_flags is None or len(fallback_flags) != len(picked):
+            fallback_flags = [False] * len(picked)
+        self._last_sibling_rank_balance_fallback_flags = list(fallback_flags)
+        self._last_sibling_rank_balance_shortage_count = int(shortage_count)
+
+    def _set_last_sampled_without_puct(self, picked: list[State]) -> None:
+        self._last_sampled_states = picked
+        self._last_sampled_indices = []
+        self._last_puct_stats = [(0, 0.0, 0.0, 0.0, 0.0) for _ in picked]
+        self._last_sampled_sibling_group_ids = [None for _ in picked]
+        self._last_sampled_sibling_group_sizes = [0 for _ in picked]
+        self._last_sampled_sibling_value_ranks = [None for _ in picked]
+        self._last_sampled_sibling_rank_buckets = [
+            "parentless_or_seed" for _ in picked
+        ]
+        self._last_sibling_rank_balance_fallback_flags = [False] * len(picked)
+        self._last_sibling_rank_balance_shortage_count = 0
+
+    def _refresh_sampled_initials(self, picked: list[State], initial_ids: set[str]) -> None:
+        for s in picked:
+            if s.id in initial_ids:
+                self._refresh_random_construction(s)
+
     def _get_lineage(self, state: State) -> set[str]:
         lineage = {state.id}
         for p in (state.parents or []):
@@ -490,37 +642,16 @@ class PUCTSampler(StateSampler):
         return lineage
 
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
@@ -537,14 +668,80 @@ class PUCTSampler(StateSampler):
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
+    def sample_states_sibling_rank_balanced(self, num_states: int) -> list[State]:
+        """Sample by PUCT rank while preferring sibling value-rank buckets."""
+        scores, initial_ids = self._ranked_puct_entries()
+
+        if not scores and not self._states:
+            picked = [
+                create_initial_state(self.env_type, self.problem_type)
+                for _ in range(num_states)
+            ]
+            self._set_last_sampled_without_puct(picked)
+            return picked
+
+        if num_states <= 1:
+            top_scores = scores[:num_states]
+            picked = [t[2] for t in top_scores]
+            self._set_last_sampled(picked, top_scores)
+            self._refresh_sampled_initials(picked, initial_ids)
+            return picked
+
+        children_map = self._build_children_map()
+        sibling_metadata = _sibling_rank_metadata(self._states, initial_ids)
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
+            _group_id, _group_size, _rank, bucket = sibling_metadata.get(
+                state.id,
+                (None, 0, None, "parentless_or_seed"),
+            )
+            if (
+                state.id in blocked_ids
+                or bucket == "parentless_or_seed"
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
+            fallback_flags=fallback_flags,
+            shortage_count=shortage_count,
+        )
+        self._refresh_sampled_initials(picked, initial_ids)
 
         return picked
 
@@ -719,11 +916,19 @@ class PUCTSampler(StateSampler):
         sampled_values = [s.value for s in self._last_sampled_states]
         sampled_timesteps = [s.timestep for s in self._last_sampled_states]
         sampled_constr_lens = [len(s.construction) if hasattr(s, 'construction') and s.construction else 0 for s in self._last_sampled_states]
+        sampled_diversity_buckets = {
+            bucket
+            for bucket in self._last_sampled_sibling_rank_buckets
+            if bucket != "parentless_or_seed"
+        }
         stats = {
             "puct/buffer_size": len(self._states),
             "puct/sampled_size": len(self._last_sampled_states),
             "puct/T": self._T,
             "puct/scale_last": float(self._last_scale),
+            "puct/sibling_rank_bucket_unique_count": len(sampled_diversity_buckets),
+            "puct/sibling_rank_balance_fallback_count": int(sum(self._last_sibling_rank_balance_fallback_flags)),
+            "puct/sibling_rank_balance_shortage_count": self._last_sibling_rank_balance_shortage_count,
         }
         stats.update(_stats(buffer_values, "puct/buffer_value"))
         stats.update(_stats(buffer_timesteps, "puct/buffer_timestep"))
@@ -734,18 +939,97 @@ class PUCTSampler(StateSampler):
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
+            "sibling_group_id",
+            "sibling_group_size",
+            "sibling_value_rank",
+            "sibling_rank_bucket",
+            "sibling_rank_balance_fallback",
+        ]
         rows = []
         if not self._last_sampled_states:
             return columns, rows
         indices = self._last_sampled_indices if len(self._last_sampled_indices) == len(self._last_sampled_states) else [-1] * len(self._last_sampled_states)
         stats = self._last_puct_stats if len(self._last_puct_stats) == len(self._last_sampled_states) else [(0, 0.0, 0.0, 0.0, 0.0)] * len(self._last_sampled_states)
-        for idx, state, (n, Q, P, bonus, score) in zip(indices, self._last_sampled_states, stats):
+        sibling_group_ids = (
+            self._last_sampled_sibling_group_ids
+            if len(self._last_sampled_sibling_group_ids) == len(self._last_sampled_states)
+            else [None] * len(self._last_sampled_states)
+        )
+        sibling_group_sizes = (
+            self._last_sampled_sibling_group_sizes
+            if len(self._last_sampled_sibling_group_sizes) == len(self._last_sampled_states)
+            else [0] * len(self._last_sampled_states)
+        )
+        sibling_value_ranks = (
+            self._last_sampled_sibling_value_ranks
+            if len(self._last_sampled_sibling_value_ranks) == len(self._last_sampled_states)
+            else [None] * len(self._last_sampled_states)
+        )
+        sibling_rank_buckets = (
+            self._last_sampled_sibling_rank_buckets
+            if len(self._last_sampled_sibling_rank_buckets) == len(self._last_sampled_states)
+            else ["parentless_or_seed"] * len(self._last_sampled_states)
+        )
+        fallback_flags = (
+            self._last_sibling_rank_balance_fallback_flags
+            if len(self._last_sibling_rank_balance_fallback_flags) == len(self._last_sampled_states)
+            else [False] * len(self._last_sampled_states)
+        )
+        for (
+            idx,
+            state,
+            (n, Q, P, bonus, score),
+            sibling_group_id,
+            sibling_group_size,
+            sibling_value_rank,
+            sibling_rank_bucket,
+            fallback,
+        ) in zip(
+            indices,
+            self._last_sampled_states,
+            stats,
+            sibling_group_ids,
+            sibling_group_sizes,
+            sibling_value_ranks,
+            sibling_rank_buckets,
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
+                sibling_group_id,
+                sibling_group_size,
+                sibling_value_rank,
+                sibling_rank_bucket,
+                fallback,
+            ))
         return columns, rows
 
 
diff --git a/ttt_discover/discovery.py b/ttt_discover/discovery.py
index 6cdb434..6b14fb6 100644
--- a/ttt_discover/discovery.py
+++ b/ttt_discover/discovery.py
@@ -63,6 +63,7 @@ class DiscoverConfig:
     codex_initial_program_paths: tuple[str, ...] = ()
     codex_initial_pool_paths: tuple[str, ...] = ()
     codex_autonomous: bool = False
+    codex_sibling_rank_balanced_sampling: bool = False
 
 
 def init_ray(num_cpus_per_task: int, env_type: str):
@@ -146,6 +147,7 @@ async def discover_impl(config: DiscoverConfig):
             initial_program_paths=config.codex_initial_program_paths,
             initial_pool_paths=config.codex_initial_pool_paths,
             autonomous=config.codex_autonomous,
+            sibling_rank_balanced_sampling=config.codex_sibling_rank_balanced_sampling,
             wandb_project=config.wandb_project,
             wandb_name=config.experiment_name,
             log_path=log_path,
diff --git a/ttt_discover/rl/codex_no_finetune.py b/ttt_discover/rl/codex_no_finetune.py
index 18a0e9e..6888667 100644
--- a/ttt_discover/rl/codex_no_finetune.py
+++ b/ttt_discover/rl/codex_no_finetune.py
@@ -137,6 +137,7 @@ class CodexNoFinetuneConfig:
     initial_program_paths: tuple[str, ...] = ()
     initial_pool_paths: tuple[str, ...] = ()
     topk_children: int = 16
+    sibling_rank_balanced_sampling: bool = False
 
     # False is TTT Discover: non-auto, many short samples. True is AutoEvolve:
     # one Codex deep-dive workspace per sample with file edits and shell access.
@@ -820,7 +821,20 @@ async def sample_batch(
     i_batch: int,
 ) -> tuple[list[CandidateResult], dict[str, Any], list[CandidateResult]]:
     metrics: dict[str, Any] = {}
-    parent_states = sampler.sample_states(cfg.groups_per_batch)
+    if cfg.sibling_rank_balanced_sampling:
+        sample_sibling_rank_balanced = getattr(
+            sampler,
+            "sample_states_sibling_rank_balanced",
+            None,
+        )
+        if not callable(sample_sibling_rank_balanced):
+            raise ValueError(
+                "sibling_rank_balanced_sampling=True requires sampler public API "
+                "sample_states_sibling_rank_balanced(num_states)"
+            )
+        parent_states = sample_sibling_rank_balanced(cfg.groups_per_batch)
+    else:
+        parent_states = sampler.sample_states(cfg.groups_per_batch)
     semaphore = (
         asyncio.Semaphore(cfg.max_concurrent_requests)
         if cfg.max_concurrent_requests is not None
````
</details>

