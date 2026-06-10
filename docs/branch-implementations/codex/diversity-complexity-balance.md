# codex/diversity-complexity-balance

## Summary

为 state construction/code 计算复杂度分数并分为 simple/medium/complex；优先覆盖复杂度桶，缺口时回退 PUCT。

## Branch State

- Worktree: `/opt/tiger/discover-complexity-balance`
- HEAD: `2234282`
- Base used for comparison: `2234282`
- Commits ahead of base: `0`
- Commits behind base: `0`
- Group: `diversity_extra`
- Implementation location: `uncommitted worktree diff`

## Implemented Behavior

- The summary above is based on reading the changed Python code, added config fields, and sampler/prompt hooks in this branch.
- For sampler branches, the common pattern is to keep baseline PUCT ordering inside a chosen surface/bucket, while changing which surface/bucket gets the next parent pick.
- Tests detected: `14` test functions across untracked test files.

## Added Markers

### Config fields

- `codex_complexity_balanced_sampling`
- `complexity_balanced_sampling`

### Constants

- None

### Classes

- None

### Functions

- `state_complexity_score`
- `complexity_score_buckets`
- `_ranked_puct_entries`
- `_complexity_info_for_entries`
- `_is_seed_or_empty_complexity_state`
- `_set_last_sampled`
- `_set_last_sampled_without_puct`
- `_refresh_sampled_initials`
- `sample_states_complexity_balanced`

## Diff Summary

- Worktree tracked shortstat: `4 files changed, 363 insertions(+), 34 deletions(-)`
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

- `repro/gpu_mode/run_0608_complexity_balance.sh (904 bytes)`
- `repro/run_discovery.py (15849 bytes)`
- `tests/test_complexity_balanced_sampling.py (9984 bytes)`

### Detected Test Functions

- `tests/test_complexity_balanced_sampling.py::test_num_states_le_one_matches_normal_puct`
- `tests/test_complexity_balanced_sampling.py::test_construction_score_precedence`
- `tests/test_complexity_balanced_sampling.py::test_code_non_empty_line_fallback`
- `tests/test_complexity_balanced_sampling.py::test_score_zero_bucket`
- `tests/test_complexity_balanced_sampling.py::test_normal_bucket_mapping`
- `tests/test_complexity_balanced_sampling.py::test_all_equal_score_not_split`
- `tests/test_complexity_balanced_sampling.py::test_repeated_boundary_score_not_split`
- `tests/test_complexity_balanced_sampling.py::test_first_pass_unique_buckets`
- `tests/test_complexity_balanced_sampling.py::test_fallback_relaxes_bucket_uniqueness_only`
- `tests/test_complexity_balanced_sampling.py::test_seed_empty_not_hard_quota`
- `tests/test_complexity_balanced_sampling.py::test_parentless_state_is_seed_or_empty_bucket`
- `tests/test_complexity_balanced_sampling.py::test_full_lineage_blocking`
- `tests/test_complexity_balanced_sampling.py::test_sample_time_table_cache_after_update_and_flush`
- `tests/test_complexity_balanced_sampling.py::test_unsupported_sampler_api_value_error`

## Untracked File Snapshots

These files are not part of `git diff` yet, but they are part of the worktree implementation state recorded here.

### `repro/gpu_mode/run_0608_complexity_balance.sh`

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
  --experiment-name "${EXPERIMENT_NAME:-gpu-mode-trimul-complexity-balance-0608}" \
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
  --codex-complexity-balanced-sampling \
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
        "--codex-complexity-balanced-sampling",
        action="store_true",
        help="Use program-complexity balanced PUCT parent sampling.",
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
        print(f"codex_complexity_balanced_sampling={args.codex_complexity_balanced_sampling}")
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
            complexity_balanced_sampling=args.codex_complexity_balanced_sampling,
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
        codex_complexity_balanced_sampling=args.codex_complexity_balanced_sampling,
        codex_autonomous=args.codex_autonomous,
        remove_constant_reward_groups=remove_constant_reward_groups,
    )
    discover(config)


if __name__ == "__main__":
    main()
````

### `tests/test_complexity_balanced_sampling.py`

````python
from __future__ import annotations

import asyncio
import os
import tempfile
import unittest
import uuid
from types import SimpleNamespace

from ttt_discover.codex_utils.runtime import State
from ttt_discover.codex_utils.sampler import (
    PUCTSampler,
    complexity_score_buckets,
    state_complexity_score,
)
from ttt_discover.rl.codex_no_finetune import sample_batch


class DummyEnv:
    state_type = State
    _next_seed = 0

    @classmethod
    def create_initial_state(cls, problem_type: str) -> State:
        del problem_type
        cls._next_seed += 1
        return State(
            timestep=0,
            construction=[],
            code="",
            value=0.0,
            id=f"seed-{cls._next_seed}",
        )


def make_state(
    state_id: str,
    value: float,
    *,
    complexity: int | None = None,
    construction: object | None = None,
    code: str = "",
    parents: list[dict] | None = None,
) -> State:
    if construction is None:
        construction = list(range(complexity or 0))
    parent_refs = [{"id": "root"}] if parents is None else parents
    return State(
        timestep=1,
        construction=construction,  # type: ignore[arg-type]
        code=code,
        value=value,
        parents=parent_refs,
        id=state_id,
    )


class ComplexityBalancedSamplingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)

    def make_sampler(
        self,
        states: list[State],
        *,
        initial_states: list[State] | None = None,
    ) -> PUCTSampler:
        sampler = PUCTSampler(
            file_path=os.path.join(self.tmpdir.name, f"{uuid.uuid4()}.json"),
            env_type=DummyEnv,
            batch_size=0,
            puct_c=0.0,
            topk_children=0,
        )
        sampler._states = list(states)
        sampler._initial_states = list(initial_states or [])
        return sampler

    def test_num_states_le_one_matches_normal_puct(self) -> None:
        states = [
            make_state("a", 3.0, complexity=1),
            make_state("b", 2.0, complexity=5),
        ]
        normal = self.make_sampler(states)
        balanced = self.make_sampler(states)

        self.assertEqual(
            [state.id for state in normal.sample_states(1)],
            [state.id for state in balanced.sample_states_complexity_balanced(1)],
        )
        self.assertEqual(
            [state.id for state in normal.sample_states(0)],
            [state.id for state in balanced.sample_states_complexity_balanced(0)],
        )

    def test_construction_score_precedence(self) -> None:
        state = make_state(
            "a",
            1.0,
            construction=["x", "y"],
            code="line1\nline2\nline3\n",
        )
        self.assertEqual(state_complexity_score(state), 2)

    def test_code_non_empty_line_fallback(self) -> None:
        state = make_state(
            "a",
            1.0,
            construction=[],
            code="\nline1\n  \nline2\n",
        )
        self.assertEqual(state_complexity_score(state), 2)

    def test_score_zero_bucket(self) -> None:
        state = make_state("empty", 1.0, construction=[], code="")
        sampler = self.make_sampler([state])
        entries, initial_ids = sampler._ranked_puct_entries()
        scores_by_id, buckets_by_id = sampler._complexity_info_for_entries(
            entries,
            initial_ids,
        )
        self.assertEqual(scores_by_id["empty"], 0)
        self.assertEqual(buckets_by_id["empty"], "seed_or_empty")

    def test_normal_bucket_mapping(self) -> None:
        buckets = complexity_score_buckets([1, 2, 3, 4, 5, 6])
        self.assertEqual(buckets[1], "simple")
        self.assertEqual(buckets[2], "simple")
        self.assertEqual(buckets[3], "medium")
        self.assertEqual(buckets[4], "medium")
        self.assertEqual(buckets[5], "complex")
        self.assertEqual(buckets[6], "complex")

    def test_all_equal_score_not_split(self) -> None:
        self.assertEqual(
            complexity_score_buckets([3, 3, 3]),
            {3: "uniform_complexity"},
        )

    def test_repeated_boundary_score_not_split(self) -> None:
        buckets = complexity_score_buckets([1, 2, 2, 3])
        self.assertEqual(buckets[2], "medium")
        self.assertEqual(len({bucket for score, bucket in buckets.items() if score == 2}), 1)

    def test_first_pass_unique_buckets(self) -> None:
        states = [
            make_state("complex_1", 100.0, complexity=6),
            make_state("complex_2", 90.0, complexity=5),
            make_state("medium_1", 80.0, complexity=3),
            make_state("simple_1", 70.0, complexity=1),
            make_state("simple_2", 1.0, complexity=2),
            make_state("medium_2", 0.5, complexity=4),
        ]
        sampler = self.make_sampler(states)

        picked = sampler.sample_states_complexity_balanced(3)

        self.assertEqual([state.id for state in picked], ["complex_1", "medium_1", "simple_1"])
        self.assertEqual(
            sampler._last_sampled_complexity_buckets,
            ["complex", "medium", "simple"],
        )
        self.assertEqual(sampler._last_complexity_balance_fallback_flags, [False, False, False])

    def test_fallback_relaxes_bucket_uniqueness_only(self) -> None:
        states = [
            make_state("complex_1", 100.0, complexity=2),
            make_state("complex_2", 90.0, complexity=2),
            make_state("simple_1", 80.0, complexity=1),
            make_state("simple_2", 70.0, complexity=1),
        ]
        sampler = self.make_sampler(states)

        picked = sampler.sample_states_complexity_balanced(4)

        self.assertEqual(
            [state.id for state in picked],
            ["complex_1", "simple_1", "complex_2", "simple_2"],
        )
        self.assertEqual(
            sampler._last_complexity_balance_fallback_flags,
            [False, False, True, True],
        )
        self.assertEqual(sampler._last_complexity_balance_shortage_count, 0)

    def test_seed_empty_not_hard_quota(self) -> None:
        seed = make_state("seed", 100.0, construction=[], code="a\nb\nc\n", parents=[])
        states = [
            seed,
            make_state("complex", 90.0, complexity=6),
            make_state("medium", 80.0, complexity=3),
            make_state("simple", 70.0, complexity=1),
        ]
        sampler = self.make_sampler(states, initial_states=[seed])

        picked = sampler.sample_states_complexity_balanced(2)

        self.assertEqual([state.id for state in picked], ["complex", "medium"])
        self.assertNotIn("seed", [state.id for state in picked])
        self.assertEqual(sampler._last_sampled_complexity_buckets, ["complex", "medium"])

    def test_parentless_state_is_seed_or_empty_bucket(self) -> None:
        parentless = make_state(
            "parentless",
            100.0,
            construction=[],
            code="a\nb\nc\n",
            parents=[],
        )
        states = [
            parentless,
            make_state("complex", 90.0, complexity=6),
            make_state("medium", 80.0, complexity=3),
            make_state("simple", 70.0, complexity=1),
        ]
        sampler = self.make_sampler(states)

        entries, initial_ids = sampler._ranked_puct_entries()
        scores_by_id, buckets_by_id = sampler._complexity_info_for_entries(
            entries,
            initial_ids,
        )
        self.assertEqual(scores_by_id["parentless"], 3)
        self.assertEqual(buckets_by_id["parentless"], "seed_or_empty")

        picked = sampler.sample_states_complexity_balanced(4)
        self.assertEqual([state.id for state in picked], ["complex", "medium", "simple", "parentless"])
        self.assertEqual(sampler._last_complexity_balance_fallback_flags, [False, False, False, True])

    def test_full_lineage_blocking(self) -> None:
        parent = make_state("parent", 100.0, complexity=1)
        child = make_state(
            "child",
            90.0,
            complexity=4,
            parents=[{"id": "parent", "timestep": 1}],
        )
        independent = make_state("independent", 80.0, complexity=6)
        sampler = self.make_sampler([parent, child, independent])

        picked = sampler.sample_states_complexity_balanced(2)

        self.assertEqual([state.id for state in picked], ["parent", "independent"])
        self.assertNotIn("child", [state.id for state in picked])

    def test_sample_time_table_cache_after_update_and_flush(self) -> None:
        parent = make_state("parent", 100.0, complexity=1)
        other = make_state("other", 90.0, complexity=3)
        sampler = self.make_sampler([parent, other])

        sampler.sample_states_complexity_balanced(2)
        columns, rows_before = sampler.get_sample_table()
        score_idx = columns.index("complexity_score")
        bucket_idx = columns.index("complexity_bucket")
        cached_scores = [row[score_idx] for row in rows_before]
        cached_buckets = [row[bucket_idx] for row in rows_before]

        parent.construction = list(range(99))
        child = make_state("new_child", 1.0, complexity=9)
        sampler.update_states([child], [parent], save=False)
        sampler.flush(step=1)
        columns, rows_after = sampler.get_sample_table()

        self.assertEqual([row[score_idx] for row in rows_after], cached_scores)
        self.assertEqual([row[bucket_idx] for row in rows_after], cached_buckets)

    def test_unsupported_sampler_api_value_error(self) -> None:
        cfg = SimpleNamespace(
            complexity_balanced_sampling=True,
            groups_per_batch=1,
        )
        sampler = SimpleNamespace(sample_states=lambda count: [])

        with self.assertRaisesRegex(ValueError, "sample_states_complexity_balanced"):
            asyncio.run(sample_batch(cfg, sampler, 0))


if __name__ == "__main__":
    unittest.main()
````

## Diff Stat

### Worktree Tracked Diff Stat

````text
 ttt_discover/codex_utils/discovery.py |   2 +
 ttt_discover/codex_utils/sampler.py   | 377 +++++++++++++++++++++++++++++++---
 ttt_discover/discovery.py             |   2 +
 ttt_discover/rl/codex_no_finetune.py  |  16 +-
 4 files changed, 363 insertions(+), 34 deletions(-)
````
## Raw Diff

<details>
<summary>Tracked worktree patch</summary>

````diff
diff --git a/ttt_discover/codex_utils/discovery.py b/ttt_discover/codex_utils/discovery.py
index b21a80f..40388d5 100644
--- a/ttt_discover/codex_utils/discovery.py
+++ b/ttt_discover/codex_utils/discovery.py
@@ -49,6 +49,7 @@ class DiscoverConfig:
     codex_max_concurrent_requests: int | None = 4
     codex_initial_program_paths: tuple[str, ...] = ()
     codex_initial_pool_paths: tuple[str, ...] = ()
+    codex_complexity_balanced_sampling: bool = False
     codex_autonomous: bool = False
 
 
@@ -84,6 +85,7 @@ def _run_codex_no_finetune(config: DiscoverConfig) -> None:
         max_concurrent_requests=config.codex_max_concurrent_requests,
         initial_program_paths=config.codex_initial_program_paths,
         initial_pool_paths=config.codex_initial_pool_paths,
+        complexity_balanced_sampling=config.codex_complexity_balanced_sampling,
         autonomous=config.codex_autonomous,
         wandb_project=config.wandb_project,
         wandb_name=experiment_name,
diff --git a/ttt_discover/codex_utils/sampler.py b/ttt_discover/codex_utils/sampler.py
index 5c5d4d1..d2bdebe 100644
--- a/ttt_discover/codex_utils/sampler.py
+++ b/ttt_discover/codex_utils/sampler.py
@@ -66,6 +66,41 @@ def _read_json_or_default(path: str, default: Any) -> Any:
         return default
 
 
+def state_complexity_score(state: State) -> int:
+    construction = getattr(state, "construction", None)
+    if construction is not None:
+        try:
+            construction_len = len(construction)
+        except TypeError:
+            construction_len = 0
+        if construction_len > 0:
+            return int(construction_len)
+
+    code = getattr(state, "code", None)
+    if code:
+        return sum(1 for line in str(code).splitlines() if line.strip())
+    return 0
+
+
+def complexity_score_buckets(scores: list[int]) -> dict[int, str]:
+    unique_scores = sorted({int(score) for score in scores if score > 0})
+    if not unique_scores:
+        return {}
+    if len(unique_scores) == 1:
+        return {unique_scores[0]: "uniform_complexity"}
+    if len(unique_scores) == 2:
+        return {
+            unique_scores[0]: "simple",
+            unique_scores[1]: "complex",
+        }
+
+    buckets = ("simple", "medium", "complex")
+    return {
+        score: buckets[(idx * len(buckets)) // len(unique_scores)]
+        for idx, score in enumerate(unique_scores)
+    }
+
+
 class StateSampler(ABC):
     """Abstract base class for sampling states."""
 
@@ -375,6 +410,10 @@ class PUCTSampler(StateSampler):
         self._T: int = 0
         self._last_scale: float = 1.0
         self._last_puct_stats: list[tuple[int, float, float, float, float]] = []
+        self._last_sampled_complexity_scores: list[int] = []
+        self._last_sampled_complexity_buckets: list[str] = []
+        self._last_complexity_balance_fallback_flags: list[bool] = []
+        self._last_complexity_balance_shortage_count: int = 0
         
         if resume_step is not None:
             self._load(resume_step)
@@ -460,6 +499,146 @@ class PUCTSampler(StateSampler):
         weights = (N - ranks).astype(np.float64)
         return weights / weights.sum()
 
+    def _ranked_puct_entries(
+        self,
+    ) -> tuple[list[tuple[float, float, State, int, float, float, float]], set[str]]:
+        initial_ids = {s.id for s in self._initial_states}
+        candidates = list(self._states)
+
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
+    def _complexity_info_for_entries(
+        self,
+        entries: list[tuple[float, float, State, int, float, float, float]],
+        initial_ids: set[str],
+    ) -> tuple[dict[str, int], dict[str, str]]:
+        scores_by_id = {
+            entry[2].id: state_complexity_score(entry[2])
+            for entry in entries
+        }
+        positive_scores = []
+        for entry in entries:
+            state = entry[2]
+            score = scores_by_id.get(state.id, 0)
+            if not self._is_seed_or_empty_complexity_state(state, initial_ids, score):
+                positive_scores.append(score)
+        bucket_by_score = complexity_score_buckets(positive_scores)
+
+        buckets_by_id: dict[str, str] = {}
+        for entry in entries:
+            state = entry[2]
+            score = scores_by_id.get(state.id, 0)
+            if self._is_seed_or_empty_complexity_state(state, initial_ids, score):
+                buckets_by_id[state.id] = "seed_or_empty"
+            else:
+                buckets_by_id[state.id] = bucket_by_score.get(score, "uniform_complexity")
+        return scores_by_id, buckets_by_id
+
+    def _is_seed_or_empty_complexity_state(
+        self,
+        state: State,
+        initial_ids: set[str],
+        complexity_score: int,
+    ) -> bool:
+        return (
+            state.id in initial_ids
+            or not getattr(state, "parents", [])
+            or complexity_score <= 0
+        )
+
+    def _set_last_sampled(
+        self,
+        picked: list[State],
+        top_scores: list[tuple[float, float, State, int, float, float, float]],
+        *,
+        complexity_scores_by_id: dict[str, int] | None = None,
+        complexity_buckets_by_id: dict[str, str] | None = None,
+        fallback_flags: list[bool] | None = None,
+        shortage_count: int = 0,
+    ) -> None:
+        state_id_to_idx = {s.id: i for i, s in enumerate(self._states)}
+        if complexity_scores_by_id is None or complexity_buckets_by_id is None:
+            entries, initial_ids = self._ranked_puct_entries()
+            complexity_scores_by_id, complexity_buckets_by_id = self._complexity_info_for_entries(
+                entries,
+                initial_ids,
+            )
+
+        self._last_sampled_states = picked
+        self._last_sampled_indices = [state_id_to_idx.get(s.id, -1) for s in picked]
+        self._last_puct_stats = [(t[3], t[4], t[5], t[6], t[0]) for t in top_scores]
+        self._last_sampled_complexity_scores = [
+            int(complexity_scores_by_id.get(s.id, state_complexity_score(s)))
+            for s in picked
+        ]
+        self._last_sampled_complexity_buckets = [
+            complexity_buckets_by_id.get(s.id, "seed_or_empty")
+            for s in picked
+        ]
+        if fallback_flags is None or len(fallback_flags) != len(picked):
+            fallback_flags = [False] * len(picked)
+        self._last_complexity_balance_fallback_flags = list(fallback_flags)
+        self._last_complexity_balance_shortage_count = int(shortage_count)
+
+    def _set_last_sampled_without_puct(self, picked: list[State]) -> None:
+        scores_by_id = {
+            state.id: state_complexity_score(state)
+            for state in picked
+        }
+        bucket_by_score = complexity_score_buckets([
+            score for score in scores_by_id.values() if score > 0
+        ])
+        buckets_by_id = {
+            state.id: (
+                "seed_or_empty"
+                if self._is_seed_or_empty_complexity_state(
+                    state,
+                    set(),
+                    scores_by_id[state.id],
+                )
+                else bucket_by_score.get(scores_by_id[state.id], "uniform_complexity")
+            )
+            for state in picked
+        }
+        self._last_sampled_states = picked
+        self._last_sampled_indices = []
+        self._last_puct_stats = [(0, 0.0, 0.0, 0.0, 0.0) for _ in picked]
+        self._last_sampled_complexity_scores = [
+            scores_by_id.get(s.id, 0)
+            for s in picked
+        ]
+        self._last_sampled_complexity_buckets = [
+            buckets_by_id.get(s.id, "seed_or_empty")
+            for s in picked
+        ]
+        self._last_complexity_balance_fallback_flags = [False] * len(picked)
+        self._last_complexity_balance_shortage_count = 0
+
+    def _refresh_sampled_initials(self, picked: list[State], initial_ids: set[str]) -> None:
+        for s in picked:
+            if s.id in initial_ids:
+                self._refresh_random_construction(s)
+
     def _get_lineage(self, state: State) -> set[str]:
         lineage = {state.id}
         for p in (state.parents or []):
@@ -490,36 +669,20 @@ class PUCTSampler(StateSampler):
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
+        complexity_scores_by_id, complexity_buckets_by_id = self._complexity_info_for_entries(
+            scores,
+            initial_ids,
+        )
 
         if num_states > 1:
             children_map = self._build_children_map()
@@ -537,14 +700,87 @@ class PUCTSampler(StateSampler):
             top_scores = scores[:num_states]
             picked = [t[2] for t in top_scores]
 
-        state_id_to_idx = {s.id: i for i, s in enumerate(self._states)}
-        self._last_sampled_states = picked
-        self._last_sampled_indices = [state_id_to_idx.get(s.id, -1) for s in picked]
-        self._last_puct_stats = [(t[3], t[4], t[5], t[6], t[0]) for t in top_scores]
+        self._set_last_sampled(
+            picked,
+            top_scores,
+            complexity_scores_by_id=complexity_scores_by_id,
+            complexity_buckets_by_id=complexity_buckets_by_id,
+        )
+        self._refresh_sampled_initials(picked, initial_ids)
 
-        for s in picked:
-            if s.id in initial_ids:
-                self._refresh_random_construction(s)
+        return picked
+
+    def sample_states_complexity_balanced(self, num_states: int) -> list[State]:
+        """Sample by PUCT rank while preferring different program-complexity buckets."""
+        if num_states <= 1:
+            return self.sample_states(num_states)
+
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
+        complexity_scores_by_id, complexity_buckets_by_id = self._complexity_info_for_entries(
+            scores,
+            initial_ids,
+        )
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
+            complexity_score = complexity_scores_by_id.get(state.id, 0)
+            bucket = complexity_buckets_by_id.get(state.id, "seed_or_empty")
+            if (
+                state.id in blocked_ids
+                or state.id in picked_ids
+                or state.id in initial_ids
+                or complexity_score <= 0
+                or bucket == "seed_or_empty"
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
+            complexity_scores_by_id=complexity_scores_by_id,
+            complexity_buckets_by_id=complexity_buckets_by_id,
+            fallback_flags=fallback_flags,
+            shortage_count=shortage_count,
+        )
+        self._refresh_sampled_initials(picked, initial_ids)
 
         return picked
 
@@ -719,12 +955,33 @@ class PUCTSampler(StateSampler):
         sampled_values = [s.value for s in self._last_sampled_states]
         sampled_timesteps = [s.timestep for s in self._last_sampled_states]
         sampled_constr_lens = [len(s.construction) if hasattr(s, 'construction') and s.construction else 0 for s in self._last_sampled_states]
+        sampled_complexity_scores = (
+            self._last_sampled_complexity_scores
+            if len(self._last_sampled_complexity_scores) == len(self._last_sampled_states)
+            else []
+        )
+        sampled_complexity_buckets = (
+            self._last_sampled_complexity_buckets
+            if len(self._last_sampled_complexity_buckets) == len(sampled_complexity_scores)
+            else []
+        )
+        unique_positive_complexity_buckets = {
+            bucket
+            for score, bucket in zip(sampled_complexity_scores, sampled_complexity_buckets)
+            if score > 0 and bucket != "seed_or_empty"
+        }
         stats = {
             "puct/buffer_size": len(self._states),
             "puct/sampled_size": len(self._last_sampled_states),
             "puct/T": self._T,
             "puct/scale_last": float(self._last_scale),
+            "puct/complexity_bucket_unique_count": len(unique_positive_complexity_buckets),
+            "puct/complexity_balance_fallback_count": int(sum(self._last_complexity_balance_fallback_flags)),
+            "puct/complexity_balance_shortage_count": self._last_complexity_balance_shortage_count,
         }
+        if sampled_complexity_scores:
+            stats["puct/complexity_score_sampled_mean"] = float(np.mean(sampled_complexity_scores))
+            stats["puct/complexity_score_sampled_max"] = int(np.max(sampled_complexity_scores))
         stats.update(_stats(buffer_values, "puct/buffer_value"))
         stats.update(_stats(buffer_timesteps, "puct/buffer_timestep"))
         stats.update(_stats(buffer_constr_lens, "puct/buffer_construction_len"))
@@ -734,18 +991,72 @@ class PUCTSampler(StateSampler):
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
+            "complexity_score",
+            "complexity_bucket",
+            "complexity_balance_fallback",
+        ]
         rows = []
         if not self._last_sampled_states:
             return columns, rows
         indices = self._last_sampled_indices if len(self._last_sampled_indices) == len(self._last_sampled_states) else [-1] * len(self._last_sampled_states)
         stats = self._last_puct_stats if len(self._last_puct_stats) == len(self._last_sampled_states) else [(0, 0.0, 0.0, 0.0, 0.0)] * len(self._last_sampled_states)
-        for idx, state, (n, Q, P, bonus, score) in zip(indices, self._last_sampled_states, stats):
+        complexity_scores = (
+            self._last_sampled_complexity_scores
+            if len(self._last_sampled_complexity_scores) == len(self._last_sampled_states)
+            else [0] * len(self._last_sampled_states)
+        )
+        complexity_buckets = (
+            self._last_sampled_complexity_buckets
+            if len(self._last_sampled_complexity_buckets) == len(self._last_sampled_states)
+            else ["seed_or_empty"] * len(self._last_sampled_states)
+        )
+        fallback_flags = (
+            self._last_complexity_balance_fallback_flags
+            if len(self._last_complexity_balance_fallback_flags) == len(self._last_sampled_states)
+            else [False] * len(self._last_sampled_states)
+        )
+        for idx, state, (n, Q, P, bonus, score), complexity_score, complexity_bucket, fallback in zip(
+            indices,
+            self._last_sampled_states,
+            stats,
+            complexity_scores,
+            complexity_buckets,
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
+                complexity_score,
+                complexity_bucket,
+                fallback,
+            ))
         return columns, rows
 
 
diff --git a/ttt_discover/discovery.py b/ttt_discover/discovery.py
index 6cdb434..81ab8ee 100644
--- a/ttt_discover/discovery.py
+++ b/ttt_discover/discovery.py
@@ -62,6 +62,7 @@ class DiscoverConfig:
     codex_max_concurrent_requests: int | None = 4
     codex_initial_program_paths: tuple[str, ...] = ()
     codex_initial_pool_paths: tuple[str, ...] = ()
+    codex_complexity_balanced_sampling: bool = False
     codex_autonomous: bool = False
 
 
@@ -145,6 +146,7 @@ async def discover_impl(config: DiscoverConfig):
             max_concurrent_requests=config.codex_max_concurrent_requests,
             initial_program_paths=config.codex_initial_program_paths,
             initial_pool_paths=config.codex_initial_pool_paths,
+            complexity_balanced_sampling=config.codex_complexity_balanced_sampling,
             autonomous=config.codex_autonomous,
             wandb_project=config.wandb_project,
             wandb_name=config.experiment_name,
diff --git a/ttt_discover/rl/codex_no_finetune.py b/ttt_discover/rl/codex_no_finetune.py
index 18a0e9e..e71b741 100644
--- a/ttt_discover/rl/codex_no_finetune.py
+++ b/ttt_discover/rl/codex_no_finetune.py
@@ -137,6 +137,7 @@ class CodexNoFinetuneConfig:
     initial_program_paths: tuple[str, ...] = ()
     initial_pool_paths: tuple[str, ...] = ()
     topk_children: int = 16
+    complexity_balanced_sampling: bool = False
 
     # False is TTT Discover: non-auto, many short samples. True is AutoEvolve:
     # one Codex deep-dive workspace per sample with file edits and shell access.
@@ -820,7 +821,20 @@ async def sample_batch(
     i_batch: int,
 ) -> tuple[list[CandidateResult], dict[str, Any], list[CandidateResult]]:
     metrics: dict[str, Any] = {}
-    parent_states = sampler.sample_states(cfg.groups_per_batch)
+    if cfg.complexity_balanced_sampling:
+        sample_complexity_balanced = getattr(
+            sampler,
+            "sample_states_complexity_balanced",
+            None,
+        )
+        if not callable(sample_complexity_balanced):
+            raise ValueError(
+                "complexity_balanced_sampling=True requires sampler public API "
+                "sample_states_complexity_balanced(num_states)"
+            )
+        parent_states = sample_complexity_balanced(cfg.groups_per_batch)
+    else:
+        parent_states = sampler.sample_states(cfg.groups_per_batch)
     semaphore = (
         asyncio.Semaphore(cfg.max_concurrent_requests)
         if cfg.max_concurrent_requests is not None
````
</details>

