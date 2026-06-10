# codex/diversity-subtree-load-balance

## Summary

计算候选在 archive 内的 subtree descendant count，并按 subtree load bucket 选样，避免只扩展超大或超小子树。

## Branch State

- Worktree: `/opt/tiger/discover-subtree-load-balance`
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

- `codex_subtree_load_balanced_sampling`
- `subtree_load_balanced_sampling`

### Constants

- None

### Classes

- None

### Functions

- `_compute_subtree_descendant_counts`
- `_subtree_load_bucket`
- `sample_states_subtree_load_balanced`

## Diff Summary

- Worktree tracked shortstat: `4 files changed, 128 insertions(+), 2 deletions(-)`
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

- `repro/gpu_mode/run_0608_subtree_load_balance.sh (1148 bytes)`
- `repro/run_discovery.py (15397 bytes)`
- `tests/test_subtree_load_balanced_sampling.py (13133 bytes)`

### Detected Test Functions

- `tests/test_subtree_load_balanced_sampling.py::test_runner_flag_uses_expected_sampler_api`
- `tests/test_subtree_load_balanced_sampling.py::test_num_states_at_most_one_matches_ordinary_puct`
- `tests/test_subtree_load_balanced_sampling.py::test_descendant_counts_and_buckets`
- `tests/test_subtree_load_balanced_sampling.py::test_bucket_uses_full_parent_chain`
- `tests/test_subtree_load_balanced_sampling.py::test_duplicate_parent_refs_do_not_double_tally`
- `tests/test_subtree_load_balanced_sampling.py::test_first_pass_balances_buckets`
- `tests/test_subtree_load_balanced_sampling.py::test_fallback_fills_without_repeating`
- `tests/test_subtree_load_balanced_sampling.py::test_full_lineage_blocking_applies_in_both_passes`
- `tests/test_subtree_load_balanced_sampling.py::test_seed_and_parentless_states_are_bucketed_by_visible_descendants`
- `tests/test_subtree_load_balanced_sampling.py::test_malformed_and_missing_parent_ids_are_ignored`
- `tests/test_subtree_load_balanced_sampling.py::test_sampling_metadata_table_and_json_schema`

## Untracked File Snapshots

These files are not part of `git diff` yet, but they are part of the worktree implementation state recorded here.

### `repro/gpu_mode/run_0608_subtree_load_balance.sh`

````bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-gpu-mode-trimul-subtree-load-balance-0608}"
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
  --codex-subtree-load-balanced-sampling
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
        "--codex-subtree-load-balanced-sampling",
        action="store_true",
        help="Use subtree-load balanced PUCT parent sampling.",
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
            "codex_subtree_load_balanced_sampling="
            f"{args.codex_subtree_load_balanced_sampling}"
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
            subtree_load_balanced_sampling=(
                args.codex_subtree_load_balanced_sampling
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
        codex_subtree_load_balanced_sampling=(
            args.codex_subtree_load_balanced_sampling
        ),
        codex_autonomous=args.codex_autonomous,
        remove_constant_reward_groups=remove_constant_reward_groups,
    )
    discover(config)


if __name__ == "__main__":
    main()
````

### `tests/test_subtree_load_balanced_sampling.py`

````python
from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path

from ttt_discover.codex_utils.runtime import State
from ttt_discover.codex_utils.sampler import PUCTSampler
from ttt_discover.rl.codex_no_finetune import CodexNoFinetuneConfig, sample_batch


class DummyEnv:
    state_type = State

    @classmethod
    def create_initial_state(cls, problem_type: str) -> State:
        del problem_type
        return State(
            timestep=-1,
            construction=["initial"],
            code="",
            value=0.0,
            id="initial",
        )


def parent_refs(*ids: str) -> list[dict]:
    return [{"id": sid, "timestep": 0} for sid in ids]


def make_state(
    sid: str,
    value: float,
    parents: list[dict] | None = None,
    *,
    timestep: int = 0,
) -> State:
    return State(
        timestep=timestep,
        construction=[sid],
        code=f"# {sid}",
        value=value,
        parents=parents or [],
        parent_values=[],
        id=sid,
    )


class RecordingSampler:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def sample_states(self, num_states: int) -> list[State]:
        self.calls.append(("ordinary", num_states))
        return []

    def update_states(
        self,
        states: list[State],
        parent_states: list[State],
        save: bool = True,
        step: int | None = None,
    ) -> None:
        del states, parent_states, save, step

    def flush(self, step: int | None = None) -> None:
        del step


class NonCallableSampler(RecordingSampler):
    sample_states_subtree_load_balanced = None


class SubtreeLoadBalancedSamplingTest(unittest.TestCase):
    def make_sampler(
        self,
        states: list[State],
        *,
        initial_states: list[State] | None = None,
    ) -> PUCTSampler:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        sampler = PUCTSampler(
            file_path=os.path.join(tmp.name, "puct_sampler.json"),
            env_type=DummyEnv,
            batch_size=1,
        )
        sampler._states = list(states)
        sampler._initial_states = list(initial_states or [])
        sampler._n = {}
        sampler._m = {}
        sampler._T = 0
        sampler._last_sampled_states = []
        sampler._last_sampled_indices = []
        sampler._last_puct_stats = []
        return sampler

    def test_runner_flag_uses_expected_sampler_api(self) -> None:
        cfg = CodexNoFinetuneConfig(
            env_type=DummyEnv,
            groups_per_batch=2,
            group_size=1,
            subtree_load_balanced_sampling=False,
        )
        sampler = RecordingSampler()
        asyncio.run(sample_batch(cfg, sampler, 0))
        self.assertEqual(sampler.calls, [("ordinary", 2)])

        enabled_cfg = CodexNoFinetuneConfig(
            env_type=DummyEnv,
            groups_per_batch=2,
            group_size=1,
            subtree_load_balanced_sampling=True,
        )
        with self.assertRaisesRegex(
            ValueError,
            "sample_states_subtree_load_balanced",
        ):
            asyncio.run(sample_batch(enabled_cfg, RecordingSampler(), 0))
        with self.assertRaisesRegex(
            ValueError,
            "sample_states_subtree_load_balanced",
        ):
            asyncio.run(sample_batch(enabled_cfg, NonCallableSampler(), 0))

    def test_num_states_at_most_one_matches_ordinary_puct(self) -> None:
        states = [
            make_state("A", 10.0),
            make_state("B", 5.0),
            make_state("C", 1.0),
        ]
        ordinary = self.make_sampler(states)
        balanced = self.make_sampler([make_state("A", 10.0), make_state("B", 5.0), make_state("C", 1.0)])

        self.assertEqual(
            [state.id for state in balanced.sample_states_subtree_load_balanced(0)],
            [state.id for state in ordinary.sample_states(0)],
        )

        ordinary = self.make_sampler(states)
        balanced = self.make_sampler([make_state("A", 10.0), make_state("B", 5.0), make_state("C", 1.0)])
        self.assertEqual(
            [state.id for state in balanced.sample_states_subtree_load_balanced(1)],
            [state.id for state in ordinary.sample_states(1)],
        )

    def test_descendant_counts_and_buckets(self) -> None:
        states = [
            make_state("A", 10.0),
            make_state("B", 9.0, parent_refs("A")),
            make_state("C", 8.0, parent_refs("B", "A")),
            make_state("D", 7.0, parent_refs("C", "B", "A")),
        ]
        sampler = self.make_sampler(states)
        counts = sampler._compute_subtree_descendant_counts()
        self.assertEqual(counts["A"], 3)
        self.assertEqual(counts["B"], 2)
        self.assertEqual(counts["C"], 1)
        self.assertEqual(counts["D"], 0)
        self.assertEqual(sampler._subtree_load_bucket(counts["A"]), "subtree_light")
        self.assertEqual(sampler._subtree_load_bucket(counts["B"]), "subtree_light")
        self.assertEqual(sampler._subtree_load_bucket(counts["C"]), "subtree_light")
        self.assertEqual(sampler._subtree_load_bucket(counts["D"]), "subtree_leaf")

        states.append(make_state("E", 6.0, parent_refs("D", "C", "B", "A")))
        sampler = self.make_sampler(states)
        counts = sampler._compute_subtree_descendant_counts()
        self.assertEqual(counts["A"], 4)
        self.assertEqual(sampler._subtree_load_bucket(counts["A"]), "subtree_heavy")

    def test_bucket_uses_full_parent_chain(self) -> None:
        states = [
            make_state("A", 10.0),
            make_state("B", 9.0, parent_refs("A")),
            make_state("C", 8.0, parent_refs("B", "A")),
            make_state("D", 7.0, parent_refs("C", "B", "A")),
            make_state("E", 6.0, parent_refs("D", "C", "B", "A")),
        ]
        sampler = self.make_sampler(states)
        counts = sampler._compute_subtree_descendant_counts()
        self.assertEqual(counts["A"], 4)
        self.assertEqual(sampler._subtree_load_bucket(counts["A"]), "subtree_heavy")

    def test_duplicate_parent_refs_do_not_double_tally(self) -> None:
        states = [
            make_state("A", 10.0),
            make_state("B", 9.0, parent_refs("A", "A", "A")),
        ]
        sampler = self.make_sampler(states)
        counts = sampler._compute_subtree_descendant_counts()
        self.assertEqual(counts["A"], 1)
        self.assertEqual(sampler._subtree_load_bucket(counts["A"]), "subtree_light")

    def test_first_pass_balances_buckets(self) -> None:
        states = [
            make_state("H1", 100.0),
            make_state("H2", 99.0),
            make_state("L", 50.0),
            make_state("F", 40.0),
        ]
        states.extend(make_state(f"H1D{i}", 1.0 - i, parent_refs("H1")) for i in range(4))
        states.extend(make_state(f"H2D{i}", -10.0 - i, parent_refs("H2")) for i in range(4))
        states.append(make_state("LD0", -20.0, parent_refs("L")))
        sampler = self.make_sampler(states)

        picked = sampler.sample_states_subtree_load_balanced(3)
        picked_ids = [state.id for state in picked]
        self.assertEqual(picked_ids, ["H1", "L", "F"])
        counts = sampler._compute_subtree_descendant_counts()
        buckets = [
            sampler._subtree_load_bucket(counts[state.id])
            for state in picked
        ]
        self.assertEqual(buckets, ["subtree_heavy", "subtree_light", "subtree_leaf"])

    def test_fallback_fills_without_repeating(self) -> None:
        states = [make_state(f"S{i}", 10.0 - i) for i in range(5)]
        sampler = self.make_sampler(states)
        picked = sampler.sample_states_subtree_load_balanced(4)
        picked_ids = [state.id for state in picked]
        self.assertEqual(picked_ids, ["S0", "S1", "S2", "S3"])
        self.assertEqual(len(picked_ids), len(set(picked_ids)))

    def test_full_lineage_blocking_applies_in_both_passes(self) -> None:
        states = [
            make_state("A", 100.0),
            make_state("B", 90.0, parent_refs("A")),
            make_state("C", 80.0),
        ]
        sampler = self.make_sampler(states)
        picked_ids = [state.id for state in sampler.sample_states_subtree_load_balanced(2)]
        self.assertIn("A", picked_ids)
        self.assertIn("C", picked_ids)
        self.assertNotEqual(set(picked_ids), {"A", "B"})

        states = [
            make_state("H1", 100.0),
            make_state("H2", 99.0),
            make_state("F", 10.0),
        ]
        states.extend(make_state(f"H1D{i}", 1.0 - i, parent_refs("H1")) for i in range(4))
        states.extend(make_state(f"H2D{i}", -10.0 - i, parent_refs("H2")) for i in range(4))
        sampler = self.make_sampler(states)
        picked_ids = [state.id for state in sampler.sample_states_subtree_load_balanced(4)]
        self.assertIn("H1", picked_ids)
        self.assertIn("H2", picked_ids)
        self.assertFalse({"H1", "H1D0"} <= set(picked_ids))
        self.assertFalse({"H2", "H2D0"} <= set(picked_ids))

    def test_seed_and_parentless_states_are_bucketed_by_visible_descendants(self) -> None:
        seed_leaf = make_state("seed_leaf", 50.0, timestep=-1)
        seed_busy = make_state("seed_busy", 40.0, timestep=-1)
        states = [seed_leaf, seed_busy]
        states.extend(make_state(f"D{i}", 10.0 - i, parent_refs("seed_busy")) for i in range(4))
        sampler = self.make_sampler(states, initial_states=[seed_leaf, seed_busy])
        counts = sampler._compute_subtree_descendant_counts()
        self.assertEqual(counts["seed_leaf"], 0)
        self.assertEqual(sampler._subtree_load_bucket(counts["seed_leaf"]), "subtree_leaf")
        self.assertEqual(counts["seed_busy"], 4)
        self.assertEqual(sampler._subtree_load_bucket(counts["seed_busy"]), "subtree_heavy")

    def test_malformed_and_missing_parent_ids_are_ignored(self) -> None:
        states = [
            make_state("A", 10.0),
            make_state(
                "B",
                9.0,
                [
                    {},
                    {"id": None, "timestep": 0},
                    {"id": "ghost", "timestep": 0},
                    "bad-ref",
                    {"id": "A", "timestep": 0},
                    {"id": "A", "timestep": 0},
                ],
            ),
        ]
        sampler = self.make_sampler(states)
        counts = sampler._compute_subtree_descendant_counts()
        self.assertEqual(counts, {"A": 1, "B": 0})
        picked = sampler.sample_states_subtree_load_balanced(2)
        self.assertLessEqual({state.id for state in picked}, {"A", "B"})

    def test_sampling_metadata_table_and_json_schema(self) -> None:
        states = [
            make_state("A", 100.0),
            make_state("B", 90.0),
            make_state("C", 80.0),
            make_state("D", 70.0),
        ]
        ordinary = self.make_sampler(states)
        ordinary.sample_states(2)
        ordinary_columns, _ = ordinary.get_sample_table()

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        sampler = PUCTSampler(
            file_path=os.path.join(tmp.name, "puct_sampler.json"),
            env_type=DummyEnv,
            batch_size=1,
        )
        sampler._states = [make_state("A", 100.0), make_state("B", 90.0), make_state("C", 80.0), make_state("D", 70.0)]
        sampler._initial_states = []
        picked = sampler.sample_states_subtree_load_balanced(3)

        self.assertEqual(sampler._last_sampled_states, picked)
        self.assertEqual(len(sampler._last_sampled_indices), len(picked))
        self.assertEqual(len(sampler._last_puct_stats), len(picked))
        for idx, state in zip(sampler._last_sampled_indices, picked):
            self.assertEqual(sampler._states[idx].id, state.id)

        columns, rows = sampler.get_sample_table()
        self.assertEqual(columns, ordinary_columns)
        self.assertEqual(len(rows), len(picked))

        sampler.flush(step=1)
        store = json.loads(
            Path(tmp.name, "puct_sampler_step_000001.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            set(store),
            {"step", "states", "initial_states", "puct_n", "puct_m", "puct_T"},
        )

        old_tmp = tempfile.TemporaryDirectory()
        self.addCleanup(old_tmp.cleanup)
        old_store = {
            "step": 0,
            "states": [make_state("old", 1.0).to_dict()],
            "initial_states": [make_state("old", 1.0).to_dict()],
            "puct_n": {},
            "puct_m": {},
            "puct_T": 0,
        }
        Path(old_tmp.name, "puct_sampler_step_000000.json").write_text(
            json.dumps(old_store),
            encoding="utf-8",
        )
        loaded = PUCTSampler(
            file_path=os.path.join(old_tmp.name, "puct_sampler.json"),
            env_type=DummyEnv,
            resume_step=0,
        )
        self.assertEqual([state.id for state in loaded._states], ["old"])


if __name__ == "__main__":
    unittest.main()
````

## Diff Stat

### Worktree Tracked Diff Stat

````text
 ttt_discover/codex_utils/discovery.py |   4 ++
 ttt_discover/codex_utils/sampler.py   | 110 +++++++++++++++++++++++++++++++++-
 ttt_discover/discovery.py             |   4 ++
 ttt_discover/rl/codex_no_finetune.py  |  12 +++-
 4 files changed, 128 insertions(+), 2 deletions(-)
````
## Raw Diff

<details>
<summary>Tracked worktree patch</summary>

````diff
diff --git a/ttt_discover/codex_utils/discovery.py b/ttt_discover/codex_utils/discovery.py
index b21a80f..b77ec1a 100644
--- a/ttt_discover/codex_utils/discovery.py
+++ b/ttt_discover/codex_utils/discovery.py
@@ -49,6 +49,7 @@ class DiscoverConfig:
     codex_max_concurrent_requests: int | None = 4
     codex_initial_program_paths: tuple[str, ...] = ()
     codex_initial_pool_paths: tuple[str, ...] = ()
+    codex_subtree_load_balanced_sampling: bool = False
     codex_autonomous: bool = False
 
 
@@ -84,6 +85,9 @@ def _run_codex_no_finetune(config: DiscoverConfig) -> None:
         max_concurrent_requests=config.codex_max_concurrent_requests,
         initial_program_paths=config.codex_initial_program_paths,
         initial_pool_paths=config.codex_initial_pool_paths,
+        subtree_load_balanced_sampling=(
+            config.codex_subtree_load_balanced_sampling
+        ),
         autonomous=config.codex_autonomous,
         wandb_project=config.wandb_project,
         wandb_name=experiment_name,
diff --git a/ttt_discover/codex_utils/sampler.py b/ttt_discover/codex_utils/sampler.py
index 5c5d4d1..d7d20c9 100644
--- a/ttt_discover/codex_utils/sampler.py
+++ b/ttt_discover/codex_utils/sampler.py
@@ -463,7 +463,7 @@ class PUCTSampler(StateSampler):
     def _get_lineage(self, state: State) -> set[str]:
         lineage = {state.id}
         for p in (state.parents or []):
-            if p.get("id"):
+            if isinstance(p, dict) and p.get("id"):
                 lineage.add(str(p["id"]))
         return lineage
 
@@ -471,6 +471,8 @@ class PUCTSampler(StateSampler):
         children: dict[str, set[str]] = {}
         for s in self._states:
             for p in (s.parents or []):
+                if not isinstance(p, dict):
+                    continue
                 pid = p.get("id")
                 if pid:
                     children.setdefault(str(pid), set()).add(s.id)
@@ -489,6 +491,31 @@ class PUCTSampler(StateSampler):
                     queue.append(child_id)
         return lineage
 
+    def _compute_subtree_descendant_counts(self) -> dict[str, int]:
+        state_ids = {str(s.id) for s in self._states}
+        counts = {sid: 0 for sid in state_ids}
+        for state in self._states:
+            parent_ids: set[str] = set()
+            for parent_ref in state.parents or []:
+                if not isinstance(parent_ref, dict):
+                    continue
+                pid = parent_ref.get("id")
+                if pid is None:
+                    continue
+                pid = str(pid)
+                if pid in state_ids and pid != state.id:
+                    parent_ids.add(pid)
+            for pid in parent_ids:
+                counts[pid] += 1
+        return counts
+
+    def _subtree_load_bucket(self, descendant_count: int) -> str:
+        if descendant_count == 0:
+            return "subtree_leaf"
+        if descendant_count <= 3:
+            return "subtree_light"
+        return "subtree_heavy"
+
     def sample_states(self, num_states: int) -> list[State]:
         initial_ids = {s.id for s in self._initial_states}
         candidates = list(self._states)
@@ -548,6 +575,87 @@ class PUCTSampler(StateSampler):
 
         return picked
 
+    def sample_states_subtree_load_balanced(self, num_states: int) -> list[State]:
+        if num_states <= 1:
+            return self.sample_states(num_states)
+
+        initial_ids = {s.id for s in self._initial_states}
+        candidates = list(self._states)
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
+        descendant_counts = self._compute_subtree_descendant_counts()
+        picked: list[State] = []
+        top_scores = []
+        picked_ids: set[str] = set()
+        blocked_ids: set[str] = set()
+        used_buckets: set[str] = set()
+
+        for entry in scores:
+            state = entry[2]
+            if state.id in blocked_ids:
+                continue
+            bucket = self._subtree_load_bucket(descendant_counts.get(str(state.id), 0))
+            if bucket in used_buckets:
+                continue
+            picked.append(state)
+            top_scores.append(entry)
+            picked_ids.add(state.id)
+            used_buckets.add(bucket)
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
index 6cdb434..0072a26 100644
--- a/ttt_discover/discovery.py
+++ b/ttt_discover/discovery.py
@@ -62,6 +62,7 @@ class DiscoverConfig:
     codex_max_concurrent_requests: int | None = 4
     codex_initial_program_paths: tuple[str, ...] = ()
     codex_initial_pool_paths: tuple[str, ...] = ()
+    codex_subtree_load_balanced_sampling: bool = False
     codex_autonomous: bool = False
 
 
@@ -145,6 +146,9 @@ async def discover_impl(config: DiscoverConfig):
             max_concurrent_requests=config.codex_max_concurrent_requests,
             initial_program_paths=config.codex_initial_program_paths,
             initial_pool_paths=config.codex_initial_pool_paths,
+            subtree_load_balanced_sampling=(
+                config.codex_subtree_load_balanced_sampling
+            ),
             autonomous=config.codex_autonomous,
             wandb_project=config.wandb_project,
             wandb_name=config.experiment_name,
diff --git a/ttt_discover/rl/codex_no_finetune.py b/ttt_discover/rl/codex_no_finetune.py
index 18a0e9e..679d63d 100644
--- a/ttt_discover/rl/codex_no_finetune.py
+++ b/ttt_discover/rl/codex_no_finetune.py
@@ -137,6 +137,7 @@ class CodexNoFinetuneConfig:
     initial_program_paths: tuple[str, ...] = ()
     initial_pool_paths: tuple[str, ...] = ()
     topk_children: int = 16
+    subtree_load_balanced_sampling: bool = False
 
     # False is TTT Discover: non-auto, many short samples. True is AutoEvolve:
     # one Codex deep-dive workspace per sample with file edits and shell access.
@@ -820,7 +821,16 @@ async def sample_batch(
     i_batch: int,
 ) -> tuple[list[CandidateResult], dict[str, Any], list[CandidateResult]]:
     metrics: dict[str, Any] = {}
-    parent_states = sampler.sample_states(cfg.groups_per_batch)
+    if cfg.subtree_load_balanced_sampling:
+        sample_method = getattr(sampler, "sample_states_subtree_load_balanced", None)
+        if not callable(sample_method):
+            raise ValueError(
+                "subtree_load_balanced_sampling=True requires sampler "
+                "sample_states_subtree_load_balanced(num_states)"
+            )
+        parent_states = sample_method(cfg.groups_per_batch)
+    else:
+        parent_states = sampler.sample_states(cfg.groups_per_batch)
     semaphore = (
         asyncio.Semaphore(cfg.max_concurrent_requests)
         if cfg.max_concurrent_requests is not None
````
</details>

