# codex/diversity-observation-surface-balance

## Summary

根据父状态 observation 文本分类（如 traceback、timeout、empty、normal 等 profile），在 parent_balance=observation_surface 时按 observation surface 平衡。

## Branch State

- Worktree: `/opt/tiger/discover-observation-surface-balance`
- HEAD: `2234282`
- Base used for comparison: `2234282`
- Commits ahead of base: `0`
- Commits behind base: `0`
- Group: `diversity_extra`
- Implementation location: `uncommitted worktree diff`

## Implemented Behavior

- The summary above is based on reading the changed Python code, added config fields, and sampler/prompt hooks in this branch.
- For sampler branches, the common pattern is to keep baseline PUCT ordering inside a chosen surface/bucket, while changing which surface/bucket gets the next parent pick.
- Tests detected: `9` test functions across untracked test files.

## Added Markers

### Config fields

- `codex_parent_balance`
- `parent_balance`

### Constants

- `_OBSERVATION_SURFACE_PROFILES`
- `_TRACEBACK_RE`
- `_TIMEOUT_RE`

### Classes

- None

### Functions

- `observation_surface_profile`
- `_prune_observation_surface_sample_counts`
- `_rank_puct_candidates`
- `_record_observation_surface_samples`
- `sample_states_observation_surface_balanced`
- `_sample_parent_states`

## Diff Summary

- Worktree tracked shortstat: `5 files changed, 223 insertions(+), 18 deletions(-)`
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

- `repro/gpu_mode/run_0608_observation_surface_balance.sh (957 bytes)`
- `repro/run_discovery.py (14309 bytes)`
- `tests/test_observation_surface_parent_balance.py (12380 bytes)`

### Detected Test Functions

- `tests/test_observation_surface_parent_balance.py::test_profile_buckets_and_markers`
- `tests/test_observation_surface_parent_balance.py::test_default_off_cli_and_sampling_match_baseline`
- `tests/test_observation_surface_parent_balance.py::test_enabled_config_calls_observation_surface_sampler`
- `tests/test_observation_surface_parent_balance.py::test_single_parent_balances_repeated_calls_and_persists_counts`
- `tests/test_observation_surface_parent_balance.py::test_resume_ignores_bad_or_unknown_observation_surface_counts`
- `tests/test_observation_surface_parent_balance.py::test_multi_parent_selects_distinct_surfaces_then_puct_fallback`
- `tests/test_observation_surface_parent_balance.py::test_full_lineage_blocking_scans_same_surface_before_consuming_it`
- `tests/test_observation_surface_parent_balance.py::test_same_surface_and_all_empty_observations_use_puct_order`
- `tests/test_observation_surface_parent_balance.py::test_sampler_save_contains_only_observation_surface_counts`

## Untracked File Snapshots

These files are not part of `git diff` yet, but they are part of the worktree implementation state recorded here.

### `repro/gpu_mode/run_0608_observation_surface_balance.sh`

````bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-gpu-mode-trimul-observation-surface-balance-0608}"
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
  --codex-parent-balance observation_surface
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
        "--codex-parent-balance",
        choices=("none", "observation_surface"),
        default="none",
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
        print(f"codex_parent_balance={args.codex_parent_balance!r}")
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
            parent_balance=args.codex_parent_balance,
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
        codex_autonomous=args.codex_autonomous,
        codex_parent_balance=args.codex_parent_balance,
        remove_constant_reward_groups=remove_constant_reward_groups,
    )
    discover(config)


if __name__ == "__main__":
    main()
````

### `tests/test_observation_surface_parent_balance.py`

````python
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

from ttt_discover.codex_utils.runtime import State
from ttt_discover.codex_utils.sampler import (
    PUCTSampler,
    observation_surface_profile,
)
from ttt_discover.rl.codex_no_finetune import (
    CodexNoFinetuneConfig,
    _sample_parent_states,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class DummyEnv:
    state_type = State
    _initial_counter = 0

    @classmethod
    def create_initial_state(cls, problem_type: str) -> State:
        cls._initial_counter += 1
        return State(
            timestep=0,
            construction=[cls._initial_counter],
            code="",
            value=0.0,
            id=f"initial-{cls._initial_counter}",
            observation="",
        )


def make_state(
    state_id: str,
    value: float,
    observation: Any,
    *,
    parents: list[dict[str, Any]] | None = None,
) -> State:
    return State(
        timestep=1,
        construction=[state_id],
        code=f"code-{state_id}",
        value=value,
        id=state_id,
        observation=observation,
        parents=parents or [],
    )


def make_sampler(tmpdir: str, states: list[State]) -> PUCTSampler:
    sampler = PUCTSampler(
        file_path=str(Path(tmpdir) / "puct_sampler.json"),
        env_type=DummyEnv,
        batch_size=0,
        puct_c=0.0,
        topk_children=0,
    )
    sampler._states = list(states)
    sampler._initial_states = []
    return sampler


class ObservationSurfaceProfileTests(unittest.TestCase):
    def test_profile_buckets_and_markers(self) -> None:
        self.assertEqual(observation_surface_profile(None), "obs_empty")
        self.assertEqual(observation_surface_profile(""), "obs_empty")
        self.assertEqual(observation_surface_profile("   \n\t"), "obs_empty")

        self.assertEqual(observation_surface_profile("a" * 512), "obs_short_plain")
        self.assertEqual(observation_surface_profile("a" * 513), "obs_medium_plain")
        self.assertEqual(observation_surface_profile("a" * 4096), "obs_medium_plain")
        self.assertEqual(observation_surface_profile("a" * 4097), "obs_long_plain")

        self.assertEqual(
            observation_surface_profile("last run had a Traceback here"),
            "obs_short_traceback",
        )
        self.assertEqual(
            observation_surface_profile("notraceback tracebacked"),
            "obs_short_plain",
        )

        self.assertEqual(observation_surface_profile("timeout"), "obs_short_timeout")
        self.assertEqual(observation_surface_profile("TIMED OUT"), "obs_short_timeout")
        self.assertEqual(
            observation_surface_profile("hit the time limit"),
            "obs_short_timeout",
        )

        ignored = "score 9999 generic error kernel triton A100"
        self.assertEqual(observation_surface_profile(ignored), "obs_short_plain")


class ParentBalanceSelectionTests(unittest.TestCase):
    def test_default_off_cli_and_sampling_match_baseline(self) -> None:
        states = [
            make_state("a", 10.0, "plain output"),
            make_state("b", 8.0, "Traceback output"),
            make_state("c", 5.0, "timeout output"),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            baseline = make_sampler(str(Path(tmpdir) / "baseline"), states)
            defaulted = make_sampler(str(Path(tmpdir) / "defaulted"), states)
            cfg = CodexNoFinetuneConfig(
                env_type=DummyEnv,
                groups_per_batch=2,
            )

            baseline_ids = [state.id for state in baseline.sample_states(2)]
            sampled_ids = [
                state.id for state in _sample_parent_states(cfg, defaulted)
            ]
            self.assertEqual(cfg.parent_balance, "none")
            self.assertEqual(sampled_ids, baseline_ids)

        default_run = subprocess.run(
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
        self.assertIn("codex_parent_balance='none'", default_run.stdout)
        self.assertNotIn(
            "codex_parent_balance='observation_surface'",
            default_run.stdout,
        )

        enabled_run = subprocess.run(
            [
                sys.executable,
                "repro/run_discovery.py",
                "dry-run",
                "--task",
                "trimul",
                "--codex-parent-balance",
                "observation_surface",
            ],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
        self.assertIn(
            "codex_parent_balance='observation_surface'",
            enabled_run.stdout,
        )

    def test_enabled_config_calls_observation_surface_sampler(self) -> None:
        class FakeSampler:
            def __init__(self) -> None:
                self.baseline_calls: list[int] = []
                self.balanced_calls: list[int] = []

            def sample_states(self, num_states: int) -> list[str]:
                self.baseline_calls.append(num_states)
                return ["baseline"]

            def sample_states_observation_surface_balanced(
                self,
                num_states: int,
            ) -> list[str]:
                self.balanced_calls.append(num_states)
                return ["balanced"]

        fake = FakeSampler()
        cfg = CodexNoFinetuneConfig(
            env_type=DummyEnv,
            groups_per_batch=3,
            parent_balance="observation_surface",
        )
        self.assertEqual(_sample_parent_states(cfg, fake), ["balanced"])
        self.assertEqual(fake.baseline_calls, [])
        self.assertEqual(fake.balanced_calls, [3])

        class MissingMethodSampler:
            def sample_states(self, num_states: int) -> list[str]:
                return ["baseline"]

        with self.assertRaisesRegex(ValueError, "observation_surface"):
            _sample_parent_states(cfg, MissingMethodSampler())

    def test_single_parent_balances_repeated_calls_and_persists_counts(self) -> None:
        states = [
            make_state("plain", 100.0, "plain output"),
            make_state("trace", 90.0, "Traceback output"),
            make_state("timeout", 80.0, "timeout output"),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            sampler = make_sampler(tmpdir, states)
            self.assertEqual(
                [
                    sampler.sample_states_observation_surface_balanced(1)[0].id
                    for _ in range(2)
                ],
                ["plain", "trace"],
            )
            sampler.flush(step=1)

            resumed = PUCTSampler(
                file_path=str(Path(tmpdir) / "puct_sampler.json"),
                env_type=DummyEnv,
                batch_size=0,
                resume_step=1,
                puct_c=0.0,
                topk_children=0,
            )
            self.assertEqual(
                resumed.observation_surface_sample_counts,
                {
                    "obs_short_plain": 1,
                    "obs_short_traceback": 1,
                },
            )
            picked = resumed.sample_states_observation_surface_balanced(1)
            self.assertEqual([state.id for state in picked], ["timeout"])
            self.assertEqual(
                resumed.observation_surface_sample_counts["obs_short_timeout"],
                1,
            )

    def test_resume_ignores_bad_or_unknown_observation_surface_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sampler = make_sampler(tmpdir, [make_state("plain", 1.0, "plain")])
            sampler.observation_surface_sample_counts = {
                "obs_short_plain": 2,
                "obs_short_timeout": -3,
                "not_allowed": 99,
            }
            sampler.flush(step=1)

            save_path = Path(tmpdir) / "puct_sampler_step_000001.json"
            store = json.loads(save_path.read_text(encoding="utf-8"))
            self.assertNotIn("not_allowed", store["observation_surface_sample_counts"])
            self.assertEqual(store["observation_surface_sample_counts"]["obs_short_timeout"], 0)
            store["observation_surface_sample_counts"]["obs_medium_plain"] = "bad"
            save_path.write_text(json.dumps(store), encoding="utf-8")

            resumed = PUCTSampler(
                file_path=str(Path(tmpdir) / "puct_sampler.json"),
                env_type=DummyEnv,
                batch_size=0,
                resume_step=1,
                puct_c=0.0,
                topk_children=0,
            )

        self.assertEqual(
            resumed.observation_surface_sample_counts,
            {
                "obs_short_plain": 2,
                "obs_short_timeout": 0,
            },
        )

    def test_multi_parent_selects_distinct_surfaces_then_puct_fallback(self) -> None:
        states = [
            make_state("plain-1", 100.0, "plain output"),
            make_state("plain-2", 99.0, "another plain output"),
            make_state("trace", 50.0, "Traceback output"),
            make_state("timeout", 40.0, "timeout output"),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            sampler = make_sampler(tmpdir, states)
            picked = sampler.sample_states_observation_surface_balanced(4)

        self.assertEqual(
            [state.id for state in picked],
            ["plain-1", "trace", "timeout", "plain-2"],
        )

    def test_full_lineage_blocking_scans_same_surface_before_consuming_it(self) -> None:
        states = [
            make_state("plain-parent", 100.0, "plain output"),
            make_state(
                "blocked-trace",
                95.0,
                "Traceback child",
                parents=[{"id": "plain-parent", "timestep": 1}],
            ),
            make_state("open-trace", 90.0, "Traceback sibling"),
            make_state("timeout", 80.0, "timeout output"),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            sampler = make_sampler(tmpdir, states)
            picked = sampler.sample_states_observation_surface_balanced(3)

        self.assertEqual(
            [state.id for state in picked],
            ["plain-parent", "open-trace", "timeout"],
        )

    def test_same_surface_and_all_empty_observations_use_puct_order(self) -> None:
        states = [
            make_state("top", 7.0, ""),
            make_state("third", 5.0, None),
            make_state("second", 6.0, "  "),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            sampler = make_sampler(tmpdir, states)
            picked = sampler.sample_states_observation_surface_balanced(3)

        self.assertEqual([state.id for state in picked], ["top", "second", "third"])

    def test_sampler_save_contains_only_observation_surface_counts(self) -> None:
        states = [
            make_state(
                "plain",
                10.0,
                "plain score 9999 generic error kernel triton A100",
            ),
            make_state("trace", 8.0, "Traceback output"),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            sampler = make_sampler(tmpdir, states)
            sampler.sample_states_observation_surface_balanced(2)
            sampler.flush(step=1)

            save_path = Path(tmpdir) / "puct_sampler_step_000001.json"
            store = json.loads(save_path.read_text(encoding="utf-8"))

        self.assertEqual(
            set(store),
            {
                "step",
                "states",
                "initial_states",
                "puct_n",
                "puct_m",
                "puct_T",
                "observation_surface_sample_counts",
            },
        )
        self.assertEqual(
            store["observation_surface_sample_counts"],
            {
                "obs_short_plain": 1,
                "obs_short_traceback": 1,
            },
        )


if __name__ == "__main__":
    unittest.main()
````

## Diff Stat

### Worktree Tracked Diff Stat

````text
 ttt_discover/codex_utils/__init__.py  |   2 +
 ttt_discover/codex_utils/discovery.py |   2 +
 ttt_discover/codex_utils/sampler.py   | 214 +++++++++++++++++++++++++++++++---
 ttt_discover/discovery.py             |   2 +
 ttt_discover/rl/codex_no_finetune.py  |  21 +++-
 5 files changed, 223 insertions(+), 18 deletions(-)
````
## Raw Diff

<details>
<summary>Tracked worktree patch</summary>

````diff
diff --git a/ttt_discover/codex_utils/__init__.py b/ttt_discover/codex_utils/__init__.py
index 1a51326..a730b0c 100644
--- a/ttt_discover/codex_utils/__init__.py
+++ b/ttt_discover/codex_utils/__init__.py
@@ -11,6 +11,7 @@ from ttt_discover.codex_utils.sampler import (
     StateSampler,
     create_sampler,
     get_or_create_sampler_with_default,
+    observation_surface_profile,
 )
 
 __all__ = [
@@ -26,6 +27,7 @@ __all__ = [
     "create_sampler",
     "discover",
     "get_or_create_sampler_with_default",
+    "observation_surface_profile",
     "state_from_dict",
     "to_json_serializable",
 ]
diff --git a/ttt_discover/codex_utils/discovery.py b/ttt_discover/codex_utils/discovery.py
index b21a80f..0488c8f 100644
--- a/ttt_discover/codex_utils/discovery.py
+++ b/ttt_discover/codex_utils/discovery.py
@@ -50,6 +50,7 @@ class DiscoverConfig:
     codex_initial_program_paths: tuple[str, ...] = ()
     codex_initial_pool_paths: tuple[str, ...] = ()
     codex_autonomous: bool = False
+    codex_parent_balance: Literal["none", "observation_surface"] = "none"
 
 
 def _run_codex_no_finetune(config: DiscoverConfig) -> None:
@@ -84,6 +85,7 @@ def _run_codex_no_finetune(config: DiscoverConfig) -> None:
         max_concurrent_requests=config.codex_max_concurrent_requests,
         initial_program_paths=config.codex_initial_program_paths,
         initial_pool_paths=config.codex_initial_pool_paths,
+        parent_balance=config.codex_parent_balance,
         autonomous=config.codex_autonomous,
         wandb_project=config.wandb_project,
         wandb_name=experiment_name,
diff --git a/ttt_discover/codex_utils/sampler.py b/ttt_discover/codex_utils/sampler.py
index 5c5d4d1..4a9f1ab 100644
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
@@ -66,6 +67,48 @@ def _read_json_or_default(path: str, default: Any) -> Any:
         return default
 
 
+_OBSERVATION_SURFACE_PROFILES = {
+    "obs_empty",
+    "obs_short_plain",
+    "obs_medium_plain",
+    "obs_long_plain",
+    "obs_short_traceback",
+    "obs_medium_traceback",
+    "obs_long_traceback",
+    "obs_short_timeout",
+    "obs_medium_timeout",
+    "obs_long_timeout",
+}
+_TRACEBACK_RE = re.compile(r"\btraceback\b", re.IGNORECASE)
+_TIMEOUT_RE = re.compile(
+    r"\btimeout\b|\btimed\s+out\b|\btime\s+limit\b",
+    re.IGNORECASE,
+)
+_PuctEntry = tuple[float, float, State, int, float, float, float]
+
+
+def observation_surface_profile(observation: Any) -> str:
+    text = "" if observation is None else str(observation).strip()
+    if not text:
+        return "obs_empty"
+
+    length = len(text)
+    if length <= 512:
+        bucket = "short"
+    elif length <= 4096:
+        bucket = "medium"
+    else:
+        bucket = "long"
+
+    if _TRACEBACK_RE.search(text):
+        kind = "traceback"
+    elif _TIMEOUT_RE.search(text):
+        kind = "timeout"
+    else:
+        kind = "plain"
+    return f"obs_{bucket}_{kind}"
+
+
 class StateSampler(ABC):
     """Abstract base class for sampling states."""
 
@@ -375,6 +418,7 @@ class PUCTSampler(StateSampler):
         self._T: int = 0
         self._last_scale: float = 1.0
         self._last_puct_stats: list[tuple[int, float, float, float, float]] = []
+        self.observation_surface_sample_counts: dict[str, int] = {}
         
         if resume_step is not None:
             self._load(resume_step)
@@ -399,10 +443,21 @@ class PUCTSampler(StateSampler):
         self._n = store.get("puct_n", {}) or {}
         self._m = store.get("puct_m", {}) or {}
         self._T = int(store.get("puct_T", 0) or 0)
+        raw_counts = store.get("observation_surface_sample_counts", {}) or {}
+        self.observation_surface_sample_counts = {}
+        for surface, count in raw_counts.items():
+            surface = str(surface)
+            if surface not in _OBSERVATION_SURFACE_PROFILES:
+                continue
+            try:
+                self.observation_surface_sample_counts[surface] = max(0, int(count))
+            except (TypeError, ValueError):
+                continue
 
     def _save(self, step: int):
         save_path = _sampler_file_for_step(self.file_path, step)
         os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
+        self._prune_observation_surface_sample_counts()
         store = {
             "step": step,
             "states": [s.to_dict() for s in self._states],
@@ -410,10 +465,24 @@ class PUCTSampler(StateSampler):
             "puct_n": self._n,
             "puct_m": self._m,
             "puct_T": self._T,
+            "observation_surface_sample_counts": (
+                self.observation_surface_sample_counts
+            ),
         }
         with _file_lock(f"{save_path}.lock"):
             _atomic_write_json(save_path, store)
 
+    def _prune_observation_surface_sample_counts(self) -> None:
+        counts: dict[str, int] = {}
+        for surface, count in self.observation_surface_sample_counts.items():
+            if surface not in _OBSERVATION_SURFACE_PROFILES:
+                continue
+            try:
+                counts[surface] = max(0, int(count))
+            except (TypeError, ValueError):
+                continue
+        self.observation_surface_sample_counts = counts
+
     def _refresh_random_construction(self, state: State) -> None:
         """Let task-specific envs refresh seed states without coupling sampler to a task."""
         refresh = getattr(self.env_type, "refresh_initial_state", None)
@@ -460,6 +529,38 @@ class PUCTSampler(StateSampler):
         weights = (N - ranks).astype(np.float64)
         return weights / weights.sum()
 
+    def _rank_puct_candidates(
+        self,
+        candidates: list[State],
+        initial_ids: set[str],
+    ) -> list[_PuctEntry]:
+        vals = np.array(
+            [
+                float(s.value if s.value is not None else float("-inf"))
+                for s in candidates
+            ]
+        )
+        non_initial_mask = np.array([s.id not in initial_ids for s in candidates])
+        scale = self._compute_scale(
+            vals,
+            non_initial_mask if non_initial_mask.any() else None,
+        )
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
     def _get_lineage(self, state: State) -> set[str]:
         lineage = {state.id}
         for p in (state.parents or []):
@@ -489,6 +590,13 @@ class PUCTSampler(StateSampler):
                     queue.append(child_id)
         return lineage
 
+    def _record_observation_surface_samples(self, states: list[State]) -> None:
+        for state in states:
+            surface = observation_surface_profile(getattr(state, "observation", None))
+            self.observation_surface_sample_counts[surface] = (
+                self.observation_surface_sample_counts.get(surface, 0) + 1
+            )
+
     def sample_states(self, num_states: int) -> list[State]:
         initial_ids = {s.id for s in self._initial_states}
         candidates = list(self._states)
@@ -503,23 +611,7 @@ class PUCTSampler(StateSampler):
             self._last_puct_stats = [(0, 0.0, 0.0, 0.0, 0.0) for _ in picked]
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
+        scores = self._rank_puct_candidates(candidates, initial_ids)
 
         if num_states > 1:
             children_map = self._build_children_map()
@@ -548,6 +640,94 @@ class PUCTSampler(StateSampler):
 
         return picked
 
+    def sample_states_observation_surface_balanced(self, num_states: int) -> list[State]:
+        if num_states <= 0:
+            return self.sample_states(num_states)
+
+        initial_ids = {s.id for s in self._initial_states}
+        candidates = list(self._states)
+        if not candidates:
+            return self.sample_states(num_states)
+
+        scores = self._rank_puct_candidates(candidates, initial_ids)
+
+        if num_states == 1:
+            best_by_surface: dict[str, tuple[int, _PuctEntry]] = {}
+            for rank, entry in enumerate(scores):
+                surface = observation_surface_profile(
+                    getattr(entry[2], "observation", None)
+                )
+                if surface not in best_by_surface:
+                    best_by_surface[surface] = (rank, entry)
+            _surface, (_rank, selected_entry) = min(
+                best_by_surface.items(),
+                key=lambda item: (
+                    self.observation_surface_sample_counts.get(item[0], 0),
+                    item[1][0],
+                    item[0],
+                ),
+            )
+            picked = [selected_entry[2]]
+            top_scores = [selected_entry]
+        else:
+            entries_by_surface: dict[
+                str,
+                list[tuple[int, _PuctEntry]],
+            ] = {}
+            for rank, entry in enumerate(scores):
+                surface = observation_surface_profile(
+                    getattr(entry[2], "observation", None)
+                )
+                entries_by_surface.setdefault(surface, []).append((rank, entry))
+
+            children_map = self._build_children_map()
+            picked: list[State] = []
+            top_scores: list[_PuctEntry] = []
+            blocked_ids: set[str] = set()
+
+            surface_order = sorted(
+                entries_by_surface.items(),
+                key=lambda item: (
+                    self.observation_surface_sample_counts.get(item[0], 0),
+                    item[1][0][0],
+                    item[0],
+                ),
+            )
+            for _surface, surface_entries in surface_order:
+                for _rank, entry in surface_entries:
+                    state = entry[2]
+                    if state.id in blocked_ids:
+                        continue
+                    picked.append(state)
+                    top_scores.append(entry)
+                    blocked_ids.update(self._get_full_lineage(state, children_map))
+                    break
+                if len(picked) >= num_states:
+                    break
+
+            if len(picked) < num_states:
+                for entry in scores:
+                    state = entry[2]
+                    if state.id in blocked_ids:
+                        continue
+                    picked.append(state)
+                    top_scores.append(entry)
+                    blocked_ids.update(self._get_full_lineage(state, children_map))
+                    if len(picked) >= num_states:
+                        break
+
+        state_id_to_idx = {s.id: i for i, s in enumerate(self._states)}
+        self._last_sampled_states = picked
+        self._last_sampled_indices = [state_id_to_idx.get(s.id, -1) for s in picked]
+        self._last_puct_stats = [(t[3], t[4], t[5], t[6], t[0]) for t in top_scores]
+        self._record_observation_surface_samples(picked)
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
index 6cdb434..52ed9ec 100644
--- a/ttt_discover/discovery.py
+++ b/ttt_discover/discovery.py
@@ -63,6 +63,7 @@ class DiscoverConfig:
     codex_initial_program_paths: tuple[str, ...] = ()
     codex_initial_pool_paths: tuple[str, ...] = ()
     codex_autonomous: bool = False
+    codex_parent_balance: Literal["none", "observation_surface"] = "none"
 
 
 def init_ray(num_cpus_per_task: int, env_type: str):
@@ -145,6 +146,7 @@ async def discover_impl(config: DiscoverConfig):
             max_concurrent_requests=config.codex_max_concurrent_requests,
             initial_program_paths=config.codex_initial_program_paths,
             initial_pool_paths=config.codex_initial_pool_paths,
+            parent_balance=config.codex_parent_balance,
             autonomous=config.codex_autonomous,
             wandb_project=config.wandb_project,
             wandb_name=config.experiment_name,
diff --git a/ttt_discover/rl/codex_no_finetune.py b/ttt_discover/rl/codex_no_finetune.py
index 18a0e9e..a5f83e1 100644
--- a/ttt_discover/rl/codex_no_finetune.py
+++ b/ttt_discover/rl/codex_no_finetune.py
@@ -137,6 +137,7 @@ class CodexNoFinetuneConfig:
     initial_program_paths: tuple[str, ...] = ()
     initial_pool_paths: tuple[str, ...] = ()
     topk_children: int = 16
+    parent_balance: Literal["none", "observation_surface"] = "none"
 
     # False is TTT Discover: non-auto, many short samples. True is AutoEvolve:
     # one Codex deep-dive workspace per sample with file edits and shell access.
@@ -814,13 +815,31 @@ def _result_metrics(results: list[CandidateResult], kept_results: list[Candidate
     return metrics
 
 
+def _sample_parent_states(cfg: CodexNoFinetuneConfig, sampler: StateSampler) -> list[Any]:
+    if cfg.parent_balance == "none":
+        return sampler.sample_states(cfg.groups_per_batch)
+    if cfg.parent_balance == "observation_surface":
+        sample_method = getattr(
+            sampler,
+            "sample_states_observation_surface_balanced",
+            None,
+        )
+        if not callable(sample_method):
+            raise ValueError(
+                "parent_balance='observation_surface' requires "
+                "sampler.sample_states_observation_surface_balanced"
+            )
+        return sample_method(cfg.groups_per_batch)
+    raise ValueError(f"Unknown parent_balance: {cfg.parent_balance!r}")
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

