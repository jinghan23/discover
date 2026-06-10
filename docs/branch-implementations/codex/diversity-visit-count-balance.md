# codex/diversity-visit-count-balance

## Summary

基于 PUCT visit count n 为候选分桶，平衡 unvisited/low/medium/high 访问程度，避免只选同一访问次数层。

## Branch State

- Worktree: `/opt/tiger/discover-visit-count-balance`
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

- `codex_visit_count_balanced_sampling`
- `cleaned`
- `positive_groups`
- `visit_count_balanced_sampling`

### Constants

- None

### Classes

- None

### Functions

- `visit_count_bucket_map`
- `sample_states_visit_count_balanced`
- `_sample_parent_states`

## Diff Summary

- Worktree tracked shortstat: `4 files changed, 151 insertions(+), 1 deletion(-)`
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

- `repro/gpu_mode/run_0608_visit_count_balance.sh (1146 bytes)`
- `repro/run_discovery.py (15400 bytes)`
- `tests/test_visit_count_balanced_sampling.py (11944 bytes)`

### Detected Test Functions

- `tests/test_visit_count_balanced_sampling.py::test_flag_off_matches_baseline_ids_stats_and_table_schema_unchanged`
- `tests/test_visit_count_balanced_sampling.py::test_enabled_flag_requires_callable_sampling_api`
- `tests/test_visit_count_balanced_sampling.py::test_num_states_one_uses_baseline`
- `tests/test_visit_count_balanced_sampling.py::test_no_candidates_uses_baseline_path`
- `tests/test_visit_count_balanced_sampling.py::test_missing_zero_and_negative_counts_clamp_to_untried`
- `tests/test_visit_count_balanced_sampling.py::test_all_zero_archive_first_pass_one_bucket_then_fallback`
- `tests/test_visit_count_balanced_sampling.py::test_all_positive_equal_archive_uses_mid_bucket`
- `tests/test_visit_count_balanced_sampling.py::test_boundary_ties_are_not_split`
- `tests/test_visit_count_balanced_sampling.py::test_num_states_less_than_bucket_count_uses_puct_order`
- `tests/test_visit_count_balanced_sampling.py::test_first_pass_covers_different_buckets_before_same_bucket`
- `tests/test_visit_count_balanced_sampling.py::test_lineage_blocking_and_blocked_candidate_does_not_claim_bucket`
- `tests/test_visit_count_balanced_sampling.py::test_fallback_lineage_blocking_can_return_subset`
- `tests/test_visit_count_balanced_sampling.py::test_fallback_fills_by_puct_without_modifying_puct_state`
- `tests/test_visit_count_balanced_sampling.py::test_persistence_schema_has_no_visit_bucket_metadata`

## Untracked File Snapshots

These files are not part of `git diff` yet, but they are part of the worktree implementation state recorded here.

### `repro/gpu_mode/run_0608_visit_count_balance.sh`

````bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-gpu-mode-trimul-visit-count-balance-0608}"
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
  --codex-visit-count-balanced-sampling
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
        "--codex-visit-count-balanced-sampling",
        action="store_true",
        help="Use visit-count bucket coverage for PUCT parent sampling.",
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
            "codex_visit_count_balanced_sampling="
            f"{args.codex_visit_count_balanced_sampling}"
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
            visit_count_balanced_sampling=(
                args.codex_visit_count_balanced_sampling
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
        codex_visit_count_balanced_sampling=(
            args.codex_visit_count_balanced_sampling
        ),
        codex_autonomous=args.codex_autonomous,
        remove_constant_reward_groups=remove_constant_reward_groups,
    )
    discover(config)


if __name__ == "__main__":
    main()
````

### `tests/test_visit_count_balanced_sampling.py`

````python
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from ttt_discover.codex_utils.runtime import State
from ttt_discover.codex_utils.sampler import (
    PUCTSampler,
    _sampler_file_for_step,
    visit_count_bucket_map,
)
from ttt_discover.rl.codex_no_finetune import _sample_parent_states


class DummyEnv:
    state_type = State

    @staticmethod
    def create_initial_state(problem_type: str) -> State:
        return State(
            timestep=0,
            construction=["initial"],
            code="",
            value=0.0,
            id=f"initial-{problem_type or 'default'}",
        )


def make_state(
    state_id: str,
    value: float,
    *,
    parents: list[dict] | None = None,
) -> State:
    return State(
        timestep=1,
        construction=[state_id],
        code="",
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


class VisitCountBalancedSamplingTest(unittest.TestCase):
    def test_flag_off_matches_baseline_ids_stats_and_table_schema_unchanged(self):
        states = [
            make_state("s0", 10.0),
            make_state("s1", 9.0),
            make_state("s2", 8.0),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            baseline = make_sampler(tmpdir, states)
            expected = baseline.sample_states(3)
            expected_stats = baseline.get_sample_stats()
            expected_columns, _ = baseline.get_sample_table()

            off_sampler = make_sampler(tmpdir, states)
            cfg = SimpleNamespace(
                groups_per_batch=3,
                visit_count_balanced_sampling=False,
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

    def test_enabled_flag_requires_callable_sampling_api(self):
        cfg = SimpleNamespace(
            groups_per_batch=2,
            visit_count_balanced_sampling=True,
        )
        sampler = SimpleNamespace(sample_states=lambda num_states: [])

        with self.assertRaisesRegex(
            ValueError,
            "sample_states_visit_count_balanced",
        ):
            _sample_parent_states(cfg, sampler)

        sampler = SimpleNamespace(
            sample_states=lambda num_states: [],
            sample_states_visit_count_balanced=None,
        )
        with self.assertRaisesRegex(
            ValueError,
            "sample_states_visit_count_balanced",
        ):
            _sample_parent_states(cfg, sampler)

    def test_num_states_one_uses_baseline(self):
        states = [
            make_state("high", 100.0),
            make_state("low", 1.0),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            sampler = make_sampler(tmpdir, states)
            sampler._n = {"high": 0, "low": 10}
            picked = sampler.sample_states_visit_count_balanced(1)

        self.assertEqual([s.id for s in picked], ["high"])

    def test_no_candidates_uses_baseline_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sampler = make_sampler(tmpdir, [])
            picked = sampler.sample_states_visit_count_balanced(2)

        self.assertEqual(len(picked), 2)
        self.assertEqual(sampler._last_sampled_indices, [])
        self.assertEqual(
            sampler._last_puct_stats,
            [(0, 0.0, 0.0, 0.0, 0.0), (0, 0.0, 0.0, 0.0, 0.0)],
        )

    def test_missing_zero_and_negative_counts_clamp_to_untried(self):
        self.assertEqual(
            visit_count_bucket_map(
                {
                    "missing": 0,
                    "zero": 0,
                    "negative": -7,
                    "positive": 3,
                }
            ),
            {
                "missing": "visit_untried",
                "zero": "visit_untried",
                "negative": "visit_untried",
                "positive": "visit_mid",
            },
        )

        states = [
            make_state("missing", 100.0),
            make_state("zero", 99.0),
            make_state("negative", 98.0),
            make_state("positive", 97.0),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            sampler = make_sampler(tmpdir, states)
            sampler._n = {"zero": 0, "negative": -2, "positive": 5}
            picked = sampler.sample_states_visit_count_balanced(3)
            stats = sampler._last_puct_stats

        self.assertEqual([s.id for s in picked], ["missing", "positive", "zero"])
        self.assertEqual([row[0] for row in stats], [0, 5, 0])

    def test_all_zero_archive_first_pass_one_bucket_then_fallback(self):
        states = [
            make_state("s0", 100.0),
            make_state("s1", 99.0),
            make_state("s2", 98.0),
        ]
        buckets = visit_count_bucket_map({state.id: 0 for state in states})
        with tempfile.TemporaryDirectory() as tmpdir:
            sampler = make_sampler(tmpdir, states)
            picked = sampler.sample_states_visit_count_balanced(3)

        self.assertEqual(set(buckets.values()), {"visit_untried"})
        self.assertEqual([s.id for s in picked], ["s0", "s1", "s2"])

    def test_all_positive_equal_archive_uses_mid_bucket(self):
        buckets = visit_count_bucket_map({"a": 4, "b": 4, "c": 4})
        self.assertEqual(
            buckets,
            {"a": "visit_mid", "b": "visit_mid", "c": "visit_mid"},
        )

        states = [
            make_state("a", 3.0),
            make_state("b", 2.0),
            make_state("c", 1.0),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            sampler = make_sampler(tmpdir, states)
            sampler._n = {"a": 4, "b": 4, "c": 4}
            picked = sampler.sample_states_visit_count_balanced(3)

        self.assertEqual([s.id for s in picked], ["a", "b", "c"])

    def test_boundary_ties_are_not_split(self):
        buckets = visit_count_bucket_map(
            {
                "a": 1,
                "b": 1,
                "c": 2,
                "d": 2,
                "e": 2,
                "f": 3,
            }
        )

        self.assertEqual(buckets["a"], "visit_low")
        self.assertEqual(buckets["b"], "visit_low")
        self.assertEqual(buckets["c"], "visit_mid")
        self.assertEqual(buckets["d"], "visit_mid")
        self.assertEqual(buckets["e"], "visit_mid")
        self.assertEqual(buckets["f"], "visit_high")

    def test_num_states_less_than_bucket_count_uses_puct_order(self):
        states = [
            make_state("mid", 100.0),
            make_state("low", 99.0),
            make_state("untried", 98.0),
            make_state("high", 97.0),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            sampler = make_sampler(tmpdir, states)
            sampler._n = {"low": 1, "mid": 2, "high": 3}
            picked = sampler.sample_states_visit_count_balanced(2)

        self.assertEqual([s.id for s in picked], ["mid", "low"])

    def test_first_pass_covers_different_buckets_before_same_bucket(self):
        states = [
            make_state("same_top", 100.0),
            make_state("same_next", 99.0),
            make_state("different_low", 10.0),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            sampler = make_sampler(tmpdir, states)
            sampler._n = {
                "same_top": 1,
                "same_next": 1,
                "different_low": 2,
            }
            picked = sampler.sample_states_visit_count_balanced(2)

        self.assertEqual([s.id for s in picked], ["same_top", "different_low"])

    def test_lineage_blocking_and_blocked_candidate_does_not_claim_bucket(self):
        states = [
            make_state("parent", 100.0),
            make_state("blocked", 99.0, parents=[{"id": "parent"}]),
            make_state("available_same_bucket", 98.0),
            make_state("other", 97.0),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            sampler = make_sampler(tmpdir, states)
            sampler._n = {
                "parent": 1,
                "blocked": 2,
                "available_same_bucket": 2,
                "other": 3,
            }
            picked = sampler.sample_states_visit_count_balanced(2)

        self.assertEqual([s.id for s in picked], ["parent", "available_same_bucket"])

    def test_fallback_lineage_blocking_can_return_subset(self):
        states = [
            make_state("parent", 100.0),
            make_state("blocked", 99.0, parents=[{"id": "parent"}]),
            make_state("other", 98.0),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            sampler = make_sampler(tmpdir, states)
            picked = sampler.sample_states_visit_count_balanced(3)

        self.assertEqual([s.id for s in picked], ["parent", "other"])

    def test_fallback_fills_by_puct_without_modifying_puct_state(self):
        states = [
            make_state("s0", 100.0),
            make_state("s1", 99.0),
            make_state("s2", 98.0),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            sampler = make_sampler(tmpdir, states)
            sampler._n = {"s0": 0, "s1": 0, "s2": 0}
            sampler._m = {"s0": 1.0}
            sampler._T = 7
            n_before = dict(sampler._n)
            m_before = dict(sampler._m)
            t_before = sampler._T
            picked = sampler.sample_states_visit_count_balanced(3)

        self.assertEqual([s.id for s in picked], ["s0", "s1", "s2"])
        self.assertEqual(sampler._n, n_before)
        self.assertEqual(sampler._m, m_before)
        self.assertEqual(sampler._T, t_before)

    def test_persistence_schema_has_no_visit_bucket_metadata(self):
        states = [
            make_state("a", 2.0),
            make_state("b", 1.0),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            sampler = make_sampler(tmpdir, states)
            sampler._n = {"a": 0, "b": 1}
            sampler.sample_states_visit_count_balanced(2)
            sampler.flush(step=3)
            store_path = _sampler_file_for_step(sampler.file_path, 3)
            with open(store_path, "r", encoding="utf-8") as f:
                store = json.load(f)
            columns, _ = sampler.get_sample_table()

        self.assertEqual(
            set(store),
            {"step", "states", "initial_states", "puct_n", "puct_m", "puct_T"},
        )
        self.assertFalse(any("visit" in key for key in store))
        self.assertFalse(any("visit" in column for column in columns))


if __name__ == "__main__":
    unittest.main()
````

## Diff Stat

### Worktree Tracked Diff Stat

````text
 ttt_discover/codex_utils/discovery.py |   2 +
 ttt_discover/codex_utils/sampler.py   | 133 ++++++++++++++++++++++++++++++++++
 ttt_discover/discovery.py             |   2 +
 ttt_discover/rl/codex_no_finetune.py  |  15 +++-
 4 files changed, 151 insertions(+), 1 deletion(-)
````
## Raw Diff

<details>
<summary>Tracked worktree patch</summary>

````diff
diff --git a/ttt_discover/codex_utils/discovery.py b/ttt_discover/codex_utils/discovery.py
index b21a80f..8f0d39a 100644
--- a/ttt_discover/codex_utils/discovery.py
+++ b/ttt_discover/codex_utils/discovery.py
@@ -50,6 +50,7 @@ class DiscoverConfig:
     codex_initial_program_paths: tuple[str, ...] = ()
     codex_initial_pool_paths: tuple[str, ...] = ()
     codex_autonomous: bool = False
+    codex_visit_count_balanced_sampling: bool = False
 
 
 def _run_codex_no_finetune(config: DiscoverConfig) -> None:
@@ -85,6 +86,7 @@ def _run_codex_no_finetune(config: DiscoverConfig) -> None:
         initial_program_paths=config.codex_initial_program_paths,
         initial_pool_paths=config.codex_initial_pool_paths,
         autonomous=config.codex_autonomous,
+        visit_count_balanced_sampling=config.codex_visit_count_balanced_sampling,
         wandb_project=config.wandb_project,
         wandb_name=experiment_name,
         log_path=log_path,
diff --git a/ttt_discover/codex_utils/sampler.py b/ttt_discover/codex_utils/sampler.py
index 5c5d4d1..2dbff3b 100644
--- a/ttt_discover/codex_utils/sampler.py
+++ b/ttt_discover/codex_utils/sampler.py
@@ -331,6 +331,60 @@ def seed_initial_program_paths(
     return len(seed_states)
 
 
+def visit_count_bucket_map(counts_by_id: dict[str, int]) -> dict[str, str]:
+    cleaned: dict[str, int] = {}
+    for state_id, raw_count in counts_by_id.items():
+        count = int(raw_count or 0)
+        if count < 0:
+            logger.debug("Clamping negative visit count for state %s: %s", state_id, count)
+            count = 0
+        cleaned[state_id] = count
+
+    buckets = {
+        state_id: "visit_untried"
+        for state_id, count in cleaned.items()
+        if count == 0
+    }
+    positive_groups: dict[int, list[str]] = {}
+    for state_id, count in cleaned.items():
+        if count > 0:
+            positive_groups.setdefault(count, []).append(state_id)
+
+    distinct_counts = sorted(positive_groups)
+    if not distinct_counts:
+        return buckets
+
+    if len(distinct_counts) == 1:
+        for state_id in positive_groups[distinct_counts[0]]:
+            buckets[state_id] = "visit_mid"
+        return buckets
+
+    if len(distinct_counts) == 2:
+        low_count, high_count = distinct_counts
+        for state_id in positive_groups[low_count]:
+            buckets[state_id] = "visit_low"
+        for state_id in positive_groups[high_count]:
+            buckets[state_id] = "visit_high"
+        return buckets
+
+    n_pos = sum(len(positive_groups[count]) for count in distinct_counts)
+    start = 0
+    for count in distinct_counts:
+        group = positive_groups[count]
+        end = start + len(group) - 1
+        midpoint = (start + end) / 2.0
+        if midpoint < n_pos / 3.0:
+            bucket = "visit_low"
+        elif midpoint < 2.0 * n_pos / 3.0:
+            bucket = "visit_mid"
+        else:
+            bucket = "visit_high"
+        for state_id in group:
+            buckets[state_id] = bucket
+        start = end + 1
+    return buckets
+
+
 class PUCTSampler(StateSampler):
     """
     PUCT-style sampler with state archive.
@@ -548,6 +602,85 @@ class PUCTSampler(StateSampler):
 
         return picked
 
+    def sample_states_visit_count_balanced(self, num_states: int) -> list[State]:
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
+        counts_by_id: dict[str, int] = {}
+        for i, s in enumerate(candidates):
+            n = int(self._n.get(s.id, 0) or 0)
+            if n < 0:
+                logger.debug("Clamping negative visit count for state %s: %s", s.id, n)
+                n = 0
+            counts_by_id[s.id] = n
+            m = self._m.get(s.id, vals[i])
+            Q = m if n > 0 else vals[i]
+            bonus = self.puct_c * scale * P[i] * sqrtT / (1.0 + n)
+            score = Q + bonus
+            scores.append((score, vals[i], s, n, Q, P[i], bonus))
+
+        scores.sort(key=lambda x: (x[0], x[1]), reverse=True)
+        buckets = visit_count_bucket_map(counts_by_id)
+        children_map = self._build_children_map()
+        picked: list[State] = []
+        top_scores = []
+        picked_ids: set[str] = set()
+        picked_buckets: set[str] = set()
+        blocked_ids: set[str] = set()
+
+        for entry in scores:
+            s = entry[2]
+            if s.id in blocked_ids:
+                continue
+            bucket = buckets.get(s.id)
+            if bucket in picked_buckets:
+                continue
+            picked.append(s)
+            top_scores.append(entry)
+            picked_ids.add(s.id)
+            if bucket is not None:
+                picked_buckets.add(bucket)
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
index 6cdb434..131472b 100644
--- a/ttt_discover/discovery.py
+++ b/ttt_discover/discovery.py
@@ -63,6 +63,7 @@ class DiscoverConfig:
     codex_initial_program_paths: tuple[str, ...] = ()
     codex_initial_pool_paths: tuple[str, ...] = ()
     codex_autonomous: bool = False
+    codex_visit_count_balanced_sampling: bool = False
 
 
 def init_ray(num_cpus_per_task: int, env_type: str):
@@ -146,6 +147,7 @@ async def discover_impl(config: DiscoverConfig):
             initial_program_paths=config.codex_initial_program_paths,
             initial_pool_paths=config.codex_initial_pool_paths,
             autonomous=config.codex_autonomous,
+            visit_count_balanced_sampling=config.codex_visit_count_balanced_sampling,
             wandb_project=config.wandb_project,
             wandb_name=config.experiment_name,
             log_path=log_path,
diff --git a/ttt_discover/rl/codex_no_finetune.py b/ttt_discover/rl/codex_no_finetune.py
index 18a0e9e..04d73a2 100644
--- a/ttt_discover/rl/codex_no_finetune.py
+++ b/ttt_discover/rl/codex_no_finetune.py
@@ -137,6 +137,7 @@ class CodexNoFinetuneConfig:
     initial_program_paths: tuple[str, ...] = ()
     initial_pool_paths: tuple[str, ...] = ()
     topk_children: int = 16
+    visit_count_balanced_sampling: bool = False
 
     # False is TTT Discover: non-auto, many short samples. True is AutoEvolve:
     # one Codex deep-dive workspace per sample with file edits and shell access.
@@ -775,6 +776,18 @@ def _sample_table(results: list[CandidateResult]) -> list[tuple[Any, ...]]:
     return rows
 
 
+def _sample_parent_states(cfg: CodexNoFinetuneConfig, sampler: StateSampler) -> list[Any]:
+    if getattr(cfg, "visit_count_balanced_sampling", False):
+        sample_balanced = getattr(sampler, "sample_states_visit_count_balanced", None)
+        if not callable(sample_balanced):
+            raise ValueError(
+                "visit_count_balanced_sampling requires sampler "
+                "sample_states_visit_count_balanced"
+            )
+        return sample_balanced(cfg.groups_per_batch)
+    return sampler.sample_states(cfg.groups_per_batch)
+
+
 def _result_metrics(results: list[CandidateResult], kept_results: list[CandidateResult]) -> dict[str, Any]:
     metrics: dict[str, Any] = {}
     rewards = [result.reward for result in kept_results]
@@ -820,7 +833,7 @@ async def sample_batch(
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

