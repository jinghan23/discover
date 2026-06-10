# codex/diversity-outcome-surface-balance

## Summary

从 CandidateResult 的 metrics/message/error 中抽取 outcome surface，记录到 state sidecar，并在 parent_balance=outcome_surface 时按结果类别平衡。

## Branch State

- Worktree: `/opt/tiger/discover-outcome-surface-balance`
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
- `outcome_surface`
- `parent_balance`

### Constants

- `_OUTCOME_SURFACE_TOKEN_RE`

### Classes

- None

### Functions

- `_prune_outcome_surface_sidecars`
- `_outcome_surface_for_state`
- `_record_outcome_surface_samples`
- `record_state_outcome_surface`
- `sample_states_outcome_surface_balanced`
- `_sanitize_outcome_surface`
- `_extract_outcome_surface`
- `_record_result_outcome_surface`
- `_sample_parent_states`

## Diff Summary

- Worktree tracked shortstat: `4 files changed, 237 insertions(+), 1 deletion(-)`
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

- `repro/gpu_mode/run_0608_outcome_surface_balance.sh (949 bytes)`
- `repro/run_discovery.py (14305 bytes)`
- `tests/test_outcome_surface_parent_balance.py (14509 bytes)`

### Detected Test Functions

- `tests/test_outcome_surface_parent_balance.py::test_default_off_matches_baseline_and_dry_run_default_is_none`
- `tests/test_outcome_surface_parent_balance.py::test_old_snapshot_without_sidecar_loads_unknown_and_samples`
- `tests/test_outcome_surface_parent_balance.py::test_num_states_one_balances_repeated_calls_across_surfaces`
- `tests/test_outcome_surface_parent_balance.py::test_within_one_surface_selection_remains_puct_order`
- `tests/test_outcome_surface_parent_balance.py::test_num_states_many_uses_surface_pass_lineage_blocking_and_fallback`
- `tests/test_outcome_surface_parent_balance.py::test_resume_preserves_sidecar_counts_and_prunes_stale_state_ids`
- `tests/test_outcome_surface_parent_balance.py::test_extractor_uses_metrics_or_valid_correct_and_failed_results_are_ignored`
- `tests/test_outcome_surface_parent_balance.py::test_duplicate_results_do_not_overwrite_existing_creation_surface`
- `tests/test_outcome_surface_parent_balance.py::test_cli_and_run_script_enable_outcome_surface_only_when_requested`

## Untracked File Snapshots

These files are not part of `git diff` yet, but they are part of the worktree implementation state recorded here.

### `repro/gpu_mode/run_0608_outcome_surface_balance.sh`

````bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-gpu-mode-trimul-outcome-surface-balance-0608}"
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
  --codex-parent-balance outcome_surface
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
        choices=("none", "outcome_surface"),
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

### `tests/test_outcome_surface_parent_balance.py`

````python
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from ttt_discover.codex_utils.runtime import State
from ttt_discover.codex_utils.sampler import PUCTSampler, _sampler_file_for_step
from ttt_discover.rl.codex_no_finetune import (
    CandidateResult,
    _extract_outcome_surface,
    _sample_parent_states,
    _update_sampler_from_results,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class DummyEnv:
    state_type = State

    @staticmethod
    def create_initial_state(problem_type: str) -> State:
        return State(
            timestep=0,
            construction=["initial", problem_type],
            code="",
            value=0.0,
            id=f"initial-{problem_type or 'default'}",
        )


def make_state(
    state_id: str,
    value: float,
    *,
    code: str = "",
    parents: list[dict] | None = None,
) -> State:
    return State(
        timestep=1,
        construction=[state_id],
        code=code,
        value=value,
        id=state_id,
        parents=parents or [],
    )


def make_sampler(
    tmpdir: str,
    states: list[State],
    *,
    max_buffer_size: int = 1000,
) -> PUCTSampler:
    sampler = PUCTSampler(
        file_path=str(Path(tmpdir) / "puct_sampler.json"),
        env_type=DummyEnv,
        problem_type="test",
        batch_size=1,
        max_buffer_size=max_buffer_size,
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
    sampler._outcome_surface_by_state_id = {}
    sampler._outcome_surface_sample_counts = {}
    return sampler


def make_result(
    parent: State,
    *,
    next_state: State | None,
    correct_format: bool = True,
    correctness: float = 1.0,
    metrics: dict | None = None,
    outcome_surface: str | None = None,
) -> CandidateResult:
    return CandidateResult(
        parent_state=parent,
        group_idx=0,
        sample_idx=0,
        prompt="prompt",
        response="response",
        parsed_code="def custom_kernel(): pass",
        reward=1.0 if correctness > 0 else 0.0,
        correctness=correctness,
        correct_format=correct_format,
        raw_score=10.0 if correctness > 0 else None,
        msg="ok" if correctness > 0 else "failed",
        metrics=metrics or {},
        outcome_surface=outcome_surface,
        next_state=next_state,
    )


class OutcomeSurfaceParentBalanceTest(unittest.TestCase):
    def test_default_off_matches_baseline_and_dry_run_default_is_none(self):
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

            sampler = make_sampler(tmpdir, states)
            cfg = SimpleNamespace(groups_per_batch=3, parent_balance="none")
            actual = _sample_parent_states(cfg, sampler)
            actual_stats = sampler.get_sample_stats()
            actual_columns, rows = sampler.get_sample_table()

        self.assertEqual([s.id for s in actual], [s.id for s in expected])
        self.assertEqual(actual_stats, expected_stats)
        self.assertEqual(actual_columns, expected_columns)
        self.assertEqual(len(rows), 3)

        proc = subprocess.run(
            [sys.executable, "repro/run_discovery.py", "dry-run"],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
        self.assertIn("codex_parent_balance='none'", proc.stdout)
        self.assertNotIn("codex_parent_balance='outcome_surface'", proc.stdout)

    def test_old_snapshot_without_sidecar_loads_unknown_and_samples(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = str(Path(tmpdir) / "puct_sampler.json")
            states = [make_state("high", 5.0), make_state("low", 1.0)]
            store_path = _sampler_file_for_step(file_path, 7)
            Path(store_path).write_text(
                json.dumps(
                    {
                        "step": 7,
                        "states": [s.to_dict() for s in states],
                        "initial_states": [],
                        "puct_n": {},
                        "puct_m": {},
                        "puct_T": 0,
                    }
                ),
                encoding="utf-8",
            )

            sampler = PUCTSampler(
                file_path=file_path,
                env_type=DummyEnv,
                problem_type="test",
                resume_step=7,
                puct_c=0.0,
                topk_children=0,
            )
            picked = sampler.sample_states_outcome_surface_balanced(1)

        self.assertEqual([s.id for s in picked], ["high"])
        self.assertEqual(sampler._outcome_surface_by_state_id, {})
        self.assertEqual(sampler._outcome_surface_sample_counts, {"unknown": 1})

    def test_num_states_one_balances_repeated_calls_across_surfaces(self):
        states = [make_state("high", 100.0), make_state("low", 1.0)]
        with tempfile.TemporaryDirectory() as tmpdir:
            sampler = make_sampler(tmpdir, states)
            sampler._outcome_surface_by_state_id = {
                "high": "surface_a",
                "low": "surface_b",
            }
            first = sampler.sample_states_outcome_surface_balanced(1)
            second = sampler.sample_states_outcome_surface_balanced(1)
            third = sampler.sample_states_outcome_surface_balanced(1)

        self.assertEqual([s.id for s in first], ["high"])
        self.assertEqual([s.id for s in second], ["low"])
        self.assertEqual([s.id for s in third], ["high"])
        self.assertEqual(
            sampler._outcome_surface_sample_counts,
            {"surface_a": 2, "surface_b": 1},
        )

    def test_within_one_surface_selection_remains_puct_order(self):
        states = [
            make_state("high", 10.0),
            make_state("mid", 5.0),
            make_state("low", 1.0),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            sampler = make_sampler(tmpdir, states)
            sampler._outcome_surface_by_state_id = {
                "high": "same_surface",
                "mid": "same_surface",
                "low": "same_surface",
            }
            picked = sampler.sample_states_outcome_surface_balanced(3)

        self.assertEqual([s.id for s in picked], ["high", "mid", "low"])

    def test_num_states_many_uses_surface_pass_lineage_blocking_and_fallback(self):
        states = [
            make_state("parent", 100.0),
            make_state("blocked_child", 99.0, parents=[{"id": "parent"}]),
            make_state("available_same_surface", 98.0),
            make_state("other_surface", 97.0),
            make_state("fallback_same_as_parent", 96.0),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            sampler = make_sampler(tmpdir, states)
            sampler._outcome_surface_by_state_id = {
                "parent": "surface_a",
                "blocked_child": "surface_b",
                "available_same_surface": "surface_b",
                "other_surface": "surface_c",
                "fallback_same_as_parent": "surface_a",
            }
            picked = sampler.sample_states_outcome_surface_balanced(4)

        self.assertEqual(
            [s.id for s in picked],
            [
                "parent",
                "available_same_surface",
                "other_surface",
                "fallback_same_as_parent",
            ],
        )

    def test_resume_preserves_sidecar_counts_and_prunes_stale_state_ids(self):
        states = [
            make_state("a", 10.0),
            make_state("b", 9.0),
            make_state("c", 1.0),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            sampler = make_sampler(tmpdir, states, max_buffer_size=2)
            sampler._outcome_surface_by_state_id = {
                "a": "surface_a",
                "b": "surface_b",
                "c": "surface_c",
                "stale": "surface_stale",
            }
            sampler._outcome_surface_sample_counts = {
                "surface_a": 2,
                "surface_b": 1,
                "surface_c": 1,
            }
            sampler.flush(step=4)
            store_path = _sampler_file_for_step(sampler.file_path, 4)
            store = json.loads(Path(store_path).read_text(encoding="utf-8"))

            reloaded = PUCTSampler(
                file_path=sampler.file_path,
                env_type=DummyEnv,
                problem_type="test",
                resume_step=4,
                puct_c=0.0,
                topk_children=0,
            )

        self.assertEqual(
            store["outcome_surface_by_state_id"],
            {"a": "surface_a", "b": "surface_b"},
        )
        self.assertNotIn("stale", store["outcome_surface_by_state_id"])
        self.assertNotIn("c", store["outcome_surface_by_state_id"])
        self.assertEqual(
            reloaded._outcome_surface_by_state_id,
            {"a": "surface_a", "b": "surface_b"},
        )
        self.assertEqual(
            reloaded._outcome_surface_sample_counts,
            {"surface_a": 2, "surface_b": 1, "surface_c": 1},
        )

    def test_extractor_uses_metrics_or_valid_correct_and_failed_results_are_ignored(self):
        self.assertEqual(
            _extract_outcome_surface(
                correct_format=True,
                correctness=1.0,
                msg="fast high score",
                metrics={},
            ),
            "valid_correct",
        )
        self.assertEqual(
            _extract_outcome_surface(
                correct_format=True,
                correctness=1.0,
                msg="",
                metrics={"outcome_surface": " Fast Pass! "},
            ),
            "fast_pass",
        )
        self.assertEqual(
            _extract_outcome_surface(
                correct_format=True,
                correctness=1.0,
                msg="",
                metrics={"codex/outcome_surface": "123 Weird Surface"},
            ),
            "outcome_123_weird_surface",
        )
        self.assertIsNone(
            _extract_outcome_surface(
                correct_format=False,
                correctness=0.0,
                msg="compile failed",
                metrics={},
            )
        )

        parent = make_state("parent", 0.0, code="plain parent")
        child = make_state(
            "child",
            9999.0,
            code="import triton\n# code tokens must not become labels",
        )
        failed = make_result(
            parent,
            next_state=None,
            correct_format=False,
            correctness=0.0,
            outcome_surface="compile_error",
        )
        accepted = make_result(
            parent,
            next_state=child,
            correct_format=True,
            correctness=1.0,
            outcome_surface="valid_correct",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            sampler = make_sampler(tmpdir, [parent])
            _update_sampler_from_results(sampler, [accepted, failed])

        self.assertEqual(
            sampler._outcome_surface_by_state_id,
            {"child": "valid_correct"},
        )
        self.assertNotIn("compile_error", sampler._outcome_surface_by_state_id.values())
        self.assertNotIn("triton", sampler._outcome_surface_by_state_id["child"])
        self.assertNotIn("9999", sampler._outcome_surface_by_state_id["child"])

    def test_duplicate_results_do_not_overwrite_existing_creation_surface(self):
        parent = make_state("parent", 0.0)
        existing_child = make_state("child", 1.0)
        duplicate_child = State(
            timestep=2,
            construction=["child"],
            code="different response with same construction",
            value=2.0,
            id="duplicate-child",
        )
        duplicate = make_result(
            parent,
            next_state=duplicate_child,
            correct_format=True,
            correctness=1.0,
            outcome_surface="late_duplicate_surface",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            sampler = make_sampler(tmpdir, [parent, existing_child])
            sampler._outcome_surface_by_state_id = {"child": "creation_surface"}
            _update_sampler_from_results(sampler, [duplicate])

        self.assertEqual(duplicate.pool_status, "duplicate")
        self.assertEqual(
            sampler._outcome_surface_by_state_id,
            {"child": "creation_surface"},
        )

    def test_cli_and_run_script_enable_outcome_surface_only_when_requested(self):
        default_proc = subprocess.run(
            [sys.executable, "repro/run_discovery.py", "dry-run", "--task", "trimul"],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
        enabled_proc = subprocess.run(
            [
                sys.executable,
                "repro/run_discovery.py",
                "dry-run",
                "--task",
                "trimul",
                "--codex-parent-balance",
                "outcome_surface",
            ],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
        script_proc = subprocess.run(
            ["bash", "repro/gpu_mode/run_0608_outcome_surface_balance.sh", "--dry-run"],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=True,
        )

        self.assertIn("codex_parent_balance='none'", default_proc.stdout)
        self.assertNotIn("codex_parent_balance='outcome_surface'", default_proc.stdout)
        self.assertIn("codex_parent_balance='outcome_surface'", enabled_proc.stdout)
        self.assertIn("codex_parent_balance='outcome_surface'", script_proc.stdout)


if __name__ == "__main__":
    unittest.main()
````

## Diff Stat

### Worktree Tracked Diff Stat

````text
 ttt_discover/codex_utils/discovery.py |   2 +
 ttt_discover/codex_utils/sampler.py   | 149 ++++++++++++++++++++++++++++++++++
 ttt_discover/discovery.py             |   2 +
 ttt_discover/rl/codex_no_finetune.py  |  85 ++++++++++++++++++-
 4 files changed, 237 insertions(+), 1 deletion(-)
````
## Raw Diff

<details>
<summary>Tracked worktree patch</summary>

````diff
diff --git a/ttt_discover/codex_utils/discovery.py b/ttt_discover/codex_utils/discovery.py
index b21a80f..9c397d3 100644
--- a/ttt_discover/codex_utils/discovery.py
+++ b/ttt_discover/codex_utils/discovery.py
@@ -50,6 +50,7 @@ class DiscoverConfig:
     codex_initial_program_paths: tuple[str, ...] = ()
     codex_initial_pool_paths: tuple[str, ...] = ()
     codex_autonomous: bool = False
+    codex_parent_balance: Literal["none", "outcome_surface"] = "none"
 
 
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
index 5c5d4d1..68cfbe2 100644
--- a/ttt_discover/codex_utils/sampler.py
+++ b/ttt_discover/codex_utils/sampler.py
@@ -375,6 +375,8 @@ class PUCTSampler(StateSampler):
         self._T: int = 0
         self._last_scale: float = 1.0
         self._last_puct_stats: list[tuple[int, float, float, float, float]] = []
+        self._outcome_surface_by_state_id: dict[str, str] = {}
+        self._outcome_surface_sample_counts: dict[str, int] = {}
         
         if resume_step is not None:
             self._load(resume_step)
@@ -399,10 +401,28 @@ class PUCTSampler(StateSampler):
         self._n = store.get("puct_n", {}) or {}
         self._m = store.get("puct_m", {}) or {}
         self._T = int(store.get("puct_T", 0) or 0)
+        state_ids = {str(s.id) for s in self._states}
+        raw_surfaces = store.get("outcome_surface_by_state_id", {}) or {}
+        self._outcome_surface_by_state_id = {
+            str(state_id): surface
+            for state_id, surface in raw_surfaces.items()
+            if str(state_id) in state_ids and isinstance(surface, str) and surface
+        }
+        raw_counts = store.get("outcome_surface_sample_counts", {}) or {}
+        counts: dict[str, int] = {}
+        for surface, count in raw_counts.items():
+            if not isinstance(surface, str) or not surface:
+                continue
+            try:
+                counts[surface] = max(0, int(count))
+            except (TypeError, ValueError):
+                continue
+        self._outcome_surface_sample_counts = counts
 
     def _save(self, step: int):
         save_path = _sampler_file_for_step(self.file_path, step)
         os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
+        self._prune_outcome_surface_sidecars()
         store = {
             "step": step,
             "states": [s.to_dict() for s in self._states],
@@ -410,10 +430,45 @@ class PUCTSampler(StateSampler):
             "puct_n": self._n,
             "puct_m": self._m,
             "puct_T": self._T,
+            "outcome_surface_by_state_id": self._outcome_surface_by_state_id,
+            "outcome_surface_sample_counts": self._outcome_surface_sample_counts,
         }
         with _file_lock(f"{save_path}.lock"):
             _atomic_write_json(save_path, store)
 
+    def _prune_outcome_surface_sidecars(self) -> None:
+        state_ids = {str(s.id) for s in self._states}
+        self._outcome_surface_by_state_id = {
+            state_id: surface
+            for state_id, surface in self._outcome_surface_by_state_id.items()
+            if state_id in state_ids
+        }
+
+    def _outcome_surface_for_state(self, state: State) -> str:
+        surface = self._outcome_surface_by_state_id.get(str(state.id))
+        return surface if surface else "unknown"
+
+    def _record_outcome_surface_samples(self, states: list[State]) -> None:
+        for state in states:
+            surface = self._outcome_surface_for_state(state)
+            self._outcome_surface_sample_counts[surface] = (
+                self._outcome_surface_sample_counts.get(surface, 0) + 1
+            )
+
+    def record_state_outcome_surface(self, state: State, surface: str | None) -> bool:
+        if not surface:
+            return False
+        surface = str(surface).strip()
+        if not surface:
+            return False
+
+        target_id = str(state.id)
+        for existing in self._states:
+            if str(existing.id) == target_id:
+                self._outcome_surface_by_state_id[target_id] = surface
+                return True
+        return False
+
     def _refresh_random_construction(self, state: State) -> None:
         """Let task-specific envs refresh seed states without coupling sampler to a task."""
         refresh = getattr(self.env_type, "refresh_initial_state", None)
@@ -548,6 +603,100 @@ class PUCTSampler(StateSampler):
 
         return picked
 
+    def sample_states_outcome_surface_balanced(self, num_states: int) -> list[State]:
+        if num_states <= 0:
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
+        surface_entries: dict[str, list[tuple[int, tuple]]] = {}
+        surface_order: list[str] = []
+        for rank, entry in enumerate(scores):
+            surface = self._outcome_surface_for_state(entry[2])
+            if surface not in surface_entries:
+                surface_order.append(surface)
+                surface_entries[surface] = []
+            surface_entries[surface].append((rank, entry))
+
+        surfaces_by_need = sorted(
+            surface_order,
+            key=lambda surface: (
+                self._outcome_surface_sample_counts.get(surface, 0),
+                surface_entries[surface][0][0],
+                surface,
+            ),
+        )
+
+        if num_states == 1:
+            surface = surfaces_by_need[0]
+            top_scores = [surface_entries[surface][0][1]]
+            picked = [top_scores[0][2]]
+        else:
+            children_map = self._build_children_map()
+            picked: list[State] = []
+            top_scores = []
+            picked_ids: set[str] = set()
+            blocked_ids: set[str] = set()
+
+            for surface in surfaces_by_need:
+                for _rank, entry in surface_entries[surface]:
+                    s = entry[2]
+                    if s.id in blocked_ids:
+                        continue
+                    picked.append(s)
+                    top_scores.append(entry)
+                    picked_ids.add(s.id)
+                    blocked_ids.update(self._get_full_lineage(s, children_map))
+                    break
+                if len(picked) >= num_states:
+                    break
+
+            if len(picked) < num_states:
+                for entry in scores:
+                    s = entry[2]
+                    if s.id in picked_ids or s.id in blocked_ids:
+                        continue
+                    picked.append(s)
+                    top_scores.append(entry)
+                    picked_ids.add(s.id)
+                    blocked_ids.update(self._get_full_lineage(s, children_map))
+                    if len(picked) >= num_states:
+                        break
+
+        state_id_to_idx = {s.id: i for i, s in enumerate(self._states)}
+        self._last_sampled_states = picked
+        self._last_sampled_indices = [state_id_to_idx.get(s.id, -1) for s in picked]
+        self._last_puct_stats = [(t[3], t[4], t[5], t[6], t[0]) for t in top_scores]
+        self._record_outcome_surface_samples(picked)
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
index 6cdb434..4860a0e 100644
--- a/ttt_discover/discovery.py
+++ b/ttt_discover/discovery.py
@@ -63,6 +63,7 @@ class DiscoverConfig:
     codex_initial_program_paths: tuple[str, ...] = ()
     codex_initial_pool_paths: tuple[str, ...] = ()
     codex_autonomous: bool = False
+    codex_parent_balance: Literal["none", "outcome_surface"] = "none"
 
 
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
index 18a0e9e..15a1c09 100644
--- a/ttt_discover/rl/codex_no_finetune.py
+++ b/ttt_discover/rl/codex_no_finetune.py
@@ -103,9 +103,11 @@ class CandidateResult:
     parsed_code: str
     reward: float
     correctness: float
+    correct_format: bool
     raw_score: float | None
     msg: str
     metrics: dict[str, Any]
+    outcome_surface: str | None = None
     next_state: Any | None = None
     error: str | None = None
     kept: bool = True
@@ -137,6 +139,7 @@ class CodexNoFinetuneConfig:
     initial_program_paths: tuple[str, ...] = ()
     initial_pool_paths: tuple[str, ...] = ()
     topk_children: int = 16
+    parent_balance: Literal["none", "outcome_surface"] = "none"
 
     # False is TTT Discover: non-auto, many short samples. True is AutoEvolve:
     # one Codex deep-dive workspace per sample with file edits and shell access.
@@ -471,6 +474,42 @@ def _maybe_create_next_state(env: Any, step_idx: int, parsed_code: str, outs: Ve
     return create_next_state(step_idx, parsed_code, outs)
 
 
+_OUTCOME_SURFACE_TOKEN_RE = re.compile(r"[^a-z0-9_.-]+")
+
+
+def _sanitize_outcome_surface(value: Any) -> str | None:
+    if not isinstance(value, str):
+        return None
+    surface = value.strip().lower()
+    if not surface:
+        return None
+    surface = _OUTCOME_SURFACE_TOKEN_RE.sub("_", surface)
+    surface = re.sub(r"_+", "_", surface).strip("_.-")
+    if not surface:
+        return None
+    if not re.match(r"^[a-z_]", surface):
+        surface = f"outcome_{surface}"
+    return surface[:80]
+
+
+def _extract_outcome_surface(
+    *,
+    correct_format: bool,
+    correctness: float,
+    msg: str,
+    metrics: dict[str, Any] | None,
+) -> str | None:
+    del msg
+    metrics = metrics or {}
+    for key in ("outcome_surface", "codex/outcome_surface"):
+        surface = _sanitize_outcome_surface(metrics.get(key))
+        if surface is not None:
+            return surface
+    if correct_format and correctness > 0:
+        return "valid_correct"
+    return None
+
+
 class AutonomousCodexCliCompleter(CodexCliCompleter):
     """Minimal Codex CLI agent mode: give Codex a workspace and evaluator."""
 
@@ -678,6 +717,14 @@ async def _run_candidate(
         metrics["codex/parsed_code_source"] = parsed_code_source
         if autonomous_submission_path is not None:
             metrics["codex/autonomous_submission_path"] = autonomous_submission_path
+        outcome_surface = _extract_outcome_surface(
+            correct_format=correct_format,
+            correctness=float(outs.correctness),
+            msg=outs.msg,
+            metrics=metrics,
+        )
+        if outcome_surface is not None:
+            metrics["codex/outcome_surface_parent_balance"] = outcome_surface
         next_state = _maybe_create_next_state(env, step_idx, parsed_code, outs)
         return CandidateResult(
             parent_state=parent_state,
@@ -688,9 +735,11 @@ async def _run_candidate(
             parsed_code=parsed_code,
             reward=float(outs.reward),
             correctness=float(outs.correctness),
+            correct_format=bool(correct_format),
             raw_score=outs.raw_score,
             msg=outs.msg,
             metrics=metrics,
+            outcome_surface=outcome_surface,
             next_state=next_state,
         )
     except Exception as exc:
@@ -711,14 +760,32 @@ async def _run_candidate(
             parsed_code=parsed_code,
             reward=0.0,
             correctness=0.0,
+            correct_format=False,
             raw_score=None,
             msg=error_msg,
             metrics={"error": error_msg},
+            outcome_surface=None,
             next_state=None,
             error=error_msg,
         )
 
 
+def _record_result_outcome_surface(sampler: StateSampler, result: CandidateResult) -> None:
+    if (
+        result.next_state is None
+        or result.outcome_surface is None
+        or result.pool_status != "added_to_pool"
+    ):
+        return
+    recorder = getattr(sampler, "record_state_outcome_surface", None)
+    if not callable(recorder):
+        return
+    try:
+        recorder(result.next_state, result.outcome_surface)
+    except Exception as exc:
+        logger.warning("Failed to record outcome surface for sampler state: %s", exc)
+
+
 def _update_sampler_from_results(sampler: StateSampler, results: list[CandidateResult]) -> None:
     for result in results:
         if result.next_state is not None:
@@ -745,6 +812,7 @@ def _update_sampler_from_results(sampler: StateSampler, results: list[CandidateR
                         result.pool_status = "updated_unknown"
                 else:
                     result.pool_status = "updated_unknown"
+                _record_result_outcome_surface(sampler, result)
                 continue
             except Exception as exc:
                 logger.warning("Failed to update sampler with new state: %s", exc)
@@ -775,6 +843,21 @@ def _sample_table(results: list[CandidateResult]) -> list[tuple[Any, ...]]:
     return rows
 
 
+def _sample_parent_states(cfg: CodexNoFinetuneConfig, sampler: StateSampler) -> list[Any]:
+    parent_balance = getattr(cfg, "parent_balance", "none")
+    if parent_balance == "none":
+        return sampler.sample_states(cfg.groups_per_batch)
+    if parent_balance == "outcome_surface":
+        sample_balanced = getattr(sampler, "sample_states_outcome_surface_balanced", None)
+        if not callable(sample_balanced):
+            raise ValueError(
+                "outcome_surface parent_balance requires sampler "
+                "sample_states_outcome_surface_balanced"
+            )
+        return sample_balanced(cfg.groups_per_batch)
+    raise ValueError(f"Unknown parent_balance: {parent_balance}")
+
+
 def _result_metrics(results: list[CandidateResult], kept_results: list[CandidateResult]) -> dict[str, Any]:
     metrics: dict[str, Any] = {}
     rewards = [result.reward for result in kept_results]
@@ -820,7 +903,7 @@ async def sample_batch(
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

