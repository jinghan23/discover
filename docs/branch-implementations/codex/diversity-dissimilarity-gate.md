# codex/diversity-dissimilarity-gate

## Summary

把 construction/code/observation 等内容 token 化，用 Jaccard similarity 对已选样本做 sequential gate；超过阈值的相似候选先跳过，最后回退。

## Branch State

- Worktree: `/opt/tiger/discover-dissimilarity-gate`
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

- `codex_dissimilarity_gate_sampling`
- `dissimilarity_gate_sampling`

### Constants

- `_DISSIMILARITY_GATE_THRESHOLD`

### Classes

- None

### Functions

- `_ranked_puct_entries`
- `_clear_dissimilarity_gate_cache`
- `_set_dissimilarity_gate_cache`
- `_set_last_sampled`
- `_set_last_sampled_without_puct`
- `_refresh_sampled_initials`
- `_state_token_set`
- `_token_set_from_value`
- `add_text`
- `visit`
- `_jaccard_similarity`
- `_max_similarity_to_prior`
- `_sequential_max_similarities`
- `sample_states_dissimilarity_gate`
- `pick`

## Diff Summary

- Worktree tracked shortstat: `4 files changed, 385 insertions(+), 34 deletions(-)`
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

- `repro/gpu_mode/run_0608_dissimilarity_gate.sh (958 bytes)`
- `repro/run_discovery.py (15871 bytes)`
- `tests/test_dissimilarity_gate_sampling.py (13739 bytes)`

### Detected Test Functions

- `tests/test_dissimilarity_gate_sampling.py::test_num_states_le_one_matches_ordinary_puct`
- `tests/test_dissimilarity_gate_sampling.py::test_duplicate_high_puct_candidate_is_recovered_by_fallback`
- `tests/test_dissimilarity_gate_sampling.py::test_jaccard_boundary_keeps_equal_threshold_and_skips_above`
- `tests/test_dissimilarity_gate_sampling.py::test_full_lineage_blocking_applies_in_first_pass_and_fallback`
- `tests/test_dissimilarity_gate_sampling.py::test_seed_parentless_and_empty_tokens_are_first_pass_ineligible`
- `tests/test_dissimilarity_gate_sampling.py::test_token_source_priority_and_generic_values`
- `tests/test_dissimilarity_gate_sampling.py::test_sample_table_cache_survives_update_and_flush`
- `tests/test_dissimilarity_gate_sampling.py::test_sample_batch_uses_gate_only_when_flag_enabled`
- `tests/test_dissimilarity_gate_sampling.py::test_ordinary_table_schema_is_unchanged`

## Untracked File Snapshots

These files are not part of `git diff` yet, but they are part of the worktree implementation state recorded here.

### `repro/gpu_mode/run_0608_dissimilarity_gate.sh`

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
  --experiment-name "${EXPERIMENT_NAME:-gpu-mode-trimul-dissimilarity-gate-0608}" \
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
  --codex-dissimilarity-gate-sampling \
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
        "--codex-dissimilarity-gate-sampling",
        action="store_true",
        help="Use dissimilarity-gated PUCT parent sampling.",
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
            "codex_dissimilarity_gate_sampling="
            f"{args.codex_dissimilarity_gate_sampling}"
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
            dissimilarity_gate_sampling=args.codex_dissimilarity_gate_sampling,
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
        codex_dissimilarity_gate_sampling=args.codex_dissimilarity_gate_sampling,
        codex_autonomous=args.codex_autonomous,
        remove_constant_reward_groups=remove_constant_reward_groups,
    )
    discover(config)


if __name__ == "__main__":
    main()
````

### `tests/test_dissimilarity_gate_sampling.py`

````python
from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ttt_discover.codex_utils.runtime import State
from ttt_discover.codex_utils.sampler import PUCTSampler
from ttt_discover.rl.codex_no_finetune import CodexNoFinetuneConfig, sample_batch


class DummyEnv:
    state_type = State
    _seed_counter = 0

    @classmethod
    def create_initial_state(cls, problem_type: str) -> State:
        cls._seed_counter += 1
        return State(
            timestep=0,
            construction=[f"initial-{cls._seed_counter}", problem_type],
            code="",
            value=0.0,
            id=f"initial-{cls._seed_counter}",
        )


def make_state(
    state_id: str,
    value: float,
    construction,
    *,
    code: str = "",
    parents: list[dict] | None = None,
    timestep: int = 1,
) -> State:
    parent_values = [0.0] if parents else []
    return State(
        timestep=timestep,
        construction=construction,
        code=code,
        value=value,
        parent_values=parent_values,
        parents=parents or [],
        id=state_id,
    )


class DissimilarityGateSamplingTest(unittest.TestCase):
    def make_sampler(
        self,
        states: list[State],
        *,
        initial_states: list[State] | None = None,
    ) -> tuple[PUCTSampler, tempfile.TemporaryDirectory[str]]:
        tmp = tempfile.TemporaryDirectory()
        sampler = PUCTSampler(
            file_path=str(Path(tmp.name) / "sampler.json"),
            env_type=DummyEnv,
            batch_size=0,
            puct_c=0.0,
            topk_children=0,
        )
        sampler._states = list(states)
        sampler._initial_states = list(initial_states or [])
        return sampler, tmp

    def table_rows_by_id(self, sampler: PUCTSampler) -> tuple[list[str], dict[str, tuple]]:
        columns, rows = sampler.get_sample_table()
        id_by_idx = {idx: state.id for idx, state in enumerate(sampler._states)}
        return columns, {id_by_idx[row[0]]: row for row in rows}

    def test_num_states_le_one_matches_ordinary_puct(self):
        states = [
            make_state("a", 10.0, ["alpha"], parents=[{"id": "root-a"}]),
            make_state("b", 9.0, ["alpha"], parents=[{"id": "root-b"}]),
        ]
        ordinary_sampler, ordinary_tmp = self.make_sampler(states)
        gate_sampler, gate_tmp = self.make_sampler(states)
        self.addCleanup(ordinary_tmp.cleanup)
        self.addCleanup(gate_tmp.cleanup)

        self.assertEqual(ordinary_sampler.sample_states(0), [])
        self.assertEqual(gate_sampler.sample_states_dissimilarity_gate(0), [])
        self.assertEqual(
            [state.id for state in ordinary_sampler.sample_states(1)],
            [state.id for state in gate_sampler.sample_states_dissimilarity_gate(1)],
        )

    def test_duplicate_high_puct_candidate_is_recovered_by_fallback(self):
        states = [
            make_state("a", 10.0, ["same"], parents=[{"id": "root-a"}]),
            make_state("b", 9.0, ["same"], parents=[{"id": "root-b"}]),
        ]
        sampler, tmp = self.make_sampler(states)
        self.addCleanup(tmp.cleanup)

        picked = sampler.sample_states_dissimilarity_gate(2)

        self.assertEqual([state.id for state in picked], ["a", "b"])
        columns, rows_by_id = self.table_rows_by_id(sampler)
        fallback_idx = columns.index("dissimilarity_gate_fallback")
        max_sim_idx = columns.index("max_similarity_to_prior")
        self.assertFalse(rows_by_id["a"][fallback_idx])
        self.assertTrue(rows_by_id["b"][fallback_idx])
        self.assertIsNone(rows_by_id["a"][max_sim_idx])
        self.assertEqual(rows_by_id["b"][max_sim_idx], 1.0)
        stats = sampler.get_sample_stats()
        self.assertEqual(stats["puct/dissimilarity_gate_fallback_count"], 1)

    def test_jaccard_boundary_keeps_equal_threshold_and_skips_above(self):
        states = [
            make_state("a", 10.0, ["a", "b", "c"], parents=[{"id": "root-a"}]),
            make_state("b", 9.0, ["a", "b", "c", "d"], parents=[{"id": "root-b"}]),
            make_state("c", 8.0, ["a", "b", "c", "d", "e"], parents=[{"id": "root-c"}]),
        ]
        sampler, tmp = self.make_sampler(states)
        self.addCleanup(tmp.cleanup)

        picked = sampler.sample_states_dissimilarity_gate(3)

        self.assertEqual([state.id for state in picked], ["a", "b", "c"])
        columns, rows_by_id = self.table_rows_by_id(sampler)
        fallback_idx = columns.index("dissimilarity_gate_fallback")
        max_sim_idx = columns.index("max_similarity_to_prior")
        self.assertFalse(rows_by_id["b"][fallback_idx])
        self.assertEqual(rows_by_id["b"][max_sim_idx], 0.75)
        self.assertTrue(rows_by_id["c"][fallback_idx])
        self.assertGreater(rows_by_id["c"][max_sim_idx], 0.75)

    def test_full_lineage_blocking_applies_in_first_pass_and_fallback(self):
        states = [
            make_state("a", 100.0, ["alpha"], parents=[{"id": "root-a"}]),
            make_state("d", 90.0, ["delta"], parents=[{"id": "a"}]),
            make_state("b", 80.0, ["alpha"], parents=[{"id": "root-b"}]),
            make_state("c", 70.0, ["alpha"], parents=[{"id": "b"}]),
            make_state("e", 60.0, ["epsilon"], parents=[{"id": "root-e"}]),
        ]
        sampler, tmp = self.make_sampler(states)
        self.addCleanup(tmp.cleanup)

        picked = sampler.sample_states_dissimilarity_gate(5)

        self.assertEqual([state.id for state in picked], ["a", "e", "b"])
        self.assertNotIn("d", [state.id for state in picked])
        self.assertNotIn("c", [state.id for state in picked])
        columns, rows_by_id = self.table_rows_by_id(sampler)
        fallback_idx = columns.index("dissimilarity_gate_fallback")
        self.assertTrue(rows_by_id["b"][fallback_idx])

    def test_seed_parentless_and_empty_tokens_are_first_pass_ineligible(self):
        seed = make_state("seed", 100.0, ["seed"], timestep=0)
        parentless = make_state("parentless", 90.0, ["parentless"])
        empty = make_state("empty", 80.0, [], parents=[{"id": "root-empty"}])
        eligible = make_state("eligible", 10.0, ["eligible"], parents=[{"id": "root-ok"}])
        sampler, tmp = self.make_sampler(
            [seed, parentless, empty, eligible],
            initial_states=[seed],
        )
        self.addCleanup(tmp.cleanup)

        picked = sampler.sample_states_dissimilarity_gate(4)

        self.assertEqual([state.id for state in picked], ["eligible", "seed", "parentless", "empty"])
        columns, rows_by_id = self.table_rows_by_id(sampler)
        fallback_idx = columns.index("dissimilarity_gate_fallback")
        empty_idx = columns.index("dissimilarity_gate_empty")
        self.assertFalse(rows_by_id["eligible"][fallback_idx])
        self.assertTrue(rows_by_id["seed"][fallback_idx])
        self.assertTrue(rows_by_id["parentless"][fallback_idx])
        self.assertTrue(rows_by_id["empty"][fallback_idx])
        self.assertTrue(rows_by_id["empty"][empty_idx])

        initial_one = make_state("initial-one", 2.0, ["one"], timestep=0)
        initial_two = make_state("initial-two", 1.0, ["two"], timestep=0)
        initial_sampler, initial_tmp = self.make_sampler(
            [initial_one, initial_two],
            initial_states=[initial_one, initial_two],
        )
        self.addCleanup(initial_tmp.cleanup)
        self.assertEqual(
            [state.id for state in initial_sampler.sample_states_dissimilarity_gate(2)],
            ["initial-one", "initial-two"],
        )

    def test_token_source_priority_and_generic_values(self):
        construction_sampler, construction_tmp = self.make_sampler(
            [
                make_state("a", 10.0, ["alpha"], code="shared", parents=[{"id": "root-a"}]),
                make_state("b", 9.0, ["beta"], code="shared", parents=[{"id": "root-b"}]),
            ]
        )
        self.addCleanup(construction_tmp.cleanup)
        construction_sampler.sample_states_dissimilarity_gate(2)
        columns, rows_by_id = self.table_rows_by_id(construction_sampler)
        fallback_idx = columns.index("dissimilarity_gate_fallback")
        self.assertFalse(rows_by_id["b"][fallback_idx])

        code_sampler, code_tmp = self.make_sampler(
            [
                make_state("c", 10.0, None, code="gamma", parents=[{"id": "root-c"}]),
                make_state("d", 9.0, None, code="gamma", parents=[{"id": "root-d"}]),
            ]
        )
        self.addCleanup(code_tmp.cleanup)
        code_sampler.sample_states_dissimilarity_gate(2)
        columns, rows_by_id = self.table_rows_by_id(code_sampler)
        self.assertTrue(rows_by_id["d"][columns.index("dissimilarity_gate_fallback")])

        empty_construction_sampler, empty_construction_tmp = self.make_sampler(
            [
                make_state("e", 10.0, [], code="zeta", parents=[{"id": "root-e"}]),
                make_state("f", 9.0, [], code="zeta", parents=[{"id": "root-f"}]),
            ]
        )
        self.addCleanup(empty_construction_tmp.cleanup)
        empty_construction_sampler.sample_states_dissimilarity_gate(2)
        columns, rows_by_id = self.table_rows_by_id(empty_construction_sampler)
        self.assertTrue(
            rows_by_id["f"][columns.index("dissimilarity_gate_fallback")]
        )

        ndarray_state = make_state("nd", 10.0, [], parents=[{"id": "root-nd"}])
        ndarray_state.construction = np.array(["array", "tokens"])
        mixed_sampler, mixed_tmp = self.make_sampler(
            [
                ndarray_state,
                make_state("dict", 9.0, {"key": ["value", 3]}, parents=[{"id": "root-dict"}]),
                make_state("list", 8.0, ["tuple", ("nested", 4)], parents=[{"id": "root-list"}]),
                make_state("none", 7.0, None, code="", parents=[{"id": "root-none"}]),
            ]
        )
        self.addCleanup(mixed_tmp.cleanup)
        picked = mixed_sampler.sample_states_dissimilarity_gate(4)
        self.assertEqual(len(picked), 4)

    def test_sample_table_cache_survives_update_and_flush(self):
        states = [
            make_state("a", 10.0, ["same"], parents=[{"id": "root-a"}]),
            make_state("b", 9.0, ["same"], parents=[{"id": "root-b"}]),
        ]
        sampler, tmp = self.make_sampler(states)
        self.addCleanup(tmp.cleanup)
        sampler.sample_states_dissimilarity_gate(2)
        before_columns, before_rows = sampler.get_sample_table()

        child = make_state("child", 11.0, ["child"], parents=[])
        sampler.update_states([child], [states[0]], save=False)
        sampler.flush(step=1)
        after_columns, after_rows = sampler.get_sample_table()

        self.assertIn("token_count", before_columns)
        self.assertEqual(before_columns, after_columns)
        self.assertEqual(len(before_rows), len(after_rows))
        self.assertEqual(
            [row[before_columns.index("token_count")] for row in before_rows],
            [row[after_columns.index("token_count")] for row in after_rows],
        )

        saved = json.loads((Path(tmp.name) / "sampler_step_000001.json").read_text())
        self.assertFalse(any("dissimilarity" in key for key in saved))

    def test_sample_batch_uses_gate_only_when_flag_enabled(self):
        class OrdinaryOnlySampler:
            def sample_states(self, num_states: int) -> list[State]:
                return []

            def update_states(self, states, parent_states, save=True, step=None):
                raise AssertionError("no results should be updated")

        cfg = CodexNoFinetuneConfig(
            env_type=DummyEnv,
            groups_per_batch=0,
            group_size=1,
            dissimilarity_gate_sampling=True,
        )
        with self.assertRaisesRegex(ValueError, "sample_states_dissimilarity_gate"):
            asyncio.run(sample_batch(cfg, OrdinaryOnlySampler(), 0))

        class TrackingSampler:
            def __init__(self):
                self.ordinary_calls = 0
                self.gate_calls = 0

            def sample_states(self, num_states: int) -> list[State]:
                self.ordinary_calls += 1
                return []

            def sample_states_dissimilarity_gate(self, num_states: int) -> list[State]:
                self.gate_calls += 1
                raise AssertionError("gate sampler should not be called")

        tracking = TrackingSampler()
        cfg = CodexNoFinetuneConfig(
            env_type=DummyEnv,
            groups_per_batch=0,
            group_size=1,
            dissimilarity_gate_sampling=False,
        )
        asyncio.run(sample_batch(cfg, tracking, 0))
        self.assertEqual(tracking.ordinary_calls, 1)
        self.assertEqual(tracking.gate_calls, 0)

    def test_ordinary_table_schema_is_unchanged(self):
        states = [
            make_state("a", 10.0, ["alpha"], parents=[{"id": "root-a"}]),
            make_state("b", 9.0, ["beta"], parents=[{"id": "root-b"}]),
        ]
        sampler, tmp = self.make_sampler(states)
        self.addCleanup(tmp.cleanup)

        sampler.sample_states(2)
        columns, rows = sampler.get_sample_table()

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
        self.assertEqual(len(rows), 2)


if __name__ == "__main__":
    unittest.main()
````

## Diff Stat

### Worktree Tracked Diff Stat

````text
 ttt_discover/codex_utils/discovery.py |   2 +
 ttt_discover/codex_utils/sampler.py   | 399 +++++++++++++++++++++++++++++++---
 ttt_discover/discovery.py             |   2 +
 ttt_discover/rl/codex_no_finetune.py  |  16 +-
 4 files changed, 385 insertions(+), 34 deletions(-)
````
## Raw Diff

<details>
<summary>Tracked worktree patch</summary>

````diff
diff --git a/ttt_discover/codex_utils/discovery.py b/ttt_discover/codex_utils/discovery.py
index b21a80f..dd6d26f 100644
--- a/ttt_discover/codex_utils/discovery.py
+++ b/ttt_discover/codex_utils/discovery.py
@@ -49,6 +49,7 @@ class DiscoverConfig:
     codex_max_concurrent_requests: int | None = 4
     codex_initial_program_paths: tuple[str, ...] = ()
     codex_initial_pool_paths: tuple[str, ...] = ()
+    codex_dissimilarity_gate_sampling: bool = False
     codex_autonomous: bool = False
 
 
@@ -84,6 +85,7 @@ def _run_codex_no_finetune(config: DiscoverConfig) -> None:
         max_concurrent_requests=config.codex_max_concurrent_requests,
         initial_program_paths=config.codex_initial_program_paths,
         initial_pool_paths=config.codex_initial_pool_paths,
+        dissimilarity_gate_sampling=config.codex_dissimilarity_gate_sampling,
         autonomous=config.codex_autonomous,
         wandb_project=config.wandb_project,
         wandb_name=experiment_name,
diff --git a/ttt_discover/codex_utils/sampler.py b/ttt_discover/codex_utils/sampler.py
index 5c5d4d1..6f42f77 100644
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
 
+_DISSIMILARITY_GATE_THRESHOLD = 0.75
+
 
 @contextmanager
 def _file_lock(lock_path: str, *, poll_s: float = 0.05, stale_s: float = 600.0):
@@ -375,6 +378,11 @@ class PUCTSampler(StateSampler):
         self._T: int = 0
         self._last_scale: float = 1.0
         self._last_puct_stats: list[tuple[int, float, float, float, float]] = []
+        self._last_dissimilarity_gate_active: bool = False
+        self._last_dissimilarity_gate_token_counts: list[int] = []
+        self._last_dissimilarity_gate_max_similarities: list[float | None] = []
+        self._last_dissimilarity_gate_fallback_flags: list[bool] = []
+        self._last_dissimilarity_gate_empty_flags: list[bool] = []
         
         if resume_step is not None:
             self._load(resume_step)
@@ -460,6 +468,191 @@ class PUCTSampler(StateSampler):
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
+    def _clear_dissimilarity_gate_cache(self) -> None:
+        self._last_dissimilarity_gate_active = False
+        self._last_dissimilarity_gate_token_counts = []
+        self._last_dissimilarity_gate_max_similarities = []
+        self._last_dissimilarity_gate_fallback_flags = []
+        self._last_dissimilarity_gate_empty_flags = []
+
+    def _set_dissimilarity_gate_cache(
+        self,
+        *,
+        token_counts: list[int],
+        max_similarities: list[float | None],
+        fallback_flags: list[bool],
+        empty_flags: list[bool],
+    ) -> None:
+        self._last_dissimilarity_gate_active = True
+        self._last_dissimilarity_gate_token_counts = list(token_counts)
+        self._last_dissimilarity_gate_max_similarities = list(max_similarities)
+        self._last_dissimilarity_gate_fallback_flags = list(fallback_flags)
+        self._last_dissimilarity_gate_empty_flags = list(empty_flags)
+
+    def _set_last_sampled(
+        self,
+        picked: list[State],
+        top_scores: list[tuple[float, float, State, int, float, float, float]],
+        *,
+        dissimilarity_gate_cache: tuple[
+            list[int],
+            list[float | None],
+            list[bool],
+            list[bool],
+        ] | None = None,
+    ) -> None:
+        state_id_to_idx = {s.id: i for i, s in enumerate(self._states)}
+        self._last_sampled_states = picked
+        self._last_sampled_indices = [state_id_to_idx.get(s.id, -1) for s in picked]
+        self._last_puct_stats = [(t[3], t[4], t[5], t[6], t[0]) for t in top_scores]
+        if dissimilarity_gate_cache is None:
+            self._clear_dissimilarity_gate_cache()
+        else:
+            token_counts, max_similarities, fallback_flags, empty_flags = dissimilarity_gate_cache
+            self._set_dissimilarity_gate_cache(
+                token_counts=token_counts,
+                max_similarities=max_similarities,
+                fallback_flags=fallback_flags,
+                empty_flags=empty_flags,
+            )
+
+    def _set_last_sampled_without_puct(
+        self,
+        picked: list[State],
+        *,
+        dissimilarity_gate_active: bool = False,
+    ) -> None:
+        self._last_sampled_states = picked
+        self._last_sampled_indices = []
+        self._last_puct_stats = [(0, 0.0, 0.0, 0.0, 0.0) for _ in picked]
+        if dissimilarity_gate_active:
+            token_sets = [self._state_token_set(s) for s in picked]
+            self._set_dissimilarity_gate_cache(
+                token_counts=[len(tokens) for tokens in token_sets],
+                max_similarities=self._sequential_max_similarities(token_sets),
+                fallback_flags=[False] * len(picked),
+                empty_flags=[not tokens for tokens in token_sets],
+            )
+        else:
+            self._clear_dissimilarity_gate_cache()
+
+    def _refresh_sampled_initials(self, picked: list[State], initial_ids: set[str]) -> None:
+        for s in picked:
+            if s.id in initial_ids:
+                self._refresh_random_construction(s)
+
+    def _state_token_set(self, state: State) -> set[str]:
+        if hasattr(state, "construction") and state.construction is not None:
+            construction_tokens = self._token_set_from_value(state.construction)
+            if construction_tokens:
+                return construction_tokens
+        return self._token_set_from_value(getattr(state, "code", None))
+
+    def _token_set_from_value(self, value: Any) -> set[str]:
+        tokens: set[str] = set()
+        seen: set[int] = set()
+
+        def add_text(raw: Any) -> None:
+            try:
+                text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)
+            except Exception:
+                text = repr(type(raw))
+            matches = re.findall(r"[A-Za-z0-9_]+", text.lower())
+            if matches:
+                tokens.update(matches)
+            elif text.strip():
+                tokens.add(text.strip().lower())
+
+        def visit(item: Any) -> None:
+            if item is None:
+                return
+            if isinstance(item, np.ndarray):
+                obj_id = id(item)
+                if obj_id in seen:
+                    return
+                seen.add(obj_id)
+                visit(item.tolist())
+                return
+            if isinstance(item, np.generic):
+                visit(item.item())
+                return
+            if isinstance(item, dict):
+                obj_id = id(item)
+                if obj_id in seen:
+                    return
+                seen.add(obj_id)
+                for key, val in item.items():
+                    visit(key)
+                    visit(val)
+                return
+            if isinstance(item, (list, tuple, set, frozenset)):
+                obj_id = id(item)
+                if obj_id in seen:
+                    return
+                seen.add(obj_id)
+                for child in item:
+                    visit(child)
+                return
+            add_text(item)
+
+        visit(value)
+        return tokens
+
+    @staticmethod
+    def _jaccard_similarity(a: set[str], b: set[str]) -> float:
+        union = a | b
+        if not union:
+            return 0.0
+        return len(a & b) / len(union)
+
+    def _max_similarity_to_prior(
+        self,
+        token_set: set[str],
+        prior_token_sets: list[set[str]],
+    ) -> float | None:
+        if not token_set or not prior_token_sets:
+            return None
+        return max(self._jaccard_similarity(token_set, prior) for prior in prior_token_sets)
+
+    def _sequential_max_similarities(
+        self,
+        token_sets: list[set[str]],
+    ) -> list[float | None]:
+        prior_token_sets: list[set[str]] = []
+        similarities: list[float | None] = []
+        for token_set in token_sets:
+            similarities.append(self._max_similarity_to_prior(token_set, prior_token_sets))
+            if token_set:
+                prior_token_sets.append(token_set)
+        return similarities
+
     def _get_lineage(self, state: State) -> set[str]:
         lineage = {state.id}
         for p in (state.parents or []):
@@ -490,37 +683,16 @@ class PUCTSampler(StateSampler):
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
@@ -537,14 +709,115 @@ class PUCTSampler(StateSampler):
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
+    def sample_states_dissimilarity_gate(self, num_states: int) -> list[State]:
+        """Sample by PUCT rank while gating near-duplicate parent token sets."""
+        scores, initial_ids = self._ranked_puct_entries()
+
+        if not scores and not self._states:
+            picked = [
+                create_initial_state(self.env_type, self.problem_type)
+                for _ in range(num_states)
+            ]
+            self._set_last_sampled_without_puct(
+                picked,
+                dissimilarity_gate_active=True,
+            )
+            return picked
+
+        if num_states <= 1:
+            top_scores = scores[:num_states]
+            picked = [t[2] for t in top_scores]
+            token_sets = [self._state_token_set(state) for state in picked]
+            self._set_last_sampled(
+                picked,
+                top_scores,
+                dissimilarity_gate_cache=(
+                    [len(tokens) for tokens in token_sets],
+                    self._sequential_max_similarities(token_sets),
+                    [False] * len(picked),
+                    [not tokens for tokens in token_sets],
+                ),
+            )
+            self._refresh_sampled_initials(picked, initial_ids)
+            return picked
+
+        children_map = self._build_children_map()
+        picked: list[State] = []
+        top_scores: list[tuple[float, float, State, int, float, float, float]] = []
+        picked_ids: set[str] = set()
+        blocked_ids: set[str] = set()
+        selected_token_sets: list[set[str]] = []
+        token_counts: list[int] = []
+        max_similarities: list[float | None] = []
+        fallback_flags: list[bool] = []
+        empty_flags: list[bool] = []
+
+        def pick(
+            entry: tuple[float, float, State, int, float, float, float],
+            token_set: set[str],
+            max_similarity: float | None,
+            *,
+            fallback: bool,
+        ) -> None:
+            state = entry[2]
+            picked.append(state)
+            top_scores.append(entry)
+            picked_ids.add(state.id)
+            blocked_ids.update(self._get_full_lineage(state, children_map))
+            token_counts.append(len(token_set))
+            max_similarities.append(max_similarity)
+            fallback_flags.append(fallback)
+            empty_flags.append(not token_set)
+            if token_set:
+                selected_token_sets.append(token_set)
+
+        for entry in scores:
+            if len(picked) >= num_states:
+                break
+            state = entry[2]
+            if state.id in blocked_ids:
+                continue
+            if state.id in initial_ids:
+                continue
+            if not (state.parents or []):
+                continue
+            token_set = self._state_token_set(state)
+            if not token_set:
+                continue
+            max_similarity = self._max_similarity_to_prior(token_set, selected_token_sets)
+            if (
+                max_similarity is not None
+                and max_similarity > _DISSIMILARITY_GATE_THRESHOLD
+            ):
+                continue
+            pick(entry, token_set, max_similarity, fallback=False)
+
+        for entry in scores:
+            if len(picked) >= num_states:
+                break
+            state = entry[2]
+            if state.id in picked_ids or state.id in blocked_ids:
+                continue
+            token_set = self._state_token_set(state)
+            max_similarity = self._max_similarity_to_prior(token_set, selected_token_sets)
+            pick(entry, token_set, max_similarity, fallback=True)
+
+        self._set_last_sampled(
+            picked,
+            top_scores,
+            dissimilarity_gate_cache=(
+                token_counts,
+                max_similarities,
+                fallback_flags,
+                empty_flags,
+            ),
+        )
+        self._refresh_sampled_initials(picked, initial_ids)
 
         return picked
 
@@ -731,21 +1004,81 @@ class PUCTSampler(StateSampler):
         stats.update(_stats(sampled_values, "puct/sampled_value"))
         stats.update(_stats(sampled_timesteps, "puct/sampled_timestep"))
         stats.update(_stats(sampled_constr_lens, "puct/sampled_construction_len"))
+        if self._last_dissimilarity_gate_active:
+            fallback_flags = (
+                self._last_dissimilarity_gate_fallback_flags
+                if len(self._last_dissimilarity_gate_fallback_flags) == len(self._last_sampled_states)
+                else [False] * len(self._last_sampled_states)
+            )
+            empty_flags = (
+                self._last_dissimilarity_gate_empty_flags
+                if len(self._last_dissimilarity_gate_empty_flags) == len(self._last_sampled_states)
+                else [False] * len(self._last_sampled_states)
+            )
+            max_similarities = (
+                self._last_dissimilarity_gate_max_similarities
+                if len(self._last_dissimilarity_gate_max_similarities) == len(self._last_sampled_states)
+                else [None] * len(self._last_sampled_states)
+            )
+            stats.update(_stats(self._last_dissimilarity_gate_token_counts, "puct/dissimilarity_gate_token_count"))
+            stats.update(
+                _stats(
+                    [value for value in max_similarities if value is not None],
+                    "puct/dissimilarity_gate_max_similarity_to_prior",
+                )
+            )
+            stats["puct/dissimilarity_gate_fallback_count"] = int(sum(fallback_flags))
+            stats["puct/dissimilarity_gate_empty_count"] = int(sum(empty_flags))
         return stats
 
     def get_sample_table(self) -> tuple[list[str], list[tuple]]:
         columns = ["buffer_idx", "timestep", "value", "terminal_value", "parent_value", "construction_len", "observation_len", "n", "Q", "P", "bonus", "score"]
+        if self._last_dissimilarity_gate_active:
+            columns += [
+                "token_count",
+                "max_similarity_to_prior",
+                "dissimilarity_gate_fallback",
+                "dissimilarity_gate_empty",
+            ]
         rows = []
         if not self._last_sampled_states:
             return columns, rows
         indices = self._last_sampled_indices if len(self._last_sampled_indices) == len(self._last_sampled_states) else [-1] * len(self._last_sampled_states)
         stats = self._last_puct_stats if len(self._last_puct_stats) == len(self._last_sampled_states) else [(0, 0.0, 0.0, 0.0, 0.0)] * len(self._last_sampled_states)
-        for idx, state, (n, Q, P, bonus, score) in zip(indices, self._last_sampled_states, stats):
+        token_counts = (
+            self._last_dissimilarity_gate_token_counts
+            if len(self._last_dissimilarity_gate_token_counts) == len(self._last_sampled_states)
+            else [0] * len(self._last_sampled_states)
+        )
+        max_similarities = (
+            self._last_dissimilarity_gate_max_similarities
+            if len(self._last_dissimilarity_gate_max_similarities) == len(self._last_sampled_states)
+            else [None] * len(self._last_sampled_states)
+        )
+        fallback_flags = (
+            self._last_dissimilarity_gate_fallback_flags
+            if len(self._last_dissimilarity_gate_fallback_flags) == len(self._last_sampled_states)
+            else [False] * len(self._last_sampled_states)
+        )
+        empty_flags = (
+            self._last_dissimilarity_gate_empty_flags
+            if len(self._last_dissimilarity_gate_empty_flags) == len(self._last_sampled_states)
+            else [False] * len(self._last_sampled_states)
+        )
+        for row_idx, (idx, state, (n, Q, P, bonus, score)) in enumerate(zip(indices, self._last_sampled_states, stats)):
             parent_val = state.parent_values[0] if state.parent_values else None
             constr = getattr(state, 'construction', None)
             constr_len = len(constr) if constr is not None else 0
             obs_len = len(state.observation) if state.observation else 0
-            rows.append((idx, state.timestep, state.value, 0, parent_val, constr_len, obs_len, n, Q, P, bonus, score))
+            row = (idx, state.timestep, state.value, 0, parent_val, constr_len, obs_len, n, Q, P, bonus, score)
+            if self._last_dissimilarity_gate_active:
+                row += (
+                    token_counts[row_idx],
+                    max_similarities[row_idx],
+                    fallback_flags[row_idx],
+                    empty_flags[row_idx],
+                )
+            rows.append(row)
         return columns, rows
 
 
diff --git a/ttt_discover/discovery.py b/ttt_discover/discovery.py
index 6cdb434..6a0794c 100644
--- a/ttt_discover/discovery.py
+++ b/ttt_discover/discovery.py
@@ -62,6 +62,7 @@ class DiscoverConfig:
     codex_max_concurrent_requests: int | None = 4
     codex_initial_program_paths: tuple[str, ...] = ()
     codex_initial_pool_paths: tuple[str, ...] = ()
+    codex_dissimilarity_gate_sampling: bool = False
     codex_autonomous: bool = False
 
 
@@ -145,6 +146,7 @@ async def discover_impl(config: DiscoverConfig):
             max_concurrent_requests=config.codex_max_concurrent_requests,
             initial_program_paths=config.codex_initial_program_paths,
             initial_pool_paths=config.codex_initial_pool_paths,
+            dissimilarity_gate_sampling=config.codex_dissimilarity_gate_sampling,
             autonomous=config.codex_autonomous,
             wandb_project=config.wandb_project,
             wandb_name=config.experiment_name,
diff --git a/ttt_discover/rl/codex_no_finetune.py b/ttt_discover/rl/codex_no_finetune.py
index 18a0e9e..4440f49 100644
--- a/ttt_discover/rl/codex_no_finetune.py
+++ b/ttt_discover/rl/codex_no_finetune.py
@@ -137,6 +137,7 @@ class CodexNoFinetuneConfig:
     initial_program_paths: tuple[str, ...] = ()
     initial_pool_paths: tuple[str, ...] = ()
     topk_children: int = 16
+    dissimilarity_gate_sampling: bool = False
 
     # False is TTT Discover: non-auto, many short samples. True is AutoEvolve:
     # one Codex deep-dive workspace per sample with file edits and shell access.
@@ -820,7 +821,20 @@ async def sample_batch(
     i_batch: int,
 ) -> tuple[list[CandidateResult], dict[str, Any], list[CandidateResult]]:
     metrics: dict[str, Any] = {}
-    parent_states = sampler.sample_states(cfg.groups_per_batch)
+    if cfg.dissimilarity_gate_sampling:
+        sample_dissimilarity_gate = getattr(
+            sampler,
+            "sample_states_dissimilarity_gate",
+            None,
+        )
+        if not callable(sample_dissimilarity_gate):
+            raise ValueError(
+                "dissimilarity_gate_sampling=True requires sampler public API "
+                "sample_states_dissimilarity_gate(num_states)"
+            )
+        parent_states = sample_dissimilarity_gate(cfg.groups_per_batch)
+    else:
+        parent_states = sampler.sample_states(cfg.groups_per_batch)
     semaphore = (
         asyncio.Semaphore(cfg.max_concurrent_requests)
         if cfg.max_concurrent_requests is not None
````
</details>

