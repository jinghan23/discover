# codex/diversity-token-rarity-balance

## Summary

从 construction/code/observation 中抽 token，按 archive 文档频率计算 token rarity，分类 token_common/mixed/rare 后平衡。

## Branch State

- Worktree: `/opt/tiger/discover-token-rarity-balance`
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

- `codex_token_rarity_balanced_sampling`
- `buckets`
- `token_sets`
- `df`
- `scores`
- `token_rarity_balanced_sampling`

### Constants

- `_TOKEN_RARITY_RE`
- `_TOKEN_RARITY_UUID_RE`
- `_TOKEN_RARITY_HEX_RE`
- `_TOKEN_RARITY_BUCKET_COMMON`
- `_TOKEN_RARITY_BUCKET_MIXED`
- `_TOKEN_RARITY_BUCKET_RARE`

### Classes

- None

### Functions

- `_token_rarity_keep_token`
- `_token_rarity_tokens`
- `_state_token_rarity_tokens`
- `_token_rarity_bucket_scores`
- `_token_rarity_scores_and_buckets`
- `_puct_sorted_scores`
- `sample_states_token_rarity_balanced`
- `pick`
- `_sample_parent_states`

## Diff Summary

- Worktree tracked shortstat: `4 files changed, 264 insertions(+), 1 deletion(-)`
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

- `repro/gpu_mode/run_0608_token_rarity_balance.sh (1148 bytes)`
- `repro/run_discovery.py (15410 bytes)`
- `tests/test_token_rarity_balanced_sampling.py (12271 bytes)`

### Detected Test Functions

- `tests/test_token_rarity_balanced_sampling.py::test_enabled_flag_requires_balanced_sampler_api`
- `tests/test_token_rarity_balanced_sampling.py::test_flag_off_matches_baseline_ids_stats_and_table_schema`
- `tests/test_token_rarity_balanced_sampling.py::test_rarity_scores_small_archive`
- `tests/test_token_rarity_balanced_sampling.py::test_bucket_tie_behavior`
- `tests/test_token_rarity_balanced_sampling.py::test_empty_tokens_skip_first_pass_but_can_fill_fallback`
- `tests/test_token_rarity_balanced_sampling.py::test_n_valid_zero_one_two`
- `tests/test_token_rarity_balanced_sampling.py::test_same_bucket_puct_order_and_cross_bucket_first_pass`
- `tests/test_token_rarity_balanced_sampling.py::test_full_lineage_blocking`
- `tests/test_token_rarity_balanced_sampling.py::test_persistence_schema_has_no_token_metadata`
- `tests/test_token_rarity_balanced_sampling.py::test_tokenization_filters_numeric_noise_and_falls_back_to_code`
- `tests/test_token_rarity_balanced_sampling.py::test_num_states_one_uses_baseline_api`

## Untracked File Snapshots

These files are not part of `git diff` yet, but they are part of the worktree implementation state recorded here.

### `repro/gpu_mode/run_0608_token_rarity_balance.sh`

````bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-gpu-mode-trimul-token-rarity-balance-0608}"
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
  --codex-token-rarity-balanced-sampling
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
        "--codex-token-rarity-balanced-sampling",
        action="store_true",
        help="Use archive-wide token-rarity balanced PUCT parent sampling.",
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
            "codex_token_rarity_balanced_sampling="
            f"{args.codex_token_rarity_balanced_sampling}"
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
            token_rarity_balanced_sampling=(
                args.codex_token_rarity_balanced_sampling
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
        codex_token_rarity_balanced_sampling=(
            args.codex_token_rarity_balanced_sampling
        ),
        codex_autonomous=args.codex_autonomous,
        remove_constant_reward_groups=remove_constant_reward_groups,
    )
    discover(config)


if __name__ == "__main__":
    main()
````

### `tests/test_token_rarity_balanced_sampling.py`

````python
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from ttt_discover.codex_utils.runtime import State
from ttt_discover.codex_utils.sampler import (
    PUCTSampler,
    _sampler_file_for_step,
    _state_token_rarity_tokens,
    _token_rarity_bucket_scores,
    _token_rarity_scores_and_buckets,
    _token_rarity_tokens,
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
    construction=None,
    *,
    code: str = "",
    parents: list[dict] | None = None,
) -> State:
    return State(
        timestep=1,
        construction=[] if construction is None else construction,
        code=code,
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


class TokenRarityBalancedSamplingTest(unittest.TestCase):
    def test_enabled_flag_requires_balanced_sampler_api(self):
        cfg = SimpleNamespace(
            groups_per_batch=2,
            token_rarity_balanced_sampling=True,
        )
        sampler = SimpleNamespace(sample_states=lambda num_states: [])

        with self.assertRaisesRegex(
            ValueError,
            "sample_states_token_rarity_balanced",
        ):
            _sample_parent_states(cfg, sampler)

        sampler = SimpleNamespace(
            sample_states=lambda num_states: [],
            sample_states_token_rarity_balanced=None,
        )
        with self.assertRaisesRegex(
            ValueError,
            "sample_states_token_rarity_balanced",
        ):
            _sample_parent_states(cfg, sampler)

    def test_flag_off_matches_baseline_ids_stats_and_table_schema(self):
        states = [
            make_state("s0", 10.0, ["alpha"]),
            make_state("s1", 9.0, ["beta"]),
            make_state("s2", 8.0, ["gamma"]),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            baseline = make_sampler(tmpdir, states)
            expected = baseline.sample_states(3)
            expected_stats = baseline.get_sample_stats()
            expected_columns, _ = baseline.get_sample_table()

            off_sampler = make_sampler(tmpdir, states)
            cfg = SimpleNamespace(
                groups_per_batch=3,
                token_rarity_balanced_sampling=False,
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

    def test_rarity_scores_small_archive(self):
        states = [
            make_state("ab", 1.0, ["a", "b"]),
            make_state("ac", 1.0, ["a", "c"]),
            make_state("d", 1.0, ["d"]),
        ]
        scores, buckets = _token_rarity_scores_and_buckets(states)

        self.assertAlmostEqual(scores["ab"], 0.5)
        self.assertAlmostEqual(scores["ac"], 0.5)
        self.assertAlmostEqual(scores["d"], 2.0 / 3.0)
        self.assertEqual(buckets["ab"], "token_common")
        self.assertEqual(buckets["ac"], "token_common")
        self.assertEqual(buckets["d"], "token_rare")

    def test_bucket_tie_behavior(self):
        one = _token_rarity_bucket_scores([("a", 0.0), ("b", 0.0)])
        self.assertEqual(set(one.values()), {"token_mixed"})

        two = _token_rarity_bucket_scores([("a", 0.1), ("b", 0.2)])
        self.assertEqual(two, {"a": "token_common", "b": "token_rare"})

        three_plus = _token_rarity_bucket_scores(
            [
                ("s0", 0.0),
                ("s1", 0.1),
                ("s2", 0.1),
                ("s3", 0.2),
                ("s4", 0.3),
                ("s5", 0.4),
            ]
        )
        self.assertEqual(three_plus["s0"], "token_common")
        self.assertEqual(three_plus["s1"], "token_common")
        self.assertEqual(three_plus["s2"], "token_common")
        self.assertEqual(three_plus["s3"], "token_mixed")
        self.assertEqual(three_plus["s4"], "token_rare")
        self.assertEqual(three_plus["s5"], "token_rare")

    def test_empty_tokens_skip_first_pass_but_can_fill_fallback(self):
        states = [
            make_state("empty", 100.0, [123, 4.5], code="123 456"),
            make_state("valid", 10.0, ["alpha"]),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            sampler = make_sampler(tmpdir, states)
            picked = sampler.sample_states_token_rarity_balanced(2)

        self.assertEqual([s.id for s in picked], ["valid", "empty"])

    def test_n_valid_zero_one_two(self):
        scores, buckets = _token_rarity_scores_and_buckets(
            [make_state("empty", 1.0, [123], code="456")]
        )
        self.assertEqual(scores, {})
        self.assertEqual(buckets, {})

        scores, buckets = _token_rarity_scores_and_buckets(
            [make_state("one", 1.0, ["alpha"])]
        )
        self.assertEqual(scores["one"], 0.0)
        self.assertEqual(buckets["one"], "token_mixed")

        scores, buckets = _token_rarity_scores_and_buckets(
            [
                make_state("ab", 1.0, ["a", "b"]),
                make_state("a", 1.0, ["a"]),
            ]
        )
        self.assertAlmostEqual(scores["a"], 0.0)
        self.assertAlmostEqual(scores["ab"], 0.25)
        self.assertEqual(buckets["a"], "token_common")
        self.assertEqual(buckets["ab"], "token_rare")

    def test_same_bucket_puct_order_and_cross_bucket_first_pass(self):
        states = [
            make_state("common_top", 100.0, ["a"]),
            make_state("common_next", 99.0, ["b"]),
            make_state("rare", 98.0, ["c"]),
        ]
        buckets = {
            "common_top": "token_common",
            "common_next": "token_common",
            "rare": "token_rare",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            sampler = make_sampler(tmpdir, states)
            with patch(
                "ttt_discover.codex_utils.sampler._token_rarity_scores_and_buckets",
                return_value=({}, buckets),
            ):
                picked = sampler.sample_states_token_rarity_balanced(2)
        self.assertEqual([s.id for s in picked], ["common_top", "rare"])

        states = [
            make_state("mixed", 100.0, ["m"]),
            make_state("common", 99.0, ["c"]),
            make_state("rare", 98.0, ["r"]),
        ]
        buckets = {
            "mixed": "token_mixed",
            "common": "token_common",
            "rare": "token_rare",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            sampler = make_sampler(tmpdir, states)
            with patch(
                "ttt_discover.codex_utils.sampler._token_rarity_scores_and_buckets",
                return_value=({}, buckets),
            ):
                picked = sampler.sample_states_token_rarity_balanced(3)
        self.assertEqual([s.id for s in picked], ["mixed", "common", "rare"])

    def test_full_lineage_blocking(self):
        states = [
            make_state("parent", 100.0, ["p"]),
            make_state("child", 99.0, ["c"], parents=[{"id": "parent"}]),
            make_state("other", 98.0, ["o"]),
        ]
        buckets = {
            "parent": "token_common",
            "child": "token_rare",
            "other": "token_mixed",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            sampler = make_sampler(tmpdir, states)
            with patch(
                "ttt_discover.codex_utils.sampler._token_rarity_scores_and_buckets",
                return_value=({}, buckets),
            ):
                picked = sampler.sample_states_token_rarity_balanced(2)

        self.assertEqual([s.id for s in picked], ["parent", "other"])

    def test_persistence_schema_has_no_token_metadata(self):
        states = [
            make_state("a", 2.0, ["alpha"]),
            make_state("b", 1.0, ["beta"]),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            sampler = make_sampler(tmpdir, states)
            sampler.sample_states_token_rarity_balanced(2)
            sampler.flush(step=3)
            store_path = _sampler_file_for_step(sampler.file_path, 3)
            with open(store_path, "r", encoding="utf-8") as f:
                store = json.load(f)
            columns, _ = sampler.get_sample_table()

        self.assertEqual(
            set(store),
            {"step", "states", "initial_states", "puct_n", "puct_m", "puct_T"},
        )
        self.assertFalse(any("token" in key for key in store))
        self.assertFalse(any("token" in column for column in columns))

    def test_tokenization_filters_numeric_noise_and_falls_back_to_code(self):
        tokens = _token_rarity_tokens(
            "123 3.14 "
            "de305d54-75b4-431b-adb2-eb6b9e546014 "
            "abcdef12 0123456789abcdef random_identifier tile_16 ValidName a"
        )
        self.assertIn("random_identifier", tokens)
        self.assertIn("tile_16", tokens)
        self.assertIn("validname", tokens)
        self.assertIn("a", tokens)
        self.assertNotIn("abcdef12", tokens)
        self.assertNotIn("de305d54", tokens)
        self.assertNotIn("adb2", tokens)

        cyclic: list[object] = []
        cyclic.append(cyclic)
        nested_tokens = _token_rarity_tokens(
            [b"Bytes_Token", np.array(["Array_Token"]), np.float64(1.5), cyclic]
        )
        self.assertIn("bytes_token", nested_tokens)
        self.assertIn("array_token", nested_tokens)

        fallback_state = make_state(
            "fallback",
            1.0,
            [123, np.array([1.0, 2.0])],
            code="def code_alpha(): return value_beta",
        )
        fallback_tokens = _state_token_rarity_tokens(fallback_state)
        self.assertIn("code_alpha", fallback_tokens)
        self.assertIn("value_beta", fallback_tokens)

        construction_state = make_state(
            "construction",
            1.0,
            ["construct_alpha"],
            code="code_beta",
        )
        construction_tokens = _state_token_rarity_tokens(construction_state)
        self.assertIn("construct_alpha", construction_tokens)
        self.assertNotIn("code_beta", construction_tokens)

    def test_num_states_one_uses_baseline_api(self):
        states = [
            make_state("empty_high", 100.0, [123], code="456"),
            make_state("valid_low", 1.0, ["alpha"]),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            sampler = make_sampler(tmpdir, states)
            picked = sampler.sample_states_token_rarity_balanced(1)

        self.assertEqual([s.id for s in picked], ["empty_high"])


if __name__ == "__main__":
    unittest.main()
````

## Diff Stat

### Worktree Tracked Diff Stat

````text
 ttt_discover/codex_utils/discovery.py |   4 +
 ttt_discover/codex_utils/sampler.py   | 235 ++++++++++++++++++++++++++++++++++
 ttt_discover/discovery.py             |   4 +
 ttt_discover/rl/codex_no_finetune.py  |  22 +++-
 4 files changed, 264 insertions(+), 1 deletion(-)
````
## Raw Diff

<details>
<summary>Tracked worktree patch</summary>

````diff
diff --git a/ttt_discover/codex_utils/discovery.py b/ttt_discover/codex_utils/discovery.py
index b21a80f..74d69c2 100644
--- a/ttt_discover/codex_utils/discovery.py
+++ b/ttt_discover/codex_utils/discovery.py
@@ -50,6 +50,7 @@ class DiscoverConfig:
     codex_initial_program_paths: tuple[str, ...] = ()
     codex_initial_pool_paths: tuple[str, ...] = ()
     codex_autonomous: bool = False
+    codex_token_rarity_balanced_sampling: bool = False
 
 
 def _run_codex_no_finetune(config: DiscoverConfig) -> None:
@@ -85,6 +86,9 @@ def _run_codex_no_finetune(config: DiscoverConfig) -> None:
         initial_program_paths=config.codex_initial_program_paths,
         initial_pool_paths=config.codex_initial_pool_paths,
         autonomous=config.codex_autonomous,
+        token_rarity_balanced_sampling=(
+            config.codex_token_rarity_balanced_sampling
+        ),
         wandb_project=config.wandb_project,
         wandb_name=experiment_name,
         log_path=log_path,
diff --git a/ttt_discover/codex_utils/sampler.py b/ttt_discover/codex_utils/sampler.py
index 5c5d4d1..137b285 100644
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
@@ -66,6 +67,159 @@ def _read_json_or_default(path: str, default: Any) -> Any:
         return default
 
 
+_TOKEN_RARITY_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,31}")
+_TOKEN_RARITY_UUID_RE = re.compile(
+    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
+    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
+)
+_TOKEN_RARITY_HEX_RE = re.compile(r"[0-9a-fA-F]+")
+_TOKEN_RARITY_BUCKET_COMMON = "token_common"
+_TOKEN_RARITY_BUCKET_MIXED = "token_mixed"
+_TOKEN_RARITY_BUCKET_RARE = "token_rare"
+
+
+def _token_rarity_keep_token(token: str) -> bool:
+    if not token or len(token) > 32:
+        return False
+    if not any(ch.isalpha() for ch in token):
+        return False
+    if token.isdigit():
+        return False
+    if len(token) >= 8 and _TOKEN_RARITY_HEX_RE.fullmatch(token):
+        return False
+    digit_count = sum(ch.isdigit() for ch in token)
+    if len(token) >= 16 and digit_count >= 4 and "_" not in token:
+        return False
+    return True
+
+
+def _token_rarity_tokens(value: Any, seen: set[int] | None = None) -> set[str]:
+    if seen is None:
+        seen = set()
+
+    if isinstance(value, bytes):
+        try:
+            value = value.decode("utf-8", errors="ignore")
+        except Exception:
+            return set()
+
+    if isinstance(value, str):
+        text = _TOKEN_RARITY_UUID_RE.sub(" ", value)
+        return {
+            token.lower()
+            for token in _TOKEN_RARITY_RE.findall(text)
+            if _token_rarity_keep_token(token)
+        }
+
+    if isinstance(value, np.generic):
+        return _token_rarity_tokens(value.item(), seen)
+
+    if isinstance(value, np.ndarray):
+        obj_id = id(value)
+        if obj_id in seen:
+            return set()
+        seen.add(obj_id)
+        return _token_rarity_tokens(value.tolist(), seen)
+
+    if isinstance(value, dict):
+        obj_id = id(value)
+        if obj_id in seen:
+            return set()
+        seen.add(obj_id)
+        tokens: set[str] = set()
+        for key, item in value.items():
+            tokens.update(_token_rarity_tokens(key, seen))
+            tokens.update(_token_rarity_tokens(item, seen))
+        return tokens
+
+    if isinstance(value, (list, tuple, set, frozenset)):
+        obj_id = id(value)
+        if obj_id in seen:
+            return set()
+        seen.add(obj_id)
+        tokens: set[str] = set()
+        for item in value:
+            tokens.update(_token_rarity_tokens(item, seen))
+        return tokens
+
+    return set()
+
+
+def _state_token_rarity_tokens(state: State) -> set[str]:
+    construction_tokens = _token_rarity_tokens(getattr(state, "construction", None))
+    if construction_tokens:
+        return construction_tokens
+    return _token_rarity_tokens(getattr(state, "code", None))
+
+
+def _token_rarity_bucket_scores(score_items: list[tuple[str, float]]) -> dict[str, str]:
+    if not score_items:
+        return {}
+
+    distinct_scores = sorted({score for _, score in score_items})
+    if len(distinct_scores) == 1:
+        return {state_id: _TOKEN_RARITY_BUCKET_MIXED for state_id, _ in score_items}
+
+    if len(distinct_scores) == 2:
+        low_score = distinct_scores[0]
+        return {
+            state_id: (
+                _TOKEN_RARITY_BUCKET_COMMON
+                if score == low_score
+                else _TOKEN_RARITY_BUCKET_RARE
+            )
+            for state_id, score in score_items
+        }
+
+    sorted_items = sorted(score_items, key=lambda item: (item[1], item[0]))
+    n = len(sorted_items)
+    buckets: dict[str, str] = {}
+    i = 0
+    while i < n:
+        score = sorted_items[i][1]
+        start = i
+        while i < n and sorted_items[i][1] == score:
+            i += 1
+        end = i - 1
+        midpoint = (start + end) / 2.0
+        if midpoint < n / 3.0:
+            bucket = _TOKEN_RARITY_BUCKET_COMMON
+        elif midpoint < 2.0 * n / 3.0:
+            bucket = _TOKEN_RARITY_BUCKET_MIXED
+        else:
+            bucket = _TOKEN_RARITY_BUCKET_RARE
+        for state_id, _ in sorted_items[start:i]:
+            buckets[state_id] = bucket
+    return buckets
+
+
+def _token_rarity_scores_and_buckets(
+    states: list[State],
+) -> tuple[dict[str, float], dict[str, str]]:
+    token_sets: list[tuple[str, set[str]]] = []
+    for state in states:
+        tokens = _state_token_rarity_tokens(state)
+        if tokens:
+            token_sets.append((state.id, tokens))
+
+    n_valid = len(token_sets)
+    if n_valid == 0:
+        return {}, {}
+
+    df: dict[str, int] = {}
+    for _, tokens in token_sets:
+        for token in tokens:
+            df[token] = df.get(token, 0) + 1
+
+    scores: dict[str, float] = {}
+    for state_id, tokens in token_sets:
+        rarity_sum = sum(1.0 - (df[token] / n_valid) for token in tokens)
+        scores[state_id] = rarity_sum / len(tokens)
+
+    buckets = _token_rarity_bucket_scores(list(scores.items()))
+    return scores, buckets
+
+
 class StateSampler(ABC):
     """Abstract base class for sampling states."""
 
@@ -548,6 +702,87 @@ class PUCTSampler(StateSampler):
 
         return picked
 
+    def _puct_sorted_scores(
+        self,
+        candidates: list[State],
+    ) -> list[tuple[float, float, State, int, float, float, float]]:
+        initial_ids = {s.id for s in self._initial_states}
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
+    def sample_states_token_rarity_balanced(self, num_states: int) -> list[State]:
+        if num_states <= 1:
+            return self.sample_states(num_states)
+
+        initial_ids = {s.id for s in self._initial_states}
+        candidates = list(self._states)
+        if not candidates:
+            return self.sample_states(num_states)
+
+        scores = self._puct_sorted_scores(candidates)
+        _, bucket_by_id = _token_rarity_scores_and_buckets(candidates)
+        children_map = self._build_children_map()
+        picked: list[State] = []
+        picked_entries: list[tuple[float, float, State, int, float, float, float]] = []
+        picked_ids: set[str] = set()
+        picked_buckets: set[str] = set()
+        blocked_ids: set[str] = set()
+
+        def pick(entry: tuple[float, float, State, int, float, float, float]) -> None:
+            state = entry[2]
+            picked.append(state)
+            picked_entries.append(entry)
+            picked_ids.add(state.id)
+            blocked_ids.update(self._get_full_lineage(state, children_map))
+
+        for entry in scores:
+            state = entry[2]
+            if state.id in blocked_ids:
+                continue
+            bucket = bucket_by_id.get(state.id)
+            if bucket is None or bucket in picked_buckets:
+                continue
+            pick(entry)
+            picked_buckets.add(bucket)
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
+        self._last_puct_stats = [(t[3], t[4], t[5], t[6], t[0]) for t in picked_entries]
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
index 6cdb434..40cc995 100644
--- a/ttt_discover/discovery.py
+++ b/ttt_discover/discovery.py
@@ -63,6 +63,7 @@ class DiscoverConfig:
     codex_initial_program_paths: tuple[str, ...] = ()
     codex_initial_pool_paths: tuple[str, ...] = ()
     codex_autonomous: bool = False
+    codex_token_rarity_balanced_sampling: bool = False
 
 
 def init_ray(num_cpus_per_task: int, env_type: str):
@@ -146,6 +147,9 @@ async def discover_impl(config: DiscoverConfig):
             initial_program_paths=config.codex_initial_program_paths,
             initial_pool_paths=config.codex_initial_pool_paths,
             autonomous=config.codex_autonomous,
+            token_rarity_balanced_sampling=(
+                config.codex_token_rarity_balanced_sampling
+            ),
             wandb_project=config.wandb_project,
             wandb_name=config.experiment_name,
             log_path=log_path,
diff --git a/ttt_discover/rl/codex_no_finetune.py b/ttt_discover/rl/codex_no_finetune.py
index 18a0e9e..af9ff5f 100644
--- a/ttt_discover/rl/codex_no_finetune.py
+++ b/ttt_discover/rl/codex_no_finetune.py
@@ -137,6 +137,7 @@ class CodexNoFinetuneConfig:
     initial_program_paths: tuple[str, ...] = ()
     initial_pool_paths: tuple[str, ...] = ()
     topk_children: int = 16
+    token_rarity_balanced_sampling: bool = False
 
     # False is TTT Discover: non-auto, many short samples. True is AutoEvolve:
     # one Codex deep-dive workspace per sample with file edits and shell access.
@@ -820,7 +821,7 @@ async def sample_batch(
     i_batch: int,
 ) -> tuple[list[CandidateResult], dict[str, Any], list[CandidateResult]]:
     metrics: dict[str, Any] = {}
-    parent_states = sampler.sample_states(cfg.groups_per_batch)
+    parent_states = _sample_parent_states(cfg, sampler)
     semaphore = (
         asyncio.Semaphore(cfg.max_concurrent_requests)
         if cfg.max_concurrent_requests is not None
@@ -874,6 +875,25 @@ async def sample_batch(
     return kept_results, metrics, results
 
 
+def _sample_parent_states(
+    cfg: CodexNoFinetuneConfig,
+    sampler: StateSampler,
+) -> list[Any]:
+    if cfg.token_rarity_balanced_sampling:
+        sample_balanced = getattr(
+            sampler,
+            "sample_states_token_rarity_balanced",
+            None,
+        )
+        if not callable(sample_balanced):
+            raise ValueError(
+                "token_rarity_balanced_sampling=True requires sampler "
+                "sample_states_token_rarity_balanced(num_states)"
+            )
+        return sample_balanced(cfg.groups_per_batch)
+    return sampler.sample_states(cfg.groups_per_batch)
+
+
 def _log_agent_tables(
     cfg: CodexNoFinetuneConfig,
     ml_logger: MetricsLogger,
````
</details>

