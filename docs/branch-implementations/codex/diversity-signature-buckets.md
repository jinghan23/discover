# codex/diversity-signature-buckets

## Summary

为 state 生成结构签名：结合 construction 形态、代码 AST/regex 特征、imports/decorators/calls、env signature 等，再按 signature bucket 平衡。

## Branch State

- Worktree: `/opt/tiger/discover-signature-buckets`
- HEAD: `2234282`
- Base used for comparison: `2234282`
- Commits ahead of base: `0`
- Commits behind base: `0`
- Group: `diversity_extra`
- Implementation location: `uncommitted worktree diff`

## Implemented Behavior

- The summary above is based on reading the changed Python code, added config fields, and sampler/prompt hooks in this branch.
- For sampler branches, the common pattern is to keep baseline PUCT ordering inside a chosen surface/bucket, while changing which surface/bucket gets the next parent pick.
- No untracked `tests/test_*.py` file was detected for this branch.

## Added Markers

### Config fields

- `codex_signature_balanced_sampling`
- `markers`
- `import_families`
- `decorator_families`
- `call_families`
- `signature_balanced_sampling`

### Constants

- `_COMMON_IMPORT_ROOTS`
- `_TL_CALL_FAMILIES`
- `_TORCH_CALL_FAMILIES`

### Classes

- None

### Functions

- `_bucket_size`
- `_safe_len`
- `_sanitize_env_signature`
- `_construction_kind`
- `_construction_len_bucket`
- `_state_code_text`
- `_join_limited`
- `_ast_dotted_name`
- `_import_family`
- `_decorator_family`
- `_call_family`
- `_regex_code_signature`
- `_ast_code_signature`
- `state_structural_signature`
- `_ranked_puct_entries`
- `_set_last_sampled`
- `sample_states_signature_balanced`

## Diff Summary

- Worktree tracked shortstat: `5 files changed, 530 insertions(+), 31 deletions(-)`
- Untracked files: `2`

### Worktree Status

````text
 M ttt_discover/codex_utils/__init__.py
 M ttt_discover/codex_utils/discovery.py
 M ttt_discover/codex_utils/sampler.py
 M ttt_discover/discovery.py
 M ttt_discover/rl/codex_no_finetune.py
?? repro/gpu_mode/
?? repro/run_discovery.py
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

- `repro/gpu_mode/run_0608_signature_buckets.sh (772 bytes)`
- `repro/run_discovery.py (15882 bytes)`

## Untracked File Snapshots

These files are not part of `git diff` yet, but they are part of the worktree implementation state recorded here.

### `repro/gpu_mode/run_0608_signature_buckets.sh`

````bash
#!/usr/bin/env bash
set -euo pipefail

# Signature-bucket balanced parent sampling: non-auto TTT Discover.
python repro/run_discovery.py \
    --task trimul \
    --runner codex_no_finetune \
    --experiment-name trimul_0608_signature_buckets_gpu2 \
    --log-root codex_runs/trimul_exec_workspaces \
    --gpu 2 \
    --cuda-device-order PCI_BUS_ID \
    --torch-cuda-arch-list 8.0 \
    --num-epochs 50 \
    --group-size 2 \
    --groups-per-batch 4 \
    --num-cpus-per-task 1 \
    --eval-timeout 1200 \
    --wandb-project "" \
    --codex-backend cli \
    --codex-model-name gpt-5.5 \
    --codex-cli-command codex \
    --codex-cli-sandbox read-only \
    --codex-cli-timeout 600 \
    --codex-max-concurrent-requests 4 \
    --codex-signature-balanced-sampling
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
        "--codex-signature-balanced-sampling",
        action="store_true",
        help="Opt in to structural-signature-balanced parent sampling.",
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
            "codex_signature_balanced_sampling="
            f"{args.codex_signature_balanced_sampling}"
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
            signature_balanced_sampling=args.codex_signature_balanced_sampling,
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
        codex_autonomous=args.codex_autonomous,
        codex_signature_balanced_sampling=args.codex_signature_balanced_sampling,
        remove_constant_reward_groups=remove_constant_reward_groups,
    )
    discover(config)


if __name__ == "__main__":
    main()
````

## Diff Stat

### Worktree Tracked Diff Stat

````text
 ttt_discover/codex_utils/__init__.py  |   2 +
 ttt_discover/codex_utils/discovery.py |   2 +
 ttt_discover/codex_utils/sampler.py   | 539 ++++++++++++++++++++++++++++++++--
 ttt_discover/discovery.py             |   2 +
 ttt_discover/rl/codex_no_finetune.py  |  16 +-
 5 files changed, 530 insertions(+), 31 deletions(-)
````
## Raw Diff

<details>
<summary>Tracked worktree patch</summary>

````diff
diff --git a/ttt_discover/codex_utils/__init__.py b/ttt_discover/codex_utils/__init__.py
index 1a51326..d633851 100644
--- a/ttt_discover/codex_utils/__init__.py
+++ b/ttt_discover/codex_utils/__init__.py
@@ -11,6 +11,7 @@ from ttt_discover.codex_utils.sampler import (
     StateSampler,
     create_sampler,
     get_or_create_sampler_with_default,
+    state_structural_signature,
 )
 
 __all__ = [
@@ -27,5 +28,6 @@ __all__ = [
     "discover",
     "get_or_create_sampler_with_default",
     "state_from_dict",
+    "state_structural_signature",
     "to_json_serializable",
 ]
diff --git a/ttt_discover/codex_utils/discovery.py b/ttt_discover/codex_utils/discovery.py
index b21a80f..8d594a4 100644
--- a/ttt_discover/codex_utils/discovery.py
+++ b/ttt_discover/codex_utils/discovery.py
@@ -50,6 +50,7 @@ class DiscoverConfig:
     codex_initial_program_paths: tuple[str, ...] = ()
     codex_initial_pool_paths: tuple[str, ...] = ()
     codex_autonomous: bool = False
+    codex_signature_balanced_sampling: bool = False
 
 
 def _run_codex_no_finetune(config: DiscoverConfig) -> None:
@@ -84,6 +85,7 @@ def _run_codex_no_finetune(config: DiscoverConfig) -> None:
         max_concurrent_requests=config.codex_max_concurrent_requests,
         initial_program_paths=config.codex_initial_program_paths,
         initial_pool_paths=config.codex_initial_pool_paths,
+        signature_balanced_sampling=config.codex_signature_balanced_sampling,
         autonomous=config.codex_autonomous,
         wandb_project=config.wandb_project,
         wandb_name=experiment_name,
diff --git a/ttt_discover/codex_utils/sampler.py b/ttt_discover/codex_utils/sampler.py
index 5c5d4d1..6f93632 100644
--- a/ttt_discover/codex_utils/sampler.py
+++ b/ttt_discover/codex_utils/sampler.py
@@ -1,11 +1,13 @@
 """Search-state sampling and seeding helpers for Codex discovery."""
 from __future__ import annotations
 from abc import ABC, abstractmethod
+import ast
 from contextlib import contextmanager
 import json
 import logging
 import os
 from pathlib import Path
+import re
 import threading
 import time
 from typing import Any, Callable
@@ -331,6 +333,315 @@ def seed_initial_program_paths(
     return len(seed_states)
 
 
+_COMMON_IMPORT_ROOTS = {
+    "collections",
+    "cupy",
+    "itertools",
+    "jax",
+    "math",
+    "numba",
+    "numpy",
+    "random",
+    "torch",
+    "triton",
+}
+_TL_CALL_FAMILIES = {
+    "abs",
+    "arange",
+    "atomic_add",
+    "atomic_cas",
+    "atomic_max",
+    "atomic_min",
+    "atomic_xchg",
+    "broadcast_to",
+    "cast",
+    "dot",
+    "exp",
+    "load",
+    "max",
+    "minimum",
+    "maximum",
+    "program_id",
+    "range",
+    "reshape",
+    "store",
+    "sum",
+    "trans",
+    "where",
+    "zeros",
+}
+_TORCH_CALL_FAMILIES = {
+    "arange",
+    "compile",
+    "empty",
+    "empty_like",
+    "matmul",
+    "mm",
+    "randn",
+    "zeros",
+    "zeros_like",
+}
+
+
+def _bucket_size(value: int | float | None) -> str:
+    try:
+        n = int(value if value is not None else 0)
+    except (TypeError, ValueError, OverflowError):
+        n = 0
+    if n <= 0:
+        return "0"
+    if n == 1:
+        return "1"
+    upper = 2
+    while n > upper and upper < 1024:
+        upper *= 2
+    if n > upper:
+        return ">1024"
+    return f"{upper // 2 + 1}-{upper}"
+
+
+def _safe_len(value: Any) -> int:
+    if value is None:
+        return 0
+    try:
+        return len(value)
+    except TypeError:
+        return 1
+
+
+def _sanitize_env_signature(value: str) -> str:
+    text = value.strip().lower()
+    text = re.sub(r"[0-9a-f]{12,}", "hash", text)
+    text = re.sub(r"\d+", "n", text)
+    text = re.sub(r"[^a-z0-9_.:=+-]+", "_", text)
+    text = re.sub(r"_+", "_", text).strip("_")
+    return text[:64] or "empty"
+
+
+def _construction_kind(state: State) -> str:
+    construction = getattr(state, "construction", None)
+    if construction is None:
+        return "none"
+    if isinstance(construction, list):
+        return "list"
+    if isinstance(construction, tuple):
+        return "tuple"
+    if isinstance(construction, dict):
+        return "dict"
+    if isinstance(construction, str):
+        return "str"
+    return type(construction).__name__.lower()
+
+
+def _construction_len_bucket(state: State) -> str:
+    return _bucket_size(_safe_len(getattr(state, "construction", None)))
+
+
+def _state_code_text(state: State) -> str:
+    code = getattr(state, "code", "")
+    if code is None:
+        return ""
+    if not isinstance(code, str):
+        code = str(code)
+    code = code.strip()
+    if not code:
+        return ""
+    matches = re.findall(
+        r"```(?:[a-zA-Z0-9_+.-]+)?\s*(.*?)```",
+        code,
+        flags=re.DOTALL,
+    )
+    if matches:
+        return matches[-1].strip()
+    return code
+
+
+def _join_limited(values: set[str], *, limit: int = 6) -> str:
+    ordered = sorted(value for value in values if value)
+    if not ordered:
+        return "none"
+    limited = ordered[:limit]
+    if len(ordered) > limit:
+        limited.append("more")
+    return "+".join(limited)
+
+
+def _ast_dotted_name(node: ast.AST | None) -> str:
+    if node is None:
+        return ""
+    if isinstance(node, ast.Name):
+        return node.id
+    if isinstance(node, ast.Attribute):
+        parent = _ast_dotted_name(node.value)
+        return f"{parent}.{node.attr}" if parent else node.attr
+    if isinstance(node, ast.Call):
+        return _ast_dotted_name(node.func)
+    if isinstance(node, ast.Subscript):
+        return _ast_dotted_name(node.value)
+    return ""
+
+
+def _import_family(root: str) -> str:
+    root = root.lower()
+    return root if root in _COMMON_IMPORT_ROOTS else "other"
+
+
+def _decorator_family(name: str) -> str:
+    name = name.lower()
+    if name in {"triton.jit", "triton.heuristics"}:
+        return "triton_jit"
+    if name in {"torch.jit.script", "torch.jit.trace", "torch.compile"}:
+        return "torch_jit"
+    if name in {"numba.jit", "numba.njit", "cuda.jit"}:
+        return "numba_jit"
+    if name.endswith(".jit") or name.endswith(".njit"):
+        return "jit"
+    return "decorated"
+
+
+def _call_family(name: str) -> str:
+    name = name.lower()
+    if name.startswith("triton.language."):
+        name = "tl." + name.split(".", 2)[2]
+    if name.startswith("tl."):
+        family = name.split(".", 1)[1].split(".", 1)[0]
+        return f"tl.{family}" if family in _TL_CALL_FAMILIES else "tl.other"
+    if name.startswith("torch."):
+        family = name.split(".", 1)[1].split(".", 1)[0]
+        return f"torch.{family}" if family in _TORCH_CALL_FAMILIES else "torch.other"
+    if name.startswith("triton."):
+        return "triton.other"
+    if name.startswith("cuda.") or name.startswith("cupy."):
+        return "cuda"
+    if name.startswith("np.") or name.startswith("numpy."):
+        return "numpy"
+    return ""
+
+
+def _regex_code_signature(code: str, state: State) -> str:
+    lowered = code.lower()
+    markers: set[str] = set()
+    marker_patterns = {
+        "triton": r"\btriton\b",
+        "tl_dot": r"\btl\.dot\b",
+        "tl_load": r"\btl\.load\b",
+        "tl_store": r"\btl\.store\b",
+        "torch": r"\btorch\b",
+        "cuda": r"\bcuda\b",
+        "numba": r"\bnumba\b|\bnjit\b",
+        "cpp": r"#include|extern\s+\"c\"|__global__",
+    }
+    for marker, pattern in marker_patterns.items():
+        if re.search(pattern, lowered):
+            markers.add(marker)
+    function_count = len(re.findall(r"(?m)^\s*(?:async\s+def|def)\s+", code))
+    for_count = len(re.findall(r"(?m)^\s*for\s+", code))
+    if_count = len(re.findall(r"(?m)^\s*if\s+", code))
+    while_count = len(re.findall(r"(?m)^\s*while\s+", code))
+    return (
+        "code_regex:"
+        f"markers={_join_limited(markers)}:"
+        f"ctrl=f{_bucket_size(for_count)}_i{_bucket_size(if_count)}_w{_bucket_size(while_count)}:"
+        f"fn={_bucket_size(function_count)}:"
+        f"lines={_bucket_size(len(code.splitlines()))}:"
+        f"chars={_bucket_size(len(code))}:"
+        f"construction={_construction_kind(state)}:"
+        f"clen={_construction_len_bucket(state)}"
+    )
+
+
+def _ast_code_signature(code: str, state: State) -> str:
+    tree = ast.parse(code)
+    import_families: set[str] = set()
+    decorator_families: set[str] = set()
+    call_families: set[str] = set()
+    function_count = 0
+    class_count = 0
+    for_count = 0
+    if_count = 0
+    while_count = 0
+    try_count = 0
+    with_count = 0
+
+    for node in ast.walk(tree):
+        if isinstance(node, ast.Import):
+            for alias in node.names:
+                root = (alias.name or "").split(".", 1)[0]
+                if root:
+                    import_families.add(_import_family(root))
+        elif isinstance(node, ast.ImportFrom):
+            root = (node.module or "").split(".", 1)[0]
+            if root:
+                import_families.add(_import_family(root))
+        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
+            function_count += 1
+            for decorator in node.decorator_list:
+                decorator_name = _ast_dotted_name(decorator)
+                if decorator_name:
+                    decorator_families.add(_decorator_family(decorator_name))
+        elif isinstance(node, ast.ClassDef):
+            class_count += 1
+        elif isinstance(node, (ast.For, ast.AsyncFor)):
+            for_count += 1
+        elif isinstance(node, ast.If):
+            if_count += 1
+        elif isinstance(node, ast.While):
+            while_count += 1
+        elif isinstance(node, ast.Try):
+            try_count += 1
+        elif isinstance(node, (ast.With, ast.AsyncWith)):
+            with_count += 1
+        elif isinstance(node, ast.Call):
+            family = _call_family(_ast_dotted_name(node.func))
+            if family:
+                call_families.add(family)
+
+    return (
+        "code_ast:"
+        f"imports={_join_limited(import_families, limit=4)}:"
+        f"decor={_join_limited(decorator_families, limit=4)}:"
+        f"calls={_join_limited(call_families, limit=6)}:"
+        f"ctrl=f{_bucket_size(for_count)}_i{_bucket_size(if_count)}_w{_bucket_size(while_count)}"
+        f"_t{_bucket_size(try_count)}_with{_bucket_size(with_count)}:"
+        f"fn={_bucket_size(function_count)}:"
+        f"class={_bucket_size(class_count)}:"
+        f"lines={_bucket_size(len(code.splitlines()))}:"
+        f"chars={_bucket_size(len(code))}:"
+        f"construction={_construction_kind(state)}:"
+        f"clen={_construction_len_bucket(state)}"
+    )
+
+
+def state_structural_signature(state: State, env_type: type | None = None) -> str:
+    """Return a deterministic, coarse structural signature for parent balancing."""
+    if env_type is not None:
+        hook = getattr(env_type, "state_signature", None)
+        if hook is not None:
+            try:
+                env_signature = hook(state)
+            except Exception as exc:
+                logger.warning("state_signature hook failed; falling back to structural signature: %s", exc)
+            else:
+                if isinstance(env_signature, str) and env_signature.strip():
+                    return f"env:{_sanitize_env_signature(env_signature)}"
+
+    code = _state_code_text(state)
+    if not code:
+        return (
+            "no_code:"
+            f"construction={_construction_kind(state)}:"
+            f"len_bucket={_construction_len_bucket(state)}"
+        )
+
+    try:
+        return _ast_code_signature(code, state)
+    except SyntaxError:
+        return _regex_code_signature(code, state)
+    except Exception as exc:
+        logger.warning("AST structural signature failed; falling back to regex signature: %s", exc)
+        return _regex_code_signature(code, state)
+
+
 class PUCTSampler(StateSampler):
     """
     PUCT-style sampler with state archive.
@@ -366,6 +677,9 @@ class PUCTSampler(StateSampler):
         self._initial_states: list[State] = []
         self._last_sampled_states: list[State] = []
         self._last_sampled_indices: list[int] = []
+        self._last_sampled_signatures: list[str] = []
+        self._last_signature_balanced_active = False
+        self._last_signature_balanced_fallback_count = 0
         self._lock = threading.Lock()
         self._current_step = resume_step if resume_step is not None else 0
         
@@ -460,6 +774,52 @@ class PUCTSampler(StateSampler):
         weights = (N - ranks).astype(np.float64)
         return weights / weights.sum()
 
+    def _ranked_puct_entries(self) -> list[tuple[float, float, State, int, float, float, float]]:
+        initial_ids = {s.id for s in self._initial_states}
+        candidates = list(self._states)
+        if not candidates:
+            return []
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
+        return scores
+
+    def _set_last_sampled(
+        self,
+        picked: list[State],
+        top_scores: list[tuple[float, float, State, int, float, float, float]],
+        *,
+        signature_balanced_active: bool = False,
+        signature_balanced_fallback_count: int = 0,
+        signatures: list[str] | None = None,
+    ) -> None:
+        state_id_to_idx = {s.id: i for i, s in enumerate(self._states)}
+        self._last_sampled_states = picked
+        self._last_sampled_indices = [state_id_to_idx.get(s.id, -1) for s in picked]
+        self._last_puct_stats = [(t[3], t[4], t[5], t[6], t[0]) for t in top_scores]
+        self._last_sampled_signatures = (
+            signatures
+            if signatures is not None
+            else [state_structural_signature(s, self.env_type) for s in picked]
+        )
+        self._last_signature_balanced_active = bool(signature_balanced_active)
+        self._last_signature_balanced_fallback_count = int(signature_balanced_fallback_count)
+
     def _get_lineage(self, state: State) -> set[str]:
         lineage = {state.id}
         for p in (state.parents or []):
@@ -491,36 +851,17 @@ class PUCTSampler(StateSampler):
 
     def sample_states(self, num_states: int) -> list[State]:
         initial_ids = {s.id for s in self._initial_states}
-        candidates = list(self._states)
+        scores = self._ranked_puct_entries()
 
-        if not candidates:
+        if not scores:
             picked = [
                 create_initial_state(self.env_type, self.problem_type)
                 for _ in range(num_states)
             ]
-            self._last_sampled_states = picked
-            self._last_sampled_indices = []
-            self._last_puct_stats = [(0, 0.0, 0.0, 0.0, 0.0) for _ in picked]
+            top_scores = [(0.0, 0.0, s, 0, 0.0, 0.0, 0.0) for s in picked]
+            self._set_last_sampled(picked, top_scores)
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
@@ -537,10 +878,97 @@ class PUCTSampler(StateSampler):
             top_scores = scores[:num_states]
             picked = [t[2] for t in top_scores]
 
-        state_id_to_idx = {s.id: i for i, s in enumerate(self._states)}
-        self._last_sampled_states = picked
-        self._last_sampled_indices = [state_id_to_idx.get(s.id, -1) for s in picked]
-        self._last_puct_stats = [(t[3], t[4], t[5], t[6], t[0]) for t in top_scores]
+        self._set_last_sampled(picked, top_scores)
+
+        for s in picked:
+            if s.id in initial_ids:
+                self._refresh_random_construction(s)
+
+        return picked
+
+    def sample_states_signature_balanced(self, num_states: int) -> list[State]:
+        initial_ids = {s.id for s in self._initial_states}
+        scores = self._ranked_puct_entries()
+
+        if not scores:
+            picked = [
+                create_initial_state(self.env_type, self.problem_type)
+                for _ in range(num_states)
+            ]
+            top_scores = [(0.0, 0.0, s, 0, 0.0, 0.0, 0.0) for s in picked]
+            signatures = [state_structural_signature(s, self.env_type) for s in picked]
+            self._set_last_sampled(
+                picked,
+                top_scores,
+                signature_balanced_active=True,
+                signatures=signatures,
+            )
+            return picked
+
+        if num_states <= 1:
+            top_scores = scores[:num_states]
+            picked = [t[2] for t in top_scores]
+            signatures = [state_structural_signature(s, self.env_type) for s in picked]
+            self._set_last_sampled(
+                picked,
+                top_scores,
+                signature_balanced_active=True,
+                signatures=signatures,
+            )
+            for s in picked:
+                if s.id in initial_ids:
+                    self._refresh_random_construction(s)
+            return picked
+
+        children_map = self._build_children_map()
+        ranked = [
+            (entry, state_structural_signature(entry[2], self.env_type))
+            for entry in scores
+        ]
+        picked: list[State] = []
+        top_scores: list[tuple[float, float, State, int, float, float, float]] = []
+        picked_signatures: list[str] = []
+        seen_signatures: set[str] = set()
+        blocked_ids: set[str] = set()
+        picked_ids: set[str] = set()
+
+        for entry, signature in ranked:
+            state = entry[2]
+            if state.id in blocked_ids or state.id in picked_ids:
+                continue
+            if signature in seen_signatures:
+                continue
+            picked.append(state)
+            top_scores.append(entry)
+            picked_signatures.append(signature)
+            seen_signatures.add(signature)
+            picked_ids.add(state.id)
+            blocked_ids.update(self._get_full_lineage(state, children_map))
+            if len(picked) >= num_states:
+                break
+
+        fallback_count = 0
+        if len(picked) < num_states:
+            for entry, signature in ranked:
+                state = entry[2]
+                if state.id in blocked_ids or state.id in picked_ids:
+                    continue
+                picked.append(state)
+                top_scores.append(entry)
+                picked_signatures.append(signature)
+                picked_ids.add(state.id)
+                blocked_ids.update(self._get_full_lineage(state, children_map))
+                fallback_count += 1
+                if len(picked) >= num_states:
+                    break
+
+        self._set_last_sampled(
+            picked,
+            top_scores,
+            signature_balanced_active=True,
+            signature_balanced_fallback_count=fallback_count,
+            signatures=picked_signatures,
+        )
 
         for s in picked:
             if s.id in initial_ids:
@@ -724,6 +1152,11 @@ class PUCTSampler(StateSampler):
             "puct/sampled_size": len(self._last_sampled_states),
             "puct/T": self._T,
             "puct/scale_last": float(self._last_scale),
+            "puct/sampled_signature_count": len(set(self._last_sampled_signatures)),
+            "puct/signature_balanced_active": bool(self._last_signature_balanced_active),
+            "puct/signature_balanced_fallback_count": int(
+                self._last_signature_balanced_fallback_count
+            ),
         }
         stats.update(_stats(buffer_values, "puct/buffer_value"))
         stats.update(_stats(buffer_timesteps, "puct/buffer_timestep"))
@@ -734,18 +1167,64 @@ class PUCTSampler(StateSampler):
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
+            "state_id",
+            "signature",
+        ]
         rows = []
         if not self._last_sampled_states:
             return columns, rows
         indices = self._last_sampled_indices if len(self._last_sampled_indices) == len(self._last_sampled_states) else [-1] * len(self._last_sampled_states)
         stats = self._last_puct_stats if len(self._last_puct_stats) == len(self._last_sampled_states) else [(0, 0.0, 0.0, 0.0, 0.0)] * len(self._last_sampled_states)
-        for idx, state, (n, Q, P, bonus, score) in zip(indices, self._last_sampled_states, stats):
+        signatures = (
+            self._last_sampled_signatures
+            if len(self._last_sampled_signatures) == len(self._last_sampled_states)
+            else [
+                state_structural_signature(state, self.env_type)
+                for state in self._last_sampled_states
+            ]
+        )
+        for idx, state, (n, Q, P, bonus, score), signature in zip(
+            indices,
+            self._last_sampled_states,
+            stats,
+            signatures,
+            strict=False,
+        ):
             parent_val = state.parent_values[0] if state.parent_values else None
             constr = getattr(state, 'construction', None)
             constr_len = len(constr) if constr is not None else 0
             obs_len = len(state.observation) if state.observation else 0
-            rows.append((idx, state.timestep, state.value, 0, parent_val, constr_len, obs_len, n, Q, P, bonus, score))
+            rows.append(
+                (
+                    idx,
+                    state.timestep,
+                    state.value,
+                    0,
+                    parent_val,
+                    constr_len,
+                    obs_len,
+                    n,
+                    Q,
+                    P,
+                    bonus,
+                    score,
+                    state.id,
+                    signature,
+                )
+            )
         return columns, rows
 
 
diff --git a/ttt_discover/discovery.py b/ttt_discover/discovery.py
index 6cdb434..66ff9eb 100644
--- a/ttt_discover/discovery.py
+++ b/ttt_discover/discovery.py
@@ -63,6 +63,7 @@ class DiscoverConfig:
     codex_initial_program_paths: tuple[str, ...] = ()
     codex_initial_pool_paths: tuple[str, ...] = ()
     codex_autonomous: bool = False
+    codex_signature_balanced_sampling: bool = False
 
 
 def init_ray(num_cpus_per_task: int, env_type: str):
@@ -145,6 +146,7 @@ async def discover_impl(config: DiscoverConfig):
             max_concurrent_requests=config.codex_max_concurrent_requests,
             initial_program_paths=config.codex_initial_program_paths,
             initial_pool_paths=config.codex_initial_pool_paths,
+            signature_balanced_sampling=config.codex_signature_balanced_sampling,
             autonomous=config.codex_autonomous,
             wandb_project=config.wandb_project,
             wandb_name=config.experiment_name,
diff --git a/ttt_discover/rl/codex_no_finetune.py b/ttt_discover/rl/codex_no_finetune.py
index 18a0e9e..620aa26 100644
--- a/ttt_discover/rl/codex_no_finetune.py
+++ b/ttt_discover/rl/codex_no_finetune.py
@@ -137,6 +137,7 @@ class CodexNoFinetuneConfig:
     initial_program_paths: tuple[str, ...] = ()
     initial_pool_paths: tuple[str, ...] = ()
     topk_children: int = 16
+    signature_balanced_sampling: bool = False
 
     # False is TTT Discover: non-auto, many short samples. True is AutoEvolve:
     # one Codex deep-dive workspace per sample with file edits and shell access.
@@ -820,7 +821,20 @@ async def sample_batch(
     i_batch: int,
 ) -> tuple[list[CandidateResult], dict[str, Any], list[CandidateResult]]:
     metrics: dict[str, Any] = {}
-    parent_states = sampler.sample_states(cfg.groups_per_batch)
+    if cfg.signature_balanced_sampling:
+        sample_signature_balanced = getattr(
+            sampler,
+            "sample_states_signature_balanced",
+            None,
+        )
+        if not callable(sample_signature_balanced):
+            raise ValueError(
+                "signature_balanced_sampling requires sampler public API "
+                "sample_states_signature_balanced(num_states)"
+            )
+        parent_states = sample_signature_balanced(cfg.groups_per_batch)
+    else:
+        parent_states = sampler.sample_states(cfg.groups_per_batch)
     semaphore = (
         asyncio.Semaphore(cfg.max_concurrent_requests)
         if cfg.max_concurrent_requests is not None
````
</details>

