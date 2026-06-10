# codex/diversity-island-pools

## Summary

引入多 island sampler 池，把状态、保存文件和父选择路由隔离到不同 island，支持按 island 采样、更新、恢复和统计。

Note: 这个分支改变 sampler 持久化和路由模型，影响范围大于单个 PUCT 后处理函数。

## Branch State

- Worktree: `/opt/tiger/discover-island-pools`
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

- `log_root`
- `codex_sampler_islands`
- `validated`
- `seen`
- `islands`
- `island`
- `sampler_islands`
- `seen_islands`
- `steps_by_island`

### Constants

- `_SAMPLER_ISLAND_RE`

### Classes

- `IslandSampler`

### Functions

- `validate_sampler_islands`
- `sampler_island_file_base`
- `__init__`
- `_clone_state`
- `_remember_route`
- `get_island_for_parent`
- `_require_island_for_parent`
- `sample_states`
- `has_state`
- `has_state_for_parent`
- `update_states`
- `flush`
- `record_failed_rollout`
- `reload_from_step`
- `get_initial_states`
- `add_initial_states`
- `get_sample_stats`
- `get_sample_table`
- `_latest_island_sampler_step`

## Diff Summary

- Worktree tracked shortstat: `5 files changed, 422 insertions(+), 4 deletions(-)`
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

- `repro/gpu_mode/run_0608_island_pools.sh (1023 bytes)`
- `repro/run_discovery.py (16428 bytes)`

## Untracked File Snapshots

These files are not part of `git diff` yet, but they are part of the worktree implementation state recorded here.

### `repro/gpu_mode/run_0608_island_pools.sh`

````bash
#!/usr/bin/env bash
set -euo pipefail

# Island/Pools: four independent PUCT sampler pools, one parent per island per batch.
# groups-per-batch must equal the number of --codex-sampler-island entries.
# group-size = Codex samples per island parent.
python repro/run_discovery.py \
    --task trimul \
    --runner codex_no_finetune \
    --experiment-name trimul_0608_island_pools_gpu2 \
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
    --codex-sampler-island explore_a \
    --codex-sampler-island explore_b \
    --codex-sampler-island exploit_a \
    --codex-sampler-island exploit_b
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
        "--codex-sampler-island",
        action="append",
        default=None,
        help=(
            "Independent Codex sampler island name. May be repeated; omit to "
            "use the ordinary single sampler."
        ),
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
    codex_sampler_islands = tuple(args.codex_sampler_island or ())
    if codex_sampler_islands:
        from ttt_discover.codex_utils.sampler import validate_sampler_islands

        validate_sampler_islands(codex_sampler_islands)
        if groups_per_batch != len(codex_sampler_islands):
            raise ValueError(
                "groups_per_batch must equal the number of sampler islands when "
                f"islands are enabled: groups_per_batch={groups_per_batch}, "
                f"islands={len(codex_sampler_islands)}"
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
        print(f"codex_sampler_islands={codex_sampler_islands!r}")
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
            sampler_islands=codex_sampler_islands,
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
        log_root=args.log_root,
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
        codex_sampler_islands=codex_sampler_islands,
        codex_autonomous=args.codex_autonomous,
        remove_constant_reward_groups=remove_constant_reward_groups,
    )
    discover(config)


if __name__ == "__main__":
    main()
````

## Diff Stat

### Worktree Tracked Diff Stat

````text
 ttt_discover/codex_utils/__init__.py  |   6 +
 ttt_discover/codex_utils/discovery.py |   5 +-
 ttt_discover/codex_utils/sampler.py   | 276 ++++++++++++++++++++++++++++++++++
 ttt_discover/discovery.py             |   5 +-
 ttt_discover/rl/codex_no_finetune.py  | 134 ++++++++++++++++-
 5 files changed, 422 insertions(+), 4 deletions(-)
````
## Raw Diff

<details>
<summary>Tracked worktree patch</summary>

````diff
diff --git a/ttt_discover/codex_utils/__init__.py b/ttt_discover/codex_utils/__init__.py
index 1a51326..1ed7cdb 100644
--- a/ttt_discover/codex_utils/__init__.py
+++ b/ttt_discover/codex_utils/__init__.py
@@ -7,10 +7,13 @@ from ttt_discover.codex_utils.discovery import DiscoverConfig, discover
 from ttt_discover.codex_utils.environment import Environment, VerifyResult
 from ttt_discover.codex_utils.runtime import State, state_from_dict, to_json_serializable
 from ttt_discover.codex_utils.sampler import (
+    IslandSampler,
     PUCTSampler,
     StateSampler,
     create_sampler,
     get_or_create_sampler_with_default,
+    sampler_island_file_base,
+    validate_sampler_islands,
 )
 
 __all__ = [
@@ -18,6 +21,7 @@ __all__ = [
     "CodexResponseCompleter",
     "DiscoverConfig",
     "Environment",
+    "IslandSampler",
     "PUCTSampler",
     "State",
     "StateSampler",
@@ -26,6 +30,8 @@ __all__ = [
     "create_sampler",
     "discover",
     "get_or_create_sampler_with_default",
+    "sampler_island_file_base",
     "state_from_dict",
     "to_json_serializable",
+    "validate_sampler_islands",
 ]
diff --git a/ttt_discover/codex_utils/discovery.py b/ttt_discover/codex_utils/discovery.py
index b21a80f..025f58d 100644
--- a/ttt_discover/codex_utils/discovery.py
+++ b/ttt_discover/codex_utils/discovery.py
@@ -26,6 +26,7 @@ class DiscoverConfig:
     remove_constant_reward_groups: bool = True
 
     experiment_name: str | None = None
+    log_root: str = "./tinker_log"
     wandb_project: str | None = "tinker-cookbook"
 
     env_type: type | None = None
@@ -49,6 +50,7 @@ class DiscoverConfig:
     codex_max_concurrent_requests: int | None = 4
     codex_initial_program_paths: tuple[str, ...] = ()
     codex_initial_pool_paths: tuple[str, ...] = ()
+    codex_sampler_islands: tuple[str, ...] = ()
     codex_autonomous: bool = False
 
 
@@ -61,7 +63,7 @@ def _run_codex_no_finetune(config: DiscoverConfig) -> None:
     if config.env_type is None:
         raise ValueError("env_type is required")
     experiment_name = config.experiment_name or "codex-no-finetune"
-    log_path = f"./tinker_log/{experiment_name}"
+    log_path = os.path.join(config.log_root, experiment_name)
     os.makedirs(log_path, exist_ok=True)
 
     codex_config = CodexNoFinetuneConfig(
@@ -84,6 +86,7 @@ def _run_codex_no_finetune(config: DiscoverConfig) -> None:
         max_concurrent_requests=config.codex_max_concurrent_requests,
         initial_program_paths=config.codex_initial_program_paths,
         initial_pool_paths=config.codex_initial_pool_paths,
+        sampler_islands=config.codex_sampler_islands,
         autonomous=config.codex_autonomous,
         wandb_project=config.wandb_project,
         wandb_name=experiment_name,
diff --git a/ttt_discover/codex_utils/sampler.py b/ttt_discover/codex_utils/sampler.py
index 5c5d4d1..181a57e 100644
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
 
+_SAMPLER_ISLAND_RE = re.compile(r"^[A-Za-z0-9_-]+$")
+
 
 @contextmanager
 def _file_lock(lock_path: str, *, poll_s: float = 0.05, stale_s: float = 600.0):
@@ -120,6 +123,35 @@ def _sampler_file_for_step(base_path: str, step: int) -> str:
     return f"{base_name}_step_{step:06d}.json"
 
 
+def validate_sampler_islands(islands: tuple[str, ...] | list[str]) -> tuple[str, ...]:
+    """Validate island names used for independent sampler pools."""
+    if not islands:
+        return ()
+
+    validated: list[str] = []
+    seen: set[str] = set()
+    for island in islands:
+        if not isinstance(island, str):
+            raise ValueError("Sampler island names must be strings")
+        if not island:
+            raise ValueError("Sampler island names must be non-empty")
+        if _SAMPLER_ISLAND_RE.fullmatch(island) is None:
+            raise ValueError(
+                f"Invalid sampler island name {island!r}; use only [A-Za-z0-9_-]+"
+            )
+        if island in seen:
+            raise ValueError(f"Duplicate sampler island name: {island!r}")
+        seen.add(island)
+        validated.append(island)
+    return tuple(validated)
+
+
+def sampler_island_file_base(log_path: str, island: str) -> str:
+    """Base JSON path for one island before the step suffix is added."""
+    (safe_name,) = validate_sampler_islands((island,))
+    return os.path.join(log_path, f"puct_sampler_island_{safe_name}.json")
+
+
 def create_initial_state(env_type: type, problem_type: str) -> State:
     """Create initial state by delegating to the env type. Custom envs implement create_initial_state on their class."""
     name = getattr(env_type, "env_name", env_type.__name__)
@@ -749,6 +781,232 @@ class PUCTSampler(StateSampler):
         return columns, rows
 
 
+class IslandSampler(StateSampler):
+    """Route sampling operations across independent PUCT sampler islands."""
+
+    def __init__(
+        self,
+        *,
+        log_path: str,
+        env_type: type,
+        problem_type: str = "",
+        islands: tuple[str, ...] | list[str],
+        resume_step: int | None = None,
+        topk_children: int = 2,
+    ):
+        self.log_path = log_path
+        self.env_type = env_type
+        self.problem_type = problem_type
+        self.islands = validate_sampler_islands(islands)
+        if not self.islands:
+            raise ValueError("IslandSampler requires at least one sampler island")
+        self.samplers: dict[str, PUCTSampler] = {
+            island: PUCTSampler(
+                file_path=sampler_island_file_base(log_path, island),
+                env_type=env_type,
+                problem_type=problem_type,
+                batch_size=1,
+                resume_step=resume_step,
+                topk_children=topk_children,
+            )
+            for island in self.islands
+        }
+        self._parent_routes_by_object: dict[int, str] = {}
+        self._parent_routes_by_state_id: dict[str, set[str]] = {}
+
+    def _clone_state(self, state: State) -> State:
+        state_type = getattr(self.env_type, "state_type", State)
+        return state_from_dict(state.to_dict(), state_type=state_type)
+
+    def _remember_route(self, state: State | None, island: str) -> None:
+        if state is None:
+            return
+        self._parent_routes_by_object[id(state)] = island
+        state_id = getattr(state, "id", None)
+        if state_id is not None:
+            self._parent_routes_by_state_id.setdefault(str(state_id), set()).add(island)
+
+    def get_island_for_parent(self, parent: State) -> str | None:
+        island = self._parent_routes_by_object.get(id(parent))
+        if island is not None:
+            return island
+
+        state_id = getattr(parent, "id", None)
+        if state_id is not None:
+            routed = self._parent_routes_by_state_id.get(str(state_id), set())
+            if len(routed) == 1:
+                return next(iter(routed))
+
+        identity_matches = [
+            island
+            for island, sampler in self.samplers.items()
+            if any(state is parent for state in sampler._states)
+        ]
+        if len(identity_matches) == 1:
+            island = identity_matches[0]
+            self._remember_route(parent, island)
+            return island
+
+        if state_id is None:
+            return None
+        id_matches = [
+            island
+            for island, sampler in self.samplers.items()
+            if any(getattr(state, "id", None) == state_id for state in sampler._states)
+        ]
+        if len(set(id_matches)) == 1:
+            island = id_matches[0]
+            self._remember_route(parent, island)
+            return island
+        return None
+
+    def _require_island_for_parent(self, parent: State) -> str:
+        island = self.get_island_for_parent(parent)
+        if island is None:
+            parent_id = getattr(parent, "id", None)
+            raise ValueError(f"Could not route parent state to a sampler island: {parent_id}")
+        return island
+
+    def sample_states(self, num_states: int) -> list[State]:
+        if num_states != len(self.islands):
+            raise ValueError(
+                "Island sampler requires groups_per_batch to equal the number "
+                f"of islands ({len(self.islands)}), got {num_states}"
+            )
+
+        sampled: list[State] = []
+        for island in self.islands:
+            parent = self.samplers[island].sample_states(1)[0]
+            self._remember_route(parent, island)
+            sampled.append(parent)
+        return sampled
+
+    def has_state(self, state: State) -> bool:
+        # No route is available from a child state alone. Returning False avoids
+        # accidental cross-island duplicate checks.
+        return False
+
+    def has_state_for_parent(self, state: State, parent: State) -> bool:
+        island = self.get_island_for_parent(parent)
+        if island is None:
+            return False
+        return self.samplers[island].has_state(state)
+
+    def update_states(
+        self,
+        states: list[State],
+        parent_states: list[State],
+        save: bool = True,
+        step: int | None = None,
+    ):
+        if not states:
+            return
+        assert len(states) == len(parent_states)
+
+        routed: list[tuple[str, State, State]] = []
+        state_object_islands: dict[int, set[str]] = {}
+        for child, parent in zip(states, parent_states):
+            island = self._require_island_for_parent(parent)
+            routed.append((island, child, parent))
+            state_object_islands.setdefault(id(child), set()).add(island)
+
+        by_island: dict[str, tuple[list[State], list[State]]] = {}
+        for island, child, parent in routed:
+            child_for_update = (
+                self._clone_state(child)
+                if len(state_object_islands.get(id(child), set())) > 1
+                else child
+            )
+            child_states, child_parents = by_island.setdefault(island, ([], []))
+            child_states.append(child_for_update)
+            child_parents.append(parent)
+            self._remember_route(child_for_update, island)
+
+        for island, (child_states, child_parents) in by_island.items():
+            self.samplers[island].update_states(
+                child_states,
+                child_parents,
+                save=save,
+                step=step,
+            )
+
+    def flush(self, step: int | None = None):
+        for sampler in self.samplers.values():
+            sampler.flush(step=step)
+
+    def record_failed_rollout(self, parent: State):
+        island = self._require_island_for_parent(parent)
+        self.samplers[island].record_failed_rollout(parent)
+
+    def reload_from_step(self, step: int):
+        self._parent_routes_by_object.clear()
+        self._parent_routes_by_state_id.clear()
+        for sampler in self.samplers.values():
+            sampler.reload_from_step(step)
+
+    def get_initial_states(self) -> list[State]:
+        states: list[State] = []
+        for island in self.islands:
+            for state in self.samplers[island].get_initial_states():
+                self._remember_route(state, island)
+                states.append(state)
+        return states
+
+    def add_initial_states(
+        self,
+        states: list[State],
+        *,
+        save: bool = True,
+        step: int | None = None,
+    ) -> int:
+        if not states:
+            return 0
+
+        total_added = 0
+        for island in self.islands:
+            cloned_states = [self._clone_state(state) for state in states]
+            for state in cloned_states:
+                self._remember_route(state, island)
+            total_added += self.samplers[island].add_initial_states(
+                cloned_states,
+                save=save,
+                step=step,
+            )
+        return total_added
+
+    def get_sample_stats(self) -> dict:
+        stats: dict[str, Any] = {
+            "puct/islands": len(self.islands),
+        }
+        aggregate_values: dict[str, list[float]] = {}
+        for island in self.islands:
+            child_stats = self.samplers[island].get_sample_stats()
+            for key, value in child_stats.items():
+                suffix = key.removeprefix("puct/")
+                stats[f"puct/island/{island}/{suffix}"] = value
+                if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
+                    aggregate_values.setdefault(suffix, []).append(float(value))
+
+        for suffix, values in aggregate_values.items():
+            if not values:
+                continue
+            stats[f"puct/aggregate/{suffix}/mean"] = float(np.mean(values))
+            stats[f"puct/aggregate/{suffix}/max"] = float(np.max(values))
+            if suffix in {"buffer_size", "sampled_size", "T"}:
+                stats[f"puct/{suffix}/total"] = float(np.sum(values))
+        return stats
+
+    def get_sample_table(self) -> tuple[list[str], list[tuple]]:
+        columns: list[str] | None = None
+        rows: list[tuple] = []
+        for island in self.islands:
+            child_columns, child_rows = self.samplers[island].get_sample_table()
+            if columns is None:
+                columns = ["island", *child_columns]
+            rows.extend((island, *row) for row in child_rows)
+        return columns or ["island"], rows
+
+
 def create_sampler(
     log_path: str,
     env_type: type,
@@ -756,10 +1014,26 @@ def create_sampler(
     batch_size: int = 1,
     resume_step: int | None = None,
     topk_children: int = 2,
+    islands: tuple[str, ...] | list[str] = (),
 ) -> StateSampler:
     """Factory function to create samplers. Pass the env type (from config.env_type)."""
     if not log_path:
         raise ValueError("log_path is required when using PUCT sampler")
+    islands = validate_sampler_islands(islands)
+    if islands:
+        if batch_size != len(islands):
+            raise ValueError(
+                "Island sampler requires groups_per_batch to equal the number "
+                f"of islands ({len(islands)}), got {batch_size}"
+            )
+        return IslandSampler(
+            log_path=log_path,
+            env_type=env_type,
+            problem_type=problem_type,
+            islands=islands,
+            resume_step=resume_step,
+            topk_children=topk_children,
+        )
     sampler_path = os.path.join(log_path, "puct_sampler.json")
     return PUCTSampler(
         file_path=sampler_path,
@@ -778,6 +1052,7 @@ def get_or_create_sampler_with_default(
     batch_size: int = 1,
     resume_step: int | None = None,
     topk_children: int = 2,
+    islands: tuple[str, ...] | list[str] = (),
 ) -> StateSampler:
     """Get sampler. Initial experience is created via env_type.create_initial_state."""
     return create_sampler(
@@ -787,4 +1062,5 @@ def get_or_create_sampler_with_default(
         batch_size=batch_size,
         resume_step=resume_step,
         topk_children=topk_children,
+        islands=islands,
     )
diff --git a/ttt_discover/discovery.py b/ttt_discover/discovery.py
index 6cdb434..d35cdb2 100644
--- a/ttt_discover/discovery.py
+++ b/ttt_discover/discovery.py
@@ -37,6 +37,7 @@ class DiscoverConfig:
 
     # Misc config
     experiment_name: str | None = None
+    log_root: str = "./tinker_log"
     wandb_project: str | None = "tinker-cookbook"
 
     # Environment-specific
@@ -62,6 +63,7 @@ class DiscoverConfig:
     codex_max_concurrent_requests: int | None = 4
     codex_initial_program_paths: tuple[str, ...] = ()
     codex_initial_pool_paths: tuple[str, ...] = ()
+    codex_sampler_islands: tuple[str, ...] = ()
     codex_autonomous: bool = False
 
 
@@ -116,7 +118,7 @@ async def discover_impl(config: DiscoverConfig):
     model_name_for_tokenizer = config.model_name
 
     # create log path if it doesn't exist
-    log_path = f"./tinker_log/{config.experiment_name}"
+    log_path = os.path.join(config.log_root, str(config.experiment_name))
     log_file = os.path.join(log_path, "train.log")
 
     misc_utils.check_log_dir(log_path, behavior_if_exists="resume")
@@ -145,6 +147,7 @@ async def discover_impl(config: DiscoverConfig):
             max_concurrent_requests=config.codex_max_concurrent_requests,
             initial_program_paths=config.codex_initial_program_paths,
             initial_pool_paths=config.codex_initial_pool_paths,
+            sampler_islands=config.codex_sampler_islands,
             autonomous=config.codex_autonomous,
             wandb_project=config.wandb_project,
             wandb_name=config.experiment_name,
diff --git a/ttt_discover/rl/codex_no_finetune.py b/ttt_discover/rl/codex_no_finetune.py
index 18a0e9e..0c437d0 100644
--- a/ttt_discover/rl/codex_no_finetune.py
+++ b/ttt_discover/rl/codex_no_finetune.py
@@ -73,8 +73,10 @@ from ttt_discover.codex_utils.runtime import to_json_serializable
 from ttt_discover.codex_utils.sampler import (
     StateSampler,
     create_sampler,
+    sampler_island_file_base,
     seed_initial_pool_paths,
     seed_initial_program_paths,
+    validate_sampler_islands,
 )
 
 logger = logging.getLogger(__name__)
@@ -112,6 +114,7 @@ class CandidateResult:
     drop_reason: str | None = None
     pool_status: str | None = None
     sampler_error: str | None = None
+    island: str | None = None
 
 
 @chz.chz
@@ -136,6 +139,7 @@ class CodexNoFinetuneConfig:
     max_concurrent_requests: int | None = 4
     initial_program_paths: tuple[str, ...] = ()
     initial_pool_paths: tuple[str, ...] = ()
+    sampler_islands: tuple[str, ...] = ()
     topk_children: int = 16
 
     # False is TTT Discover: non-auto, many short samples. True is AutoEvolve:
@@ -249,6 +253,7 @@ def append_agent_outputs(log_path: str, step: int, results: list[CandidateResult
                 "row": row_idx,
                 "group_idx": result.group_idx,
                 "sample_idx": result.sample_idx,
+                "island": result.island,
                 "kept": result.kept,
                 "drop_reason": result.drop_reason,
                 "parent_id": getattr(parent_state, "id", None),
@@ -338,6 +343,64 @@ def _latest_sampler_step(log_path: str) -> int:
     return latest
 
 
+def _latest_island_sampler_step(log_path: str, islands: tuple[str, ...]) -> int:
+    islands = validate_sampler_islands(islands)
+    ordinary = glob.glob(os.path.join(log_path, "puct_sampler_step_*.json"))
+    if ordinary:
+        raise ValueError(
+            "Cannot resume island sampler from a log directory containing ordinary "
+            f"PUCT sampler snapshots: {ordinary[:3]}"
+        )
+
+    island_pattern = os.path.join(log_path, "puct_sampler_island_*_step_*.json")
+    configured = set(islands)
+    seen_islands: set[str] = set()
+    for path in glob.glob(island_pattern):
+        match = re.search(r"puct_sampler_island_([A-Za-z0-9_-]+)_step_(\d+)\.json$", path)
+        if match:
+            seen_islands.add(match.group(1))
+    extra_islands = seen_islands - configured
+    if extra_islands:
+        raise ValueError(
+            "Cannot resume island sampler with snapshots for unconfigured islands: "
+            f"{sorted(extra_islands)}"
+        )
+
+    steps_by_island: dict[str, list[int]] = {}
+    for island in islands:
+        base_file = sampler_island_file_base(log_path, island)
+        base_path = base_file[:-5] if base_file.endswith(".json") else base_file
+        pattern = f"{base_path}_step_*.json"
+        steps: list[int] = []
+        for path in glob.glob(pattern):
+            match = re.search(
+                rf"puct_sampler_island_{re.escape(island)}_step_(\d+)\.json$",
+                path,
+            )
+            if match:
+                steps.append(int(match.group(1)))
+        steps_by_island[island] = sorted(steps)
+
+    populated = {island: steps for island, steps in steps_by_island.items() if steps}
+    if not populated:
+        return 0
+    missing = [island for island, steps in steps_by_island.items() if not steps]
+    if missing:
+        raise ValueError(
+            "Cannot resume island sampler; missing snapshots for islands: "
+            f"{missing}"
+        )
+
+    latest_by_island = {island: steps[-1] for island, steps in steps_by_island.items()}
+    latest_steps = set(latest_by_island.values())
+    if len(latest_steps) != 1:
+        raise ValueError(
+            "Cannot resume island sampler; latest steps differ by island: "
+            f"{latest_by_island}"
+        )
+    return latest_steps.pop()
+
+
 def _make_env(cfg: CodexNoFinetuneConfig, state: Any, sampler: StateSampler) -> Any:
     env = object.__new__(cfg.env_type)
     env.config = cfg
@@ -635,6 +698,7 @@ async def _run_candidate(
     group_idx: int,
     sample_idx: int,
     step_idx: int,
+    island: str | None,
     semaphore: asyncio.Semaphore | None,
 ) -> CandidateResult:
     prompt = ""
@@ -676,6 +740,8 @@ async def _run_candidate(
         outs = await _safe_grade(cfg, env, parsed_code, correct_format)
         metrics = _build_metrics(env, outs, response, parsed_code, correct_format)
         metrics["codex/parsed_code_source"] = parsed_code_source
+        if island is not None:
+            metrics["codex/island"] = island
         if autonomous_submission_path is not None:
             metrics["codex/autonomous_submission_path"] = autonomous_submission_path
         next_state = _maybe_create_next_state(env, step_idx, parsed_code, outs)
@@ -692,6 +758,7 @@ async def _run_candidate(
             msg=outs.msg,
             metrics=metrics,
             next_state=next_state,
+            island=island,
         )
     except Exception as exc:
         error_msg = f"{exc}\n{traceback.format_exc()}"
@@ -716,15 +783,24 @@ async def _run_candidate(
             metrics={"error": error_msg},
             next_state=None,
             error=error_msg,
+            island=island,
         )
 
 
 def _update_sampler_from_results(sampler: StateSampler, results: list[CandidateResult]) -> None:
     for result in results:
         if result.next_state is not None:
+            has_state_for_parent = getattr(sampler, "has_state_for_parent", None)
             has_state = getattr(sampler, "has_state", None)
             was_present = False
-            if has_state is not None:
+            if has_state_for_parent is not None:
+                try:
+                    was_present = bool(
+                        has_state_for_parent(result.next_state, result.parent_state)
+                    )
+                except Exception as exc:
+                    logger.warning("Failed to check sampler state before update: %s", exc)
+            elif has_state is not None:
                 try:
                     was_present = bool(has_state(result.next_state))
                 except Exception as exc:
@@ -733,6 +809,16 @@ def _update_sampler_from_results(sampler: StateSampler, results: list[CandidateR
                 sampler.update_states([result.next_state], [result.parent_state], save=False)
                 if was_present:
                     result.pool_status = "duplicate"
+                elif has_state_for_parent is not None:
+                    try:
+                        result.pool_status = (
+                            "added_to_pool"
+                            if has_state_for_parent(result.next_state, result.parent_state)
+                            else "sampler_filtered"
+                        )
+                    except Exception as exc:
+                        logger.warning("Failed to check sampler state after update: %s", exc)
+                        result.pool_status = "updated_unknown"
                 elif has_state is not None:
                     try:
                         result.pool_status = (
@@ -762,6 +848,7 @@ def _sample_table(results: list[CandidateResult]) -> list[tuple[Any, ...]]:
     for result in results:
         rows.append(
             (
+                result.island,
                 result.prompt,
                 result.response,
                 result.reward,
@@ -811,6 +898,27 @@ def _result_metrics(results: list[CandidateResult], kept_results: list[Candidate
         metrics[key] = float(np.mean(values))
         metrics[f"{key}/min"] = float(np.min(values))
         metrics[f"{key}/max"] = float(np.max(values))
+
+    island_names = sorted({result.island for result in results if result.island})
+    for island in island_names:
+        island_results = [result for result in results if result.island == island]
+        island_kept = [result for result in kept_results if result.island == island]
+        island_rewards = [result.reward for result in island_kept]
+        metrics[f"codex/island/{island}/total_samples"] = len(island_results)
+        metrics[f"codex/island/{island}/kept_samples"] = len(island_kept)
+        metrics[f"codex/island/{island}/failed_samples"] = sum(
+            1 for result in island_results if result.error
+        )
+        metrics[f"codex/island/{island}/correct_samples"] = sum(
+            1 for result in island_kept if result.correctness > 0
+        )
+        if island_rewards:
+            metrics[f"codex/island/{island}/reward_mean"] = float(
+                np.mean(island_rewards)
+            )
+            metrics[f"codex/island/{island}/reward_max"] = float(
+                np.max(island_rewards)
+            )
     return metrics
 
 
@@ -828,7 +936,13 @@ async def sample_batch(
     )
 
     tasks = []
+    get_island_for_parent = getattr(sampler, "get_island_for_parent", None)
     for group_idx, parent_state in enumerate(parent_states):
+        island = (
+            get_island_for_parent(parent_state)
+            if get_island_for_parent is not None
+            else None
+        )
         for sample_idx in range(cfg.group_size):
             tasks.append(
                 asyncio.create_task(
@@ -839,6 +953,7 @@ async def sample_batch(
                         group_idx=group_idx,
                         sample_idx=sample_idx,
                         step_idx=i_batch,
+                        island=island,
                         semaphore=semaphore,
                     ),
                     name=f"codex_sample_{group_idx}_{sample_idx}",
@@ -884,6 +999,7 @@ def _log_agent_tables(
     if not table_data:
         return
     columns = [
+        "Island",
         "Prompt",
         "Gen Sequence",
         "Reward",
@@ -960,6 +1076,7 @@ def _build_sampler(cfg: CodexNoFinetuneConfig, start_batch: int) -> StateSampler
         batch_size=cfg.groups_per_batch,
         resume_step=start_batch if start_batch > 0 else None,
         topk_children=cfg.topk_children,
+        islands=cfg.sampler_islands,
     )
 
 
@@ -987,10 +1104,24 @@ async def main(cfg: CodexNoFinetuneConfig) -> None:
         raise ValueError("num_epochs must be >= 1")
     if not cfg.log_path:
         raise ValueError("log_path is required")
+    sampler_islands = validate_sampler_islands(cfg.sampler_islands)
+    object.__setattr__(cfg, "sampler_islands", sampler_islands)
+    if sampler_islands and cfg.groups_per_batch != len(sampler_islands):
+        raise ValueError(
+            "groups_per_batch must equal the number of sampler islands when "
+            f"islands are enabled: groups_per_batch={cfg.groups_per_batch}, "
+            f"islands={len(sampler_islands)}"
+        )
 
     object.__setattr__(cfg, "log_path", os.path.expanduser(cfg.log_path))
     os.makedirs(cfg.log_path, exist_ok=True)
 
+    start_batch = (
+        _latest_island_sampler_step(cfg.log_path, sampler_islands)
+        if sampler_islands
+        else _latest_sampler_step(cfg.log_path)
+    )
+
     ml_logger = MetricsLogger(
         log_path=cfg.log_path,
         wandb_project=cfg.wandb_project,
@@ -998,7 +1129,6 @@ async def main(cfg: CodexNoFinetuneConfig) -> None:
         config=cfg,
     )
 
-    start_batch = _latest_sampler_step(cfg.log_path)
     sampler = _build_sampler(cfg, start_batch)
     _seed_sampler(cfg, sampler)
 
````
</details>

