# codex/diversity-parent-portfolio

## Summary

提供 parent portfolio lanes（puct/top/recent/underexplored），按 lane 选择父节点并记录 parent_lane，让一个 batch 覆盖不同父选择策略。

## Branch State

- Worktree: `/opt/tiger/discover-parent-portfolio`
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

- `parent_lane`
- `parent_portfolio`

### Constants

- `PARENT_PORTFOLIO_LANES`

### Classes

- `_PUCTEntry`

### Functions

- `validate_parent_portfolio_lanes`
- `sample_states_portfolio`
- `_finite_value`
- `_value_or_neg_inf`
- `_state_timestep`
- `_has_parent_payload`
- `_is_valid_parent`
- `_portfolio_candidate_pool`
- `_puct_entries`
- `_zero_puct_entry`
- `_set_last_sampled`
- `_refresh_sampled_initial_states`
- `_normalized_values`
- `_rank_portfolio_lane`
- `_select_from_rankings`
- `sample_states`
- `_parent_group_count`

## Diff Summary

- Worktree tracked shortstat: `2 files changed, 372 insertions(+), 41 deletions(-)`
- Untracked files: `2`

### Worktree Status

````text
 M ttt_discover/codex_utils/sampler.py
 M ttt_discover/rl/codex_no_finetune.py
?? repro/gpu_mode/
?? repro/run_discovery.py
````
### Tracked Worktree Files

````text
M	ttt_discover/codex_utils/sampler.py
M	ttt_discover/rl/codex_no_finetune.py
````
### Untracked Files

- `repro/gpu_mode/run_0608_parent_portfolio.sh (936 bytes)`
- `repro/run_discovery.py (16044 bytes)`

## Untracked File Snapshots

These files are not part of `git diff` yet, but they are part of the worktree implementation state recorded here.

### `repro/gpu_mode/run_0608_parent_portfolio.sh`

````bash
#!/usr/bin/env bash
set -euo pipefail

# TTT Discover parent portfolio: non-auto, read-only Codex samples.
# Portfolio lanes determine the parent group count for each outer round.
python repro/run_discovery.py \
    --task trimul \
    --runner codex_no_finetune \
    --experiment-name trimul_0608_parent_portfolio_gpu2 \
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
    --codex-parent-lane puct \
    --codex-parent-lane top \
    --codex-parent-lane recent \
    --codex-parent-lane underexplored
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

from ttt_discover.codex_utils.sampler import (  # noqa: E402
    PARENT_PORTFOLIO_LANES,
    validate_parent_portfolio_lanes,
)


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
        "--codex-parent-lane",
        action="append",
        choices=PARENT_PORTFOLIO_LANES,
        default=None,
        help="Parent portfolio lane. May be repeated; omit to use normal PUCT sampling.",
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
    codex_parent_portfolio = validate_parent_portfolio_lanes(
        tuple(args.codex_parent_lane or ())
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
        print(f"codex_parent_portfolio={codex_parent_portfolio!r}")
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
            parent_portfolio=codex_parent_portfolio,
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
 ttt_discover/codex_utils/sampler.py  | 363 +++++++++++++++++++++++++++++++----
 ttt_discover/rl/codex_no_finetune.py |  50 ++++-
 2 files changed, 372 insertions(+), 41 deletions(-)
````
## Raw Diff

<details>
<summary>Tracked worktree patch</summary>

````diff
diff --git a/ttt_discover/codex_utils/sampler.py b/ttt_discover/codex_utils/sampler.py
index 5c5d4d1..776c465 100644
--- a/ttt_discover/codex_utils/sampler.py
+++ b/ttt_discover/codex_utils/sampler.py
@@ -2,13 +2,14 @@
 from __future__ import annotations
 from abc import ABC, abstractmethod
 from contextlib import contextmanager
+from dataclasses import dataclass
 import json
 import logging
 import os
 from pathlib import Path
 import threading
 import time
-from typing import Any, Callable
+from typing import Any, Callable, Sequence
 
 import numpy as np
 
@@ -16,6 +17,29 @@ from ttt_discover.codex_utils.runtime import State, state_from_dict, to_json_ser
 
 logger = logging.getLogger(__name__)
 
+PARENT_PORTFOLIO_LANES = ("puct", "top", "recent", "underexplored")
+
+
+def validate_parent_portfolio_lanes(lanes: Sequence[str]) -> tuple[str, ...]:
+    lanes = tuple(lanes)
+    unknown = [lane for lane in lanes if lane not in PARENT_PORTFOLIO_LANES]
+    if unknown:
+        valid = ", ".join(PARENT_PORTFOLIO_LANES)
+        raise ValueError(f"Unknown parent portfolio lane(s): {unknown}. Valid lanes: {valid}")
+    return lanes
+
+
+@dataclass(frozen=True)
+class _PUCTEntry:
+    state: State
+    index: int
+    value: float
+    n: int
+    Q: float
+    P: float
+    bonus: float
+    score: float
+
 
 @contextmanager
 def _file_lock(lock_path: str, *, poll_s: float = 0.05, stale_s: float = 600.0):
@@ -74,6 +98,20 @@ class StateSampler(ABC):
         """Sample states to continue search from."""
         pass
 
+    def sample_states_portfolio(self, lanes: Sequence[str]) -> list[tuple[State, str]]:
+        """Sample states by named parent-selection lanes."""
+        lanes = validate_parent_portfolio_lanes(lanes)
+        if not lanes:
+            return []
+        if any(lane != "puct" for lane in lanes):
+            raise NotImplementedError(
+                f"{self.__class__.__name__} supports only puct parent portfolio lanes"
+            )
+        return [
+            (state, "puct")
+            for state in self.sample_states(len(lanes))
+        ]
+
     @abstractmethod
     def update_states(self, states: list[State], parent_states: list[State], save: bool = True, step: int | None = None):
         """Update internal storage with new states. Sets parent info automatically."""
@@ -450,6 +488,7 @@ class PUCTSampler(StateSampler):
         if values.size == 0:
             return 1.0
         v = values[mask] if mask is not None else values
+        v = v[np.isfinite(v)]
         return float(max(np.max(v) - np.min(v), 1e-6)) if v.size > 0 else 1.0
 
     def _compute_prior(self, values: np.ndarray, scale: float) -> np.ndarray:
@@ -489,65 +528,315 @@ class PUCTSampler(StateSampler):
                     queue.append(child_id)
         return lineage
 
-    def sample_states(self, num_states: int) -> list[State]:
-        initial_ids = {s.id for s in self._initial_states}
-        candidates = list(self._states)
+    def _finite_value(self, state: State) -> float | None:
+        try:
+            value = float(state.value)
+        except (TypeError, ValueError):
+            return None
+        if not np.isfinite(value):
+            return None
+        return value
 
+    def _value_or_neg_inf(self, state: State) -> float:
+        value = self._finite_value(state)
+        return value if value is not None else float("-inf")
+
+    def _state_timestep(self, state: State) -> int:
+        try:
+            return int(state.timestep)
+        except (TypeError, ValueError):
+            return -10**18
+
+    def _has_parent_payload(self, state: State) -> bool:
+        code = getattr(state, "code", None)
+        if isinstance(code, str) and code.strip():
+            return True
+        if code:
+            return True
+        construction = getattr(state, "construction", None)
+        if construction is None:
+            return False
+        try:
+            return len(construction) > 0
+        except TypeError:
+            return bool(construction)
+
+    def _is_valid_parent(self, state: State) -> bool:
+        return self._finite_value(state) is not None and self._has_parent_payload(state)
+
+    def _portfolio_candidate_pool(self) -> list[State]:
+        valid = [state for state in self._states if self._is_valid_parent(state)]
+        if valid:
+            return valid
+
+        if self._initial_states:
+            return list(self._initial_states)
+        finite = [state for state in self._states if self._finite_value(state) is not None]
+        if finite:
+            return finite
+        return list(self._states)
+
+    def _puct_entries(self) -> list[_PUCTEntry]:
+        candidates = list(self._states)
         if not candidates:
-            picked = [
-                create_initial_state(self.env_type, self.problem_type)
-                for _ in range(num_states)
-            ]
-            self._last_sampled_states = picked
-            self._last_sampled_indices = []
-            self._last_puct_stats = [(0, 0.0, 0.0, 0.0, 0.0) for _ in picked]
-            return picked
+            self._last_scale = 1.0
+            return []
 
-        vals = np.array([float(s.value if s.value is not None else float("-inf")) for s in candidates])
+        initial_ids = {s.id for s in self._initial_states}
+        vals = np.array([self._value_or_neg_inf(s) for s in candidates])
         non_initial_mask = np.array([s.id not in initial_ids for s in candidates])
-        scale = self._compute_scale(vals, non_initial_mask if non_initial_mask.any() else None)
+        scale = self._compute_scale(
+            vals,
+            non_initial_mask if non_initial_mask.any() else None,
+        )
         self._last_scale = scale
         P = self._compute_prior(vals, scale)
         sqrtT = np.sqrt(1.0 + self._T)
 
-        scores = []
-        for i, s in enumerate(candidates):
-            n = self._n.get(s.id, 0)
-            m = self._m.get(s.id, vals[i])
-            Q = m if n > 0 else vals[i]
+        entries: list[_PUCTEntry] = []
+        for i, state in enumerate(candidates):
+            n = self._n.get(state.id, 0)
+            value = float(vals[i])
+            m = self._m.get(state.id, value)
+            try:
+                m_value = float(m)
+            except (TypeError, ValueError):
+                m_value = value
+            if not np.isfinite(m_value):
+                m_value = value
+            Q = m_value if n > 0 else value
             bonus = self.puct_c * scale * P[i] * sqrtT / (1.0 + n)
             score = Q + bonus
-            scores.append((score, vals[i], s, n, Q, P[i], bonus))
+            entries.append(
+                _PUCTEntry(
+                    state=state,
+                    index=i,
+                    value=value,
+                    n=n,
+                    Q=Q,
+                    P=float(P[i]),
+                    bonus=float(bonus),
+                    score=float(score),
+                )
+            )
+
+        entries.sort(key=lambda entry: (entry.score, entry.value), reverse=True)
+        return entries
+
+    def _zero_puct_entry(self, state: State) -> _PUCTEntry:
+        value = self._finite_value(state)
+        if value is None:
+            value = 0.0
+        n = self._n.get(state.id, 0)
+        try:
+            Q = float(self._m.get(state.id, value)) if n > 0 else value
+        except (TypeError, ValueError):
+            Q = value
+        if not np.isfinite(Q):
+            Q = value
+        return _PUCTEntry(
+            state=state,
+            index=-1,
+            value=value,
+            n=n,
+            Q=Q,
+            P=0.0,
+            bonus=0.0,
+            score=Q,
+        )
+
+    def _set_last_sampled(
+        self,
+        picked: list[State],
+        entries: list[_PUCTEntry | None],
+    ) -> None:
+        state_id_to_idx = {s.id: i for i, s in enumerate(self._states)}
+        self._last_sampled_states = list(picked)
+        self._last_sampled_indices = [state_id_to_idx.get(s.id, -1) for s in picked]
+        padded_entries = list(entries)
+        while len(padded_entries) < len(picked):
+            padded_entries.append(None)
+        stats = []
+        for state, entry in zip(picked, padded_entries, strict=False):
+            if entry is None:
+                entry = self._zero_puct_entry(state)
+            stats.append((entry.n, entry.Q, entry.P, entry.bonus, entry.score))
+        self._last_puct_stats = stats
+
+    def _refresh_sampled_initial_states(self, picked: list[State]) -> None:
+        initial_ids = {s.id for s in self._initial_states}
+        refreshed: set[str] = set()
+        for state in picked:
+            if state.id in initial_ids and state.id not in refreshed:
+                self._refresh_random_construction(state)
+                refreshed.add(state.id)
+
+    def _normalized_values(self, candidates: list[State]) -> dict[str, float]:
+        values: dict[str, float] = {}
+        finite_values: list[float] = []
+        for state in candidates:
+            value = self._finite_value(state)
+            if value is None:
+                continue
+            values[state.id] = value
+            finite_values.append(value)
+        if not finite_values:
+            return {state.id: float("-inf") for state in candidates}
+        lo = min(finite_values)
+        scale = max(max(finite_values) - lo, 1e-6)
+        return {
+            state.id: (
+                (values[state.id] - lo) / scale
+                if state.id in values
+                else float("-inf")
+            )
+            for state in candidates
+        }
+
+    def _rank_portfolio_lane(
+        self,
+        lane: str,
+        candidates: list[State],
+        puct_ranked: list[State],
+    ) -> list[State]:
+        if lane == "puct":
+            return puct_ranked
+        if lane == "top":
+            normalized = self._normalized_values(candidates)
+            return sorted(
+                candidates,
+                key=lambda state: (
+                    normalized.get(state.id, float("-inf")),
+                    self._value_or_neg_inf(state),
+                    self._state_timestep(state),
+                    state.id,
+                ),
+                reverse=True,
+            )
+        if lane == "recent":
+            return sorted(
+                candidates,
+                key=lambda state: (
+                    self._state_timestep(state),
+                    self._value_or_neg_inf(state),
+                    state.id,
+                ),
+                reverse=True,
+            )
+        if lane == "underexplored":
+            return sorted(
+                candidates,
+                key=lambda state: (
+                    self._n.get(state.id, 0),
+                    -self._value_or_neg_inf(state),
+                    -self._state_timestep(state),
+                    state.id,
+                ),
+            )
+        raise ValueError(f"Unknown parent portfolio lane: {lane}")
+
+    def _select_from_rankings(
+        self,
+        lane_ranked: list[State],
+        puct_ranked: list[State],
+        *,
+        blocked_ids: set[str],
+        picked_ids: set[str],
+    ) -> State | None:
+        for state in lane_ranked:
+            if state.id not in blocked_ids:
+                return state
+        for state in lane_ranked:
+            if state.id not in picked_ids:
+                return state
+        for state in puct_ranked:
+            if state.id not in picked_ids:
+                return state
+        if lane_ranked:
+            return lane_ranked[0]
+        if puct_ranked:
+            return puct_ranked[0]
+        return None
 
-        scores.sort(key=lambda x: (x[0], x[1]), reverse=True)
+    def sample_states(self, num_states: int) -> list[State]:
+        candidates = list(self._states)
+
+        if not candidates:
+            picked = [
+                create_initial_state(self.env_type, self.problem_type)
+                for _ in range(num_states)
+            ]
+            self._set_last_sampled(picked, [])
+            return picked
+
+        entries = self._puct_entries()
 
         if num_states > 1:
             children_map = self._build_children_map()
             picked, top_scores, blocked_ids = [], [], set()
-            for entry in scores:
-                s = entry[2]
-                if s.id in blocked_ids:
+            for entry in entries:
+                state = entry.state
+                if state.id in blocked_ids:
                     continue
-                picked.append(s)
+                picked.append(state)
                 top_scores.append(entry)
-                blocked_ids.update(self._get_full_lineage(s, children_map))
+                blocked_ids.update(self._get_full_lineage(state, children_map))
                 if len(picked) >= num_states:
                     break
         else:
-            top_scores = scores[:num_states]
-            picked = [t[2] for t in top_scores]
-
-        state_id_to_idx = {s.id: i for i, s in enumerate(self._states)}
-        self._last_sampled_states = picked
-        self._last_sampled_indices = [state_id_to_idx.get(s.id, -1) for s in picked]
-        self._last_puct_stats = [(t[3], t[4], t[5], t[6], t[0]) for t in top_scores]
-
-        for s in picked:
-            if s.id in initial_ids:
-                self._refresh_random_construction(s)
+            top_scores = entries[:num_states]
+            picked = [entry.state for entry in top_scores]
 
+        self._set_last_sampled(picked, top_scores)
+        self._refresh_sampled_initial_states(picked)
         return picked
 
+    def sample_states_portfolio(self, lanes: Sequence[str]) -> list[tuple[State, str]]:
+        lanes = validate_parent_portfolio_lanes(lanes)
+        if not lanes:
+            self._set_last_sampled([], [])
+            return []
+
+        puct_entries = self._puct_entries()
+        entry_by_id = {entry.state.id: entry for entry in puct_entries}
+        candidate_pool = self._portfolio_candidate_pool()
+        candidate_ids = {state.id for state in candidate_pool}
+        puct_ranked = [
+            entry.state
+            for entry in puct_entries
+            if entry.state.id in candidate_ids
+        ]
+        if not puct_ranked:
+            puct_ranked = list(candidate_pool)
+
+        children_map = self._build_children_map()
+        picked: list[State] = []
+        picked_ids: set[str] = set()
+        blocked_ids: set[str] = set()
+
+        for lane in lanes:
+            lane_ranked = self._rank_portfolio_lane(
+                lane,
+                candidate_pool,
+                puct_ranked,
+            )
+            selected = self._select_from_rankings(
+                lane_ranked,
+                puct_ranked,
+                blocked_ids=blocked_ids,
+                picked_ids=picked_ids,
+            )
+            if selected is None:
+                selected = create_initial_state(self.env_type, self.problem_type)
+            picked.append(selected)
+            picked_ids.add(selected.id)
+            blocked_ids.update(self._get_full_lineage(selected, children_map))
+
+        selected_entries = [entry_by_id.get(state.id) for state in picked]
+        self._set_last_sampled(picked, selected_entries)
+        self._refresh_sampled_initial_states(picked)
+        return list(zip(picked, lanes, strict=False))
+
     def update_states(self, states: list[State], parent_states: list[State], save: bool = True, step: int | None = None):
         if not states:
             return
diff --git a/ttt_discover/rl/codex_no_finetune.py b/ttt_discover/rl/codex_no_finetune.py
index 18a0e9e..65438c6 100644
--- a/ttt_discover/rl/codex_no_finetune.py
+++ b/ttt_discover/rl/codex_no_finetune.py
@@ -75,6 +75,7 @@ from ttt_discover.codex_utils.sampler import (
     create_sampler,
     seed_initial_pool_paths,
     seed_initial_program_paths,
+    validate_parent_portfolio_lanes,
 )
 
 logger = logging.getLogger(__name__)
@@ -112,6 +113,7 @@ class CandidateResult:
     drop_reason: str | None = None
     pool_status: str | None = None
     sampler_error: str | None = None
+    parent_lane: str | None = None
 
 
 @chz.chz
@@ -136,6 +138,7 @@ class CodexNoFinetuneConfig:
     max_concurrent_requests: int | None = 4
     initial_program_paths: tuple[str, ...] = ()
     initial_pool_paths: tuple[str, ...] = ()
+    parent_portfolio: tuple[str, ...] = ()
     topk_children: int = 16
 
     # False is TTT Discover: non-auto, many short samples. True is AutoEvolve:
@@ -237,6 +240,12 @@ def all_same(xs: list[Any]) -> bool:
     return bool(xs) and all(x == xs[0] for x in xs)
 
 
+def _parent_group_count(cfg: CodexNoFinetuneConfig) -> int:
+    if cfg.parent_portfolio:
+        return len(cfg.parent_portfolio)
+    return cfg.groups_per_batch
+
+
 def append_agent_outputs(log_path: str, step: int, results: list[CandidateResult]) -> None:
     os.makedirs(log_path, exist_ok=True)
     output_path = os.path.join(log_path, "agent_outputs.jsonl")
@@ -251,6 +260,7 @@ def append_agent_outputs(log_path: str, step: int, results: list[CandidateResult
                 "sample_idx": result.sample_idx,
                 "kept": result.kept,
                 "drop_reason": result.drop_reason,
+                "parent_lane": result.parent_lane,
                 "parent_id": getattr(parent_state, "id", None),
                 "parent_timestep": getattr(parent_state, "timestep", None),
                 "parent_value": getattr(parent_state, "value", None),
@@ -636,6 +646,7 @@ async def _run_candidate(
     sample_idx: int,
     step_idx: int,
     semaphore: asyncio.Semaphore | None,
+    parent_lane: str | None = None,
 ) -> CandidateResult:
     prompt = ""
     response = ""
@@ -676,6 +687,8 @@ async def _run_candidate(
         outs = await _safe_grade(cfg, env, parsed_code, correct_format)
         metrics = _build_metrics(env, outs, response, parsed_code, correct_format)
         metrics["codex/parsed_code_source"] = parsed_code_source
+        if parent_lane is not None:
+            metrics["codex/parent_lane"] = parent_lane
         if autonomous_submission_path is not None:
             metrics["codex/autonomous_submission_path"] = autonomous_submission_path
         next_state = _maybe_create_next_state(env, step_idx, parsed_code, outs)
@@ -692,6 +705,7 @@ async def _run_candidate(
             msg=outs.msg,
             metrics=metrics,
             next_state=next_state,
+            parent_lane=parent_lane,
         )
     except Exception as exc:
         error_msg = f"{exc}\n{traceback.format_exc()}"
@@ -702,6 +716,9 @@ async def _run_candidate(
             sample_idx,
             exc,
         )
+        metrics = {"error": error_msg}
+        if parent_lane is not None:
+            metrics["codex/parent_lane"] = parent_lane
         return CandidateResult(
             parent_state=parent_state,
             group_idx=group_idx,
@@ -713,9 +730,10 @@ async def _run_candidate(
             correctness=0.0,
             raw_score=None,
             msg=error_msg,
-            metrics={"error": error_msg},
+            metrics=metrics,
             next_state=None,
             error=error_msg,
+            parent_lane=parent_lane,
         )
 
 
@@ -820,7 +838,14 @@ async def sample_batch(
     i_batch: int,
 ) -> tuple[list[CandidateResult], dict[str, Any], list[CandidateResult]]:
     metrics: dict[str, Any] = {}
-    parent_states = sampler.sample_states(cfg.groups_per_batch)
+    if cfg.parent_portfolio:
+        parent_pairs = sampler.sample_states_portfolio(cfg.parent_portfolio)
+        parent_states = [state for state, _lane in parent_pairs]
+        parent_lanes: list[str | None] = [lane for _state, lane in parent_pairs]
+    else:
+        parent_states = sampler.sample_states(cfg.groups_per_batch)
+        parent_lanes = [None] * len(parent_states)
+
     semaphore = (
         asyncio.Semaphore(cfg.max_concurrent_requests)
         if cfg.max_concurrent_requests is not None
@@ -828,7 +853,8 @@ async def sample_batch(
     )
 
     tasks = []
-    for group_idx, parent_state in enumerate(parent_states):
+    parent_groups = zip(parent_states, parent_lanes, strict=False)
+    for group_idx, (parent_state, parent_lane) in enumerate(parent_groups):
         for sample_idx in range(cfg.group_size):
             tasks.append(
                 asyncio.create_task(
@@ -840,6 +866,7 @@ async def sample_batch(
                         sample_idx=sample_idx,
                         step_idx=i_batch,
                         semaphore=semaphore,
+                        parent_lane=parent_lane,
                     ),
                     name=f"codex_sample_{group_idx}_{sample_idx}",
                 )
@@ -869,6 +896,10 @@ async def sample_batch(
         kept_results.extend(group_results)
 
     metrics["codex/parent_states"] = len(parent_states)
+    if cfg.parent_portfolio:
+        metrics["codex/parent_portfolio"] = ",".join(cfg.parent_portfolio)
+        for lane in cfg.parent_portfolio:
+            metrics[f"codex/parent_lane_count/{lane}"] = parent_lanes.count(lane)
     metrics["codex/dropped_constant_groups"] = dropped_constant_groups
     metrics.update(_result_metrics(results, kept_results))
     return kept_results, metrics, results
@@ -957,7 +988,7 @@ def _build_sampler(cfg: CodexNoFinetuneConfig, start_batch: int) -> StateSampler
         log_path=cfg.log_path,
         env_type=cfg.env_type,
         problem_type=cfg.problem_type,
-        batch_size=cfg.groups_per_batch,
+        batch_size=_parent_group_count(cfg),
         resume_step=start_batch if start_batch > 0 else None,
         topk_children=cfg.topk_children,
     )
@@ -987,6 +1018,17 @@ async def main(cfg: CodexNoFinetuneConfig) -> None:
         raise ValueError("num_epochs must be >= 1")
     if not cfg.log_path:
         raise ValueError("log_path is required")
+    object.__setattr__(
+        cfg,
+        "parent_portfolio",
+        validate_parent_portfolio_lanes(cfg.parent_portfolio),
+    )
+    if cfg.parent_portfolio and len(cfg.parent_portfolio) != cfg.groups_per_batch:
+        logger.info(
+            "Parent portfolio lanes determine group count (%s); groups_per_batch=%s",
+            len(cfg.parent_portfolio),
+            cfg.groups_per_batch,
+        )
 
     object.__setattr__(cfg, "log_path", os.path.expanduser(cfg.log_path))
     os.makedirs(cfg.log_path, exist_ok=True)
````
</details>

