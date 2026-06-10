# codex/diversity-focus-slices

## Summary

不是 sampler 平衡，而是 prompt 层多样性：为 GPUMode trimul 定义多个 evaluation focus slice，并按样本序号循环追加到 prompt。

Note: 这个分支主要增加任务 focus prompt，不是父节点平衡 sampler。

## Branch State

- Worktree: `/opt/tiger/discover-focus-slices`
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

- `focus_slices`

### Constants

- None

### Classes

- None

### Functions

- `get_diversity_focus_blocks`
- `_get_focus_slice_blocks`
- `validate_focus_slices`
- `_focus_slice_for_sample`
- `_append_focus_slice_prompt`

## Diff Summary

- Worktree tracked shortstat: `2 files changed, 109 insertions(+), 1 deletion(-)`
- Untracked files: `2`

### Worktree Status

````text
 M examples/gpu_mode/env.py
 M ttt_discover/rl/codex_no_finetune.py
?? repro/gpu_mode/
?? repro/run_discovery.py
````
### Tracked Worktree Files

````text
M	examples/gpu_mode/env.py
M	ttt_discover/rl/codex_no_finetune.py
````
### Untracked Files

- `repro/gpu_mode/run_0608_focus_slices.sh (1044 bytes)`
- `repro/run_discovery.py (16152 bytes)`

## Untracked File Snapshots

These files are not part of `git diff` yet, but they are part of the worktree implementation state recorded here.

### `repro/gpu_mode/run_0608_focus_slices.sh`

````bash
#!/usr/bin/env bash
set -euo pipefail

# TTT Discover focus-slice run: six non-auto Codex samples per parent, one
# prompt-only evaluation slice per sample.
python repro/run_discovery.py \
    --task trimul \
    --runner codex_no_finetune \
    --experiment-name trimul_0608_focus_slices_gpu2 \
    --log-root codex_runs/trimul_focus_slices \
    --gpu 2 \
    --cuda-device-order PCI_BUS_ID \
    --torch-cuda-arch-list 8.0 \
    --num-epochs 50 \
    --group-size 6 \
    --groups-per-batch 1 \
    --num-cpus-per-task 1 \
    --eval-timeout 1200 \
    --wandb-project "" \
    --codex-backend cli \
    --codex-model-name gpt-5.5 \
    --codex-cli-command codex \
    --codex-cli-sandbox read-only \
    --codex-cli-timeout 600 \
    --codex-max-concurrent-requests 4 \
    --codex-focus-slice balanced \
    --codex-focus-slice n256_n512_dim128_nomask \
    --codex-focus-slice n256_dim384_masked \
    --codex-focus-slice n768_dim384_masked \
    --codex-focus-slice n768_n1024_dim128_cauchy \
    --codex-focus-slice n1024_dim384_nomask
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
        "--codex-focus-slice",
        action="append",
        default=None,
        help=(
            "Prompt-only Codex evaluation-slice focus. May be repeated; omit to disable."
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
    codex_focus_slices = tuple(args.codex_focus_slice or ())
    if codex_focus_slices:
        from ttt_discover.rl.codex_no_finetune import validate_focus_slices

        validate_focus_slices(
            argparse.Namespace(
                env_type=env_type,
                problem_type=problem_type,
                focus_slices=codex_focus_slices,
            )
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
        print(f"codex_focus_slices={codex_focus_slices!r}")
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
            focus_slices=codex_focus_slices,
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
 examples/gpu_mode/env.py             | 50 ++++++++++++++++++++++++++++++
 ttt_discover/rl/codex_no_finetune.py | 60 +++++++++++++++++++++++++++++++++++-
 2 files changed, 109 insertions(+), 1 deletion(-)
````
## Raw Diff

<details>
<summary>Tracked worktree patch</summary>

````diff
diff --git a/examples/gpu_mode/env.py b/examples/gpu_mode/env.py
index a37953f..4bd1d4e 100644
--- a/examples/gpu_mode/env.py
+++ b/examples/gpu_mode/env.py
@@ -338,6 +338,56 @@ class GpuModeEnv(Environment):
             )
         raise ValueError(f"Unknown problem_type: {problem_type}")
 
+    @classmethod
+    def get_diversity_focus_blocks(cls, problem_type: str) -> dict[str, str]:
+        if problem_type != "trimul":
+            return {}
+
+        mitigation = (
+            "Bias this sample's reasoning toward this slice. Keep correctness and "
+            "broad performance across all listed cases; any shape-specific dispatch "
+            "must preserve correct fallbacks. Do not branch on distribution; use "
+            "cauchy only as numerical-stability guidance."
+        )
+        return {
+            "balanced": (
+                "Focus on balanced TriMul performance across all listed runtime "
+                "cases. Treat every listed shape, mask setting, and input "
+                "distribution as important; no single shape should dominate "
+                f"tradeoffs. {mitigation}"
+            ),
+            "n256_n512_dim128_nomask": (
+                "Focus on N=256 and N=512 with dim=128, hidden_dim=128, no mask, "
+                "and normal inputs. Pay special attention to launch/setup cost, "
+                "temporary allocation overhead, and small-to-medium shape "
+                f"efficiency. {mitigation}"
+            ),
+            "n256_dim384_masked": (
+                "Focus on N=256 with dim=384, hidden_dim=128, masked normal "
+                "inputs. Pay special attention to projection, gate, layernorm, "
+                "and mask overhead where fixed costs can dominate the contraction. "
+                f"{mitigation}"
+            ),
+            "n768_dim384_masked": (
+                "Focus on N=768 with dim=384, hidden_dim=128, masked normal "
+                "inputs. Pay special attention to memory pressure, mask handling, "
+                "and avoiding extra materialization on the larger masked path. "
+                f"{mitigation}"
+            ),
+            "n768_n1024_dim128_cauchy": (
+                "Focus on N=768 and N=1024 with dim=128, hidden_dim=128, no mask, "
+                "and cauchy inputs. Pay special attention to large contraction "
+                "throughput and numerical robustness under heavy-tailed values. "
+                f"{mitigation}"
+            ),
+            "n1024_dim384_nomask": (
+                "Focus on N=1024 with dim=384, hidden_dim=128, no mask, and "
+                "normal inputs. Pay special attention to the largest memory and "
+                "projection path, including bandwidth, tiling, and avoiding "
+                f"unnecessary full-size intermediates. {mitigation}"
+            ),
+        }
+
     def _should_keep_code_separators(self) -> bool:
         return False
 
diff --git a/ttt_discover/rl/codex_no_finetune.py b/ttt_discover/rl/codex_no_finetune.py
index 18a0e9e..773cbc4 100644
--- a/ttt_discover/rl/codex_no_finetune.py
+++ b/ttt_discover/rl/codex_no_finetune.py
@@ -137,6 +137,7 @@ class CodexNoFinetuneConfig:
     initial_program_paths: tuple[str, ...] = ()
     initial_pool_paths: tuple[str, ...] = ()
     topk_children: int = 16
+    focus_slices: tuple[str, ...] = ()
 
     # False is TTT Discover: non-auto, many short samples. True is AutoEvolve:
     # one Codex deep-dive workspace per sample with file edits and shell access.
@@ -352,6 +353,54 @@ def _make_env(cfg: CodexNoFinetuneConfig, state: Any, sampler: StateSampler) ->
     return env
 
 
+def _get_focus_slice_blocks(env_type: type, problem_type: str) -> dict[str, str]:
+    get_blocks = getattr(env_type, "get_diversity_focus_blocks", None)
+    if get_blocks is None:
+        return {}
+    blocks = get_blocks(problem_type)
+    if blocks is None:
+        return {}
+    if not isinstance(blocks, dict):
+        raise TypeError(
+            f"{env_type.__name__}.get_diversity_focus_blocks must return dict[str, str]"
+        )
+    return {str(name): str(block) for name, block in blocks.items()}
+
+
+def validate_focus_slices(cfg: Any) -> None:
+    focus_slices = tuple(getattr(cfg, "focus_slices", ()) or ())
+    if not focus_slices:
+        return
+
+    blocks = _get_focus_slice_blocks(cfg.env_type, cfg.problem_type)
+    unknown = sorted({name for name in focus_slices if name not in blocks})
+    if unknown:
+        available = ", ".join(sorted(blocks)) if blocks else "<none>"
+        raise ValueError(
+            "Unknown Codex focus slice(s): "
+            f"{', '.join(unknown)}. Available for problem_type "
+            f"{cfg.problem_type!r}: {available}"
+        )
+
+
+def _focus_slice_for_sample(cfg: CodexNoFinetuneConfig, sample_idx: int) -> str | None:
+    if not cfg.focus_slices:
+        return None
+    return cfg.focus_slices[sample_idx % len(cfg.focus_slices)]
+
+
+def _append_focus_slice_prompt(
+    prompt: str,
+    cfg: CodexNoFinetuneConfig,
+    focus_slice: str | None,
+) -> str:
+    if focus_slice is None:
+        return prompt
+    blocks = _get_focus_slice_blocks(cfg.env_type, cfg.problem_type)
+    block = blocks[focus_slice]
+    return f"{prompt}\n\n--- Evaluation Slice Focus: {focus_slice} ---\n{block}"
+
+
 def _invalid_result(msg: str) -> VerifyResult:
     return VerifyResult(
         reward=0.0,
@@ -640,9 +689,11 @@ async def _run_candidate(
     prompt = ""
     response = ""
     parsed_code = ""
+    focus_slice = _focus_slice_for_sample(cfg, sample_idx)
     try:
         env = _make_env(cfg, parent_state, sampler)
         prompt = env.get_question()
+        prompt = _append_focus_slice_prompt(prompt, cfg, focus_slice)
         completer = _make_completer(
             cfg,
             semaphore=semaphore,
@@ -676,6 +727,8 @@ async def _run_candidate(
         outs = await _safe_grade(cfg, env, parsed_code, correct_format)
         metrics = _build_metrics(env, outs, response, parsed_code, correct_format)
         metrics["codex/parsed_code_source"] = parsed_code_source
+        metrics["codex/focus_slice"] = focus_slice
+        metrics["codex/focus_slice_enabled"] = focus_slice is not None
         if autonomous_submission_path is not None:
             metrics["codex/autonomous_submission_path"] = autonomous_submission_path
         next_state = _maybe_create_next_state(env, step_idx, parsed_code, outs)
@@ -713,7 +766,11 @@ async def _run_candidate(
             correctness=0.0,
             raw_score=None,
             msg=error_msg,
-            metrics={"error": error_msg},
+            metrics={
+                "error": error_msg,
+                "codex/focus_slice": focus_slice,
+                "codex/focus_slice_enabled": focus_slice is not None,
+            },
             next_state=None,
             error=error_msg,
         )
@@ -987,6 +1044,7 @@ async def main(cfg: CodexNoFinetuneConfig) -> None:
         raise ValueError("num_epochs must be >= 1")
     if not cfg.log_path:
         raise ValueError("log_path is required")
+    validate_focus_slices(cfg)
 
     object.__setattr__(cfg, "log_path", os.path.expanduser(cfg.log_path))
     os.makedirs(cfg.log_path, exist_ok=True)
````
</details>

