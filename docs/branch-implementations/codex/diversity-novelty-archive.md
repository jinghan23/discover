# codex/diversity-novelty-archive

## Summary

从 sampler archive 中选择 top/recent 候选摘要，提取函数名、kernel 名、markers、hash 等，作为 Novelty Archive 附加到 prompt 以避免近重复实现。

Note: 这个分支主要通过 prompt 增加 archive 摘要，不改变 sampler 排序算法。

## Branch State

- Worktree: `/opt/tiger/discover-novelty-archive`
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

- `novelty_archive_count`
- `out`
- `seen`
- `decorators`
- `candidates`
- `selected`
- `seen_shas`

### Constants

- `NOVELTY_ARCHIVE_MAX_COUNT`
- `NOVELTY_ARCHIVE_MARKERS`

### Classes

- None

### Functions

- `validate_novelty_archive_count`
- `_sanitize_archive_text`
- `_state_code`
- `_code_sha`
- `_finite_float`
- `_int_or_default`
- `_short_id`
- `_function_names`
- `_triton_kernel_names`
- `_docstring_summary`
- `_markers`
- `_novelty_archive_entry`
- `_select_novelty_archive_entries`
- `add_entries`
- `_format_archive_list`
- `_format_novelty_archive_prompt`
- `_novelty_archive_metrics`
- `_append_novelty_archive_prompt`

## Diff Summary

- Worktree tracked shortstat: `1 file changed, 294 insertions(+), 1 deletion(-)`
- Untracked files: `2`

### Worktree Status

````text
 M ttt_discover/rl/codex_no_finetune.py
?? repro/gpu_mode/
?? repro/run_discovery.py
````
### Tracked Worktree Files

````text
M	ttt_discover/rl/codex_no_finetune.py
````
### Untracked Files

- `repro/gpu_mode/run_0608_novelty_archive.sh (770 bytes)`
- `repro/run_discovery.py (16195 bytes)`

## Untracked File Snapshots

These files are not part of `git diff` yet, but they are part of the worktree implementation state recorded here.

### `repro/gpu_mode/run_0608_novelty_archive.sh`

````bash
#!/usr/bin/env bash
set -euo pipefail

# TTT Discover novelty-archive run: non-auto, read-only Codex samples.
python repro/run_discovery.py \
    --task trimul \
    --runner codex_no_finetune \
    --experiment-name trimul_0608_novelty_archive_gpu2 \
    --log-root codex_runs/trimul_exec_workspaces \
    --gpu 2 \
    --cuda-device-order PCI_BUS_ID \
    --torch-cuda-arch-list 8.0 \
    --num-epochs 50 \
    --group-size 4 \
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
    --codex-novelty-archive-count 24
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


def _novelty_archive_count(value: str) -> int:
    try:
        count = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if not 0 <= count <= 32:
        raise argparse.ArgumentTypeError("must be between 0 and 32")
    return count


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
        "--codex-novelty-archive-count",
        type=_novelty_archive_count,
        default=0,
        help=(
            "Append metadata for this many previous accepted/surviving sampler "
            "pool candidates to the Codex prompt. 0 disables it."
        ),
    )
    parser.add_argument(
        "--codex-autonomous",
        action="store_true",
        help="AutoEvolve mode: give Codex a writable workspace for deep dives.",
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
        print(f"codex_novelty_archive_count={args.codex_novelty_archive_count}")
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
            novelty_archive_count=args.codex_novelty_archive_count,
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
 ttt_discover/rl/codex_no_finetune.py | 295 ++++++++++++++++++++++++++++++++++-
 1 file changed, 294 insertions(+), 1 deletion(-)
````
## Raw Diff

<details>
<summary>Tracked worktree patch</summary>

````diff
diff --git a/ttt_discover/rl/codex_no_finetune.py b/ttt_discover/rl/codex_no_finetune.py
index 18a0e9e..7b45263 100644
--- a/ttt_discover/rl/codex_no_finetune.py
+++ b/ttt_discover/rl/codex_no_finetune.py
@@ -46,8 +46,10 @@ from __future__ import annotations
 
 import asyncio
 import glob
+import hashlib
 import json
 import logging
+import math
 import os
 import re
 import shlex
@@ -80,6 +82,20 @@ from ttt_discover.codex_utils.sampler import (
 logger = logging.getLogger(__name__)
 
 SAFE_GRADE_EXECUTOR = ThreadPoolExecutor(max_workers=4096)
+NOVELTY_ARCHIVE_MAX_COUNT = 32
+NOVELTY_ARCHIVE_MARKERS = (
+    "triton",
+    "torch",
+    "load_inline",
+    "cublas",
+    "cupy",
+    "torch.compile",
+    "@triton.autotune",
+    "@triton.jit",
+    "custom_kernel",
+    "bmm",
+    "einsum",
+)
 
 
 @dataclass
@@ -137,6 +153,7 @@ class CodexNoFinetuneConfig:
     initial_program_paths: tuple[str, ...] = ()
     initial_pool_paths: tuple[str, ...] = ()
     topk_children: int = 16
+    novelty_archive_count: int = 0
 
     # False is TTT Discover: non-auto, many short samples. True is AutoEvolve:
     # one Codex deep-dive workspace per sample with file edits and shell access.
@@ -338,6 +355,272 @@ def _latest_sampler_step(log_path: str) -> int:
     return latest
 
 
+def validate_novelty_archive_count(count: int) -> None:
+    if not 0 <= int(count) <= NOVELTY_ARCHIVE_MAX_COUNT:
+        raise ValueError(
+            "novelty_archive_count must be between 0 and "
+            f"{NOVELTY_ARCHIVE_MAX_COUNT}, got {count}"
+        )
+
+
+def _sanitize_archive_text(text: Any, *, max_chars: int = 120) -> str:
+    clean = str(text)
+    clean = clean.replace("```", "")
+    clean = clean.replace('"""', "")
+    clean = clean.replace("'''", "")
+    clean = clean.replace("`", "")
+    clean = re.sub(r"\s+", " ", clean).strip()
+    if len(clean) > max_chars:
+        clean = clean[: max(0, max_chars - 3)].rstrip() + "..."
+    return clean
+
+
+def _state_code(state: Any) -> str:
+    code = getattr(state, "code", "")
+    if code is None:
+        return ""
+    if isinstance(code, str):
+        return code
+    return str(code)
+
+
+def _code_sha(code: str) -> str:
+    return hashlib.sha256(code.encode("utf-8", errors="replace")).hexdigest()
+
+
+def _finite_float(value: Any) -> float | None:
+    try:
+        out = float(value)
+    except (TypeError, ValueError):
+        return None
+    if not math.isfinite(out):
+        return None
+    return out
+
+
+def _int_or_default(value: Any, default: int = -1) -> int:
+    try:
+        return int(value)
+    except (TypeError, ValueError):
+        return default
+
+
+def _short_id(state: Any, code_sha: str) -> str:
+    raw_id = getattr(state, "id", None)
+    if raw_id is None:
+        return code_sha[:12]
+    return _sanitize_archive_text(str(raw_id), max_chars=16)
+
+
+def _function_names(code: str) -> list[str]:
+    names = re.findall(r"(?m)^\s*(?:async\s+)?def\s+([A-Za-z_]\w*)\s*\(", code)
+    out: list[str] = []
+    seen: set[str] = set()
+    for name in names:
+        if name in seen:
+            continue
+        out.append(_sanitize_archive_text(name, max_chars=48))
+        seen.add(name)
+        if len(out) >= 12:
+            break
+    return out
+
+
+def _triton_kernel_names(code: str) -> list[str]:
+    out: list[str] = []
+    seen: set[str] = set()
+    decorators: list[str] = []
+    for line in code.splitlines():
+        stripped = line.strip()
+        if stripped.startswith("@"):
+            decorators.append(stripped)
+            decorators = decorators[-4:]
+            continue
+        match = re.match(r"(?:async\s+)?def\s+([A-Za-z_]\w*)\s*\(", stripped)
+        if match:
+            nearby = "\n".join(decorators[-4:])
+            if "@triton.jit" in nearby or "@triton.autotune" in nearby:
+                name = match.group(1)
+                if name not in seen:
+                    out.append(_sanitize_archive_text(name, max_chars=48))
+                    seen.add(name)
+            decorators = []
+            continue
+        if stripped and not stripped.startswith("#"):
+            decorators = []
+    return out
+
+
+def _docstring_summary(code: str) -> str:
+    module_doc = re.match(
+        r"\A\s*(?P<quote>\"\"\"|''')(?P<body>.*?)(?P=quote)",
+        code,
+        flags=re.DOTALL,
+    )
+    if module_doc is not None:
+        for line in module_doc.group("body").splitlines():
+            summary = _sanitize_archive_text(line, max_chars=120)
+            if summary:
+                return summary
+
+    for line in code.splitlines():
+        stripped = line.strip()
+        if not stripped:
+            continue
+        if stripped.startswith("#"):
+            summary = _sanitize_archive_text(stripped.lstrip("#").strip(), max_chars=120)
+            if summary:
+                return summary
+            continue
+        break
+    return ""
+
+
+def _markers(code: str) -> list[str]:
+    lowered = code.lower()
+    return [marker for marker in NOVELTY_ARCHIVE_MARKERS if marker.lower() in lowered]
+
+
+def _novelty_archive_entry(state: Any) -> dict[str, Any] | None:
+    code = _state_code(state)
+    if not code.strip():
+        return None
+
+    value = _finite_float(getattr(state, "value", None))
+    if value is None:
+        return None
+
+    code_sha = _code_sha(code)
+    return {
+        "id": _short_id(state, code_sha),
+        "value": value,
+        "timestep": _int_or_default(getattr(state, "timestep", -1)),
+        "code_sha": code_sha,
+        "code_len": len(code),
+        "function_names": _function_names(code),
+        "triton_kernel_names": _triton_kernel_names(code),
+        "docstring_summary": _docstring_summary(code),
+        "markers": _markers(code),
+    }
+
+
+def _select_novelty_archive_entries(
+    sampler: StateSampler,
+    parent_state: Any,
+    count: int,
+) -> list[dict[str, Any]]:
+    validate_novelty_archive_count(count)
+    if count <= 0:
+        return []
+
+    parent_id = getattr(parent_state, "id", None)
+    candidates: list[dict[str, Any]] = []
+    for state in list(getattr(sampler, "_states", []) or []):
+        if parent_id is not None and getattr(state, "id", None) == parent_id:
+            continue
+        entry = _novelty_archive_entry(state)
+        if entry is not None:
+            candidates.append(entry)
+
+    top_sorted = sorted(
+        candidates,
+        key=lambda item: (item["value"], item["timestep"]),
+        reverse=True,
+    )
+    recent_sorted = sorted(
+        candidates,
+        key=lambda item: (item["timestep"], item["value"]),
+        reverse=True,
+    )
+
+    selected: list[dict[str, Any]] = []
+    seen_shas: set[str] = set()
+
+    def add_entries(entries: list[dict[str, Any]], limit: int) -> None:
+        for entry in entries:
+            if len(selected) >= limit:
+                return
+            code_sha = entry["code_sha"]
+            if code_sha in seen_shas:
+                continue
+            selected.append(entry)
+            seen_shas.add(code_sha)
+
+    top_quota = min(count, (count + 1) // 2)
+    add_entries(top_sorted, top_quota)
+    add_entries(recent_sorted, count)
+    add_entries(top_sorted, count)
+    return selected[:count]
+
+
+def _format_archive_list(values: list[str]) -> str:
+    if not values:
+        return "-"
+    return ", ".join(_sanitize_archive_text(value, max_chars=48) for value in values)
+
+
+def _format_novelty_archive_prompt(entries: list[dict[str, Any]]) -> str:
+    if not entries:
+        return ""
+
+    lines = [
+        "--- Novelty Archive ---",
+        (
+            "The entries below summarize previous accepted/surviving candidates "
+            "from the sampler pool."
+        ),
+        (
+            "Use them only to avoid near-identical implementations or superficial "
+            "rewrites; do not optimize for novelty at the cost of correctness; "
+            "keep the original task rules; return one final valid program/code block."
+        ),
+        "",
+    ]
+    for idx, entry in enumerate(entries, start=1):
+        value = f"{entry['value']:.6g}"
+        lines.append(
+            f"{idx}. "
+            f"id={entry['id']}; "
+            f"value={value}; "
+            f"timestep={entry['timestep']}; "
+            f"code_sha={entry['code_sha'][:16]}; "
+            f"code_len={entry['code_len']}; "
+            f"function_names={_format_archive_list(entry['function_names'])}; "
+            f"triton_kernel_names={_format_archive_list(entry['triton_kernel_names'])}; "
+            f"docstring_summary={entry['docstring_summary'] or '-'}; "
+            f"markers={_format_archive_list(entry['markers'])}"
+        )
+    return "\n".join(lines)
+
+
+def _novelty_archive_metrics(
+    entries: list[dict[str, Any]],
+    block: str,
+) -> dict[str, Any]:
+    return {
+        "codex/novelty_archive_count": len(entries),
+        "codex/novelty_archive_chars": len(block),
+        "codex/novelty_archive_ids": [entry["id"] for entry in entries],
+        "codex/novelty_archive_code_shas": [
+            entry["code_sha"][:16] for entry in entries
+        ],
+    }
+
+
+def _append_novelty_archive_prompt(
+    prompt: str,
+    sampler: StateSampler,
+    parent_state: Any,
+    count: int,
+) -> tuple[str, dict[str, Any]]:
+    entries = _select_novelty_archive_entries(sampler, parent_state, count)
+    block = _format_novelty_archive_prompt(entries)
+    metrics = _novelty_archive_metrics(entries, block)
+    if not block:
+        return prompt, metrics
+    return f"{prompt.rstrip()}\n\n{block}\n", metrics
+
+
 def _make_env(cfg: CodexNoFinetuneConfig, state: Any, sampler: StateSampler) -> Any:
     env = object.__new__(cfg.env_type)
     env.config = cfg
@@ -640,9 +923,16 @@ async def _run_candidate(
     prompt = ""
     response = ""
     parsed_code = ""
+    archive_metrics = _novelty_archive_metrics([], "")
     try:
         env = _make_env(cfg, parent_state, sampler)
         prompt = env.get_question()
+        prompt, archive_metrics = _append_novelty_archive_prompt(
+            prompt,
+            sampler,
+            parent_state,
+            cfg.novelty_archive_count,
+        )
         completer = _make_completer(
             cfg,
             semaphore=semaphore,
@@ -675,6 +965,8 @@ async def _run_candidate(
         correct_format = _check_candidate_format(env, parsed_code)
         outs = await _safe_grade(cfg, env, parsed_code, correct_format)
         metrics = _build_metrics(env, outs, response, parsed_code, correct_format)
+        metrics["prompt"] = prompt
+        metrics.update(archive_metrics)
         metrics["codex/parsed_code_source"] = parsed_code_source
         if autonomous_submission_path is not None:
             metrics["codex/autonomous_submission_path"] = autonomous_submission_path
@@ -713,7 +1005,7 @@ async def _run_candidate(
             correctness=0.0,
             raw_score=None,
             msg=error_msg,
-            metrics={"error": error_msg},
+            metrics={"error": error_msg, **archive_metrics},
             next_state=None,
             error=error_msg,
         )
@@ -987,6 +1279,7 @@ async def main(cfg: CodexNoFinetuneConfig) -> None:
         raise ValueError("num_epochs must be >= 1")
     if not cfg.log_path:
         raise ValueError("log_path is required")
+    validate_novelty_archive_count(cfg.novelty_archive_count)
 
     object.__setattr__(cfg, "log_path", os.path.expanduser(cfg.log_path))
     os.makedirs(cfg.log_path, exist_ok=True)
````
</details>

