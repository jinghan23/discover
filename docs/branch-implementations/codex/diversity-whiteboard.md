# codex/diversity-whiteboard

## Summary

prompt/memory 层 whiteboard：在 log_path 下维护 codex_whiteboard.md，压缩保存历史候选结果摘要，并在后续 prompt 中注入。

Note: 这个分支主要增加跨 step 的 prompt memory/日志白板，不是父节点平衡 sampler。

## Branch State

- Worktree: `/opt/tiger/discover-whiteboard`
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

- `whiteboard_enabled`
- `whiteboard_max_chars`
- `whiteboard_update_every`
- `whiteboard_snapshot`

### Constants

- `WHITEBOARD_FILENAME`
- `WHITEBOARD_HEADER`
- `WHITEBOARD_COMPACTION_MARKER`
- `WHITEBOARD_PROMPT_REMINDER`

### Classes

- None

### Functions

- `_validate_whiteboard_config`
- `_whiteboard_path`
- `_compact_whiteboard_text`
- `_read_whiteboard_snapshot`
- `_with_whiteboard_prompt`
- `_sanitize_whiteboard_value`
- `_format_whiteboard_number`
- `_short_whiteboard_id`
- `_whiteboard_code_hash`
- `_whiteboard_error_category`
- `_format_whiteboard_fact`
- `_format_whiteboard_entry`
- `_write_text_atomic`
- `_append_whiteboard_facts`

## Diff Summary

- Worktree tracked shortstat: `1 file changed, 212 insertions(+), 1 deletion(-)`
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

- `repro/gpu_mode/run_0608_whiteboard.sh (827 bytes)`
- `repro/run_discovery.py (16626 bytes)`

## Untracked File Snapshots

These files are not part of `git diff` yet, but they are part of the worktree implementation state recorded here.

### `repro/gpu_mode/run_0608_whiteboard.sh`

````bash
#!/usr/bin/env bash
set -euo pipefail

# TTT Discover whiteboard run: non-auto, read-only Codex samples.
python repro/run_discovery.py \
    --task trimul \
    --runner codex_no_finetune \
    --experiment-name trimul_0608_whiteboard_gpu2 \
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
    --codex-whiteboard \
    --codex-whiteboard-max-chars 12000 \
    --codex-whiteboard-update-every 1
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
        "--codex-whiteboard",
        action="store_true",
        help="Enable deterministic fact-only shared whiteboard prompting.",
    )
    parser.add_argument(
        "--codex-whiteboard-max-chars",
        type=int,
        default=12000,
        help="Maximum characters stored/read from codex_whiteboard.md.",
    )
    parser.add_argument("--codex-whiteboard-update-every", type=int, default=1)

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


def _validate_whiteboard_args(args: argparse.Namespace) -> None:
    if args.codex_whiteboard_max_chars <= 0:
        raise ValueError("--codex-whiteboard-max-chars must be > 0")
    if args.codex_whiteboard_update_every <= 0:
        raise ValueError("--codex-whiteboard-update-every must be > 0")


def main() -> None:
    args = parse_args()
    _validate_whiteboard_args(args)
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
        print(f"codex_whiteboard={args.codex_whiteboard}")
        print(f"codex_whiteboard_max_chars={args.codex_whiteboard_max_chars}")
        print(f"codex_whiteboard_update_every={args.codex_whiteboard_update_every}")
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
            whiteboard_enabled=args.codex_whiteboard,
            whiteboard_max_chars=args.codex_whiteboard_max_chars,
            whiteboard_update_every=args.codex_whiteboard_update_every,
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
 ttt_discover/rl/codex_no_finetune.py | 213 ++++++++++++++++++++++++++++++++++-
 1 file changed, 212 insertions(+), 1 deletion(-)
````
## Raw Diff

<details>
<summary>Tracked worktree patch</summary>

````diff
diff --git a/ttt_discover/rl/codex_no_finetune.py b/ttt_discover/rl/codex_no_finetune.py
index 18a0e9e..a8da662 100644
--- a/ttt_discover/rl/codex_no_finetune.py
+++ b/ttt_discover/rl/codex_no_finetune.py
@@ -46,6 +46,7 @@ from __future__ import annotations
 
 import asyncio
 import glob
+import hashlib
 import json
 import logging
 import os
@@ -80,6 +81,18 @@ from ttt_discover.codex_utils.sampler import (
 logger = logging.getLogger(__name__)
 
 SAFE_GRADE_EXECUTOR = ThreadPoolExecutor(max_workers=4096)
+WHITEBOARD_FILENAME = "codex_whiteboard.md"
+WHITEBOARD_HEADER = (
+    "# Codex Shared Search Whiteboard\n\n"
+    "Deterministic fact-only observations from previous completed batches.\n"
+)
+WHITEBOARD_COMPACTION_MARKER = (
+    "\n... older whiteboard facts compacted deterministically ...\n"
+)
+WHITEBOARD_PROMPT_REMINDER = (
+    "Reminder: facts are previous-run observations only; keep original task rules; "
+    "return one final code block."
+)
 
 
 @dataclass
@@ -137,6 +150,9 @@ class CodexNoFinetuneConfig:
     initial_program_paths: tuple[str, ...] = ()
     initial_pool_paths: tuple[str, ...] = ()
     topk_children: int = 16
+    whiteboard_enabled: bool = False
+    whiteboard_max_chars: int = 12000
+    whiteboard_update_every: int = 1
 
     # False is TTT Discover: non-auto, many short samples. True is AutoEvolve:
     # one Codex deep-dive workspace per sample with file edits and shell access.
@@ -237,6 +253,180 @@ def all_same(xs: list[Any]) -> bool:
     return bool(xs) and all(x == xs[0] for x in xs)
 
 
+def _validate_whiteboard_config(cfg: CodexNoFinetuneConfig) -> None:
+    if cfg.whiteboard_max_chars <= 0:
+        raise ValueError("whiteboard_max_chars must be > 0")
+    if cfg.whiteboard_update_every <= 0:
+        raise ValueError("whiteboard_update_every must be > 0")
+
+
+def _whiteboard_path(cfg: CodexNoFinetuneConfig) -> Path:
+    return Path(cfg.log_path) / WHITEBOARD_FILENAME
+
+
+def _compact_whiteboard_text(text: str, max_chars: int) -> str:
+    if max_chars <= 0:
+        raise ValueError("max_chars must be > 0")
+    if len(text) <= max_chars:
+        return text
+
+    prefix = WHITEBOARD_HEADER
+    tail_budget = max_chars - len(prefix) - len(WHITEBOARD_COMPACTION_MARKER)
+    if tail_budget <= 0:
+        return text[-max_chars:]
+
+    tail = text[-tail_budget:]
+    first_newline = tail.find("\n")
+    if 0 < first_newline < len(tail) - 1:
+        tail = tail[first_newline + 1 :]
+
+    compacted = prefix + WHITEBOARD_COMPACTION_MARKER + tail.lstrip("\n")
+    if len(compacted) > max_chars:
+        compacted = compacted[-max_chars:]
+    return compacted
+
+
+def _read_whiteboard_snapshot(cfg: CodexNoFinetuneConfig) -> str:
+    if not cfg.whiteboard_enabled:
+        return ""
+
+    path = _whiteboard_path(cfg)
+    try:
+        text = path.read_text(encoding="utf-8")
+    except FileNotFoundError:
+        return ""
+    except OSError as exc:
+        logger.warning("Could not read Codex whiteboard %s: %s", path, exc)
+        return ""
+
+    return _compact_whiteboard_text(text, cfg.whiteboard_max_chars).strip()
+
+
+def _with_whiteboard_prompt(prompt: str, snapshot: str) -> str:
+    return (
+        f"{prompt.rstrip()}\n\n"
+        "--- Shared Search Whiteboard ---\n"
+        f"{snapshot.strip()}\n\n"
+        f"{WHITEBOARD_PROMPT_REMINDER}"
+    )
+
+
+def _sanitize_whiteboard_value(value: Any, max_chars: int = 160) -> str:
+    text = str(value).replace("```", "'''").replace("`", "'")
+    text = re.sub(r"\s+", " ", text).strip()
+    if len(text) <= max_chars:
+        return text
+    if max_chars <= 3:
+        return text[-max_chars:]
+    return "..." + text[-(max_chars - 3) :]
+
+
+def _format_whiteboard_number(value: Any) -> str:
+    if value is None:
+        return "None"
+    if isinstance(value, bool):
+        return str(value)
+    if isinstance(value, (int, float, np.integer, np.floating)):
+        return f"{float(value):.6g}"
+    return _sanitize_whiteboard_value(value, max_chars=40)
+
+
+def _short_whiteboard_id(value: Any) -> str:
+    if value is None:
+        return "None"
+    return _sanitize_whiteboard_value(value, max_chars=12)
+
+
+def _whiteboard_code_hash(parsed_code: str) -> str:
+    if not parsed_code:
+        return "none"
+    return hashlib.sha256(parsed_code.encode("utf-8", errors="replace")).hexdigest()[:12]
+
+
+def _whiteboard_error_category(text: str | None) -> str:
+    if not text:
+        return "none"
+    first_line = _sanitize_whiteboard_value(text.splitlines()[0], max_chars=80)
+    return first_line.split(":", 1)[0] or "error"
+
+
+def _format_whiteboard_fact(step: int, result: CandidateResult) -> str:
+    parent_state = result.parent_state
+    msg_source = result.error or result.sampler_error or result.msg
+    fields = [
+        f"step={step}",
+        f"group={result.group_idx}",
+        f"sample={result.sample_idx}",
+        f"parent={_short_whiteboard_id(getattr(parent_state, 'id', None))}",
+        f"parent_value={_format_whiteboard_number(getattr(parent_state, 'value', None))}",
+        f"reward={_format_whiteboard_number(result.reward)}",
+        f"correctness={_format_whiteboard_number(result.correctness)}",
+        f"raw_score={_format_whiteboard_number(result.raw_score)}",
+        f"pool_status={_sanitize_whiteboard_value(result.pool_status or 'unknown', 40)}",
+        f"code_sha={_whiteboard_code_hash(result.parsed_code)}",
+        f"code_len={len(result.parsed_code)}",
+        f"response_len={len(result.response)}",
+    ]
+    if msg_source:
+        fields.append(f"msg_kind={_whiteboard_error_category(msg_source)}")
+        fields.append(f"msg_tail={_sanitize_whiteboard_value(msg_source, 160)!r}")
+    return "- " + " ".join(fields)
+
+
+def _format_whiteboard_entry(step: int, results: list[CandidateResult]) -> str:
+    lines = [f"## step={step}"]
+    for result in sorted(results, key=lambda item: (item.group_idx, item.sample_idx)):
+        lines.append(_format_whiteboard_fact(step, result))
+    return "\n".join(lines) + "\n"
+
+
+def _write_text_atomic(path: Path, text: str) -> bool:
+    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
+    try:
+        path.parent.mkdir(parents=True, exist_ok=True)
+        tmp_path.write_text(text, encoding="utf-8")
+        os.replace(tmp_path, path)
+        return True
+    except OSError as exc:
+        logger.warning("Could not write Codex whiteboard %s: %s", path, exc)
+        try:
+            tmp_path.unlink()
+        except OSError:
+            pass
+        return False
+
+
+def _append_whiteboard_facts(
+    cfg: CodexNoFinetuneConfig,
+    step: int,
+    results: list[CandidateResult],
+) -> int | None:
+    if not cfg.whiteboard_enabled:
+        return None
+    if (step + 1) % cfg.whiteboard_update_every != 0:
+        return None
+
+    path = _whiteboard_path(cfg)
+    try:
+        text = path.read_text(encoding="utf-8")
+    except FileNotFoundError:
+        text = WHITEBOARD_HEADER
+    except OSError as exc:
+        logger.warning("Could not read Codex whiteboard before update %s: %s", path, exc)
+        return None
+
+    if not text.strip():
+        text = WHITEBOARD_HEADER
+    elif not text.startswith(WHITEBOARD_HEADER):
+        text = WHITEBOARD_HEADER + "\n" + text.strip() + "\n"
+
+    text = f"{text.rstrip()}\n\n{_format_whiteboard_entry(step, results)}"
+    text = _compact_whiteboard_text(text, cfg.whiteboard_max_chars)
+    if not _write_text_atomic(path, text):
+        return None
+    return len(text)
+
+
 def append_agent_outputs(log_path: str, step: int, results: list[CandidateResult]) -> None:
     os.makedirs(log_path, exist_ok=True)
     output_path = os.path.join(log_path, "agent_outputs.jsonl")
@@ -636,6 +826,7 @@ async def _run_candidate(
     sample_idx: int,
     step_idx: int,
     semaphore: asyncio.Semaphore | None,
+    whiteboard_snapshot: str = "",
 ) -> CandidateResult:
     prompt = ""
     response = ""
@@ -643,6 +834,8 @@ async def _run_candidate(
     try:
         env = _make_env(cfg, parent_state, sampler)
         prompt = env.get_question()
+        if cfg.whiteboard_enabled and whiteboard_snapshot:
+            prompt = _with_whiteboard_prompt(prompt, whiteboard_snapshot)
         completer = _make_completer(
             cfg,
             semaphore=semaphore,
@@ -676,6 +869,7 @@ async def _run_candidate(
         outs = await _safe_grade(cfg, env, parsed_code, correct_format)
         metrics = _build_metrics(env, outs, response, parsed_code, correct_format)
         metrics["codex/parsed_code_source"] = parsed_code_source
+        metrics["codex/whiteboard_snapshot_chars"] = len(whiteboard_snapshot)
         if autonomous_submission_path is not None:
             metrics["codex/autonomous_submission_path"] = autonomous_submission_path
         next_state = _maybe_create_next_state(env, step_idx, parsed_code, outs)
@@ -713,7 +907,10 @@ async def _run_candidate(
             correctness=0.0,
             raw_score=None,
             msg=error_msg,
-            metrics={"error": error_msg},
+            metrics={
+                "error": error_msg,
+                "codex/whiteboard_snapshot_chars": len(whiteboard_snapshot),
+            },
             next_state=None,
             error=error_msg,
         )
@@ -820,6 +1017,7 @@ async def sample_batch(
     i_batch: int,
 ) -> tuple[list[CandidateResult], dict[str, Any], list[CandidateResult]]:
     metrics: dict[str, Any] = {}
+    whiteboard_snapshot = _read_whiteboard_snapshot(cfg)
     parent_states = sampler.sample_states(cfg.groups_per_batch)
     semaphore = (
         asyncio.Semaphore(cfg.max_concurrent_requests)
@@ -840,6 +1038,7 @@ async def sample_batch(
                         sample_idx=sample_idx,
                         step_idx=i_batch,
                         semaphore=semaphore,
+                        whiteboard_snapshot=whiteboard_snapshot,
                     ),
                     name=f"codex_sample_{group_idx}_{sample_idx}",
                 )
@@ -847,6 +1046,12 @@ async def sample_batch(
 
     results = await asyncio.gather(*tasks)
     _update_sampler_from_results(sampler, results)
+    whiteboard_chars = len(whiteboard_snapshot)
+    whiteboard_updated = False
+    updated_whiteboard_chars = _append_whiteboard_facts(cfg, i_batch, results)
+    if updated_whiteboard_chars is not None:
+        whiteboard_chars = updated_whiteboard_chars
+        whiteboard_updated = True
 
     results_by_group: dict[int, list[CandidateResult]] = {}
     for result in results:
@@ -871,6 +1076,11 @@ async def sample_batch(
     metrics["codex/parent_states"] = len(parent_states)
     metrics["codex/dropped_constant_groups"] = dropped_constant_groups
     metrics.update(_result_metrics(results, kept_results))
+    metrics["codex/whiteboard_enabled"] = bool(cfg.whiteboard_enabled)
+    metrics["codex/whiteboard_chars"] = whiteboard_chars if cfg.whiteboard_enabled else 0
+    metrics["codex/whiteboard_updated"] = whiteboard_updated
+    if whiteboard_updated:
+        metrics["codex/whiteboard_version_step"] = i_batch
     return kept_results, metrics, results
 
 
@@ -987,6 +1197,7 @@ async def main(cfg: CodexNoFinetuneConfig) -> None:
         raise ValueError("num_epochs must be >= 1")
     if not cfg.log_path:
         raise ValueError("log_path is required")
+    _validate_whiteboard_config(cfg)
 
     object.__setattr__(cfg, "log_path", os.path.expanduser(cfg.log_path))
     os.makedirs(cfg.log_path, exist_ok=True)
````
</details>

