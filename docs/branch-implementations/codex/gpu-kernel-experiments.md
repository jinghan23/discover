# codex/gpu-kernel-experiments

## Summary

GPU kernel 实验支持分支：调整 GPU Mode 环境、Codex completer 调用环境/日志/超时处理，并扩展 no-finetune 路径以跑 kernel 实验。

## Branch State

- Worktree: `/opt/tiger/discover-gpu-kernel-experiments`
- HEAD: `2234282`
- Base used for comparison: `2234282`
- Commits ahead of base: `0`
- Commits behind base: `0`
- Group: `other`
- Implementation location: `uncommitted worktree diff`

## Implemented Behavior

- The summary above is based on reading the changed Python code, added config fields, and sampler/prompt hooks in this branch.

## Added Markers

### Config fields

- `config_overrides`
- `env`
- `skip_git_repo_check`
- `cli_timeout_error`

### Constants

- `PUBLIC_TESTS`
- `PUBLIC_BENCHMARKS`

### Classes

- `CodexCliTimeoutError`

### Functions

- `make_case`
- `compute_score_us`
- `__init__`
- `_build_process_env`
- `_log_env`
- `_build_command`
- `_build_inner_dangerous_command`
- `_prepare_isolated_codex_home`
- `_build_isolated_command`

## Diff Summary

- Worktree tracked shortstat: `3 files changed, 315 insertions(+), 71 deletions(-)`
- Untracked files: `2`

### Worktree Status

````text
 M examples/gpu_mode/env.py
 M ttt_discover/codex_utils/completers.py
 M ttt_discover/rl/codex_no_finetune.py
?? repro/gpu_mode/
?? repro/run_discovery.py
````
### Tracked Worktree Files

````text
M	examples/gpu_mode/env.py
M	ttt_discover/codex_utils/completers.py
M	ttt_discover/rl/codex_no_finetune.py
````
### Untracked Files

- `repro/gpu_mode/run_0608.sh (1921 bytes)`
- `repro/run_discovery.py (15409 bytes)`

## Untracked File Snapshots

These files are not part of `git diff` yet, but they are part of the worktree implementation state recorded here.

### `repro/gpu_mode/run_0608.sh`

````bash
#!/usr/bin/env bash
set -euo pipefail

# TTT Discover: non-auto, many short Codex samples driven by the sampler/evaluator.
# groups-per-batch = parent states sampled per outer round.
# group-size = Codex samples per parent; product is samples/evals per round.
python repro/run_discovery.py \
    --task trimul \
    --runner codex_no_finetune \
    --experiment-name trimul_0608_ttt_discover_gpu2 \
    --log-root codex_runs/trimul_exec_workspaces \
    --gpu 2 \
    --cuda-device-order PCI_BUS_ID \
    --torch-cuda-arch-list 8.0 \
    --num-epochs 50 \
    --group-size 8 \
    --groups-per-batch 1 \
    --num-cpus-per-task 1 \
    --eval-timeout 1200 \
    --wandb-project "" \
    --codex-backend cli \
    --codex-model-name gpt-5.5 \
    --codex-cli-command codex \
    --codex-cli-sandbox read-only \
    --codex-cli-timeout 600 \
    --codex-max-concurrent-requests 4

# Codex AutoEvolve: autonomous deep dive, one writable workspace per sample.
# For GPUMode, danger-full-access is wrapped by the completer in an external
# unshare mount namespace: Codex only sees the sample workspace, while CUDA
# remains visible.
# groups-per-batch = parent states sampled per outer round.
# group-size = Codex samples per parent; product is samples/evals per round.
python repro/run_discovery.py \
    --task trimul \
    --runner codex_no_finetune \
    --experiment-name trimul_0608_codex_autoevolve_gpu2 \
    --log-root codex_runs/trimul_exec_workspaces \
    --gpu 3 \
    --cuda-device-order PCI_BUS_ID \
    --torch-cuda-arch-list 8.0 \
    --num-epochs 8 \
    --group-size 2 \
    --groups-per-batch 1 \
    --num-cpus-per-task 1 \
    --eval-timeout 1200 \
    --wandb-project "" \
    --codex-backend cli \
    --codex-model-name gpt-5.5 \
    --codex-cli-command codex \
    --codex-cli-sandbox danger-full-access \
    --codex-cli-timeout 7200 \
    --codex-max-concurrent-requests 2 \
    --codex-autonomous
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
        remove_constant_reward_groups=remove_constant_reward_groups,
    )
    discover(config)


if __name__ == "__main__":
    main()
````

## Diff Stat

### Worktree Tracked Diff Stat

````text
 examples/gpu_mode/env.py               | 145 +++++++++++++++------------
 ttt_discover/codex_utils/completers.py |  67 ++++++++++++-
 ttt_discover/rl/codex_no_finetune.py   | 174 ++++++++++++++++++++++++++++++++-
 3 files changed, 315 insertions(+), 71 deletions(-)
````
## Raw Diff

<details>
<summary>Tracked worktree patch</summary>

````diff
diff --git a/examples/gpu_mode/env.py b/examples/gpu_mode/env.py
index a37953f..e05e8fc 100644
--- a/examples/gpu_mode/env.py
+++ b/examples/gpu_mode/env.py
@@ -238,72 +238,77 @@ class GpuModeRewardEvaluator(BaseRewardEvaluator):
 
 
 def _autonomous_eval_script(problem_type: str) -> str:
-    task_yaml = _task_yaml(problem_type)
     return f"""from pathlib import Path
 import math
 import os
-import sys
-import tempfile
 
-ROOT = Path({str(REPO_ROOT)!r})
-sys.path.insert(0, str(ROOT / "examples/gpu_mode/lib"))
+import torch
 
-from libkernelbot.consts import RankCriterion, SubmissionMode
-from libkernelbot.run_eval import run_config
-from libkernelbot.task import build_task_config, make_task_definition
+from eval import Stats, TestCase, _run_single_benchmark, _run_single_test
 
+PUBLIC_TESTS = [
+    {{"seqlen": 32, "bs": 1, "dim": 128, "hiddendim": 128, "seed": 101, "nomask": True, "distribution": "normal"}},
+    {{"seqlen": 32, "bs": 1, "dim": 128, "hiddendim": 128, "seed": 103, "nomask": False, "distribution": "normal"}},
+    {{"seqlen": 64, "bs": 1, "dim": 384, "hiddendim": 128, "seed": 107, "nomask": False, "distribution": "normal"}},
+    {{"seqlen": 64, "bs": 1, "dim": 128, "hiddendim": 128, "seed": 109, "nomask": True, "distribution": "cauchy"}},
+]
 
-def compute_score_us(result, task):
-    run = result.runs["leaderboard"].run
-    n = int(run.result["benchmark-count"])
-    means_ns = [float(run.result[f"benchmark.{{i}}.mean"]) for i in range(n)]
-    if task.ranking_by == RankCriterion.LAST:
-        score_ns = means_ns[-1]
-    elif task.ranking_by == RankCriterion.MEAN:
-        score_ns = sum(means_ns) / len(means_ns)
-    elif task.ranking_by == RankCriterion.GEOM:
-        score_ns = math.exp(sum(math.log(x) for x in means_ns) / len(means_ns))
-    else:
-        raise ValueError(f"Unsupported ranking_by: {{task.ranking_by}}")
-    return score_ns / 1000.0
+PUBLIC_BENCHMARKS = [
+    {{"seqlen": 256, "bs": 2, "dim": 128, "hiddendim": 128, "seed": 1009, "nomask": True, "distribution": "normal"}},
+    {{"seqlen": 768, "bs": 1, "dim": 128, "hiddendim": 128, "seed": 1013, "nomask": True, "distribution": "cauchy"}},
+    {{"seqlen": 256, "bs": 2, "dim": 384, "hiddendim": 128, "seed": 1019, "nomask": False, "distribution": "normal"}},
+    {{"seqlen": 512, "bs": 1, "dim": 128, "hiddendim": 128, "seed": 1021, "nomask": True, "distribution": "normal"}},
+    {{"seqlen": 1024, "bs": 1, "dim": 128, "hiddendim": 128, "seed": 1031, "nomask": True, "distribution": "cauchy"}},
+    {{"seqlen": 768, "bs": 1, "dim": 384, "hiddendim": 128, "seed": 1033, "nomask": False, "distribution": "normal"}},
+    {{"seqlen": 1024, "bs": 1, "dim": 384, "hiddendim": 128, "seed": 1039, "nomask": True, "distribution": "normal"}},
+]
+
+
+def make_case(args):
+    spec = "; ".join(f"{{k}}: {{v}}" for k, v in args.items())
+    return TestCase(args=dict(args), spec=spec)
+
+
+def compute_score_us(means_ns):
+    return math.exp(sum(math.log(x) for x in means_ns) / len(means_ns)) / 1000.0
 
 
 def main():
-    task = make_task_definition(Path({str(task_yaml)!r})).task
-    code = Path("submission.py").read_text()
-    config = build_task_config(
-        task=task,
-        submission_content=code,
-        arch=None,
-        mode=SubmissionMode.LEADERBOARD,
-    )
-    workspace = Path.cwd()
-    eval_root = workspace / "eval_tmp"
-    eval_root.mkdir(exist_ok=True)
-    with tempfile.TemporaryDirectory(prefix="run_", dir=eval_root) as tmp_dir:
-        old_cwd = os.getcwd()
-        os.chdir(tmp_dir)
-        try:
-            result = run_config(config)
-        finally:
-            os.chdir(old_cwd)
-    test = result.runs.get("test")
-    if test is None or test.run is None or not test.run.passed:
-        print("TEST FAILED")
-        if test is not None and test.run is not None:
-            print(test.run.stdout)
-            print(test.run.stderr)
-            print(test.run.result)
-        raise SystemExit(1)
-    leaderboard = result.runs.get("leaderboard")
-    if leaderboard is None or leaderboard.run is None or not leaderboard.run.passed:
-        print("LEADERBOARD FAILED")
-        if leaderboard is not None and leaderboard.run is not None:
-            print(leaderboard.run.stdout)
-            print(leaderboard.run.stderr)
-            print(leaderboard.run.result)
-        raise SystemExit(1)
-    print(f"score_us {{compute_score_us(result, task):.6f}}")
+    print(f"CUDA_VISIBLE_DEVICES {{os.environ.get('CUDA_VISIBLE_DEVICES')}}")
+    print(f"TORCH_CUDA_ARCH_LIST {{os.environ.get('TORCH_CUDA_ARCH_LIST')}}")
+    print(f"torch_cuda_available {{torch.cuda.is_available()}}")
+    print(f"torch_cuda_device_count {{torch.cuda.device_count()}}")
+    if not torch.cuda.is_available():
+        raise SystemExit("CUDA is not available in the Codex self-eval process")
+    print(f"torch_cuda_device {{torch.cuda.get_device_name(0)}}")
+    if not Path("submission.py").exists():
+        raise SystemExit("submission.py is missing")
+
+    for idx, args in enumerate(PUBLIC_TESTS):
+        good, message = _run_single_test(make_case(args))
+        print(f"public_test.{{idx}} {{'pass' if good else 'fail'}} {{args}}")
+        if not good:
+            print(message)
+            raise SystemExit(1)
+
+    means_ns = []
+    for idx, args in enumerate(PUBLIC_BENCHMARKS):
+        result = _run_single_benchmark(
+            make_case(args),
+            recheck=True,
+            max_repeats=20,
+            max_time_ns=3e9,
+        )
+        if not isinstance(result, Stats):
+            print(f"public_benchmark.{{idx}} fail {{args}}")
+            print(result)
+            raise SystemExit(1)
+        means_ns.append(float(result.mean))
+        print(
+            f"public_benchmark.{{idx}} mean_us {{result.mean / 1000.0:.6f}} "
+            f"runs {{result.runs}} {{args}}"
+        )
+    print(f"score_us {{compute_score_us(means_ns):.6f}}")
 
 
 if __name__ == "__main__":
@@ -367,7 +372,7 @@ class GpuModeEnv(Environment):
     ) -> str:
         del eval_timeout, num_cpus_per_task
         task_dir = _task_dir(self.problem_type)
-        for name in ("task.py", "utils.py", "reference.py", "eval.py", "task.yml"):
+        for name in ("task.py", "utils.py", "reference.py", "eval.py"):
             shutil.copy2(task_dir / name, workspace / name)
         readme = task_dir / "README.md"
         if readme.exists():
@@ -381,7 +386,15 @@ class GpuModeEnv(Environment):
             encoding="utf-8",
         )
 
-        evaluator_cmd = f"cd {shlex.quote(str(workspace))} && python eval_candidate.py"
+        env_parts = []
+        for key in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "TORCH_CUDA_ARCH_LIST"):
+            value = os.environ.get(key)
+            if value:
+                env_parts.append(f"{key}={shlex.quote(value)}")
+        env_prefix = (" ".join(env_parts) + " ") if env_parts else ""
+        evaluator_cmd = (
+            f"cd {shlex.quote(str(workspace))} && {env_prefix}python eval_candidate.py"
+        )
         return f"""{prompt}
 
 --- Autonomous GPUMode Search Mode ---
@@ -391,19 +404,25 @@ You may inspect files and run shell commands, but keep all edits inside this wor
 Editable candidate:
 {workspace / "submission.py"}
 
-The copied task files (`task.py`, `utils.py`, `reference.py`, `eval.py`, `task.yml`)
-are for inspection and local testing. The evaluator below reloads trusted task files
-from the repository and runs in `eval_tmp/`, so editing copied task files will not
-change the official score or delete your candidate.
+The copied task files (`task.py`, `utils.py`, `reference.py`, `eval.py`) are for
+inspection and local testing. `task.yml`, official seeds, historical runs, and
+top solutions are intentionally not present in this isolated Codex workspace.
+The evaluator below uses public smoke configs with synthetic local seeds; editing
+copied task/eval files will not change the official score or delete your candidate.
 
 Run this evaluator after each revision:
 {evaluator_cmd}
 
+Do not edit `eval_candidate.py` or the copied task/eval files to improve a score.
+Only `submission.py` is a valid candidate artifact, and the outer runner will
+re-score it with trusted repository files.
+
 When done, put the best implementation in:
 {workspace / "submission.py"}
 
 The outer discovery runner will score the final contents of that `submission.py`
-before considering any code block in your final response.
+with trusted repository files outside the Codex workspace before considering any
+code block in your final response.
 """
 
     def get_question(self) -> str:
diff --git a/ttt_discover/codex_utils/completers.py b/ttt_discover/codex_utils/completers.py
index 3e9b014..7d510f4 100644
--- a/ttt_discover/codex_utils/completers.py
+++ b/ttt_discover/codex_utils/completers.py
@@ -31,6 +31,27 @@ class TextCompleter:
         raise NotImplementedError
 
 
+class CodexCliTimeoutError(RuntimeError):
+    """Raised when `codex exec` times out after writing logs/workspace files."""
+
+    def __init__(
+        self,
+        *,
+        timeout: float | None,
+        call_dir: Path | None,
+        stdout: str,
+        stderr: str,
+    ):
+        self.timeout = timeout
+        self.call_dir = call_dir
+        self.stdout = stdout
+        self.stderr = stderr
+        super().__init__(
+            f"codex exec timed out after {timeout}s; "
+            f"log_dir={call_dir}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
+        )
+
+
 @dataclass
 class CodexResponseCompleter(TextCompleter):
     """Text completer backed by OpenAI's Responses API."""
@@ -110,6 +131,9 @@ class CodexCliCompleter(TextCompleter):
     append_final_answer_instruction: bool = True
     log_dir: str | None = None
     call_name: str | None = None
+    config_overrides: tuple[str, ...] = ()
+    env: dict[str, str] | None = None
+    skip_git_repo_check: bool = False
     _call_idx: int = field(default=0, init=False, repr=False)
 
     def _build_prompt(self, prompt: str) -> str:
@@ -132,6 +156,8 @@ class CodexCliCompleter(TextCompleter):
         ]
         if self.cwd:
             cmd.extend(["-C", self.cwd])
+        if self.skip_git_repo_check:
+            cmd.append("--skip-git-repo-check")
         if self.ignore_user_config:
             cmd.append("--ignore-user-config")
         if self.ignore_rules:
@@ -142,9 +168,40 @@ class CodexCliCompleter(TextCompleter):
                 "-c",
                 f"model_reasoning_effort={json.dumps(self.reasoning_effort)}",
             ])
+        for override in self.config_overrides:
+            cmd.extend(["-c", override])
         cmd.append("-")
         return cmd
 
+    def _build_process_env(self) -> dict[str, str] | None:
+        if self.env is None:
+            return None
+        process_env = os.environ.copy()
+        process_env.update(self.env)
+        return process_env
+
+    def _log_env(self, call_dir: Path) -> None:
+        keys = (
+            "CUDA_VISIBLE_DEVICES",
+            "CUDA_DEVICE_ORDER",
+            "TORCH_CUDA_ARCH_LIST",
+            "PYTHONUNBUFFERED",
+            "PYTHONPATH",
+            "CUDA_HOME",
+            "LD_LIBRARY_PATH",
+            "TRITON_CACHE_DIR",
+            "TORCH_EXTENSIONS_DIR",
+        )
+        effective = os.environ.copy()
+        if self.env is not None:
+            effective.update(self.env)
+        logged = {key: effective[key] for key in keys if key in effective}
+        if logged:
+            (call_dir / "codex_env.json").write_text(
+                json.dumps(logged, indent=2, sort_keys=True),
+                encoding="utf-8",
+            )
+
     def _next_call_dir(self) -> Path | None:
         if not self.log_dir:
             return None
@@ -192,6 +249,7 @@ class CodexCliCompleter(TextCompleter):
                 json.dumps(cmd, indent=2),
                 encoding="utf-8",
             )
+            self._log_env(call_dir)
 
         try:
             try:
@@ -200,6 +258,7 @@ class CodexCliCompleter(TextCompleter):
                     stdin=asyncio.subprocess.PIPE,
                     stdout=stdout_target,
                     stderr=stderr_target,
+                    env=self._build_process_env(),
                     start_new_session=True,
                 )
                 communicate = process.communicate(prompt.encode("utf-8"))
@@ -219,9 +278,11 @@ class CodexCliCompleter(TextCompleter):
                             f"codex exec timed out after {self.timeout}s\n",
                             encoding="utf-8",
                         )
-                    raise RuntimeError(
-                        f"codex exec timed out after {self.timeout}s; "
-                        f"log_dir={call_dir}\nSTDOUT:\n{stdout_text}\nSTDERR:\n{stderr_text}"
+                    raise CodexCliTimeoutError(
+                        timeout=self.timeout,
+                        call_dir=call_dir,
+                        stdout=stdout_text,
+                        stderr=stderr_text,
                     ) from exc
                 except asyncio.CancelledError:
                     await _kill_process_tree(process)
diff --git a/ttt_discover/rl/codex_no_finetune.py b/ttt_discover/rl/codex_no_finetune.py
index 18a0e9e..f126496 100644
--- a/ttt_discover/rl/codex_no_finetune.py
+++ b/ttt_discover/rl/codex_no_finetune.py
@@ -50,7 +50,9 @@ import json
 import logging
 import os
 import re
+import shutil
 import shlex
+import sys
 import time
 import traceback
 import uuid
@@ -66,6 +68,7 @@ import numpy as np
 
 from ttt_discover.codex_utils.completers import (
     CodexCliCompleter,
+    CodexCliTimeoutError,
     CodexResponseCompleter,
     TextCompleter,
 )
@@ -474,6 +477,9 @@ def _maybe_create_next_state(env: Any, step_idx: int, parsed_code: str, outs: Ve
 class AutonomousCodexCliCompleter(CodexCliCompleter):
     """Minimal Codex CLI agent mode: give Codex a workspace and evaluator."""
 
+    _ISOLATED_REPO_ROOT = Path("/opt/tiger/discover")
+    _ISOLATED_WORKSPACE = _ISOLATED_REPO_ROOT / "workspace"
+
     def __init__(
         self,
         *,
@@ -482,7 +488,9 @@ class AutonomousCodexCliCompleter(CodexCliCompleter):
         step_idx: int,
         eval_timeout: int,
         num_cpus_per_task: int,
+        repo_cwd: str | os.PathLike[str] | None = None,
         prompt_builder: Callable[..., str] | None = None,
+        isolate_danger_full_access: bool = False,
         **kwargs: Any,
     ):
         super().__init__(append_final_answer_instruction=False, **kwargs)
@@ -491,12 +499,14 @@ class AutonomousCodexCliCompleter(CodexCliCompleter):
         self.step_idx = step_idx
         self.eval_timeout = eval_timeout
         self.num_cpus_per_task = num_cpus_per_task
+        self.repo_cwd = Path(repo_cwd or os.getcwd()).resolve()
         self.prompt_builder = prompt_builder
+        self.isolate_danger_full_access = isolate_danger_full_access
         self._call_idx = 0
 
     def _next_workspace(self) -> Path:
         self._call_idx += 1
-        root = Path(self.log_path) / "codex_autonomous_workspaces"
+        root = (Path(self.log_path) / "codex_autonomous_workspaces").resolve()
         workspace = (
             root
             / f"step_{max(0, self.step_idx):06d}"
@@ -517,6 +527,11 @@ class AutonomousCodexCliCompleter(CodexCliCompleter):
                 eval_timeout=self.eval_timeout,
                 num_cpus_per_task=self.num_cpus_per_task,
             )
+            if self.isolate_danger_full_access:
+                full_prompt = full_prompt.replace(
+                    str(workspace),
+                    str(self._ISOLATED_WORKSPACE),
+                )
             (workspace / "prompt.txt").write_text(full_prompt, encoding="utf-8")
             return full_prompt
 
@@ -530,8 +545,11 @@ class AutonomousCodexCliCompleter(CodexCliCompleter):
 
         candidate = shlex.quote(str(workspace / "candidate.py"))
         eval_dir = shlex.quote(str(workspace / "eval_tmp"))
+        python_exe = shlex.quote(sys.executable)
+        repo_cwd = shlex.quote(str(self.repo_cwd))
         evaluator_cmd = (
-            ".venv/bin/python -m repro.cap_set.self_loop_eval "
+            f"PYTHONPATH={repo_cwd}${{PYTHONPATH:+:$PYTHONPATH}} "
+            f"{python_exe} -m repro.cap_set.self_loop_eval "
             f"--candidate {candidate} "
             f"--dimension {shlex.quote(str(self.problem_type))} "
             f"--log-dir {eval_dir} "
@@ -555,6 +573,11 @@ When done, save the best code to:
 
 Finish with exactly one Python code block defining that best `priority(el, n)`.
 """
+        if self.isolate_danger_full_access:
+            full_prompt = full_prompt.replace(
+                str(workspace),
+                str(self._ISOLATED_WORKSPACE),
+            )
         (workspace / "prompt.txt").write_text(full_prompt, encoding="utf-8")
         return full_prompt
 
@@ -564,6 +587,126 @@ Finish with exactly one Python code block defining that best `priority(el, n)`.
             raise RuntimeError("Autonomous Codex workspace was not initialized.")
         return workspace
 
+    def _build_command(self, output_path: str) -> list[str]:
+        workspace = getattr(self, "_workspace", None)
+        if workspace is None:
+            raise RuntimeError("Autonomous Codex workspace was not initialized.")
+        if self.isolate_danger_full_access:
+            return self._build_isolated_command(workspace, output_path)
+        original_cwd = self.cwd
+        self.cwd = str(workspace)
+        try:
+            return super()._build_command(output_path)
+        finally:
+            self.cwd = original_cwd
+
+    def _build_inner_dangerous_command(self, output_path: str) -> list[str]:
+        cmd = [
+            self.codex_command,
+            "exec",
+            "--ephemeral",
+            "--dangerously-bypass-approvals-and-sandbox",
+            "-o",
+            output_path,
+            "-C",
+            str(self._ISOLATED_WORKSPACE),
+            "--skip-git-repo-check",
+        ]
+        if self.ignore_user_config:
+            cmd.append("--ignore-user-config")
+        if self.ignore_rules:
+            cmd.append("--ignore-rules")
+        cmd.extend(["-m", self.model_name or "gpt-5.5"])
+        if self.reasoning_effort:
+            cmd.extend([
+                "-c",
+                f"model_reasoning_effort={json.dumps(self.reasoning_effort)}",
+            ])
+        for override in self.config_overrides:
+            cmd.extend(["-c", override])
+        cmd.append("-")
+        return cmd
+
+    def _prepare_isolated_codex_home(self, workspace: Path) -> Path:
+        codex_home = workspace.parent / f"{workspace.name}_codex_home"
+        codex_home.mkdir(parents=True, exist_ok=True)
+        auth_src = Path.home() / ".codex" / "auth.json"
+        if auth_src.exists():
+            shutil.copy2(auth_src, codex_home / "auth.json")
+        for name in ("installation_id", "version.json", "models_cache.json"):
+            src = Path.home() / ".codex" / name
+            if src.exists():
+                shutil.copy2(src, codex_home / name)
+        return codex_home
+
+    def _build_isolated_command(self, workspace: Path, output_path: str) -> list[str]:
+        codex_home = self._prepare_isolated_codex_home(workspace)
+        isolated_output = str(self._ISOLATED_WORKSPACE / Path(output_path).name)
+        inner_cmd = self._build_inner_dangerous_command(isolated_output)
+
+        mask_roots = []
+        for root in (self.repo_cwd, Path("/opt/tiger/discover")):
+            root = root.resolve()
+            if root.exists() and root not in mask_roots:
+                mask_roots.append(root)
+
+        workspace_shell = shlex.quote(str(workspace))
+        codex_home_shell = shlex.quote(str(codex_home))
+        nvm_shell = shlex.quote(str(Path.home() / ".nvm"))
+        isolated_repo_shell = shlex.quote(str(self._ISOLATED_REPO_ROOT))
+        isolated_workspace_shell = shlex.quote(str(self._ISOLATED_WORKSPACE))
+        mask_roots_shell = " ".join(shlex.quote(str(path)) for path in mask_roots)
+        inner_cmd_shell = shlex.join(inner_cmd)
+        script = f"""
+set -euo pipefail
+
+mount --make-rprivate /
+
+iso_root="$(mktemp -d /tmp/ttt-codex-iso.XXXXXX)"
+mkdir -p "$iso_root/ws" "$iso_root/codex_home" "$iso_root/nvm"
+mount --bind {workspace_shell} "$iso_root/ws"
+mount --bind {codex_home_shell} "$iso_root/codex_home"
+mount --bind {nvm_shell} "$iso_root/nvm"
+
+for path in {mask_roots_shell}; do
+    if [ -d "$path" ]; then
+        mount -t tmpfs tmpfs "$path"
+    fi
+done
+if [ -d /home/tiger ]; then
+    mount -t tmpfs tmpfs /home/tiger
+fi
+
+mkdir -p {isolated_workspace_shell} /home/tiger/.codex /home/tiger/.nvm
+mount --bind "$iso_root/ws" {isolated_workspace_shell}
+mount --bind "$iso_root/codex_home" /home/tiger/.codex
+mount --bind "$iso_root/nvm" /home/tiger/.nvm
+
+export CODEX_HOME=/home/tiger/.codex
+cd {isolated_workspace_shell}
+{inner_cmd_shell}
+"""
+        workspace.joinpath("isolated_inner_command.json").write_text(
+            json.dumps(inner_cmd, indent=2),
+            encoding="utf-8",
+        )
+        workspace.joinpath("isolation.json").write_text(
+            json.dumps(
+                {
+                    "mode": "unshare-mount-namespace",
+                    "real_workspace": str(workspace),
+                    "isolated_workspace": str(self._ISOLATED_WORKSPACE),
+                    "codex_home": str(codex_home),
+                    "masked_roots": [str(path) for path in mask_roots]
+                    + ["/home/tiger"],
+                },
+                indent=2,
+                sort_keys=True,
+            ),
+            encoding="utf-8",
+        )
+        return ["unshare", "-Ur", "-m", "bash", "-lc", script]
+
 
 def _make_completer(
     cfg: CodexNoFinetuneConfig,
@@ -585,7 +728,8 @@ def _make_completer(
                 model_name=cfg.model_name,
                 codex_command=cfg.cli_command,
                 sandbox=cfg.cli_sandbox,
-                cwd=os.getcwd(),
+                cwd=None,
+                repo_cwd=os.getcwd(),
                 timeout=cfg.cli_timeout,
                 semaphore=semaphore,
                 log_path=cfg.log_path,
@@ -593,6 +737,9 @@ def _make_completer(
                 step_idx=step_idx,
                 eval_timeout=cfg.eval_timeout,
                 num_cpus_per_task=max(1, int(cfg.num_cpus_per_task)),
+                config_overrides=("shell_environment_policy.inherit=all",),
+                isolate_danger_full_access=cfg.cli_sandbox == "danger-full-access",
+                skip_git_repo_check=True,
                 prompt_builder=(
                     getattr(env, "build_autonomous_prompt", None)
                     if env is not None
@@ -640,6 +787,7 @@ async def _run_candidate(
     prompt = ""
     response = ""
     parsed_code = ""
+    cli_timeout_error: CodexCliTimeoutError | None = None
     try:
         env = _make_env(cfg, parent_state, sampler)
         prompt = env.get_question()
@@ -651,7 +799,13 @@ async def _run_candidate(
             sample_idx=sample_idx,
             env=env,
         )
-        response = await completer(prompt)
+        try:
+            response = await completer(prompt)
+        except CodexCliTimeoutError as exc:
+            if not cfg.autonomous:
+                raise
+            cli_timeout_error = exc
+            response = ""
         get_languages = getattr(env, "_get_code_languages", None)
         languages = get_languages() if get_languages is not None else ["python"]
         keep_separators_fn = getattr(env, "_should_keep_code_separators", None)
@@ -671,11 +825,21 @@ async def _run_candidate(
             )
             if autonomous_submission is not None:
                 parsed_code = autonomous_submission
-                parsed_code_source = "workspace_submission.py"
+                parsed_code_source = (
+                    "workspace_submission.py_after_cli_timeout"
+                    if cli_timeout_error is not None
+                    else "workspace_submission.py"
+                )
         correct_format = _check_candidate_format(env, parsed_code)
         outs = await _safe_grade(cfg, env, parsed_code, correct_format)
         metrics = _build_metrics(env, outs, response, parsed_code, correct_format)
         metrics["codex/parsed_code_source"] = parsed_code_source
+        metrics["codex/cli_timeout_salvaged"] = cli_timeout_error is not None
+        if cli_timeout_error is not None:
+            metrics["codex/cli_timeout_error"] = (
+                f"codex exec timed out after {cli_timeout_error.timeout}s; "
+                f"log_dir={cli_timeout_error.call_dir}"
+            )
         if autonomous_submission_path is not None:
             metrics["codex/autonomous_submission_path"] = autonomous_submission_path
         next_state = _maybe_create_next_state(env, step_idx, parsed_code, outs)
````
</details>

