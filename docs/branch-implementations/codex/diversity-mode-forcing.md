# codex/diversity-mode-forcing

## Summary

prompt 层 diversity mode forcing：维护一组模式说明，按 sample index 轮转注入 prompt；同时扩展 Codex CLI 环境、日志和输出 token 相关配置。

Note: 注意：该 worktree 中还混有若干 GPUMode/notes/tools 相关 untracked 文件，文档中已单独列出；这些文件不一定属于 mode forcing 本身。

## Branch State

- Worktree: `/opt/tiger/discover`
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

- `max_output_tokens`
- `config_overrides`
- `env`
- `skip_git_repo_check`
- `log_root`
- `codex_max_output_tokens`
- `diversity_modes`
- `cli_timeout_error`

### Constants

- None

### Classes

- `CodexCliTimeoutError`

### Functions

- `__init__`
- `_build_process_env`
- `_log_env`
- `validate_diversity_modes`
- `_diversity_mode_for_sample`
- `_append_diversity_mode_block`
- `_build_command`

## Diff Summary

- Worktree tracked shortstat: `5 files changed, 207 insertions(+), 16 deletions(-)`
- Untracked files: `8`

### Worktree Status

````text
 M examples/gpu_mode/env.py
 M ttt_discover/codex_utils/completers.py
 M ttt_discover/codex_utils/discovery.py
 M ttt_discover/discovery.py
 M ttt_discover/rl/codex_no_finetune.py
?? GPUMode/top_solutions/trimul/web_top5/EVAL_README.md
?? kernelbench_self_evolve_integrated.md
?? my_handbook.md
?? repro/gpu_mode/
?? repro/run_discovery.py
?? results/kernel-engineering/trimul_subagent_attempt.py
?? tools/
````
### Tracked Worktree Files

````text
M	examples/gpu_mode/env.py
M	ttt_discover/codex_utils/completers.py
M	ttt_discover/codex_utils/discovery.py
M	ttt_discover/discovery.py
M	ttt_discover/rl/codex_no_finetune.py
````
### Untracked Files

- `GPUMode/top_solutions/trimul/web_top5/EVAL_README.md (726 bytes)`
- `kernelbench_self_evolve_integrated.md (1404 bytes)`
- `my_handbook.md (297 bytes)`
- `repro/gpu_mode/run_0608.sh (1910 bytes)`
- `repro/gpu_mode/run_0608_mode_forcing.sh (1148 bytes)`
- `repro/run_discovery.py (16041 bytes)`
- `results/kernel-engineering/trimul_subagent_attempt.py (3890 bytes)`
- `tools/gpu_occupy.py (5331 bytes)`

## Untracked File Snapshots

These files are not part of `git diff` yet, but they are part of the worktree implementation state recorded here.

### `GPUMode/top_solutions/trimul/web_top5/EVAL_README.md`

````markdown
# TriMul Top5 Local Eval Notes
Correct local eval must follow `/opt/tiger/discover/codex_runs/trimul_handbook.md`: build `SubmissionMode.LEADERBOARD` and call `libkernelbot.run_eval.run_config`.
Prior official-entry A800 top5 rerun: `/opt/tiger/discover/codex_runs/trimul_official_entry_a800/summary.json`; current repeat: `/opt/tiger/discover/codex_runs/trimul_official_run_config/results/summary.json`.
Bool-string leaderboard rerun matching official `nomask` parsing: `/opt/tiger/discover/codex_runs/trimul_leaderboard_boolstrings_a800/summary.json`.
Numeric-`nomask` rerun `/opt/tiger/discover/codex_runs/trimul_leaderboard_a800/summary.json` can show rank1/rank3 null and must not be used as official pass/fail evidence.
````

### `kernelbench_self_evolve_integrated.md`

````markdown
  本地已拉回并解压到：

  /opt/tiger/discover/GPUMode/data/kernelbench_self_evolve_integrated

  本地 zip 副本在：

  /opt/tiger/discover/GPUMode/data/kernelbench_self_evolve_integrated.zip

  已推回 HDFS：

  hdfs://haruna/home/byte_data_seed/ssd_hldy/user/zhangjinghan.23/autoresearch_tasks/extracted/kernelbench_self_evolve_integrated

  对应挂载路径：

  /mnt/hdfs/hl/autoresearch_tasks/extracted/kernelbench_self_evolve_integrated

  结论：

  - 共 270 个任务：L1=100、L2=100、L3=50、L4=20
  - HDFS 最终大小约 54.3G，文件约 4140 个
  - 每个任务都有 problem.md、solution.py、verifier.py、10 个 test_cases/*.txt
  - KB_L1_001 到 KB_L1_010 是旧格式：stdin JSON、stdout JSON、runner.py
  - 其余 260 个是 self-evolve 格式：写 submission.py，定义 solve(*inputs) -> torch.Tensor，用 verifier.py --submission ...
  - baseline_solution.py 和 task_spec.json 只存在于后 260 个新格式任务
  - 原 zip 里有 __pycache__，解压时排除了；smoke test 生成的 pycache 也已从本地和 HDFS 删除

  校验已做：

  - 本地旧格式 L1 smoke test：10/10 passed
  - 本地新格式 L1_011 verifier：通过
  - 本地 L2_001 verifier：通过
  - HDFS 挂载路径旧格式 L1 smoke test：10/10 passed
  - HDFS 挂载路径新格式 L1_011 verifier：通过
  - 最终确认本地和 HDFS 都没有 __pycache__
````

### `my_handbook.md`

````markdown
  官方 benchmark 才是固定/权威的，调用的是：

  libkernelbot.run_eval.run_config(...)
  SubmissionMode.LEADERBOARD

  对应地址主要是：

  - /opt/tiger/discover/examples/gpu_mode/lib/libkernelbot/run_eval.py
  - /opt/tiger/discover/examples/gpu_mode/lib/bioml/trimul/task.yml
````

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
# The Codex completer forces -C to that per-sample workspace. danger-full-access
# is required here because the workspace-write sandbox cannot see CUDA devices.
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

### `repro/gpu_mode/run_0608_mode_forcing.sh`

````bash
#!/usr/bin/env bash
set -euo pipefail

# Mode forcing: non-auto TTT Discover samples with prompt-only diversity modes.
# groups-per-batch = parent states sampled per outer round.
# group-size = Codex samples per parent; here each batch covers the mode list once.
python repro/run_discovery.py \
    --task trimul \
    --runner codex_no_finetune \
    --experiment-name trimul_0608_mode_forcing_gpu2 \
    --log-root codex_runs/trimul_exec_workspaces \
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
    --codex-diversity-mode baseline \
    --codex-diversity-mode minimal_change \
    --codex-diversity-mode local_refactor \
    --codex-diversity-mode algorithmic_shift \
    --codex-diversity-mode correctness_first \
    --codex-diversity-mode performance_probe
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
        "--codex-diversity-mode",
        action="append",
        default=None,
        help=(
            "Prompt-only Codex diversity mode. May be repeated; omit to disable "
            "mode forcing."
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
    codex_diversity_modes = tuple(args.codex_diversity_mode or ())
    if codex_diversity_modes:
        from ttt_discover.rl.codex_no_finetune import validate_diversity_modes

        validate_diversity_modes(codex_diversity_modes)

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
        print(f"codex_diversity_modes={codex_diversity_modes!r}")
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
            diversity_modes=codex_diversity_modes,
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

### `results/kernel-engineering/trimul_subagent_attempt.py`

````python
"""
Outgoing TriMul forward attempt.

This implementation keeps the numerically sensitive normalization and linear
layers in PyTorch float32, and uses a Triton kernel to fuse sigmoid gating,
mask application, and packing of left/right activations into the [B, H, N, N]
layout needed by the batched triangle multiplication.
"""

from typing import Dict, Tuple

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _gate_mask_pack_kernel(
    left_proj_ptr,
    right_proj_ptr,
    left_gate_ptr,
    right_gate_ptr,
    mask_ptr,
    left_out_ptr,
    right_out_ptr,
    total: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < total

    row = offsets // H
    h = offsets - row * H

    mask_val = tl.load(mask_ptr + row, mask=valid, other=0.0).to(tl.float32)
    lp = tl.load(left_proj_ptr + offsets, mask=valid, other=0.0).to(tl.float32)
    rp = tl.load(right_proj_ptr + offsets, mask=valid, other=0.0).to(tl.float32)
    lg = tl.load(left_gate_ptr + offsets, mask=valid, other=0.0).to(tl.float32)
    rg = tl.load(right_gate_ptr + offsets, mask=valid, other=0.0).to(tl.float32)

    left = lp * (1.0 / (1.0 + tl.exp(-lg))) * mask_val
    right = rp * (1.0 / (1.0 + tl.exp(-rg))) * mask_val

    n2 = N * N
    b = row // n2
    rem = row - b * n2
    i = rem // N
    k = rem - i * N

    packed_offsets = ((b * H + h) * N + i) * N + k
    tl.store(left_out_ptr + packed_offsets, left, mask=valid)
    tl.store(right_out_ptr + packed_offsets, right, mask=valid)


def _pack_gated_sides(
    left_proj: torch.Tensor,
    right_proj: torch.Tensor,
    left_gate_logits: torch.Tensor,
    right_gate_logits: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    bsz, n, _, hdim = left_proj.shape
    total = bsz * n * n * hdim

    left = torch.empty((bsz, hdim, n, n), device=left_proj.device, dtype=torch.bfloat16)
    right = torch.empty_like(left)

    block = 256
    grid = (triton.cdiv(total, block),)
    _gate_mask_pack_kernel[grid](
        left_proj.contiguous(),
        right_proj.contiguous(),
        left_gate_logits.contiguous(),
        right_gate_logits.contiguous(),
        mask.reshape(bsz * n * n).contiguous(),
        left,
        right,
        total,
        n,
        hdim,
        BLOCK=block,
        num_warps=4,
    )

    return left, right


@torch.no_grad()
def custom_kernel(
    data: Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], Dict]
) -> torch.Tensor:
    input_tensor, mask, weights, config = data
    dim = config["dim"]
    hidden_dim = config["hidden_dim"]
    bsz, n, _, _ = input_tensor.shape

    x = F.layer_norm(
        input_tensor,
        (dim,),
        weights["norm.weight"],
        weights["norm.bias"],
        eps=1e-5,
    )

    left_proj = F.linear(x, weights["left_proj.weight"])
    right_proj = F.linear(x, weights["right_proj.weight"])
    left_gate_logits = F.linear(x, weights["left_gate.weight"])
    right_gate_logits = F.linear(x, weights["right_gate.weight"])
    out_gate = torch.sigmoid(F.linear(x, weights["out_gate.weight"]))

    left, right = _pack_gated_sides(
        left_proj,
        right_proj,
        left_gate_logits,
        right_gate_logits,
        mask,
    )

    tri = torch.bmm(
        left.view(bsz * hidden_dim, n, n),
        right.view(bsz * hidden_dim, n, n).transpose(1, 2),
    )
    tri = tri.view(bsz, hidden_dim, n, n).permute(0, 2, 3, 1).contiguous()
    tri = tri.to(torch.float32)

    tri = F.layer_norm(
        tri,
        (hidden_dim,),
        weights["to_out_norm.weight"],
        weights["to_out_norm.bias"],
        eps=1e-5,
    )
    tri = tri * out_gate
    out = F.linear(tri, weights["to_out.weight"])
    return out.to(torch.float32)
````

### `tools/gpu_occupy.py`

````python
#!/usr/bin/env python3
"""Reserve GPU memory and optionally keep CUDA cores busy.

Use only on machines where you are allowed to consume the selected GPUs.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import signal
import sys
import time
from dataclasses import dataclass


MiB = 1024 * 1024


@dataclass(frozen=True)
class WorkerConfig:
    gpu_id: int
    memory_fraction: float
    chunk_mib: int
    matrix_size: int
    compute: bool


def parse_gpu_list(value: str, device_count: int) -> list[int]:
    if value == "all":
        return list(range(device_count))

    gpu_ids: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        gpu_id = int(part)
        if gpu_id < 0 or gpu_id >= device_count:
            raise ValueError(f"GPU {gpu_id} is out of range; found {device_count} CUDA device(s)")
        gpu_ids.append(gpu_id)

    if not gpu_ids:
        raise ValueError("No GPUs selected")
    return gpu_ids


def reserve_memory(torch, gpu_id: int, memory_fraction: float, chunk_mib: int):
    free_bytes, total_bytes = torch.cuda.mem_get_info(gpu_id)
    target_bytes = int(free_bytes * memory_fraction)
    chunk_bytes = max(1, chunk_mib) * MiB

    blocks = []
    allocated = 0
    while allocated < target_bytes:
        this_chunk = min(chunk_bytes, target_bytes - allocated)
        try:
            block = torch.empty((this_chunk,), dtype=torch.uint8, device=f"cuda:{gpu_id}")
            block.fill_(1)
            blocks.append(block)
            allocated += this_chunk
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            break

    torch.cuda.synchronize(gpu_id)
    print(
        f"[gpu {gpu_id}] reserved {allocated / MiB:.0f} MiB "
        f"of {total_bytes / MiB:.0f} MiB total",
        flush=True,
    )
    return blocks


def worker(config: WorkerConfig, stop_event: mp.Event) -> None:
    import torch

    torch.cuda.set_device(config.gpu_id)
    blocks = reserve_memory(
        torch,
        config.gpu_id,
        config.memory_fraction,
        config.chunk_mib,
    )

    if not config.compute:
        while not stop_event.is_set():
            time.sleep(1)
        return

    dtype = torch.float16
    n = config.matrix_size
    a = torch.randn((n, n), device=f"cuda:{config.gpu_id}", dtype=dtype)
    b = torch.randn((n, n), device=f"cuda:{config.gpu_id}", dtype=dtype)

    i = 0
    while not stop_event.is_set():
        a = a @ b
        if i % 8 == 0:
            a = a / a.norm().clamp_min(1)
        i += 1

    del a, b, blocks
    torch.cuda.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gpus",
        default="all",
        help="Comma-separated GPU ids, for example 0,1,2,3,4,5,6,7; default: all",
    )
    parser.add_argument(
        "--memory-fraction",
        type=float,
        default=0.90,
        help="Fraction of currently free memory to reserve on each GPU; default: 0.90",
    )
    parser.add_argument(
        "--chunk-mib",
        type=int,
        default=256,
        help="Allocation chunk size in MiB; default: 256",
    )
    parser.add_argument(
        "--matrix-size",
        type=int,
        default=4096,
        help="Square matrix size for compute load; default: 4096",
    )
    parser.add_argument(
        "--no-compute",
        action="store_true",
        help="Only reserve GPU memory; do not run matrix multiplication",
    )
    args = parser.parse_args()

    if not 0 < args.memory_fraction < 1:
        parser.error("--memory-fraction must be between 0 and 1")

    import torch

    if not torch.cuda.is_available():
        print("CUDA is not available to this Python process.", file=sys.stderr)
        return 1

    try:
        gpu_ids = parse_gpu_list(args.gpus, torch.cuda.device_count())
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    stop_event = mp.Event()

    def request_stop(signum, _frame):
        print(f"received signal {signum}; stopping workers", flush=True)
        stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    configs = [
        WorkerConfig(
            gpu_id=gpu_id,
            memory_fraction=args.memory_fraction,
            chunk_mib=args.chunk_mib,
            matrix_size=args.matrix_size,
            compute=not args.no_compute,
        )
        for gpu_id in gpu_ids
    ]

    processes = [mp.Process(target=worker, args=(config, stop_event)) for config in configs]
    for process in processes:
        process.start()

    print(f"started {len(processes)} worker(s), parent pid={os.getpid()}", flush=True)

    try:
        while any(process.is_alive() for process in processes):
            time.sleep(1)
    finally:
        stop_event.set()
        for process in processes:
            process.join(timeout=10)
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join()

    return 0


if __name__ == "__main__":
    mp.set_start_method("spawn")
    raise SystemExit(main())
````

## Diff Stat

### Worktree Tracked Diff Stat

````text
 examples/gpu_mode/env.py               |  23 ++++++-
 ttt_discover/codex_utils/completers.py |  72 ++++++++++++++++++--
 ttt_discover/codex_utils/discovery.py  |   5 +-
 ttt_discover/discovery.py              |   6 +-
 ttt_discover/rl/codex_no_finetune.py   | 117 +++++++++++++++++++++++++++++++--
 5 files changed, 207 insertions(+), 16 deletions(-)
````
## Raw Diff

<details>
<summary>Tracked worktree patch</summary>

````diff
diff --git a/examples/gpu_mode/env.py b/examples/gpu_mode/env.py
index a37953f..7c766ac 100644
--- a/examples/gpu_mode/env.py
+++ b/examples/gpu_mode/env.py
@@ -245,6 +245,8 @@ import os
 import sys
 import tempfile
 
+import torch
+
 ROOT = Path({str(REPO_ROOT)!r})
 sys.path.insert(0, str(ROOT / "examples/gpu_mode/lib"))
 
@@ -269,6 +271,13 @@ def compute_score_us(result, task):
 
 
 def main():
+    print(f"CUDA_VISIBLE_DEVICES {{os.environ.get('CUDA_VISIBLE_DEVICES')}}")
+    print(f"TORCH_CUDA_ARCH_LIST {{os.environ.get('TORCH_CUDA_ARCH_LIST')}}")
+    print(f"torch_cuda_available {{torch.cuda.is_available()}}")
+    print(f"torch_cuda_device_count {{torch.cuda.device_count()}}")
+    if not torch.cuda.is_available():
+        raise SystemExit("CUDA is not available in the Codex self-eval process")
+    print(f"torch_cuda_device {{torch.cuda.get_device_name(0)}}")
     task = make_task_definition(Path({str(task_yaml)!r})).task
     code = Path("submission.py").read_text()
     config = build_task_config(
@@ -381,7 +390,15 @@ class GpuModeEnv(Environment):
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
@@ -399,6 +416,10 @@ change the official score or delete your candidate.
 Run this evaluator after each revision:
 {evaluator_cmd}
 
+Do not edit `eval_candidate.py` or the copied task/eval files to improve a score.
+Only `submission.py` is a valid candidate artifact, and the outer runner will
+re-score it with trusted repository files.
+
 When done, put the best implementation in:
 {workspace / "submission.py"}
 
diff --git a/ttt_discover/codex_utils/completers.py b/ttt_discover/codex_utils/completers.py
index 3e9b014..78e1094 100644
--- a/ttt_discover/codex_utils/completers.py
+++ b/ttt_discover/codex_utils/completers.py
@@ -31,12 +31,33 @@ class TextCompleter:
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
 
     model_name: str | None = "gpt-5.2-codex"
-    max_output_tokens: int = 8192
+    max_output_tokens: int | None = None
     reasoning_effort: str | None = "medium"
     temperature: float | None = None
     api_key_env: str = "OPENAI_API_KEY"
@@ -64,9 +85,10 @@ class CodexResponseCompleter(TextCompleter):
         request: dict[str, Any] = {
             "model": self.model_name or "gpt-5.2-codex",
             "input": prompt,
-            "max_output_tokens": self.max_output_tokens,
             "store": False,
         }
+        if self.max_output_tokens is not None:
+            request["max_output_tokens"] = self.max_output_tokens
         if self.reasoning_effort is not None:
             request["reasoning"] = {"effort": self.reasoning_effort}
         if self.temperature is not None:
@@ -110,6 +132,9 @@ class CodexCliCompleter(TextCompleter):
     append_final_answer_instruction: bool = True
     log_dir: str | None = None
     call_name: str | None = None
+    config_overrides: tuple[str, ...] = ()
+    env: dict[str, str] | None = None
+    skip_git_repo_check: bool = False
     _call_idx: int = field(default=0, init=False, repr=False)
 
     def _build_prompt(self, prompt: str) -> str:
@@ -132,6 +157,8 @@ class CodexCliCompleter(TextCompleter):
         ]
         if self.cwd:
             cmd.extend(["-C", self.cwd])
+        if self.skip_git_repo_check:
+            cmd.append("--skip-git-repo-check")
         if self.ignore_user_config:
             cmd.append("--ignore-user-config")
         if self.ignore_rules:
@@ -142,9 +169,40 @@ class CodexCliCompleter(TextCompleter):
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
@@ -192,6 +250,7 @@ class CodexCliCompleter(TextCompleter):
                 json.dumps(cmd, indent=2),
                 encoding="utf-8",
             )
+            self._log_env(call_dir)
 
         try:
             try:
@@ -200,6 +259,7 @@ class CodexCliCompleter(TextCompleter):
                     stdin=asyncio.subprocess.PIPE,
                     stdout=stdout_target,
                     stderr=stderr_target,
+                    env=self._build_process_env(),
                     start_new_session=True,
                 )
                 communicate = process.communicate(prompt.encode("utf-8"))
@@ -219,9 +279,11 @@ class CodexCliCompleter(TextCompleter):
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
diff --git a/ttt_discover/codex_utils/discovery.py b/ttt_discover/codex_utils/discovery.py
index b21a80f..8d0453e 100644
--- a/ttt_discover/codex_utils/discovery.py
+++ b/ttt_discover/codex_utils/discovery.py
@@ -26,6 +26,7 @@ class DiscoverConfig:
     remove_constant_reward_groups: bool = True
 
     experiment_name: str | None = None
+    log_root: str = "tinker_log"
     wandb_project: str | None = "tinker-cookbook"
 
     env_type: type | None = None
@@ -35,7 +36,7 @@ class DiscoverConfig:
 
     codex_backend: Literal["cli", "responses"] = "cli"
     codex_model_name: str | None = None
-    codex_max_output_tokens: int = 8192
+    codex_max_output_tokens: int | None = None
     codex_temperature: float | None = None
     codex_api_key_env: str = "OPENAI_API_KEY"
     codex_base_url: str | None = None
@@ -61,7 +62,7 @@ def _run_codex_no_finetune(config: DiscoverConfig) -> None:
     if config.env_type is None:
         raise ValueError("env_type is required")
     experiment_name = config.experiment_name or "codex-no-finetune"
-    log_path = f"./tinker_log/{experiment_name}"
+    log_path = os.path.join(os.path.expanduser(config.log_root), experiment_name)
     os.makedirs(log_path, exist_ok=True)
 
     codex_config = CodexNoFinetuneConfig(
diff --git a/ttt_discover/discovery.py b/ttt_discover/discovery.py
index 6cdb434..62a01da 100644
--- a/ttt_discover/discovery.py
+++ b/ttt_discover/discovery.py
@@ -37,6 +37,7 @@ class DiscoverConfig:
 
     # Misc config
     experiment_name: str | None = None
+    log_root: str = "tinker_log"
     wandb_project: str | None = "tinker-cookbook"
 
     # Environment-specific
@@ -48,7 +49,7 @@ class DiscoverConfig:
     # Codex no-finetune config.
     codex_backend: Literal["cli", "responses"] = "cli"
     codex_model_name: str | None = None
-    codex_max_output_tokens: int = 8192
+    codex_max_output_tokens: int | None = None
     codex_temperature: float | None = None
     codex_api_key_env: str = "OPENAI_API_KEY"
     codex_base_url: str | None = None
@@ -116,7 +117,8 @@ async def discover_impl(config: DiscoverConfig):
     model_name_for_tokenizer = config.model_name
 
     # create log path if it doesn't exist
-    log_path = f"./tinker_log/{config.experiment_name}"
+    log_root = os.path.expanduser(getattr(config, "log_root", "tinker_log"))
+    log_path = os.path.join(log_root, str(config.experiment_name))
     log_file = os.path.join(log_path, "train.log")
 
     misc_utils.check_log_dir(log_path, behavior_if_exists="resume")
diff --git a/ttt_discover/rl/codex_no_finetune.py b/ttt_discover/rl/codex_no_finetune.py
index 18a0e9e..9c8f90a 100644
--- a/ttt_discover/rl/codex_no_finetune.py
+++ b/ttt_discover/rl/codex_no_finetune.py
@@ -51,6 +51,7 @@ import logging
 import os
 import re
 import shlex
+import sys
 import time
 import traceback
 import uuid
@@ -66,6 +67,7 @@ import numpy as np
 
 from ttt_discover.codex_utils.completers import (
     CodexCliCompleter,
+    CodexCliTimeoutError,
     CodexResponseCompleter,
     TextCompleter,
 )
@@ -81,6 +83,33 @@ logger = logging.getLogger(__name__)
 
 SAFE_GRADE_EXECUTOR = ThreadPoolExecutor(max_workers=4096)
 
+DIVERSITY_MODE_REGISTRY: dict[str, str] = {
+    "baseline": (
+        "a balanced improvement that preserves the current solution's broad "
+        "intent while considering one useful new idea."
+    ),
+    "minimal_change": (
+        "small, targeted edits that improve the current approach while keeping "
+        "its structure recognizable."
+    ),
+    "local_refactor": (
+        "refining nearby logic, helper structure, or data flow to make a modest "
+        "improvement without changing the main strategy."
+    ),
+    "algorithmic_shift": (
+        "a different high-level algorithmic angle or scoring heuristic while "
+        "staying compatible with the task interface."
+    ),
+    "correctness_first": (
+        "robustness, edge cases, valid formatting, and conservative improvements "
+        "likely to pass the evaluator."
+    ),
+    "performance_probe": (
+        "one focused performance-oriented idea, such as reducing overhead, "
+        "simplifying hot-path work, or improving memory access patterns."
+    ),
+}
+
 
 @dataclass
 class VerifyResult:
@@ -136,6 +165,7 @@ class CodexNoFinetuneConfig:
     max_concurrent_requests: int | None = 4
     initial_program_paths: tuple[str, ...] = ()
     initial_pool_paths: tuple[str, ...] = ()
+    diversity_modes: tuple[str, ...] = ()
     topk_children: int = 16
 
     # False is TTT Discover: non-auto, many short samples. True is AutoEvolve:
@@ -237,6 +267,36 @@ def all_same(xs: list[Any]) -> bool:
     return bool(xs) and all(x == xs[0] for x in xs)
 
 
+def validate_diversity_modes(modes: tuple[str, ...]) -> None:
+    unknown = sorted({mode for mode in modes if mode not in DIVERSITY_MODE_REGISTRY})
+    if unknown:
+        known = ", ".join(DIVERSITY_MODE_REGISTRY)
+        raise ValueError(
+            "Unknown Codex diversity mode(s): "
+            f"{', '.join(unknown)}. Known modes: {known}"
+        )
+
+
+def _diversity_mode_for_sample(cfg: CodexNoFinetuneConfig, sample_idx: int) -> str:
+    if not cfg.diversity_modes:
+        return ""
+    validate_diversity_modes(cfg.diversity_modes)
+    return cfg.diversity_modes[sample_idx % len(cfg.diversity_modes)]
+
+
+def _append_diversity_mode_block(prompt: str, mode: str) -> str:
+    if not mode:
+        return prompt
+    guidance = DIVERSITY_MODE_REGISTRY[mode]
+    block = (
+        f"--- Diversity Mode: {mode} ---\n"
+        f"For this sample, bias your exploration toward {guidance}\n"
+        "Keep all original task rules and return exactly the required final code block."
+    )
+    prompt = prompt.rstrip()
+    return f"{prompt}\n\n{block}" if prompt else block
+
+
 def append_agent_outputs(log_path: str, step: int, results: list[CandidateResult]) -> None:
     os.makedirs(log_path, exist_ok=True)
     output_path = os.path.join(log_path, "agent_outputs.jsonl")
@@ -482,6 +542,7 @@ class AutonomousCodexCliCompleter(CodexCliCompleter):
         step_idx: int,
         eval_timeout: int,
         num_cpus_per_task: int,
+        repo_cwd: str | os.PathLike[str] | None = None,
         prompt_builder: Callable[..., str] | None = None,
         **kwargs: Any,
     ):
@@ -491,12 +552,13 @@ class AutonomousCodexCliCompleter(CodexCliCompleter):
         self.step_idx = step_idx
         self.eval_timeout = eval_timeout
         self.num_cpus_per_task = num_cpus_per_task
+        self.repo_cwd = Path(repo_cwd or os.getcwd()).resolve()
         self.prompt_builder = prompt_builder
         self._call_idx = 0
 
     def _next_workspace(self) -> Path:
         self._call_idx += 1
-        root = Path(self.log_path) / "codex_autonomous_workspaces"
+        root = (Path(self.log_path) / "codex_autonomous_workspaces").resolve()
         workspace = (
             root
             / f"step_{max(0, self.step_idx):06d}"
@@ -530,8 +592,11 @@ class AutonomousCodexCliCompleter(CodexCliCompleter):
 
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
@@ -564,6 +629,17 @@ Finish with exactly one Python code block defining that best `priority(el, n)`.
             raise RuntimeError("Autonomous Codex workspace was not initialized.")
         return workspace
 
+    def _build_command(self, output_path: str) -> list[str]:
+        workspace = getattr(self, "_workspace", None)
+        if workspace is None:
+            raise RuntimeError("Autonomous Codex workspace was not initialized.")
+        original_cwd = self.cwd
+        self.cwd = str(workspace)
+        try:
+            return super()._build_command(output_path)
+        finally:
+            self.cwd = original_cwd
+
 
 def _make_completer(
     cfg: CodexNoFinetuneConfig,
@@ -585,7 +661,8 @@ def _make_completer(
                 model_name=cfg.model_name,
                 codex_command=cfg.cli_command,
                 sandbox=cfg.cli_sandbox,
-                cwd=os.getcwd(),
+                cwd=None,
+                repo_cwd=os.getcwd(),
                 timeout=cfg.cli_timeout,
                 semaphore=semaphore,
                 log_path=cfg.log_path,
@@ -593,6 +670,8 @@ def _make_completer(
                 step_idx=step_idx,
                 eval_timeout=cfg.eval_timeout,
                 num_cpus_per_task=max(1, int(cfg.num_cpus_per_task)),
+                config_overrides=("shell_environment_policy.inherit=all",),
+                skip_git_repo_check=True,
                 prompt_builder=(
                     getattr(env, "build_autonomous_prompt", None)
                     if env is not None
@@ -640,9 +719,12 @@ async def _run_candidate(
     prompt = ""
     response = ""
     parsed_code = ""
+    diversity_mode = _diversity_mode_for_sample(cfg, sample_idx)
+    cli_timeout_error: CodexCliTimeoutError | None = None
     try:
         env = _make_env(cfg, parent_state, sampler)
         prompt = env.get_question()
+        prompt = _append_diversity_mode_block(prompt, diversity_mode)
         completer = _make_completer(
             cfg,
             semaphore=semaphore,
@@ -651,7 +733,13 @@ async def _run_candidate(
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
@@ -671,11 +759,23 @@ async def _run_candidate(
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
+        metrics["prompt"] = prompt
+        metrics["codex/diversity_mode"] = diversity_mode
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
@@ -713,7 +813,11 @@ async def _run_candidate(
             correctness=0.0,
             raw_score=None,
             msg=error_msg,
-            metrics={"error": error_msg},
+            metrics={
+                "error": error_msg,
+                "prompt": prompt,
+                "codex/diversity_mode": diversity_mode,
+            },
             next_state=None,
             error=error_msg,
         )
@@ -987,6 +1091,7 @@ async def main(cfg: CodexNoFinetuneConfig) -> None:
         raise ValueError("num_epochs must be >= 1")
     if not cfg.log_path:
         raise ValueError("log_path is required")
+    validate_diversity_modes(cfg.diversity_modes)
 
     object.__setattr__(cfg, "log_path", os.path.expanduser(cfg.log_path))
     os.makedirs(cfg.log_path, exist_ok=True)
````
</details>

