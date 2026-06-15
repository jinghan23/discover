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
        "--codex-autonomous-blackbox",
        action="store_true",
        help="In AutoEvolve mode, use a blackbox verifier service for local evals.",
    )
    parser.add_argument(
        "--blackbox-eval-socket",
        default=None,
        help="Unix socket path for the blackbox verifier service.",
    )
    parser.add_argument("--blackbox-eval-host", default="127.0.0.1")
    parser.add_argument("--blackbox-eval-port", type=int, default=None)
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
    if args.codex_autonomous_blackbox and not args.codex_autonomous:
        raise ValueError("--codex-autonomous-blackbox requires --codex-autonomous")

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
        print(f"codex_autonomous_blackbox={args.codex_autonomous_blackbox}")
        print(f"blackbox_eval_socket={args.blackbox_eval_socket!r}")
        print(f"blackbox_eval_host={args.blackbox_eval_host!r}")
        print(f"blackbox_eval_port={args.blackbox_eval_port!r}")
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
            autonomous_blackbox=args.codex_autonomous_blackbox,
            blackbox_eval_socket=args.blackbox_eval_socket,
            blackbox_eval_host=args.blackbox_eval_host,
            blackbox_eval_port=args.blackbox_eval_port,
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
        codex_autonomous_blackbox=args.codex_autonomous_blackbox,
        blackbox_eval_socket=args.blackbox_eval_socket,
        blackbox_eval_host=args.blackbox_eval_host,
        blackbox_eval_port=args.blackbox_eval_port,
        remove_constant_reward_groups=remove_constant_reward_groups,
    )
    discover(config)


if __name__ == "__main__":
    main()
