from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

from examples.cap_set_priority.env import CapSetPriorityEnv
from ttt_discover.rl.codex_no_finetune import (
    CodexNoFinetuneConfig,
    main as codex_no_finetune_main,
)


DEFAULT_INITIAL_POOL = Path(__file__).with_name("initial_pool_400_to_512.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run autonomous Codex cap-set discovery through codex_no_finetune."
    )
    parser.add_argument("--dimension", type=int, default=8)
    parser.add_argument(
        "--experiment-name",
        default="capset-codex-experimental",
        help="Experiment name used under tinker_log/.",
    )
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--group-size", type=int, default=1)
    parser.add_argument("--groups-per-batch", type=int, default=1)
    parser.add_argument(
        "--wandb-project",
        default=os.environ.get("WANDB_PROJECT"),
        help="WANDB project name. Omit or pass '' to disable.",
    )
    parser.add_argument("--codex-model-name", default=None)
    parser.add_argument("--codex-cli-command", default="codex")
    parser.add_argument(
        "--codex-cli-sandbox",
        choices=("read-only", "workspace-write", "danger-full-access"),
        default="workspace-write",
    )
    parser.add_argument("--codex-cli-timeout", type=float, default=None)
    parser.add_argument("--codex-max-concurrent-requests", type=int, default=None)
    parser.add_argument("--num-cpus-per-task", type=int, default=1)
    parser.add_argument("--eval-timeout", type=int, default=45)
    parser.add_argument(
        "--codex-initial-program",
        action="append",
        default=None,
        help="Additional seed priority program to verify and add to the sampler pool.",
    )
    parser.add_argument(
        "--log-root",
        default="tinker_log",
        help="Directory root for experiment logs.",
    )
    return parser.parse_args()


def default_initial_programs(args: argparse.Namespace) -> list[str]:
    return list(args.codex_initial_program or [])


def main() -> None:
    args = parse_args()
    log_path = Path(args.log_root) / args.experiment_name

    config_kwargs = {}
    if args.codex_max_concurrent_requests is not None:
        config_kwargs["max_concurrent_requests"] = args.codex_max_concurrent_requests

    cfg = CodexNoFinetuneConfig(
        env_type=CapSetPriorityEnv,
        problem_type=str(args.dimension),
        backend="cli",
        model_name=args.codex_model_name,
        groups_per_batch=args.groups_per_batch,
        group_size=args.group_size,
        num_cpus_per_task=args.num_cpus_per_task,
        eval_timeout=args.eval_timeout,
        num_epochs=args.num_epochs,
        cli_command=args.codex_cli_command,
        cli_sandbox=args.codex_cli_sandbox,
        cli_timeout=args.codex_cli_timeout,
        autonomous=True,
        initial_program_paths=tuple(default_initial_programs(args)),
        initial_pool_paths=(str(DEFAULT_INITIAL_POOL),),
        wandb_project=args.wandb_project,
        wandb_name=args.experiment_name,
        log_path=str(log_path),
        **config_kwargs,
    )
    asyncio.run(codex_no_finetune_main(cfg))


if __name__ == "__main__":
    main()
