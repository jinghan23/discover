from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.erdos_min_overlap.env import ErdosMinOverlapEnv
from ttt_discover.rl.codex_no_finetune import (
    CodexNoFinetuneConfig,
    main as codex_no_finetune_main,
)


DEFAULT_INITIAL_POOL = Path(__file__).with_name(
    "initial_pool_reference_plus_codex_20260603.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Codex no-finetune Erdős minimum-overlap discovery."
    )
    parser.add_argument(
        "--experiment-name",
        default="erdos-codex-experimental",
        help="Experiment name used under tinker_log/.",
    )
    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--group-size", type=int, default=1)
    parser.add_argument("--groups-per-batch", type=int, default=1)
    parser.add_argument(
        "--wandb-project",
        default=os.environ.get("WANDB_PROJECT"),
        help="WANDB project name. Omit or pass '' to disable.",
    )
    parser.add_argument("--codex-model-name", default="gpt-5.5")
    parser.add_argument("--codex-cli-command", default="codex")
    parser.add_argument(
        "--codex-cli-sandbox",
        choices=("read-only", "workspace-write", "danger-full-access"),
        default="read-only",
    )
    parser.add_argument("--codex-cli-timeout", type=float, default=None)
    parser.add_argument("--codex-max-concurrent-requests", type=int, default=1)
    parser.add_argument("--codex-max-output-tokens", type=int, default=8192)
    parser.add_argument("--codex-temperature", type=float, default=None)
    parser.add_argument("--num-cpus-per-task", type=int, default=1)
    parser.add_argument(
        "--eval-timeout",
        type=int,
        default=500,
        help="Seconds passed to the Erdős candidate run(seed, budget_s=...).",
    )
    parser.add_argument(
        "--codex-initial-program",
        action="append",
        default=None,
        help="Additional seed program or construction file to verify and add to the sampler pool.",
    )
    parser.add_argument(
        "--codex-initial-pool",
        action="append",
        default=None,
        help="Additional reusable Codex initial-pool state JSON. May be repeated.",
    )
    parser.add_argument(
        "--no-default-initial-pool",
        action="store_true",
        help="Do not load repro/erdos/initial_pool_reference_plus_codex_20260603.json.",
    )
    parser.add_argument(
        "--log-root",
        default="tinker_log",
        help="Directory root for experiment logs.",
    )
    return parser.parse_args()


def default_initial_programs(args: argparse.Namespace) -> list[str]:
    return list(args.codex_initial_program or [])


def default_initial_pools(args: argparse.Namespace) -> list[str]:
    pools: list[str] = []
    if not args.no_default_initial_pool and DEFAULT_INITIAL_POOL.exists():
        pools.append(str(DEFAULT_INITIAL_POOL))
    pools.extend(args.codex_initial_pool or [])
    return pools


def main() -> None:
    args = parse_args()
    log_path = Path(args.log_root) / args.experiment_name

    cfg = CodexNoFinetuneConfig(
        env_type=ErdosMinOverlapEnv,
        problem_type="",
        backend="cli",
        model_name=args.codex_model_name,
        groups_per_batch=args.groups_per_batch,
        group_size=args.group_size,
        num_cpus_per_task=args.num_cpus_per_task,
        eval_timeout=args.eval_timeout,
        num_epochs=args.num_epochs,
        max_output_tokens=args.codex_max_output_tokens,
        temperature=args.codex_temperature,
        cli_command=args.codex_cli_command,
        cli_sandbox=args.codex_cli_sandbox,
        cli_timeout=args.codex_cli_timeout,
        max_concurrent_requests=args.codex_max_concurrent_requests,
        autonomous=False,
        initial_program_paths=tuple(default_initial_programs(args)),
        initial_pool_paths=tuple(default_initial_pools(args)),
        wandb_project=args.wandb_project,
        wandb_name=args.experiment_name,
        log_path=str(log_path),
        remove_constant_reward_groups=args.group_size > 1,
    )
    asyncio.run(codex_no_finetune_main(cfg))


if __name__ == "__main__":
    main()
