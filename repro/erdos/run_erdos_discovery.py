from __future__ import annotations

import argparse
import os

from examples.erdos_min_overlap.env import ErdosMinOverlapEnv
from ttt_discover import DiscoverConfig, discover


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch the TTT-Discover Erdős minimum-overlap task."
    )
    parser.add_argument(
        "--experiment-name",
        default="erdos-min-overlap-repro",
        help="Experiment name used for tinker_log and WANDB.",
    )
    parser.add_argument(
        "--wandb-project",
        default=os.environ.get("WANDB_PROJECT", "erdos-min-overlap"),
        help="WANDB project name.",
    )
    parser.add_argument(
        "--model-name",
        choices=("openai/gpt-oss-120b", "openai/gpt-oss-20b"),
        default="openai/gpt-oss-120b",
        help="GPT-OSS model served through Tinker.",
    )
    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--groups-per-batch", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=4e-5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--kl-penalty-coef", type=float, default=0.1)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--save-every", type=int, default=2)
    parser.add_argument("--phase1-max-tokens", type=int, default=26000)
    parser.add_argument("--num-cpus-per-task", type=int, default=1)
    parser.add_argument("--eval-timeout", type=int, default=1100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DiscoverConfig(
        env_type=ErdosMinOverlapEnv,
        problem_type="",
        model_name=args.model_name,
        lora_rank=args.lora_rank,
        group_size=args.group_size,
        groups_per_batch=args.groups_per_batch,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        temperature=args.temperature,
        kl_penalty_coef=args.kl_penalty_coef,
        phase1_max_tokens=args.phase1_max_tokens,
        save_every=args.save_every,
        num_cpus_per_task=args.num_cpus_per_task,
        eval_timeout=args.eval_timeout,
        experiment_name=args.experiment_name,
        wandb_project=args.wandb_project,
    )
    discover(config)


if __name__ == "__main__":
    main()
