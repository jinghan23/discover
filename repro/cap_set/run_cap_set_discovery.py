from __future__ import annotations

import argparse
import os
from pathlib import Path

from examples.cap_set_priority.env import CapSetPriorityEnv
from ttt_discover import DiscoverConfig, discover


DEFAULT_INITIAL_POOL = Path(__file__).with_name("initial_pool_400_to_512.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch the TTT-Discover cap-set priority-function task."
    )
    parser.add_argument("--dimension", type=int, default=8)
    parser.add_argument(
        "--experiment-name",
        default="cap-set-priority-repro",
        help="Experiment name used for local logs and WANDB.",
    )
    parser.add_argument(
        "--wandb-project",
        default=os.environ.get("WANDB_PROJECT", "cap-set-priority"),
        help="WANDB project name. Pass '' to disable in local smoke runs.",
    )
    parser.add_argument(
        "--model-name",
        choices=("openai/gpt-oss-120b", "openai/gpt-oss-20b"),
        default="openai/gpt-oss-120b",
        help="GPT-OSS model served through Tinker.",
    )
    parser.add_argument(
        "--runner",
        choices=("tinker_rl", "codex_no_finetune"),
        default="codex_no_finetune",
        help="Use Tinker RL fine-tuning or Codex sampling without fine-tuning.",
    )
    parser.add_argument(
        "--codex-model-name",
        default=None,
        help="Codex model override for --runner codex_no_finetune.",
    )
    parser.add_argument(
        "--codex-backend",
        choices=("cli", "responses"),
        default="cli",
        help="Use local Codex CLI account or direct OpenAI Responses API.",
    )
    parser.add_argument("--codex-max-output-tokens", type=int, default=8192)
    parser.add_argument("--codex-temperature", type=float, default=None)
    parser.add_argument("--codex-cli-timeout", type=float, default=None)
    parser.add_argument(
        "--codex-autonomous",
        action="store_true",
        help="Let Codex run the local evaluator in a workspace before returning code.",
    )
    parser.add_argument("--codex-max-concurrent-requests", type=int, default=4)
    parser.add_argument(
        "--codex-initial-program",
        action="append",
        default=None,
        help=(
            "Additional seed priority program to verify and add to the Codex "
            "sampler pool."
        ),
    )
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--group-size", type=int, default=1)
    parser.add_argument("--groups-per-batch", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=4e-5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--kl-penalty-coef", type=float, default=0.1)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--save-every", type=int, default=2)
    parser.add_argument("--phase1-max-tokens", type=int, default=26000)
    parser.add_argument("--num-cpus-per-task", type=int, default=1)
    parser.add_argument("--eval-timeout", type=int, default=45)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    initial_programs = list(args.codex_initial_program or [])
    discovery_cpus = args.num_cpus_per_task
    if args.runner == "codex_no_finetune":
        discovery_cpus = 0
    codex_cli_sandbox = "workspace-write" if args.codex_autonomous else "read-only"

    config = DiscoverConfig(
        env_type=CapSetPriorityEnv,
        problem_type=str(args.dimension),
        model_name=args.model_name,
        runner=args.runner,
        lora_rank=args.lora_rank,
        group_size=args.group_size,
        groups_per_batch=args.groups_per_batch,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        temperature=args.temperature,
        kl_penalty_coef=args.kl_penalty_coef,
        phase1_max_tokens=args.phase1_max_tokens,
        save_every=args.save_every,
        num_cpus_per_task=discovery_cpus,
        eval_timeout=args.eval_timeout,
        experiment_name=args.experiment_name,
        wandb_project=args.wandb_project,
        codex_model_name=args.codex_model_name,
        codex_backend=args.codex_backend,
        codex_max_output_tokens=args.codex_max_output_tokens,
        codex_temperature=args.codex_temperature,
        codex_cli_sandbox=codex_cli_sandbox,
        codex_cli_timeout=args.codex_cli_timeout,
        codex_max_concurrent_requests=args.codex_max_concurrent_requests,
        codex_initial_program_paths=tuple(initial_programs),
        codex_initial_pool_paths=(str(DEFAULT_INITIAL_POOL),),
        codex_autonomous=args.codex_autonomous,
    )
    discover(config)


if __name__ == "__main__":
    main()
