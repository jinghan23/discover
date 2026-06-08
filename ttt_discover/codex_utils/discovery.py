"""Codex-friendly top-level discovery wrapper."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Literal


@dataclass
class DiscoverConfig:
    model_name: str = "openai/gpt-oss-120b"
    runner: Literal["tinker_rl", "codex_no_finetune"] = "tinker_rl"
    lora_rank: int = 32
    renderer_name: str | None = "gpt_oss_high_reasoning"
    save_every: int = 2

    group_size: int = 64
    groups_per_batch: int = 8
    learning_rate: float = 4e-5
    num_epochs: int = 50
    temperature: float = 1.0
    kl_penalty_coef: float = 0.1
    phase1_max_tokens: int = 26000
    remove_constant_reward_groups: bool = True

    experiment_name: str | None = None
    wandb_project: str | None = "tinker-cookbook"

    env_type: type | None = None
    problem_type: str = "26"
    num_cpus_per_task: int = 0
    eval_timeout: int = 1000

    codex_backend: Literal["cli", "responses"] = "cli"
    codex_model_name: str | None = None
    codex_max_output_tokens: int = 8192
    codex_temperature: float | None = None
    codex_api_key_env: str = "OPENAI_API_KEY"
    codex_base_url: str | None = None
    codex_cli_command: str = "codex"
    codex_cli_sandbox: Literal[
        "read-only",
        "workspace-write",
        "danger-full-access",
    ] = "read-only"
    codex_cli_timeout: float | None = None
    codex_max_concurrent_requests: int | None = 4
    codex_initial_program_paths: tuple[str, ...] = ()
    codex_initial_pool_paths: tuple[str, ...] = ()
    codex_autonomous: bool = False


def _run_codex_no_finetune(config: DiscoverConfig) -> None:
    from ttt_discover.rl.codex_no_finetune import (
        CodexNoFinetuneConfig,
        main as codex_no_finetune_main,
    )

    if config.env_type is None:
        raise ValueError("env_type is required")
    experiment_name = config.experiment_name or "codex-no-finetune"
    log_path = f"./tinker_log/{experiment_name}"
    os.makedirs(log_path, exist_ok=True)

    codex_config = CodexNoFinetuneConfig(
        env_type=config.env_type,
        problem_type=config.problem_type,
        backend=config.codex_backend,
        model_name=config.codex_model_name,
        groups_per_batch=config.groups_per_batch,
        group_size=config.group_size,
        num_cpus_per_task=max(1, int(config.num_cpus_per_task)),
        eval_timeout=config.eval_timeout,
        num_epochs=config.num_epochs,
        max_output_tokens=config.codex_max_output_tokens,
        temperature=config.codex_temperature,
        api_key_env=config.codex_api_key_env,
        base_url=config.codex_base_url,
        cli_command=config.codex_cli_command,
        cli_sandbox=config.codex_cli_sandbox,
        cli_timeout=config.codex_cli_timeout,
        max_concurrent_requests=config.codex_max_concurrent_requests,
        initial_program_paths=config.codex_initial_program_paths,
        initial_pool_paths=config.codex_initial_pool_paths,
        autonomous=config.codex_autonomous,
        wandb_project=config.wandb_project,
        wandb_name=experiment_name,
        log_path=log_path,
        remove_constant_reward_groups=(
            config.remove_constant_reward_groups and config.group_size > 1
        ),
    )
    asyncio.run(codex_no_finetune_main(codex_config))


def discover(config: DiscoverConfig) -> None:
    if config.runner == "codex_no_finetune":
        _run_codex_no_finetune(config)
        return

    from ttt_discover.discovery import discover as tinker_discover

    tinker_discover(config)
