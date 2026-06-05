import asyncio
import logging
import os
from typing import Literal

import chz
from ttt_discover.environments.utils.cpu_scheduler import CpuScheduler
import ttt_discover.tinker_utils.misc_utils as misc_utils
from ttt_discover.rl.codex_no_finetune import (
    CodexNoFinetuneConfig,
    main as codex_no_finetune_main,
)

logger = logging.getLogger(__name__)


@chz.chz
class DiscoverConfig:
    """Simple config for discovery with RL training."""

    # Model config
    model_name: str = "openai/gpt-oss-120b"
    runner: Literal["tinker_rl", "codex_no_finetune"] = "tinker_rl"
    lora_rank: int = 32
    renderer_name: str | None = "gpt_oss_high_reasoning"
    save_every: int = 2

    # Training hyperparameters
    group_size: int = 64
    groups_per_batch: int = 8
    learning_rate: float = 4e-5
    num_epochs: int = 50
    temperature: float = 1.0
    kl_penalty_coef: float = 0.1
    phase1_max_tokens: int = 26000  # Two-phase sampling: total prompt + thinking token budget
    remove_constant_reward_groups: bool = True

    # Misc config
    experiment_name: str | None = None
    wandb_project: str | None = "tinker-cookbook"

    # Environment-specific
    env_type: type | None = None
    problem_type: str = "26"
    num_cpus_per_task: int = 0
    eval_timeout: int = 1000

    # Codex no-finetune config.
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


def init_ray(num_cpus_per_task: int, env_type: str):
    import ray

    if not ray.is_initialized():
        ray.init(
            num_cpus=max(2, int(num_cpus_per_task) + 1),
            include_dashboard=False,
            log_to_driver=False,
        )

    try:
        # Try to get existing actor by name
        _scheduler = ray.get_actor("cpu_scheduler")
        print("Found existing cpu_scheduler actor.")
    except ValueError:
        # If not found, create a new one
        print("Creating new cpu_scheduler actor.")
        _scheduler = CpuScheduler.options(
            name="cpu_scheduler",
            lifetime="detached",
        ).remote(
            num_cpus_per_task=num_cpus_per_task,
            num_persistent_workers=0,
        )


async def discover_impl(config: DiscoverConfig):
    """Convert discover config to full config and run training."""

    if config.env_type is None:
        raise ValueError("env_type is required")

    if config.runner == "tinker_rl":
        assert config.model_name in {
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
        }, "Only supporting GPT-OSS models for now."
    elif config.runner != "codex_no_finetune":
        raise ValueError(f"Unknown discovery runner: {config.runner}")

    # Ray is needed to dispatch jobs across cpus
    if config.num_cpus_per_task > 0:
        init_ray(config.num_cpus_per_task, config.env_type)

    logging.getLogger().handlers.clear()
    logging.getLogger().addHandler(logging.NullHandler())

    renderer_name = config.renderer_name
    model_name_for_tokenizer = config.model_name

    # create log path if it doesn't exist
    log_path = f"./tinker_log/{config.experiment_name}"
    log_file = os.path.join(log_path, "train.log")

    misc_utils.check_log_dir(log_path, behavior_if_exists="resume")
    os.makedirs(log_path, exist_ok=True)
    logging.basicConfig(level=logging.INFO, filename=log_file, filemode="a", force=True)
    logger.info("Logging to %s", log_file)

    if config.runner == "codex_no_finetune":
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
            wandb_name=config.experiment_name,
            log_path=log_path,
            remove_constant_reward_groups=(
                config.remove_constant_reward_groups and config.group_size > 1
            ),
        )
        await codex_no_finetune_main(codex_config)
        return

    from ttt_discover.rl.train import Config, main
    from ttt_discover.tinker_utils.dataset_builder import (
        DatasetConfig,
        get_single_problem_dataset_builder,
    )

    # Resolve env_name -> env type and build dataset
    dataset_config = DatasetConfig(
        env_type=config.env_type,
        problem_type=config.problem_type,
        batch_size=config.groups_per_batch,
        group_size=config.group_size,
        model_name_for_tokenizer=model_name_for_tokenizer,
        renderer_name=renderer_name,
        num_cpus_per_task=config.num_cpus_per_task,
        eval_timeout=config.eval_timeout,
        log_path=log_path,
    )
    dataset_builder = get_single_problem_dataset_builder(dataset_config)

    rl_config = Config(
        env_type=dataset_config.env_type,
        problem_type=config.problem_type,
        learning_rate=config.learning_rate,
        dataset_builder=dataset_builder,
        model_name=config.model_name,
        lora_rank=config.lora_rank,
        temperature=config.temperature,
        wandb_project=config.wandb_project,
        wandb_name=config.experiment_name,
        log_path=log_path,
        load_checkpoint_path=None,
        kl_penalty_coef=config.kl_penalty_coef,
        num_substeps=1,
        save_every=config.save_every,
        num_epochs=config.num_epochs,
        loss_fn="importance_sampling",
        adv_estimator="entropic_adaptive_beta",
        adv_estimator_beta=2.0, # Unused with entropic_adaptive_beta
        remove_constant_reward_groups=config.remove_constant_reward_groups,
        phase1_max_tokens=config.phase1_max_tokens,
        local_model_path=None,
    )

    # Run training
    await main(rl_config)
    
def discover(config: DiscoverConfig):
    asyncio.run(discover_impl(config))
