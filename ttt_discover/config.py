"""Configuration for the discovery runtime."""

from __future__ import annotations

from typing import Any, Literal

import chz

AlgorithmName = Literal["ttt_discover", "autoevolve", "openevolve", "abmcts"]
EvalRunnerName = Literal["auto", "in_process", "blackbox"]
BackendName = Literal["cli", "responses"]
CliSandboxName = Literal["read-only", "workspace-write", "danger-full-access"]
ABMCTSStrategyName = Literal[
    "stack",
    "multiarm_bandit_thompson",
    "multiarm_bandit_ucb",
]
ABMCTSDistName = Literal["gaussian", "beta"]
ABMCTSVariantName = Literal["a", "m"]


@chz.chz
class DiscoverConfig:
    """Single config object for Codex-backed discovery."""

    # Run identity
    experiment_name: str | None = None
    log_path: str = ""

    # Task / environment
    env_type: type | None = None
    problem_type: str = ""
    num_cpus_per_task: int = 1
    eval_timeout: int = 45
    timeout: float = 8000.0

    # Algorithm / search
    algorithm: AlgorithmName = "ttt_discover"
    initial_state_file: str | None = None
    num_epochs: int = 1
    groups_per_batch: int = 1
    group_size: int = 1
    inner_iterations: int = 1
    max_concurrent_requests: int | None = 4
    topk_children: int = 16
    remove_constant_reward_groups: bool = False

    # AB-MCTS local sampler
    abmcts_variant: ABMCTSVariantName = "a"
    abmcts_actions: Any = ("default",)
    abmcts_dist_type: ABMCTSDistName = "gaussian"
    abmcts_model_selection_strategy: ABMCTSStrategyName = "multiarm_bandit_thompson"
    abmcts_invalid_score: float = 0.0
    abmcts_prior_mean: float = 0.0
    abmcts_prior_std: float = 1.0
    abmcts_prior_strength: float = 1.0
    abmcts_beta_a: float = 0.5
    abmcts_beta_b: float = 0.5
    abmcts_seed: int | None = None

    # OpenEvolve adapter
    openevolve_initial_program: str | None = None
    openevolve_config_path: str | None = None
    openevolve_output_dir: str | None = None
    openevolve_checkpoint_interval: int | None = None
    openevolve_max_code_length: int | None = None
    openevolve_save_db: bool = True
    openevolve_export_history: bool = True
    openevolve_trace: bool = False
    openevolve_trace_format: str = "jsonl"
    openevolve_oe: dict[str, Any] | None = None

    # Codex backend
    backend: BackendName = "cli"
    model_name: str | None = None
    max_output_tokens: int | None = 8192
    temperature: float | None = None
    api_key_env: str = "OPENAI_API_KEY"
    base_url: str | None = None
    cli_command: str = "codex"
    cli_sandbox: CliSandboxName = "read-only"
    cli_timeout: float | None = None

    # Evaluation runner
    eval_runner: EvalRunnerName = "auto"
    blackbox_eval_socket: str | None = None
    blackbox_eval_host: str = "127.0.0.1"
    blackbox_eval_port: int | None = None

    # Logging
    wandb_project: str | None = "tinker-cookbook"
    wandb_name: str | None = None

    # Variants (see ttt_discover/algorithms/variants/): opt-in algorithm hooks
    # applied by stable hook phases. Adding a variant never edits this schema --
    # only enable it here per run, e.g.
    # variants=("reward_shaping",),
    # variant_params={"reward_shaping": {"threshold_start": 0.01, ...}}.
    variants: tuple[str, ...] = ()
    variant_params: dict[str, Any] | None = None

    def to_runtime_config(
        self,
        *,
        log_path: str | None = None,
        wandb_name: str | None = None,
    ) -> "DiscoverConfig":
        kwargs = chz.asdict(self)
        kwargs.update(
            num_cpus_per_task=max(1, int(self.num_cpus_per_task)),
            inner_iterations=max(1, int(self.inner_iterations)),
            remove_constant_reward_groups=(
                bool(self.remove_constant_reward_groups) and int(self.group_size) > 1
            ),
        )
        if log_path is not None:
            kwargs["log_path"] = log_path
        if wandb_name is not None:
            kwargs["wandb_name"] = wandb_name

        return type(self)(**kwargs)


DISCOVER_CONFIG_FIELDS = tuple(DiscoverConfig.__annotations__)


__all__ = [
    "AlgorithmName",
    "ABMCTSDistName",
    "ABMCTSStrategyName",
    "ABMCTSVariantName",
    "BackendName",
    "CliSandboxName",
    "DISCOVER_CONFIG_FIELDS",
    "DiscoverConfig",
    "EvalRunnerName",
]
