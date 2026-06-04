"""
Sampling-only discovery loop backed by OpenAI Codex.

This runner keeps the TTT-Discover environment, reward, PUCT sampler, and
logging stack, but deliberately skips all Tinker training and checkpointing.
"""

import asyncio
import glob
import json
import logging
import os
import re
import shlex
import time
import uuid
from pathlib import Path
from typing import Any, Literal, Sequence

import chz
import wandb

from ttt_discover.codex_utils import ml_log
from ttt_discover.codex_utils.completers import CodexCliTokenCompleter, CodexTokenCompleter
from ttt_discover.codex_utils.misc_utils import Tokenizer, all_same, get_tokenizer, timed
from ttt_discover.codex_utils.ml_log import WandbLogger
from ttt_discover.codex_utils.runtime import (
    EnvGroupBuilder,
    RLDataset,
    RLDatasetBuilder,
    TrajectoryGroup,
    TokensWithLogprobs,
    compute_trajectory_metrics,
    do_group_rollout,
)

logger = logging.getLogger(__name__)


def append_agent_outputs(log_path: str, step: int, table_data: list[tuple[Any, ...]]) -> None:
    """Persist raw agent rollouts locally."""
    os.makedirs(log_path, exist_ok=True)
    output_path = os.path.join(log_path, "agent_outputs.jsonl")
    columns = [
        "prompt",
        "response",
        "reward",
        "correctness",
        "parsed_code",
        "msg",
        "initial_raw_score",
        "advantage",
    ]
    with open(output_path, "a", encoding="utf-8") as f:
        for row_idx, row in enumerate(table_data):
            entry = {"step": step, "row": row_idx}
            entry.update({key: value for key, value in zip(columns, row, strict=False)})
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


@chz.chz
class CodexNoFinetuneConfig:
    env_type: type
    problem_type: str
    dataset_builder: RLDatasetBuilder
    backend: Literal["cli", "responses"] = "cli"
    model_name: str | None = None
    tokenizer_model_name: str = "openai/gpt-oss-20b"
    num_cpus_per_task: int = 1
    eval_timeout: int = 45
    num_epochs: int = 1
    max_output_tokens: int = 8192
    temperature: float | None = None
    api_key_env: str = "OPENAI_API_KEY"
    base_url: str | None = None
    cli_command: str = "codex"
    cli_sandbox: Literal["read-only", "workspace-write", "danger-full-access"] = "read-only"
    cli_timeout: float | None = None
    max_concurrent_requests: int | None = 4

    autonomous: bool = False

    wandb_project: str | None = None
    wandb_name: str | None = None

    log_path: str = chz.field(munger=lambda _, s: os.path.expanduser(s))
    remove_constant_reward_groups: bool = False


def _latest_sampler_step(log_path: str) -> int:
    pattern = os.path.join(log_path, "puct_sampler_step_*.json")
    latest = 0
    for path in glob.glob(pattern):
        match = re.search(r"puct_sampler_step_(\d+)\.json$", path)
        if match:
            latest = max(latest, int(match.group(1)))
    return latest


class AutonomousCodexCliTokenCompleter(CodexCliTokenCompleter):
    """Minimal Codex CLI agent mode: give Codex a workspace and evaluator."""

    def __init__(
        self,
        *,
        tokenizer: Tokenizer,
        log_path: str,
        problem_type: str,
        step_idx: int,
        eval_timeout: int,
        num_cpus_per_task: int,
        **kwargs: Any,
    ):
        super().__init__(tokenizer=tokenizer, **kwargs)
        self.log_path = log_path
        self.problem_type = problem_type
        self.step_idx = step_idx
        self.eval_timeout = eval_timeout
        self.num_cpus_per_task = num_cpus_per_task
        self._call_idx = 0

    def _next_workspace(self) -> Path:
        self._call_idx += 1
        root = Path(self.log_path) / "codex_autonomous_workspaces"
        workspace = (
            root
            / f"step_{max(0, self.step_idx):06d}"
            / f"call_{self._call_idx:04d}_{uuid.uuid4().hex[:8]}"
        )
        workspace.mkdir(parents=True, exist_ok=True)
        (workspace / "eval_tmp").mkdir(parents=True, exist_ok=True)
        return workspace

    def _build_prompt(self, prompt_text: str) -> str:
        workspace = self._next_workspace()
        self._workspace = workspace
        request_input = CodexTokenCompleter(tokenizer=self.tokenizer)._build_input(prompt_text)
        if isinstance(request_input, str):
            base_prompt = request_input
        else:
            lines: list[str] = []
            for message in request_input:
                role = message["role"].upper()
                lines.append(f"{role}:\n{message['content']}")
            base_prompt = "\n\n".join(lines)

        matches = re.findall(r"```python\s+([\s\S]*?)\s*```", base_prompt or "")
        parent_matches = [match for match in matches if "def priority(" in match]
        if not parent_matches:
            parent_source = "def priority(el, n):\n    return 0.0\n"
        else:
            parent_source = parent_matches[-1].strip() + "\n"
        (workspace / "candidate.py").write_text(parent_source, encoding="utf-8")
        (workspace / "prompt.txt").write_text(base_prompt, encoding="utf-8")

        candidate = shlex.quote(str(workspace / "candidate.py"))
        eval_dir = shlex.quote(str(workspace / "eval_tmp"))
        evaluator_cmd = (
            ".venv/bin/python -m repro.cap_set.self_loop_eval "
            f"--candidate {candidate} "
            f"--dimension {shlex.quote(str(self.problem_type))} "
            f"--log-dir {eval_dir} "
            f"--eval-timeout {int(self.eval_timeout)} "
            f"--num-cpus-per-task {int(self.num_cpus_per_task)}"
        )
        prompt = f"""{base_prompt}

--- Autonomous Codex Search Mode ---
You may edit files and run shell commands, but only inside this workspace:
{workspace}

Network IO is not allowed. The current candidate is:
{workspace / "candidate.py"}

Run this evaluator after each revision:
{evaluator_cmd}

When done, save the best code to:
{workspace / "best_priority.py"}

Finish with exactly one Python code block defining that best `priority(el, n)`.
"""
        (workspace / "prompt.txt").write_text(prompt, encoding="utf-8")
        return prompt

    async def _call_unlocked(self, model_input, stop) -> TokensWithLogprobs:
        prompt = self._build_prompt(self._decode_model_input(model_input))
        workspace = self._workspace
        output_path = workspace / "final_response.txt"

        cmd = [
            self.codex_command,
            "exec",
            "--ephemeral",
            "--sandbox",
            self.sandbox,
            "-o",
            str(output_path),
        ]
        if self.cwd:
            cmd.extend(["-C", self.cwd])
        if self.ignore_user_config:
            cmd.append("--ignore-user-config")
        if self.ignore_rules:
            cmd.append("--ignore-rules")
        cmd.extend(["-m", self.model_name or "gpt-5.5"])
        if self.reasoning_effort:
            cmd.extend([
                "-c",
                f"model_reasoning_effort={json.dumps(self.reasoning_effort)}",
            ])
        cmd.append("-")

        (workspace / "command.json").write_text(json.dumps(cmd, indent=2), encoding="utf-8")
        stdout_log_path = workspace / "codex.stdout.log"
        stderr_log_path = workspace / "codex.stderr.log"
        with stdout_log_path.open("wb") as stdout_log, stderr_log_path.open("wb") as stderr_log:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=stdout_log,
                stderr=stderr_log,
            )
            communicate = process.communicate(prompt.encode("utf-8"))
            try:
                if self.timeout is not None:
                    await asyncio.wait_for(communicate, timeout=self.timeout)
                else:
                    await communicate
            except asyncio.TimeoutError as exc:
                if process.returncode is None:
                    try:
                        process.kill()
                    except ProcessLookupError:
                        pass
                await process.wait()
                stdout_log.flush()
                stderr_log.flush()
                stdout_text = stdout_log_path.read_bytes().decode(errors="replace")
                stderr_text = stderr_log_path.read_bytes().decode(errors="replace")
                raise RuntimeError(
                    "codex exec timed out after "
                    f"{self.timeout}s; workspace={workspace}\n"
                    f"STDOUT:\n{stdout_text}\nSTDERR:\n{stderr_text}"
                ) from exc
            except asyncio.CancelledError:
                if process.returncode is None:
                    try:
                        process.kill()
                    except ProcessLookupError:
                        pass
                await process.wait()
                raise
            finally:
                stdout_log.flush()
                stderr_log.flush()

        stdout_text = stdout_log_path.read_bytes().decode(errors="replace")
        stderr_text = stderr_log_path.read_bytes().decode(errors="replace")

        if process.returncode != 0:
            raise RuntimeError(
                "codex exec failed with exit code "
                f"{process.returncode}; workspace={workspace}\n"
                f"STDOUT:\n{stdout_text}\nSTDERR:\n{stderr_text}"
            )

        text = output_path.read_text(encoding="utf-8").strip() if output_path.exists() else ""
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        return TokensWithLogprobs(
            tokens=self._append_stop_sequence(tokens, stop),
            maybe_logprobs=None,
        )


async def do_group_rollout_with_codex(
    tokenizer: Tokenizer,
    env_group_builder: EnvGroupBuilder,
    backend: Literal["cli", "responses"],
    model_name: str | None,
    max_output_tokens: int,
    temperature: float | None,
    api_key_env: str,
    base_url: str | None,
    cli_command: str,
    cli_sandbox: Literal["read-only", "workspace-write", "danger-full-access"],
    cli_timeout: float | None,
    autonomous: bool,
    log_path: str,
    problem_type: str,
    eval_timeout: int,
    num_cpus_per_task: int,
    cwd: str,
    semaphore: asyncio.Semaphore | None,
    step_idx: int = -1,
) -> TrajectoryGroup:
    if backend == "cli":
        if autonomous:
            if cli_sandbox == "read-only":
                raise ValueError(
                    "Codex autonomous mode requires --codex-cli-sandbox workspace-write "
                    "or danger-full-access so Codex can write candidates."
                )
            policy = AutonomousCodexCliTokenCompleter(
                tokenizer=tokenizer,
                model_name=model_name,
                codex_command=cli_command,
                sandbox=cli_sandbox,
                cwd=cwd,
                timeout=cli_timeout,
                semaphore=semaphore,
                log_path=log_path,
                problem_type=problem_type,
                step_idx=step_idx,
                eval_timeout=eval_timeout,
                num_cpus_per_task=num_cpus_per_task,
            )
        else:
            policy = CodexCliTokenCompleter(
                tokenizer=tokenizer,
                model_name=model_name,
                codex_command=cli_command,
                sandbox=cli_sandbox,
                cwd=cwd,
                timeout=cli_timeout,
                semaphore=semaphore,
            )
    elif backend == "responses":
        if autonomous:
            raise ValueError("Codex autonomous mode requires --codex-backend cli.")
        policy = CodexTokenCompleter(
            tokenizer=tokenizer,
            model_name=model_name,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            api_key_env=api_key_env,
            base_url=base_url,
            semaphore=semaphore,
        )
    else:
        raise ValueError(f"Unknown Codex backend: {backend}")
    return await do_group_rollout(env_group_builder, policy, step_idx)


async def sample_batch(
    cfg: CodexNoFinetuneConfig,
    tokenizer: Tokenizer,
    i_batch: int,
    env_group_builders: Sequence[EnvGroupBuilder],
) -> tuple[list[TrajectoryGroup], dict[str, Any]]:
    metrics: dict[str, Any] = {}
    semaphore = (
        asyncio.Semaphore(cfg.max_concurrent_requests)
        if cfg.max_concurrent_requests is not None
        else None
    )
    with timed("sampling", metrics):
        trajectory_groups = await asyncio.gather(
            *[
                asyncio.create_task(
                    do_group_rollout_with_codex(
                        tokenizer=tokenizer,
                        env_group_builder=builder,
                        backend=cfg.backend,
                        model_name=cfg.model_name,
                        max_output_tokens=cfg.max_output_tokens,
                        temperature=cfg.temperature,
                        api_key_env=cfg.api_key_env,
                        base_url=cfg.base_url,
                        cli_command=cfg.cli_command,
                        cli_sandbox=cfg.cli_sandbox,
                        cli_timeout=cfg.cli_timeout,
                        autonomous=cfg.autonomous,
                        log_path=cfg.log_path,
                        problem_type=cfg.problem_type,
                        eval_timeout=cfg.eval_timeout,
                        num_cpus_per_task=max(1, int(cfg.num_cpus_per_task)),
                        cwd=os.getcwd(),
                        semaphore=semaphore,
                        step_idx=i_batch,
                    ),
                    name=f"codex_sample_task_{i}",
                )
                for i, builder in enumerate(env_group_builders)
            ],
            return_exceptions=True,
        )

    kept_pairs: list[tuple[EnvGroupBuilder, TrajectoryGroup]] = []
    failed_groups = 0
    for env_group_builder, trajectory_group in zip(
        env_group_builders,
        trajectory_groups,
        strict=True,
    ):
        if isinstance(trajectory_group, BaseException):
            failed_groups += 1
            logger.warning(
                "Skipping failed Codex rollout group at step %s: %r",
                i_batch,
                trajectory_group,
            )
            continue
        if trajectory_group is None:
            continue
        if cfg.remove_constant_reward_groups and all_same(trajectory_group.get_total_rewards()):
            continue
        kept_pairs.append((env_group_builder, trajectory_group))

    metrics["codex/failed_groups"] = failed_groups
    metrics["codex/succeeded_groups"] = len(kept_pairs)
    trajectory_groups = [trajectory_group for _, trajectory_group in kept_pairs]
    taglist = [env_group_builder.logging_tags() for env_group_builder, _ in kept_pairs]
    if trajectory_groups:
        metrics.update(compute_trajectory_metrics(trajectory_groups, taglist))
    else:
        metrics["env/all/total_episodes"] = 0
    return trajectory_groups, metrics


def _log_agent_tables(
    cfg: CodexNoFinetuneConfig,
    ml_logger: ml_log.Logger,
    i_batch: int,
    metrics: dict[str, Any],
) -> None:
    table_data = metrics.pop("table", None)
    if table_data is None:
        return

    table_data_with_advantage = [(*row, None) for row in table_data]
    append_agent_outputs(cfg.log_path, i_batch, table_data_with_advantage)

    if not hasattr(ml_logger, "loggers"):
        return
    for logger_instance in ml_logger.loggers:
        if isinstance(logger_instance, WandbLogger):
            logger_instance.log_metrics(
                {
                    f"gen&score_codex_{i_batch}": wandb.Table(
                        columns=[
                            "Prompt",
                            "Gen Sequence",
                            "Reward",
                            "Correctness",
                            "Gen Sequence PostProc",
                            "Message",
                            "Initial Raw Score",
                            "Advantage",
                        ],
                        data=table_data_with_advantage,
                    )
                },
                step=i_batch,
            )
            break


def _log_sampler_table(
    ml_logger: ml_log.Logger,
    i_batch: int,
    sampler_table_columns: list[str] | None,
    sampler_table_data: list[tuple] | None,
) -> None:
    if sampler_table_columns is None or sampler_table_data is None:
        return
    if not hasattr(ml_logger, "loggers"):
        return
    for logger_instance in ml_logger.loggers:
        if isinstance(logger_instance, WandbLogger):
            logger_instance.log_metrics(
                {
                    f"sampler_states_{i_batch}": wandb.Table(
                        columns=sampler_table_columns,
                        data=sampler_table_data,
                    )
                },
                step=i_batch,
            )
            break


async def do_sampling_only(
    start_batch: int,
    end_batch: int,
    num_batches: int,
    cfg: CodexNoFinetuneConfig,
    dataset: RLDataset,
    ml_logger: ml_log.Logger,
    tokenizer: Tokenizer,
) -> None:
    num_batches_per_epoch = len(dataset)
    if num_batches_per_epoch == 0:
        raise ValueError("RLDataset must contain at least one batch")

    for i_batch in range(start_batch, end_batch):
        metrics: dict[str, Any] = {
            "progress/batch": i_batch,
            "progress/done_frac": (i_batch + 1) / num_batches,
        }
        t_start = time.time()

        print("Load dataset batch...")
        dataset_batch_idx = i_batch % num_batches_per_epoch
        env_group_builders = dataset.get_batch(dataset_batch_idx)

        print("Log sampler stats...")
        sampler_table_columns, sampler_table_data = None, None
        if hasattr(dataset, "sampler") and hasattr(dataset.sampler, "get_sample_stats"):
            metrics.update(dataset.sampler.get_sample_stats())
            if hasattr(dataset.sampler, "get_sample_table"):
                sampler_table_columns, sampler_table_data = dataset.sampler.get_sample_table()

        print("Sampling with Codex...")
        _trajectory_groups, sampling_metrics = await sample_batch(
            cfg,
            tokenizer,
            i_batch,
            env_group_builders,
        )
        metrics.update(sampling_metrics)

        if hasattr(dataset, "flush"):
            dataset.flush(step=i_batch + 1)

        _log_agent_tables(cfg, ml_logger, i_batch, metrics)
        _log_sampler_table(ml_logger, i_batch, sampler_table_columns, sampler_table_data)

        metrics["time/total"] = time.time() - t_start
        ml_logger.log_metrics(metrics, step=i_batch)


async def main(cfg: CodexNoFinetuneConfig) -> None:
    """Main sampling-only loop for Codex-backed discovery."""
    if cfg.num_epochs < 1:
        raise ValueError("num_epochs must be >= 1")

    ml_logger = ml_log.setup_logging(
        log_dir=cfg.log_path,
        wandb_project=cfg.wandb_project,
        config=cfg,
        wandb_name=cfg.wandb_name,
    )

    tokenizer = get_tokenizer(cfg.tokenizer_model_name)

    print("Create dataset...")
    dataset = await cfg.dataset_builder()
    print("Dataset created!")

    start_batch = _latest_sampler_step(cfg.log_path)
    if (
        start_batch > 0
        and hasattr(dataset, "sampler")
        and hasattr(dataset.sampler, "reload_from_step")
    ):
        logger.info("Reloading sampler state from step %s", start_batch)
        dataset.sampler.reload_from_step(start_batch)

    num_batches_per_epoch = len(dataset)
    if num_batches_per_epoch == 0:
        raise ValueError("RLDataset must contain at least one batch")
    num_batches_total = num_batches_per_epoch * cfg.num_epochs
    logger.info(
        "Will sample for %s epoch(s) x %s batches = %s steps",
        cfg.num_epochs,
        num_batches_per_epoch,
        num_batches_total,
    )

    if start_batch < num_batches_total:
        await do_sampling_only(
            start_batch=start_batch,
            end_batch=num_batches_total,
            num_batches=num_batches_total,
            cfg=cfg,
            dataset=dataset,
            ml_logger=ml_logger,
            tokenizer=tokenizer,
        )
    else:
        logger.info("Sampling-only run was already complete; nothing to do")

    ml_logger.close()
    logger.info("Codex no-finetune discovery completed successfully")
