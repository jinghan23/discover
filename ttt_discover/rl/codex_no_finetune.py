"""Sampling-only discovery loop backed by Codex.

This runner keeps the TTT-Discover task hooks, reward functions, PUCT sampler,
and local metrics, but does not use RL dataset, trajectory, or trainer
abstractions.
"""

from __future__ import annotations

import asyncio
import glob
import json
import logging
import os
import re
import shlex
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Literal

import chz
import numpy as np

from ttt_discover.codex_utils.completers import (
    CodexCliCompleter,
    CodexResponseCompleter,
    TextCompleter,
    _kill_process_tree,
)
from ttt_discover.codex_utils.runtime import to_json_serializable
from ttt_discover.codex_utils.sampler import (
    StateSampler,
    create_sampler,
    seed_initial_pool_paths,
    seed_initial_program_paths,
)

logger = logging.getLogger(__name__)

SAFE_GRADE_EXECUTOR = ThreadPoolExecutor(max_workers=4096)


@dataclass
class VerifyResult:
    reward: float
    msg: str
    correctness: float
    raw_score: float
    result_construction: Any
    stdout: str
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class CandidateResult:
    parent_state: Any
    group_idx: int
    sample_idx: int
    prompt: str
    response: str
    parsed_code: str
    reward: float
    correctness: float
    raw_score: float | None
    msg: str
    metrics: dict[str, Any]
    next_state: Any | None = None
    error: str | None = None
    kept: bool = True
    drop_reason: str | None = None
    pool_status: str | None = None
    sampler_error: str | None = None


@chz.chz
class CodexNoFinetuneConfig:
    env_type: type
    problem_type: str = ""
    backend: Literal["cli", "responses"] = "cli"
    model_name: str | None = None
    groups_per_batch: int = 1
    group_size: int = 1
    num_cpus_per_task: int = 1
    eval_timeout: int = 45
    timeout: float = 8000.0
    num_epochs: int = 1
    max_output_tokens: int = 8192
    temperature: float | None = None
    api_key_env: str = "OPENAI_API_KEY"
    base_url: str | None = None
    cli_command: str = "codex"
    cli_sandbox: Literal["read-only", "workspace-write", "danger-full-access"] = "read-only"
    cli_timeout: float | None = None
    max_concurrent_requests: int | None = 4
    initial_program_paths: tuple[str, ...] = ()
    initial_pool_paths: tuple[str, ...] = ()
    topk_children: int = 16

    autonomous: bool = False

    wandb_project: str | None = None
    wandb_name: str | None = None

    log_path: str = ""
    remove_constant_reward_groups: bool = False


class MetricsLogger:
    def __init__(
        self,
        *,
        log_path: str,
        wandb_project: str | None,
        wandb_name: str | None,
        config: Any,
    ):
        self.log_path = log_path
        os.makedirs(log_path, exist_ok=True)
        self.metrics_path = os.path.join(log_path, "metrics.jsonl")
        self._wandb = None
        self._run = None
        if wandb_project:
            import wandb

            self._wandb = wandb
            self._run = wandb.init(
                project=wandb_project,
                name=wandb_name,
                config=_config_for_logging(config),
            )

    def log_metrics(self, metrics: dict[str, Any], step: int) -> None:
        row = {"step": step, **to_json_serializable(metrics)}
        with open(self.metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

        if self._wandb is None:
            return
        wandb_metrics = {
            k: v
            for k, v in metrics.items()
            if isinstance(v, (int, float, str, bool)) or v is None
        }
        if wandb_metrics:
            self._wandb.log(wandb_metrics, step=step)

    def log_table(
        self,
        name: str,
        *,
        columns: list[str],
        data: list[tuple[Any, ...]],
        step: int,
    ) -> None:
        table_path = os.path.join(self.log_path, f"{name}.jsonl")
        with open(table_path, "a", encoding="utf-8") as f:
            for row_idx, row in enumerate(data):
                entry = {"step": step, "row": row_idx}
                entry.update({key: value for key, value in zip(columns, row, strict=False)})
                f.write(json.dumps(to_json_serializable(entry), ensure_ascii=False) + "\n")

        if self._wandb is not None:
            self._wandb.log(
                {f"{name}_{step}": self._wandb.Table(columns=columns, data=data)},
                step=step,
            )

    def close(self) -> None:
        if self._run is not None:
            self._run.finish()


def _config_for_logging(config: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in vars(config).items():
        if key.startswith("X_"):
            continue
        if isinstance(value, type):
            out[key] = f"{value.__module__}.{value.__name__}"
        else:
            out[key] = to_json_serializable(value)
    return out


@contextmanager
def timed(key: str, metrics: dict[str, Any]):
    start = time.time()
    yield
    metrics[f"time/{key}"] = time.time() - start


def all_same(xs: list[Any]) -> bool:
    return bool(xs) and all(x == xs[0] for x in xs)


def append_agent_outputs(log_path: str, step: int, results: list[CandidateResult]) -> None:
    os.makedirs(log_path, exist_ok=True)
    output_path = os.path.join(log_path, "agent_outputs.jsonl")
    with open(output_path, "a", encoding="utf-8") as f:
        for row_idx, result in enumerate(results):
            parent_state = result.parent_state
            next_state = result.next_state
            entry = {
                "step": step,
                "row": row_idx,
                "group_idx": result.group_idx,
                "sample_idx": result.sample_idx,
                "kept": result.kept,
                "drop_reason": result.drop_reason,
                "parent_id": getattr(parent_state, "id", None),
                "parent_timestep": getattr(parent_state, "timestep", None),
                "parent_value": getattr(parent_state, "value", None),
                "parent_construction_len": (
                    len(parent_state.construction)
                    if getattr(parent_state, "construction", None) is not None
                    else None
                ),
                "next_state_id": getattr(next_state, "id", None),
                "next_state_timestep": getattr(next_state, "timestep", None),
                "next_state_value": getattr(next_state, "value", None),
                "pool_status": result.pool_status,
                "prompt": result.prompt,
                "response": result.response,
                "reward": result.reward,
                "correctness": result.correctness,
                "parsed_code": result.parsed_code,
                "msg": result.msg,
                "initial_raw_score": result.metrics.get(
                    "initial_raw_score", getattr(parent_state, "value", None)
                ),
                "raw_score": result.raw_score,
                "advantage": None,
                "error": result.error,
                "sampler_error": result.sampler_error,
                "metrics": result.metrics,
            }
            f.write(json.dumps(to_json_serializable(entry), ensure_ascii=False) + "\n")


def last_codeblock_postprocess(
    input_text: str,
    codeblock_seps: list[str] | tuple[str, ...] = ("python", "cpp", "java", "cuda"),
    last_response_strict: bool = True,
    keep_separators: bool = True,
):
    languages_pattern = "|".join(map(re.escape, codeblock_seps))
    pattern = re.compile(
        f"```({languages_pattern})" + r"\n(?!```)(.*?)(?:\n```)?(?=\n```|$)",
        re.DOTALL,
    )
    matches = list(pattern.finditer(input_text))

    if matches:
        last_match = matches[-1]
        language = last_match.group(1)
        code_content = last_match.group(2).rstrip()
        if not code_content or code_content.strip() == "":
            return "" if last_response_strict else input_text
        if keep_separators:
            return f"```{language}\n{code_content}\n```"
        return code_content

    return "" if last_response_strict else input_text


def _latest_sampler_step(log_path: str) -> int:
    pattern = os.path.join(log_path, "puct_sampler_step_*.json")
    latest = 0
    for path in glob.glob(pattern):
        match = re.search(r"puct_sampler_step_(\d+)\.json$", path)
        if match:
            latest = max(latest, int(match.group(1)))
    return latest


def _make_env(cfg: CodexNoFinetuneConfig, state: Any, sampler: StateSampler) -> Any:
    env = object.__new__(cfg.env_type)
    env.config = cfg
    env.timeout = cfg.timeout
    env.num_cpus_per_task = max(1, int(cfg.num_cpus_per_task))
    env.eval_timeout = cfg.eval_timeout
    env.log_path = cfg.log_path
    env.initial_state = state
    env.sampler = sampler
    env.state = state
    env.problem_type = cfg.problem_type
    return env


def _invalid_result(msg: str) -> VerifyResult:
    return VerifyResult(
        reward=0.0,
        msg=msg,
        correctness=0.0,
        raw_score=0.0,
        result_construction=None,
        stdout="",
    )


def _run_verification(env: Any, generation: str) -> VerifyResult:
    task = env.reward_function(
        problem_type=env.problem_type,
        log_dir=env.log_path,
        eval_timeout=env.eval_timeout,
        num_cpus_per_task=env.num_cpus_per_task,
    )
    out = task.get_reward(generation, state=env.state)
    if not isinstance(out, dict):
        out = {
            "reward": float(out),
            "msg": "",
            "correctness": 1.0,
            "raw_score": float(out),
            "result_construction": None,
            "stdout": "",
        }
    return VerifyResult(
        reward=out["reward"],
        msg=out.get("msg", ""),
        correctness=out.get("correctness", 0.0),
        raw_score=out.get("raw_score", out.get("reward", 0.0)),
        result_construction=out.get("result_construction", None),
        stdout=out.get("stdout", ""),
        metrics=out.get("metrics", {}),
    )


def _check_candidate_format(env: Any, parsed_code: str) -> bool:
    if parsed_code is None or parsed_code.strip() == "":
        return False
    check_format = getattr(env, "check_format", None)
    if check_format is None:
        return True
    try:
        return bool(check_format(parsed_code))
    except Exception as exc:
        logger.warning("Candidate format check failed: %s", exc)
        return False


async def _safe_grade(
    cfg: CodexNoFinetuneConfig,
    env: Any,
    parsed_code: str,
    correct_format: bool,
) -> VerifyResult:
    if not correct_format:
        return _invalid_result("Invalid code")

    loop = asyncio.get_running_loop()
    start_time = time.time()
    try:
        return await asyncio.wait_for(
            loop.run_in_executor(
                SAFE_GRADE_EXECUTOR,
                partial(_run_verification, env, parsed_code),
            ),
            timeout=cfg.timeout,
        )
    except asyncio.TimeoutError:
        elapsed = time.time() - start_time
        logger.warning("Timeout grading: took %.1fs, limit was %.1fs", elapsed, cfg.timeout)
        return _invalid_result("Timeout grading")
    except Exception as exc:
        error_msg = f"Error grading: {exc}\n{traceback.format_exc()}"
        logger.warning("Exception while grading: %s", exc)
        return _invalid_result(error_msg)


def _build_metrics(
    env: Any,
    outs: VerifyResult,
    response: str,
    parsed_code: str,
    correct_format: bool,
) -> dict[str, Any]:
    format_score = float(correct_format)
    message = {"role": "assistant", "content": response}
    build_metrics = getattr(env, "_build_metrics", None)
    if build_metrics is not None:
        try:
            return build_metrics(outs, format_score, message, parsed_code)
        except Exception as exc:
            logger.warning("Falling back after _build_metrics failed: %s", exc)
    return {
        "format": format_score,
        "reward": outs.reward,
        "correctness": outs.correctness,
        "raw_score": outs.raw_score if outs.correctness > 0 else None,
        "initial_raw_score": getattr(env.initial_state, "value", None),
        "msg": outs.msg,
        "prompt": env.get_question(),
        "response": response,
        "parsed_code": parsed_code,
        **(outs.metrics or {}),
    }


def _maybe_create_next_state(env: Any, step_idx: int, parsed_code: str, outs: VerifyResult) -> Any | None:
    if outs.correctness <= 0:
        return None
    create_next_state = getattr(env, "_create_next_state", None)
    if create_next_state is None:
        return None
    return create_next_state(step_idx, parsed_code, outs)


class AutonomousCodexCliCompleter(CodexCliCompleter):
    """Minimal Codex CLI agent mode: give Codex a workspace and evaluator."""

    def __init__(
        self,
        *,
        log_path: str,
        problem_type: str,
        step_idx: int,
        eval_timeout: int,
        num_cpus_per_task: int,
        **kwargs: Any,
    ):
        super().__init__(append_final_answer_instruction=False, **kwargs)
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

    def _build_prompt(self, prompt: str) -> str:
        workspace = self._next_workspace()
        self._workspace = workspace

        matches = re.findall(r"```python\s+([\s\S]*?)\s*```", prompt or "")
        parent_matches = [match for match in matches if "def priority(" in match]
        if not parent_matches:
            parent_source = "def priority(el, n):\n    return 0.0\n"
        else:
            parent_source = parent_matches[-1].strip() + "\n"
        (workspace / "candidate.py").write_text(parent_source, encoding="utf-8")

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
        full_prompt = f"""{prompt}

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
        (workspace / "prompt.txt").write_text(full_prompt, encoding="utf-8")
        return full_prompt

    async def _call_unlocked(self, prompt: str) -> str:
        prompt = self._build_prompt(prompt)
        workspace = self._workspace
        output_path = workspace / "final_response.txt"
        cmd = self._build_command(str(output_path))

        (workspace / "command.json").write_text(json.dumps(cmd, indent=2), encoding="utf-8")
        stdout_log_path = workspace / "codex.stdout.log"
        stderr_log_path = workspace / "codex.stderr.log"
        with stdout_log_path.open("wb") as stdout_log, stderr_log_path.open("wb") as stderr_log:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=stdout_log,
                stderr=stderr_log,
                start_new_session=True,
            )
            communicate = process.communicate(prompt.encode("utf-8"))
            try:
                if self.timeout is not None:
                    await asyncio.wait_for(communicate, timeout=self.timeout)
                else:
                    await communicate
            except asyncio.TimeoutError as exc:
                await _kill_process_tree(process)
                stdout_text = stdout_log_path.read_bytes().decode(errors="replace")
                stderr_text = stderr_log_path.read_bytes().decode(errors="replace")
                raise RuntimeError(
                    "codex exec timed out after "
                    f"{self.timeout}s; workspace={workspace}\n"
                    f"STDOUT:\n{stdout_text}\nSTDERR:\n{stderr_text}"
                ) from exc
            except asyncio.CancelledError:
                await _kill_process_tree(process)
                raise

        stdout_text = stdout_log_path.read_bytes().decode(errors="replace")
        stderr_text = stderr_log_path.read_bytes().decode(errors="replace")
        if process.returncode != 0:
            raise RuntimeError(
                "codex exec failed with exit code "
                f"{process.returncode}; workspace={workspace}\n"
                f"STDOUT:\n{stdout_text}\nSTDERR:\n{stderr_text}"
            )

        return output_path.read_text(encoding="utf-8").strip() if output_path.exists() else ""


def _make_completer(
    cfg: CodexNoFinetuneConfig,
    *,
    semaphore: asyncio.Semaphore | None,
    step_idx: int,
) -> TextCompleter:
    if cfg.backend == "cli":
        if cfg.autonomous:
            if cfg.cli_sandbox == "read-only":
                raise ValueError(
                    "Codex autonomous mode requires --codex-cli-sandbox workspace-write "
                    "or danger-full-access so Codex can write candidates."
                )
            return AutonomousCodexCliCompleter(
                model_name=cfg.model_name,
                codex_command=cfg.cli_command,
                sandbox=cfg.cli_sandbox,
                cwd=os.getcwd(),
                timeout=cfg.cli_timeout,
                semaphore=semaphore,
                log_path=cfg.log_path,
                problem_type=cfg.problem_type,
                step_idx=step_idx,
                eval_timeout=cfg.eval_timeout,
                num_cpus_per_task=max(1, int(cfg.num_cpus_per_task)),
            )
        return CodexCliCompleter(
            model_name=cfg.model_name,
            codex_command=cfg.cli_command,
            sandbox=cfg.cli_sandbox,
            cwd=os.getcwd(),
            timeout=cfg.cli_timeout,
            semaphore=semaphore,
        )
    if cfg.backend == "responses":
        if cfg.autonomous:
            raise ValueError("Codex autonomous mode requires --codex-backend cli.")
        return CodexResponseCompleter(
            model_name=cfg.model_name,
            max_output_tokens=cfg.max_output_tokens,
            temperature=cfg.temperature,
            api_key_env=cfg.api_key_env,
            base_url=cfg.base_url,
            semaphore=semaphore,
        )
    raise ValueError(f"Unknown Codex backend: {cfg.backend}")


async def _run_candidate(
    cfg: CodexNoFinetuneConfig,
    sampler: StateSampler,
    parent_state: Any,
    *,
    group_idx: int,
    sample_idx: int,
    step_idx: int,
    semaphore: asyncio.Semaphore | None,
) -> CandidateResult:
    prompt = ""
    response = ""
    parsed_code = ""
    try:
        env = _make_env(cfg, parent_state, sampler)
        prompt = env.get_question()
        completer = _make_completer(cfg, semaphore=semaphore, step_idx=step_idx)
        response = await completer(prompt)
        get_languages = getattr(env, "_get_code_languages", None)
        languages = get_languages() if get_languages is not None else ["python"]
        keep_separators_fn = getattr(env, "_should_keep_code_separators", None)
        keep_separators = (
            keep_separators_fn() if keep_separators_fn is not None else True
        )
        parsed_code = last_codeblock_postprocess(
            response,
            codeblock_seps=languages,
            keep_separators=keep_separators,
        )
        correct_format = _check_candidate_format(env, parsed_code)
        outs = await _safe_grade(cfg, env, parsed_code, correct_format)
        metrics = _build_metrics(env, outs, response, parsed_code, correct_format)
        next_state = _maybe_create_next_state(env, step_idx, parsed_code, outs)
        return CandidateResult(
            parent_state=parent_state,
            group_idx=group_idx,
            sample_idx=sample_idx,
            prompt=prompt,
            response=response,
            parsed_code=parsed_code,
            reward=float(outs.reward),
            correctness=float(outs.correctness),
            raw_score=outs.raw_score,
            msg=outs.msg,
            metrics=metrics,
            next_state=next_state,
        )
    except Exception as exc:
        error_msg = f"{exc}\n{traceback.format_exc()}"
        logger.warning(
            "Codex candidate failed at step %s group %s sample %s: %s",
            step_idx,
            group_idx,
            sample_idx,
            exc,
        )
        return CandidateResult(
            parent_state=parent_state,
            group_idx=group_idx,
            sample_idx=sample_idx,
            prompt=prompt,
            response=response,
            parsed_code=parsed_code,
            reward=0.0,
            correctness=0.0,
            raw_score=None,
            msg=error_msg,
            metrics={"error": error_msg},
            next_state=None,
            error=error_msg,
        )


def _update_sampler_from_results(sampler: StateSampler, results: list[CandidateResult]) -> None:
    for result in results:
        if result.next_state is not None:
            has_state = getattr(sampler, "has_state", None)
            was_present = False
            if has_state is not None:
                try:
                    was_present = bool(has_state(result.next_state))
                except Exception as exc:
                    logger.warning("Failed to check sampler state before update: %s", exc)
            try:
                sampler.update_states([result.next_state], [result.parent_state], save=False)
                if was_present:
                    result.pool_status = "duplicate"
                elif has_state is not None:
                    try:
                        result.pool_status = (
                            "added_to_pool"
                            if has_state(result.next_state)
                            else "sampler_filtered"
                        )
                    except Exception as exc:
                        logger.warning("Failed to check sampler state after update: %s", exc)
                        result.pool_status = "updated_unknown"
                else:
                    result.pool_status = "updated_unknown"
                continue
            except Exception as exc:
                logger.warning("Failed to update sampler with new state: %s", exc)
                result.pool_status = "sampler_update_error"
                result.sampler_error = str(exc)
        record_failed = getattr(sampler, "record_failed_rollout", None)
        if record_failed is not None:
            record_failed(result.parent_state)
        if result.pool_status is None:
            result.pool_status = "no_valid_state"


def _sample_table(results: list[CandidateResult]) -> list[tuple[Any, ...]]:
    rows: list[tuple[Any, ...]] = []
    for result in results:
        rows.append(
            (
                result.prompt,
                result.response,
                result.reward,
                result.correctness,
                result.parsed_code,
                result.msg,
                result.metrics.get("initial_raw_score", getattr(result.parent_state, "value", None)),
                None,
            )
        )
    return rows


def _result_metrics(results: list[CandidateResult], kept_results: list[CandidateResult]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    rewards = [result.reward for result in kept_results]
    raw_scores = [
        result.raw_score
        for result in kept_results
        if isinstance(result.raw_score, (int, float))
    ]
    response_chars = [len(result.response) for result in kept_results]
    prompt_chars = [len(result.prompt) for result in kept_results]

    metrics["codex/total_samples"] = len(results)
    metrics["codex/kept_samples"] = len(kept_results)
    metrics["codex/failed_samples"] = sum(1 for result in results if result.error)
    metrics["codex/correct_samples"] = sum(1 for result in kept_results if result.correctness > 0)
    if rewards:
        metrics["reward/mean"] = float(np.mean(rewards))
        metrics["reward/max"] = float(np.max(rewards))
        metrics["reward/min"] = float(np.min(rewards))
    if raw_scores:
        metrics["raw_score/mean"] = float(np.mean(raw_scores))
        metrics["raw_score/max"] = float(np.max(raw_scores))
        metrics["raw_score/min"] = float(np.min(raw_scores))
    if response_chars:
        metrics["text/response_chars_mean"] = float(np.mean(response_chars))
        metrics["text/prompt_chars_mean"] = float(np.mean(prompt_chars))

    numeric_values: dict[str, list[float]] = {}
    for result in kept_results:
        for key, value in result.metrics.items():
            if isinstance(value, (int, float)):
                numeric_values.setdefault(key, []).append(float(value))
    for key, values in numeric_values.items():
        metrics[key] = float(np.mean(values))
        metrics[f"{key}/min"] = float(np.min(values))
        metrics[f"{key}/max"] = float(np.max(values))
    return metrics


async def sample_batch(
    cfg: CodexNoFinetuneConfig,
    sampler: StateSampler,
    i_batch: int,
) -> tuple[list[CandidateResult], dict[str, Any], list[CandidateResult]]:
    metrics: dict[str, Any] = {}
    parent_states = sampler.sample_states(cfg.groups_per_batch)
    semaphore = (
        asyncio.Semaphore(cfg.max_concurrent_requests)
        if cfg.max_concurrent_requests is not None
        else None
    )

    tasks = []
    for group_idx, parent_state in enumerate(parent_states):
        for sample_idx in range(cfg.group_size):
            tasks.append(
                asyncio.create_task(
                    _run_candidate(
                        cfg,
                        sampler,
                        parent_state,
                        group_idx=group_idx,
                        sample_idx=sample_idx,
                        step_idx=i_batch,
                        semaphore=semaphore,
                    ),
                    name=f"codex_sample_{group_idx}_{sample_idx}",
                )
            )

    results = await asyncio.gather(*tasks)
    _update_sampler_from_results(sampler, results)

    results_by_group: dict[int, list[CandidateResult]] = {}
    for result in results:
        results_by_group.setdefault(result.group_idx, []).append(result)

    kept_results: list[CandidateResult] = []
    dropped_constant_groups = 0
    for group_results in results_by_group.values():
        rewards = [result.reward for result in group_results]
        if (
            cfg.remove_constant_reward_groups
            and len(group_results) > 1
            and all_same(rewards)
        ):
            dropped_constant_groups += 1
            for result in group_results:
                result.kept = False
                result.drop_reason = "constant_reward_group"
            continue
        kept_results.extend(group_results)

    metrics["codex/parent_states"] = len(parent_states)
    metrics["codex/dropped_constant_groups"] = dropped_constant_groups
    metrics.update(_result_metrics(results, kept_results))
    return kept_results, metrics, results


def _log_agent_tables(
    cfg: CodexNoFinetuneConfig,
    ml_logger: MetricsLogger,
    i_batch: int,
    results: list[CandidateResult],
) -> None:
    table_data = _sample_table(results)
    if not table_data:
        return
    columns = [
        "Prompt",
        "Gen Sequence",
        "Reward",
        "Correctness",
        "Gen Sequence PostProc",
        "Message",
        "Initial Raw Score",
        "Advantage",
    ]
    append_agent_outputs(cfg.log_path, i_batch, results)
    ml_logger.log_table(
        "gen_score_codex",
        columns=columns,
        data=table_data,
        step=i_batch,
    )


def _log_sampler_table(
    ml_logger: MetricsLogger,
    i_batch: int,
    sampler_table_columns: list[str] | None,
    sampler_table_data: list[tuple] | None,
) -> None:
    if sampler_table_columns is None or sampler_table_data is None:
        return
    ml_logger.log_table(
        "sampler_states",
        columns=sampler_table_columns,
        data=sampler_table_data,
        step=i_batch,
    )


async def do_sampling_only(
    *,
    start_batch: int,
    end_batch: int,
    num_batches: int,
    cfg: CodexNoFinetuneConfig,
    sampler: StateSampler,
    ml_logger: MetricsLogger,
) -> None:
    for i_batch in range(start_batch, end_batch):
        metrics: dict[str, Any] = {
            "progress/batch": i_batch,
            "progress/done_frac": (i_batch + 1) / num_batches,
        }
        t_start = time.time()

        with timed("sampling", metrics):
            _results, sampling_metrics, all_results = await sample_batch(cfg, sampler, i_batch)
        metrics.update(sampling_metrics)

        sampler_table_columns, sampler_table_data = None, None
        if hasattr(sampler, "get_sample_stats"):
            metrics.update(sampler.get_sample_stats())
            if hasattr(sampler, "get_sample_table"):
                sampler_table_columns, sampler_table_data = sampler.get_sample_table()

        sampler.flush(step=i_batch + 1)
        _log_agent_tables(cfg, ml_logger, i_batch, all_results)
        _log_sampler_table(ml_logger, i_batch, sampler_table_columns, sampler_table_data)

        metrics["time/total"] = time.time() - t_start
        ml_logger.log_metrics(metrics, step=i_batch)


def _build_sampler(cfg: CodexNoFinetuneConfig, start_batch: int) -> StateSampler:
    return create_sampler(
        log_path=cfg.log_path,
        env_type=cfg.env_type,
        problem_type=cfg.problem_type,
        batch_size=cfg.groups_per_batch,
        resume_step=start_batch if start_batch > 0 else None,
        topk_children=cfg.topk_children,
    )


def _seed_sampler(cfg: CodexNoFinetuneConfig, sampler: StateSampler) -> None:
    seed_initial_pool_paths(
        sampler,
        cfg.initial_pool_paths,
        env_type=cfg.env_type,
        save=False,
    )
    seed_initial_program_paths(
        sampler,
        cfg.initial_program_paths,
        env_type=cfg.env_type,
        problem_type=cfg.problem_type,
        log_path=cfg.log_path,
        eval_timeout=cfg.eval_timeout,
        make_env=lambda state: _make_env(cfg, state, sampler),
        save=False,
    )


async def main(cfg: CodexNoFinetuneConfig) -> None:
    if cfg.num_epochs < 1:
        raise ValueError("num_epochs must be >= 1")
    if not cfg.log_path:
        raise ValueError("log_path is required")

    object.__setattr__(cfg, "log_path", os.path.expanduser(cfg.log_path))
    os.makedirs(cfg.log_path, exist_ok=True)

    ml_logger = MetricsLogger(
        log_path=cfg.log_path,
        wandb_project=cfg.wandb_project,
        wandb_name=cfg.wandb_name,
        config=cfg,
    )

    start_batch = _latest_sampler_step(cfg.log_path)
    sampler = _build_sampler(cfg, start_batch)
    _seed_sampler(cfg, sampler)

    num_batches_total = cfg.num_epochs
    logger.info("Will sample for %s steps", num_batches_total)
    if start_batch < num_batches_total:
        await do_sampling_only(
            start_batch=start_batch,
            end_batch=num_batches_total,
            num_batches=num_batches_total,
            cfg=cfg,
            sampler=sampler,
            ml_logger=ml_logger,
        )
    else:
        logger.info("Sampling-only run was already complete; nothing to do")

    ml_logger.close()
    logger.info("Codex no-finetune discovery completed successfully")
