"""Shared runtime utilities for Codex-backed algorithms."""

from __future__ import annotations

import asyncio
import glob
import json
import logging
import os
import re
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np

from ttt_discover.algorithms.variants import apply_prompt_hooks, apply_state_value_hooks
from ttt_discover.codex_utils.completers import CodexCliCompleter, CodexResponseCompleter
from ttt_discover.config import DISCOVER_CONFIG_FIELDS, DiscoverConfig
from ttt_discover.algorithms.state import to_json_serializable
from ttt_discover.eval_runners import EvalRunner
from ttt_discover.tasks import Task, VerifyResult

logger = logging.getLogger(__name__)

SAFE_GRADE_EXECUTOR = ThreadPoolExecutor(max_workers=4096)


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
                config=config_for_logging(config),
            )

    @classmethod
    def from_config(cls, cfg: DiscoverConfig) -> "MetricsLogger":
        return cls(
            log_path=cfg.log_path,
            wandb_project=cfg.wandb_project,
            wandb_name=cfg.wandb_name,
            config=cfg,
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


def config_for_logging(config: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in DISCOVER_CONFIG_FIELDS:
        if not hasattr(config, key):
            continue
        value = getattr(config, key)
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


def latest_sampler_step(log_path: str) -> int:
    pattern = os.path.join(log_path, "puct_sampler_step_*.json")
    latest = 0
    for path in glob.glob(pattern):
        match = re.search(r"puct_sampler_step_(\d+)\.json$", path)
        if match:
            latest = max(latest, int(match.group(1)))
    return latest


def evaluator_call_budget(cfg: Any) -> int | None:
    budget = getattr(cfg, "max_evaluator_calls", None)
    if budget is None:
        return None
    budget = int(budget)
    if budget < 1:
        raise ValueError("max_evaluator_calls must be >= 1")
    return budget


def _metric_int(metrics: dict[str, Any], key: str) -> int | None:
    value = metrics.get(key)
    if isinstance(value, bool) or value is None:
        return None
    try:
        out = int(value)
    except (TypeError, ValueError):
        return None
    return out if out >= 0 else None


def _metrics_evaluator_call_count(metrics: dict[str, Any]) -> int:
    explicit = _metric_int(metrics, "budget/evaluator_calls")
    if explicit is not None:
        return explicit

    inner = _metric_int(metrics, "inner/completed_iterations")
    if inner is not None:
        return inner

    return 1


def evaluator_call_count(results: list[CandidateResult]) -> int:
    return sum(_metrics_evaluator_call_count(result.metrics) for result in results)


def logged_evaluator_call_count(log_path: str) -> int:
    output_path = os.path.join(log_path, "agent_outputs.jsonl")
    if not os.path.exists(output_path):
        return 0

    count = 0
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            metrics = entry.get("metrics")
            if not isinstance(metrics, dict):
                metrics = {}
            count += _metrics_evaluator_call_count(metrics)
    return count


@dataclass
class BudgetTracker:
    max_calls: int | None
    used: int = 0

    @classmethod
    def from_config(cls, cfg: Any, log_path: str) -> "BudgetTracker":
        return cls(
            max_calls=evaluator_call_budget(cfg),
            used=logged_evaluator_call_count(log_path),
        )

    @property
    def enabled(self) -> bool:
        return self.max_calls is not None

    @property
    def exceeded(self) -> bool:
        return self.max_calls is not None and self.used >= self.max_calls

    def add(self, results: list[CandidateResult]) -> dict[str, Any]:
        start_used = self.used
        epoch_calls = evaluator_call_count(results)
        self.used += epoch_calls
        metrics: dict[str, Any] = {
            "budget/evaluator_calls_used_start": start_used,
            "budget/evaluator_calls_epoch": epoch_calls,
            "budget/evaluator_calls_used": self.used,
        }
        if self.max_calls is not None:
            metrics.update(
                {
                    "budget/max_evaluator_calls": self.max_calls,
                    "budget/evaluator_calls_remaining": max(
                        0, self.max_calls - self.used
                    ),
                    "budget/stop_after_epoch": self.exceeded,
                }
            )
        return metrics

    def done_frac(self, fallback: float) -> float:
        if self.max_calls is None:
            return fallback
        return min(1.0, self.used / self.max_calls)

    def log_start(self, run_name: str) -> None:
        if self.max_calls is None:
            return
        logger.info(
            "%s evaluator call budget: %s, already used: %s",
            run_name,
            self.max_calls,
            self.used,
        )


def invalid_result(msg: str) -> VerifyResult:
    return VerifyResult(
        reward=0.0,
        msg=msg,
        correctness=0.0,
        raw_score=0.0,
        result_construction=None,
        stdout="",
    )


async def safe_grade(
    eval_runner: EvalRunner,
    env: Any,
    parsed_code: str,
    correct_format: bool,
    *,
    timeout: float,
) -> VerifyResult:
    if not correct_format:
        return invalid_result("Invalid code")

    loop = asyncio.get_running_loop()
    start_time = time.time()
    try:
        return await asyncio.wait_for(
            loop.run_in_executor(
                SAFE_GRADE_EXECUTOR,
                partial(eval_runner.evaluate, env, parsed_code),
            ),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        elapsed = time.time() - start_time
        logger.warning("Timeout grading: took %.1fs, limit was %.1fs", elapsed, timeout)
        return invalid_result("Timeout grading")
    except Exception as exc:
        error_msg = f"Error grading: {exc}\n{traceback.format_exc()}"
        logger.warning("Exception while grading: %s", exc)
        return invalid_result(error_msg)


def update_sampler_from_results(sampler: Any, results: list[CandidateResult]) -> None:
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


def _finite_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _candidate_state_value(result: CandidateResult) -> float | None:
    return _finite_float_or_none(getattr(result.next_state, "value", None))


def _best_inner_candidate(results: list[CandidateResult]) -> CandidateResult:
    if not results:
        raise ValueError("Cannot select from an empty inner candidate list")

    def sort_key(result: CandidateResult) -> tuple[float, float, float, float]:
        state_value = _candidate_state_value(result)
        reward = _finite_float_or_none(result.reward)
        correctness = _finite_float_or_none(result.correctness)
        reward = reward if reward is not None else float("-inf")
        correctness = correctness if correctness is not None else float("-inf")
        if state_value is not None:
            return (1.0, state_value, reward, correctness)
        return (0.0, reward, correctness, 0.0)

    return max(results, key=sort_key)


def _inner_iteration_metrics(
    results: list[CandidateResult],
    selected: CandidateResult,
    *,
    requested_iterations: int | None = None,
    failed_iterations: int = 0,
) -> dict[str, Any]:
    rewards = [float(result.reward) for result in results]
    correctness = [float(result.correctness) for result in results]
    state_values = [_candidate_state_value(result) for result in results]
    valid_state_values = [value for value in state_values if value is not None]
    selected_idx = selected.metrics.get("inner/idx", 0)
    requested_iterations = (
        len(results) if requested_iterations is None else requested_iterations
    )

    metrics: dict[str, Any] = {
        "inner/iterations": requested_iterations,
        "inner/completed_iterations": len(results),
        "inner/failed_iterations": failed_iterations,
        "inner/selected_idx": int(selected_idx),
        "inner/valid_states": len(valid_state_values),
        "inner/attempt_rewards": rewards,
        "inner/attempt_correctness": correctness,
        "inner/attempt_state_values": state_values,
    }
    if rewards:
        metrics.update(
            {
                "inner/reward_mean": float(np.mean(rewards)),
                "inner/reward_max": float(np.max(rewards)),
                "inner/reward_min": float(np.min(rewards)),
            }
        )
    if valid_state_values:
        metrics["inner/best_state_value"] = float(np.max(valid_state_values))
    return metrics


def sample_table(results: list[CandidateResult]) -> list[tuple[Any, ...]]:
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


def result_metrics(
    results: list[CandidateResult],
    kept_results: list[CandidateResult],
) -> dict[str, Any]:
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


def log_agent_tables(
    log_path: str,
    ml_logger: MetricsLogger,
    i_batch: int,
    results: list[CandidateResult],
) -> None:
    table_data = sample_table(results)
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
    append_agent_outputs(log_path, i_batch, results)
    ml_logger.log_table(
        "gen_score_codex",
        columns=columns,
        data=table_data,
        step=i_batch,
    )


def log_sampler_table(
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


class Loop:
    """Shared search loop: sample states, run candidates, update pool, log, repeat."""

    sampler_cls: type | None = None

    def __init__(
        self,
        cfg: DiscoverConfig,
        *,
        eval_runner: EvalRunner,
        task: Task | None = None,
    ):
        if cfg.env_type is None and task is None:
            raise ValueError("env_type is required")
        self.cfg = cfg
        self.task = task or Task.from_config(cfg)
        self.eval_runner = eval_runner
        self.sampler: Any | None = None
        self.ml_logger: MetricsLogger | None = None

    def latest_step(self) -> int:
        return latest_sampler_step(self.cfg.log_path)

    def build_sampler(self, start_batch: int) -> Any:
        if self.sampler_cls is None:
            raise NotImplementedError(f"{type(self).__name__}.sampler_cls is not set")
        from_config = getattr(self.sampler_cls, "from_config", None)
        if from_config is None:
            raise TypeError(f"{self.sampler_cls.__name__} must define from_config")
        return from_config(self.cfg, task=self.task, start_batch=start_batch)

    def build_prompt(
        self,
        env: Any,
        parent_state: Any,
        *,
        step_idx: int,
        group_idx: int,
        sample_idx: int,
    ) -> str:
        del parent_state, step_idx, group_idx, sample_idx
        return self.task.get_prompt(env)

    def make_completer(
        self,
        *,
        semaphore: asyncio.Semaphore | None,
        step_idx: int,
        group_idx: int,
        sample_idx: int,
    ):
        cfg = self.cfg
        if cfg.backend == "cli":
            return CodexCliCompleter.from_discover_config(
                cfg,
                cwd=os.getcwd(),
                semaphore=semaphore,
                step_idx=step_idx,
                group_idx=group_idx,
                sample_idx=sample_idx,
            )
        if cfg.backend == "responses":
            return CodexResponseCompleter.from_discover_config(
                cfg,
                semaphore=semaphore,
            )
        raise ValueError(f"Unknown Codex backend: {cfg.backend}")

    async def run_candidate(
        self,
        parent_state: Any,
        *,
        group_idx: int,
        sample_idx: int,
        step_idx: int,
        semaphore: asyncio.Semaphore | None,
    ) -> CandidateResult:
        if self.sampler is None:
            raise RuntimeError("Loop sampler is not initialized")

        prompt = ""
        response = ""
        parsed_code = ""
        evaluator_calls = 0
        try:
            env = self.task.make_env(parent_state, sampler=self.sampler)
            prompt = self.build_prompt(
                env,
                parent_state,
                step_idx=step_idx,
                group_idx=group_idx,
                sample_idx=sample_idx,
            )
            prompt, prompt_metrics = apply_prompt_hooks(
                prompt,
                env=env,
                parent=parent_state,
                step_idx=step_idx,
                group_idx=group_idx,
                sample_idx=sample_idx,
                cfg=self.cfg,
            )
            completer = self.make_completer(
                semaphore=semaphore,
                step_idx=step_idx,
                group_idx=group_idx,
                sample_idx=sample_idx,
            )

            inner_iterations = max(1, int(getattr(self.cfg, "inner_iterations", 1)))
            inner_results: list[CandidateResult] = []
            inner_errors: list[str] = []
            for inner_idx in range(inner_iterations):
                try:
                    response = await completer(prompt)
                    parsed_code = last_codeblock_postprocess(
                        response,
                        codeblock_seps=self.task.code_languages(env),
                        keep_separators=self.task.keep_code_separators(env),
                    )
                    correct_format = self.task.check_candidate_format(env, parsed_code)
                    evaluator_call = 1 if correct_format else 0
                    evaluator_calls += evaluator_call
                    outs = await safe_grade(
                        self.eval_runner,
                        env,
                        parsed_code,
                        correct_format,
                        timeout=self.cfg.timeout,
                    )
                    metrics = self.task.build_metrics(
                        env,
                        outs,
                        response=response,
                        parsed_code=parsed_code,
                        correct_format=correct_format,
                    )
                    metrics["codex/parsed_code_source"] = "final_response_codeblock"
                    metrics["codex/cli_timeout_salvaged"] = False
                    metrics["inner/idx"] = inner_idx
                    metrics["budget/evaluator_calls"] = evaluator_call
                    metrics.update(prompt_metrics)
                    next_state = self.task.create_next_state(
                        env,
                        step_idx=step_idx,
                        parsed_code=parsed_code,
                        outs=outs,
                    )
                    inner_results.append(
                        CandidateResult(
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
                    )
                except Exception as exc:
                    inner_error = f"inner_idx={inner_idx}: {exc}\n{traceback.format_exc()}"
                    inner_errors.append(inner_error)
                    logger.warning(
                        "Inner candidate failed at step %s group %s sample %s inner %s: %s",
                        step_idx,
                        group_idx,
                        sample_idx,
                        inner_idx,
                        exc,
                    )

            if not inner_results:
                raise RuntimeError(
                    "All inner candidate attempts failed:\n" + "\n".join(inner_errors)
                )

            selected_result = _best_inner_candidate(inner_results)
            selected_result.metrics.update(
                _inner_iteration_metrics(
                    inner_results,
                    selected_result,
                    requested_iterations=inner_iterations,
                    failed_iterations=len(inner_errors),
                )
            )
            selected_result.metrics.update(
                apply_state_value_hooks(
                    selected_result.next_state,
                    parent_state,
                    step_idx=step_idx,
                    cfg=self.cfg,
                )
            )
            selected_result.metrics["budget/evaluator_calls"] = evaluator_calls
            return selected_result
        except Exception as exc:
            error_msg = f"{exc}\n{traceback.format_exc()}"
            logger.warning(
                "Candidate failed at step %s group %s sample %s: %s",
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
                metrics={
                    "error": error_msg,
                    "budget/evaluator_calls": evaluator_calls,
                },
                next_state=None,
                error=error_msg,
            )

    async def sample_batch(
        self,
        i_batch: int,
    ) -> tuple[list[CandidateResult], dict[str, Any], list[CandidateResult]]:
        if self.sampler is None:
            raise RuntimeError("Loop sampler is not initialized")

        metrics: dict[str, Any] = {}
        parent_states = self.sampler.sample_states(self.cfg.groups_per_batch)
        semaphore = (
            asyncio.Semaphore(self.cfg.max_concurrent_requests)
            if self.cfg.max_concurrent_requests is not None
            else None
        )

        tasks = []
        for group_idx, parent_state in enumerate(parent_states):
            for sample_idx in range(self.cfg.group_size):
                tasks.append(
                    asyncio.create_task(
                        self.run_candidate(
                            parent_state,
                            group_idx=group_idx,
                            sample_idx=sample_idx,
                            step_idx=i_batch,
                            semaphore=semaphore,
                        ),
                        name=f"{type(self).__name__}_sample_{group_idx}_{sample_idx}",
                    )
                )

        results = await asyncio.gather(*tasks)
        update_sampler_from_results(self.sampler, results)

        results_by_group: dict[int, list[CandidateResult]] = {}
        for result in results:
            results_by_group.setdefault(result.group_idx, []).append(result)

        kept_results: list[CandidateResult] = []
        dropped_constant_groups = 0
        for group_results in results_by_group.values():
            rewards = [result.reward for result in group_results]
            if (
                self.cfg.remove_constant_reward_groups
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
        metrics.update(result_metrics(results, kept_results))
        return kept_results, metrics, results

    async def run(self) -> None:
        if self.cfg.num_epochs < 1:
            raise ValueError("num_epochs must be >= 1")
        if not self.cfg.log_path:
            raise ValueError("log_path is required")

        object.__setattr__(
            self.cfg,
            "log_path",
            os.path.abspath(os.path.expanduser(self.cfg.log_path)),
        )
        os.makedirs(self.cfg.log_path, exist_ok=True)

        budget = BudgetTracker.from_config(self.cfg, self.cfg.log_path)
        self.ml_logger = MetricsLogger.from_config(self.cfg)

        start_batch = self.latest_step()
        self.sampler = self.build_sampler(start_batch)

        num_batches_total = self.cfg.num_epochs
        logger.info(
            "Will run %s for up to %s steps",
            type(self).__name__,
            num_batches_total,
        )
        budget.log_start(type(self).__name__)
        try:
            if budget.exceeded:
                logger.info(
                    "Evaluator call budget already reached (%s/%s); nothing to run",
                    budget.used,
                    budget.max_calls,
                )
                return

            for i_batch in range(start_batch, num_batches_total):
                metrics: dict[str, Any] = {
                    "progress/batch": i_batch,
                }

                t_start = time.time()
                with timed("sampling", metrics):
                    _results, sampling_metrics, all_results = await self.sample_batch(
                        i_batch,
                    )
                metrics.update(sampling_metrics)
                metrics.update(budget.add(all_results))
                metrics["progress/done_frac"] = budget.done_frac(
                    (i_batch + 1) / num_batches_total
                )

                sampler_table_columns, sampler_table_data = None, None
                if hasattr(self.sampler, "get_sample_stats"):
                    metrics.update(self.sampler.get_sample_stats())
                    if hasattr(self.sampler, "get_sample_table"):
                        sampler_table_columns, sampler_table_data = (
                            self.sampler.get_sample_table()
                        )

                log_agent_tables(
                    self.cfg.log_path,
                    self.ml_logger,
                    i_batch,
                    all_results,
                )
                self.sampler.flush(step=i_batch + 1)
                log_sampler_table(
                    self.ml_logger,
                    i_batch,
                    sampler_table_columns,
                    sampler_table_data,
                )
                metrics["time/total"] = time.time() - t_start
                self.ml_logger.log_metrics(metrics, step=i_batch)
                if budget.exceeded:
                    logger.info(
                        "Stopping after step %s: evaluator call budget reached (%s/%s)",
                        i_batch,
                        budget.used,
                        budget.max_calls,
                    )
                    break
        finally:
            self.ml_logger.close()

        logger.info("%s completed successfully", type(self).__name__)
