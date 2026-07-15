from __future__ import annotations

import asyncio
import glob
import json
import logging
import os
import re
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ttt_discover.algorithms.variants import apply_prompt_hooks, apply_state_value_hooks
from ttt_discover.algorithms.runtime import (
    BudgetTracker,
    CandidateResult,
    MetricsLogger,
    last_codeblock_postprocess,
    log_agent_tables,
    result_metrics,
    safe_grade,
    timed,
)
from ttt_discover.codex_utils.autonomous import AutonomousCodexCliCompleter
from ttt_discover.codex_utils.completers import CodexCliTimeoutError
from ttt_discover.algorithms.state import state_from_dict, to_json_serializable
from ttt_discover.config import DiscoverConfig
from ttt_discover.eval_runners import EvalRunner
from ttt_discover.tasks import Task
from ttt_discover.algorithms.ttt_discover.sampler import load_initial_states

logger = logging.getLogger(__name__)


def _pool_file_for_step(log_path: str, step: int) -> str:
    return os.path.join(log_path, f"autoevolve_pool_step_{step:06d}.json")


def _latest_pool_step(log_path: str) -> int:
    latest = 0
    for path in glob.glob(os.path.join(log_path, "autoevolve_pool_step_*.json")):
        match = re.search(r"autoevolve_pool_step_(\d+)\.json$", path)
        if match:
            latest = max(latest, int(match.group(1)))
    return latest


def _state_value(state: Any) -> float:
    value = getattr(state, "value", None)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("-inf")


class AutoEvolvePool:
    """Simple AutoEvolve-owned state pool.

    This intentionally does not use TTT-Discover's PUCT sampler. AutoEvolve can
    append every externally verified state it produces; parent selection is a
    simple best-value pick.
    """

    def __init__(
        self,
        *,
        log_path: str,
        task: Task,
        batch_size: int,
        initial_state_file: str | None = None,
        resume_step: int | None = None,
    ):
        self.log_path = log_path
        self.task = task
        self.batch_size = max(1, int(batch_size))
        self._states: list[Any] = []
        self._initial_states: list[Any] = []
        self._last_sampled_states: list[Any] = []
        self._current_step = resume_step if resume_step is not None else 0

        if resume_step is not None:
            self._load(resume_step)
        if not self._states:
            initial_states = (
                load_initial_states(initial_state_file, self.task.env_type)
                if initial_state_file
                else [self.task.create_initial_state() for _ in range(self.batch_size)]
            )
            for state in initial_states:
                self._states.append(state)
                self._initial_states.append(state)
            self.flush(step=self._current_step)

    def _load(self, step: int) -> None:
        path = _pool_file_for_step(self.log_path, step)
        with open(path, "r", encoding="utf-8") as f:
            store = json.load(f)
        state_type = self.task.state_type
        self._states = [
            state_from_dict(item, state_type=state_type)
            for item in store.get("states", [])
            if isinstance(item, dict)
        ]
        self._initial_states = [
            state_from_dict(item, state_type=state_type)
            for item in store.get("initial_states", [])
            if isinstance(item, dict)
        ]

    def _state_to_dict(self, state: Any) -> Any:
        to_dict = getattr(state, "to_dict", None)
        if callable(to_dict):
            return to_dict()
        return to_json_serializable(state)

    def sample_states(self, num_states: int) -> list[Any]:
        if not self._states:
            state = self.task.create_initial_state()
            self._states.append(state)
            self._initial_states.append(state)

        ranked = sorted(self._states, key=_state_value, reverse=True)
        picked = ranked[: max(1, int(num_states))]
        self._last_sampled_states = picked
        return picked

    def add_results(self, results: list[CandidateResult]) -> int:
        added = 0
        for result in results:
            child = result.next_state
            if child is None:
                result.pool_status = result.pool_status or "no_valid_state"
                continue
            self._set_parent_info(child, result.parent_state)
            self._states.append(child)
            result.pool_status = "added_to_autoevolve_pool"
            added += 1
        return added

    def flush(self, step: int | None = None) -> None:
        if step is not None:
            self._current_step = step
        os.makedirs(self.log_path, exist_ok=True)
        store = {
            "step": self._current_step,
            "states": [self._state_to_dict(state) for state in self._states],
            "initial_states": [
                self._state_to_dict(state) for state in self._initial_states
            ],
        }
        path = _pool_file_for_step(self.log_path, self._current_step)
        tmp_path = f"{path}.tmp.{os.getpid()}"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(to_json_serializable(store), f, indent=2, sort_keys=True)
        os.replace(tmp_path, path)

    @staticmethod
    def _set_parent_info(child: Any, parent: Any) -> None:
        if parent is None:
            return
        parent_value = getattr(parent, "value", None)
        parent_values = getattr(parent, "parent_values", []) or []
        parents = getattr(parent, "parents", []) or []
        child.parent_values = (
            [parent_value] + parent_values if parent_value is not None else parent_values
        )
        child.parents = [
            {"id": getattr(parent, "id", None), "timestep": getattr(parent, "timestep", None)}
        ] + parents

    def get_sample_stats(self) -> dict[str, Any]:
        values = [_state_value(state) for state in self._states]
        finite_values = [value for value in values if value != float("-inf")]
        stats: dict[str, Any] = {
            "autoevolve_pool/size": len(self._states),
            "autoevolve_pool/initial_size": len(self._initial_states),
            "autoevolve_pool/sampled_size": len(self._last_sampled_states),
        }
        if finite_values:
            stats.update(
                {
                    "autoevolve_pool/value_mean": sum(finite_values) / len(finite_values),
                    "autoevolve_pool/value_min": min(finite_values),
                    "autoevolve_pool/value_max": max(finite_values),
                }
            )
        return stats

    def get_pool_table(self) -> tuple[list[str], list[tuple[Any, ...]]]:
        columns = [
            "pool_idx",
            "id",
            "timestep",
            "value",
            "parent_value",
            "construction_len",
            "observation_len",
        ]
        rows = []
        for idx, state in enumerate(self._states):
            construction = getattr(state, "construction", None)
            parents = getattr(state, "parent_values", []) or []
            observation = getattr(state, "observation", "") or ""
            rows.append(
                (
                    idx,
                    getattr(state, "id", None),
                    getattr(state, "timestep", None),
                    getattr(state, "value", None),
                    parents[0] if parents else None,
                    len(construction) if construction is not None else 0,
                    len(observation),
                )
            )
        return columns, rows


@dataclass
class GeneratedCandidate:
    name: str
    code: str
    source: str
    path: str | None = None


def _safe_workspace_path(workspace: Path, raw_path: str) -> Path | None:
    root = workspace.resolve()
    path = (workspace / raw_path).resolve()
    if path == root or str(path).startswith(str(root) + os.sep):
        return path
    return None


def _candidate_from_file(path: Path, *, name: str, source: str) -> GeneratedCandidate | None:
    try:
        code = path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("Could not read AutoEvolve candidate file %s: %s", path, exc)
        return None
    if not code.strip():
        return None
    return GeneratedCandidate(name=name, code=code, source=source, path=str(path))


def _manifest_candidates(workspace: Path) -> list[GeneratedCandidate]:
    manifest_path = workspace / "candidate_pool.json"
    if not manifest_path.exists():
        return []
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read AutoEvolve candidate manifest %s: %s", manifest_path, exc)
        return []

    raw_candidates = payload.get("candidates", payload) if isinstance(payload, dict) else payload
    if not isinstance(raw_candidates, list):
        logger.warning("AutoEvolve candidate_pool.json must contain a list or candidates list")
        return []

    candidates: list[GeneratedCandidate] = []
    for idx, entry in enumerate(raw_candidates):
        name = f"manifest_{idx:04d}"
        if isinstance(entry, dict):
            name = str(entry.get("name") or name)
            if isinstance(entry.get("code"), str) and entry["code"].strip():
                candidates.append(
                    GeneratedCandidate(
                        name=name,
                        code=entry["code"],
                        source="candidate_pool.json:code",
                        path=str(manifest_path),
                    )
                )
                continue
            raw_path = entry.get("path") or entry.get("file")
            if isinstance(raw_path, str):
                path = _safe_workspace_path(workspace, raw_path)
                if path is not None and path.is_file():
                    candidate = _candidate_from_file(
                        path,
                        name=name,
                        source="candidate_pool.json:path",
                    )
                    if candidate is not None:
                        candidates.append(candidate)
                continue
        if isinstance(entry, str) and entry.strip():
            path = _safe_workspace_path(workspace, entry)
            if path is not None and path.is_file():
                candidate = _candidate_from_file(
                    path,
                    name=name,
                    source="candidate_pool.json:path",
                )
                if candidate is not None:
                    candidates.append(candidate)
            else:
                candidates.append(
                    GeneratedCandidate(
                        name=name,
                        code=entry,
                        source="candidate_pool.json:code",
                        path=str(manifest_path),
                    )
                )
    return candidates


def _workspace_candidates(completer: Any) -> list[GeneratedCandidate]:
    workspace_raw = getattr(completer, "_workspace", None)
    if workspace_raw is None:
        return []
    workspace = Path(workspace_raw)

    candidates = _manifest_candidates(workspace)
    for dirname in ("state_pool", "candidate_pool", "pool"):
        pool_dir = workspace / dirname
        if not pool_dir.is_dir():
            continue
        for path in sorted(pool_dir.glob("*.py")):
            candidate = _candidate_from_file(
                path,
                name=f"{dirname}/{path.name}",
                source=f"{dirname}/*.py",
            )
            if candidate is not None:
                candidates.append(candidate)

    submission = workspace / "submission.py"
    if submission.is_file():
        candidate = _candidate_from_file(
            submission,
            name="submission.py",
            source="workspace_submission.py",
        )
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def _response_candidate(task: Task, env: Any, response: str) -> GeneratedCandidate | None:
    parsed_code = last_codeblock_postprocess(
        response,
        codeblock_seps=task.code_languages(env),
        keep_separators=task.keep_code_separators(env),
    )
    if not parsed_code.strip():
        return None
    return GeneratedCandidate(
        name="final_response_codeblock",
        code=parsed_code,
        source="final_response_codeblock",
    )


def _make_completer(
    cfg: DiscoverConfig,
    *,
    task: Task,
    eval_runner: EvalRunner,
    env: Any,
    semaphore: asyncio.Semaphore | None,
    step_idx: int,
):
    if cfg.backend != "cli":
        raise ValueError("AutoEvolve requires --codex-backend cli.")
    if cfg.cli_sandbox == "read-only":
        raise ValueError(
            "AutoEvolve requires --codex-cli-sandbox workspace-write "
            "or danger-full-access so Codex can write candidates."
        )
    return AutonomousCodexCliCompleter.from_discover_config(
        cfg,
        task=task,
        eval_runner=eval_runner,
        env=env,
        semaphore=semaphore,
        step_idx=step_idx,
    )


async def _evaluate_generated_candidate(
    cfg: DiscoverConfig,
    task: Task,
    eval_runner: EvalRunner,
    env: Any,
    parent_state: Any,
    candidate: GeneratedCandidate,
    *,
    group_idx: int,
    sample_idx: int,
    candidate_idx: int,
    step_idx: int,
    response: str,
    cli_timeout_error: CodexCliTimeoutError | None,
    prompt_metrics: dict[str, Any] | None = None,
) -> CandidateResult:
    parsed_code = candidate.code
    evaluator_calls = 0
    try:
        correct_format = task.check_candidate_format(env, parsed_code)
        evaluator_calls = 1 if correct_format else 0
        outs = await safe_grade(
            eval_runner,
            env,
            parsed_code,
            correct_format,
            timeout=cfg.timeout,
        )
        metrics = task.build_metrics(
            env,
            outs,
            response=response,
            parsed_code=parsed_code,
            correct_format=correct_format,
        )
        metrics["autoevolve/candidate_idx"] = candidate_idx
        metrics["autoevolve/candidate_name"] = candidate.name
        metrics["autoevolve/candidate_path"] = candidate.path
        metrics["codex/parsed_code_source"] = candidate.source
        metrics["codex/cli_timeout_salvaged"] = cli_timeout_error is not None
        metrics["budget/evaluator_calls"] = evaluator_calls
        if cli_timeout_error is not None:
            metrics["codex/cli_timeout_error"] = (
                f"codex exec timed out after {cli_timeout_error.timeout}s; "
                f"log_dir={cli_timeout_error.call_dir}"
            )

        next_state = task.create_next_state(
            env,
            step_idx=step_idx,
            parsed_code=parsed_code,
            outs=outs,
        )
        metrics.update(
            apply_state_value_hooks(next_state, parent_state, step_idx=step_idx, cfg=cfg)
        )
        metrics.update(prompt_metrics or {})

        return CandidateResult(
            parent_state=parent_state,
            group_idx=group_idx,
            sample_idx=sample_idx,
            prompt=task.get_prompt(env),
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
            "AutoEvolve generated candidate failed at step %s group %s sample %s "
            "candidate %s: %s",
            step_idx,
            group_idx,
            sample_idx,
            candidate_idx,
            exc,
        )
        return CandidateResult(
            parent_state=parent_state,
            group_idx=group_idx,
            sample_idx=sample_idx,
            prompt="",
            response=response,
            parsed_code=parsed_code,
            reward=0.0,
            correctness=0.0,
            raw_score=None,
            msg=error_msg,
            metrics={
                "error": error_msg,
                "autoevolve/candidate_idx": candidate_idx,
                "autoevolve/candidate_name": candidate.name,
                "autoevolve/candidate_path": candidate.path,
                "codex/parsed_code_source": candidate.source,
                "budget/evaluator_calls": evaluator_calls,
                **(prompt_metrics or {}),
            },
            next_state=None,
            error=error_msg,
        )


async def _run_candidate(
    cfg: DiscoverConfig,
    task: Task,
    eval_runner: EvalRunner,
    pool: AutoEvolvePool,
    parent_state: Any,
    *,
    group_idx: int,
    sample_idx: int,
    step_idx: int,
    semaphore: asyncio.Semaphore | None,
) -> list[CandidateResult]:
    prompt = ""
    response = ""
    cli_timeout_error: CodexCliTimeoutError | None = None
    try:
        env = task.make_env(parent_state, sampler=pool)
        prompt = task.get_prompt(env)
        prompt, prompt_metrics = apply_prompt_hooks(
            prompt,
            env=env,
            parent=parent_state,
            step_idx=step_idx,
            group_idx=group_idx,
            sample_idx=sample_idx,
            cfg=cfg,
        )
        completer = _make_completer(
            cfg,
            task=task,
            eval_runner=eval_runner,
            env=env,
            semaphore=semaphore,
            step_idx=step_idx,
        )
        try:
            response = await completer(prompt)
        except CodexCliTimeoutError as exc:
            cli_timeout_error = exc
            response = ""

        generated_candidates = _workspace_candidates(completer)
        response_candidate = _response_candidate(task, env, response)
        if response_candidate is not None:
            generated_candidates.append(response_candidate)

        if not generated_candidates:
            generated_candidates = [
                GeneratedCandidate(
                    name="missing_candidate",
                    code="",
                    source="missing_candidate",
                )
            ]

        results: list[CandidateResult] = []
        for candidate_idx, candidate in enumerate(generated_candidates):
            results.append(
                await _evaluate_generated_candidate(
                    cfg,
                    task,
                    eval_runner,
                    env,
                    parent_state,
                    candidate,
                    group_idx=group_idx,
                    sample_idx=sample_idx,
                    candidate_idx=candidate_idx,
                    step_idx=step_idx,
                    response=response,
                    cli_timeout_error=cli_timeout_error,
                    prompt_metrics=prompt_metrics,
                )
            )
        return results
    except Exception as exc:
        error_msg = f"{exc}\n{traceback.format_exc()}"
        logger.warning(
            "AutoEvolve candidate failed at step %s group %s sample %s: %s",
            step_idx,
            group_idx,
            sample_idx,
            exc,
        )
        return [
            CandidateResult(
                parent_state=parent_state,
                group_idx=group_idx,
                sample_idx=sample_idx,
                prompt=prompt,
                response=response,
                parsed_code="",
                reward=0.0,
                correctness=0.0,
                raw_score=None,
                msg=error_msg,
                metrics={"error": error_msg, "budget/evaluator_calls": 0},
                next_state=None,
                error=error_msg,
            )
        ]


async def sample_batch(
    cfg: DiscoverConfig,
    task: Task,
    eval_runner: EvalRunner,
    pool: AutoEvolvePool,
    i_batch: int,
) -> tuple[list[CandidateResult], dict[str, Any], list[CandidateResult]]:
    metrics: dict[str, Any] = {}
    parent_states = pool.sample_states(cfg.groups_per_batch)
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
                        task,
                        eval_runner,
                        pool,
                        parent_state,
                        group_idx=group_idx,
                        sample_idx=sample_idx,
                        step_idx=i_batch,
                        semaphore=semaphore,
                    ),
                    name=f"autoevolve_sample_{group_idx}_{sample_idx}",
                )
            )

    nested_results = await asyncio.gather(*tasks)
    all_results = [result for results in nested_results for result in results]
    added_states = pool.add_results(all_results)

    metrics["autoevolve/parent_states"] = len(parent_states)
    metrics["autoevolve/agent_runs"] = len(tasks)
    metrics["autoevolve/generated_candidates"] = len(all_results)
    metrics["autoevolve/added_states"] = added_states
    metrics.update(result_metrics(all_results, all_results))
    return all_results, metrics, all_results


async def run(
    cfg: DiscoverConfig,
    *,
    task: Task,
    eval_runner: EvalRunner,
) -> None:
    if cfg.num_epochs < 1:
        raise ValueError("num_epochs must be >= 1")
    if not cfg.log_path:
        raise ValueError("log_path is required")

    object.__setattr__(
        cfg,
        "log_path",
        os.path.abspath(os.path.expanduser(cfg.log_path)),
    )
    os.makedirs(cfg.log_path, exist_ok=True)

    budget = BudgetTracker.from_config(cfg, cfg.log_path)
    ml_logger = MetricsLogger.from_config(cfg)

    start_batch = _latest_pool_step(cfg.log_path)
    pool = AutoEvolvePool(
        log_path=cfg.log_path,
        task=task,
        batch_size=cfg.groups_per_batch,
        initial_state_file=cfg.initial_state_file,
        resume_step=start_batch if start_batch > 0 else None,
    )

    num_batches_total = cfg.num_epochs
    logger.info("Will run AutoEvolve for up to %s steps", num_batches_total)
    budget.log_start("AutoEvolve")

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
                _results, sampling_metrics, all_results = await sample_batch(
                    cfg,
                    task,
                    eval_runner,
                    pool,
                    i_batch,
                )
            metrics.update(sampling_metrics)
            metrics.update(budget.add(all_results))
            metrics["progress/done_frac"] = budget.done_frac(
                (i_batch + 1) / num_batches_total
            )
            metrics.update(pool.get_sample_stats())

            log_agent_tables(cfg.log_path, ml_logger, i_batch, all_results)
            pool.flush(step=i_batch + 1)
            pool_columns, pool_rows = pool.get_pool_table()
            ml_logger.log_table(
                "autoevolve_pool_states",
                columns=pool_columns,
                data=pool_rows,
                step=i_batch,
            )
            metrics["time/total"] = time.time() - t_start
            ml_logger.log_metrics(metrics, step=i_batch)
            if budget.exceeded:
                logger.info(
                    "Stopping after step %s: evaluator call budget reached (%s/%s)",
                    i_batch,
                    budget.used,
                    budget.max_calls,
                )
                break

    finally:
        ml_logger.close()
    logger.info("AutoEvolve completed successfully")
