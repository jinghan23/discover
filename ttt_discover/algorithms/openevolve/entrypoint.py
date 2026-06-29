from __future__ import annotations

import importlib
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any


def _ensure_repo_on_syspath() -> None:
    repo_root = os.environ.get("TTT_DISCOVER_ROOT")
    if repo_root:
        root = str(Path(repo_root).expanduser().resolve())
        if root not in sys.path:
            sys.path.insert(0, root)


def _load_object(spec: str) -> Any:
    module_name, sep, qualname = spec.partition(":")
    if not sep or not module_name or not qualname:
        raise RuntimeError(f"Invalid object spec {spec!r}; expected 'module:qualname'")
    obj: Any = importlib.import_module(module_name)
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


def _state_from_env(env_type: type) -> Any:
    from ttt_discover.algorithms.state import state_from_dict

    raw = os.environ.get("TTT_OPENEVOLVE_INITIAL_STATE_JSON")
    if raw:
        payload = json.loads(raw)
        return state_from_dict(payload, state_type=getattr(env_type, "state_type", None))
    problem_type = os.environ.get("TTT_OPENEVOLVE_PROBLEM_TYPE", "")
    return env_type.create_initial_state(problem_type)


def _make_config() -> SimpleNamespace:
    return SimpleNamespace(
        problem_type=os.environ.get("TTT_OPENEVOLVE_PROBLEM_TYPE", ""),
        log_path=os.environ.get("TTT_OPENEVOLVE_LOG_DIR", ""),
        eval_timeout=int(float(os.environ.get("TTT_OPENEVOLVE_EVAL_TIMEOUT", "300"))),
        num_cpus_per_task=max(
            1,
            int(float(os.environ.get("TTT_OPENEVOLVE_NUM_CPUS_PER_TASK", "1"))),
        ),
        timeout=float(os.environ.get("TTT_OPENEVOLVE_TIMEOUT", "8000")),
    )


def _is_maximize(env_type: type, state: Any, cfg: SimpleNamespace) -> bool:
    try:
        env = env_type(initial_state=state, config=cfg)
        return bool(env.is_maximize())
    except Exception:
        return True


def _as_dict(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        return result
    if isinstance(result, (int, float)) and not isinstance(result, bool):
        return {
            "reward": float(result),
            "correctness": 1.0,
            "raw_score": float(result),
        }
    data: dict[str, Any] = {}
    for key in (
        "reward",
        "msg",
        "correctness",
        "raw_score",
        "result_construction",
        "stdout",
        "metrics",
    ):
        if hasattr(result, key):
            data[key] = getattr(result, key)
    return data


def _numeric(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else default
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return numeric if math.isfinite(numeric) else default


def _should_retry_with_fence(result: dict[str, Any], program: str) -> bool:
    if "```" in program:
        return False
    if _numeric(result.get("correctness"), 0.0) > 0:
        return False
    msg = str(result.get("msg", "") or "").lower()
    return "extract" in msg and "code" in msg


def _evaluate_candidate(reward_evaluator: Any, program: str, state: Any) -> dict[str, Any]:
    result = _as_dict(reward_evaluator.get_reward(program, state=state))
    if _should_retry_with_fence(result, program):
        fenced = f"```python\n{program.rstrip()}\n```"
        result = _as_dict(reward_evaluator.get_reward(fenced, state=state))
    return result


def _split_metrics_and_artifacts(result: dict[str, Any]) -> tuple[dict[str, float], dict[str, Any]]:
    metrics: dict[str, float] = {}
    artifacts: dict[str, Any] = {}
    extra_metrics = result.get("metrics") or {}
    if isinstance(extra_metrics, dict):
        for key, value in extra_metrics.items():
            if isinstance(value, bool):
                metrics[str(key)] = float(value)
            elif isinstance(value, (int, float)):
                numeric = float(value)
                if math.isfinite(numeric):
                    metrics[str(key)] = numeric
                else:
                    artifacts[f"metric/{key}"] = value
            else:
                artifacts[f"metric/{key}"] = value
    for key in ("msg", "stdout", "result_construction"):
        value = result.get(key)
        if value not in (None, ""):
            artifacts[key] = value
    return metrics, artifacts


def _to_evaluation_result(result: dict[str, Any], *, is_maximize: bool) -> Any:
    try:
        from openevolve.evaluation_result import EvaluationResult
    except Exception:
        EvaluationResult = None

    correctness = _numeric(result.get("correctness"), 0.0)
    reward = _numeric(result.get("reward"), 0.0)
    raw_score = _numeric(result.get("raw_score", reward), reward)
    if correctness > 0:
        combined_score = raw_score if is_maximize else -raw_score
    else:
        combined_score = -1.0e12 + reward

    metrics, artifacts = _split_metrics_and_artifacts(result)
    metrics.update(
        {
            "combined_score": float(combined_score),
            "score": float(combined_score),
            "reward": float(reward),
            "raw_score": float(raw_score),
            "correctness": float(correctness),
        }
    )
    artifacts = {
        key: value if isinstance(value, (str, bytes)) else json.dumps(value, default=str)
        for key, value in artifacts.items()
    }

    if EvaluationResult is None:
        return metrics
    return EvaluationResult(metrics=metrics, artifacts=artifacts)


def evaluate(program_path: str) -> Any:
    _ensure_repo_on_syspath()

    env_spec = os.environ.get("TTT_OPENEVOLVE_ENV_TYPE", "")
    if not env_spec:
        raise RuntimeError("Missing TTT_OPENEVOLVE_ENV_TYPE for OpenEvolve evaluation")
    env_type = _load_object(env_spec)
    state = _state_from_env(env_type)
    cfg = _make_config()
    reward_evaluator = env_type.reward_function(
        problem_type=cfg.problem_type,
        log_dir=cfg.log_path,
        eval_timeout=cfg.eval_timeout,
        num_cpus_per_task=cfg.num_cpus_per_task,
    )
    program = Path(program_path).expanduser().resolve().read_text(
        encoding="utf-8",
        errors="replace",
    )
    result = _evaluate_candidate(reward_evaluator, program, state)
    return _to_evaluation_result(
        result,
        is_maximize=_is_maximize(env_type, state, cfg),
    )
