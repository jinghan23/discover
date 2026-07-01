"""Prompt variant that nudges search toward exploration on plateaus.

Enable per run with::

    cfg.variants = ("exploration_prompt",)
    cfg.variant_params = {
        "exploration_prompt": {
            "top_k": 16,
            "top_gap_threshold": 20.0,
            "stale_steps": 3,
            "prompt_suffix": "...",
        },
    }

The default ``top_gap_threshold`` is in the same units as ``State.value``. For
latency tasks that store verifier results in microseconds, ``20.0`` means 20us.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from ttt_discover.algorithms.variants.phases import (
    PromptContext,
    register_prompt_hook,
)

_DEFAULTS = {
    "top_k": 16,
    "top_gap_threshold": 20.0,
    "stale_steps": 3,
    "improvement_epsilon": 0.0,
    "separator": "\n\n",
    "prompt_suffix": (
        "[Exploration prompt placeholder]\n"
        "The current search appears to be close to a plateau. Try a materially "
        "different approach, explore less-tested assumptions, and prioritize "
        "candidates that could escape the current local neighborhood."
    ),
}


@dataclass
class _RunState:
    best_value: float
    best_step: int
    last_observed_step: int


_RUN_STATES: dict[str, _RunState] = {}


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _pool_states(env: Any) -> list[Any]:
    sampler = getattr(env, "sampler", None)
    if sampler is None:
        return []

    states = getattr(sampler, "_states", None)
    if states is not None:
        return list(states.values()) if isinstance(states, dict) else list(states)

    nodes = getattr(sampler, "_nodes", None)
    if isinstance(nodes, dict):
        return [getattr(node, "state", None) for node in nodes.values()]

    return []


def _top_values(states: list[Any], top_k: int) -> list[float]:
    values = [
        value
        for value in (_finite_float(getattr(state, "value", None)) for state in states)
        if value is not None
    ]
    values.sort(reverse=True)
    return values[:top_k]


def _run_key(ctx: PromptContext) -> str:
    env = ctx.env
    log_path = str(
        getattr(env, "log_path", "")
        or getattr(getattr(env, "config", None), "log_path", "")
        or ""
    )
    if log_path:
        return f"log_path:{log_path}"
    return f"sampler:{id(getattr(env, 'sampler', None))}"


def _stale_steps_since_best(
    ctx: PromptContext,
    best_value: float | None,
    *,
    improvement_epsilon: float,
) -> int:
    # No readable best value (e.g. the pool could not be read) means no plateau
    # signal, so treat it as "not stale" rather than letting the counter run away.
    if best_value is None:
        return 0

    step_idx = int(ctx.step_idx)
    key = _run_key(ctx)
    state = _RUN_STATES.get(key)
    if state is None or step_idx < state.last_observed_step:
        _RUN_STATES[key] = _RunState(
            best_value=best_value,
            best_step=step_idx,
            last_observed_step=step_idx,
        )
        return 0

    if best_value > state.best_value + improvement_epsilon:
        state.best_value = best_value
        state.best_step = step_idx

    state.last_observed_step = max(state.last_observed_step, step_idx)
    return max(0, step_idx - state.best_step)


def reset_exploration_prompt_state() -> None:
    """Clear in-process plateau tracking; useful for direct unit tests."""
    _RUN_STATES.clear()


@register_prompt_hook("exploration_prompt")
def exploration_prompt(
    prompt: str,
    ctx: PromptContext,
    params: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    top_k = max(1, int(params.get("top_k", _DEFAULTS["top_k"])))
    top_gap_threshold = float(
        params.get("top_gap_threshold", _DEFAULTS["top_gap_threshold"])
    )
    stale_steps_threshold = int(params.get("stale_steps", _DEFAULTS["stale_steps"]))
    improvement_epsilon = max(
        0.0,
        float(params.get("improvement_epsilon", _DEFAULTS["improvement_epsilon"])),
    )

    top_values = _top_values(_pool_states(ctx.env), top_k)
    best_value = top_values[0] if top_values else None
    top_gap = top_values[0] - top_values[-1] if len(top_values) >= top_k else None
    stale_steps = _stale_steps_since_best(
        ctx,
        best_value,
        improvement_epsilon=improvement_epsilon,
    )

    top_gap_trigger = top_gap is not None and top_gap <= top_gap_threshold
    stale_trigger = (
        stale_steps_threshold > 0 and stale_steps >= stale_steps_threshold
    )
    should_inject = top_gap_trigger or stale_trigger

    metrics: dict[str, Any] = {
        "exploration_prompt/enabled": True,
        "exploration_prompt/injected": should_inject,
        "exploration_prompt/reason_top_gap": top_gap_trigger,
        "exploration_prompt/reason_stale_best": stale_trigger,
        "exploration_prompt/top_k": top_k,
        "exploration_prompt/top_k_available": len(top_values),
        "exploration_prompt/top_gap_threshold": top_gap_threshold,
        "exploration_prompt/stale_steps_threshold": stale_steps_threshold,
        "exploration_prompt/stale_steps_since_best": stale_steps,
    }
    if best_value is not None:
        metrics["exploration_prompt/best_value"] = best_value
    if top_gap is not None:
        metrics["exploration_prompt/top_gap"] = top_gap

    if not should_inject:
        return prompt, metrics

    suffix = str(params.get("prompt_suffix", _DEFAULTS["prompt_suffix"])).strip()
    if not suffix:
        return prompt, metrics

    separator = str(params.get("separator", _DEFAULTS["separator"]))
    return f"{prompt.rstrip()}{separator}{suffix}\n", metrics


__all__ = [
    "exploration_prompt",
    "reset_exploration_prompt_state",
]
