"""Reward-shaping state-value variant.

Dampens small improvements and rewards large ones, relative to the parent node,
with a threshold that decays over steps. Enable per run with::

    cfg.variants = ("reward_shaping",)
    cfg.variant_params = {
        "reward_shaping": {
            "threshold_start": 0.010,
            "threshold_end": 0.002,
            "decay_steps": 20,
            "small_weight": 0.1,
            "large_weight": 1.2,
        },
    }

All params are optional; the defaults below match the original implementation.
"""

from __future__ import annotations

from typing import Any

from ttt_discover.algorithms.variants.phases import (
    StateValueContext,
    register_state_value_hook,
)

# Defaults preserved from the original config fields.
_DEFAULTS = {
    "threshold_start": 0.010,
    "threshold_end": 0.002,
    "decay_steps": 20,
    "small_weight": 0.1,
    "large_weight": 1.2,
}


def reward_shaping_threshold(
    step_idx: int,
    *,
    threshold_start: float = 0.010,
    threshold_end: float = 0.002,
    decay_steps: int = 20,
) -> float:
    start = max(0.0, float(threshold_start))
    end = max(0.0, float(threshold_end))
    if decay_steps <= 0:
        return end

    step = max(0.0, float(step_idx))
    progress = min(step / decay_steps, 1.0)
    return start + (end - start) * progress


@register_state_value_hook("reward_shaping")
def reward_shaping(
    value: float | None,
    ctx: StateValueContext,
    params: dict[str, Any],
) -> tuple[float | None, dict[str, Any]]:
    threshold_start = float(params.get("threshold_start", _DEFAULTS["threshold_start"]))
    threshold_end = float(params.get("threshold_end", _DEFAULTS["threshold_end"]))
    decay_steps = int(params.get("decay_steps", _DEFAULTS["decay_steps"]))
    small = float(params.get("small_weight", _DEFAULTS["small_weight"]))
    large = float(params.get("large_weight", _DEFAULTS["large_weight"]))

    metrics: dict[str, Any] = {"reward_shaping/enabled": False}
    if value is None:
        return value, metrics

    raw_value = float(value)
    metrics["reward_shaping/value_raw"] = raw_value

    parent_value = getattr(ctx.parent, "value", None)
    if parent_value is None:
        metrics["reward_shaping/value_shaped"] = raw_value
        return raw_value, metrics

    parent = float(parent_value)
    delta = raw_value - parent
    threshold = reward_shaping_threshold(
        ctx.step_idx,
        threshold_start=threshold_start,
        threshold_end=threshold_end,
        decay_steps=decay_steps,
    )

    if delta <= 0:
        shaped_delta = delta
    elif delta <= threshold:
        shaped_delta = delta * small
    else:
        shaped_delta = threshold * small + (delta - threshold) * large

    shaped_value = parent + shaped_delta
    metrics.update(
        {
            "reward_shaping/enabled": True,
            "reward_shaping/parent_value": parent,
            "reward_shaping/delta": delta,
            "reward_shaping/threshold": threshold,
            "reward_shaping/small_weight": small,
            "reward_shaping/large_weight": large,
            "reward_shaping/value_shaped": shaped_value,
        }
    )
    return shaped_value, metrics


__all__ = ["reward_shaping", "reward_shaping_threshold"]
