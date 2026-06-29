from __future__ import annotations

from typing import Any


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


def shape_state_value(
    child_value: float | None,
    parent_value: float | None,
    step_idx: int,
    *,
    enabled: bool = False,
    threshold_start: float = 0.010,
    threshold_end: float = 0.002,
    decay_steps: int = 20,
    small_weight: float = 0.1,
    large_weight: float = 1.2,
) -> tuple[float | None, dict[str, Any]]:
    metrics: dict[str, Any] = {"reward_shaping/enabled": False}
    if child_value is None:
        return child_value, metrics

    raw_value = float(child_value)
    metrics["reward_shaping/value_raw"] = raw_value

    if not enabled or parent_value is None:
        metrics["reward_shaping/value_shaped"] = raw_value
        return raw_value, metrics

    parent = float(parent_value)
    delta = raw_value - parent
    threshold = reward_shaping_threshold(
        step_idx,
        threshold_start=threshold_start,
        threshold_end=threshold_end,
        decay_steps=decay_steps,
    )
    small = float(small_weight)
    large = float(large_weight)

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


def shape_state_value_from_config(
    child_value: float | None,
    parent_value: float | None,
    step_idx: int,
    cfg: Any,
) -> tuple[float | None, dict[str, Any]]:
    return shape_state_value(
        child_value,
        parent_value,
        step_idx,
        enabled=cfg.reward_shaping,
        threshold_start=cfg.reward_shaping_threshold_start,
        threshold_end=cfg.reward_shaping_threshold_end,
        decay_steps=cfg.reward_shaping_decay_steps,
        small_weight=cfg.reward_shaping_small_weight,
        large_weight=cfg.reward_shaping_large_weight,
    )


__all__ = [
    "reward_shaping_threshold",
    "shape_state_value",
    "shape_state_value_from_config",
]
