"""Concrete hook phases: each phase's context object and typed wrappers.

Adding a new phase means adding, here: a ``<PHASE>`` constant, a ``Context``
dataclass, a ``register_<phase>_hook`` decorator, and an ``apply_<phase>_hooks``
wrapper -- then calling that wrapper from the one trunk spot that owns the step.
The generic engine in ``base.py`` is not touched.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from ttt_discover.algorithms.variants.base import Hook, apply_hooks, register_hook

# ---------------------------------------------------------------------------
# state_value phase: rewrite the tree-node value assigned to a new state.
# ---------------------------------------------------------------------------
STATE_VALUE_PHASE = "state_value"


@dataclass(frozen=True)
class StateValueContext:
    state: Any            # the freshly created child state
    parent: Any | None    # the parent state (None at roots)
    step_idx: int


StateValueHook = Callable[
    ["float | None", StateValueContext, "dict[str, Any]"],
    "tuple[float | None, dict[str, Any]]",
]


def register_state_value_hook(name: str) -> Callable[[Hook], Hook]:
    """Decorator: register a ``state_value`` hook for variant ``name``."""
    return register_hook(STATE_VALUE_PHASE, name)


def apply_state_value_hooks(
    next_state: Any,
    parent_state: Any,
    *,
    step_idx: int,
    cfg: Any,
) -> dict[str, Any]:
    """Apply enabled ``state_value`` hooks to ``next_state.value`` in place.

    Returns merged metrics. No-op (``{}``) when ``next_state`` is None or no
    variants are enabled. This is the trunk's single state_value touch-point.
    """
    if next_state is None:
        return {}
    ctx = StateValueContext(state=next_state, parent=parent_state, step_idx=step_idx)
    new_value, metrics = apply_hooks(
        STATE_VALUE_PHASE, getattr(next_state, "value", None), ctx, cfg
    )
    next_state.value = new_value
    return metrics


# ---------------------------------------------------------------------------
# prompt phase: rewrite the prompt shown to the model before it is sent.
# ---------------------------------------------------------------------------
PROMPT_PHASE = "prompt"


@dataclass(frozen=True)
class PromptContext:
    env: Any
    parent: Any | None
    step_idx: int
    group_idx: int | None = None
    sample_idx: int | None = None


PromptHook = Callable[
    [str, PromptContext, "dict[str, Any]"],
    "tuple[str, dict[str, Any]]",
]


def register_prompt_hook(name: str) -> Callable[[Hook], Hook]:
    """Decorator: register a ``prompt`` hook for variant ``name``."""
    return register_hook(PROMPT_PHASE, name)


def apply_prompt_hooks(
    prompt: str,
    *,
    env: Any,
    parent: Any | None,
    step_idx: int,
    group_idx: int | None = None,
    sample_idx: int | None = None,
    cfg: Any,
) -> tuple[str, dict[str, Any]]:
    """Thread ``prompt`` through enabled ``prompt`` hooks; return (prompt, metrics)."""
    ctx = PromptContext(
        env=env,
        parent=parent,
        step_idx=step_idx,
        group_idx=group_idx,
        sample_idx=sample_idx,
    )
    return apply_hooks(PROMPT_PHASE, prompt, ctx, cfg)


__all__ = [
    "STATE_VALUE_PHASE",
    "StateValueContext",
    "StateValueHook",
    "register_state_value_hook",
    "apply_state_value_hooks",
    "PROMPT_PHASE",
    "PromptContext",
    "PromptHook",
    "register_prompt_hook",
    "apply_prompt_hooks",
]
