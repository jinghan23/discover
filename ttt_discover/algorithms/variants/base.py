"""Generic hook engine for search *variants*.

A *variant* is a self-contained experiment that plugs into one or more *phases*
of the search. The trunk only ever references stable phase names; it never knows
about individual variants. Each phase threads a single value through an ordered
chain of hooks:

    hook(value, ctx, params) -> (new_value, metrics)

``value`` is the thing the phase transforms (a tree-node value, a prompt string,
...). ``ctx`` is phase-specific and shared across the chain. ``params`` is this
variant's per-run config.

This module is phase-agnostic: it stores hooks per phase, validates which
variants a run enabled, and runs the chain. Concrete phases (their context +
typed wrappers) live in ``phases.py``; variants live in their own modules.
Adding a phase or a variant never edits this file.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable, TypeVar

T = TypeVar("T")

# A hook transforms the carried value for its phase and returns metrics.
Hook = Callable[[Any, Any, "dict[str, Any]"], "tuple[Any, dict[str, Any]]"]

# phase -> variant name -> hook
_HOOKS: dict[str, dict[str, Hook]] = {}


def register_hook(phase: str, name: str) -> Callable[[Hook], Hook]:
    """Decorator: register ``fn`` as variant ``name``'s hook for ``phase``."""
    if not isinstance(phase, str) or not phase:
        raise ValueError(f"phase must be a non-empty string; got {phase!r}")
    if not isinstance(name, str) or not name:
        raise ValueError(f"variant name must be a non-empty string; got {name!r}")

    def deco(fn: Hook) -> Hook:
        phase_hooks = _HOOKS.setdefault(phase, {})
        if name in phase_hooks:
            raise ValueError(f"{phase!r} hook {name!r} is already registered")
        phase_hooks[name] = fn
        return fn

    return deco


def list_variants() -> list[str]:
    """All variant names registered in any phase."""
    return sorted({name for hooks in _HOOKS.values() for name in hooks})


def list_phase_variants(phase: str) -> list[str]:
    """Variant names that registered a hook for ``phase``."""
    return sorted(_HOOKS.get(phase, {}))


def is_known_variant(name: str) -> bool:
    """True if ``name`` registered a hook in at least one phase."""
    return any(name in hooks for hooks in _HOOKS.values())


def ensure_known_variant(name: str) -> None:
    if not is_known_variant(name):
        raise KeyError(
            f"Unknown variant {name!r}. Registered variants: {list_variants()}"
        )


def _normalize_variants(raw: Any) -> tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        items: tuple[str, ...] = (raw,)
    else:
        try:
            items = tuple(raw)
        except TypeError as exc:
            raise TypeError(
                "cfg.variants must be a string, an iterable of strings, or None"
            ) from exc
    for item in items:
        if not isinstance(item, str) or not item:
            raise TypeError(
                f"cfg.variants must contain non-empty strings; got {item!r}"
            )
    return items


def _normalize_variant_params(raw: Any) -> dict[str, dict[str, Any]]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise TypeError("cfg.variant_params must be a mapping or None")
    out: dict[str, dict[str, Any]] = {}
    for name, params in raw.items():
        if not isinstance(name, str) or not name:
            raise TypeError(
                f"cfg.variant_params keys must be non-empty strings; got {name!r}"
            )
        if params is None:
            out[name] = {}
        elif isinstance(params, Mapping):
            out[name] = dict(params)
        else:
            raise TypeError(
                f"cfg.variant_params[{name!r}] must be a mapping or None"
            )
    return out


def resolve_variant_config(cfg: Any) -> tuple[tuple[str, ...], dict[str, dict[str, Any]]]:
    """Return validated ``(variants, params_by_name)`` from a run config."""
    return (
        _normalize_variants(getattr(cfg, "variants", None)),
        _normalize_variant_params(getattr(cfg, "variant_params", None)),
    )


def apply_hooks(phase: str, value: T, ctx: Any, cfg: Any) -> tuple[T, dict[str, Any]]:
    """Thread ``value`` through ``phase``'s enabled hooks, in ``cfg.variants`` order.

    Unknown variant names raise (typo protection). A variant enabled for the run
    but with no hook in *this* phase is skipped -- that is exactly how a variant
    can touch some phases and not others. Returns ``(new_value, merged_metrics)``.
    """
    variants, params_by_name = resolve_variant_config(cfg)
    if not variants:
        return value, {}
    phase_hooks = _HOOKS.get(phase, {})
    metrics: dict[str, Any] = {}
    for name in variants:
        ensure_known_variant(name)
        hook = phase_hooks.get(name)
        if hook is None:
            continue
        value, hook_metrics = hook(value, ctx, params_by_name.get(name, {}))
        metrics.update(hook_metrics)
    return value, metrics


__all__ = [
    "Hook",
    "register_hook",
    "apply_hooks",
    "resolve_variant_config",
    "list_variants",
    "list_phase_variants",
    "is_known_variant",
    "ensure_known_variant",
]
