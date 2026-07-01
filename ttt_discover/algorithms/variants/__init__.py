"""Search variants, confined to the algorithms package.

Two rates of change, cleanly separated:

* Add a *variant* (common): create ``variants/<name>.py`` with a function
  decorated by ``@register_<phase>_hook("<name>")``, then add one import below so
  the decorator runs on import. Enable per run via ``cfg.variants=("<name>",)``
  and optional ``cfg.variant_params={"<name>": {...}}``.
* Add a *phase* (rare): define its context + ``register_<phase>_hook`` /
  ``apply_<phase>_hooks`` in ``phases.py`` and call the ``apply_*`` wrapper from
  the one trunk spot that owns that step. The engine in ``base.py`` is untouched.

A variant may register hooks in several phases; a variant enabled for a run but
with no hook in a given phase is simply skipped there.
"""

from ttt_discover.algorithms.variants.base import (
    apply_hooks,
    ensure_known_variant,
    is_known_variant,
    list_phase_variants,
    list_variants,
    register_hook,
    resolve_variant_config,
)
from ttt_discover.algorithms.variants.phases import (
    PROMPT_PHASE,
    PromptContext,
    PromptHook,
    STATE_VALUE_PHASE,
    StateValueContext,
    StateValueHook,
    apply_prompt_hooks,
    apply_state_value_hooks,
    register_prompt_hook,
    register_state_value_hook,
)

# Import builtin variants so their decorators run on import.
from ttt_discover.algorithms.variants import exploration_prompt  # noqa: F401
from ttt_discover.algorithms.variants import reward_shaping  # noqa: F401

__all__ = [
    # generic engine
    "register_hook",
    "apply_hooks",
    "resolve_variant_config",
    "list_variants",
    "list_phase_variants",
    "is_known_variant",
    "ensure_known_variant",
    # state_value phase
    "STATE_VALUE_PHASE",
    "StateValueContext",
    "StateValueHook",
    "register_state_value_hook",
    "apply_state_value_hooks",
    # prompt phase
    "PROMPT_PHASE",
    "PromptContext",
    "PromptHook",
    "register_prompt_hook",
    "apply_prompt_hooks",
]
