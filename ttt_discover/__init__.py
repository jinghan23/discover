from ttt_discover.codex_utils.discovery import DiscoverConfig, discover


def _tinker_import_failed(exc: ModuleNotFoundError) -> bool:
    return exc.name == "tinker"


def __getattr__(name: str):
    if name == "BaseRewardEvaluator":
        from ttt_discover.environments.base_reward_evaluator import BaseRewardEvaluator

        return BaseRewardEvaluator
    if name == "SandboxRewardEvaluator":
        from ttt_discover.environments.sandbox_reward_evaluator import SandboxRewardEvaluator

        return SandboxRewardEvaluator
    if name == "Environment":
        try:
            from ttt_discover.tinker_utils.dataset_builder import Environment
        except ModuleNotFoundError as exc:
            if not _tinker_import_failed(exc):
                raise
            from ttt_discover.codex_utils.environment import Environment

        return Environment
    if name == "State":
        try:
            from ttt_discover.tinker_utils.state import State
        except ModuleNotFoundError as exc:
            if not _tinker_import_failed(exc):
                raise
            from ttt_discover.codex_utils.runtime import State

        return State
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "Environment",
    "DiscoverConfig",
    "discover",
    "State",
    "BaseRewardEvaluator",
    "SandboxRewardEvaluator",
]
