from ttt_discover.discovery import DiscoverConfig, discover
from ttt_discover.environments.base_reward_evaluator import BaseRewardEvaluator
from ttt_discover.environments.sandbox_reward_evaluator import SandboxRewardEvaluator


def __getattr__(name: str):
    if name == "Environment":
        from ttt_discover.tinker_utils.dataset_builder import Environment

        return Environment
    if name == "State":
        from ttt_discover.tinker_utils.state import State

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
