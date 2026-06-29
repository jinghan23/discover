from ttt_discover.config import DiscoverConfig
from ttt_discover.runner import discover


def __getattr__(name: str):
    if name == "BaseRewardEvaluator":
        from ttt_discover.tasks.reward_evaluator import BaseRewardEvaluator

        return BaseRewardEvaluator
    if name == "SandboxRewardEvaluator":
        from ttt_discover.tasks.sandbox_reward_evaluator import SandboxRewardEvaluator

        return SandboxRewardEvaluator
    if name == "Environment":
        from ttt_discover.tasks import Environment

        return Environment
    if name == "Task":
        from ttt_discover.tasks import Task

        return Task
    if name == "State":
        from ttt_discover.algorithms.state import State

        return State
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "Environment",
    "Task",
    "DiscoverConfig",
    "discover",
    "State",
    "BaseRewardEvaluator",
    "SandboxRewardEvaluator",
]
