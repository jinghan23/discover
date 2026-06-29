from ttt_discover.tasks.base import Environment, Task, VerifyResult


def __getattr__(name: str):
    if name == "BaseRewardEvaluator":
        from ttt_discover.tasks.reward_evaluator import BaseRewardEvaluator

        return BaseRewardEvaluator
    if name == "SandboxRewardEvaluator":
        from ttt_discover.tasks.sandbox_reward_evaluator import SandboxRewardEvaluator

        return SandboxRewardEvaluator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BaseRewardEvaluator",
    "Environment",
    "SandboxRewardEvaluator",
    "Task",
    "VerifyResult",
]
