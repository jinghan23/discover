"""ARC WhestBench task integration."""

__all__ = ["WhestBenchEnv", "WhestBenchRewardEvaluator", "discover_whestbench"]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(name)
    from examples.aicrowd_whestbench import env

    return getattr(env, name)
