"""Algorithm runtime package."""


def __getattr__(name: str):
    if name == "Algorithm":
        from ttt_discover.algorithms.base import Algorithm

        return Algorithm
    if name in {"get_algorithm", "list_algorithms", "run_algorithm"}:
        from ttt_discover.algorithms import registry

        return getattr(registry, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["Algorithm", "get_algorithm", "list_algorithms", "run_algorithm"]
