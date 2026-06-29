from __future__ import annotations

from typing import Any, Type

from ttt_discover.algorithms.abmcts.algorithm import ABMCTSAlgorithm
from ttt_discover.algorithms.autoevolve.algorithm import AutoEvolveAlgorithm
from ttt_discover.algorithms.base import Algorithm
from ttt_discover.algorithms.openevolve.algorithm import OpenEvolveAlgorithm
from ttt_discover.algorithms.ttt_discover.algorithm import TTTDiscoverAlgorithm
from ttt_discover.eval_runners import EvalRunner
from ttt_discover.tasks import Task

_ALGORITHMS: dict[str, Type[Algorithm]] = {
    algo.NAME: algo
    for algo in (
        TTTDiscoverAlgorithm,
        ABMCTSAlgorithm,
        AutoEvolveAlgorithm,
        OpenEvolveAlgorithm,
    )
}


def get_algorithm(name: str) -> Type[Algorithm]:
    if name not in _ALGORITHMS:
        raise KeyError(f"Unknown algorithm {name!r}. Available: {sorted(_ALGORITHMS)}")
    return _ALGORITHMS[name]


def list_algorithms() -> list[str]:
    return sorted(_ALGORITHMS)


async def run_algorithm(
    cfg: Any,
    *,
    task: Task,
    eval_runner: EvalRunner,
) -> None:
    await get_algorithm(cfg.algorithm)(cfg).run(task, eval_runner)
