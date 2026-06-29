from __future__ import annotations

from ttt_discover.algorithms.base import Algorithm
from ttt_discover.eval_runners import EvalRunner
from ttt_discover.tasks import Task


class TTTDiscoverAlgorithm(Algorithm):
    NAME = "ttt_discover"

    async def run(self, task: Task, eval_runner: EvalRunner) -> None:
        from ttt_discover.algorithms.ttt_discover.loop import run

        await run(
            self.cfg,
            task=task,
            eval_runner=eval_runner,
        )
