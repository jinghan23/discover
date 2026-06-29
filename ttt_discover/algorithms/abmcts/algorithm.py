from __future__ import annotations

from ttt_discover.algorithms.base import Algorithm
from ttt_discover.eval_runners import EvalRunner
from ttt_discover.tasks import Task


class ABMCTSAlgorithm(Algorithm):
    NAME = "abmcts"

    async def run(self, task: Task, eval_runner: EvalRunner) -> None:
        from ttt_discover.algorithms.abmcts.loop import run

        await run(
            self.cfg,
            task=task,
            eval_runner=eval_runner,
        )
