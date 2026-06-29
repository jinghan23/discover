from __future__ import annotations

from ttt_discover.algorithms.base import Algorithm
from ttt_discover.eval_runners import EvalRunner
from ttt_discover.tasks import Task


class OpenEvolveAlgorithm(Algorithm):
    NAME = "openevolve"

    async def run(self, task: Task, eval_runner: EvalRunner) -> None:
        del eval_runner

        from ttt_discover.algorithms.openevolve.loop import run

        await run(self.cfg, task=task)
