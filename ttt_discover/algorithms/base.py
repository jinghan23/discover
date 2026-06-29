from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from ttt_discover.eval_runners import EvalRunner
from ttt_discover.tasks import Task


class Algorithm(ABC):
    NAME: ClassVar[str]

    def __init__(self, cfg: Any):
        self.cfg = cfg

    @abstractmethod
    async def run(self, task: Task, eval_runner: EvalRunner) -> None:
        """Run this search strategy against a task and eval_runner."""
