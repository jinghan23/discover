"""Evaluation runner interfaces.

Eval runners own the transport used to turn a candidate into a reward. Algorithms
do not call ``reward_function.get_reward`` directly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable

from ttt_discover.tasks.base import VerifyResult

AutonomousPromptBuilder = Callable[..., str]


class EvalRunner(ABC):
    NAME = "base"

    @abstractmethod
    def evaluate(self, env: Any, generation: str) -> VerifyResult:
        """Score ``generation`` for the task context represented by ``env``."""

    def build_autonomous_prompt(
        self,
        env: Any,
        *,
        prompt: str,
        workspace: Path,
        eval_timeout: int,
        num_cpus_per_task: int,
    ) -> str | None:
        """Optionally build an eval-runner-specific autonomous prompt."""
        del env, prompt, workspace, eval_timeout, num_cpus_per_task
        return None
