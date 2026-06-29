from __future__ import annotations

from pathlib import Path
from typing import Any

from ttt_discover.eval_runners.base import EvalRunner
from ttt_discover.tasks.base import VerifyResult


class InProcessRunner(EvalRunner):
    NAME = "in_process"

    def evaluate(self, env: Any, generation: str) -> VerifyResult:
        reward_evaluator = env.reward_function(
            problem_type=env.problem_type,
            log_dir=env.log_path,
            eval_timeout=env.eval_timeout,
            num_cpus_per_task=env.num_cpus_per_task,
        )
        return VerifyResult.from_reward_dict(
            reward_evaluator.get_reward(generation, state=env.state)
        )

    def build_autonomous_prompt(
        self,
        env: Any,
        *,
        prompt: str,
        workspace: Path,
        eval_timeout: int,
        num_cpus_per_task: int,
    ) -> str | None:
        builder = getattr(env, "build_autonomous_prompt", None)
        if builder is None:
            return None
        return builder(
            prompt=prompt,
            workspace=workspace,
            eval_timeout=eval_timeout,
            num_cpus_per_task=num_cpus_per_task,
        )
