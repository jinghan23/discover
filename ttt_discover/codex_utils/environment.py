"""Lightweight environment helpers for Codex-only discovery.

This mirrors the task-facing parts of ``tinker_utils.dataset_builder.Environment``
without importing Tinker. The Codex no-finetune runner constructs env objects
with ``object.__new__`` and uses these hooks directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ttt_discover.codex_utils.runtime import State


@dataclass
class VerifyResult:
    reward: float
    msg: str
    correctness: float
    raw_score: float
    result_construction: Any
    stdout: str
    metrics: dict[str, Any] = field(default_factory=dict)


class Environment:
    state_type: type = State

    @classmethod
    def create_initial_state(cls, problem_type: str) -> State:
        return cls.state_type(timestep=-1, construction=None, code="", value=0.0)

    def __init__(
        self,
        renderer: Any | None = None,
        initial_state: State | None = None,
        sampler: Any | None = None,
        config: Any | None = None,
    ):
        self.renderer = renderer
        self.config = config
        self.timeout = getattr(config, "timeout", 8000.0)
        self.num_cpus_per_task = getattr(config, "num_cpus_per_task", 1)
        self.eval_timeout = getattr(config, "eval_timeout", 300)
        self.log_path = getattr(config, "log_path", "")
        self.initial_state = initial_state
        self.sampler = sampler
        self.state = initial_state
        self.problem_type = getattr(config, "problem_type", "")

    def get_question(self) -> str:
        raise NotImplementedError

    def is_maximize(self) -> bool:
        return True

    def _create_next_state(
        self,
        step_idx: int,
        parsed_code: str,
        outs: VerifyResult,
    ) -> State:
        return self.state_type(
            timestep=step_idx,
            construction=outs.result_construction,
            code=parsed_code,
            value=outs.raw_score if self.is_maximize() else -outs.raw_score,
            observation=outs.stdout,
        )

    def _build_metrics(
        self,
        outs: VerifyResult,
        correct_format: bool,
        message: dict,
        parsed_code: str,
    ) -> dict[str, Any]:
        return {
            "format": float(correct_format),
            "reward": outs.reward,
            "correctness": outs.correctness,
            "raw_score": outs.raw_score if outs.correctness > 0 else None,
            "initial_raw_score": getattr(self.initial_state, "value", None),
            "msg": outs.msg,
            "prompt": self.get_question(),
            "response": message["content"],
            "parsed_code": parsed_code,
            **(outs.metrics or {}),
        }

    def _get_code_languages(self) -> list[str]:
        return ["python"]

    def _should_keep_code_separators(self) -> bool:
        return True

    def check_format(self, parsed_code: str) -> bool:
        return bool(parsed_code and parsed_code.strip())

    def _run_verification(
        self,
        generation: str,
        problem_type: str,
        log_path: str,
        state: State,
    ) -> VerifyResult:
        task = self.reward_function(
            problem_type=problem_type,
            log_dir=log_path,
            eval_timeout=self.eval_timeout,
            num_cpus_per_task=self.num_cpus_per_task,
        )
        out = task.get_reward(generation, state=state)
        return VerifyResult(
            reward=out["reward"],
            msg=out.get("msg", ""),
            correctness=out.get("correctness", 0.0),
            raw_score=out.get("raw_score", out.get("reward", 0.0)),
            result_construction=out.get("result_construction", None),
            stdout=out.get("stdout", ""),
            metrics=out.get("metrics", {}),
        )
