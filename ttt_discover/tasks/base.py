"""Task-side interfaces for Codex discovery.

``Task`` is the problem-level object algorithms consume. It owns the environment
type and builds per-state environment contexts. ``Environment`` is retained as a
compatibility base class for the existing examples; new tasks should implement
the same hooks there and be wrapped by ``Task`` through ``Task.from_env_type``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ttt_discover.algorithms.state import State


@dataclass
class VerifyResult:
    reward: float
    msg: str
    correctness: float
    raw_score: float
    result_construction: Any
    stdout: str
    metrics: dict[str, Any] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_reward_dict(cls, out: Any) -> "VerifyResult":
        if not isinstance(out, dict):
            reward = float(out)
            return cls(
                reward=reward,
                msg="",
                correctness=1.0,
                raw_score=reward,
                result_construction=None,
                stdout="",
                metrics={},
                details={},
            )
        return cls(
            reward=out["reward"],
            msg=out.get("msg", ""),
            correctness=out.get("correctness", 0.0),
            raw_score=out.get("raw_score", out.get("reward", 0.0)),
            result_construction=out.get("result_construction", None),
            stdout=out.get("stdout", ""),
            metrics=out.get("metrics", {}),
            details=out.get("details") or {},
        )


@dataclass
class Task:
    """Problem-level adapter exposed to algorithms."""

    env_type: type
    problem_type: str
    config: Any

    @classmethod
    def from_env_type(cls, env_type: type, *, problem_type: str, config: Any) -> "Task":
        return cls(env_type=env_type, problem_type=problem_type, config=config)

    @classmethod
    def from_config(cls, cfg: Any) -> "Task":
        if cfg.env_type is None:
            raise ValueError("env_type is required")
        return cls(env_type=cfg.env_type, problem_type=cfg.problem_type, config=cfg)

    @property
    def state_type(self) -> type:
        return getattr(self.env_type, "state_type", State)

    def create_initial_state(self) -> State:
        return self.env_type.create_initial_state(self.problem_type)

    def make_env(self, state: State, *, sampler: Any | None = None) -> Any:
        return self.env_type(initial_state=state, sampler=sampler, config=self.config)

    def get_prompt(self, env: Any) -> str:
        return env.get_question()

    def code_languages(self, env: Any) -> list[str]:
        get_languages = getattr(env, "_get_code_languages", None)
        return get_languages() if get_languages is not None else ["python"]

    def keep_code_separators(self, env: Any) -> bool:
        should_keep = getattr(env, "_should_keep_code_separators", None)
        return bool(should_keep()) if should_keep is not None else True

    def check_candidate_format(self, env: Any, parsed_code: str) -> bool:
        if parsed_code is None or parsed_code.strip() == "":
            return False
        check_format = getattr(env, "check_format", None)
        if check_format is None:
            return True
        try:
            return bool(check_format(parsed_code))
        except Exception:
            return False

    def build_metrics(
        self,
        env: Any,
        outs: VerifyResult,
        *,
        response: str,
        parsed_code: str,
        correct_format: bool,
    ) -> dict[str, Any]:
        format_score = float(correct_format)
        message = {"role": "assistant", "content": response}
        build_metrics = getattr(env, "_build_metrics", None)
        if build_metrics is not None:
            try:
                return build_metrics(outs, format_score, message, parsed_code)
            except Exception:
                pass
        return {
            "format": format_score,
            "reward": outs.reward,
            "correctness": outs.correctness,
            "raw_score": outs.raw_score if outs.correctness > 0 else None,
            "initial_raw_score": getattr(env.initial_state, "value", None),
            "msg": outs.msg,
            "prompt": self.get_prompt(env),
            "response": response,
            "parsed_code": parsed_code,
            **(outs.metrics or {}),
        }

    def create_next_state(
        self,
        env: Any,
        *,
        step_idx: int,
        parsed_code: str,
        outs: VerifyResult,
    ) -> State | None:
        if outs.correctness <= 0:
            return None
        create_next_state = getattr(env, "_create_next_state", None)
        if create_next_state is None:
            return None
        return create_next_state(step_idx, parsed_code, outs)

    def autonomous_prompt_builder(self, eval_runner: Any, env: Any) -> Any | None:
        if (
            getattr(eval_runner, "NAME", None) == "blackbox"
            or getattr(env, "build_autonomous_prompt", None) is not None
        ):
            return lambda **kwargs: eval_runner.build_autonomous_prompt(env, **kwargs)
        return None


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
