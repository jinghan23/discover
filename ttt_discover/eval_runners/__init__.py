from __future__ import annotations

from ttt_discover.config import DiscoverConfig, EvalRunnerName
from ttt_discover.eval_runners.base import EvalRunner
from ttt_discover.eval_runners.blackbox import BlackboxRunner
from ttt_discover.eval_runners.in_process import InProcessRunner


def build_eval_runner(
    cfg: DiscoverConfig,
    *,
    socket_path: str | None = None,
) -> EvalRunner:
    name: EvalRunnerName = cfg.eval_runner
    if name == "auto":
        name = "blackbox" if cfg.algorithm == "autoevolve" else "in_process"
    if name == "in_process":
        return InProcessRunner()
    if name == "blackbox":
        return BlackboxRunner.from_config(cfg, socket_path=socket_path)
    raise ValueError(f"Unknown eval runner: {name}")


__all__ = [
    "BlackboxRunner",
    "EvalRunner",
    "EvalRunnerName",
    "InProcessRunner",
    "build_eval_runner",
]
