from __future__ import annotations

from ttt_discover.algorithms.runtime import Loop
from ttt_discover.algorithms.ttt_discover.sampler import PUCTSampler
from ttt_discover.config import DiscoverConfig
from ttt_discover.eval_runners import EvalRunner
from ttt_discover.tasks import Task


class TTTDiscoverLoop(Loop):
    """TTT-Discover uses the shared Loop with the PUCT state sampler."""

    sampler_cls = PUCTSampler


async def run(
    cfg: DiscoverConfig,
    *,
    task: Task | None = None,
    eval_runner: EvalRunner,
) -> None:
    await TTTDiscoverLoop(cfg, eval_runner=eval_runner, task=task).run()


__all__ = ["TTTDiscoverLoop", "run"]
