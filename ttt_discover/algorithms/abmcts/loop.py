from __future__ import annotations

import glob
import os
import re
from typing import Any

from ttt_discover.algorithms.runtime import Loop
from ttt_discover.algorithms.abmcts.sampler import ABMCTSSampler
from ttt_discover.config import DiscoverConfig
from ttt_discover.eval_runners import EvalRunner
from ttt_discover.tasks import Task


def latest_abmcts_step(log_path: str) -> int:
    latest = 0
    pattern = os.path.join(log_path, "abmcts_sampler_step_*.json")
    for path in glob.glob(pattern):
        match = re.search(r"abmcts_sampler_step_(\d+)\.json$", path)
        if match:
            latest = max(latest, int(match.group(1)))
    return latest


class ABMCTSLoop(Loop):
    """Shared Codex loop using an AB-MCTS-A tree sampler."""

    sampler_cls = ABMCTSSampler

    def latest_step(self) -> int:
        return latest_abmcts_step(self.cfg.log_path)

    def build_prompt(
        self,
        env: Any,
        parent_state: Any,
        *,
        step_idx: int,
        group_idx: int,
        sample_idx: int,
    ) -> str:
        build_abmcts_prompt = getattr(env, "get_abmcts_question", None)
        if build_abmcts_prompt is None:
            return super().build_prompt(
                env,
                parent_state,
                step_idx=step_idx,
                group_idx=group_idx,
                sample_idx=sample_idx,
            )
        meta = getattr(parent_state, "metadata", {}) or {}
        abmcts_meta = meta.get("abmcts", {}) if isinstance(meta, dict) else {}
        return build_abmcts_prompt(
            action=abmcts_meta.get("action"),
            parent_state=parent_state,
            step_idx=step_idx,
            group_idx=group_idx,
            sample_idx=sample_idx,
        )


async def run(
    cfg: DiscoverConfig,
    *,
    task: Task | None = None,
    eval_runner: EvalRunner,
) -> None:
    await ABMCTSLoop(cfg, eval_runner=eval_runner, task=task).run()


__all__ = ["ABMCTSLoop", "latest_abmcts_step", "run"]
