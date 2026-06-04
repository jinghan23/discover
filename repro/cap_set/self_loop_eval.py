from __future__ import annotations

import argparse
import json
from pathlib import Path

from examples.cap_set_priority.env import CapSetPriorityRewardEvaluator
from ttt_discover import State


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one cap-set priority candidate and print compact JSON."
    )
    parser.add_argument("--candidate", required=True, help="Path to a Python file.")
    parser.add_argument("--dimension", type=int, default=8)
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--eval-timeout", type=int, default=45)
    parser.add_argument("--num-cpus-per-task", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidate_path = Path(args.candidate)
    code = candidate_path.read_text(encoding="utf-8")
    if "```python" not in code:
        code = f"```python\n{code.rstrip()}\n```"

    log_dir = args.log_dir or str(candidate_path.parent / "eval_tmp")
    evaluator = CapSetPriorityRewardEvaluator(
        problem_type=str(args.dimension),
        log_dir=log_dir,
        num_cpus_per_task=args.num_cpus_per_task,
        eval_timeout=args.eval_timeout,
    )
    out = evaluator.get_reward(
        code,
        state=State(timestep=-1, construction=[], code="", value=0.0),
    )
    compact = {
        "reward": out.get("reward"),
        "raw_score": out.get("raw_score"),
        "correctness": out.get("correctness"),
        "msg": out.get("msg"),
        "metrics": out.get("metrics", {}),
    }
    print(json.dumps(compact, sort_keys=True))


if __name__ == "__main__":
    main()
