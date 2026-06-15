from pathlib import Path
import math
import os
import sys
import tempfile

sys.path.insert(0, str(Path.cwd()))

from libkernelbot.consts import RankCriterion, SubmissionMode
from libkernelbot.run_eval import run_config
from libkernelbot.task import build_task_config, make_task_definition


def compute_score_us(result, task):
    run = result.runs["leaderboard"].run
    n = int(run.result["benchmark-count"])
    means_ns = [float(run.result[f"benchmark.{i}.mean"]) for i in range(n)]
    if task.ranking_by == RankCriterion.LAST:
        score_ns = means_ns[-1]
    elif task.ranking_by == RankCriterion.MEAN:
        score_ns = sum(means_ns) / len(means_ns)
    elif task.ranking_by == RankCriterion.GEOM:
        score_ns = math.exp(sum(math.log(x) for x in means_ns) / len(means_ns))
    else:
        raise ValueError(f"Unsupported ranking_by: {task.ranking_by}")
    return score_ns / 1000.0


def main():
    print(f"CUDA_VISIBLE_DEVICES {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"TORCH_CUDA_ARCH_LIST {os.environ.get('TORCH_CUDA_ARCH_LIST')}")
    task_path = Path("task.yml")
    if not task_path.exists():
        raise SystemExit("task.yml is missing")
    if not Path("submission.py").exists():
        raise SystemExit("submission.py is missing")

    task = make_task_definition(task_path).task
    code = Path("submission.py").read_text()
    config = build_task_config(
        task=task,
        submission_content=code,
        arch=None,
        mode=SubmissionMode.LEADERBOARD,
    )
    eval_root = Path.cwd() / "eval_tmp"
    eval_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="run_", dir=eval_root) as tmp_dir:
        old_cwd = os.getcwd()
        os.chdir(tmp_dir)
        try:
            result = run_config(config)
        finally:
            os.chdir(old_cwd)

    test = result.runs.get("test")
    if test is None or test.run is None or not test.run.passed:
        print("TEST FAILED")
        if test is not None and test.run is not None:
            print(test.run.stdout)
            print(test.run.stderr)
            print(test.run.result)
        raise SystemExit(1)
    leaderboard = result.runs.get("leaderboard")
    if leaderboard is None or leaderboard.run is None or not leaderboard.run.passed:
        print("LEADERBOARD FAILED")
        if leaderboard is not None and leaderboard.run is not None:
            print(leaderboard.run.stdout)
            print(leaderboard.run.stderr)
            print(leaderboard.run.result)
        raise SystemExit(1)
    print(f"score_us {compute_score_us(result, task):.6f}")


if __name__ == "__main__":
    main()
