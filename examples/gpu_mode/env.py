from __future__ import annotations

import contextlib
import math
import os
import re
import shlex
import shutil
import sys
import tempfile
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any

from ttt_discover import BaseRewardEvaluator, DiscoverConfig, Environment, State, discover
from ttt_discover.eval_runners.blackbox_eval import build_eval_client_source

GPU_MODE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = GPU_MODE_ROOT.parents[1]
LIB_ROOT = GPU_MODE_ROOT / "lib"
if str(LIB_ROOT) not in sys.path:
    sys.path.insert(0, str(LIB_ROOT))

from libkernelbot.consts import RankCriterion, SubmissionMode
from libkernelbot.run_eval import FullResult, run_config
from libkernelbot.task import LeaderboardTask, build_task_config, make_task_definition

from examples.gpu_mode.prompt import (
    MLA_DECODE_PROMPT,
    MLA_DECODE_PROMPT_END,
    TRIMUL_PROMPT,
)


_RUN_CONFIG_LOCK = threading.Lock()
_CODE_BLOCK_RE = re.compile(r"```(?:python|py|cuda)?\s*([\s\S]*?)\s*```")


def _task_yaml(problem_type: str) -> Path:
    task_map = {
        "trimul": GPU_MODE_ROOT / "lib" / "bioml" / "trimul" / "task.yml",
        "mla_decode_nvidia": GPU_MODE_ROOT / "lib" / "mla-decode" / "task.yml",
    }
    try:
        return task_map[problem_type]
    except KeyError as exc:
        raise ValueError(
            f"Unknown problem_type: {problem_type}. "
            "Must be 'trimul' or 'mla_decode_nvidia'"
        ) from exc


def _task_dir(problem_type: str) -> Path:
    return _task_yaml(problem_type).parent


@lru_cache(maxsize=None)
def load_task(problem_type: str = "trimul") -> LeaderboardTask:
    """Load a LeaderboardTask from its YAML definition."""
    return make_task_definition(_task_yaml(problem_type)).task


def _extract_submission_code(text: str) -> str:
    matches = list(_CODE_BLOCK_RE.finditer(text or ""))
    if matches:
        return matches[-1].group(1).strip() + "\n"
    return (text or "").strip() + "\n"


def _score_scale(problem_type: str) -> float:
    if problem_type == "trimul":
        return 1500.0
    if problem_type == "mla_decode_nvidia":
        return 5000.0
    raise ValueError(f"Unknown problem_type: {problem_type}")


def _target_us(problem_type: str) -> float:
    return 1000.0 if problem_type == "trimul" else 1700.0


def _compute_score_us(result: FullResult, task: LeaderboardTask) -> float:
    leaderboard = result.runs["leaderboard"].run
    if leaderboard is None:
        raise ValueError("Missing leaderboard run result")

    num_benchmarks = int(leaderboard.result["benchmark-count"])
    means_ns = [
        float(leaderboard.result[f"benchmark.{idx}.mean"])
        for idx in range(num_benchmarks)
    ]
    if not means_ns:
        raise ValueError("No benchmark means in leaderboard result")

    if task.ranking_by == RankCriterion.LAST:
        score_ns = means_ns[-1]
    elif task.ranking_by == RankCriterion.MEAN:
        score_ns = sum(means_ns) / len(means_ns)
    elif task.ranking_by == RankCriterion.GEOM:
        if any(value <= 0 for value in means_ns):
            raise ValueError(f"Cannot compute geometric mean for {means_ns}")
        score_ns = math.exp(sum(math.log(value) for value in means_ns) / len(means_ns))
    else:
        raise ValueError(f"Unsupported ranking criterion: {task.ranking_by}")

    return score_ns / 1000.0


def _format_run_outputs(result: FullResult) -> str:
    chunks: list[str] = []
    for name, eval_result in result.runs.items():
        run = eval_result.run
        if run is None:
            continue
        chunks.append(
            f"--- {name} ---\n"
            f"passed={run.passed} success={run.success} exit_code={run.exit_code}\n"
            f"stdout:\n{run.stdout}\n"
            f"stderr:\n{run.stderr}\n"
        )
    return "\n".join(chunks)


@contextlib.contextmanager
def _pushd(path: str | os.PathLike[str]):
    old_cwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_cwd)


def get_gpu_mode_error(msg: str, stdout: str = "") -> dict:
    return {
        "reward": 0.0,
        "msg": msg,
        "correctness": 0.0,
        "raw_score": -1_000_000.0,
        "result_construction": [],
        "stdout": stdout,
    }


class GpuModeRewardEvaluator(BaseRewardEvaluator):
    def __init__(self, *args, **kwargs):
        self.problem_type = kwargs.get("problem_type")
        self.log_dir = kwargs.get("log_dir") or ""
        self.eval_timeout = int(kwargs.get("eval_timeout", 1200))
        self.num_cpus_per_task = max(1, int(kwargs.get("num_cpus_per_task", 1)))
        self.score_scale = _score_scale(self.problem_type)
        self.task = load_task(self.problem_type)

    def _tmp_root(self) -> Path:
        base = Path(self.log_dir) if self.log_dir else Path(tempfile.gettempdir())
        root = base / "gpu_mode_local_eval"
        root.mkdir(parents=True, exist_ok=True)
        return root

    def get_reward(self, code: str, state: State) -> dict:
        del state
        submission_code = _extract_submission_code(code)
        if not submission_code.strip():
            return get_gpu_mode_error("Empty submission.")
        if "def custom_kernel" not in submission_code:
            return get_gpu_mode_error("Code must define custom_kernel.")
        if "@triton.jit" not in submission_code:
            return get_gpu_mode_error("Code must contain @triton.jit.")
        if self.problem_type == "trimul" and "identity" in submission_code.lower():
            return get_gpu_mode_error("Identity kernel is not allowed.")

        config = build_task_config(
            task=self.task,
            submission_content=submission_code,
            arch=None,
            mode=SubmissionMode.LEADERBOARD,
        )

        try:
            with tempfile.TemporaryDirectory(
                prefix=f"{self.problem_type}_",
                dir=self._tmp_root(),
            ) as tmp_dir:
                # run_config writes task files into cwd, so guard cwd globally.
                with _RUN_CONFIG_LOCK:
                    with _pushd(tmp_dir):
                        result = run_config(config)
        except Exception as exc:
            return get_gpu_mode_error(f"Error: Failed to run local eval: {exc}")

        stdout = _format_run_outputs(result)
        if not result.success:
            return get_gpu_mode_error(f"Error: Failed to run test: {result.error}.", stdout)

        test_result = result.runs.get("test")
        if test_result is None or test_result.run is None:
            return get_gpu_mode_error("Unexpected result: Failed to find test results.", stdout)
        if not test_result.run.success:
            return get_gpu_mode_error(f"Failed to run tests: {test_result.run.stderr}", stdout)
        if not test_result.run.passed:
            return get_gpu_mode_error("Failed to pass test cases.", stdout)

        leaderboard_result = result.runs.get("leaderboard")
        if leaderboard_result is None or leaderboard_result.run is None:
            return get_gpu_mode_error("No leaderboard run in result.", stdout)
        if not leaderboard_result.run.success:
            return get_gpu_mode_error(
                f"Failed to run leaderboard: {leaderboard_result.run.stderr}",
                stdout,
            )
        if not leaderboard_result.run.passed:
            return get_gpu_mode_error("Failed leaderboard correctness checks.", stdout)

        try:
            score_us = _compute_score_us(result, self.task)
        except Exception as exc:
            return get_gpu_mode_error(f"Could not compute leaderboard score: {exc}", stdout)

        reward = self.score_scale / max(score_us, 1e-9)
        return {
            "reward": float(reward),
            "msg": (
                f"Overall leaderboard score (microseconds, "
                f"{self.task.ranking_by.value}): {score_us:.6f} us"
            ),
            "correctness": 1.0,
            "raw_score": float(score_us),
            "result_construction": [],
            "stdout": stdout,
            "metrics": {
                "gpu_mode/score_us": float(score_us),
                "gpu_mode/reward": float(reward),
                "gpu_mode/task": self.problem_type,
                "gpu_mode/system_gpu": result.system.gpu,
                "gpu_mode/device_count": result.system.device_count,
            },
        }


def _autonomous_eval_script(problem_type: str) -> str:
    del problem_type
    return """from pathlib import Path
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
"""


class GpuModeEnv(Environment):
    reward_function = GpuModeRewardEvaluator
    state_type = State

    @classmethod
    def create_initial_state(cls, problem_type: str) -> State:
        if problem_type == "mla_decode_nvidia":
            from examples.gpu_mode.prompt import (
                MLA_DECODE_INITIAL_STATE,
                MLA_DECODE_INITIAL_VALUE,
            )

            return State(
                timestep=-1,
                code=MLA_DECODE_INITIAL_STATE,
                value=MLA_DECODE_INITIAL_VALUE,
                construction=None,
            )
        if problem_type == "trimul":
            return State(
                timestep=-1,
                code="",
                value=-1_000_000.0,
                construction=None,
            )
        raise ValueError(f"Unknown problem_type: {problem_type}")

    def _should_keep_code_separators(self) -> bool:
        return False

    def _get_code_languages(self) -> list[str]:
        return ["python"]

    def is_maximize(self) -> bool:
        return False

    def check_format(self, parsed_code: str) -> bool:
        return bool(parsed_code and "def custom_kernel" in parsed_code)

    def _initial_submission_code(self) -> str:
        if self.initial_state and self.initial_state.code:
            code = _extract_submission_code(self.initial_state.code)
            if code.strip():
                return code
        return (_task_dir(self.problem_type) / "submission.py").read_text()

    def build_autonomous_prompt(
        self,
        *,
        prompt: str,
        workspace: Path,
        eval_timeout: int,
        num_cpus_per_task: int,
    ) -> str:
        del eval_timeout, num_cpus_per_task
        task_dir = _task_dir(self.problem_type)
        for name in ("task.py", "utils.py", "reference.py", "eval.py", "task.yml"):
            shutil.copy2(task_dir / name, workspace / name)
        shutil.copytree(
            LIB_ROOT / "libkernelbot",
            workspace / "libkernelbot",
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        readme = task_dir / "README.md"
        if readme.exists():
            shutil.copy2(readme, workspace / "README.md")
        (workspace / "submission.py").write_text(
            self._initial_submission_code(),
            encoding="utf-8",
        )
        (workspace / "eval_candidate.py").write_text(
            _autonomous_eval_script(self.problem_type),
            encoding="utf-8",
        )

        env_parts = []
        for key in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "TORCH_CUDA_ARCH_LIST"):
            value = os.environ.get(key)
            if value:
                env_parts.append(f"{key}={shlex.quote(value)}")
        env_prefix = (" ".join(env_parts) + " ") if env_parts else ""
        evaluator_cmd = (
            f"cd {shlex.quote(str(workspace))} && {env_prefix}python eval_candidate.py"
        )
        return f"""{prompt}

--- Autonomous GPUMode Search Mode ---
You may inspect files and run shell commands, but keep all edits inside this workspace:
{workspace}

Editable candidate:
{workspace / "submission.py"}

The copied task files (`task.py`, `utils.py`, `reference.py`, `eval.py`, `task.yml`)
and `libkernelbot/` are for inspection and local testing. The evaluator below uses
the copied `task.yml` and runs leaderboard mode, so it checks both tests and
leaderboard benchmarks for this task.

Run this evaluator after each revision:
{evaluator_cmd}

Do not edit `eval_candidate.py`, `task.yml`, `libkernelbot/`, or the copied task/eval
files to improve a score. Only `submission.py` is a valid candidate artifact, and
the outer runner will re-score it with trusted repository files.

When done, put the best implementation in:
{workspace / "submission.py"}

The outer discovery runner will score the final contents of that `submission.py`
with trusted repository files outside the Codex workspace before considering any
code block in your final response.
"""

    def build_blackbox_autonomous_prompt(
        self,
        *,
        prompt: str,
        workspace: Path,
        eval_timeout: int,
        num_cpus_per_task: int,
        socket_path: str | None = None,
        host: str = "127.0.0.1",
        port: int | None = None,
    ) -> str:
        del num_cpus_per_task
        socket_path = socket_path or os.environ.get("TTT_BLACKBOX_EVAL_SOCKET")
        host = os.environ.get("TTT_BLACKBOX_EVAL_HOST") or host
        env_port = os.environ.get("TTT_BLACKBOX_EVAL_PORT")
        if port is None and env_port:
            port = int(env_port)
        if socket_path is None and port is None:
            raise ValueError(
                "Blackbox autonomous GPUMode requires a socket path or TCP port."
            )

        (workspace / "submission.py").write_text(
            self._initial_submission_code(),
            encoding="utf-8",
        )
        (workspace / "eval_client.py").write_text(
            build_eval_client_source(
                problem_type=self.problem_type,
                socket_path=socket_path,
                host=host,
                port=port,
                timeout_s=max(1.0, float(eval_timeout)),
            ),
            encoding="utf-8",
        )

        evaluator_cmd = f"cd {shlex.quote(str(workspace))} && python eval_client.py"
        return f"""{prompt}

--- Autonomous GPUMode Blackbox Search Mode ---
You may inspect files and run shell commands, but keep all edits inside this workspace:
{workspace}

Editable candidate:
{workspace / "submission.py"}

The local evaluator is a blackbox service. This workspace intentionally contains
only `submission.py` and `eval_client.py`; hidden task files, reference code,
test cases, and evaluator internals are not available here.

Run this evaluator after each revision:
{evaluator_cmd}

Do not edit `eval_client.py` to improve a score. Only `submission.py` is a valid
candidate artifact. The evaluator returns only compact pass/fail/score feedback,
and the outer runner will re-score the final `submission.py` with trusted files.

When done, put the best implementation in:
{workspace / "submission.py"}
"""

    def get_question(self) -> str:
        """Build prompt from template, injecting previous code from state."""
        state = self.initial_state
        target = _target_us(self.problem_type)
        state_ctx = state.to_prompt(
            target,
            metric_name="runtime (microseconds)",
            maximize=False,
            language="python",
        )

        if self.problem_type == "trimul":
            # Original hardware prompt kept for reference:
            # - You must use Triton 3.3.1 and these kernels will be run on an H100-class NVIDIA GPU.
            return f"""{TRIMUL_PROMPT}

{state_ctx}

Rules:
- The tensor arguments passed in will already be on your CUDA device.
- Define all of your code in one final ```python ``` block.
- We will test correctness on multiple input shapes; support all potential test cases.
- You are allowed to use mixed precision computations, but make sure your final output is float32.
- You must use Triton 3.3.1. These kernels will be run on an A800/A100-class NVIDIA GPU (Ampere SM80, CUDA arch 8.0; A800-SXM4-80GB target). Do not use Hopper/SM90-only features such as TMA, WGMMA, or FP8-only paths.
- You do not have to implement everything in Triton; PyTorch helper operations are allowed. However, implement at least part of the computation in a kernel.
- Include a short docstring at the top summarizing your algorithm.
"""

        if self.problem_type == "mla_decode_nvidia":
            return f"""{MLA_DECODE_PROMPT}

{state_ctx}

{MLA_DECODE_PROMPT_END}
"""

        raise ValueError(
            f"Unknown problem_type: {self.problem_type}. "
            "Must be 'trimul' or 'mla_decode_nvidia'"
        )


def discover_gpu_mode(problem_type: str):
    config = DiscoverConfig(
        env_type=GpuModeEnv,
        problem_type=problem_type,
        eval_timeout=530,
        experiment_name=f"test-gpu-mode-{problem_type}-run",
        wandb_project="gpu-mode",
    )
    discover(config)


if __name__ == "__main__":
    discover_gpu_mode("trimul")
    # discover_gpu_mode("mla_decode_nvidia")
