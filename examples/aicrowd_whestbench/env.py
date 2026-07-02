from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np

from ttt_discover import BaseRewardEvaluator, DiscoverConfig, Environment, State, discover

from examples.aicrowd_whestbench.prompt import WHESTBENCH_PROMPT


_CODE_BLOCK_RE = re.compile(r"```(?:python|py)?\s*([\s\S]*?)```")
_DEFAULT_WIDTH = 64
_DEFAULT_DEPTH = 8
_DEFAULT_REFERENCE_SAMPLES = 4096
_DEFAULT_BUDGET = int(1e9)
_DEFAULT_SEEDS = (0, 1, 2)
_TARGET_FINAL_LAYER_MSE = 1.0e-6


@dataclass(frozen=True)
class WhestMlp:
    width: int
    depth: int
    weights: list[np.ndarray]
    seed: int


class CandidateError(ValueError):
    """Candidate code failed the local WhestBench contract."""


def _extract_python_code(text: str) -> str:
    matches = list(_CODE_BLOCK_RE.finditer(text or ""))
    if matches:
        return matches[-1].group(1).strip() + "\n"
    return (text or "").strip() + "\n"


def build_mlp(width: int, depth: int, seed: int) -> WhestMlp:
    if width < 1 or depth < 1:
        raise ValueError(f"width and depth must be positive, got {width=} {depth=}")
    rng = np.random.default_rng(seed)
    scale = math.sqrt(2.0 / width)
    weights = [
        (rng.standard_normal((width, width)) * scale).astype(np.float32)
        for _ in range(depth)
    ]
    return WhestMlp(width=width, depth=depth, weights=weights, seed=seed)


def monte_carlo_layer_means(
    mlp: WhestMlp,
    n_samples: int,
    *,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n_samples, mlp.width)).astype(np.float32)
    rows: list[np.ndarray] = []
    for w in mlp.weights:
        x = np.maximum(x @ w, 0.0)
        rows.append(np.mean(x, axis=0))
    return np.stack(rows, axis=0)


def _normal_pdf(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _normal_cdf_approx(x: np.ndarray) -> np.ndarray:
    # GELU-style tanh approximation; avoids a scipy dependency for local smoke tests.
    return 0.5 * (
        1.0
        + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * np.power(x, 3)))
    )


def mean_propagation_estimate(mlp: WhestMlp, budget: int = _DEFAULT_BUDGET) -> np.ndarray:
    del budget
    mu = np.zeros(mlp.width, dtype=np.float64)
    var = np.ones(mlp.width, dtype=np.float64)
    rows: list[np.ndarray] = []
    for w in mlp.weights:
        w64 = w.astype(np.float64, copy=False)
        mu_pre = w64.T @ mu
        var_pre = (w64 * w64).T @ var
        var_pre = np.maximum(var_pre, 1e-12)
        sigma_pre = np.sqrt(var_pre)
        alpha = mu_pre / sigma_pre
        phi = _normal_pdf(alpha)
        cdf = _normal_cdf_approx(alpha)
        mu = mu_pre * cdf + sigma_pre * phi
        ez2 = (mu_pre * mu_pre + var_pre) * cdf + mu_pre * sigma_pre * phi
        var = np.maximum(ez2 - mu * mu, 0.0)
        rows.append(mu.astype(np.float64, copy=True))
    return np.stack(rows, axis=0)


INITIAL_ESTIMATOR_CODE = '''import math
import numpy as np


def _normal_pdf(x):
    return np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _normal_cdf_approx(x):
    return 0.5 * (
        1.0
        + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * np.power(x, 3)))
    )


def estimate(mlp, budget):
    """Diagonal mean/variance propagation baseline for WhestBench."""
    del budget
    mu = np.zeros(mlp.width, dtype=np.float64)
    var = np.ones(mlp.width, dtype=np.float64)
    rows = []
    for w in mlp.weights:
        w64 = w.astype(np.float64, copy=False)
        mu_pre = w64.T @ mu
        var_pre = (w64 * w64).T @ var
        var_pre = np.maximum(var_pre, 1e-12)
        sigma_pre = np.sqrt(var_pre)
        alpha = mu_pre / sigma_pre
        phi = _normal_pdf(alpha)
        cdf = _normal_cdf_approx(alpha)
        mu = mu_pre * cdf + sigma_pre * phi
        ez2 = (mu_pre * mu_pre + var_pre) * cdf + mu_pre * sigma_pre * phi
        var = np.maximum(ez2 - mu * mu, 0.0)
        rows.append(mu.astype(np.float64, copy=True))
    return np.stack(rows, axis=0)
'''


def _call_estimator(fn: Callable[..., Any], mlp: WhestMlp, budget: int) -> np.ndarray:
    try:
        prediction = fn(mlp, budget)
    except TypeError:
        prediction = fn(mlp)
    arr = np.asarray(prediction, dtype=np.float64)
    if arr.shape != (mlp.depth, mlp.width):
        raise CandidateError(
            f"prediction has shape {arr.shape}, expected {(mlp.depth, mlp.width)}"
        )
    if not np.all(np.isfinite(arr)):
        raise CandidateError("prediction contains non-finite values")
    return arr


def _load_estimator(code: str) -> Callable[[WhestMlp, int], np.ndarray]:
    namespace: dict[str, Any] = {
        "__builtins__": __builtins__,
        "math": math,
        "np": np,
        "numpy": np,
    }
    try:
        exec(compile(code, "<whestbench-candidate>", "exec"), namespace, namespace)
    except Exception as exc:
        raise CandidateError(f"could not execute candidate code: {exc}") from exc

    if callable(namespace.get("estimate")):
        return namespace["estimate"]
    if callable(namespace.get("predict")):
        return namespace["predict"]

    estimator_cls = namespace.get("Estimator")
    if estimator_cls is not None:
        try:
            estimator = estimator_cls()
            if hasattr(estimator, "setup"):
                estimator.setup(SimpleNamespace(seed=0))
        except Exception as exc:
            raise CandidateError(f"could not construct Estimator: {exc}") from exc
        predict = getattr(estimator, "predict", None)
        if callable(predict):
            return predict

    raise CandidateError(
        "candidate must define estimate(mlp, budget), predict(mlp, budget), "
        "or class Estimator with predict()"
    )


def _score_callable(
    estimator: Callable[[WhestMlp, int], np.ndarray],
    *,
    width: int,
    depth: int,
    reference_samples: int,
    seeds: tuple[int, ...],
    budget: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    final_mses: list[float] = []
    all_mses: list[float] = []

    for seed in seeds:
        mlp = build_mlp(width=width, depth=depth, seed=seed)
        reference = monte_carlo_layer_means(
            mlp,
            reference_samples,
            seed=seed + 10_000,
        )
        prediction = _call_estimator(estimator, mlp, budget)
        err = np.square(prediction - reference)
        final_mse = float(np.mean(err[-1]))
        all_mse = float(np.mean(err))
        final_mses.append(final_mse)
        all_mses.append(all_mse)
        rows.append(
            {
                "seed": seed,
                "final_layer_mse": final_mse,
                "all_layer_mse": all_mse,
            }
        )

    return {
        "final_layer_mse": float(np.mean(final_mses)),
        "all_layer_mse": float(np.mean(all_mses)),
        "rows": rows,
    }


def _parse_seed_list(value: str | None) -> tuple[int, ...]:
    if not value:
        return _DEFAULT_SEEDS
    seeds = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    return seeds or _DEFAULT_SEEDS


def _suite_from_env() -> dict[str, Any]:
    return {
        "width": int(os.environ.get("WHEST_LOCAL_WIDTH", _DEFAULT_WIDTH)),
        "depth": int(os.environ.get("WHEST_LOCAL_DEPTH", _DEFAULT_DEPTH)),
        "reference_samples": int(
            os.environ.get("WHEST_LOCAL_REFERENCE_SAMPLES", _DEFAULT_REFERENCE_SAMPLES)
        ),
        "seeds": _parse_seed_list(os.environ.get("WHEST_LOCAL_SEEDS")),
        "budget": int(os.environ.get("WHEST_LOCAL_BUDGET", _DEFAULT_BUDGET)),
    }


def _format_rows(rows: list[dict[str, Any]]) -> str:
    lines = ["seed | final_layer_mse | all_layer_mse"]
    for row in rows:
        lines.append(
            f"{row['seed']} | {row['final_layer_mse']:.8g} | "
            f"{row['all_layer_mse']:.8g}"
        )
    return "\n".join(lines)


class WhestBenchRewardEvaluator(BaseRewardEvaluator):
    def __init__(self, *args, **kwargs):
        del args
        self.problem_type = kwargs.get("problem_type", "arc_whestbench_2026")
        self.suite = _suite_from_env()

    def get_reward(self, code: str, state: State) -> dict[str, Any]:
        del state
        candidate_code = _extract_python_code(code)
        if not candidate_code.strip():
            return self._failure("Empty candidate.")

        try:
            estimator = _load_estimator(candidate_code)
            candidate = _score_callable(estimator, **self.suite)
            baseline = _score_callable(mean_propagation_estimate, **self.suite)
        except CandidateError as exc:
            return self._failure(str(exc))
        except Exception as exc:
            return self._failure(f"Error while scoring candidate: {exc}")

        score = candidate["final_layer_mse"]
        baseline_score = baseline["final_layer_mse"]
        reward = baseline_score / max(score, 1e-12)
        stdout = _format_rows(candidate["rows"])
        return {
            "reward": float(reward),
            "msg": (
                f"final_layer_mse={score:.8g}; "
                f"baseline_final_layer_mse={baseline_score:.8g}; "
                f"all_layer_mse={candidate['all_layer_mse']:.8g}"
            ),
            "correctness": 1.0,
            "raw_score": float(score),
            "result_construction": candidate["rows"],
            "stdout": stdout,
            "metrics": {
                "whestbench/final_layer_mse": float(score),
                "whestbench/all_layer_mse": float(candidate["all_layer_mse"]),
                "whestbench/baseline_final_layer_mse": float(baseline_score),
                "whestbench/reward_vs_baseline": float(reward),
                "whestbench/width": self.suite["width"],
                "whestbench/depth": self.suite["depth"],
                "whestbench/reference_samples": self.suite["reference_samples"],
            },
        }

    @staticmethod
    def _failure(msg: str) -> dict[str, Any]:
        return {
            "reward": 0.0,
            "msg": msg,
            "correctness": 0.0,
            "raw_score": float("inf"),
            "result_construction": [],
            "stdout": "",
            "metrics": {},
        }


class WhestBenchEnv(Environment):
    reward_function = WhestBenchRewardEvaluator
    state_type = State

    @classmethod
    def create_initial_state(cls, problem_type: str) -> State:
        del problem_type
        suite = _suite_from_env()
        baseline = _score_callable(mean_propagation_estimate, **suite)
        return State(
            timestep=-1,
            construction=baseline["rows"],
            code=INITIAL_ESTIMATOR_CODE,
            value=-baseline["final_layer_mse"],
            metadata={
                "challenge": "ARC White-Box Estimation Challenge 2026",
                "local_suite": suite,
                "baseline_all_layer_mse": baseline["all_layer_mse"],
            },
        )

    def is_maximize(self) -> bool:
        return False

    def _should_keep_code_separators(self) -> bool:
        return False

    def _get_code_languages(self) -> list[str]:
        return ["python"]

    def check_format(self, parsed_code: str) -> bool:
        if not parsed_code or not parsed_code.strip():
            return False
        code = _extract_python_code(parsed_code)
        return any(
            marker in code
            for marker in ("def estimate", "def predict", "class Estimator")
        )

    def get_question(self) -> str:
        state_ctx = self.initial_state.to_prompt(
            _TARGET_FINAL_LAYER_MSE,
            metric_name="final-layer MSE",
            maximize=False,
            language="python",
        )
        suite = _suite_from_env()
        return f"""{WHESTBENCH_PROMPT}

Local evaluation suite for this run:
- width: {suite['width']}
- depth: {suite['depth']}
- reference samples per MLP: {suite['reference_samples']}
- seeds: {suite['seeds']}

{state_ctx}

Return one final Python code block. Keep the official `Estimator.predict` path in
mind, but the local evaluator accepts a simpler `estimate(mlp, budget)` function
for fast iteration.
"""


def discover_whestbench(problem_type: str = "arc_whestbench_2026") -> None:
    config = DiscoverConfig(
        env_type=WhestBenchEnv,
        problem_type=problem_type,
        eval_timeout=300,
        experiment_name=f"test-{problem_type}-run",
        wandb_project="whestbench",
    )
    discover(config)


def smoke_test() -> None:
    evaluator = WhestBenchRewardEvaluator(problem_type="arc_whestbench_2026")
    result = evaluator.get_reward(INITIAL_ESTIMATOR_CODE, State(-1, [], "", 0.0))
    print(result["msg"])
    print(result["stdout"])


if __name__ == "__main__":
    smoke_test()
