from __future__ import annotations

import os
import re
import shlex
import subprocess
import tempfile
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any

import flopscope.numpy as fnp
from whestbench import BaseEstimator, SetupContext, load_dataset, metadata
from whestbench.runner import (
    EstimatorEntrypoint,
    LocalRunner,
    ResourceLimits,
    SubprocessRunner,
)
from whestbench.scoring import (
    ContestData,
    ContestSpec,
    evaluate_estimator,
    make_contest,
    make_contest_from_dataset,
)

from examples.aicrowd_whestbench.prompt import WHESTBENCH_PROMPT
try:
    from ttt_discover import BaseRewardEvaluator, DiscoverConfig, Environment, State, discover
    from ttt_discover.eval_runners.blackbox_eval import build_eval_client_source
except ModuleNotFoundError:
    # The standalone official reproduction runner only needs the suite loading
    # and scoring helpers above.  Keep that path usable in the lightweight
    # whestbench/flopscope environment without installing the full discovery
    # stack (Ray, Tinker, etc.).
    BaseRewardEvaluator = object  # type: ignore[assignment,misc]
    Environment = object  # type: ignore[assignment,misc]

    class State:  # type: ignore[no-redef]
        pass

    DiscoverConfig = None  # type: ignore[assignment,misc]

    def discover(*args, **kwargs):  # type: ignore[no-redef]
        del args, kwargs
        raise RuntimeError("the optional discovery dependencies are not installed")

    def build_eval_client_source(*args, **kwargs):  # type: ignore[no-redef]
        del args, kwargs
        raise RuntimeError("the optional discovery dependencies are not installed")


_CODE_BLOCK_RE = re.compile(r"```(?:python|py)?\s*([\s\S]*?)```")
_DEFAULT_DATASET = "aicrowd/arc-whestbench-public-2026"
_DEFAULT_REVISION = "v1-phase1"
_DEFAULT_SPLIT = "mini"
_DEFAULT_N_MLPS = 50
_DEFAULT_FLOP_BUDGET = 272_000_000_000
_DEFAULT_LAMBDA_FLOPS_PER_SECOND = 1e11
_DEFAULT_ACCEPTANCE_THRESHOLD = 1e-7
_TARGET_ADJUSTED_SCORE = 1e-7
_INITIAL_ESTIMATOR_PATH_ENV = "WHEST_INITIAL_ESTIMATOR_PATH"


@dataclass(frozen=True)
class OfficialSuiteConfig:
    dataset: str = _DEFAULT_DATASET
    revision: str | None = _DEFAULT_REVISION
    split: str = _DEFAULT_SPLIT
    n_mlps: int = _DEFAULT_N_MLPS
    mlp_offset: int = 0
    flop_budget: int = _DEFAULT_FLOP_BUDGET
    setup_timeout_s: float = 5.0
    predict_timeout_s: float = 30.0
    memory_limit_mb: int = 65_536
    wall_time_limit_s: float | None = 60.0
    residual_wall_time_limit_s: float | None = None
    lambda_flops_per_second: float = _DEFAULT_LAMBDA_FLOPS_PER_SECOND
    seed: int = 0
    runner: str = "subprocess"
    streaming: bool = False

    def validate(self) -> None:
        if not self.dataset:
            raise ValueError("WHEST_DATASET must not be empty")
        if not self.split:
            raise ValueError("WHEST_DATASET_SPLIT must not be empty")
        if self.n_mlps <= 0:
            raise ValueError("WHEST_N_MLPS must be positive")
        if self.mlp_offset < 0:
            raise ValueError("MLP suite offset must be non-negative")
        if self.flop_budget <= 0:
            raise ValueError("WHEST_FLOP_BUDGET must be positive")
        if self.lambda_flops_per_second <= 0:
            raise ValueError("WHEST_LAMBDA_FLOPS_PER_SECOND must be positive")
        if self.runner not in {"local", "subprocess"}:
            raise ValueError("WHEST_RUNNER must be 'local' or 'subprocess'")


@dataclass(frozen=True)
class WhestBenchGateConfig:
    test_threshold: float = _DEFAULT_ACCEPTANCE_THRESHOLD
    holdout_threshold: float = _DEFAULT_ACCEPTANCE_THRESHOLD

    def validate(self) -> None:
        if self.test_threshold < 0:
            raise ValueError("WHEST_TEST_ACCEPTANCE_THRESHOLD must be non-negative")
        if self.holdout_threshold < 0:
            raise ValueError("WHEST_HOLDOUT_ACCEPTANCE_THRESHOLD must be non-negative")


INITIAL_ESTIMATOR_CODE = '''from __future__ import annotations

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import BaseEstimator


class Estimator(BaseEstimator):
    """Official diagonal mean/variance propagation baseline."""

    def predict(self, mlp, budget):
        del budget
        mu = fnp.zeros(mlp.width)
        var = fnp.ones(mlp.width)
        rows = []
        for w in mlp.weights:
            mu_pre = w.T @ mu
            var_pre = (w * w).T @ var
            var_pre = fnp.maximum(var_pre, 1e-12)
            sigma_pre = fnp.sqrt(var_pre)
            alpha = mu_pre / sigma_pre
            phi = flops.stats.norm.pdf(alpha)
            cdf = flops.stats.norm.cdf(alpha)
            mu = mu_pre * cdf + sigma_pre * phi
            ez2 = (mu_pre * mu_pre + var_pre) * cdf + mu_pre * sigma_pre * phi
            var = fnp.maximum(ez2 - mu * mu, 0.0)
            rows.append(mu)
        return fnp.stack(rows, axis=0)
'''


def _extract_python_code(text: str) -> str:
    matches = list(_CODE_BLOCK_RE.finditer(text or ""))
    if matches:
        return matches[-1].group(1).strip() + "\n"
    return (text or "").strip() + "\n"


def _optional_float_env(name: str, default: float | None) -> float | None:
    value = os.environ.get(name)
    if value is None:
        return default
    if value.strip().lower() in {"", "none", "null", "off"}:
        return None
    return float(value)


def _bool_env(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _suite_from_env() -> OfficialSuiteConfig:
    dataset = os.environ.get("WHEST_DATASET", _DEFAULT_DATASET).strip()
    revision_value = os.environ.get("WHEST_DATASET_REVISION", _DEFAULT_REVISION).strip()
    config = OfficialSuiteConfig(
        dataset=dataset,
        revision=revision_value or None,
        split=os.environ.get("WHEST_DATASET_SPLIT", _DEFAULT_SPLIT).strip(),
        n_mlps=int(
            os.environ.get(
                "WHEST_TEST_N_MLPS",
                os.environ.get("WHEST_N_MLPS", _DEFAULT_N_MLPS),
            )
        ),
        mlp_offset=int(
            os.environ.get(
                "WHEST_TEST_MLP_OFFSET",
                os.environ.get("WHEST_MLP_OFFSET", 0),
            )
        ),
        flop_budget=int(os.environ.get("WHEST_FLOP_BUDGET", _DEFAULT_FLOP_BUDGET)),
        setup_timeout_s=float(os.environ.get("WHEST_SETUP_TIMEOUT", 5.0)),
        predict_timeout_s=float(os.environ.get("WHEST_PREDICT_TIMEOUT", 30.0)),
        memory_limit_mb=int(os.environ.get("WHEST_MEMORY_LIMIT_MB", 65_536)),
        wall_time_limit_s=_optional_float_env("WHEST_WALL_TIME_LIMIT", 60.0),
        residual_wall_time_limit_s=_optional_float_env(
            "WHEST_RESIDUAL_WALL_TIME_LIMIT", None
        ),
        lambda_flops_per_second=float(
            os.environ.get(
                "WHEST_LAMBDA_FLOPS_PER_SECOND", _DEFAULT_LAMBDA_FLOPS_PER_SECOND
            )
        ),
        seed=int(os.environ.get("WHEST_SETUP_SEED", 0)),
        runner=os.environ.get("WHEST_RUNNER", "subprocess").strip().lower(),
        streaming=_bool_env("WHEST_DATASET_STREAMING", False),
    )
    config.validate()
    return config


def _holdout_suite_from_env(
    test_config: OfficialSuiteConfig,
) -> OfficialSuiteConfig | None:
    raw_n_mlps = os.environ.get("WHEST_HOLDOUT_N_MLPS")
    if raw_n_mlps is None or int(raw_n_mlps) <= 0:
        return None

    dataset = os.environ.get("WHEST_HOLDOUT_DATASET", test_config.dataset).strip()
    revision_value = os.environ.get(
        "WHEST_HOLDOUT_DATASET_REVISION",
        test_config.revision or "",
    ).strip()
    split = os.environ.get("WHEST_HOLDOUT_DATASET_SPLIT", test_config.split).strip()
    same_dataset = (
        dataset == test_config.dataset
        and (revision_value or None) == test_config.revision
        and split == test_config.split
    )
    default_offset = (
        test_config.mlp_offset + test_config.n_mlps if same_dataset else 0
    )
    config = replace(
        test_config,
        dataset=dataset,
        revision=revision_value or None,
        split=split,
        n_mlps=int(raw_n_mlps),
        mlp_offset=int(
            os.environ.get("WHEST_HOLDOUT_MLP_OFFSET", default_offset)
        ),
        streaming=_bool_env(
            "WHEST_HOLDOUT_DATASET_STREAMING",
            test_config.streaming,
        ),
    )
    config.validate()
    return config


def _gate_config_from_env() -> WhestBenchGateConfig:
    config = WhestBenchGateConfig(
        test_threshold=float(
            os.environ.get(
                "WHEST_TEST_ACCEPTANCE_THRESHOLD",
                _DEFAULT_ACCEPTANCE_THRESHOLD,
            )
        ),
        holdout_threshold=float(
            os.environ.get(
                "WHEST_HOLDOUT_ACCEPTANCE_THRESHOLD",
                _DEFAULT_ACCEPTANCE_THRESHOLD,
            )
        ),
    )
    config.validate()
    return config


def _resolve_dataset_source(config: OfficialSuiteConfig) -> tuple[str, str | None]:
    source = config.dataset
    revision = config.revision
    if source.startswith("hf://"):
        source = source[len("hf://") :]
    if not Path(source).exists() and "@" in source:
        source, embedded_revision = source.rsplit("@", 1)
        revision = embedded_revision or revision
    return source, revision


@lru_cache(maxsize=4)
def _load_contest_data(config: OfficialSuiteConfig) -> ContestData:
    config.validate()
    source, revision = _resolve_dataset_source(config)
    is_local = Path(source).exists()
    dataset = load_dataset(
        source,
        revision=revision,
        split=config.split,
        streaming=config.streaming and not is_local,
    )
    dataset_metadata = metadata(dataset)
    available = int(dataset_metadata.get("n_mlps") or 0)
    required = config.mlp_offset + config.n_mlps
    if available and required > available:
        raise ValueError(
            f"MLP range [{config.mlp_offset}, {required}) exceeds "
            f"dataset split size {available}"
        )
    if config.mlp_offset:
        if config.streaming:
            dataset = dataset.skip(config.mlp_offset)
        else:
            dataset = dataset.select(range(config.mlp_offset, required))

    spec = ContestSpec(
        width=int(dataset_metadata["width"]),
        depth=int(dataset_metadata["depth"]),
        n_mlps=config.n_mlps,
        flop_budget=config.flop_budget,
        ground_truth_samples=max(1, int(dataset_metadata.get("n_samples") or 1)),
        setup_timeout_s=config.setup_timeout_s,
        predict_timeout_s=config.predict_timeout_s,
        memory_limit_mb=config.memory_limit_mb,
        wall_time_limit_s=config.wall_time_limit_s,
        residual_wall_time_limit_s=config.residual_wall_time_limit_s,
        seed=config.seed,
        lambda_flops_per_second=config.lambda_flops_per_second,
    )
    seed_protocol = dataset_metadata.get("seed_protocol") or {}
    seed_protocol_version = (
        seed_protocol.get("version") if isinstance(seed_protocol, dict) else None
    )
    return make_contest_from_dataset(
        spec,
        dataset,
        config.n_mlps,
        seed_protocol_version=seed_protocol_version,
    )


class _RunnerEstimator(BaseEstimator):
    def __init__(self, runner: LocalRunner | SubprocessRunner):
        self.runner = runner

    def predict(self, mlp, budget):
        return self.runner.predict(mlp, budget)

    def last_predict_stats(self):
        return self.runner.last_predict_stats()


class _FailedEstimator(BaseEstimator):
    def __init__(self, error: Exception):
        self.error = error

    def predict(self, mlp, budget):
        del mlp, budget
        raise self.error


def _runner_for(config: OfficialSuiteConfig) -> LocalRunner | SubprocessRunner:
    return SubprocessRunner() if config.runner == "subprocess" else LocalRunner()


def _close_runner(runner: LocalRunner | SubprocessRunner) -> None:
    # whestbench 0.12.0rc5 terminates the worker but leaves its Popen pipes open.
    process = getattr(runner, "_process", None)
    try:
        runner.close()
    except subprocess.TimeoutExpired:
        # The worker may take longer than the package's hard-coded one-second
        # post-kill wait while large NumPy allocations are being released.
        # It has already received SIGKILL, so finish reaping it here instead of
        # discarding a successfully computed evaluation.
        if process is not None:
            try:
                process.wait(timeout=30.0)
            except subprocess.TimeoutExpired:
                pass
    if process is None:
        return
    for stream_name in ("stdin", "stdout", "stderr"):
        stream = getattr(process, stream_name, None)
        if stream is not None and not stream.closed:
            stream.close()


def _score_code(
    code: str,
    data: ContestData,
    config: OfficialSuiteConfig,
    submission_assets: dict[str, bytes] | None = None,
) -> dict[str, Any]:
    runner = _runner_for(config)
    with tempfile.TemporaryDirectory(prefix="ttt-whestbench-") as tmp:
        submission_dir = Path(tmp)
        estimator_path = submission_dir / "estimator.py"
        estimator_path.write_text(code, encoding="utf-8")
        for name, payload in (submission_assets or {}).items():
            asset_path = Path(name)
            if asset_path.is_absolute() or asset_path.name != name:
                raise ValueError(f"submission asset must be a basename: {name!r}")
            (submission_dir / name).write_bytes(payload)
        scratch_dir = submission_dir / "scratch"
        scratch_dir.mkdir()

        context = SetupContext(
            width=data.spec.width,
            depth=data.spec.depth,
            flop_budget=data.spec.flop_budget,
            api_version="1.0",
            scratch_dir=str(scratch_dir),
            submission_dir=str(submission_dir),
            seed=config.seed,
        )
        limits = ResourceLimits(
            setup_timeout_s=config.setup_timeout_s,
            predict_timeout_s=config.predict_timeout_s,
            memory_limit_mb=config.memory_limit_mb,
            flop_budget=config.flop_budget,
            wall_time_limit_s=config.wall_time_limit_s,
            residual_wall_time_limit_s=config.residual_wall_time_limit_s,
        )

        try:
            runner.start(
                EstimatorEntrypoint(file_path=estimator_path, class_name="Estimator"),
                context,
                limits,
            )
        except Exception as exc:
            _close_runner(runner)
            return evaluate_estimator(_FailedEstimator(exc), data)

        try:
            return evaluate_estimator(_RunnerEstimator(runner), data)
        finally:
            _close_runner(runner)


@lru_cache(maxsize=256)
def _score_code_for_config(
    code: str,
    config: OfficialSuiteConfig,
) -> dict[str, Any]:
    return _score_code(code, _load_contest_data(config), config)


@lru_cache(maxsize=4)
def _official_baseline_report(config: OfficialSuiteConfig) -> dict[str, Any]:
    return _score_code_for_config(INITIAL_ESTIMATOR_CODE, config)


@lru_cache(maxsize=8)
def _configured_initial_estimator(
    config: OfficialSuiteConfig,
    estimator_path: str,
) -> tuple[str, dict[str, Any]]:
    if not estimator_path:
        return INITIAL_ESTIMATOR_CODE, _official_baseline_report(config)

    path = Path(estimator_path).expanduser().resolve()
    code = _extract_python_code(path.read_text(encoding="utf-8"))
    if "class Estimator" not in code or "def predict" not in code:
        raise ValueError(
            f"{_INITIAL_ESTIMATOR_PATH_ENV} must define an Estimator.predict method: {path}"
        )
    return code, _score_code_for_config(code, config)


def _per_mlp_summary(report: dict[str, Any]) -> list[dict[str, Any]]:
    keys = (
        "mlp_index",
        "mlp_name",
        "adjusted_final_layer_score",
        "final_layer_mse",
        "all_layers_mse",
        "flops_used",
        "effective_compute",
        "budget_exhausted",
        "time_exhausted",
        "residual_wall_time_exhausted",
        "combined_budget_exhausted",
        "error_code",
    )
    return [{key: row.get(key) for key in keys if key in row} for row in report["per_mlp"]]


def _row_identity(row: dict[str, Any]) -> tuple[str, Any]:
    name = row.get("mlp_name")
    if name is not None:
        return ("name", name)
    return ("index", row.get("mlp_index"))


def _gate_decision(
    candidate_rows: list[dict[str, Any]],
    incumbent_rows: list[dict[str, Any]],
    *,
    threshold: float,
) -> dict[str, Any]:
    incumbent_by_id = {_row_identity(row): row for row in incumbent_rows}
    deltas: list[float] = []
    for candidate in candidate_rows:
        incumbent = incumbent_by_id.get(_row_identity(candidate))
        if incumbent is None:
            raise ValueError("candidate and incumbent MLP suites do not match")
        deltas.append(
            float(candidate["adjusted_final_layer_score"])
            - float(incumbent["adjusted_final_layer_score"])
        )
    if not deltas or len(deltas) != len(incumbent_rows):
        raise ValueError("candidate and incumbent MLP suites do not match")

    mean_delta = sum(deltas) / len(deltas)
    improvement = -mean_delta
    margin = improvement - threshold
    return {
        "evaluated": True,
        "passed": bool(mean_delta + threshold < 0.0),
        "n_mlps": len(deltas),
        "threshold": float(threshold),
        "candidate_minus_incumbent": float(mean_delta),
        "improvement": float(improvement),
        "margin_over_threshold": float(margin),
    }


def _same_dataset(
    left: OfficialSuiteConfig,
    right: OfficialSuiteConfig,
) -> bool:
    return (
        left.dataset == right.dataset
        and left.revision == right.revision
        and left.split == right.split
    )


def _validate_disjoint_suites(
    test_config: OfficialSuiteConfig,
    holdout_config: OfficialSuiteConfig | None,
) -> None:
    if holdout_config is None or not _same_dataset(test_config, holdout_config):
        return
    test_range = range(
        test_config.mlp_offset,
        test_config.mlp_offset + test_config.n_mlps,
    )
    holdout_range = range(
        holdout_config.mlp_offset,
        holdout_config.mlp_offset + holdout_config.n_mlps,
    )
    if test_range.start < holdout_range.stop and holdout_range.start < test_range.stop:
        raise ValueError("WhestBench test and holdout MLP ranges must not overlap")


def _format_report(report: dict[str, Any]) -> str:
    lines = ["mlp | adjusted_score | final_mse | effective_compute | status"]
    for row in report["per_mlp"]:
        failed = bool(row.get("error_code")) or any(
            row.get(flag)
            for flag in (
                "budget_exhausted",
                "time_exhausted",
                "residual_wall_time_exhausted",
                "combined_budget_exhausted",
            )
        )
        lines.append(
            f"{row.get('mlp_name') or row['mlp_index']} | "
            f"{row['adjusted_final_layer_score']:.8g} | "
            f"{row['final_layer_mse']:.8g} | "
            f"{row.get('effective_compute', 0.0):.8g} | "
            f"{'failed' if failed else 'ok'}"
        )
    return "\n".join(lines)


class WhestBenchRewardResult(dict[str, Any]):
    """Structured WhestBench result compatible with the generic reward contract."""

    SCHEMA_VERSION = 2

    @classmethod
    def from_report(
        cls,
        *,
        candidate: dict[str, Any],
        baseline: dict[str, Any],
        config: OfficialSuiteConfig,
        test_gate: dict[str, Any],
        holdout_gate: dict[str, Any],
    ) -> "WhestBenchRewardResult":
        score = float(candidate["adjusted_final_layer_score"])
        baseline_score = float(baseline["adjusted_final_layer_score"])
        reward = baseline_score / max(score, 1e-30)
        per_mlp = _per_mlp_summary(candidate)
        n_mlps = len(per_mlp)
        n_failed = int(candidate["n_failed_mlps"])
        correctness = (n_mlps - n_failed) / n_mlps if n_mlps else 0.0
        promotion_passed = bool(
            test_gate["passed"]
            and (
                holdout_gate["passed"]
                if holdout_gate["enabled"]
                else True
            )
        )
        details = {
            "whestbench": {
                "schema_version": cls.SCHEMA_VERSION,
                "suite_role": "test",
                "suite": {
                    "dataset": config.dataset,
                    "revision": config.revision,
                    "split": config.split,
                    "n_mlps": n_mlps,
                    "mlp_offset": config.mlp_offset,
                },
                "score_key": "adjusted_final_layer_score",
                "score_direction": "minimize",
                "adjusted_final_layer_score": score,
                "baseline_adjusted_final_layer_score": baseline_score,
                "test_gate": test_gate,
                "holdout_gate": holdout_gate,
                "promotion_passed": promotion_passed,
            }
        }
        return cls(
            reward=float(reward),
            msg=(
                f"adjusted_final_layer_score={score:.8g}; "
                f"final_layer_mse={candidate['final_layer_mse']:.8g}; "
                f"baseline_adjusted_score={baseline_score:.8g}; "
                f"failed_mlps={n_failed}/{n_mlps}; "
                f"test_gate_passed={test_gate['passed']}; "
                f"holdout_evaluated={holdout_gate['evaluated']}; "
                f"holdout_gate_passed={holdout_gate['passed']}; "
                f"promotion_passed={promotion_passed}"
            ),
            correctness=float(correctness),
            raw_score=score,
            result_construction=per_mlp,
            stdout=_format_report(candidate),
            threshold_passed=promotion_passed,
            metrics={
                "whestbench/adjusted_final_layer_score": score,
                "whestbench/final_layer_mse": float(candidate["final_layer_mse"]),
                "whestbench/all_layers_mse": float(candidate["all_layers_mse"]),
                "whestbench/baseline_adjusted_score": baseline_score,
                "whestbench/reward_vs_baseline": float(reward),
                "whestbench/mean_score_multiplier": float(
                    candidate["mean_score_multiplier"]
                ),
                "whestbench/mean_compute_utilization": float(
                    candidate["mean_compute_utilization"]
                ),
                "whestbench/mean_effective_compute": float(
                    candidate["mean_effective_compute"]
                ),
                "whestbench/n_failed_mlps": n_failed,
                "whestbench/n_mlps": n_mlps,
                "whestbench/flop_budget": config.flop_budget,
                "whestbench/lambda_flops_per_second": config.lambda_flops_per_second,
                "whestbench/test_gate_passed": bool(test_gate["passed"]),
                "whestbench/holdout_evaluated": bool(holdout_gate["evaluated"]),
                "whestbench/holdout_gate_passed": holdout_gate["passed"],
                "whestbench/promotion_passed": promotion_passed,
            },
            details=details,
        )

    @classmethod
    def infrastructure_failure(cls, msg: str) -> "WhestBenchRewardResult":
        return cls(
            reward=0.0,
            msg=msg,
            correctness=0.0,
            raw_score=float("inf"),
            result_construction=[],
            stdout="",
            metrics={},
            details={
                "whestbench": {
                    "schema_version": cls.SCHEMA_VERSION,
                    "suite_role": "test",
                    "status": "infrastructure_failure",
                    "promotion_passed": False,
                }
            },
            threshold_passed=False,
        )


class WhestBenchRewardEvaluator(BaseRewardEvaluator):
    def __init__(self, *args, **kwargs):
        del args
        self.problem_type = kwargs.pop("problem_type", "arc_whestbench_2026")
        provided_suite_config = kwargs.pop("suite_config", None)
        self.config = provided_suite_config or _suite_from_env()
        self.gate_config = kwargs.pop("gate_config", None) or _gate_config_from_env()
        self.gate_config.validate()
        provided_holdout_config = kwargs.pop("holdout_config", None)
        self.holdout_config = (
            provided_holdout_config
            if provided_suite_config is not None
            else (
                provided_holdout_config
                or _holdout_suite_from_env(self.config)
            )
        )
        _validate_disjoint_suites(self.config, self.holdout_config)

        provided_contest_data = kwargs.pop("contest_data", None)
        self._has_provided_contest_data = provided_contest_data is not None
        self.contest_data = provided_contest_data or _load_contest_data(self.config)
        provided_holdout_data = kwargs.pop("holdout_contest_data", None)
        self._has_provided_holdout_data = provided_holdout_data is not None
        self.holdout_contest_data = provided_holdout_data

        self.baseline = kwargs.pop("baseline_report", None)
        if self.baseline is None:
            if provided_contest_data is None:
                self.baseline = _official_baseline_report(self.config)
            else:
                self.baseline = _score_code(
                    INITIAL_ESTIMATOR_CODE, self.contest_data, self.config
                )

    @staticmethod
    def _incumbent_code(state: State | None) -> str:
        code = _extract_python_code(getattr(state, "code", "") or "")
        return code if code.strip() else INITIAL_ESTIMATOR_CODE

    def _score_test(self, code: str) -> dict[str, Any]:
        if self._has_provided_contest_data:
            return _score_code(code, self.contest_data, self.config)
        return _score_code_for_config(code, self.config)

    def _score_holdout(self, code: str) -> dict[str, Any]:
        if self.holdout_config is None:
            raise RuntimeError("WhestBench holdout suite is not configured")
        if self._has_provided_holdout_data:
            assert self.holdout_contest_data is not None
            return _score_code(code, self.holdout_contest_data, self.holdout_config)
        return _score_code_for_config(code, self.holdout_config)

    def _test_incumbent_rows(
        self,
        state: State | None,
        candidate_rows: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        construction = getattr(state, "construction", None)
        if isinstance(construction, list) and construction:
            try:
                _gate_decision(
                    candidate_rows,
                    construction,
                    threshold=self.gate_config.test_threshold,
                )
                return construction
            except (KeyError, TypeError, ValueError):
                pass
        incumbent = self._score_test(self._incumbent_code(state))
        return _per_mlp_summary(incumbent)

    def get_reward(self, code: str, state: State) -> dict[str, Any]:
        candidate_code = _extract_python_code(code)
        if not candidate_code.strip():
            return WhestBenchRewardResult.infrastructure_failure("Empty candidate.")

        try:
            candidate = self._score_test(candidate_code)
            candidate_rows = _per_mlp_summary(candidate)
            incumbent_rows = self._test_incumbent_rows(state, candidate_rows)
            test_gate = _gate_decision(
                candidate_rows,
                incumbent_rows,
                threshold=self.gate_config.test_threshold,
            )
            test_failures = int(candidate["n_failed_mlps"])
            if test_failures:
                test_gate["passed"] = False
                test_gate["failure_reason"] = "candidate_failed_mlps"
                test_gate["n_failed_mlps"] = test_failures

            holdout_gate: dict[str, Any] = {
                "enabled": self.holdout_config is not None,
                "evaluated": False,
                "passed": None,
            }
            if self.holdout_config is not None:
                holdout_gate["threshold"] = self.gate_config.holdout_threshold
                if not test_gate["passed"]:
                    holdout_gate["passed"] = False
                    holdout_gate["skip_reason"] = "test_gate_failed"
                else:
                    holdout_candidate = self._score_holdout(candidate_code)
                    holdout_incumbent = self._score_holdout(
                        self._incumbent_code(state)
                    )
                    private_decision = _gate_decision(
                        _per_mlp_summary(holdout_candidate),
                        _per_mlp_summary(holdout_incumbent),
                        threshold=self.gate_config.holdout_threshold,
                    )
                    holdout_failures = int(holdout_candidate["n_failed_mlps"])
                    private_passed = bool(
                        private_decision["passed"] and not holdout_failures
                    )
                    holdout_gate.update(
                        {
                            "evaluated": True,
                            "passed": private_passed,
                            "n_mlps": private_decision["n_mlps"],
                        }
                    )
                    if holdout_failures:
                        holdout_gate["failure_reason"] = "candidate_failed_mlps"
                        holdout_gate["n_failed_mlps"] = holdout_failures
        except Exception as exc:
            return WhestBenchRewardResult.infrastructure_failure(
                f"Official evaluator failed: {exc}"
            )

        return WhestBenchRewardResult.from_report(
            candidate=candidate,
            baseline=self.baseline,
            config=self.config,
            test_gate=test_gate,
            holdout_gate=holdout_gate,
        )


class WhestBenchEnv(Environment):
    reward_function = WhestBenchRewardEvaluator
    state_type = State

    @classmethod
    def create_initial_state(cls, problem_type: str) -> State:
        del problem_type
        config = _suite_from_env()
        data = _load_contest_data(config)
        baseline = _official_baseline_report(config)
        estimator_path = os.environ.get(_INITIAL_ESTIMATOR_PATH_ENV, "").strip()
        initial_code, initial = _configured_initial_estimator(config, estimator_path)
        spec = data.spec
        return State(
            timestep=-1,
            construction=_per_mlp_summary(initial),
            code=initial_code,
            value=-float(initial["adjusted_final_layer_score"]),
            metadata={
                "challenge": "ARC White-Box Estimation Challenge 2026",
                "official_suite": {
                    "dataset": config.dataset,
                    "revision": config.revision,
                    "split": config.split,
                    "n_mlps": spec.n_mlps,
                    "mlp_offset": config.mlp_offset,
                    "width": spec.width,
                    "depth": spec.depth,
                    "flop_budget": spec.flop_budget,
                    "lambda_flops_per_second": spec.lambda_flops_per_second,
                    "runner": config.runner,
                },
                "baseline_final_layer_mse": baseline["final_layer_mse"],
                "baseline_all_layers_mse": baseline["all_layers_mse"],
                "initial_estimator_path": estimator_path or None,
                "initial_adjusted_final_layer_score": initial[
                    "adjusted_final_layer_score"
                ],
                "initial_final_layer_mse": initial["final_layer_mse"],
                "initial_all_layers_mse": initial["all_layers_mse"],
            },
        )

    def is_maximize(self) -> bool:
        return False

    def _create_next_state(
        self,
        step_idx: int,
        parsed_code: str,
        outs,
    ) -> State | None:
        whestbench = (outs.details or {}).get("whestbench", {})
        if not bool(whestbench.get("promotion_passed", False)):
            return None
        return super()._create_next_state(step_idx, parsed_code, outs)

    def _should_keep_code_separators(self) -> bool:
        return False

    def _get_code_languages(self) -> list[str]:
        return ["python"]

    def check_format(self, parsed_code: str) -> bool:
        if not parsed_code or not parsed_code.strip():
            return False
        code = _extract_python_code(parsed_code)
        return "class Estimator" in code and "def predict" in code

    def _initial_estimator_code(self) -> str:
        code = _extract_python_code(getattr(self.initial_state, "code", "") or "")
        if "class Estimator" in code and "def predict" in code:
            return code
        return INITIAL_ESTIMATOR_CODE

    def build_autonomous_prompt(
        self,
        *,
        prompt: str,
        workspace: Path,
        eval_timeout: int,
        num_cpus_per_task: int,
    ) -> str:
        del eval_timeout, num_cpus_per_task
        (workspace / "submission.py").write_text(
            self._initial_estimator_code(),
            encoding="utf-8",
        )
        return f"""{prompt}

--- Autonomous WhestBench Isolated Search Mode ---
Work only inside this workspace:
{workspace}

Editable candidate:
{workspace / "submission.py"}

This isolated workspace intentionally contains no evaluation data or evaluator.
Derive an improved estimator analytically from the public task description. Do
not inspect files outside the workspace or search for datasets, caches, reports,
or target values. The outer AutoEvolve runner will score the final candidate with
the trusted official evaluator.

Leave the best plain Python implementation in:
{workspace / "submission.py"}
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
                "Blackbox autonomous WhestBench requires a socket path or TCP port."
            )

        (workspace / "submission.py").write_text(
            self._initial_estimator_code(),
            encoding="utf-8",
        )
        (workspace / "eval_client.py").write_text(
            build_eval_client_source(
                problem_type=self.problem_type,
                socket_path=socket_path,
                host=host,
                port=port,
                timeout_s=max(1.0, float(eval_timeout)),
                state=(
                    self.state.to_dict()
                    if hasattr(self.state, "to_dict")
                    else None
                ),
                print_details=True,
            ),
            encoding="utf-8",
        )

        evaluator_cmd = f"cd {shlex.quote(str(workspace))} && python eval_client.py"
        return f"""{prompt}

--- Autonomous WhestBench Blackbox Search Mode ---
Work only inside this workspace:
{workspace}

Editable candidate:
{workspace / "submission.py"}

The trusted official evaluator is exposed only through a local blackbox service.
Run it after each meaningful revision:
{evaluator_cmd}

Do not edit `eval_client.py`. Only `submission.py` is a candidate artifact. It
must be plain Python source defining the official `Estimator(BaseEstimator)`
class, without Markdown fences. Lower `raw_score` is better; `reward` is the
official baseline score divided by the candidate score.
The evaluator prints a `details` JSON object after each valid run. A candidate
must pass the test gate before the hidden holdout is evaluated, and only
`promotion_passed=true` is eligible to replace the incumbent.
Do not inspect evaluation datasets, caches, reports, or target values; derive
the estimator only from the supplied task description and blackbox scores.

When done, leave the best implementation in:
{workspace / "submission.py"}
"""

    def get_question(self) -> str:
        state_ctx = self.initial_state.to_prompt(
            _TARGET_ADJUSTED_SCORE,
            metric_name="budget-adjusted final-layer score",
            maximize=False,
            language="python",
        )
        suite = self.initial_state.metadata.get("official_suite", {})
        return f"""{WHESTBENCH_PROMPT}

Official evaluation suite for this run:
- dataset: {suite.get('dataset')}@{suite.get('revision')}
- split / MLPs: {suite.get('split')} / first {suite.get('n_mlps')}
- width / depth: {suite.get('width')} / {suite.get('depth')}
- FLOP budget per MLP: {suite.get('flop_budget')}
- residual penalty lambda: {suite.get('lambda_flops_per_second')} FLOPs/second
- runner: {suite.get('runner')}

{state_ctx}

Return one final Python code block containing an official `Estimator` class.
"""


def discover_whestbench(problem_type: str = "arc_whestbench_2026") -> None:
    config = DiscoverConfig(
        env_type=WhestBenchEnv,
        problem_type=problem_type,
        eval_timeout=4000,
        experiment_name=f"test-{problem_type}-run",
        wandb_project="whestbench",
    )
    discover(config)


def smoke_test() -> None:
    smoke_config = replace(
        _suite_from_env(),
        dataset="generated-smoke",
        revision=None,
        split="smoke",
        n_mlps=2,
        flop_budget=10_000_000_000,
        runner="local",
        streaming=False,
    )
    spec = ContestSpec(
        width=16,
        depth=4,
        n_mlps=2,
        flop_budget=smoke_config.flop_budget,
        ground_truth_samples=512,
        seed=0,
        wall_time_limit_s=smoke_config.wall_time_limit_s,
        lambda_flops_per_second=smoke_config.lambda_flops_per_second,
    )
    evaluator = WhestBenchRewardEvaluator(
        problem_type="arc_whestbench_2026",
        suite_config=smoke_config,
        contest_data=make_contest(spec),
    )
    result = evaluator.get_reward(INITIAL_ESTIMATOR_CODE, State(-1, [], "", 0.0))
    print(result["msg"])
    print(result["stdout"])


if __name__ == "__main__":
    smoke_test()
