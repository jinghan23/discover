from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples.aicrowd_whestbench import env as whest_env


ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, relative_path: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_seed_offset_rewrites_plain_and_constant_rng_streams() -> None:
    source = "\n".join(
        (
            "a = fnp.random.default_rng(mlp.seed)",
            "b = fnp.random.default_rng(mlp.seed + 999)",
            "c = fnp.random.default_rng(mlp.seed + 20260721)",
            "dataset_seed = mlp.seed",
        )
    )

    rewritten, count = whest_env._offset_estimator_rng_seeds(source, 1_000_003)

    assert count == 3
    assert "default_rng(mlp.seed + 1000003)" in rewritten
    assert "default_rng(mlp.seed + 999 + 1000003)" in rewritten
    assert "default_rng(mlp.seed + 20260721 + 1000003)" in rewritten
    assert "dataset_seed = mlp.seed" in rewritten


def test_seed_offset_zero_keeps_source_byte_identical() -> None:
    source = "rng = fnp.random.default_rng(mlp.seed)\n"
    rewritten, count = whest_env._offset_estimator_rng_seeds(source, 0)
    assert rewritten == source
    assert count == 0


def test_seed_offset_allows_setup_only_estimator_without_supported_rng() -> None:
    source = "return mlp.seed\n"
    rewritten, count = whest_env._offset_estimator_rng_seeds(source, 1)
    assert rewritten == source
    assert count == 0


def test_blackbox_evaluator_scores_one_requested_seed_offset(monkeypatch) -> None:
    evaluator = whest_env.WhestBenchRewardEvaluator.__new__(
        whest_env.WhestBenchRewardEvaluator
    )
    evaluator.config = whest_env.OfficialSuiteConfig(seed=17)
    evaluator.contest_data = object()
    evaluator._has_provided_contest_data = True
    evaluator.baseline = {"adjusted_final_layer_score": 4.0}
    evaluator.gate_config = whest_env._gate_config_from_env()
    evaluator.holdout_config = None
    evaluator.holdout_contest_data = None
    evaluator._has_provided_holdout_data = False
    seen: list[tuple[int, str]] = []

    def fake_score(code, contest_data, config):
        del contest_data
        seen.append((config.seed, code))
        value = 2.0
        row = {
            "mlp_index": 0,
            "mlp_name": "mlp-0",
            "adjusted_final_layer_score": value,
            "final_layer_mse": value,
            "all_layers_mse": value,
            "effective_compute": 1.0,
        }
        return {
            "adjusted_final_layer_score": value,
            "final_layer_mse": value,
            "all_layers_mse": value,
            "mean_compute_utilization": 0.1,
            "mean_effective_compute": 1.0,
            "mean_score_multiplier": 0.1,
            "n_failed_mlps": 0,
            "per_mlp": [row],
        }

    monkeypatch.setattr(whest_env, "_score_code", fake_score)
    # Non-empty construction lets the test gate reuse the incumbent rows, so the
    # candidate is the only suite that has to be scored.
    incumbent_rows = [
        {
            "mlp_index": 0,
            "mlp_name": "mlp-0",
            "adjusted_final_layer_score": 3.0,
            "final_layer_mse": 3.0,
            "all_layers_mse": 3.0,
            "effective_compute": 1.0,
        }
    ]
    result = evaluator.get_reward(
        "rng = fnp.random.default_rng(mlp.seed)\nclass Estimator: pass",
        state=whest_env.State(
            timestep=-1,
            construction=incumbent_rows,
            code="",
            value=0.0,
            metadata={"whestbench_estimator_seed_offset": 1_000_003},
        ),
    )

    assert len(seen) == 1
    assert seen[0][0] == 1_000_020
    assert "mlp.seed + 1000003" in seen[0][1]
    assert result["raw_score"] == pytest.approx(2.0)
    assert result["correctness"] == 1.0
    assert result["result_construction"][0]["estimator_seed_offset"] == 1_000_003


def test_blackbox_client_accepts_one_matching_seed_response() -> None:
    evaluator = _load_module(
        "whest_three_seed_blackbox_evaluator",
        "repro/aicrowd_whestbench/evaluate_public_reproduction.py",
    )
    rows = [
        {
            "mlp_index": 0,
            "adjusted_final_layer_score": 1.0,
            "estimator_seed_offset": 1_000_003,
        }
    ]

    assert evaluator._validate_rows(
        rows,
        expected_n_mlps=1,
        expected_seed_offset=1_000_003,
    ) is rows


def test_blackbox_client_rejects_wrong_seed_response() -> None:
    evaluator = _load_module(
        "whest_three_seed_blackbox_protocol",
        "repro/aicrowd_whestbench/evaluate_public_reproduction.py",
    )
    rows = [
        {
            "mlp_index": 0,
            "adjusted_final_layer_score": 1.0,
            "estimator_seed_offset": 0,
        }
    ]

    with pytest.raises(RuntimeError, match="did not honor estimator_seed_offset"):
        evaluator._validate_rows(
            rows,
            expected_n_mlps=1,
            expected_seed_offset=1_000_003,
        )


def test_clustered_bootstrap_resamples_mlp_means() -> None:
    evaluator = _load_module(
        "whest_three_seed_bootstrap",
        "repro/aicrowd_whestbench/evaluate_public_reproduction.py",
    )
    paired_rows = [
        {"mean_paired_adjusted_difference": -1.0},
        {"mean_paired_adjusted_difference": -1.0},
        {"mean_paired_adjusted_difference": -1.0},
    ]

    observed, low, high = evaluator._clustered_bootstrap_ci(
        paired_rows,
        samples=100,
        seed=7,
    )

    assert observed == pytest.approx(-1.0)
    assert low == pytest.approx(-1.0)
    assert high == pytest.approx(-1.0)


def test_blackbox_schedule_interleaves_with_symmetric_cooldown(monkeypatch) -> None:
    evaluator = _load_module(
        "whest_three_seed_interleaved_schedule",
        "repro/aicrowd_whestbench/evaluate_public_reproduction.py",
    )
    calls = []
    sleeps = []

    def fake_evaluate_seed(
        runner,
        source,
        *,
        seed_offset,
        expected_n_mlps,
    ):
        del runner
        calls.append((source, seed_offset, expected_n_mlps))
        return {"source": source, "estimator_seed_offset": seed_offset}

    monkeypatch.setattr(evaluator, "_evaluate_seed", fake_evaluate_seed)
    monkeypatch.setattr(evaluator.time, "sleep", sleeps.append)

    candidate, baseline, order, attempts = evaluator._evaluate_schedule(
        object(),
        "candidate-source",
        "baseline-source",
        seed_offsets=(0, 1_000_003, 2_000_003),
        expected_n_mlps=100,
        interleave_baseline=True,
        inter_run_cooldown_s=15.0,
        max_infra_retries=0,
        infra_retry_cooldown_s=30.0,
    )

    assert calls == [
        ("candidate-source", 0, 100),
        ("baseline-source", 0, 100),
        ("candidate-source", 1_000_003, 100),
        ("baseline-source", 1_000_003, 100),
        ("candidate-source", 2_000_003, 100),
        ("baseline-source", 2_000_003, 100),
    ]
    assert sleeps == [15.0] * 6
    assert [run["estimator_seed_offset"] for run in candidate] == [
        0,
        1_000_003,
        2_000_003,
    ]
    assert [run["estimator_seed_offset"] for run in baseline] == [
        0,
        1_000_003,
        2_000_003,
    ]
    assert order == [
        {"estimator": "candidate", "estimator_seed_offset": 0},
        {"estimator": "baseline", "estimator_seed_offset": 0},
        {"estimator": "candidate", "estimator_seed_offset": 1_000_003},
        {"estimator": "baseline", "estimator_seed_offset": 1_000_003},
        {"estimator": "candidate", "estimator_seed_offset": 2_000_003},
        {"estimator": "baseline", "estimator_seed_offset": 2_000_003},
    ]
    assert len(attempts) == 6
    assert all(attempt["attempt_number"] == 1 for attempt in attempts)
    assert all(attempt["used_for_aggregate"] is True for attempt in attempts)


def test_whole_suite_setup_timeout_retry_is_narrow() -> None:
    evaluator = _load_module(
        "whest_three_seed_infra_retry_predicate",
        "repro/aicrowd_whestbench/evaluate_public_reproduction.py",
    )

    def row(**updates):
        value = {
            "error_code": "SETUP_TIMEOUT",
            "flops_used": 0,
            "effective_compute": 0.0,
            "budget_exhausted": False,
            "time_exhausted": False,
            "residual_wall_time_exhausted": False,
            "combined_budget_exhausted": False,
        }
        value.update(updates)
        return value

    assert evaluator._whole_suite_setup_timeout_before_compute(
        {"per_mlp": [row(), row()]},
        expected_n_mlps=2,
    )
    assert not evaluator._whole_suite_setup_timeout_before_compute(
        {"per_mlp": [row()]},
        expected_n_mlps=2,
    )
    assert not evaluator._whole_suite_setup_timeout_before_compute(
        {"per_mlp": [row(), row(flops_used=1)]},
        expected_n_mlps=2,
    )
    assert not evaluator._whole_suite_setup_timeout_before_compute(
        {"per_mlp": [row(), row(error_code="PREDICT_ERROR")]},
        expected_n_mlps=2,
    )
    assert not evaluator._whole_suite_setup_timeout_before_compute(
        {"per_mlp": [row(), row(time_exhausted=True)]},
        expected_n_mlps=2,
    )


def test_schedule_retries_only_whole_suite_zero_compute_setup_timeout(
    monkeypatch,
) -> None:
    evaluator = _load_module(
        "whest_three_seed_infra_retry_schedule",
        "repro/aicrowd_whestbench/evaluate_public_reproduction.py",
    )
    calls = []
    sleeps = []

    def make_run(source, seed_offset, *, timeout):
        row = {
            "adjusted_final_layer_score": 1.0,
            "all_layers_mse": 1.0,
            "effective_compute": 0.0 if timeout else 1.0,
            "error_code": "SETUP_TIMEOUT" if timeout else None,
            "estimator_seed_offset": seed_offset,
            "final_layer_mse": 1.0,
            "flops_used": 0 if timeout else 1,
        }
        return {
            "elapsed_s": 5.0 if timeout else 10.0,
            "per_mlp": [row],
            "source": source,
            "summary": {"adjusted_final_layer_score": 1.0, "n_failed_mlps": int(timeout)},
        }

    def fake_evaluate_seed(
        runner,
        source,
        *,
        seed_offset,
        expected_n_mlps,
    ):
        del runner, expected_n_mlps
        calls.append((source, seed_offset))
        first_candidate_attempt = source == "candidate-source" and sum(
            call[0] == "candidate-source" for call in calls
        ) == 1
        return make_run(source, seed_offset, timeout=first_candidate_attempt)

    monkeypatch.setattr(evaluator, "_evaluate_seed", fake_evaluate_seed)
    monkeypatch.setattr(evaluator.time, "sleep", sleeps.append)

    candidate, baseline, order, attempts = evaluator._evaluate_schedule(
        object(),
        "candidate-source",
        "baseline-source",
        seed_offsets=(0,),
        expected_n_mlps=1,
        interleave_baseline=True,
        inter_run_cooldown_s=15.0,
        max_infra_retries=2,
        infra_retry_cooldown_s=30.0,
    )

    assert calls == [
        ("candidate-source", 0),
        ("candidate-source", 0),
        ("baseline-source", 0),
    ]
    assert sleeps == [15.0, 30.0, 15.0]
    assert len(candidate) == len(baseline) == 1
    assert order == [
        {"estimator": "candidate", "estimator_seed_offset": 0},
        {"estimator": "baseline", "estimator_seed_offset": 0},
    ]
    assert [attempt["used_for_aggregate"] for attempt in attempts] == [
        False,
        True,
        True,
    ]
    assert attempts[0]["whole_suite_setup_timeout_before_compute"] is True
    assert attempts[1]["whole_suite_setup_timeout_before_compute"] is False


def test_blackbox_client_runs_paired_three_seed_gate(tmp_path, monkeypatch) -> None:
    evaluator = _load_module(
        "whest_three_seed_blackbox_main",
        "repro/aicrowd_whestbench/evaluate_public_reproduction.py",
    )
    estimator_path = tmp_path / "estimator.py"
    baseline_path = tmp_path / "german_current.py"
    output_path = tmp_path / "report.json"
    estimator_path.write_text("class Estimator: pass\n", encoding="utf-8")
    baseline_path.write_text("class Estimator: german_current\n", encoding="utf-8")
    calls = []

    class FakeBlackboxRunner:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def evaluate(self, env, source):
            calls.append((self.kwargs, env, source))
            seed_position = (len(calls) - 1) % 3
            is_baseline = "german_current" in source
            score = float(seed_position + 1 + int(is_baseline))
            seed_offset = env.state.metadata["whestbench_estimator_seed_offset"]
            row = {
                "mlp_index": 0,
                "mlp_name": "mlp-0",
                "adjusted_final_layer_score": score,
                "final_layer_mse": score * 2.0,
                "all_layers_mse": 3.0,
                "effective_compute": 4.0,
                "flop_budget": 100,
                "lambda_flops_per_second": 10.0,
                "estimator_seed_offset": seed_offset,
                "estimator_seed_replacement_count": 0,
            }
            return SimpleNamespace(result_construction=[row], raw_score=score)

    monkeypatch.setattr(evaluator, "BlackboxRunner", FakeBlackboxRunner)
    monkeypatch.setattr(
        evaluator.sys,
        "argv",
        [
            "evaluate_public_reproduction.py",
            str(estimator_path),
            str(output_path),
            "--baseline",
            str(baseline_path),
            "--n-mlps",
            "1",
            "--bootstrap-samples",
            "100",
            "--socket",
            "/tmp/test-whestbench.sock",
        ],
    )

    assert evaluator.main() == 0
    assert len(calls) == 6
    assert [
        request_env.state.metadata["whestbench_estimator_seed_offset"]
        for _, request_env, _ in calls
    ] == [0, 1_000_003, 2_000_003, 0, 1_000_003, 2_000_003]
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["candidate_mean_adjusted"] == pytest.approx(2.0)
    assert report["baseline_mean_adjusted"] == pytest.approx(3.0)
    assert report["bootstrap"]["percentile_95_ci"] == pytest.approx([-1.0, -1.0])
    assert report["gate"]["all_three_seed_aggregates_beat_baseline"] is True
    assert report["gate"]["official_submission_eligible"] is True
    assert report["evaluation"]["n_suite_evaluations"] == 6
    assert len(report["results"]["per_mlp"]) == 1
    assert len(report["results"]["seed_runs"]) == 3
    assert len(report["baseline_results"]["seed_runs"]) == 3
