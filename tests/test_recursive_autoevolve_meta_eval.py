from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from examples.aicrowd_whestbench import env as whest_env
from experiments.recursive_autoevolve import meta_eval
from ttt_discover import State


def _evaluation(
    erdos: float,
    whest_private: float,
    kernel: float,
    *,
    valid: bool = True,
) -> dict:
    return {
        "valid": valid,
        "tasks": {
            "erdos": {"best_score": erdos},
            "arc_whestbench": {"private_best_score": whest_private},
            "kernel": {"best_score": kernel},
        },
    }


def test_protocol_fixes_budget_model_tasks_and_gpu_pool() -> None:
    protocol = meta_eval._protocol_payload()

    assert protocol["inner"]["num_epochs"] == 1
    assert protocol["inner"]["max_evaluator_calls"] == 25
    assert protocol["inner"]["model"] == "gpt-5.5"
    assert protocol["kernel"] == {
        "task": "trimul",
        "gpu_devices": (7,),
    }
    assert set(protocol["whestbench_manifest"]["public_seeds"]).isdisjoint(
        protocol["whestbench_manifest"]["private_seeds"]
    )
    assert len(protocol["whestbench_manifest"]["public_seeds"]) == 50
    assert len(protocol["whestbench_manifest"]["private_seeds"]) == 50


def test_inner_config_cannot_raise_candidate_budget(tmp_path: Path) -> None:
    for spec in meta_eval.TASK_SPECS:
        payload = meta_eval._inner_config(spec, tmp_path / spec.name, None)
        config = payload["runs"]["inner"]

        assert config["algorithm"] == "autoevolve"
        assert config["eval_runner"] == "blackbox"
        assert config["num_epochs"] == 1
        assert config["max_evaluator_calls"] == 25
        assert config["group_size"] == 1
        assert config["groups_per_batch"] == 1
        assert config["max_concurrent_requests"] == 1


def test_editable_scope_is_exact() -> None:
    assert meta_eval._path_is_editable(
        "ttt_discover/algorithms/autoevolve/loop.py"
    )
    assert meta_eval._path_is_editable("ttt_discover/codex_utils/autonomous.py")
    assert not meta_eval._path_is_editable("ttt_discover/runner.py")
    assert not meta_eval._path_is_editable(
        "experiments/recursive_autoevolve/meta_eval.py"
    )
    assert not meta_eval._path_is_editable("examples/aicrowd_whestbench/env.py")


def test_selection_accepts_broad_improvement() -> None:
    incumbent = _evaluation(100.0, 100.0, 100.0)
    candidate = _evaluation(98.0, 98.0, 99.0)

    result = meta_eval.select_candidate(incumbent, candidate)

    assert result["accepted"] is True
    assert result["selection_score"] >= meta_eval.MIN_GEOMEAN_IMPROVEMENT
    assert result["improved_tasks"] == 3


def test_selection_rejects_single_task_regression() -> None:
    incumbent = _evaluation(100.0, 100.0, 100.0)
    candidate = _evaluation(90.0, 90.0, 103.0)

    result = meta_eval.select_candidate(incumbent, candidate)

    assert result["accepted"] is False
    assert any("one-task ratio" in reason for reason in result["reasons"])


def test_selection_rejects_invalid_candidate() -> None:
    result = meta_eval.select_candidate(
        _evaluation(1.0, 1.0, 1.0),
        _evaluation(0.9, 0.9, 0.9, valid=False),
    )

    assert result["accepted"] is False
    assert result["selection_score"] is None


def test_run_summary_uses_best_valid_kernel_state_and_server_budget(
    tmp_path: Path,
) -> None:
    pool = {
        "states": [
            {"timestep": -1, "value": -1_000_000.0, "code": ""},
            {
                "timestep": 0,
                "value": -12.0,
                "code": "def custom_kernel(data):\n    return data\n",
            },
            {
                "timestep": 0,
                "value": -10.0,
                "code": "def custom_kernel(data):\n    return data[0]\n",
            },
        ]
    }
    (tmp_path / "autoevolve_pool_step_000001.json").write_text(
        json.dumps(pool), encoding="utf-8"
    )
    (tmp_path / "agent_outputs.jsonl").write_text(
        json.dumps(
            {"metrics": {"budget/evaluator_calls_server": 25}}
        )
        + "\n",
        encoding="utf-8",
    )

    result = meta_eval.summarize_inner_run(
        "kernel",
        tmp_path,
        {
            "returncode": 0,
            "timed_out": False,
            "wall_time_seconds": 1.0,
            "gpu_device": 4,
        },
    )

    assert result["valid"] is True
    assert result["best_score"] == 10.0
    assert result["evaluator_calls"] == 25
    assert Path(result["best_submission"]).read_text(encoding="utf-8").endswith(
        "return data[0]\n"
    )


def test_whestbench_reference_cache_does_not_recompute(monkeypatch) -> None:
    calls = 0
    original = whest_env.monte_carlo_layer_means

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(whest_env, "monte_carlo_layer_means", counted)
    whest_env._cached_reference.cache_clear()

    first = whest_env._cached_reference(2, 1, 4, 12345)
    second = whest_env._cached_reference(2, 1, 4, 12345)

    assert calls == 1
    assert first is second
    assert np.array_equal(first, second)
    assert first.flags.writeable is False


def test_whestbench_blackbox_workspace_hides_private_suite(tmp_path: Path) -> None:
    state = State(-1, [], whest_env.INITIAL_ESTIMATOR_CODE, -0.1)
    env = whest_env.WhestBenchEnv(
        initial_state=state,
        config=SimpleNamespace(problem_type="arc_whestbench_2026"),
    )

    prompt = env.build_blackbox_autonomous_prompt(
        prompt="Improve the estimator.",
        workspace=tmp_path,
        eval_timeout=30,
        num_cpus_per_task=1,
        socket_path="/tmp/whest-public-test.sock",
    )

    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "eval_client.py",
        "submission.py",
    ]
    assert "public-50 evaluator" in prompt
    assert "private-50 instances" in prompt
    assert "not\navailable through this evaluator" in prompt
    assert "50,51,52" not in prompt
    compile(
        (tmp_path / "eval_client.py").read_text(encoding="utf-8"),
        "eval_client.py",
        "exec",
    )
