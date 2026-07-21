import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import flopscope.numpy as fnp
from whestbench.domain import MLP
from whestbench.scoring import ContestData, ContestSpec

from examples.aicrowd_whestbench.env import (
    OfficialSuiteConfig,
    WhestBenchEnv,
    WhestBenchRewardEvaluator,
)
from examples.aicrowd_whestbench.prompt import build_whestbench_prompt
from ttt_discover import State


VALID_ZERO_ESTIMATOR = '''import flopscope.numpy as fnp
from whestbench import BaseEstimator


class Estimator(BaseEstimator):
    def predict(self, mlp, budget):
        del budget
        return fnp.zeros((mlp.depth, mlp.width))
'''


WRONG_SHAPE_ESTIMATOR = '''import flopscope.numpy as fnp
from whestbench import BaseEstimator


class Estimator(BaseEstimator):
    def predict(self, mlp, budget):
        del budget
        return fnp.zeros((mlp.width, mlp.depth))
'''


FUNCTION_ONLY_CANDIDATE = '''import flopscope.numpy as fnp


def estimate(mlp, budget):
    del budget
    return fnp.zeros((mlp.depth, mlp.width))
'''


def tiny_contest() -> ContestData:
    width = 2
    depth = 1
    flop_budget = 1_000_000_000_000_000
    mlp = MLP(
        width=width,
        depth=depth,
        weights=[fnp.zeros((width, width), dtype=fnp.float32)],
        seed=7,
        name="tiny-mlp",
    )
    all_target = fnp.array([[1.0, 2.0]], dtype=fnp.float32)
    return ContestData(
        spec=ContestSpec(
            width=width,
            depth=depth,
            n_mlps=1,
            flop_budget=flop_budget,
            ground_truth_samples=100,
            seed=0,
        ),
        mlps=[mlp],
        all_layer_targets=[all_target],
        final_targets=[all_target[-1]],
        avg_variances=[0.0],
    )


def tiny_config(data: ContestData) -> OfficialSuiteConfig:
    return OfficialSuiteConfig(
        dataset="tiny-test",
        revision=None,
        split="test",
        n_mlps=1,
        flop_budget=data.spec.flop_budget,
        runner="local",
        streaming=False,
    )


class WhestBenchOfficialScoringTest(unittest.TestCase):
    def setUp(self):
        self.data = tiny_contest()
        self.evaluator = WhestBenchRewardEvaluator(
            problem_type="arc_whestbench_2026",
            suite_config=tiny_config(self.data),
            contest_data=self.data,
        )
        self.state = State(-1, [], "", 0.0)

    def test_valid_candidate_uses_official_adjusted_score(self):
        result = self.evaluator.get_reward(VALID_ZERO_ESTIMATOR, self.state)

        self.assertEqual(result["correctness"], 1.0)
        self.assertAlmostEqual(result["metrics"]["whestbench/final_layer_mse"], 2.5)
        self.assertAlmostEqual(result["raw_score"], 0.25)
        self.assertAlmostEqual(
            result["metrics"]["whestbench/mean_score_multiplier"], 0.1
        )
        self.assertEqual(result["metrics"]["whestbench/n_failed_mlps"], 0)

    def test_wrong_shape_gets_zero_prediction_without_discount(self):
        result = self.evaluator.get_reward(WRONG_SHAPE_ESTIMATOR, self.state)

        self.assertEqual(result["correctness"], 0.0)
        self.assertAlmostEqual(result["raw_score"], 2.5)
        self.assertAlmostEqual(
            result["metrics"]["whestbench/mean_score_multiplier"], 1.0
        )
        self.assertEqual(result["metrics"]["whestbench/n_failed_mlps"], 1)

    def test_function_only_candidate_fails_official_contract(self):
        result = self.evaluator.get_reward(FUNCTION_ONLY_CANDIDATE, self.state)

        self.assertEqual(result["correctness"], 0.0)
        self.assertAlmostEqual(result["raw_score"], 2.5)
        self.assertEqual(result["metrics"]["whestbench/n_failed_mlps"], 1)

    def test_official_subprocess_runner_path(self):
        config = replace(tiny_config(self.data), runner="subprocess")
        evaluator = WhestBenchRewardEvaluator(
            problem_type="arc_whestbench_2026",
            suite_config=config,
            contest_data=self.data,
        )

        result = evaluator.get_reward(VALID_ZERO_ESTIMATOR, self.state)

        self.assertEqual(result["correctness"], 1.0)
        self.assertAlmostEqual(result["raw_score"], 0.25)
        self.assertEqual(result["metrics"]["whestbench/n_failed_mlps"], 0)


class WhestBenchAutonomousWorkspaceTest(unittest.TestCase):
    def _env(self) -> WhestBenchEnv:
        state = State(-1, [], VALID_ZERO_ESTIMATOR, -0.25)
        return WhestBenchEnv(
            initial_state=state,
            config=SimpleNamespace(problem_type="arc_whestbench_2026"),
        )

    def test_isolated_workspace_contains_only_submission(self):
        env = self._env()
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            prompt = env.build_autonomous_prompt(
                prompt="improve the estimator",
                workspace=workspace,
                eval_timeout=30,
                num_cpus_per_task=1,
            )
            files = sorted(path.name for path in workspace.iterdir())
            submission = (workspace / "submission.py").read_text(encoding="utf-8")

        self.assertEqual(files, ["submission.py"])
        self.assertIn("class Estimator", submission)
        self.assertIn("no evaluation data or evaluator", prompt)

    def test_blackbox_workspace_contains_submission_and_client(self):
        env = self._env()

        with TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            prompt = env.build_blackbox_autonomous_prompt(
                prompt="improve the estimator",
                workspace=workspace,
                eval_timeout=30,
                num_cpus_per_task=1,
                socket_path="/tmp/test-whestbench.sock",
            )

            submission = (workspace / "submission.py").read_text(encoding="utf-8")
            client = (workspace / "eval_client.py").read_text(encoding="utf-8")

        self.assertIn("class Estimator", submission)
        self.assertIn("/tmp/test-whestbench.sock", client)
        self.assertIn("python eval_client.py", prompt)
        self.assertIn("Lower `raw_score` is better", prompt)

    def test_launch_time_mode_forcing_loads_one_prompt_file(self):
        with TemporaryDirectory() as tmp:
            mode_file = Path(tmp) / "exact_radial.md"
            mode_file.write_text(
                "--- Diversity Mode ---\nMode: `exact_radial`\n",
                encoding="utf-8",
            )

            prompt = build_whestbench_prompt(mode_file)

            self.assertIn("Mode: `exact_radial`", prompt)
            self.assertEqual(prompt.count("--- Diversity Mode ---"), 1)
            with self.assertRaisesRegex(FileNotFoundError, "not found"):
                build_whestbench_prompt(Path(tmp) / "missing.md")


if __name__ == "__main__":
    unittest.main()
