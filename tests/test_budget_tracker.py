import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from ttt_discover.algorithms.runtime import (
    BudgetTracker,
    CandidateResult,
    evaluator_call_count,
    logged_evaluator_call_count,
)


def candidate(metrics):
    return CandidateResult(
        parent_state=None,
        group_idx=0,
        sample_idx=0,
        prompt="",
        response="",
        parsed_code="",
        reward=0.0,
        correctness=0.0,
        raw_score=None,
        msg="",
        metrics=metrics,
    )


class BudgetTrackerTest(unittest.TestCase):
    def test_evaluator_call_count_prefers_explicit_metric(self):
        results = [
            candidate({"budget/evaluator_calls": 0}),
            candidate({"budget/evaluator_calls": 3, "inner/completed_iterations": 9}),
            candidate({"inner/completed_iterations": 4}),
            candidate({}),
        ]

        self.assertEqual(evaluator_call_count(results), 8)

    def test_logged_evaluator_call_count_recovers_resume_total(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "agent_outputs.jsonl"
            rows = [
                {"metrics": {"budget/evaluator_calls": 2}},
                {"metrics": {"inner/completed_iterations": 4}},
                {"metrics": {}},
                {"metrics": {"budget/evaluator_calls": 0}},
            ]
            with path.open("w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")
                f.write("{not json}\n")

            self.assertEqual(logged_evaluator_call_count(tmp), 7)

    def test_budget_tracker_adds_results_and_reports_exceeded(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "agent_outputs.jsonl"
            with path.open("w", encoding="utf-8") as f:
                f.write(json.dumps({"metrics": {"budget/evaluator_calls": 3}}) + "\n")

            cfg = SimpleNamespace(max_evaluator_calls=5)
            tracker = BudgetTracker.from_config(cfg, tmp)

            self.assertFalse(tracker.exceeded)
            metrics = tracker.add(
                [
                    candidate({"budget/evaluator_calls": 1}),
                    candidate({"budget/evaluator_calls": 1}),
                ]
            )

            self.assertTrue(tracker.exceeded)
            self.assertEqual(metrics["budget/evaluator_calls_used_start"], 3)
            self.assertEqual(metrics["budget/evaluator_calls_epoch"], 2)
            self.assertEqual(metrics["budget/evaluator_calls_used"], 5)
            self.assertEqual(metrics["budget/evaluator_calls_remaining"], 0)
            self.assertTrue(metrics["budget/stop_after_epoch"])
            self.assertEqual(tracker.done_frac(0.25), 1.0)


if __name__ == "__main__":
    unittest.main()
