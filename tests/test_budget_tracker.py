import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from ttt_discover.algorithms.runtime import (
    BudgetTracker,
    CandidateResult,
    evaluator_call_count,
    logged_evaluator_call_count,
)
from ttt_discover.eval_runners.blackbox_eval.server import (
    BlackboxVerifier,
    VerifierConfig,
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
            self.assertEqual(tracker.remaining, 2)
            metrics = tracker.add(
                [
                    candidate({"budget/evaluator_calls": 1}),
                    candidate({"budget/evaluator_calls": 1}),
                ]
            )

            self.assertTrue(tracker.exceeded)
            self.assertEqual(tracker.remaining, 0)
            self.assertEqual(metrics["budget/evaluator_calls_used_start"], 3)
            self.assertEqual(metrics["budget/evaluator_calls_epoch"], 2)
            self.assertEqual(metrics["budget/evaluator_calls_used"], 5)
            self.assertEqual(metrics["budget/evaluator_calls_remaining"], 0)
            self.assertTrue(metrics["budget/stop_after_epoch"])
            self.assertEqual(tracker.done_frac(0.25), 1.0)

    def test_budget_tracker_uses_server_high_water_mark(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = SimpleNamespace(max_evaluator_calls=500)
            tracker = BudgetTracker.from_config(cfg, tmp)

            metrics = tracker.add(
                [
                    candidate(
                        {
                            "budget/evaluator_calls": 1,
                            "budget/evaluator_calls_server": 120,
                        }
                    ),
                    candidate(
                        {
                            "budget/evaluator_calls": 1,
                            "budget/evaluator_calls_server": 137,
                        }
                    ),
                ]
            )

            # Server owns the count: take the high-water mark (137), not the
            # per-candidate sum (2), so inner eval_client.py calls are covered.
            self.assertEqual(tracker.used, 137)
            self.assertEqual(metrics["budget/evaluator_calls_epoch"], 137)
            self.assertFalse(tracker.exceeded)

            # A stale, smaller server report must never lower the count.
            tracker.add([candidate({"budget/evaluator_calls_server": 100})])
            self.assertEqual(tracker.used, 137)

    def test_logged_count_prefers_server_max(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "agent_outputs.jsonl"
            rows = [
                {
                    "metrics": {
                        "budget/evaluator_calls": 1,
                        "budget/evaluator_calls_server": 10,
                    }
                },
                {
                    "metrics": {
                        "budget/evaluator_calls": 1,
                        "budget/evaluator_calls_server": 42,
                    }
                },
                {"metrics": {"budget/evaluator_calls": 1}},
            ]
            with path.open("w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")

            # Authoritative server max (42), not the per-candidate sum (3).
            self.assertEqual(logged_evaluator_call_count(tmp), 42)


class _CountingEvaluator:
    """Minimal trusted-verifier stand-in that always succeeds."""

    def __init__(self, **kwargs):
        pass

    def get_reward(self, submission, state=None):
        time.sleep(0.002)  # widen the race window for the concurrency test
        return {"reward": 1.0, "correctness": 1.0, "raw_score": 1.0, "msg": "ok"}


def _make_verifier(tmp, max_evaluations):
    return BlackboxVerifier(
        VerifierConfig(
            evaluator_type=_CountingEvaluator,
            problem_type="",
            log_dir=Path(tmp),
            eval_timeout=10,
            num_cpus_per_task=1,
            evaluator_kwargs={},
            state_factory=lambda: None,
            max_evaluations=max_evaluations,
        )
    )


class BlackboxBudgetCapTest(unittest.TestCase):
    def test_hard_cap_never_exceeded_under_concurrency(self):
        cap = 50
        extra = 37
        with tempfile.TemporaryDirectory() as tmp:
            verifier = _make_verifier(tmp, cap)
            responses = []
            lock = threading.Lock()

            def worker(i):
                resp = verifier.evaluate(
                    {"request_id": str(i), "submission": "code"}
                )
                with lock:
                    responses.append(resp)

            threads = [
                threading.Thread(target=worker, args=(i,))
                for i in range(cap + extra)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            granted = [r for r in responses if r.get("ok")]
            denied = [r for r in responses if r.get("budget_exhausted")]
            self.assertEqual(len(granted), cap)  # exactly cap, never cap + 1
            self.assertEqual(len(denied), extra)
            self.assertEqual(verifier._eval_count, cap)
            self.assertEqual(
                sorted(r["budget_used"] for r in granted),
                list(range(1, cap + 1)),
            )

    def test_no_cap_allows_all(self):
        with tempfile.TemporaryDirectory() as tmp:
            verifier = _make_verifier(tmp, None)
            for i in range(200):
                resp = verifier.evaluate(
                    {"request_id": str(i), "submission": "code"}
                )
                self.assertTrue(resp["ok"])
            self.assertEqual(verifier._eval_count, 200)

    def test_malformed_request_does_not_consume_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            verifier = _make_verifier(tmp, 5)
            resp = verifier.evaluate({"request_id": "x", "submission": "   "})
            self.assertFalse(resp["ok"])
            self.assertEqual(resp["stage"], "request")
            self.assertEqual(verifier._eval_count, 0)


if __name__ == "__main__":
    unittest.main()
