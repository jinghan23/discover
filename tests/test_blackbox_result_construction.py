from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from ttt_discover import State
from ttt_discover.eval_runners.blackbox import BlackboxRunner
from ttt_discover.eval_runners.blackbox_eval.server import (
    BlackboxVerifier,
    VerifierConfig,
)
from ttt_discover.tasks.base import VerifyResult


class _ResponseSocket:
    def __init__(self, response: dict[str, object]):
        self.response = json.dumps(response).encode("utf-8") + b"\n"
        self.sent = b""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        del exc_type, exc, traceback

    def sendall(self, data: bytes) -> None:
        self.sent += data

    def recv(self, size: int) -> bytes:
        del size
        response, self.response = self.response, b""
        return response


def test_verify_result_preserves_task_details() -> None:
    result = VerifyResult.from_reward_dict(
        {
            "reward": 2.0,
            "raw_score": 0.5,
            "correctness": 1.0,
            "details": {"task": {"suite": "search"}},
        }
    )

    assert result.details == {"task": {"suite": "search"}}


def test_blackbox_verifier_returns_result_construction(tmp_path: Path) -> None:
    class Evaluator:
        def __init__(self, **kwargs):
            del kwargs

        def get_reward(self, submission, state):
            del submission, state
            return {
                "reward": 2.0,
                "raw_score": 0.5,
                "correctness": 1.0,
                "result_construction": [0.4, 0.6],
                "details": {"task": {"suite": "search"}},
            }

    verifier = BlackboxVerifier(
        VerifierConfig(
            evaluator_type=Evaluator,
            problem_type="",
            log_dir=tmp_path,
            eval_timeout=10,
            num_cpus_per_task=1,
            evaluator_kwargs={},
            state_factory=lambda: None,
        )
    )

    response = verifier.evaluate({"submission": "candidate"})

    assert response["result_construction"] == [0.4, 0.6]
    assert response["details"] == {"task": {"suite": "search"}}


def test_blackbox_runner_preserves_result_construction() -> None:
    socket = _ResponseSocket(
        {
            "ok": True,
            "stage": "complete",
            "message": "pass score=0.5",
            "reward": 2.0,
            "raw_score": 0.5,
            "correctness": 1.0,
            "result_construction": [0.4, 0.6],
            "details": {"task": {"suite": "search"}},
        }
    )
    runner = BlackboxRunner(socket_path="/tmp/not-used.sock")
    runner._connect = lambda: socket
    env = SimpleNamespace(
        problem_type="",
        state=State(timestep=-1, construction=[0.5, 0.5], code="", value=-0.5),
    )

    result = runner.evaluate(env, "candidate")

    assert result.result_construction == [0.4, 0.6]
    assert result.details == {"task": {"suite": "search"}}
    request = json.loads(socket.sent.decode("utf-8"))
    assert request["state"]["construction"] == [0.5, 0.5]
