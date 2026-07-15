from __future__ import annotations

import json
from pathlib import Path

from examples.erdos_min_overlap.env import (
    ErdosMinOverlapEnv,
    ErdosMinOverlapRewardEvaluator,
    _erdos_blackbox_eval_client_source,
)
from ttt_discover import DiscoverConfig, State


def test_erdos_blackbox_client_forwards_parent_state() -> None:
    state = State(timestep=-1, construction=[0.5, 0.5], code="", value=-0.5)
    source = _erdos_blackbox_eval_client_source(
        socket_path="/tmp/not-used.sock",
        state=state.to_dict(),
    )

    assert '"submission": source' in source
    assert '"state": STATE' in source
    state_json = json.dumps(
        state.to_dict(), ensure_ascii=False, separators=(",", ":")
    )
    assert state_json in source
    compile(source, "eval_client.py", "exec")


def test_erdos_blackbox_autonomous_workspace_is_task_specific(tmp_path: Path) -> None:
    state = State(timestep=-1, construction=[0.5, 0.5], code="", value=-0.5)
    config = DiscoverConfig(
        env_type=ErdosMinOverlapEnv,
        problem_type="",
        eval_timeout=30,
    )
    env = ErdosMinOverlapEnv(initial_state=state, config=config)

    prompt = env.build_blackbox_autonomous_prompt(
        prompt="Improve the bound.",
        workspace=tmp_path,
        eval_timeout=30,
        num_cpus_per_task=1,
        socket_path="/tmp/erdos-test.sock",
    )

    submission = (tmp_path / "submission.py").read_text(encoding="utf-8")
    client = (tmp_path / "eval_client.py").read_text(encoding="utf-8")
    assert "def run(" in submission
    assert "initial_h_values" in submission
    assert "Autonomous Erdos Blackbox Search Mode" in prompt
    assert "repro.cap_set" not in prompt
    assert "repro.cap_set" not in client
    compile(client, str(tmp_path / "eval_client.py"), "exec")


def test_erdos_reward_evaluator_accepts_plain_workspace_python() -> None:
    plain = "def run():\n    return None\n"
    fenced = f"```python\n{plain}```"

    assert ErdosMinOverlapRewardEvaluator._extract_code(None, plain) == plain.strip()
    assert ErdosMinOverlapRewardEvaluator._extract_code(None, fenced) == plain.strip()
