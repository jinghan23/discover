from __future__ import annotations

import json
from pathlib import Path

from ttt_discover import DiscoverConfig, State
from ttt_discover.algorithms.autoevolve.loop import AutoEvolvePool
from ttt_discover.algorithms.ttt_discover.sampler import load_initial_states
from ttt_discover.tasks import Task


class _InitialStateEnv:
    state_type = State

    @classmethod
    def create_initial_state(cls, problem_type: str) -> State:
        del problem_type
        return State(timestep=-1, construction=[], code="", value=0.0)


def test_initial_state_loader_accepts_reproduction_pool_wrapper(
    tmp_path: Path,
) -> None:
    states = [
        State(timestep=-1, construction=[0.5, 0.5], code="", value=-0.5),
        State(timestep=-1, construction=[0.4, 0.6], code="", value=-0.4),
    ]
    pool_path = tmp_path / "initial_pool.json"
    pool_path.write_text(
        json.dumps({"name": "wrapped", "states": [state.to_dict() for state in states]}),
        encoding="utf-8",
    )

    loaded = load_initial_states(str(pool_path), _InitialStateEnv)

    assert [state.value for state in loaded] == [-0.5, -0.4]


def test_autoevolve_pool_uses_configured_initial_pool(tmp_path: Path) -> None:
    state = State(timestep=-1, construction=[0.4, 0.6], code="", value=-0.4)
    pool_path = tmp_path / "initial_pool.json"
    pool_path.write_text(
        json.dumps({"states": [state.to_dict()]}),
        encoding="utf-8",
    )
    config = DiscoverConfig(env_type=_InitialStateEnv, problem_type="")
    task = Task.from_config(config)

    pool = AutoEvolvePool(
        log_path=str(tmp_path / "run"),
        task=task,
        batch_size=1,
        initial_state_file=str(pool_path),
    )

    assert pool.sample_states(1)[0].construction == [0.4, 0.6]
