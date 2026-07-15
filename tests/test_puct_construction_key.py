from __future__ import annotations

from examples.cap_set_priority.env import CapSetPriorityEnv
from ttt_discover.algorithms.state import State
from ttt_discover.algorithms.ttt_discover.sampler import PUCTSampler


class _OrderedConstructionEnv:
    state_type = State

    @classmethod
    def create_initial_state(cls, problem_type: str) -> State:
        del problem_type
        return State(
            timestep=-1,
            construction=[[0], [1]],
            code="initial",
            value=2.0,
        )


def _state(construction: list[list[int]], *, value: float = 2.0) -> State:
    return State(
        timestep=0,
        construction=construction,
        code="candidate",
        value=value,
    )


def test_cap_set_construction_key_ignores_point_order() -> None:
    construction = [[0, 1, 2], [2, 0, 1], [1, 2, 0]]

    assert CapSetPriorityEnv.construction_key(construction) == (
        (0, 1, 2),
        (1, 2, 0),
        (2, 0, 1),
    )
    assert CapSetPriorityEnv.construction_key(
        list(reversed(construction))
    ) == CapSetPriorityEnv.construction_key(construction)
    assert CapSetPriorityEnv.construction_key(
        [[0, 1, 2], [2, 0, 1], [1, 1, 0]]
    ) != CapSetPriorityEnv.construction_key(construction)


def test_puct_sampler_rejects_reordered_cap_set(tmp_path) -> None:
    sampler = PUCTSampler(
        file_path=str(tmp_path / "puct_sampler.json"),
        env_type=CapSetPriorityEnv,
        problem_type="1",
        topk_children=0,
    )
    parent = sampler.get_initial_states()[0]
    reordered = _state(list(reversed(parent.construction)))

    assert sampler.has_state(reordered)
    sampler.update_states([reordered], [parent], save=False)

    assert len(sampler._states) == 1


def test_puct_sampler_keeps_default_order_sensitive_behavior(tmp_path) -> None:
    sampler = PUCTSampler(
        file_path=str(tmp_path / "puct_sampler.json"),
        env_type=_OrderedConstructionEnv,
        topk_children=0,
    )
    parent = sampler.get_initial_states()[0]
    reordered = _state([[1], [0]])

    assert not sampler.has_state(reordered)
    sampler.update_states([reordered], [parent], save=False)

    assert len(sampler._states) == 2
