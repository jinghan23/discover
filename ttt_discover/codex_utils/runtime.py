"""Core runtime types and rollout helpers for Codex discovery."""

from __future__ import annotations

import asyncio
import itertools
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Sequence, TypeAlias

import chz
import numpy as np

import ttt_discover.codex_utils.logtree as logtree
from ttt_discover.codex_utils.misc_utils import all_same, dict_mean, safezip

logger = logging.getLogger(__name__)


@dataclass
class EncodedTextChunk:
    tokens: list[int]

    @property
    def length(self) -> int:
        return len(self.tokens)


ModelInputChunk = EncodedTextChunk


@dataclass
class ModelInput:
    chunks: list[ModelInputChunk] = field(default_factory=list)

    @property
    def length(self) -> int:
        return sum(chunk.length for chunk in self.chunks)

    @classmethod
    def empty(cls) -> "ModelInput":
        return cls(chunks=[])

    def append_int(self, token: int) -> "ModelInput":
        chunks = list(self.chunks)
        chunks.append(EncodedTextChunk(tokens=[token]))
        return ModelInput(chunks=chunks)


class _TypesNamespace:
    EncodedTextChunk = EncodedTextChunk
    ModelInputChunk = ModelInputChunk


types = _TypesNamespace()


StopCondition: TypeAlias = list[str] | list[int]
Action: TypeAlias = list[int]
Observation: TypeAlias = ModelInput
Logprobs: TypeAlias = list[float]
Metrics: TypeAlias = dict[str, Any]


@dataclass
class TokensWithLogprobs:
    tokens: list[int]
    maybe_logprobs: list[float] | None
    maybe_mask: list[float] | None = None

    @property
    def logprobs(self) -> list[float]:
        if self.maybe_logprobs is None:
            raise ValueError("Logprobs are not available")
        return self.maybe_logprobs

    @property
    def mask(self) -> list[float]:
        if self.maybe_mask is None:
            return [1.0] * len(self.tokens)
        return self.maybe_mask


def to_json_serializable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json_serializable(v) for v in obj]
    return obj


class State(ABC):
    id: str
    timestep: int
    value: float
    code: str
    construction: list[Any]
    parent_values: list[float]
    parents: list[dict]
    observation: str

    def __init__(
        self,
        timestep: int,
        construction: list[Any],
        code: str,
        value: float = None,
        parent_values: list[float] = None,
        parents: list[dict] = None,
        id: str = None,
        observation: str = "",
    ):
        self.id = id if id is not None else str(uuid.uuid4())
        self.timestep = timestep
        self.value = value
        self.construction = to_json_serializable(construction)
        self.code = code
        self.parent_values = parent_values if parent_values is not None else []
        self.parents = parents if parents is not None else []
        self.observation = observation

    def to_dict(self) -> dict:
        return {
            "type": "State",
            "id": self.id,
            "timestep": self.timestep,
            "value": self.value,
            "parent_values": self.parent_values,
            "parents": self.parents,
            "observation": self.observation,
            "construction": to_json_serializable(self.construction),
            "code": self.code,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "State":
        return cls(
            timestep=d["timestep"],
            construction=d["construction"],
            code=d["code"],
            value=d.get("value"),
            parent_values=d.get("parent_values", []),
            parents=d.get("parents", []),
            id=d.get("id"),
            observation=d.get("observation", ""),
        )

    def to_prompt(self, target, metric_name: str = "value", maximize: bool = True, language: str = ""):
        value_ctx = f"You are iteratively optimizing {metric_name}."
        improvement_direction = "higher" if maximize else "lower"

        has_code = self.code and self.code.strip()
        if has_code:
            value_ctx += "\nHere is the last code we ran:\n"
            if language:
                value_ctx += f"```{language}\n{self.code}\n```"
            else:
                value_ctx += f"{self.code}"
        else:
            value_ctx += "\nNo previous code available."

        if self.parent_values and self.value is not None and self.construction:
            before_value = self.parent_values[0] if maximize else -self.parent_values[0]
            after_value = self.value if maximize else -self.value
            current_gap = target - after_value if maximize else after_value - target
            value_ctx += f"\nHere is the {metric_name} before and after running the code above ({improvement_direction} is better): {before_value:.6f} -> {after_value:.6f}"
            value_ctx += f"\nTarget: {target}. Current gap: {current_gap:.6f}. Further improvements will also be generously rewarded."
        elif self.value is not None:
            after_value = self.value if maximize else -self.value
            current_gap = target - after_value if maximize else after_value - target
            value_ctx += f"\nCurrent {metric_name} ({improvement_direction} is better): {after_value:.6f}"
            value_ctx += f"\nTarget: {target}. Current gap: {current_gap:.6f}. Further improvements will also be generously rewarded."
        else:
            value_ctx += f"\nTarget {metric_name}: {target}"

        if self.observation and self.observation.strip():
            stdout = self.observation.strip()
            if len(stdout) > 500:
                stdout = "\n\n\t\t ...(TRUNCATED)...\n" + stdout[-500:]
            value_ctx += f"\n\n--- Previous Program Output ---\n{stdout}\n--- End Output ---"

        return value_ctx


def _state_class_by_name(name: str) -> type:
    def _all_subclasses(cls: type) -> set[type]:
        return set(cls.__subclasses__()) | {s for c in cls.__subclasses__() for s in _all_subclasses(c)}

    for cls in [State] + list(_all_subclasses(State)):
        if cls.__name__ == name:
            return cls
    raise ValueError(f"Unknown state type: {name}")


def state_from_dict(d: dict | None, state_type: type | None = None) -> State | None:
    if d is None:
        return None
    cls = state_type if state_type is not None else _state_class_by_name(d.get("type", "State"))
    return cls.from_dict(d)


@dataclass
class StepResult:
    reward: float
    episode_done: bool
    next_observation: Observation
    next_stop_condition: StopCondition
    metrics: Metrics = field(default_factory=dict)


@dataclass
class Transition:
    ob: Observation
    ac: TokensWithLogprobs
    reward: float
    episode_done: bool
    metrics: Metrics = field(default_factory=dict)


@dataclass(frozen=True)
class Trajectory:
    transitions: list[Transition]
    final_ob: Observation


@dataclass
class TrajectoryGroup:
    trajectories_G: list[Trajectory]
    final_rewards_G: list[float]
    metrics_G: list[Metrics]

    def get_total_rewards(self) -> list[float]:
        return [
            sum(transition.reward for transition in trajectory.transitions) + final_reward
            for trajectory, final_reward in safezip(self.trajectories_G, self.final_rewards_G)
        ]


@dataclass
class Experience:
    prev_state: State | None
    action: Action | None
    step_result: StepResult
    next_state: State
    is_initial: bool = False

    def to_dict(self) -> dict:
        return {
            "prev_state": self.prev_state.to_dict() if self.prev_state else None,
            "action": to_json_serializable(self.action) if self.action is not None else None,
            "step_result": {
                "reward": to_json_serializable(self.step_result.reward),
                "episode_done": self.step_result.episode_done,
                "metrics": to_json_serializable(self.step_result.metrics),
            },
            "next_state": self.next_state.to_dict(),
            "is_initial": self.is_initial,
        }

    @classmethod
    def from_dict(
        cls,
        d: dict,
        *,
        state_type: type[State] | None = None,
    ) -> "Experience":
        step_result = StepResult(
            reward=d["step_result"]["reward"],
            episode_done=d["step_result"]["episode_done"],
            next_observation=None,
            next_stop_condition=None,
            metrics=d["step_result"].get("metrics", {}),
        )
        next_state = state_from_dict(d["next_state"], state_type=state_type)
        if next_state is None:
            raise ValueError("Experience is missing next_state")
        return cls(
            prev_state=state_from_dict(d["prev_state"], state_type=state_type),
            action=d.get("action"),
            step_result=step_result,
            next_state=next_state,
            is_initial=d.get("is_initial", False),
        )


class Env(ABC):
    @abstractmethod
    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        pass

    @abstractmethod
    async def step(self, action: Action, *args: Any, **kwargs: Any) -> StepResult:
        pass


class EnvGroupBuilder(ABC):
    @abstractmethod
    async def make_envs(self) -> Sequence[Env]:
        pass

    async def compute_group_rewards(
        self,
        trajectory_group: list[Trajectory],
        env_group: Sequence[Env],
    ) -> list[tuple[float, Metrics]]:
        return [(0.0, {}) for _ in trajectory_group]

    def logging_tags(self) -> list[str]:
        return []


class RLDataset(ABC):
    @abstractmethod
    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass


@chz.chz
class RLDatasetBuilder:
    @abstractmethod
    async def __call__(self) -> RLDataset:
        pass


@logtree.scope_header_decorator
async def do_single_rollout(policy: Any, env: Env, step_idx: int) -> Trajectory:
    transitions = []
    ob, stop_condition = await env.initial_observation()
    while True:
        t_policy_start = time.time()
        ac_with_logprobs = await policy(ob, stop_condition)
        t_policy = time.time() - t_policy_start

        t_env_start = time.time()
        step_result = await env.step(ac_with_logprobs.tokens, step_idx)
        t_env = time.time() - t_env_start

        step_metrics = dict(step_result.metrics) if step_result.metrics else {}
        step_metrics["time/policy"] = t_policy
        step_metrics["time/env_step"] = t_env
        transitions.append(
            Transition(
                ob=ob,
                ac=ac_with_logprobs,
                reward=step_result.reward,
                episode_done=step_result.episode_done,
                metrics=step_metrics,
            )
        )
        ob = step_result.next_observation
        stop_condition = step_result.next_stop_condition
        if step_result.episode_done:
            break
    return Trajectory(transitions=transitions, final_ob=ob)


@logtree.scope_header_decorator
async def do_group_rollout(
    env_group_builder: EnvGroupBuilder,
    policy: Any,
    step_idx: int,
) -> TrajectoryGroup:
    envs_G: Sequence[Env] = await env_group_builder.make_envs()
    rollout_results = await asyncio.gather(
        *[do_single_rollout(policy, env, step_idx) for env in envs_G],
        return_exceptions=True,
    )
    trajectories_G: list[Trajectory] = []
    successful_envs_G: list[Env] = []
    for env, result in zip(envs_G, rollout_results, strict=True):
        if isinstance(result, BaseException):
            logger.warning(
                "Skipping failed trajectory in group rollout at step %s: %r",
                step_idx,
                result,
            )
            continue
        trajectories_G.append(result)
        successful_envs_G.append(env)

    if not trajectories_G:
        raise RuntimeError(f"All {len(envs_G)} trajectories failed at step {step_idx}")

    t_reward_start = time.time()
    rewards_and_metrics_G = await env_group_builder.compute_group_rewards(
        trajectories_G,
        successful_envs_G,
    )
    t_reward = time.time() - t_reward_start
    rewards_G, metrics_G = zip(*rewards_and_metrics_G, strict=True)

    per_traj_group_reward = t_reward / max(1, len(trajectories_G))
    for metrics in metrics_G:
        if isinstance(metrics, dict):
            metrics["time/reward_compute"] = per_traj_group_reward

    with logtree.scope_header("Trajectory Summary"):
        for i, (traj, final_reward, reward_metrics) in enumerate(
            zip(trajectories_G, rewards_G, metrics_G, strict=True)
        ):
            rows = []
            step_reward_sum = 0.0
            for t_idx, t in enumerate(traj.transitions):
                step_reward_sum += t.reward
                rows.append(
                    {
                        "step": t_idx,
                        "ob_len": t.ob.length,
                        "ac_len": len(t.ac.tokens),
                        "reward": f"{t.reward:.3f}",
                    }
                )
            rows.append(
                {
                    "step": "final",
                    "ob_len": traj.final_ob.length,
                    "ac_len": "-",
                    "reward": f"{final_reward:.3f}",
                }
            )
            rows.append(
                {
                    "step": "total",
                    "ob_len": "-",
                    "ac_len": "-",
                    "reward": f"{step_reward_sum + final_reward:.3f}",
                }
            )
            logtree.table(rows, caption=f"Trajectory {i}")

            metrics_payload = {
                "final_reward": final_reward,
                "total_reward": step_reward_sum + final_reward,
            }
            if reward_metrics:
                metrics_payload["reward_metrics"] = reward_metrics
            step_metrics = [
                {"step": idx, **transition.metrics}
                for idx, transition in enumerate(traj.transitions)
                if transition.metrics
            ]
            if step_metrics:
                metrics_payload["step_metrics"] = step_metrics
            logtree.details(
                json.dumps(to_json_serializable(metrics_payload), indent=2),
                summary=f"Trajectory {i} metrics",
            )

    return TrajectoryGroup(trajectories_G, list(rewards_G), list(metrics_G))


def _compute_by_group_metrics(trajectory_groups_P: list[TrajectoryGroup], good_thresh: float = 0.5):
    n_groups = len(trajectory_groups_P)
    n_mixed = n_good = n_bad = 0
    for tg in trajectory_groups_P:
        grp_rewards = tg.get_total_rewards()
        if all_same(grp_rewards):
            if grp_rewards[0] >= good_thresh:
                n_good += 1
            else:
                n_bad += 1
        else:
            n_mixed += 1
    return {
        "by_group/frac_mixed": n_mixed / n_groups,
        "by_group/frac_all_good": n_good / n_groups,
        "by_group/frac_all_bad": n_bad / n_groups,
    }


def get_log_table(trajectory_groups_P: list[TrajectoryGroup]):
    table_row_list = []
    for trajectory_group in trajectory_groups_P:
        if "response" not in trajectory_group.trajectories_G[0].transitions[0].metrics:
            return None
        for trajectory in trajectory_group.trajectories_G:
            traj_metrics = trajectory.transitions[0].metrics
            table_row_list.append(
                (
                    traj_metrics["prompt"],
                    traj_metrics["response"],
                    traj_metrics["reward"],
                    traj_metrics["correctness"],
                    traj_metrics["parsed_code"],
                    str(traj_metrics["msg"]),
                    traj_metrics.get("initial_raw_score"),
                )
            )
    return None if len(table_row_list) == 0 else table_row_list


def remove_non_numerical_field(trajectory_groups_P: list[TrajectoryGroup]):
    for trajectory_group in trajectory_groups_P:
        for trajectory in trajectory_group.trajectories_G:
            traj_metrics = trajectory.transitions[0].metrics
            for key in ["prompt_hash", "predicted_grid", "prompt", "response", "ref"]:
                traj_metrics.pop(key, None)


def compute_trajectory_metrics(
    trajectory_groups_P: list[TrajectoryGroup],
    taglist_P: list[list[str]],
) -> dict[str, float]:
    tag2trajgroups = defaultdict(list)
    for taglist, trajectory_group in zip(taglist_P, trajectory_groups_P):
        for tag in taglist:
            tag2trajgroups[tag].append(trajectory_group)

    out = {}
    have_nontrivial_tags = any(
        len(trajgroups) < len(trajectory_groups_P)
        for trajgroups in tag2trajgroups.values()
    )
    if have_nontrivial_tags:
        for tag, trajectory_groups in tag2trajgroups.items():
            out.update(
                {
                    f"env/{tag}/{k}": v
                    for k, v in _compute_trajectory_metrics(trajectory_groups).items()
                }
            )

    log_table = get_log_table(trajectory_groups_P)
    if log_table is not None:
        out.update({"table": log_table})

    remove_non_numerical_field(trajectory_groups_P)
    out.update(
        {f"env/all/{k}": v for k, v in _compute_trajectory_metrics(trajectory_groups_P).items()}
    )
    return out


def _compute_trajectory_metrics(trajectory_groups_P: list[TrajectoryGroup]) -> dict[str, float]:
    flat_trajs_PG = [traj for tg in trajectory_groups_P for traj in tg.trajectories_G]
    flat_transitions = [t for traj in flat_trajs_PG for t in traj.transitions]
    ac_tokens_by_turn = [len(t.ac.tokens) for t in flat_transitions]
    ob_tokens_by_turn = [t.ob.length for t in flat_transitions]
    turns_by_trajectory = [len(traj.transitions) for traj in flat_trajs_PG]
    policy_times = [(t.metrics or {}).get("time/policy", 0.0) for t in flat_transitions]
    env_step_times = [(t.metrics or {}).get("time/env_step", 0.0) for t in flat_transitions]

    metrics = {
        "ac_tokens_per_turn": sum(ac_tokens_by_turn) / sum(turns_by_trajectory),
        "ob_tokens_per_turn": sum(ob_tokens_by_turn) / sum(turns_by_trajectory),
        "turns_per_episode": sum(turns_by_trajectory) / len(flat_trajs_PG),
        "total_episodes": len(flat_trajs_PG),
        "total_turns": sum(turns_by_trajectory),
        "total_ac_tokens": sum(ac_tokens_by_turn),
        "total_ob_tokens": sum(ob_tokens_by_turn),
        "time/sampling_mean": np.mean(policy_times).item() if policy_times else 0.0,
        "time/sampling_max": np.max(policy_times).item() if policy_times else 0.0,
        "time/env_step_mean": np.mean(env_step_times).item() if env_step_times else 0.0,
        "time/env_step_max": np.max(env_step_times).item() if env_step_times else 0.0,
    }

    gr_rewards = [reward for tg in trajectory_groups_P for reward in tg.get_total_rewards()]
    metrics["reward/mean"] = np.mean(gr_rewards).item()
    metrics["reward/max"] = np.max(gr_rewards).item()
    metrics["reward/min"] = np.min(gr_rewards).item()

    transition_metrics = [
        transition.metrics
        for tg in trajectory_groups_P
        for traj in tg.trajectories_G
        for transition in traj.transitions
    ]
    traj_metrics = [metrics for tg in trajectory_groups_P for metrics in tg.metrics_G]
    metrics.update(dict_mean(transition_metrics + traj_metrics))
    metrics.update(_compute_by_group_metrics(trajectory_groups_P))
    return metrics


def dataset_to_env_group_builders(dataset: RLDataset) -> list[EnvGroupBuilder]:
    return list(itertools.chain(*[dataset.get_batch(i) for i in range(len(dataset))]))
