"""AB-MCTS-A sampler implemented on the local discovery loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
import math
import os
from typing import Any, Literal
import uuid

import numpy as np

from ttt_discover.algorithms.state import State, state_from_dict, to_json_serializable
from ttt_discover.algorithms.ttt_discover.sampler import (
    StateSampler,
    _atomic_write_json,
    _file_lock,
    _read_json_or_default,
    _sampler_file_for_step,
    create_initial_state,
)

logger = logging.getLogger(__name__)

ModelSelectionStrategy = Literal[
    "stack",
    "multiarm_bandit_thompson",
    "multiarm_bandit_ucb",
]
DistributionType = Literal["gaussian", "beta"]


def _utc_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _as_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


@dataclass
class RewardModel:
    dist_type: str = "gaussian"
    prior_mean: float = 0.0
    prior_std: float = 1.0
    prior_strength: float = 1.0
    beta_a: float = 0.5
    beta_b: float = 0.5
    observations: list[float] = field(default_factory=list)

    def tell(self, reward: float) -> None:
        reward = _as_float(reward)
        if self.dist_type == "beta":
            reward = max(0.0, min(1.0, reward))
        self.observations.append(reward)

    def mean(self) -> float:
        if self.observations:
            return float(np.mean(self.observations))
        return float(self.prior_mean)

    def draw(self, rng: np.random.Generator) -> float:
        if self.dist_type == "beta":
            successes = sum(max(0.0, min(1.0, x)) for x in self.observations)
            failures = len(self.observations) - successes
            return float(rng.beta(self.beta_a + successes, self.beta_b + failures))

        if not self.observations:
            return float(rng.normal(self.prior_mean, max(self.prior_std, 1e-6)))

        n = len(self.observations)
        mean = float(np.mean(self.observations))
        obs_std = float(np.std(self.observations, ddof=1)) if n > 1 else self.prior_std
        scale = max(obs_std, 1e-6) / math.sqrt(max(n + self.prior_strength, 1.0))
        return float(rng.normal(mean, scale))

    def to_dict(self) -> dict[str, Any]:
        return {
            "dist_type": self.dist_type,
            "prior_mean": self.prior_mean,
            "prior_std": self.prior_std,
            "prior_strength": self.prior_strength,
            "beta_a": self.beta_a,
            "beta_b": self.beta_b,
            "observations": list(self.observations),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RewardModel":
        return cls(
            dist_type=str(data.get("dist_type", "gaussian")),
            prior_mean=_as_float(data.get("prior_mean"), 0.0),
            prior_std=max(_as_float(data.get("prior_std"), 1.0), 1e-6),
            prior_strength=max(_as_float(data.get("prior_strength"), 1.0), 1e-6),
            beta_a=max(_as_float(data.get("beta_a"), 0.5), 1e-6),
            beta_b=max(_as_float(data.get("beta_b"), 0.5), 1e-6),
            observations=[_as_float(x) for x in data.get("observations", [])],
        )


class BanditNodeState:
    """TreeQuest-style per-node AB-MCTS-A posterior state."""

    def __init__(
        self,
        *,
        actions: list[str],
        dist_type: str = "gaussian",
        model_selection_strategy: str = "multiarm_bandit_thompson",
        prior_mean: float = 0.0,
        prior_std: float = 1.0,
        prior_strength: float = 1.0,
        beta_a: float = 0.5,
        beta_b: float = 0.5,
    ):
        if model_selection_strategy not in {
            "stack",
            "multiarm_bandit_thompson",
            "multiarm_bandit_ucb",
        }:
            raise ValueError(
                "abmcts_model_selection_strategy must be one of "
                "stack, multiarm_bandit_thompson, multiarm_bandit_ucb"
            )
        if not actions:
            raise ValueError("AB-MCTS requires at least one action")
        self.actions = list(actions)
        self.dist_type = dist_type
        self.model_selection_strategy = model_selection_strategy
        self.prior_mean = float(prior_mean)
        self.prior_std = max(float(prior_std), 1e-6)
        self.prior_strength = max(float(prior_strength), 1e-6)
        self.beta_a = max(float(beta_a), 1e-6)
        self.beta_b = max(float(beta_b), 1e-6)
        self.action_models = {action: self._new_model() for action in self.actions}
        if self.model_selection_strategy.startswith("multiarm_bandit_"):
            self.gen_vs_cont_models = {
                "shared": {"GEN": self._new_model(), "CONT": self._new_model()}
            }
            self.node_models: dict[str, dict[int, RewardModel]] = {"shared": {}}
        else:
            self.gen_vs_cont_models = {
                action: {"GEN": self._new_model(), "CONT": self._new_model()}
                for action in self.actions
            }
            self.node_models = {action: {} for action in self.actions}
        self.child_node_to_action: dict[int, str] = {}

    def _new_model(self) -> RewardModel:
        return RewardModel(
            dist_type=self.dist_type,
            prior_mean=self.prior_mean,
            prior_std=self.prior_std,
            prior_strength=self.prior_strength,
            beta_a=self.beta_a,
            beta_b=self.beta_b,
        )

    def update_action_reward(self, action: str, reward: float) -> None:
        if action not in self.action_models:
            self.action_models[action] = self._new_model()
        if self.model_selection_strategy.startswith("multiarm_bandit_"):
            self.gen_vs_cont_models["shared"]["GEN"].tell(reward)
        else:
            self.gen_vs_cont_models.setdefault(
                action, {"GEN": self._new_model(), "CONT": self._new_model()}
            )["GEN"].tell(reward)
        self.action_models[action].tell(reward)

    def update_node_reward(self, child_node_id: int, reward: float) -> None:
        action = self.child_node_to_action.get(child_node_id)
        if action is None:
            return
        if self.model_selection_strategy.startswith("multiarm_bandit_"):
            self.node_models.setdefault("shared", {}).setdefault(
                child_node_id, self._new_model()
            ).tell(reward)
            self.gen_vs_cont_models["shared"]["CONT"].tell(reward)
        else:
            self.node_models.setdefault(action, {}).setdefault(
                child_node_id, self._new_model()
            ).tell(reward)
            self.gen_vs_cont_models.setdefault(
                action, {"GEN": self._new_model(), "CONT": self._new_model()}
            )["CONT"].tell(reward)
        self.action_models.setdefault(action, self._new_model()).tell(reward)

    def register_new_child_node(self, action: str, child_node_id: int, score: float) -> None:
        self.child_node_to_action[child_node_id] = action
        if self.model_selection_strategy == "stack":
            self.node_models.setdefault(action, {}).setdefault(
                child_node_id, self._new_model()
            ).tell(score)
        else:
            self.node_models.setdefault("shared", {}).setdefault(
                child_node_id, self._new_model()
            ).tell(score)

    def select_next(
        self,
        all_rewards_store: dict[str, list[float]],
        rng: np.random.Generator,
    ) -> str | int:
        if self.model_selection_strategy == "stack":
            return self._select_next_stack(rng)
        if self.model_selection_strategy == "multiarm_bandit_thompson":
            return self._select_next_multiarm("thompson", all_rewards_store, rng)
        if self.model_selection_strategy == "multiarm_bandit_ucb":
            return self._select_next_multiarm("ucb", all_rewards_store, rng)
        raise ValueError(f"Unknown AB-MCTS strategy: {self.model_selection_strategy}")

    def _select_next_stack(self, rng: np.random.Generator) -> str | int:
        action = self._thompson(self.action_models, rng)
        gen_or_cont = self._thompson(self.gen_vs_cont_models[action], rng)
        node_models = self.node_models.get(action, {})
        if gen_or_cont == "GEN" or not node_models:
            return action
        return self._thompson(node_models, rng)

    def _select_next_multiarm(
        self,
        strategy: str,
        all_rewards_store: dict[str, list[float]],
        rng: np.random.Generator,
    ) -> str | int:
        choice = self._thompson(self.gen_vs_cont_models["shared"], rng)
        node_models = self.node_models.get("shared", {})
        if choice == "GEN" or not node_models:
            return self._select_best_action(strategy, all_rewards_store, rng)
        return self._thompson(node_models, rng)

    def _select_best_action(
        self,
        strategy: str,
        all_rewards_store: dict[str, list[float]],
        rng: np.random.Generator,
    ) -> str:
        if len(self.actions) == 1:
            return self.actions[0]
        if strategy == "thompson":
            return self._thompson(self.action_models, rng)
        total = max(1, sum(len(all_rewards_store.get(action, [])) for action in self.actions))
        scores: dict[str, float] = {}
        for action in self.actions:
            rewards = all_rewards_store.get(action, [])
            if not rewards:
                scores[action] = float("inf")
            else:
                scores[action] = float(np.mean(rewards)) + math.sqrt(2.0) * math.sqrt(
                    math.log(max(total, 2)) / len(rewards)
                )
        return max(scores, key=scores.__getitem__)

    def _thompson(
        self,
        models: dict[Any, RewardModel],
        rng: np.random.Generator,
    ) -> Any:
        best_key = None
        best_sample = None
        for key, model in models.items():
            sample = model.draw(rng)
            if best_sample is None or sample > best_sample:
                best_key = key
                best_sample = sample
        if best_key is None:
            raise RuntimeError("Cannot sample from an empty AB-MCTS posterior set")
        return best_key

    def to_dict(self) -> dict[str, Any]:
        return {
            "actions": self.actions,
            "dist_type": self.dist_type,
            "model_selection_strategy": self.model_selection_strategy,
            "prior_mean": self.prior_mean,
            "prior_std": self.prior_std,
            "prior_strength": self.prior_strength,
            "beta_a": self.beta_a,
            "beta_b": self.beta_b,
            "action_models": {
                action: model.to_dict() for action, model in self.action_models.items()
            },
            "gen_vs_cont_models": {
                key: {name: model.to_dict() for name, model in value.items()}
                for key, value in self.gen_vs_cont_models.items()
            },
            "node_models": {
                key: {str(node_id): model.to_dict() for node_id, model in value.items()}
                for key, value in self.node_models.items()
            },
            "child_node_to_action": {
                str(node_id): action
                for node_id, action in self.child_node_to_action.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BanditNodeState":
        out = cls(
            actions=[str(x) for x in data.get("actions", ["default"])],
            dist_type=str(data.get("dist_type", "gaussian")),
            model_selection_strategy=str(
                data.get("model_selection_strategy", "multiarm_bandit_thompson")
            ),
            prior_mean=_as_float(data.get("prior_mean"), 0.0),
            prior_std=max(_as_float(data.get("prior_std"), 1.0), 1e-6),
            prior_strength=max(_as_float(data.get("prior_strength"), 1.0), 1e-6),
            beta_a=max(_as_float(data.get("beta_a"), 0.5), 1e-6),
            beta_b=max(_as_float(data.get("beta_b"), 0.5), 1e-6),
        )
        out.action_models = {
            str(action): RewardModel.from_dict(model)
            for action, model in data.get("action_models", {}).items()
        }
        out.gen_vs_cont_models = {
            str(key): {
                str(name): RewardModel.from_dict(model)
                for name, model in value.items()
            }
            for key, value in data.get("gen_vs_cont_models", {}).items()
        }
        out.node_models = {
            str(key): {
                int(node_id): RewardModel.from_dict(model)
                for node_id, model in value.items()
            }
            for key, value in data.get("node_models", {}).items()
        }
        out.child_node_to_action = {
            int(node_id): str(action)
            for node_id, action in data.get("child_node_to_action", {}).items()
        }
        return out


@dataclass
class ABMCTSNode:
    node_id: int
    state: State
    score: float
    parent_id: int | None
    children: list[int] = field(default_factory=list)
    action: str | None = None
    trial_id: str | None = None
    depth: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "state": self.state.to_dict(),
            "score": self.score,
            "parent_id": self.parent_id,
            "children": list(self.children),
            "action": self.action,
            "trial_id": self.trial_id,
            "depth": self.depth,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, state_type: type) -> "ABMCTSNode":
        return cls(
            node_id=int(data["node_id"]),
            state=state_from_dict(data["state"], state_type=state_type),
            score=_as_float(data.get("score"), 0.0),
            parent_id=(
                int(data["parent_id"]) if data.get("parent_id") is not None else None
            ),
            children=[int(x) for x in data.get("children", [])],
            action=data.get("action"),
            trial_id=data.get("trial_id"),
            depth=int(data.get("depth", 0) or 0),
        )


@dataclass
class ABMCTSTrial:
    trial_id: str
    node_to_expand: int
    action: str
    parent_state_id: str
    score: float | None = None
    trial_status: str = "RUNNING"
    created_at: str = field(default_factory=_utc_now)
    completed_at: str | None = None

    def finish(self, score: float, status: str = "COMPLETE") -> "ABMCTSTrial":
        return ABMCTSTrial(
            trial_id=self.trial_id,
            node_to_expand=self.node_to_expand,
            action=self.action,
            parent_state_id=self.parent_state_id,
            score=score,
            trial_status=status,
            created_at=self.created_at,
            completed_at=_utc_now(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "node_to_expand": self.node_to_expand,
            "action": self.action,
            "parent_state_id": self.parent_state_id,
            "score": self.score,
            "trial_status": self.trial_status,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ABMCTSTrial":
        return cls(
            trial_id=str(data["trial_id"]),
            node_to_expand=int(data["node_to_expand"]),
            action=str(data["action"]),
            parent_state_id=str(data.get("parent_state_id", "")),
            score=(
                _as_float(data["score"])
                if data.get("score") is not None
                else None
            ),
            trial_status=str(data.get("trial_status", "RUNNING")),
            created_at=str(data.get("created_at") or _utc_now()),
            completed_at=data.get("completed_at"),
        )


class ABMCTSSampler(StateSampler):
    """
    AB-MCTS-A tree sampler.

    `sample_states()` returns clones of tree node states and annotates each clone
    with the trial/action/node metadata needed by `update_states()`.
    """

    def __init__(
        self,
        file_path: str,
        env_type: type,
        problem_type: str = "",
        *,
        batch_size: int = 1,
        resume_step: int | None = None,
        actions: list[str] | tuple[str, ...] | str | None = None,
        dist_type: DistributionType = "gaussian",
        model_selection_strategy: ModelSelectionStrategy = "multiarm_bandit_thompson",
        invalid_score: float = 0.0,
        prior_mean: float = 0.0,
        prior_std: float = 1.0,
        prior_strength: float = 1.0,
        beta_a: float = 0.5,
        beta_b: float = 0.5,
        seed: int | None = None,
    ):
        self.file_path = file_path
        self.env_type = env_type
        self.problem_type = problem_type
        self.batch_size = int(batch_size)
        self.actions = self._normalize_actions(actions)
        self.dist_type = dist_type
        self.model_selection_strategy = model_selection_strategy
        self.invalid_score = float(invalid_score)
        self.prior_mean = float(prior_mean)
        self.prior_std = max(float(prior_std), 1e-6)
        self.prior_strength = max(float(prior_strength), 1e-6)
        self.beta_a = max(float(beta_a), 1e-6)
        self.beta_b = max(float(beta_b), 1e-6)
        self._rng = np.random.default_rng(seed)
        self._current_step = resume_step if resume_step is not None else 0

        self._root_node_id = 0
        self._next_node_id = 1
        self._nodes: dict[int, ABMCTSNode] = {}
        self._prob_states: dict[int, BanditNodeState] = {}
        self._all_rewards_store: dict[str, list[float]] = {
            action: [] for action in self.actions
        }
        self._running_trials: dict[str, ABMCTSTrial] = {}
        self._finished_trials: dict[str, ABMCTSTrial] = {}
        self._last_sampled_states: list[State] = []
        self._last_sampled_records: list[dict[str, Any]] = []
        self._last_reflected: list[dict[str, Any]] = []

        if resume_step is not None:
            self._load(resume_step)
        if not self._nodes:
            self._initialize_root()
            self._save(self._current_step)

    @classmethod
    def from_config(cls, cfg: Any, *, task: Any, start_batch: int) -> "ABMCTSSampler":
        if not cfg.log_path:
            raise ValueError("log_path is required when using AB-MCTS sampler")
        variant = str(getattr(cfg, "abmcts_variant", "a") or "a").lower()
        if variant != "a":
            raise NotImplementedError(
                "The local AB-MCTS sampler currently implements AB-MCTS-A only. "
                "Set abmcts_variant='a'."
            )
        return cls(
            file_path=os.path.join(cfg.log_path, "abmcts_sampler.json"),
            env_type=task.env_type,
            problem_type=task.problem_type,
            batch_size=cfg.groups_per_batch,
            resume_step=start_batch if start_batch > 0 else None,
            actions=getattr(cfg, "abmcts_actions", ("default",)),
            dist_type=getattr(cfg, "abmcts_dist_type", "gaussian"),
            model_selection_strategy=getattr(
                cfg,
                "abmcts_model_selection_strategy",
                "multiarm_bandit_thompson",
            ),
            invalid_score=getattr(cfg, "abmcts_invalid_score", 0.0),
            prior_mean=getattr(cfg, "abmcts_prior_mean", 0.0),
            prior_std=getattr(cfg, "abmcts_prior_std", 1.0),
            prior_strength=getattr(cfg, "abmcts_prior_strength", 1.0),
            beta_a=getattr(cfg, "abmcts_beta_a", 0.5),
            beta_b=getattr(cfg, "abmcts_beta_b", 0.5),
            seed=getattr(cfg, "abmcts_seed", None),
        )

    def _normalize_actions(
        self,
        actions: list[str] | tuple[str, ...] | str | None,
    ) -> list[str]:
        if actions is None:
            return ["default"]
        if isinstance(actions, str):
            items = [part.strip() for part in actions.split(",")]
        else:
            items = [str(part).strip() for part in actions]
        out = [item for item in items if item]
        return out or ["default"]

    def _new_prob_state(self) -> BanditNodeState:
        return BanditNodeState(
            actions=self.actions,
            dist_type=self.dist_type,
            model_selection_strategy=self.model_selection_strategy,
            prior_mean=self.prior_mean,
            prior_std=self.prior_std,
            prior_strength=self.prior_strength,
            beta_a=self.beta_a,
            beta_b=self.beta_b,
        )

    def _initialize_root(self) -> None:
        root_state = create_initial_state(self.env_type, self.problem_type)
        root_score = self._score_from_state(root_state)
        self._nodes[self._root_node_id] = ABMCTSNode(
            node_id=self._root_node_id,
            state=root_state,
            score=root_score,
            parent_id=None,
            depth=0,
        )
        self._next_node_id = self._root_node_id + 1
        self._prob_states[self._root_node_id] = self._new_prob_state()

    def _load(self, step: int) -> None:
        file_path = _sampler_file_for_step(self.file_path, step)
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Cannot resume from step {step}: AB-MCTS sampler file not found: {file_path}"
            )
        with _file_lock(f"{file_path}.lock"):
            store = _read_json_or_default(file_path, default=None)
        if store is None:
            raise ValueError(f"Failed to load AB-MCTS sampler state from {file_path}")
        state_cls = self.env_type.state_type
        self._root_node_id = int(store.get("root_node_id", 0))
        self._next_node_id = int(store.get("next_node_id", 1))
        self.actions = [str(x) for x in store.get("actions", self.actions)] or ["default"]
        self.dist_type = str(store.get("dist_type", self.dist_type))
        self.model_selection_strategy = str(
            store.get("model_selection_strategy", self.model_selection_strategy)
        )
        self.invalid_score = _as_float(store.get("invalid_score"), self.invalid_score)
        self._nodes = {
            int(item["node_id"]): ABMCTSNode.from_dict(item, state_type=state_cls)
            for item in store.get("nodes", [])
        }
        self._prob_states = {
            int(node_id): BanditNodeState.from_dict(data)
            for node_id, data in store.get("prob_states", {}).items()
        }
        self._all_rewards_store = {
            str(action): [_as_float(x) for x in rewards]
            for action, rewards in store.get("all_rewards_store", {}).items()
        }
        for action in self.actions:
            self._all_rewards_store.setdefault(action, [])
        self._running_trials = {
            str(trial_id): ABMCTSTrial.from_dict(data)
            for trial_id, data in store.get("running_trials", {}).items()
        }
        self._finished_trials = {
            str(trial_id): ABMCTSTrial.from_dict(data)
            for trial_id, data in store.get("finished_trials", {}).items()
        }
        rng_state = store.get("rng_state")
        if rng_state is not None:
            try:
                self._rng.bit_generator.state = rng_state
            except Exception as exc:
                logger.warning("Failed to restore AB-MCTS RNG state: %s", exc)

    def _save(self, step: int) -> None:
        save_path = _sampler_file_for_step(self.file_path, step)
        store = {
            "step": step,
            "root_node_id": self._root_node_id,
            "next_node_id": self._next_node_id,
            "actions": list(self.actions),
            "dist_type": self.dist_type,
            "model_selection_strategy": self.model_selection_strategy,
            "invalid_score": self.invalid_score,
            "prior_mean": self.prior_mean,
            "prior_std": self.prior_std,
            "prior_strength": self.prior_strength,
            "beta_a": self.beta_a,
            "beta_b": self.beta_b,
            "nodes": [
                self._nodes[node_id].to_dict()
                for node_id in sorted(self._nodes)
            ],
            "prob_states": {
                str(node_id): prob_state.to_dict()
                for node_id, prob_state in sorted(self._prob_states.items())
            },
            "all_rewards_store": self._all_rewards_store,
            "running_trials": {
                trial_id: trial.to_dict()
                for trial_id, trial in self._running_trials.items()
            },
            "finished_trials": {
                trial_id: trial.to_dict()
                for trial_id, trial in self._finished_trials.items()
            },
            "rng_state": self._rng.bit_generator.state,
        }
        with _file_lock(f"{save_path}.lock"):
            _atomic_write_json(save_path, store)

    def _score_from_state(self, state: State | None) -> float:
        if state is None:
            return self.invalid_score
        score = _as_float(getattr(state, "value", None), self.invalid_score)
        if self.dist_type == "beta":
            return max(0.0, min(1.0, score))
        return score

    def _get_prob_state(self, node_id: int) -> BanditNodeState:
        if node_id not in self._prob_states:
            self._prob_states[node_id] = self._new_prob_state()
        return self._prob_states[node_id]

    def _state_key(self, state: State) -> tuple | str | None:
        construction = getattr(state, "construction", None)
        if construction:
            return self._freeze_key(construction)
        code = getattr(state, "code", None)
        if code:
            return str(code)
        return None

    def _freeze_key(self, value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return self._freeze_key(value.tolist())
        if isinstance(value, (list, tuple)):
            return tuple(self._freeze_key(item) for item in value)
        if isinstance(value, dict):
            return tuple(
                sorted((key, self._freeze_key(item)) for key, item in value.items())
            )
        return value

    def has_state(self, state: State) -> bool:
        key = self._state_key(state)
        if key is None:
            return False
        existing = {self._state_key(node.state) for node in self._nodes.values()}
        existing.discard(None)
        return key in existing

    def _select_expansion(self) -> tuple[ABMCTSNode, str]:
        node = self._nodes[self._root_node_id]
        while node.children:
            selection = self._get_prob_state(node.node_id).select_next(
                self._all_rewards_store,
                self._rng,
            )
            if isinstance(selection, str):
                return node, selection
            if selection not in node.children:
                logger.warning(
                    "AB-MCTS posterior selected child %s not present under node %s; "
                    "falling back to first child",
                    selection,
                    node.node_id,
                )
                selection = node.children[0]
            node = self._nodes[int(selection)]

        selection = self._get_prob_state(node.node_id).select_next(
            self._all_rewards_store,
            self._rng,
        )
        if isinstance(selection, str):
            return node, selection
        return node, self.actions[0]

    def _create_trial(self, node: ABMCTSNode, action: str) -> ABMCTSTrial:
        trial = ABMCTSTrial(
            trial_id=str(uuid.uuid4()),
            node_to_expand=node.node_id,
            action=action,
            parent_state_id=node.state.id,
        )
        self._running_trials[trial.trial_id] = trial
        return trial

    def _sample_clone(self, node: ABMCTSNode, trial: ABMCTSTrial) -> State:
        metadata = {
            "trial_id": trial.trial_id,
            "action": trial.action,
            "node_id": node.node_id,
            "node_to_expand": trial.node_to_expand,
            "parent_state_id": node.state.id,
            "tree_depth": node.depth,
        }
        return node.state.clone(metadata_updates={"abmcts": metadata})

    def sample_states(self, num_states: int) -> list[State]:
        picked: list[State] = []
        records: list[dict[str, Any]] = []
        for _ in range(max(0, int(num_states))):
            node, action = self._select_expansion()
            trial = self._create_trial(node, action)
            picked_state = self._sample_clone(node, trial)
            picked.append(picked_state)
            records.append(
                {
                    "trial_id": trial.trial_id,
                    "node_id": node.node_id,
                    "action": action,
                    "parent_state_id": node.state.id,
                    "parent_value": node.state.value,
                    "tree_depth": node.depth,
                    "children": len(node.children),
                }
            )
        self._last_sampled_states = picked
        self._last_sampled_records = records
        return picked

    def _trial_from_parent(self, parent: State) -> ABMCTSTrial | None:
        metadata = getattr(parent, "metadata", {}) or {}
        abmcts_meta = metadata.get("abmcts", {}) if isinstance(metadata, dict) else {}
        trial_id = abmcts_meta.get("trial_id")
        if not trial_id:
            return None
        return self._running_trials.get(str(trial_id))

    def _finish_trial(
        self,
        trial: ABMCTSTrial,
        *,
        score: float,
        status: str = "COMPLETE",
    ) -> ABMCTSTrial | None:
        if trial.trial_id not in self._running_trials:
            return None
        self._running_trials.pop(trial.trial_id)
        finished = trial.finish(score=score, status=status)
        self._finished_trials[finished.trial_id] = finished
        return finished

    def _backpropagate(
        self,
        *,
        parent_node_id: int,
        child_node_id: int | None,
        action: str,
        score: float,
    ) -> None:
        self._all_rewards_store.setdefault(action, []).append(score)
        parent_prob = self._get_prob_state(parent_node_id)
        parent_prob.update_action_reward(action, score)
        if child_node_id is not None:
            parent_prob.register_new_child_node(action, child_node_id, score)

        current_id = parent_node_id
        while True:
            current = self._nodes[current_id]
            if current.parent_id is None:
                break
            ancestor_prob = self._get_prob_state(current.parent_id)
            ancestor_prob.update_node_reward(current_id, score)
            current_id = current.parent_id

    def _annotate_child(
        self,
        child: State,
        *,
        node_id: int,
        parent_node_id: int,
        trial: ABMCTSTrial,
    ) -> None:
        metadata = dict(getattr(child, "metadata", {}) or {})
        metadata["abmcts"] = {
            "node_id": node_id,
            "parent_node_id": parent_node_id,
            "action": trial.action,
            "trial_id": trial.trial_id,
            "tree_depth": self._nodes[parent_node_id].depth + 1,
        }
        child.metadata = to_json_serializable(metadata)

    def _reflect_success(self, child: State, parent: State) -> dict[str, Any]:
        trial = self._trial_from_parent(parent)
        if trial is None:
            return {"status": "missing_trial"}

        score = self._score_from_state(child)
        finished = self._finish_trial(trial, score=score)
        if finished is None:
            return {"status": "stale_trial", "trial_id": trial.trial_id}

        parent_node = self._nodes.get(finished.node_to_expand)
        if parent_node is None:
            return {"status": "missing_parent_node", "trial_id": finished.trial_id}

        if self.has_state(child):
            self._backpropagate(
                parent_node_id=parent_node.node_id,
                child_node_id=None,
                action=finished.action,
                score=score,
            )
            return {
                "status": "duplicate",
                "trial_id": finished.trial_id,
                "parent_node_id": parent_node.node_id,
                "action": finished.action,
                "score": score,
            }

        self._set_parent_info(child, parent_node.state)
        node_id = self._next_node_id
        self._next_node_id += 1
        self._annotate_child(
            child,
            node_id=node_id,
            parent_node_id=parent_node.node_id,
            trial=finished,
        )
        node = ABMCTSNode(
            node_id=node_id,
            state=child,
            score=score,
            parent_id=parent_node.node_id,
            children=[],
            action=finished.action,
            trial_id=finished.trial_id,
            depth=parent_node.depth + 1,
        )
        self._nodes[node_id] = node
        parent_node.children.append(node_id)
        self._get_prob_state(node_id)
        self._backpropagate(
            parent_node_id=parent_node.node_id,
            child_node_id=node_id,
            action=finished.action,
            score=score,
        )
        return {
            "status": "added",
            "trial_id": finished.trial_id,
            "node_id": node_id,
            "parent_node_id": parent_node.node_id,
            "action": finished.action,
            "score": score,
        }

    def update_states(
        self,
        states: list[State],
        parent_states: list[State],
        save: bool = True,
        step: int | None = None,
    ) -> None:
        if not states:
            return
        if len(states) != len(parent_states):
            raise ValueError("states and parent_states must have the same length")

        reflected = []
        for child, parent in zip(states, parent_states):
            if child.value is None:
                reflected.append({"status": "missing_value"})
                continue
            reflected.append(self._reflect_success(child, parent))
        self._last_reflected = reflected
        if save:
            self.flush(step=step)

    def record_failed_rollout(self, parent: State) -> None:
        trial = self._trial_from_parent(parent)
        if trial is None:
            return
        finished = self._finish_trial(trial, score=self.invalid_score, status="FAILED")
        if finished is None:
            return
        if finished.node_to_expand not in self._nodes:
            return
        self._backpropagate(
            parent_node_id=finished.node_to_expand,
            child_node_id=None,
            action=finished.action,
            score=self.invalid_score,
        )
        self._last_reflected = [
            {
                "status": "failed",
                "trial_id": finished.trial_id,
                "parent_node_id": finished.node_to_expand,
                "action": finished.action,
                "score": self.invalid_score,
            }
        ]

    def flush(self, step: int | None = None) -> None:
        if step is not None:
            self._current_step = step
        self._save(self._current_step)

    def reload_from_step(self, step: int) -> None:
        self._nodes = {}
        self._prob_states = {}
        self._running_trials = {}
        self._finished_trials = {}
        self._current_step = step
        self._load(step)

    def get_initial_states(self) -> list[State]:
        return [self._nodes[self._root_node_id].state]

    def get_sample_stats(self) -> dict[str, Any]:
        values = [node.state.value for node in self._nodes.values()]
        depths = [node.depth for node in self._nodes.values()]
        sampled_values = [state.value for state in self._last_sampled_states]

        def _summary(prefix: str, xs: list[Any]) -> dict[str, float]:
            arr = np.array([_as_float(x, float("nan")) for x in xs], dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                return {}
            return {
                f"{prefix}/mean": float(np.mean(arr)),
                f"{prefix}/min": float(np.min(arr)),
                f"{prefix}/max": float(np.max(arr)),
            }

        stats: dict[str, Any] = {
            "abmcts/tree_size": len(self._nodes),
            "abmcts/running_trials": len(self._running_trials),
            "abmcts/finished_trials": len(self._finished_trials),
            "abmcts/actions": len(self.actions),
            "abmcts/last_sampled": len(self._last_sampled_states),
            "abmcts/max_depth": max(depths) if depths else 0,
        }
        stats.update(_summary("abmcts/node_value", values))
        stats.update(_summary("abmcts/sampled_value", sampled_values))
        return stats

    def get_sample_table(self) -> tuple[list[str], list[tuple[Any, ...]]]:
        columns = [
            "trial_id",
            "node_id",
            "action",
            "tree_depth",
            "parent_state_id",
            "parent_value",
            "children",
        ]
        rows = [
            (
                record.get("trial_id"),
                record.get("node_id"),
                record.get("action"),
                record.get("tree_depth"),
                record.get("parent_state_id"),
                record.get("parent_value"),
                record.get("children"),
            )
            for record in self._last_sampled_records
        ]
        return columns, rows


__all__ = [
    "ABMCTSSampler",
    "BanditNodeState",
    "RewardModel",
]
