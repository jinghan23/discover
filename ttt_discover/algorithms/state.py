"""State objects shared by algorithm loops, samplers, tasks, and eval runners."""

from __future__ import annotations

from typing import Any
import uuid

import numpy as np


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


class State:
    id: str
    timestep: int
    value: float
    code: str
    construction: list[Any]
    parent_values: list[float]
    parents: list[dict]
    observation: str
    metadata: dict[str, Any]

    def __init__(
        self,
        timestep: int,
        construction: list[Any],
        code: str,
        value: float | None = None,
        parent_values: list[float] | None = None,
        parents: list[dict] | None = None,
        id: str | None = None,
        observation: str = "",
        metadata: dict[str, Any] | None = None,
    ):
        self.id = id if id is not None else str(uuid.uuid4())
        self.timestep = timestep
        self.value = value
        self.construction = to_json_serializable(construction)
        self.code = code
        self.parent_values = parent_values if parent_values is not None else []
        self.parents = parents if parents is not None else []
        self.observation = observation
        self.metadata = to_json_serializable(metadata or {})

    def to_dict(self) -> dict:
        return {
            "type": self.__class__.__name__,
            "id": self.id,
            "timestep": self.timestep,
            "value": self.value,
            "parent_values": self.parent_values,
            "parents": self.parents,
            "observation": self.observation,
            "construction": to_json_serializable(self.construction),
            "code": self.code,
            "metadata": to_json_serializable(self.metadata),
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
            metadata=d.get("metadata", {}),
        )

    def clone(self, *, metadata_updates: dict[str, Any] | None = None) -> "State":
        data = self.to_dict()
        if metadata_updates:
            metadata = dict(data.get("metadata") or {})
            metadata.update(metadata_updates)
            data["metadata"] = metadata
        return type(self).from_dict(data)

    def to_prompt(
        self,
        target,
        metric_name: str = "value",
        maximize: bool = True,
        language: str = "",
    ):
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
        return set(cls.__subclasses__()) | {
            s for c in cls.__subclasses__() for s in _all_subclasses(c)
        }

    for cls in [State] + list(_all_subclasses(State)):
        if cls.__name__ == name:
            return cls
    raise ValueError(f"Unknown state type: {name}")


def state_from_dict(d: dict | None, state_type: type | None = None) -> Any | None:
    if d is None:
        return None
    cls = state_type if state_type is not None else _state_class_by_name(d.get("type", "State"))
    return cls.from_dict(d)


__all__ = ["State", "state_from_dict", "to_json_serializable"]
