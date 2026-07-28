"""PushT-specific conversion utilities."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional

from .schema import SceneGraph, graph_from_pusht_state, render_graph_text


@dataclass(frozen=True)
class PushTTransition:
    observation: str
    action: str
    next_observation: str
    reward: float
    done: bool
    metadata: Mapping[str, Any]

    def to_patchworld(self) -> Dict[str, Any]:
        return {
            "observation": self.observation,
            "action": self.action,
            "next_observation": self.next_observation,
            "reward": float(self.reward),
            "done": bool(self.done),
            "metadata": dict(self.metadata),
        }


def discretize_action(action: Iterable[float], *, hold_threshold: float = 0.05) -> str:
    """Convert a continuous PushT action vector into a stable symbolic action string."""

    values = list(float(v) for v in action)
    dx = values[0] if values else 0.0
    dy = values[1] if len(values) > 1 else 0.0
    speed = math.hypot(dx, dy)
    if speed < hold_threshold:
        label = "hold"
    elif abs(dx) >= abs(dy):
        label = "move_right" if dx > 0 else "move_left"
    else:
        label = "move_up" if dy > 0 else "move_down"
    return f"{label}[dx={dx:.4f},dy={dy:.4f},speed={speed:.4f}]"


def aggregate_action(action_window: Any) -> List[float]:
    """Average a raw action window into one low-level delta vector."""

    try:
        import numpy as np
    except ModuleNotFoundError:
        values = action_window
        if isinstance(values, list) and values and isinstance(values[0], list):
            cols = list(zip(*values))
            return [sum(col) / max(1, len(col)) for col in cols]
        if isinstance(values, list):
            return [float(v) for v in values]
        return [float(values)]

    arr = np.asarray(action_window, dtype=float)
    if arr.ndim == 0:
        return [float(arr)]
    if arr.ndim == 1:
        return [float(v) for v in arr.tolist()]
    return [float(v) for v in arr.mean(axis=0).tolist()]


def graph_from_vlm_payload(
    payload: Mapping[str, Any],
    *,
    step: int,
    metadata: Optional[Mapping[str, Any]] = None,
    source: str = "vlm",
) -> SceneGraph:
    """Normalize a cached visual JSON payload into the canonical scene graph."""

    state = payload.get("state")
    if state is not None:
        return graph_from_pusht_state(state, source=source, step=step, metadata=metadata)

    objects = payload.get("objects")
    if not isinstance(objects, list):
        raise ValueError("VLM payload must contain either `state` or an `objects` list")

    # When the VLM emits object dictionaries, reuse the oracle constructor by
    # mapping known IDs back into the PushT state vector.
    by_name: Dict[str, Mapping[str, Any]] = {}
    for obj in objects:
        if not isinstance(obj, Mapping):
            continue
        raw_id = str(obj.get("id") or obj.get("name") or obj.get("category") or "").lower()
        if "agent" in raw_id or "pusher" in raw_id:
            by_name["agent"] = obj
        elif "block" in raw_id or "t" in raw_id:
            by_name["block"] = obj

    agent = by_name.get("agent")
    block = by_name.get("block")
    if not agent or not block:
        raise ValueError("VLM payload must identify both agent/pusher and T block")

    def pos(obj: Mapping[str, Any], key: str) -> float:
        position = obj.get("position", obj)
        if not isinstance(position, Mapping):
            raise ValueError(f"Invalid position for object {obj!r}")
        return float(position[key])

    state = [
        pos(agent, "x"),
        pos(agent, "y"),
        pos(block, "x"),
        pos(block, "y"),
        float(block.get("angle", 0.0)),
        float(agent.get("vx", 0.0)),
        float(agent.get("vy", 0.0)),
    ]
    return graph_from_pusht_state(state, source=source, step=step, metadata=metadata)


def make_transition(
    graph: SceneGraph,
    action_values: Iterable[float],
    next_graph: SceneGraph,
    *,
    reward: float = 0.0,
    done: bool = False,
    metadata: Optional[Mapping[str, Any]] = None,
    action_text: Optional[str] = None,
) -> PushTTransition:
    action_values_list = [float(v) for v in action_values]
    action = str(action_text) if action_text else discretize_action(action_values_list)
    rich_metadata = dict(metadata or {})
    rich_metadata.setdefault("action_values", action_values_list)
    rich_metadata["observation_graph"] = graph.to_json()
    rich_metadata["next_observation_graph"] = next_graph.to_json()
    return PushTTransition(
        observation=render_graph_text(graph),
        action=action,
        next_observation=render_graph_text(next_graph),
        reward=reward,
        done=done,
        metadata=rich_metadata,
    )


def make_patchworld_record(
    transitions: List[PushTTransition],
    *,
    split: str,
    episode_index: int,
    source: str,
    dataset: str,
) -> Dict[str, Any]:
    total_reward = sum(t.reward for t in transitions)
    return {
        "metadata": {
            "env": "pusht",
            "item_id": f"pusht-{source}-episode-{episode_index}",
            "task_idx": int(episode_index),
            "rollout_index": 0,
            "split": split,
            "success": bool(transitions and transitions[-1].done),
            "total_reward": float(total_reward),
            "source": source,
            "dataset": dataset,
        },
        "transitions": [transition.to_patchworld() for transition in transitions],
    }


def record_to_jsonl(record: Mapping[str, Any]) -> str:
    return json.dumps(record, sort_keys=True, separators=(",", ":"))
