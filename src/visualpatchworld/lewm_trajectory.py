"""PatchWorld JSONL record helpers for multi-environment exports."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Mapping, Optional

from .lewm_graph import render_env_graph_text
from .schema import SceneGraph


def make_env_transition(
    graph: SceneGraph,
    action_values: Iterable[float],
    next_graph: SceneGraph,
    *,
    env_name: str,
    reward: float = 0.0,
    done: bool = False,
    metadata: Optional[Mapping[str, Any]] = None,
    action_text: Optional[str] = None,
) -> Dict[str, Any]:
    from .lewm_env_specs import get_env_spec
    from .lewm_graph import action_string_from_vector

    spec = get_env_spec(env_name)
    action_values_list = [float(v) for v in action_values]
    action = str(action_text) if action_text else action_string_from_vector(action_values_list, spec)
    rich_metadata = dict(metadata or {})
    rich_metadata.setdefault("action_values", action_values_list)
    rich_metadata["observation_graph"] = graph.to_json()
    rich_metadata["next_observation_graph"] = next_graph.to_json()
    return {
        "observation": render_env_graph_text(graph),
        "action": action,
        "next_observation": render_env_graph_text(next_graph),
        "reward": float(reward),
        "done": bool(done),
        "metadata": rich_metadata,
    }


def make_env_record(
    transitions: List[Mapping[str, Any]],
    *,
    env_name: str,
    split: str,
    episode_index: int,
    source: str,
    dataset: str,
) -> Dict[str, Any]:
    total_reward = sum(float(t.get("reward", 0.0)) for t in transitions)
    return {
        "metadata": {
            "env": env_name,
            "item_id": f"{env_name}-{source}-episode-{episode_index}",
            "task_idx": int(episode_index),
            "rollout_index": 0,
            "split": split,
            "source": source,
            "dataset": dataset,
            "total_reward": total_reward,
        },
        "transitions": transitions,
    }


def record_to_jsonl(record: Mapping[str, Any]) -> str:
    return json.dumps(record, sort_keys=True)
