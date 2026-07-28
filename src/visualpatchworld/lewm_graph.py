"""Scene-graph construction for LeWM four-environment suite."""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .lewm_env_specs import LeWMEnvSpec, get_env_spec
from .schema import Relation, SceneGraph, SceneObject


def _objects_from_mapping(objects: Mapping[str, Mapping[str, float]]) -> List[SceneObject]:
    out: List[SceneObject] = []
    for object_id, fields in objects.items():
        out.append(
            SceneObject(
                object_id=object_id,
                category=str(fields.get("category", object_id)),
                x=float(fields.get("x", 0.0)),
                y=float(fields.get("y", 0.0)),
                # Cube (and other 3D) objects carry z; previously this was dropped
                # from attributes and never reached rendered scene-graph text.
                z=float(fields["z"]) if "z" in fields else None,
                angle=float(fields["angle"]) if "angle" in fields else None,
                vx=float(fields["vx"]) if "vx" in fields else None,
                vy=float(fields["vy"]) if "vy" in fields else None,
                attributes={
                    k: v
                    for k, v in fields.items()
                    if k not in {"category", "x", "y", "z", "angle", "vx", "vy"}
                },
            )
        )
    return out


def graph_from_row(
    env_name: str,
    h5_row: Mapping[str, Any],
    *,
    source: str,
    step: int,
    metadata: Optional[Mapping[str, Any]] = None,
) -> SceneGraph:
    spec = get_env_spec(env_name)
    meta = dict(metadata or {})

    if spec.name == "pusht":
        from .schema import graph_from_pusht_state

        return graph_from_pusht_state(h5_row["state"], source=source, step=step, metadata=meta)

    if spec.name == "tworoom":
        obs = np.asarray(h5_row["observation"], dtype=np.float64)
        agent_x, agent_y = float(obs[0]), float(obs[1])
        target_x, target_y = float(obs[2]), float(obs[3])
        objects = _objects_from_mapping(
            {
                "agent": {"category": "navigator", "x": agent_x, "y": agent_y},
                "target": {"category": "goal", "x": target_x, "y": target_y},
            }
        )
        relations = [
            Relation("agent", "distance_to", "target", math.hypot(target_x - agent_x, target_y - agent_y)),
            Relation("agent", "near", "target", math.hypot(target_x - agent_x, target_y - agent_y) <= 16.0),
        ]
        return SceneGraph(
            env=spec.name,
            source=source,
            step=step,
            objects=objects,
            relations=relations,
            goal="navigate agent to target",
            metadata={**meta, "door_coords": [float(v) for v in obs[4:10].tolist()]},
        )

    if spec.name == "cube":
        effector = np.asarray(h5_row["effector_pos"], dtype=np.float64)
        block = np.asarray(h5_row["block_pos"], dtype=np.float64)
        goal = np.asarray(h5_row["goal_block_pos"], dtype=np.float64)
        objects = _objects_from_mapping(
            {
                "effector": {
                    "category": "gripper",
                    "x": float(effector[0]),
                    "y": float(effector[1]),
                    "z": float(effector[2]),
                },
                "block": {
                    "category": "cube",
                    "x": float(block[0]),
                    "y": float(block[1]),
                    "z": float(block[2]),
                },
                "goal": {
                    "category": "goal_pose",
                    "x": float(goal[0]),
                    "y": float(goal[1]),
                    "z": float(goal[2]),
                },
            }
        )
        dist = float(np.linalg.norm(block - goal))
        relations = [
            Relation("effector", "distance_to", "block", float(np.linalg.norm(effector - block))),
            Relation("block", "distance_to", "goal", dist),
            Relation("block", "near", "goal", dist <= 0.05),
        ]
        return SceneGraph(
            env=spec.name,
            source=source,
            step=step,
            objects=objects,
            relations=relations,
            goal="move block to goal pose",
            metadata=meta,
        )

    if spec.name == "reacher":
        qpos = np.asarray(h5_row["qpos"], dtype=np.float64)
        finger = np.asarray(h5_row["finger_pos"], dtype=np.float64)
        target = np.asarray(h5_row["target_pos"], dtype=np.float64)
        objects = _objects_from_mapping(
            {
                "arm": {"category": "joint_state", "x": float(qpos[0]), "y": float(qpos[1])},
                "finger": {"category": "end_effector", "x": float(finger[0]), "y": float(finger[1])},
                "target": {"category": "goal", "x": float(target[0]), "y": float(target[1])},
            }
        )
        dist = float(np.linalg.norm(finger - target))
        relations = [
            Relation("finger", "distance_to", "target", dist),
            Relation("finger", "near", "target", dist <= 0.05),
        ]
        return SceneGraph(
            env=spec.name,
            source=source,
            step=step,
            objects=objects,
            relations=relations,
            goal="match finger to target",
            metadata=meta,
        )

    raise ValueError(f"No graph builder for env {env_name}")


def render_env_graph_text(graph: SceneGraph) -> str:
    data = graph.to_json()
    object_lines = []
    for obj in sorted(data["objects"], key=lambda item: item["id"]):
        pos = obj["position"]
        parts = [
            f"{obj['id']}:{obj['category']}",
            f"x={pos['x']:.3f}",
            f"y={pos['y']:.3f}",
        ]
        if "z" in pos:
            parts.append(f"z={float(pos['z']):.3f}")
        attrs = obj.get("attributes", {})
        if "z" in attrs and "z" not in pos:
            parts.append(f"z={float(attrs['z']):.3f}")
        if "angle" in obj:
            parts.append(f"angle={obj['angle']:.3f}")
        if "velocity" in obj:
            vel = obj["velocity"]
            parts.append(f"vx={vel['x']:.3f}")
            parts.append(f"vy={vel['y']:.3f}")
        parts.append(f"conf={obj.get('confidence', 1.0):.3f}")
        object_lines.append("- " + " ".join(parts))

    relation_lines = []
    for rel in sorted(data["relations"], key=lambda item: (item["subject"], item["predicate"], item["object"])):
        value = rel.get("value")
        if isinstance(value, float):
            value_text = f"{value:.3f}"
        elif isinstance(value, bool):
            value_text = str(value).lower()
        else:
            value_text = str(value)
        relation_lines.append(f"- {rel['subject']} {rel['predicate']} {rel['object']} = {value_text}")

    metadata = data.get("metadata", {})
    meta_text = json.dumps(metadata, sort_keys=True, separators=(",", ":")) if metadata else "{}"
    return "\n".join(
        [
            f"Environment: {graph.env}",
            f"Source: {graph.source}",
            f"Step: {graph.step}",
            f"Goal: {graph.goal}",
            "Objects:",
            *object_lines,
            "Relations:",
            *relation_lines,
            f"Metadata: {meta_text}",
        ]
    )


def graph_text_from_row(
    env_name: str,
    h5_row: Mapping[str, Any],
    *,
    source: str = "oracle",
    step: int = 0,
    metadata: Optional[Mapping[str, Any]] = None,
) -> str:
    graph = graph_from_row(env_name, h5_row, source=source, step=step, metadata=metadata)
    return render_env_graph_text(graph)


def row_dict_from_h5(h5: Any, idx: int, spec: LeWMEnvSpec) -> Dict[str, Any]:
    idx = int(idx)
    if spec.name == "pusht":
        return {"state": np.asarray(h5["state"][idx], dtype=np.float64)}
    if spec.name == "tworoom":
        return {"observation": np.asarray(h5["observation"][idx], dtype=np.float64)}
    if spec.name == "cube":
        return {
            "effector_pos": np.asarray(h5["proprio_effector_pos"][idx], dtype=np.float64),
            "block_pos": np.asarray(h5["privileged_block_0_pos"][idx], dtype=np.float64),
            "goal_block_pos": np.asarray(h5["privileged_target_block_pos"][idx], dtype=np.float64),
            "qpos": np.asarray(h5["qpos"][idx], dtype=np.float64),
            "qvel": np.asarray(h5["qvel"][idx], dtype=np.float64),
        }
    if spec.name == "reacher":
        return {
            "qpos": np.asarray(h5["qpos"][idx], dtype=np.float64),
            "qvel": np.asarray(h5["qvel"][idx], dtype=np.float64),
            "finger_pos": np.asarray(h5["finger_pos"][idx], dtype=np.float64),
            "target_pos": np.asarray(h5["target_pos"][idx], dtype=np.float64),
        }
    raise ValueError(spec.name)


def aggregate_action_window(h5: Any, start: int, stride: int) -> np.ndarray:
    block = np.asarray(h5["action"][start : start + stride], dtype=np.float64)
    if block.ndim == 1:
        return block
    return block.mean(axis=0)


def action_string_from_vector(action: Sequence[float], spec: LeWMEnvSpec) -> str:
    values = [float(v) for v in action]
    if spec.action_dim == 2:
        from .pusht import discretize_action

        return discretize_action(values)
    parts = ",".join(f"a{i}={values[i]:.4f}" for i in range(len(values)))
    speed = math.sqrt(sum(v * v for v in values))
    return f"act[{parts},speed={speed:.4f}]"


def parse_action_vector(action: str, spec: LeWMEnvSpec) -> np.ndarray:
    import re

    if spec.action_dim == 2:
        from .planning_eval import parse_action_deltas

        dx, dy = parse_action_deltas(action)
        return np.array([dx, dy], dtype=np.float64)
    out = np.zeros(spec.action_dim, dtype=np.float64)
    for match in re.finditer(r"a(\d+)=(-?\d+(?:\.\d+)?)", action):
        idx = int(match.group(1))
        if 0 <= idx < spec.action_dim:
            out[idx] = float(match.group(2))
    return out


def goal_row_from_start(h5: Any, spec: LeWMEnvSpec, episode_index: int, start_step: int, goal_offset: int) -> Dict[str, Any]:
    goal_step = int(start_step + goal_offset)
    offset = int(h5["ep_offset"][episode_index])
    goal_idx = offset + goal_step
    return row_dict_from_h5(h5, goal_idx, spec)


def goal_distance(
    env_name: str,
    goal_row: Mapping[str, Any],
    current_row: Mapping[str, Any],
    *,
    metric: str = "default",
) -> float:
    spec = get_env_spec(env_name)
    if spec.name == "pusht":
        from .planning_eval import pusht_goal_distance

        return pusht_goal_distance(goal_row["state"], current_row["state"])
    if spec.name == "tworoom":
        g = np.asarray(goal_row["observation"][:2], dtype=np.float64)
        c = np.asarray(current_row["observation"][:2], dtype=np.float64)
        return float(np.linalg.norm(g - c))
    if spec.name == "cube":
        g = np.asarray(goal_row["block_pos"], dtype=np.float64)
        c = np.asarray(current_row["block_pos"], dtype=np.float64)
        return float(np.linalg.norm(g - c))
    if spec.name == "reacher":
        if metric == "finger_match":
            g = np.asarray(goal_row["target_pos"], dtype=np.float64)
            c = np.asarray(current_row["finger_pos"], dtype=np.float64)
            return float(np.linalg.norm(g - c))
        g = np.asarray(goal_row["qpos"], dtype=np.float64)
        c = np.asarray(current_row["qpos"], dtype=np.float64)
        return float(np.linalg.norm(g - c))
    raise ValueError(env_name)


def is_success(
    env_name: str,
    goal_row: Mapping[str, Any],
    current_row: Mapping[str, Any],
    *,
    metric: str = "default",
) -> bool:
    spec = get_env_spec(env_name)
    if spec.name == "pusht":
        from .planning_eval import pusht_success

        return pusht_success(goal_row["state"], current_row["state"])
    if spec.name == "tworoom":
        return goal_distance(env_name, goal_row, current_row, metric=metric) < 16.0
    if spec.name == "cube":
        return goal_distance(env_name, goal_row, current_row, metric=metric) < 0.05
    if spec.name == "reacher":
        if metric == "finger_match":
            return goal_distance(env_name, goal_row, current_row, metric=metric) < 0.05
        g = np.asarray(goal_row["qpos"], dtype=np.float64)
        c = np.asarray(current_row["qpos"], dtype=np.float64)
        return bool(np.all(np.abs(g - c) < 0.05))
    raise ValueError(env_name)


def reset_env_from_rows(env: Any, spec: LeWMEnvSpec, start_row: Mapping[str, Any], goal_row: Mapping[str, Any]) -> np.ndarray:
    if spec.name == "pusht":
        env.reset(options={"state": np.asarray(start_row["state"], dtype=np.float64)})
        return np.asarray(env.unwrapped._get_obs(), dtype=np.float64)
    if spec.name == "tworoom":
        agent = np.asarray(start_row["observation"][:2], dtype=np.float32)
        target = np.asarray(goal_row["observation"][2:4], dtype=np.float32)
        obs, _ = env.reset(options={"state": agent, "target_state": target})
        return np.asarray(obs, dtype=np.float64)
    if spec.name == "cube":
        # Gymnasium requires one env.reset() per env instance; after that, CEM candidates
        # can restore via set_state only (full reset each candidate was prohibitively slow).
        unwrapped = env.unwrapped
        if not getattr(unwrapped, "_vpw_has_reset", False):
            env.reset()
            setattr(unwrapped, "_vpw_has_reset", True)
        unwrapped.set_state(
            np.asarray(start_row["qpos"], dtype=np.float64),
            np.asarray(start_row["qvel"], dtype=np.float64),
        )
        goal = np.asarray(goal_row["goal_block_pos"], dtype=np.float64)
        unwrapped.set_target_pos(0, goal)
        if hasattr(unwrapped, "_get_obs"):
            return np.asarray(unwrapped._get_obs(), dtype=np.float64)
        return np.asarray(unwrapped.compute_observation(), dtype=np.float64)
    if spec.name == "reacher":
        qpos = np.asarray(start_row["qpos"], dtype=np.float64)
        qvel = np.asarray(start_row["qvel"], dtype=np.float64)
        state = np.concatenate([qpos, qvel])
        target_qpos = np.asarray(goal_row["qpos"], dtype=np.float64)
        obs, _ = env.reset(options={"state": state, "target_qpos": target_qpos})
        return np.asarray(obs, dtype=np.float64)
    raise RuntimeError(f"Live sim reset not implemented for {spec.name}")


def _sim_obs_vector(env: Any, spec: LeWMEnvSpec, obs: Any) -> np.ndarray:
    """Normalize simulator observations to a flat state vector for planning rollouts."""
    if spec.name == "pusht":
        return np.asarray(env.unwrapped._get_obs(), dtype=np.float64)
    return np.asarray(obs, dtype=np.float64)


def sim_step(env: Any, spec: LeWMEnvSpec, action: np.ndarray) -> np.ndarray:
    if spec.action_dim == 2:
        obs, _, _, _, _ = env.step(np.asarray(action[:2], dtype=np.float32))
    else:
        obs, _, _, _, _ = env.step(np.asarray(action, dtype=np.float32))
    return _sim_obs_vector(env, spec, obs)


def sim_row_from_obs(env_name: str, obs: np.ndarray, *, template_row: Mapping[str, Any], info: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    spec = get_env_spec(env_name)
    if spec.name == "pusht":
        return {"state": np.asarray(obs, dtype=np.float64)}
    if spec.name == "tworoom":
        row = dict(template_row)
        row["observation"] = np.asarray(obs, dtype=np.float64).copy()
        return row
    if spec.name == "cube":
        row = dict(template_row)
        if info and "privileged/block_0_pos" in info:
            row["block_pos"] = np.asarray(info["privileged/block_0_pos"], dtype=np.float64)
        if info and "proprio/effector_pos" in info:
            row["effector_pos"] = np.asarray(info["proprio/effector_pos"], dtype=np.float64)
        if info and "qpos" in info:
            row["qpos"] = np.asarray(info["qpos"], dtype=np.float64)
        if info and "qvel" in info:
            row["qvel"] = np.asarray(info["qvel"], dtype=np.float64)
        return row
    if spec.name == "reacher":
        row = dict(template_row)
        if info and "qpos" in info:
            row["qpos"] = np.asarray(info["qpos"], dtype=np.float64)
        if info and "qvel" in info:
            row["qvel"] = np.asarray(info["qvel"], dtype=np.float64)
        if info and "finger_pos" in info:
            row["finger_pos"] = np.asarray(info["finger_pos"], dtype=np.float64)
        if info and "target_pos" in info:
            row["target_pos"] = np.asarray(info["target_pos"], dtype=np.float64)
        return row
    raise ValueError(env_name)


def sim_step_with_info(env: Any, spec: LeWMEnvSpec, action: np.ndarray) -> Tuple[np.ndarray, Mapping[str, Any]]:
    if spec.action_dim == 2:
        obs, _, _, _, info = env.step(np.asarray(action[:2], dtype=np.float32))
    else:
        obs, _, _, _, info = env.step(np.asarray(action, dtype=np.float32))
    return _sim_obs_vector(env, spec, obs), info if isinstance(info, Mapping) else {}
