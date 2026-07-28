"""Live tool-based scene-graph observations for cross-environment planning."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .lewm_env_specs import LeWMEnvSpec, get_env_spec
from .lewm_graph import graph_text_from_row, row_dict_from_h5
from .visual_tools import extractor_for_env


def graph_text_from_tool_payload(
    env_name: str,
    payload: Mapping[str, Any],
    *,
    template_row: Mapping[str, Any],
    goal_row: Mapping[str, Any],
    step: int = 0,
    metadata: Optional[Mapping[str, Any]] = None,
) -> str:
    """Convert a tool extractor payload into a scene-graph string for planning."""

    spec = get_env_spec(env_name)
    meta = dict(metadata or {})
    objects = {str(obj.get("id", "")): obj for obj in payload.get("objects", []) if isinstance(obj, Mapping)}

    if spec.name == "tworoom":
        obs = np.asarray(template_row["observation"], dtype=np.float64).copy()
        agent = objects.get("agent", {})
        agent_pos = agent.get("position", agent)
        if isinstance(agent_pos, Mapping):
            obs[0] = float(agent_pos.get("x", obs[0]))
            obs[1] = float(agent_pos.get("y", obs[1]))
        goal_obs = np.asarray(goal_row["observation"], dtype=np.float64)
        obs[2:4] = goal_obs[2:4]
        row = {"observation": obs}
    elif spec.name == "cube":
        row = dict(template_row)
        effector = objects.get("effector", {})
        block = objects.get("block", {})
        eff_pos = effector.get("position", effector)
        blk_pos = block.get("position", block)
        if isinstance(eff_pos, Mapping):
            row["effector_pos"] = np.array(
                [float(eff_pos.get("x", 0)), float(eff_pos.get("y", 0)), float(eff_pos.get("z", 0))],
                dtype=np.float64,
            )
        if isinstance(blk_pos, Mapping):
            row["block_pos"] = np.array(
                [float(blk_pos.get("x", 0)), float(blk_pos.get("y", 0)), float(blk_pos.get("z", 0))],
                dtype=np.float64,
            )
        row["goal_block_pos"] = np.asarray(goal_row["goal_block_pos"], dtype=np.float64)
    elif spec.name == "reacher":
        row = dict(template_row)
        finger = objects.get("finger", {})
        arm = objects.get("arm", {})
        finger_pos = finger.get("position", finger)
        if isinstance(finger_pos, Mapping):
            row["finger_pos"] = np.array(
                [float(finger_pos.get("x", 0)), float(finger_pos.get("y", 0))],
                dtype=np.float64,
            )
        if isinstance(arm, Mapping) and "qpos" in arm:
            row["qpos"] = np.asarray(arm["qpos"], dtype=np.float64)
        elif "qpos" in objects.get("arm", {}):
            row["qpos"] = np.asarray(objects["arm"]["qpos"], dtype=np.float64)
        row["target_pos"] = np.asarray(goal_row.get("target_pos", goal_row["qpos"]), dtype=np.float64)
        if "qpos" in goal_row:
            row.setdefault("goal_qpos", np.asarray(goal_row["qpos"], dtype=np.float64))
    else:
        raise ValueError(f"Unsupported env for tool payload conversion: {env_name}")

    return graph_text_from_row(spec.name, row, source="tool", step=step, metadata=meta)


def observation_text_from_env_tool_render(
    env: Any,
    spec: LeWMEnvSpec,
    *,
    template_row: Mapping[str, Any],
    goal_row: Mapping[str, Any],
    episode_index: Optional[int] = None,
    step: int = 0,
    fallback_observation_text: Optional[str] = None,
) -> Tuple[str, bool]:
    """Render the live env frame, run the env tool extractor, return (graph text, ok)."""

    from PIL import Image

    frame = np.asarray(env.render(), dtype=np.uint8)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
        path = Path(handle.name)
    try:
        Image.fromarray(frame).save(path)
        metadata = {
            "episode_index": episode_index,
            "goal_row": goal_row,
            "template_row": template_row,
        }
        payload = extractor_for_env(spec.name).extract(path, metadata=metadata)
    except Exception:
        meta = {"episode_index": episode_index} if episode_index is not None else {}
        if spec.name == "reacher" and _sim_row_has_reacher_geometry(template_row):
            return _graph_text_from_sim_row_reacher(
                template_row, goal_row, step=step, metadata=meta
            ), False
        if spec.name in {"cube", "tworoom"}:
            return _graph_text_from_sim_row(
                spec.name, template_row, goal_row, step=step, metadata=meta
            ), False
        if fallback_observation_text is not None:
            return fallback_observation_text, False
        raise
    finally:
        path.unlink(missing_ok=True)

    meta = {"episode_index": episode_index} if episode_index is not None else {}
    text = graph_text_from_tool_payload(
        spec.name,
        payload,
        template_row=template_row,
        goal_row=goal_row,
        step=step,
        metadata=meta,
    )
    return text, True


def _sim_row_has_reacher_geometry(row: Mapping[str, Any]) -> bool:
    return "finger_pos" in row or "qpos" in row


def _graph_text_from_sim_row(
    env_name: str,
    row: Mapping[str, Any],
    goal_row: Mapping[str, Any],
    *,
    step: int,
    metadata: Optional[Mapping[str, Any]] = None,
) -> str:
    """Build a tool-format graph from simulator state when CV extraction fails."""

    sim_row = dict(row)
    if env_name == "tworoom" and "observation" in sim_row and "observation" in goal_row:
        obs = np.asarray(sim_row["observation"], dtype=np.float64).copy()
        goal_obs = np.asarray(goal_row["observation"], dtype=np.float64)
        if obs.shape[0] >= 4 and goal_obs.shape[0] >= 4:
            obs[2:4] = goal_obs[2:4]
        sim_row["observation"] = obs
    elif env_name == "cube" and "goal_block_pos" not in sim_row and "goal_block_pos" in goal_row:
        sim_row["goal_block_pos"] = np.asarray(goal_row["goal_block_pos"], dtype=np.float64)
    return graph_text_from_row(env_name, sim_row, source="tool_fallback", step=step, metadata=metadata or {})


def _graph_text_from_sim_row_reacher(
    row: Mapping[str, Any],
    goal_row: Mapping[str, Any],
    *,
    step: int,
    metadata: Optional[Mapping[str, Any]] = None,
) -> str:
    """Build a tool-format graph from simulator state when CV extraction fails."""

    sim_row = dict(row)
    if "target_pos" not in sim_row:
        if "target_pos" in goal_row:
            sim_row["target_pos"] = np.asarray(goal_row["target_pos"], dtype=np.float64)
        elif "qpos" in goal_row:
            sim_row["target_pos"] = np.asarray(goal_row["qpos"], dtype=np.float64)
    return graph_text_from_row("reacher", sim_row, source="tool", step=step, metadata=metadata or {})
