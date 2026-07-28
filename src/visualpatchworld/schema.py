"""Scene-graph schema and deterministic text rendering for VisualPatchWorld."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional


def _round(value: float, digits: int = 3) -> float:
    return round(float(value), digits)


def _fmt(value: float, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


@dataclass(frozen=True)
class SceneObject:
    """An object tracked in a PushT / LeWM scene graph."""

    object_id: str
    category: str
    x: float
    y: float
    z: Optional[float] = None
    angle: Optional[float] = None
    vx: Optional[float] = None
    vy: Optional[float] = None
    w: Optional[float] = None
    confidence: float = 1.0
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "id": self.object_id,
            "category": self.category,
            "position": {"x": _round(self.x), "y": _round(self.y)},
            "confidence": _round(self.confidence),
        }
        if self.z is not None:
            data["position"]["z"] = _round(self.z)
        if self.angle is not None:
            data["angle"] = _round(self.angle)
        if self.vx is not None or self.vy is not None or self.w is not None:
            data["velocity"] = {
                "x": _round(self.vx or 0.0),
                "y": _round(self.vy or 0.0),
            }
            if self.w is not None:
                data["velocity"]["w"] = _round(self.w)
        if self.attributes:
            data["attributes"] = dict(self.attributes)
        return data


@dataclass(frozen=True)
class Relation:
    subject: str
    predicate: str
    object: str
    value: Optional[Any] = None

    def to_json(self) -> Dict[str, Any]:
        data = {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
        }
        if self.value is not None:
            data["value"] = _round(self.value) if isinstance(self.value, (int, float)) else self.value
        return data


@dataclass(frozen=True)
class SceneGraph:
    """Canonical PushT symbolic observation used by PatchWorld."""

    env: str
    source: str
    step: int
    objects: List[SceneObject]
    relations: List[Relation]
    goal: str = "move block to the target T pose"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        return {
            "env": self.env,
            "source": self.source,
            "step": int(self.step),
            "goal": self.goal,
            "objects": [obj.to_json() for obj in self.objects],
            "relations": [rel.to_json() for rel in self.relations],
            "metadata": dict(self.metadata),
        }


def build_relations(objects: Iterable[SceneObject], near_threshold: float = 35.0) -> List[Relation]:
    by_id = {obj.object_id: obj for obj in objects}
    agent = by_id.get("agent")
    block = by_id.get("block")
    if not agent or not block:
        return []

    dx = block.x - agent.x
    dy = block.y - agent.y
    distance = math.hypot(dx, dy)
    if abs(dx) >= abs(dy):
        direction = "right" if dx >= 0 else "left"
    else:
        direction = "up" if dy >= 0 else "down"

    return [
        Relation("agent", "relative_direction_to", "block", direction),
        Relation("agent", "distance_to", "block", distance),
        Relation("agent", "near", "block", distance <= near_threshold),
    ]


def graph_from_pusht_state(
    state: Iterable[float],
    *,
    source: str,
    step: int,
    metadata: Optional[Mapping[str, Any]] = None,
    block_velocity: Optional[Iterable[float]] = None,
) -> SceneGraph:
    """Build a PushT graph from LeWM's state vector.

    Expected state order:
    [agent_x, agent_y, block_x, block_y, block_angle, agent_vx, agent_vy].

    ``block_velocity`` optionally supplies the block's per-macro-step velocity
    ``(vx, vy, w)`` (e.g. estimated by finite difference at re-observation). The
    base state vector carries agent velocity but not block velocity, which is a
    hidden state variable for the dynamics; exposing it lets second-order
    dynamics models (e.g. pushing-mechanics) condition on incoming momentum.
    """

    values = list(float(v) for v in state)
    if len(values) < 5:
        raise ValueError(f"PushT state must have at least 5 values, got {len(values)}")
    agent_vx = values[5] if len(values) > 5 else 0.0
    agent_vy = values[6] if len(values) > 6 else 0.0
    bvx = bvy = bw = None
    if block_velocity is not None:
        bv = list(float(v) for v in block_velocity)
        bvx = bv[0] if len(bv) > 0 else 0.0
        bvy = bv[1] if len(bv) > 1 else 0.0
        bw = bv[2] if len(bv) > 2 else 0.0
    objects = [
        SceneObject(
            object_id="agent",
            category="circular_pusher",
            x=values[0],
            y=values[1],
            vx=agent_vx,
            vy=agent_vy,
        ),
        SceneObject(
            object_id="block",
            category="t_block",
            x=values[2],
            y=values[3],
            angle=values[4],
            vx=bvx,
            vy=bvy,
            w=bw,
        ),
    ]
    return SceneGraph(
        env="pusht",
        source=source,
        step=step,
        objects=objects,
        relations=build_relations(objects),
        metadata=metadata or {},
    )


def render_graph_text(graph: SceneGraph) -> str:
    """Render a graph as stable text for PatchWorld induction."""

    data = graph.to_json()
    object_lines = []
    for obj in sorted(data["objects"], key=lambda item: item["id"]):
        pos = obj["position"]
        parts = [
            f"{obj['id']}:{obj['category']}",
            f"x={_fmt(pos['x'])}",
            f"y={_fmt(pos['y'])}",
        ]
        if "angle" in obj:
            parts.append(f"angle={_fmt(obj['angle'])}")
        if "velocity" in obj:
            vel = obj["velocity"]
            parts.append(f"vx={_fmt(vel['x'])}")
            parts.append(f"vy={_fmt(vel['y'])}")
            if "w" in vel:
                parts.append(f"w={_fmt(vel['w'])}")
        parts.append(f"conf={_fmt(obj.get('confidence', 1.0))}")
        object_lines.append("- " + " ".join(parts))

    relation_lines = []
    for rel in sorted(data["relations"], key=lambda item: (item["subject"], item["predicate"], item["object"])):
        value = rel.get("value")
        if isinstance(value, float):
            value_text = _fmt(value)
        else:
            value_text = json.dumps(value, sort_keys=True) if isinstance(value, bool) else str(value)
        relation_lines.append(f"- {rel['subject']} {rel['predicate']} {rel['object']} = {value_text}")

    metadata = data.get("metadata", {})
    meta_text = json.dumps(metadata, sort_keys=True, separators=(",", ":")) if metadata else "{}"
    sections = [
        "Environment: pusht",
        f"Source: {graph.source}",
        f"Step: {graph.step}",
        f"Goal: {graph.goal}",
        "Objects:",
        *object_lines,
        "Relations:",
        *relation_lines,
        f"Metadata: {meta_text}",
    ]
    return "\n".join(sections)


def canonical_graph_json(graph: SceneGraph) -> str:
    return json.dumps(graph.to_json(), sort_keys=True, separators=(",", ":"))
