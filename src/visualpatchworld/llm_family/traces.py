"""Load oracle scene-graph traces as flat ``{object.field: value}`` states."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

_OBJ_RE = re.compile(r"^-\s*(\w+):(\w+)\s*(.*)$")
_KV_RE = re.compile(r"(\w+)=(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")
_STATE_FIELDS = ("x", "y", "z", "angle")


def parse_graph_text(text: str) -> Dict[str, Any]:
    """Return ``{"state": {obj.field: v}, "categories": {...}, "metadata": {...}}``."""
    state: Dict[str, float] = {}
    categories: Dict[str, str] = {}
    metadata: Dict[str, Any] = {}
    section = ""
    for raw in str(text or "").splitlines():
        line = raw.strip()
        if line.startswith("Objects:"):
            section = "objects"
        elif line.startswith("Relations:"):
            section = "relations"
        elif line.startswith("Metadata:"):
            try:
                metadata = json.loads(line.split(":", 1)[1].strip())
            except Exception:
                metadata = {}
            section = ""
        elif section == "objects" and line.startswith("-"):
            m = _OBJ_RE.match(line)
            if not m:
                continue
            categories[m.group(1)] = m.group(2)
            for key, value in _KV_RE.findall(m.group(3)):
                if key in _STATE_FIELDS:
                    state[f"{m.group(1)}.{key}"] = float(value)
    return {"state": state, "categories": categories, "metadata": metadata}


@dataclass
class Transition:
    state: Dict[str, float]
    action: List[float]
    next_state: Dict[str, float]
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Episode:
    transitions: List[Transition]

    def __len__(self) -> int:
        return len(self.transitions)


def _action_values(transition: Mapping[str, Any]) -> List[float]:
    md = transition.get("metadata") or {}
    values = md.get("action_values") if isinstance(md, Mapping) else None
    if values is None:
        values = transition.get("action_values")
    if values is None:
        raise ValueError("transition has no action_values")
    return [float(v) for v in values]


def load_episodes(path: Path, *, max_episodes: Optional[int] = None) -> List[Episode]:
    episodes: List[Episode] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            items: List[Transition] = []
            for t in record.get("transitions", []):
                cur = parse_graph_text(t["observation"])
                nxt = parse_graph_text(t["next_observation"])
                items.append(
                    Transition(
                        state=cur["state"],
                        action=_action_values(t),
                        next_state=nxt["state"],
                        meta=cur["metadata"],
                    )
                )
            if items:
                episodes.append(Episode(items))
            if max_episodes and len(episodes) >= max_episodes:
                break
    return episodes


def state_keys(episodes: Sequence[Episode]) -> List[str]:
    keys = set()
    for ep in episodes:
        for t in ep.transitions:
            keys.update(t.state)
    return sorted(keys)


def state_scales(episodes: Sequence[Episode], keys: Sequence[str]) -> Dict[str, float]:
    """Per-key std of one-step deltas, used to normalize prediction errors."""
    deltas: Dict[str, List[float]] = {k: [] for k in keys}
    for ep in episodes:
        for t in ep.transitions:
            for k in keys:
                if k in t.state and k in t.next_state:
                    deltas[k].append(t.next_state[k] - t.state[k])
    scales = {}
    for k, vals in deltas.items():
        s = float(np.std(vals)) if vals else 0.0
        scales[k] = s if s > 1e-6 else 1.0
    return scales


def schema_summary(episodes: Sequence[Episode]) -> Dict[str, Any]:
    keys = state_keys(episodes)
    stats: Dict[str, Dict[str, float]] = {}
    for k in keys:
        vals = [t.state[k] for ep in episodes for t in ep.transitions if k in t.state]
        stats[k] = {"min": float(np.min(vals)), "max": float(np.max(vals))}
    acts = np.asarray([t.action for ep in episodes for t in ep.transitions], dtype=np.float64)
    return {
        "state_keys": keys,
        "state_ranges": stats,
        "action_dim": int(acts.shape[1]),
        "action_min": acts.min(axis=0).round(3).tolist(),
        "action_max": acts.max(axis=0).round(3).tolist(),
        "num_episodes": len(episodes),
        "num_transitions": int(acts.shape[0]),
    }
