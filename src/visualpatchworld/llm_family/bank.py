"""Mechanism bank: reusable parametric dynamics mechanisms shown to the proposer.

General entries are written by an LLM from a textbook-style topic list that never
mentions the evaluation environments. Experience entries are mechanisms that won
selection in *other* environments (leave-one-environment-out).
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .propose import _strip_think, chat

TOPICS = (
    "point-mass and rigid-body kinematics (first-order velocity control, integration)",
    "second-order dynamics (inertia, damping, drag, momentum)",
    "articulated chains and joint-space vs. task-space coordinates",
    "actuation and feedback control (gains, set-point tracking, saturation, dead zones)",
    "geometric constraints (bounds, barriers, obstacles, passages, joint limits)",
    "contact and collision between bodies (impulses, penetration, restitution)",
    "friction (Coulomb, viscous, static vs. kinetic) and sliding",
    "planar rigid-body motion with rotation (torque, moment of inertia, contact point)",
    "gravity, support and resting contact",
    "attachment and detachment between bodies (grasping, carrying, release)",
    "springs, elastic coupling and potential fields",
    "gated or regime-switching dynamics (thresholds, mode changes)",
)

BANK_PROMPT = """You are compiling a reference library of parametric mechanisms that appear in
classical mechanics and in robot control simulators. The library will later be shown to a
modeler who must write transition programs for environments you are not told about, so
entries must be GENERAL: use generic variable names (e.g. pos_x, vel_x, angle_i, act_i,
other_x), not names from any particular task.

Cover these topics, with 2-3 distinct mechanisms per topic:
{topics}

For each mechanism return a JSON object with keys:
  "name": short_snake_case,
  "topic": one of the topics above (short form),
  "mechanism": one sentence on the qualitative law,
  "equations": the discrete-time update in plain text (one step of length dt),
  "params": list of {{"name": ..., "meaning": ...}} (at most 6),
  "snippet": a short Python function (<= 15 lines, only `math`) implementing one step on
             generic variables,
  "tags": 5-10 lowercase keywords describing when it applies.

Return ONLY a JSON list of these objects, no prose."""


@dataclass
class BankEntry:
    name: str
    mechanism: str
    equations: str = ""
    snippet: str = ""
    params: List[Dict[str, str]] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    topic: str = ""
    source: str = "general"

    def render(self, with_snippet: bool = True) -> str:
        head = f"- {self.name} [{self.source}]: {self.mechanism}"
        if self.equations:
            head += f"\n    update: {self.equations.strip()}"
        if self.params:
            head += "\n    params: " + ", ".join(f"{p.get('name')} ({p.get('meaning')})" for p in self.params)
        if with_snippet and self.snippet:
            body = "\n".join("    " + line for line in self.snippet.strip().splitlines())
            head += f"\n    code:\n{body}"
        return head


def _parse_json_list(text: str) -> List[Dict[str, Any]]:
    text = _strip_think(text)
    m = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.S)
    if m:
        text = m.group(1)
    start, end = text.find("["), text.rfind("]")
    return json.loads(text[start : end + 1])


def build_general_bank(model: Optional[str], *, log: Optional[List[Dict[str, Any]]] = None) -> List[BankEntry]:
    prompt = BANK_PROMPT.format(topics="\n".join(f"  - {t}" for t in TOPICS))
    reply = chat(prompt, model=model, temperature=0.3, max_tokens=32000)
    if log is not None:
        log.append({"kind": "bank", "prompt": prompt, **reply})
    entries = []
    for obj in _parse_json_list(reply["text"]):
        if not isinstance(obj, dict) or not obj.get("name"):
            continue
        entries.append(
            BankEntry(
                name=str(obj["name"]),
                mechanism=str(obj.get("mechanism", "")),
                equations=str(obj.get("equations", "")),
                snippet=str(obj.get("snippet", "")),
                params=[p for p in obj.get("params", []) if isinstance(p, dict)],
                tags=[str(t).lower() for t in obj.get("tags", [])],
                topic=str(obj.get("topic", "")),
                source="general",
            )
        )
    return entries


def experience_entries(run_dirs: Sequence[Path], *, selection: str = "offline_multi_step") -> List[BankEntry]:
    """Winning programs from previously modeled environments."""
    out = []
    for d in run_dirs:
        fam = json.loads((Path(d) / "family.json").read_text())
        winner = fam["selections"][selection]["winner"]
        cand = next(c for c in fam["candidates"] if c["name"] == winner)
        out.append(
            BankEntry(
                name=winner,
                mechanism=cand["mechanism"],
                snippet=cand["code"],
                source=f"experience:{fam['env']}",
            )
        )
    return out


def save_bank(entries: Sequence[BankEntry], path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps([asdict(e) for e in entries], indent=2), encoding="utf-8")


def load_bank(path: Path) -> List[BankEntry]:
    return [BankEntry(**obj) for obj in json.loads(Path(path).read_text())]


def render_bank(entries: Sequence[BankEntry], *, with_snippet: bool = True) -> str:
    if not entries:
        return ""
    lines = "\n".join(e.render(with_snippet) for e in entries)
    return (
        "\nMechanism bank. These parametric mechanisms were collected from general physics and\n"
        "robot control, and (if marked experience) from other environments modeled earlier. They\n"
        "are NOT specific to this environment and most will not apply. You may instantiate, adapt\n"
        "or compose any that are consistent with the data, rename variables to this environment's\n"
        "state fields, and you should still propose mechanisms that are not in the bank when the\n"
        "data suggests them. The diversity requirement below still holds.\n"
        f"{lines}\n"
    )
