"""Append-only interaction history (Schema Timeline) with curated views.

VPW wants exact executable models too. Continuous domains cannot use ARC-style
exact grid match, so Timeline stores residuals and trust tags. Fit / promotion
use curated slices; live surprises are induction fuel, not a plan-permission gate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from patchworld.worldmodel_data import Trajectory, Transition


@dataclass
class TaggedTransition:
    transition: Transition
    source: str = "live"  # seed | surprise | live | probe
    residual: Optional[float] = None
    weight: float = 1.0


@dataclass
class Timeline:
    """Immutable-append log of real (obs, action, next_obs) transitions."""

    env: str
    records: List[TaggedTransition] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def transitions(self) -> List[Transition]:
        return [r.transition for r in self.records]

    @property
    def n_transitions(self) -> int:
        return len(self.records)

    def append(
        self,
        observation: str,
        action: str,
        next_observation: str,
        *,
        reward: float = 0.0,
        done: bool = False,
        source: str = "live",
        residual: Optional[float] = None,
        weight: float = 1.0,
    ) -> None:
        self.records.append(
            TaggedTransition(
                transition=Transition(
                    observation=str(observation),
                    action=str(action),
                    next_observation=str(next_observation),
                    reward=float(reward),
                    done=bool(done),
                ),
                source=str(source),
                residual=None if residual is None else float(residual),
                weight=float(weight),
            )
        )

    def extend_from_trajectories(
        self,
        trajectories: Sequence[Trajectory],
        *,
        max_transitions: int = 0,
        source: str = "seed",
    ) -> int:
        added = 0
        for traj in trajectories:
            for step in traj.transitions:
                self.records.append(TaggedTransition(transition=step, source=source, weight=1.0))
                added += 1
                if max_transitions > 0 and added >= max_transitions:
                    return added
        return added

    def curated_for_fit(
        self,
        *,
        include_sources: Sequence[str] = ("seed", "surprise"),
        max_surprise_frac: float = 0.3,
        max_transitions: int = 0,
    ) -> List[Trajectory]:
        """Return a fit/promotion view that caps surprise pollution.

        Always keep seed transitions; add surprises up to max_surprise_frac of
        the returned mass. Live transitions are excluded from promotion fits
        unless explicitly listed in include_sources.
        """
        seed = [r for r in self.records if r.source == "seed"]
        surprise = [r for r in self.records if r.source == "surprise"]
        other = [r for r in self.records if r.source in include_sources and r.source not in ("seed", "surprise")]

        chosen: List[TaggedTransition] = list(seed) + list(other)
        if surprise and max_surprise_frac > 0:
            cap = max(1, int(len(chosen) * max_surprise_frac / max(1e-6, 1.0 - max_surprise_frac))) if chosen else len(surprise)
            # Prefer largest residuals (most informative counterexamples).
            ranked = sorted(
                surprise,
                key=lambda r: float(r.residual if r.residual is not None else r.weight),
                reverse=True,
            )
            chosen.extend(ranked[:cap])

        if max_transitions > 0:
            chosen = chosen[:max_transitions]
        if not chosen:
            # Fallback: seed-only or empty.
            chosen = list(seed)[: max_transitions or None]

        transitions = [r.transition for r in chosen]
        if not transitions:
            return []
        return [
            Trajectory(
                transitions=transitions,
                env=self.env,
                item_id=f"{self.env}-schema-curated",
                task_idx=0,
                rollout_index=0,
                split="curated",
                success=False,
                total_reward=sum(t.reward for t in transitions),
            )
        ]

    def as_trajectories(self) -> List[Trajectory]:
        if not self.records:
            return []
        return [
            Trajectory(
                transitions=list(self.transitions),
                env=self.env,
                item_id=f"{self.env}-schema-timeline",
                task_idx=0,
                rollout_index=0,
                split="timeline",
                success=False,
                total_reward=sum(t.reward for t in self.transitions),
            )
        ]

    def add_note(self, text: str) -> None:
        self.notes.append(str(text))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "env": self.env,
            "notes": list(self.notes),
            "records": [
                {
                    "source": r.source,
                    "residual": r.residual,
                    "weight": r.weight,
                    "observation": r.transition.observation,
                    "action": r.transition.action,
                    "next_observation": r.transition.next_observation,
                    "reward": r.transition.reward,
                    "done": r.transition.done,
                }
                for r in self.records
            ],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Timeline":
        payload = json.loads(path.read_text(encoding="utf-8"))
        tl = cls(env=str(payload.get("env", "")))
        tl.notes = [str(x) for x in payload.get("notes", [])]
        rows = payload.get("records") or payload.get("transitions", [])
        for raw in rows:
            tl.append(
                raw.get("observation", ""),
                raw.get("action", ""),
                raw.get("next_observation", ""),
                reward=float(raw.get("reward", 0.0)),
                done=bool(raw.get("done", False)),
                source=str(raw.get("source", "live")),
                residual=raw.get("residual"),
                weight=float(raw.get("weight", 1.0)),
            )
        return tl


def transitions_from_mapping(rows: Iterable[Mapping[str, Any]]) -> List[Transition]:
    out: List[Transition] = []
    for row in rows:
        out.append(
            Transition(
                observation=str(row.get("observation", "")),
                action=str(row.get("action", "")),
                next_observation=str(row.get("next_observation", "")),
                reward=float(row.get("reward", 0.0)),
                done=bool(row.get("done", False)),
            )
        )
    return out
