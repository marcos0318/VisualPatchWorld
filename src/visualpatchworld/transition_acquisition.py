"""Unified transition acquisition for dynamics refitting.

Combines simulator counterexamples, planning-failure regret, and near-contact
supervision into one priority-ranked pool instead of merging separate JSONLs.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .active_dynamics_induction import (
    SimCounterexample,
    collect_sim_counterexamples,
    summarize_counterexamples,
)
from .counterexample_store import CounterexampleStore, StoredCounterexample
from .planning_failure_mining import MinedFailure, mine_planning_failures, summarize_failures
from .pusht_local_induction import _parse_observation_text, _state_from_graph
from .sim_contact_supervision import (
    SimContactTransition,
    collect_sim_contact_supervision,
    summarize_sim_contact_supervision,
)


@dataclass(frozen=True)
class AcquisitionConfig:
    """Weights and collection knobs for unified acquisition."""

    top_k: int = 200
    contact_threshold: float = 70.0
    sim_error_weight: float = 1.0
    regret_weight: float = 1.5
    near_contact_weight: float = 0.8
    block_motion_weight: float = 0.4
    # Candidate collection (pre-ranking).
    num_explore: int = 96
    num_supervision: int = 400
    planning_num_eval: int = 16
    planning_num_samples: int = 64
    planning_cem_iters: int = 3
    error_threshold: float = 5.0
    regret_threshold: float = 20.0
    supervision_base_weight: float = 2.0
    counterexample_base_weight: float = 6.0
    planning_failure_weight: float = 5.0
    exclude_modules: Tuple[str, ...] = ("contact_gate",)
    seed: int = 42


@dataclass(frozen=True)
class AcquiredTransition:
    observation: str
    action: str
    next_observation: str
    weight: float
    module: str
    source: str
    priority: float
    contact_dist: float
    metadata: Mapping[str, Any] = field(default_factory=dict)


def _contact_distance_from_observation(observation: str) -> float:
    state = _state_from_graph(_parse_observation_text(observation))
    return float(math.hypot(state[2] - state[0], state[3] - state[1]))


def compute_priority(
    *,
    sim_error: float = 0.0,
    regret: float = 0.0,
    contact_dist: float = 999.0,
    block_motion: float = 0.0,
    config: AcquisitionConfig,
) -> float:
    """Score a candidate transition for acquisition ranking."""

    near_contact = 1.0 if contact_dist <= config.contact_threshold else 0.0
    coverage = config.near_contact_weight * near_contact
    if block_motion > 0.0:
        coverage += config.block_motion_weight * min(block_motion / 5.0, 1.0)
    return (
        config.sim_error_weight * max(0.0, sim_error)
        + config.regret_weight * max(0.0, regret)
        + coverage
    )


def _weight_from_priority(priority: float, *, base_weight: float, priority_scale: float = 10.0) -> float:
    return float(base_weight * (1.0 + priority / max(priority_scale, 1e-6)))


def from_sim_counterexample(row: SimCounterexample, *, config: AcquisitionConfig) -> AcquiredTransition:
    priority = compute_priority(
        sim_error=row.sim_error,
        contact_dist=row.contact_dist,
        config=config,
    )
    return AcquiredTransition(
        observation=row.observation,
        action=row.action,
        next_observation=row.next_observation,
        weight=_weight_from_priority(priority, base_weight=config.counterexample_base_weight),
        module=row.module,
        source=row.source,
        priority=priority,
        contact_dist=row.contact_dist,
        metadata={
            "sim_error": row.sim_error,
            "agent_error": row.agent_error,
            "block_error": row.block_error,
            "angle_error": row.angle_error,
        },
    )


def from_planning_failure(row: MinedFailure, *, config: AcquisitionConfig) -> AcquiredTransition:
    contact_dist = row.contact_dist if row.contact_dist >= 0.0 else _contact_distance_from_observation(row.observation)
    priority = compute_priority(
        regret=row.regret,
        contact_dist=contact_dist,
        config=config,
    )
    return AcquiredTransition(
        observation=row.observation,
        action=row.action,
        next_observation=row.next_observation,
        weight=_weight_from_priority(priority, base_weight=config.planning_failure_weight),
        module=row.module,
        source="planning_failure",
        priority=priority,
        contact_dist=contact_dist,
        metadata={
            "regret": row.regret,
            "model_goal_distance": row.model_goal_distance,
            "sim_goal_distance": row.sim_goal_distance,
        },
    )


def from_sim_contact(row: SimContactTransition, *, config: AcquisitionConfig) -> AcquiredTransition:
    priority = compute_priority(
        contact_dist=row.contact_dist,
        block_motion=row.block_motion,
        config=config,
    )
    return AcquiredTransition(
        observation=row.observation,
        action=row.action,
        next_observation=row.next_observation,
        weight=_weight_from_priority(priority, base_weight=config.supervision_base_weight),
        module=row.module,
        source=row.source,
        priority=priority,
        contact_dist=row.contact_dist,
        metadata={
            "block_motion": row.block_motion,
        },
    )


def merge_acquired_transitions(rows: Sequence[AcquiredTransition]) -> List[AcquiredTransition]:
    """Deduplicate by (observation, action), keeping the highest-priority row."""

    best: Dict[Tuple[str, str], AcquiredTransition] = {}
    for row in rows:
        key = (row.observation.strip(), row.action.strip())
        existing = best.get(key)
        if existing is None or row.priority > existing.priority:
            best[key] = row
    return sorted(best.values(), key=lambda item: item.priority, reverse=True)


def acquire_transitions(
    *,
    model: Any,
    config: AcquisitionConfig,
    trajectories_jsonl: Optional[Path] = None,
    dataset: str = "pusht_expert_train.h5",
    stablewm_home: Optional[str] = None,
    frameskip: int = 5,
    include_counterexamples: bool = True,
    include_planning_failures: bool = True,
    include_supervision: bool = True,
) -> Tuple[List[AcquiredTransition], Dict[str, Any]]:
    """Collect, score, merge, and rank transitions under a unified acquisition policy."""

    candidates: List[AcquiredTransition] = []
    diagnostics: Dict[str, Any] = {"config": asdict(config)}

    if include_counterexamples:
        sim_rows = collect_sim_counterexamples(
            model=model,
            trajectories_jsonl=trajectories_jsonl,
            dataset=dataset,
            stablewm_home=stablewm_home,
            frameskip=frameskip,
            num_explore=config.num_explore,
            error_threshold=config.error_threshold,
            top_k=0,
            base_weight=config.counterexample_base_weight,
            seed=config.seed,
        )
        candidates.extend(from_sim_counterexample(row, config=config) for row in sim_rows)
        diagnostics["sim_counterexamples"] = summarize_counterexamples(sim_rows)

    if include_planning_failures:
        failures = mine_planning_failures(
            model=model,
            dataset=dataset,
            stablewm_home=stablewm_home,
            num_eval=config.planning_num_eval,
            frameskip=frameskip,
            regret_threshold=config.regret_threshold,
            failure_weight=config.planning_failure_weight,
            num_samples=config.planning_num_samples,
            cem_iters=config.planning_cem_iters,
            seed=config.seed,
        )
        candidates.extend(from_planning_failure(row, config=config) for row in failures)
        diagnostics["planning_failures"] = summarize_failures(failures)

    if include_supervision:
        supervision = collect_sim_contact_supervision(
            dataset=dataset,
            stablewm_home=stablewm_home,
            frameskip=frameskip,
            num_samples=config.num_supervision,
            base_weight=config.supervision_base_weight,
            seed=config.seed,
            top_k=0,
        )
        candidates.extend(from_sim_contact(row, config=config) for row in supervision)
        diagnostics["sim_contact_supervision"] = summarize_sim_contact_supervision(supervision)

    merged = merge_acquired_transitions(candidates)
    if config.exclude_modules:
        excluded = set(config.exclude_modules)
        merged = [row for row in merged if row.module not in excluded]
    if config.top_k > 0:
        merged = merged[: config.top_k]

    diagnostics["candidate_count"] = len(candidates)
    diagnostics["merged_count"] = len(merged)
    diagnostics["top_k"] = config.top_k
    if merged:
        diagnostics["mean_priority"] = float(sum(row.priority for row in merged) / len(merged))
        diagnostics["max_priority"] = float(max(row.priority for row in merged))
        modules: Dict[str, int] = {}
        sources: Dict[str, int] = {}
        for row in merged:
            modules[row.module] = modules.get(row.module, 0) + 1
            sources[row.source] = sources.get(row.source, 0) + 1
        diagnostics["modules"] = dict(sorted(modules.items()))
        diagnostics["sources"] = dict(sorted(sources.items()))
        diagnostics["recommended_patch"] = _recommended_patch(modules)
    else:
        diagnostics["recommended_patch"] = "none"

    return merged, diagnostics


def _recommended_patch(modules: Mapping[str, int]) -> str:
    if not modules:
        return "none"
    top_module = max(modules.items(), key=lambda item: item[1])[0]
    mapping = {
        "contact": "refit_contact_head",
        "contact_angle": "refit_contact_mlp",
        "contact_gate": "tighten_contact_gate",
        "agent": "refit_agent_head",
        "angle": "refit_angle_head",
        "block_far": "audit_contact_gate",
    }
    return mapping.get(top_module, "refit_contact_head")


def acquired_to_stored(rows: Sequence[AcquiredTransition]) -> List[StoredCounterexample]:
    return [
        StoredCounterexample(
            observation=row.observation,
            action=row.action,
            next_observation=row.next_observation,
            weight=row.weight,
            module=row.module,
            source=row.source,
            metadata={
                "priority": row.priority,
                "contact_dist": row.contact_dist,
                **dict(row.metadata),
            },
        )
        for row in rows
    ]


def write_acquired_jsonl(rows: Sequence[AcquiredTransition], path: Path, *, item_id: str) -> None:
    record = {
        "metadata": {
            "env": "pusht",
            "source": "unified_transition_acquisition",
            "item_id": item_id,
            "split": "train",
        },
        "transitions": [
            {
                "observation": row.observation,
                "action": row.action,
                "next_observation": row.next_observation,
                "reward": 0.0,
                "done": False,
                "metadata": {
                    "weight": row.weight,
                    "module": row.module,
                    "source": row.source,
                    "priority": row.priority,
                    "contact_dist": row.contact_dist,
                    **dict(row.metadata),
                },
            }
            for row in rows
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")


def load_acquired_jsonl(path: Path | str) -> CounterexampleStore:
    return CounterexampleStore.from_jsonl_paths([path])
