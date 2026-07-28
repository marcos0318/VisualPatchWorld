"""Active counterexample-guided dynamics induction helpers."""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .hdf5 import require_h5py, resolve_dataset_path
from .lewm_sampling import graph_text_from_state, state_vector_from_payload
from .planning_eval import observation_text_from_state, rollout_sim_macro_values
from .pusht import discretize_action
from .pusht_local_induction import (
    WeightedTransition,
    _parse_observation_text,
    _state_from_graph,
)
from .sim_eval import _make_pusht_env, _predict_next_graph


@dataclass(frozen=True)
class SimCounterexample:
    observation: str
    action: str
    next_observation: str
    sim_error: float
    agent_error: float
    block_error: float
    angle_error: float
    contact_dist: float
    module: str
    weight: float
    source: str = "sim_counterexample"


def diagnose_failure_module(
    *,
    agent_error: float,
    block_error: float,
    angle_error: float,
    contact_dist: float,
    contact_threshold: float = 70.0,
) -> str:
    """Route a transition error to the dynamics module that should be patched."""

    if contact_dist > contact_threshold:
        if block_error > 0.5:
            return "contact_gate"
        if agent_error >= block_error:
            return "agent"
        return "block_far"

    if block_error >= agent_error and block_error >= angle_error * 8.0:
        if angle_error > 0.15 and angle_error >= block_error * 0.2:
            return "contact_angle"
        return "contact"
    if angle_error > max(agent_error, block_error):
        return "angle"
    if agent_error >= block_error:
        return "agent"
    return "contact"


def _state_errors(
    predicted: Sequence[float],
    target: Sequence[float],
) -> Tuple[float, float, float, float]:
    pred = np.asarray(predicted, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    agent_error = float(np.linalg.norm(pred[:2] - truth[:2]))
    block_error = float(np.linalg.norm(pred[2:4] - truth[2:4]))
    angle_error = abs(float(pred[4]) - float(truth[4]))
    angle_error = min(angle_error, abs(2.0 * math.pi - angle_error))
    total = float(agent_error + block_error + 10.0 * angle_error)
    return total, agent_error, block_error, angle_error


def _contact_distance_from_observation(observation: str) -> float:
    state = _state_from_graph(_parse_observation_text(observation))
    return float(math.hypot(state[2] - state[0], state[3] - state[1]))


def _weight_from_error(
    error: float,
    *,
    base_weight: float,
    error_scale: float,
) -> float:
    return float(base_weight * (1.0 + max(0.0, error) / max(error_scale, 1e-6)))


def _macro_action_string(action_block: np.ndarray) -> str:
    mean_action = np.asarray(action_block, dtype=np.float64).mean(axis=0)
    return discretize_action(mean_action.tolist())


def _candidate_from_states(
    model: Any,
    *,
    observation: str,
    action: str,
    start_state: np.ndarray,
    sim_next_state: np.ndarray,
    base_weight: float,
    error_scale: float,
    source: str,
    error_threshold: float,
) -> Optional[SimCounterexample]:
    try:
        predicted_graph = _predict_next_graph(model, observation, action)
        predicted_state = np.asarray(state_vector_from_payload(predicted_graph), dtype=np.float64)
    except Exception:
        return None

    sim_error, agent_error, block_error, angle_error = _state_errors(predicted_state, sim_next_state)
    if sim_error < error_threshold:
        return None

    contact_dist = _contact_distance_from_observation(observation)
    module = diagnose_failure_module(
        agent_error=agent_error,
        block_error=block_error,
        angle_error=angle_error,
        contact_dist=contact_dist,
    )
    next_observation = observation_text_from_state(sim_next_state)
    return SimCounterexample(
        observation=observation,
        action=action,
        next_observation=next_observation,
        sim_error=sim_error,
        agent_error=agent_error,
        block_error=block_error,
        angle_error=angle_error,
        contact_dist=contact_dist,
        module=module,
        weight=_weight_from_error(sim_error, base_weight=base_weight, error_scale=error_scale),
        source=source,
    )


def collect_sim_counterexamples_from_trajectories(
    *,
    model: Any,
    trajectories_jsonl: Path,
    dataset: str,
    stablewm_home: Optional[str],
    frameskip: int = 5,
    max_transitions: int = 0,
    error_threshold: float = 5.0,
    top_k: int = 128,
    base_weight: float = 6.0,
    error_scale: float = 10.0,
) -> List[SimCounterexample]:
    """Collect simulator-labeled counterexamples from trajectory transitions."""

    dataset_path = resolve_dataset_path(dataset, stablewm_home)
    h5py = require_h5py()
    candidates: List[SimCounterexample] = []

    with h5py.File(dataset_path, "r") as h5:
            states = h5["state"]
            actions_h5 = h5.get("action")
            with trajectories_jsonl.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    for transition in record.get("transitions", []):
                        metadata = transition.get("metadata", {})
                        if not isinstance(metadata, Mapping):
                            continue
                        if "raw_index" not in metadata or "next_raw_index" not in metadata:
                            continue
                        raw_index = int(metadata["raw_index"])
                        next_raw_index = int(metadata["next_raw_index"])
                        start_state = np.asarray(states[raw_index], dtype=np.float64)
                        sim_next_state = np.asarray(states[next_raw_index], dtype=np.float64)

                        observation = str(transition.get("observation", ""))
                        if not observation:
                            observation = graph_text_from_state(
                                start_state,
                                step=int(metadata.get("step", 0)),
                                metadata={"raw_index": raw_index},
                            )
                        action = str(transition.get("action", ""))
                        if not action:
                            if metadata.get("action_block"):
                                action_block = np.asarray(metadata["action_block"], dtype=np.float32)
                            elif actions_h5 is not None:
                                action_block = np.asarray(
                                    actions_h5[raw_index : raw_index + frameskip],
                                    dtype=np.float32,
                                )
                            else:
                                continue
                            action = _macro_action_string(action_block)

                        item = _candidate_from_states(
                            model,
                            observation=observation,
                            action=action,
                            start_state=start_state,
                            sim_next_state=sim_next_state,
                            base_weight=base_weight,
                            error_scale=error_scale,
                            source="trajectory",
                            error_threshold=error_threshold,
                        )
                        if item is not None:
                            candidates.append(item)
                        if max_transitions > 0 and len(candidates) >= max_transitions * 4:
                            break
    candidates.sort(key=lambda row: row.sim_error, reverse=True)
    if top_k > 0:
        candidates = candidates[:top_k]
    return candidates


def collect_sim_counterexamples_from_exploration(
    *,
    model: Any,
    dataset: str,
    stablewm_home: Optional[str],
    frameskip: int = 5,
    num_samples: int = 96,
    error_threshold: float = 5.0,
    top_k: int = 96,
    base_weight: float = 8.0,
    error_scale: float = 10.0,
    seed: int = 42,
    contact_bias: float = 0.7,
) -> List[SimCounterexample]:
    """Actively explore random macro-actions and keep high model-vs-sim error states."""

    dataset_path = resolve_dataset_path(dataset, stablewm_home)
    h5py = require_h5py()
    env = _make_pusht_env()
    rng = np.random.default_rng(seed)
    candidates: List[SimCounterexample] = []

    try:
        with h5py.File(dataset_path, "r") as h5:
            states = h5["state"]
            total = int(len(states))
            for sample_idx in range(num_samples):
                raw_index = int(rng.integers(0, max(1, total - frameskip - 1)))
                start_state = np.asarray(states[raw_index], dtype=np.float64)
                agent = start_state[:2]
                block = start_state[2:4]
                dist = float(np.linalg.norm(block - agent))
                if dist > 90.0 and rng.random() > contact_bias:
                    continue

                macro_action = rng.uniform(-1.0, 1.0, size=(1, 2))
                if dist <= 70.0:
                    direction = block - agent
                    norm = float(np.linalg.norm(direction))
                    if norm > 1e-6:
                        direction = direction / norm
                        macro_action[0] = direction * rng.uniform(0.2, 1.0)

                sim_next_state = rollout_sim_macro_values(
                    env,
                    start_state,
                    macro_action,
                    frameskip=frameskip,
                )
                action = _macro_action_string(np.tile(macro_action, (frameskip, 1)))
                observation = graph_text_from_state(
                    start_state,
                    step=0,
                    metadata={"raw_index": raw_index, "active_explore": sample_idx},
                )
                item = _candidate_from_states(
                    model,
                    observation=observation,
                    action=action,
                    start_state=start_state,
                    sim_next_state=sim_next_state,
                    base_weight=base_weight,
                    error_scale=error_scale,
                    source="active_explore",
                    error_threshold=error_threshold,
                )
                if item is not None:
                    candidates.append(item)
    finally:
        env.close()

    candidates.sort(key=lambda row: row.sim_error, reverse=True)
    if top_k > 0:
        candidates = candidates[:top_k]
    return candidates


def collect_sim_counterexamples(
    *,
    model: Any,
    trajectories_jsonl: Optional[Path] = None,
    dataset: str = "pusht_expert_train.h5",
    stablewm_home: Optional[str] = None,
    frameskip: int = 5,
    max_transitions: int = 0,
    num_explore: int = 96,
    error_threshold: float = 5.0,
    top_k: int = 160,
    base_weight: float = 6.0,
    error_scale: float = 10.0,
    seed: int = 42,
) -> List[SimCounterexample]:
    rows: List[SimCounterexample] = []
    if trajectories_jsonl is not None and trajectories_jsonl.is_file():
        rows.extend(
            collect_sim_counterexamples_from_trajectories(
                model=model,
                trajectories_jsonl=trajectories_jsonl,
                dataset=dataset,
                stablewm_home=stablewm_home,
                frameskip=frameskip,
                max_transitions=max_transitions,
                error_threshold=error_threshold,
                top_k=0,
                base_weight=base_weight,
                error_scale=error_scale,
            )
        )
    if num_explore > 0:
        rows.extend(
            collect_sim_counterexamples_from_exploration(
                model=model,
                dataset=dataset,
                stablewm_home=stablewm_home,
                frameskip=frameskip,
                num_samples=num_explore,
                error_threshold=error_threshold,
                top_k=0,
                base_weight=base_weight + 2.0,
                error_scale=error_scale,
                seed=seed,
            )
        )

    dedup: Dict[Tuple[str, str], SimCounterexample] = {}
    for row in rows:
        key = (row.observation.strip(), row.action.strip())
        existing = dedup.get(key)
        if existing is None or row.sim_error > existing.sim_error:
            dedup[key] = row
    merged = sorted(dedup.values(), key=lambda item: item.sim_error, reverse=True)
    if top_k > 0:
        merged = merged[:top_k]
    return merged


def summarize_counterexamples(rows: Sequence[SimCounterexample]) -> Dict[str, Any]:
    if not rows:
        return {
            "count": 0,
            "mean_sim_error": 0.0,
            "modules": {},
            "sources": {},
        }
    modules = Counter(row.module for row in rows)
    sources = Counter(row.source for row in rows)
    errors = [row.sim_error for row in rows]
    contact_rows = [row for row in rows if row.module in {"contact", "contact_angle"}]
    return {
        "count": len(rows),
        "mean_sim_error": float(sum(errors) / len(errors)),
        "max_sim_error": float(max(errors)),
        "mean_contact_dist": float(sum(row.contact_dist for row in rows) / len(rows)),
        "contact_bucket_count": len(contact_rows),
        "modules": dict(sorted(modules.items())),
        "sources": dict(sorted(sources.items())),
        "recommended_patch": _recommended_patch(modules),
    }


def _recommended_patch(modules: Counter) -> str:
    if not modules:
        return "none"
    top_module, _ = modules.most_common(1)[0]
    mapping = {
        "contact": "refit_contact_head",
        "contact_angle": "refit_contact_mlp",
        "contact_gate": "tighten_contact_gate",
        "agent": "refit_agent_head",
        "angle": "refit_angle_head",
        "block_far": "audit_contact_gate",
    }
    return mapping.get(top_module, "refit_contact_head")


def counterexamples_to_weighted_transitions(
    rows: Sequence[SimCounterexample],
) -> List[WeightedTransition]:
    return [
        WeightedTransition(
            observation=row.observation,
            action=row.action,
            next_observation=row.next_observation,
            weight=row.weight,
        )
        for row in rows
    ]


def write_counterexample_jsonl(rows: Sequence[SimCounterexample], path: Path, *, item_id: str) -> None:
    record = {
        "metadata": {
            "env": "pusht",
            "source": "sim_counterexample_mining",
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
                    "sim_error": row.sim_error,
                    "agent_error": row.agent_error,
                    "block_error": row.block_error,
                    "angle_error": row.angle_error,
                    "contact_dist": row.contact_dist,
                    "module": row.module,
                    "source": row.source,
                },
            }
            for row in rows
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")


def evaluate_acceptance(
    *,
    baseline_report: Mapping[str, Any],
    candidate_report: Mapping[str, Any],
    min_success_gain: float = 0.0,
    max_goal_distance_regression: float = 3.0,
) -> Dict[str, Any]:
    """Return whether a refit round should be accepted for planning."""

    baseline_success = float(baseline_report.get("success_rate", 0.0))
    candidate_success = float(candidate_report.get("success_rate", 0.0))
    baseline_goal = float(baseline_report.get("mean_goal_distance", float("inf")))
    candidate_goal = float(candidate_report.get("mean_goal_distance", float("inf")))
    success_delta = candidate_success - baseline_success
    goal_delta = candidate_goal - baseline_goal
    accepted = success_delta >= min_success_gain - 1e-9 and goal_delta <= max_goal_distance_regression + 1e-9
    return {
        "accepted": accepted,
        "baseline_success_rate": baseline_success,
        "candidate_success_rate": candidate_success,
        "success_delta": success_delta,
        "baseline_mean_goal_distance": baseline_goal,
        "candidate_mean_goal_distance": candidate_goal,
        "goal_distance_delta": goal_delta,
        "min_success_gain": min_success_gain,
        "max_goal_distance_regression": max_goal_distance_regression,
    }
