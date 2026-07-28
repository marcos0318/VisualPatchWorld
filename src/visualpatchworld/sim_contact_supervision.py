"""Simulator-labeled near-contact transitions for dynamics distillation."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .hdf5 import require_h5py, resolve_dataset_path
from .lewm_sampling import graph_text_from_state
from .planning_eval import observation_text_from_state, rollout_sim_macro_values
from .pusht import discretize_action
from .pusht_local_induction import WeightedTransition
from .sim_eval import _make_pusht_env


@dataclass(frozen=True)
class SimContactTransition:
    observation: str
    action: str
    next_observation: str
    contact_dist: float
    block_motion: float
    weight: float
    module: str = "contact"
    source: str = "sim_contact_supervision"


def _macro_action_string(action_block: np.ndarray) -> str:
    mean_action = np.asarray(action_block, dtype=np.float64).mean(axis=0)
    return discretize_action(mean_action.tolist())


def _contact_distance_state(state: np.ndarray) -> float:
    agent = state[:2]
    block = state[2:4]
    return float(np.linalg.norm(block - agent))


def _block_motion(state: np.ndarray, next_state: np.ndarray) -> float:
    return float(np.linalg.norm(next_state[2:4] - state[2:4]))


def collect_sim_contact_supervision(
    *,
    dataset: str = "pusht_expert_train.h5",
    stablewm_home: Optional[str] = None,
    frameskip: int = 5,
    num_samples: int = 3000,
    contact_threshold: float = 70.0,
    contact_bias: float = 0.92,
    min_block_motion: float = 0.2,
    base_weight: float = 2.0,
    push_toward_block_prob: float = 0.75,
    seed: int = 42,
    top_k: int = 0,
) -> List[SimContactTransition]:
    """Roll the true simulator near contact and keep ground-truth push transitions."""

    dataset_path = resolve_dataset_path(dataset, stablewm_home)
    h5py = require_h5py()
    env = _make_pusht_env()
    rng = np.random.default_rng(seed)
    rows: List[SimContactTransition] = []

    try:
        with h5py.File(dataset_path, "r") as h5:
            states = h5["state"]
            total = int(len(states))
            attempts = 0
            max_attempts = max(num_samples * 8, num_samples)
            while len(rows) < num_samples and attempts < max_attempts:
                attempts += 1
                raw_index = int(rng.integers(0, max(1, total - frameskip - 1)))
                start_state = np.asarray(states[raw_index], dtype=np.float64)
                dist = _contact_distance_state(start_state)
                if dist > contact_threshold and rng.random() > contact_bias:
                    continue
                if dist > contact_threshold * 1.4:
                    continue

                agent = start_state[:2]
                block = start_state[2:4]
                direction = block - agent
                norm = float(np.linalg.norm(direction))
                if norm > 1e-6 and rng.random() < push_toward_block_prob:
                    direction = direction / norm
                    speed = float(rng.uniform(0.25, 1.0))
                    macro_action = direction * speed
                else:
                    macro_action = rng.uniform(-0.8, 0.8, size=2)
                    if dist <= contact_threshold:
                        macro_action = macro_action * rng.uniform(0.3, 1.0)

                macro_block = np.tile(macro_action.reshape(1, 2), (1, 1))
                sim_next = rollout_sim_macro_values(
                    env,
                    start_state,
                    macro_block,
                    frameskip=frameskip,
                )
                block_motion = _block_motion(start_state, sim_next)
                if block_motion < min_block_motion and dist > contact_threshold * 0.85:
                    continue

                action = _macro_action_string(np.tile(macro_block, (frameskip, 1)))
                observation = graph_text_from_state(
                    start_state,
                    step=0,
                    metadata={"raw_index": raw_index, "sim_contact": len(rows)},
                )
                next_observation = observation_text_from_state(sim_next)
                weight = float(base_weight * (1.0 + block_motion / 2.0))
                if dist <= contact_threshold:
                    weight *= 1.5
                rows.append(
                    SimContactTransition(
                        observation=observation,
                        action=action,
                        next_observation=next_observation,
                        contact_dist=dist,
                        block_motion=block_motion,
                        weight=weight,
                    )
                )
    finally:
        env.close()

    rows.sort(key=lambda row: (row.contact_dist <= contact_threshold, row.block_motion), reverse=True)
    if top_k > 0:
        rows = rows[:top_k]
    return rows


def sim_contact_to_weighted_transitions(rows: Sequence[SimContactTransition]) -> List[WeightedTransition]:
    return [
        WeightedTransition(
            observation=row.observation,
            action=row.action,
            next_observation=row.next_observation,
            weight=row.weight,
        )
        for row in rows
    ]


def summarize_sim_contact_supervision(rows: Sequence[SimContactTransition]) -> Dict[str, Any]:
    if not rows:
        return {"count": 0, "total_weight": 0.0}
    near = [row for row in rows if row.contact_dist <= 70.0]
    return {
        "count": len(rows),
        "near_contact_count": len(near),
        "total_weight": float(sum(row.weight for row in rows)),
        "mean_contact_dist": float(sum(row.contact_dist for row in rows) / len(rows)),
        "mean_block_motion": float(sum(row.block_motion for row in rows) / len(rows)),
    }


def write_sim_contact_jsonl(rows: Sequence[SimContactTransition], path: Path, *, item_id: str) -> None:
    record = {
        "metadata": {
            "env": "pusht",
            "source": "sim_contact_supervision",
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
                    "contact_dist": row.contact_dist,
                    "block_motion": row.block_motion,
                    "module": row.module,
                    "source": row.source,
                },
            }
            for row in rows
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
