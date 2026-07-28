"""Mine high-regret planning transitions for weighted dynamics refitting."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

import math

from .active_dynamics_induction import diagnose_failure_module
from .lewm_sampling import graph_text_from_state, state_vector_from_payload
from .planning_eval import (
  action_values_to_strings,
  observation_text_from_state,
  plan_cem,
  pusht_goal_distance,
  rollout_model_state,
  rollout_sim_macro_values,
  select_lewm_eval_starts,
  _global_index,
)
from .pusht_local_induction import WeightedTransition
from .sim_eval import _make_pusht_env


def _contact_distance_from_state(state: np.ndarray) -> float:
  return float(np.linalg.norm(state[2:4] - state[:2]))


def _state_errors(predicted: np.ndarray, target: np.ndarray) -> tuple[float, float, float]:
  agent_error = float(np.linalg.norm(predicted[:2] - target[:2]))
  block_error = float(np.linalg.norm(predicted[2:4] - target[2:4]))
  angle_error = abs(float(predicted[4]) - float(target[4]))
  angle_error = min(angle_error, abs(2.0 * math.pi - angle_error))
  return agent_error, block_error, angle_error


@dataclass(frozen=True)
class MinedFailure:
  observation: str
  action: str
  next_observation: str
  model_goal_distance: float
  sim_goal_distance: float
  regret: float
  weight: float
  module: str = "contact"
  contact_dist: float = -1.0
  agent_error: float = 0.0
  block_error: float = 0.0
  angle_error: float = 0.0


def mine_planning_failures(
  *,
  model: Any,
  dataset: str,
  stablewm_home: Optional[str],
  num_eval: int = 16,
  goal_offset_steps: int = 25,
  eval_budget: int = 50,
  frameskip: int = 5,
  num_samples: int = 64,
  cem_iters: int = 3,
  elite_frac: float = 0.1,
  failure_weight: float = 5.0,
  regret_threshold: float = 20.0,
  seed: int = 42,
) -> List[MinedFailure]:
  from .hdf5 import require_h5py, resolve_dataset_path

  dataset_path = resolve_dataset_path(dataset, stablewm_home)
  h5py = require_h5py()
  env = _make_pusht_env()
  failures: List[MinedFailure] = []

  try:
    with h5py.File(dataset_path, "r") as h5:
      states = h5["state"]
      eval_starts = select_lewm_eval_starts(
        h5,
        num_eval=num_eval,
        goal_offset_steps=goal_offset_steps,
        seed=seed,
      )
      macro_budget = max(1, eval_budget // frameskip)

      for trial_id, start in enumerate(eval_starts):
        start_global = _global_index(h5, start.episode_index, start.start_step)
        goal_global = _global_index(h5, start.episode_index, start.start_step + goal_offset_steps)
        start_state = np.asarray(states[start_global], dtype=np.float64)
        goal_state = np.asarray(states[goal_global], dtype=np.float64)
        obs_text = graph_text_from_state(
          start_state,
          step=0,
          metadata={"episode_index": start.episode_index},
        )
        current_state = start_state.copy()
        current_obs = obs_text

        for step_idx in range(macro_budget):
          plan_values, _, _ = plan_cem(
            model,
            current_obs,
            goal_state,
            horizon=1,
            num_samples=num_samples,
            cem_iters=cem_iters,
            elite_frac=elite_frac,
            seed=seed + trial_id * 100 + step_idx,
          )
          action = action_values_to_strings(plan_values)[0]
          try:
            model_next = rollout_model_state(model, current_obs, [action])
            model_score = pusht_goal_distance(goal_state, model_next)
          except Exception:
            continue
          sim_next = rollout_sim_macro_values(env, current_state, plan_values[:1], frameskip=frameskip)
          sim_score = pusht_goal_distance(goal_state, sim_next)
          regret = float(sim_score - model_score)
          contact_dist = _contact_distance_from_state(current_state)
          agent_error, block_error, angle_error = _state_errors(model_next, sim_next)
          module = diagnose_failure_module(
            agent_error=agent_error,
            block_error=block_error,
            angle_error=angle_error,
            contact_dist=contact_dist,
          )
          if regret < regret_threshold:
            current_state = sim_next
            current_obs = observation_text_from_state(
              current_state,
              episode_index=start.episode_index,
              step=step_idx + 1,
            )
            continue

          failures.append(
            MinedFailure(
              observation=current_obs,
              action=action,
              next_observation=observation_text_from_state(
                sim_next,
                episode_index=start.episode_index,
                step=step_idx + 1,
              ),
              model_goal_distance=model_score,
              sim_goal_distance=sim_score,
              regret=regret,
              weight=failure_weight,
              module=module,
              contact_dist=contact_dist,
              agent_error=agent_error,
              block_error=block_error,
              angle_error=angle_error,
            )
          )
          current_state = sim_next
          current_obs = observation_text_from_state(
            current_state,
            episode_index=start.episode_index,
            step=step_idx + 1,
          )
  finally:
    env.close()

  return failures


def failures_to_weighted_transitions(failures: Sequence[MinedFailure]) -> List[WeightedTransition]:
  return [
    WeightedTransition(
      observation=item.observation,
      action=item.action,
      next_observation=item.next_observation,
      weight=item.weight,
    )
    for item in failures
  ]


def write_failure_jsonl(failures: Sequence[MinedFailure], path: Path) -> None:
  record = {
    "metadata": {
      "env": "pusht",
      "source": "planning_failure_mining",
      "item_id": "planning-failures",
      "split": "train",
    },
    "transitions": [
      {
        "observation": item.observation,
        "action": item.action,
        "next_observation": item.next_observation,
        "reward": 0.0,
        "done": False,
        "metadata": {
          "weight": item.weight,
          "model_goal_distance": item.model_goal_distance,
          "sim_goal_distance": item.sim_goal_distance,
          "regret": item.regret,
          "module": item.module,
          "contact_dist": item.contact_dist,
          "agent_error": item.agent_error,
          "block_error": item.block_error,
          "angle_error": item.angle_error,
          "source": "planning_failure_mining",
        },
      }
      for item in failures
    ],
  }
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")


def summarize_failures(failures: Sequence[MinedFailure]) -> Dict[str, Any]:
  if not failures:
    return {"count": 0, "mean_regret": 0.0, "max_regret": 0.0, "modules": {}}
  regrets = [item.regret for item in failures]
  modules: Dict[str, int] = {}
  for item in failures:
    modules[item.module] = modules.get(item.module, 0) + 1
  return {
    "count": len(failures),
    "mean_regret": float(sum(regrets) / len(regrets)),
    "max_regret": float(max(regrets)),
    "mean_model_goal_distance": float(sum(item.model_goal_distance for item in failures) / len(failures)),
    "mean_sim_goal_distance": float(sum(item.sim_goal_distance for item in failures) / len(failures)),
    "modules": dict(sorted(modules.items())),
  }
