"""Simulator-backed evaluation for induced PushT world models."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

from .lewm_sampling import (
    belief_from_state_vector,
    graph_text_from_state,
    state_field_errors,
    state_vector_from_payload,
)
from .patchworld_adapter import apply_pusht_patchworld_adapter

@dataclass(frozen=True)
class SimStepResult:
    raw_index: int
    next_raw_index: int
    action: str
    hdf5_replay_max_error: float
    model_agent_position_error: float
    model_block_position_error: float
    model_block_angle_error: float
    model_state_match: bool
    predict_success: bool


def _load_jsonl(path: Path) -> List[Mapping[str, Any]]:
    records: List[Mapping[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _transition_action_block(
    transition: Mapping[str, Any],
    frameskip: int,
    *,
    actions_h5: Any = None,
) -> np.ndarray:
    metadata = transition.get("metadata", {})
    if isinstance(metadata, Mapping) and metadata.get("action_block"):
        block = np.asarray(metadata["action_block"], dtype=np.float32)
        if block.shape == (frameskip, 2):
            return block
    values = transition.get("action_values")
    if isinstance(values, list) and len(values) == frameskip * 2:
        return np.asarray(values, dtype=np.float32).reshape(frameskip, 2)
    if actions_h5 is not None and isinstance(metadata, Mapping) and "raw_index" in metadata:
        raw_index = int(metadata["raw_index"])
        return np.asarray(actions_h5[raw_index : raw_index + frameskip], dtype=np.float32)
    raise ValueError("Transition is missing LeWM-aligned action block metadata.")


def _transition_state_index(transition: Mapping[str, Any], key: str) -> int:
    metadata = transition.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ValueError(f"Transition missing metadata for {key}")
    if key == "raw_index":
        return int(metadata["raw_index"])
    if key == "next_raw_index":
        return int(metadata["next_raw_index"])
    raise KeyError(key)


def _predict_next_graph(model: Any, observation_text: str, action: str) -> Dict[str, Any]:
    belief = model.init_belief_from_observation(observation_text)
    predicted = model.predict_belief(belief, action)
    if isinstance(predicted, dict) and predicted.get("objects"):
        return predicted
    readout = model.readout_observation(predicted, action)
    parsed = model.parse_observation(readout)
    if isinstance(parsed, dict) and parsed.get("objects"):
        return parsed
    return predicted if isinstance(predicted, dict) else {}


def _max_state_error(sim_state: Sequence[float], target_state: Sequence[float]) -> float:
    sim = np.asarray(sim_state, dtype=np.float64)
    target = np.asarray(target_state, dtype=np.float64)
    angle_err = abs(sim[4] - target[4])
    angle_err = min(angle_err, abs(2 * math.pi - angle_err))
    pos_err = np.linalg.norm(sim[:4] - target[:4])
    vel_err = np.linalg.norm(sim[5:7] - target[5:7]) if sim.shape[0] > 6 else 0.0
    return float(max(pos_err, vel_err, angle_err))


def _make_pusht_env():
    import gymnasium as gym
    import stable_worldmodel.envs  # noqa: F401

    return gym.make("swm/PushT-v1", render_mode="rgb_array")


def _rollout_sim(env: Any, state: Sequence[float], action_block: np.ndarray) -> np.ndarray:
    env.reset(options={"state": np.asarray(state, dtype=np.float64)})
    for action in action_block:
        env.step(np.asarray(action, dtype=np.float32))
    return env.unwrapped._get_obs()


def resolve_generated_model_path(results_dir: Path, source: str, model_path: Optional[str]) -> Path:
    if model_path:
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"Model path not found: {path}")
        return path
    model_dir = results_dir / source / "generated_models"
    candidates = sorted(model_dir.glob("*.py"), key=lambda item: item.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No generated models found under {model_dir}")
    return candidates[-1]


def load_induced_model(model_path: Path) -> Any:
    apply_pusht_patchworld_adapter()
    from patchworld.worldmodel_validator import load_model_from_code

    code = model_path.read_text(encoding="utf-8")
    model, error = load_model_from_code(code)
    if model is None:
        raise RuntimeError(f"Failed to load generated model: {error}")
    return model


def evaluate_sim_steps(
    *,
    model: Any,
    states_h5: Any,
    trajectories_jsonl: Path,
    actions_h5: Any = None,
    frameskip: int = 5,
    max_transitions: int = 0,
    position_tolerance: float = 5.0,
    velocity_tolerance: float = 25.0,
) -> Dict[str, Any]:
    records = _load_jsonl(trajectories_jsonl)
    results: List[SimStepResult] = []
    env = _make_pusht_env()

    try:
        for record in records:
            for transition in record.get("transitions", []):
                if not isinstance(transition, Mapping):
                    continue
                raw_index = _transition_state_index(transition, "raw_index")
                next_raw_index = _transition_state_index(transition, "next_raw_index")
                action = str(transition.get("action", ""))
                action_block = _transition_action_block(transition, frameskip, actions_h5=actions_h5)

                start_state = np.asarray(states_h5[raw_index], dtype=np.float64)
                target_state = np.asarray(states_h5[next_raw_index], dtype=np.float64)
                sim_state = _rollout_sim(env, start_state, action_block)
                hdf5_replay_error = _max_state_error(sim_state, target_state)

                observation_text = transition.get("observation")
                if not observation_text:
                    observation_text = graph_text_from_state(
                        start_state,
                        step=int(transition.get("metadata", {}).get("step", 0)),
                        metadata={"raw_index": raw_index},
                    )

                predict_success = True
                try:
                    predicted_graph = _predict_next_graph(model, str(observation_text), action)
                except Exception:
                    predict_success = False
                    predicted_graph = {}

                target_graph = belief_from_state_vector(sim_state, step=int(transition.get("metadata", {}).get("step", 0)) + 1)
                errors = state_field_errors(
                    predicted_graph,
                    sim_state,
                    tolerance=position_tolerance,
                    velocity_tolerance=velocity_tolerance,
                )
                results.append(
                    SimStepResult(
                        raw_index=raw_index,
                        next_raw_index=next_raw_index,
                        action=action,
                        hdf5_replay_max_error=hdf5_replay_error,
                        model_agent_position_error=errors["agent_position_error"],
                        model_block_position_error=errors["block_position_error"],
                        model_block_angle_error=errors["block_angle_error"],
                        model_state_match=bool(errors["state_match"]),
                        predict_success=predict_success,
                    )
                )
                if max_transitions > 0 and len(results) >= max_transitions:
                    break
            if max_transitions > 0 and len(results) >= max_transitions:
                break
    finally:
        env.close()

    if not results:
        raise SystemExit(f"No transitions evaluated from {trajectories_jsonl}")

    def _mean(values: Iterable[float]) -> float:
        values = list(values)
        return sum(values) / max(1, len(values))

    return {
        "trajectories_jsonl": str(trajectories_jsonl),
        "n_transitions": len(results),
        "predict_success_rate": _mean(1.0 if item.predict_success else 0.0 for item in results),
        "hdf5_replay_max_error_mean": _mean(item.hdf5_replay_max_error for item in results),
        "model_agent_position_error_mean": _mean(item.model_agent_position_error for item in results),
        "model_block_position_error_mean": _mean(item.model_block_position_error for item in results),
        "model_block_angle_error_mean": _mean(item.model_block_angle_error for item in results),
        "model_state_match_rate": _mean(1.0 if item.model_state_match else 0.0 for item in results),
        "position_tolerance": position_tolerance,
        "velocity_tolerance": velocity_tolerance,
        "frameskip": frameskip,
    }


def compare_model_to_oracle_graphs(
    *,
    model: Any,
    trajectories_jsonl: Path,
    max_transitions: int = 0,
    position_tolerance: float = 5.0,
    velocity_tolerance: float = 25.0,
) -> Dict[str, Any]:
    """Compare model predictions against oracle next graphs already stored in JSONL."""

    records = _load_jsonl(trajectories_jsonl)
    agent_errors: List[float] = []
    block_errors: List[float] = []
    angle_errors: List[float] = []
    matches: List[float] = []
    predict_ok: List[float] = []

    count = 0
    for record in records:
        for transition in record.get("transitions", []):
            if not isinstance(transition, Mapping):
                continue
            metadata = transition.get("metadata", {})
            if not isinstance(metadata, Mapping) or "next_observation_graph" not in metadata:
                continue
            action = str(transition.get("action", ""))
            observation_text = str(transition.get("observation", ""))
            oracle_next = metadata["next_observation_graph"]
            try:
                predicted_graph = _predict_next_graph(model, observation_text, action)
                predict_ok.append(1.0)
            except Exception:
                predict_ok.append(0.0)
                predicted_graph = {}
            pred_errors = state_field_errors(
                predicted_graph,
                oracle_next,
                tolerance=position_tolerance,
                velocity_tolerance=velocity_tolerance,
            )
            agent_errors.append(pred_errors["agent_position_error"])
            block_errors.append(pred_errors["block_position_error"])
            angle_errors.append(pred_errors["block_angle_error"])
            matches.append(pred_errors["state_match"])
            count += 1
            if max_transitions > 0 and count >= max_transitions:
                break
        if max_transitions > 0 and count >= max_transitions:
            break

    if count == 0:
        raise SystemExit("No oracle transitions with next_observation_graph metadata found.")

    return {
        "trajectories_jsonl": str(trajectories_jsonl),
        "n_transitions": count,
        "predict_success_rate": sum(predict_ok) / count,
        "model_agent_position_error_mean": sum(agent_errors) / count,
        "model_block_position_error_mean": sum(block_errors) / count,
        "model_block_angle_error_mean": sum(angle_errors) / count,
        "model_state_match_rate": sum(matches) / count,
        "position_tolerance": position_tolerance,
        "velocity_tolerance": velocity_tolerance,
    }


def _state_from_graph_json(graph: Mapping[str, Any]) -> List[float]:
    return state_vector_from_payload(graph)
