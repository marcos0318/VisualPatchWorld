"""Cross-environment planning evaluation for induced VisualPatchWorld models."""

from __future__ import annotations

import gc
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .hdf5 import episode_slices, require_h5py, resolve_dataset_path
from .lewm_env_specs import LeWMEnvSpec, get_env_spec, make_gym_env, sim_planning_render_mode
from .lewm_graph import (
    action_string_from_vector,
    goal_distance,
    goal_row_from_start,
    graph_text_from_row,
    is_success,
    reset_env_from_rows,
    row_dict_from_h5,
    sim_row_from_obs,
    sim_step_with_info,
)
from .planning_eval import EvalStart, select_lewm_eval_starts
from .sim_eval import _predict_next_graph


def _global_index(h5: Mapping[str, Any], episode_index: int, local_step: int) -> int:
    offset = int(h5["ep_offset"][episode_index])
    return offset + local_step


def _rollout_model_macro(
    model: Any,
    observation_text: str,
    action_matrix: np.ndarray,
    spec: LeWMEnvSpec,
) -> str:
    obs = observation_text
    for action in np.asarray(action_matrix):
        action_str = action_string_from_vector(action, spec)
        predicted = _predict_next_graph_for_planning(model, obs, action_str, spec)
        obs = model.readout_observation(predicted, action_str)
    return obs


def _rollout_model_row(
    model: Any,
    observation_text: str,
    action_matrix: np.ndarray,
    spec: LeWMEnvSpec,
    template_row: Mapping[str, Any],
) -> Dict[str, Any]:
    obs_for_init = observation_text
    try:
        belief = model.init_belief_from_observation(obs_for_init)
    except Exception:
        if spec.name != "pusht":
            raise
        belief = model.init_belief_from_observation(_strip_nonnumeric_relation_lines(obs_for_init))
    for action in np.asarray(action_matrix):
        action_str = action_string_from_vector(action, spec)
        predicted = model.predict_belief(belief, action_str)
        if isinstance(predicted, Mapping) and predicted.get("objects"):
            belief = predicted
            continue
        readout = model.readout_observation(predicted, action_str)
        try:
            parsed = model.parse_observation(readout)
        except Exception:
            if spec.name != "pusht":
                raise
            parsed = model.parse_observation(_strip_nonnumeric_relation_lines(readout))
        belief = parsed if isinstance(parsed, Mapping) and parsed.get("objects") else predicted
    return _model_row_from_belief(belief, template_row, spec.name)


def _predict_next_graph_for_planning(model: Any, observation_text: str, action: str, spec: LeWMEnvSpec) -> Dict[str, Any]:
    try:
        return _predict_next_graph(model, observation_text, action)
    except Exception:
        if spec.name != "pusht":
            raise
        return _predict_next_graph(model, _strip_nonnumeric_relation_lines(observation_text), action)


def _strip_nonnumeric_relation_lines(obs_text: str) -> str:
    lines: List[str] = []
    in_relations = False
    for raw_line in str(obs_text or "").splitlines():
        stripped = raw_line.strip()
        if stripped == "Relations:":
            in_relations = True
            lines.append(raw_line)
            continue
        if stripped == "Metadata:" or stripped.startswith("Metadata:"):
            in_relations = False
            lines.append(raw_line)
            continue
        if in_relations and stripped.startswith("-") and "=" in stripped:
            value = stripped.rsplit("=", 1)[1].strip()
            try:
                float(value)
            except ValueError:
                continue
        lines.append(raw_line)
    return "\n".join(lines)


def _model_row_from_obs_text(model: Any, obs_text: str, template_row: Mapping[str, Any], env_name: str) -> Dict[str, Any]:
    # Reconstruct row from parsed belief - use graph fields
    spec = get_env_spec(env_name)
    try:
        belief = model.parse_observation(obs_text)
    except Exception:
        if spec.name != "pusht":
            raise
        belief = model.parse_observation(_strip_nonnumeric_relation_lines(obs_text))
    return _model_row_from_belief(belief, template_row, env_name)


def _model_row_from_belief(belief: Any, template_row: Mapping[str, Any], env_name: str) -> Dict[str, Any]:
    spec = get_env_spec(env_name)
    if spec.name == "pusht":
        objects = belief.get("objects", {}) if isinstance(belief, Mapping) else {}
        agent = _find_belief_object(objects, {"agent", "circular_pusher", "pusher"})
        block = _find_belief_object(objects, {"block", "t_block", "tee", "t"})
        state = np.asarray(template_row["state"], dtype=np.float64).copy()
        state[0] = _object_float(agent, "x", state[0])
        state[1] = _object_float(agent, "y", state[1])
        state[2] = _object_float(block, "x", state[2])
        state[3] = _object_float(block, "y", state[3])
        state[4] = _object_float(block, "angle", state[4])
        if len(state) > 5:
            state[5] = _object_float(agent, "vx", state[5])
        if len(state) > 6:
            state[6] = _object_float(agent, "vy", state[6])
        return {"state": state}
    if spec.name == "tworoom":
        objects = belief.get("objects", {}) if isinstance(belief, Mapping) else {}
        agent = objects.get("agent", {}) if isinstance(objects, Mapping) else {}
        target = objects.get("target", {}) if isinstance(objects, Mapping) else {}
        obs = np.asarray(template_row["observation"], dtype=np.float64).copy()
        obs[0] = float(agent.get("x", obs[0]))
        obs[1] = float(agent.get("y", obs[1]))
        obs[2] = float(target.get("x", obs[2]))
        obs[3] = float(target.get("y", obs[3]))
        return {"observation": obs}
    if spec.name == "cube":
        row = dict(template_row)
        effector = belief.get("objects", {}).get("effector", {})
        block = belief.get("objects", {}).get("block", {})
        row["effector_pos"] = np.array([effector.get("x", 0), effector.get("y", 0), effector.get("z", 0)], dtype=np.float64)
        row["block_pos"] = np.array([block.get("x", 0), block.get("y", 0), block.get("z", 0)], dtype=np.float64)
        return row
    if spec.name == "reacher":
        row = dict(template_row)
        arm = belief.get("objects", {}).get("arm", {})
        finger = belief.get("objects", {}).get("finger", {})
        target = belief.get("objects", {}).get("target", {})
        row["qpos"] = np.array([arm.get("x", 0), arm.get("y", 0)], dtype=np.float64)
        row["finger_pos"] = np.array([finger.get("x", 0), finger.get("y", 0)], dtype=np.float64)
        row["target_pos"] = np.array([target.get("x", 0), target.get("y", 0)], dtype=np.float64)
        return row
    raise ValueError(env_name)


def _find_belief_object(objects: Any, aliases: set[str]) -> Mapping[str, Any]:
    if not isinstance(objects, Mapping):
        return {}
    normalized = {alias.lower() for alias in aliases}
    for key, obj in objects.items():
        if str(key).lower() in normalized and isinstance(obj, Mapping):
            return obj
    for obj in objects.values():
        if not isinstance(obj, Mapping):
            continue
        labels = {
            str(obj.get("id", "")).lower(),
            str(obj.get("type", "")).lower(),
            str(obj.get("category", "")).lower(),
        }
        if labels & normalized:
            return obj
    return {}


def _object_float(obj: Mapping[str, Any], key: str, default: float) -> float:
    value: Any = obj.get(key, default) if isinstance(obj, Mapping) else default
    if key in {"x", "y"} and isinstance(obj, Mapping) and isinstance(obj.get("position"), Mapping):
        value = obj["position"].get(key, value)
    if key in {"vx", "vy"} and isinstance(obj, Mapping) and isinstance(obj.get("velocity"), Mapping):
        vel_key = "x" if key == "vx" else "y"
        value = obj["velocity"].get(vel_key, value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def plan_cem_general(
    model: Any,
    observation_text: str,
    goal_row: Mapping[str, Any],
    template_row: Mapping[str, Any],
    spec: LeWMEnvSpec,
    *,
    horizon: int,
    num_samples: int,
    cem_iters: int,
    elite_frac: float,
    seed: int,
    action_low: float,
    action_high: float,
    sim_env: Any = None,
    sim_start_row: Optional[Mapping[str, Any]] = None,
    sim_verify_frac: float = 0.0,
    frameskip: int = 1,
    metric: str = "default",
    init_mean: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float, int]:
    rng = np.random.default_rng(seed)
    action_dim = spec.action_dim
    if init_mean is not None and np.asarray(init_mean).shape == (horizon, action_dim):
        mean = np.asarray(init_mean, dtype=np.float64).copy()
        std = np.full((horizon, action_dim), 0.25 * (action_high - action_low), dtype=np.float64)
    else:
        mean = np.zeros((horizon, action_dim), dtype=np.float64)
        std = np.full((horizon, action_dim), 0.5 * (action_high - action_low), dtype=np.float64)
    elite_count = max(1, int(num_samples * elite_frac))
    best_score = float("inf")
    best_values = mean.copy()
    failed = 0
    verify_count = max(0, int(math.ceil(num_samples * sim_verify_frac))) if sim_verify_frac > 0 else 0

    for _ in range(max(1, cem_iters)):
        candidates = rng.normal(loc=mean, scale=std, size=(num_samples, horizon, action_dim))
        candidates = np.clip(candidates, action_low, action_high)
        candidates[0] = mean
        scores = np.full((num_samples,), np.inf, dtype=np.float64)
        for sample_idx, candidate in enumerate(candidates):
            try:
                cur_row = _rollout_model_row(model, observation_text, candidate, spec, template_row)
                scores[sample_idx] = goal_distance(spec.name, goal_row, cur_row, metric=metric)
            except Exception:
                failed += 1
        if verify_count > 0 and sim_env is not None and sim_start_row is not None:
            ranked = np.argsort(scores)
            for sample_idx in ranked[:verify_count]:
                if not np.isfinite(scores[sample_idx]):
                    continue
                try:
                    reset_env_from_rows(sim_env, spec, sim_start_row, goal_row)
                    cur_row = dict(sim_start_row)
                    for action in candidates[sample_idx]:
                        info: Mapping[str, Any] = {}
                        for _ in range(max(1, frameskip)):
                            obs, info = sim_step_with_info(sim_env, spec, action)
                        if spec.name == "tworoom":
                            cur_row = sim_row_from_obs(spec.name, obs, template_row=cur_row)
                        elif spec.name in ("cube", "reacher"):
                            cur_row = sim_row_from_obs(spec.name, obs, template_row=cur_row, info=info)
                    scores[sample_idx] = goal_distance(spec.name, goal_row, cur_row, metric=metric)
                except Exception:
                    failed += 1
        finite = np.isfinite(scores)
        if not finite.any():
            continue
        elite_indices = np.argsort(scores)[:elite_count]
        elites = candidates[elite_indices]
        mean = elites.mean(axis=0)
        std = np.maximum(elites.std(axis=0), 1e-3)
        if float(scores[elite_indices[0]]) < best_score:
            best_score = float(scores[elite_indices[0]])
            best_values = candidates[int(elite_indices[0])]
    return best_values, best_score, failed


def evaluate_dynamics_on_jsonl(model: Any, trajectories_jsonl: Path, env_name: str, max_transitions: int = 500) -> Dict[str, Any]:
    spec = get_env_spec(env_name)
    errors: List[float] = []
    count = 0
    with trajectories_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            for transition in record.get("transitions", []):
                obs_text = str(transition.get("observation", ""))
                action = str(transition.get("action", ""))
                next_text = str(transition.get("next_observation", ""))
                try:
                    pred_text = model.readout_observation(_predict_next_graph(model, obs_text, action), action)
                    pred = model.parse_observation(pred_text)
                    tgt = model.parse_observation(next_text)
                    _, pred_vals = _state_values_from_belief(pred)
                    _, tgt_vals = _state_values_from_belief(tgt)
                    if pred_vals and tgt_vals:
                        err = float(np.linalg.norm(np.asarray(pred_vals) - np.asarray(tgt_vals)))
                        errors.append(err)
                except Exception:
                    pass
                count += 1
                if max_transitions > 0 and count >= max_transitions:
                    break
            if max_transitions > 0 and count >= max_transitions:
                break
    return {
        "env": env_name,
        "n_transitions": count,
        "mean_state_error": float(np.mean(errors)) if errors else None,
        "trajectories_jsonl": str(trajectories_jsonl),
    }


def _state_values_from_belief(belief: Mapping[str, Any]) -> Tuple[List[str], List[float]]:
    keys: List[str] = []
    values: List[float] = []
    for object_id in sorted(belief.get("objects", {})):
        obj = belief["objects"][object_id]
        for field in ("x", "y", "z", "angle"):
            if field in obj:
                keys.append(f"{object_id}.{field}")
                values.append(float(obj[field]))
    return keys, values


def evaluate_general_planning(
    *,
    model: Any,
    env_name: str,
    dataset: Optional[str] = None,
    stablewm_home: Optional[str] = None,
    num_eval: int = 50,
    goal_offset_steps: int = 25,
    eval_budget: int = 50,
    frameskip: int = 0,
    horizon: int = 5,
    num_samples: int = 300,
    cem_iters: int = 10,
    elite_frac: float = 0.1,
    receding_horizon: int = 5,
    sim_verify_frac: float = 0.0,
    seed: int = 42,
    action_low: float = -1.0,
    action_high: float = 1.0,
    observation_mode: str = "oracle",
    success_metric: str = "default",
    warm_start: bool = False,
    induced_planner: str = "cem",
    beam_width: int = 16,
) -> Dict[str, Any]:
    spec = get_env_spec(env_name)
    dataset = dataset or spec.dataset
    frameskip = frameskip or spec.frameskip
    dataset_path = resolve_dataset_path(dataset, stablewm_home)
    h5py = require_h5py()

    if not spec.can_plan_in_sim:
        val_path = Path(dataset_path).parent / "data" / f"{spec.name}_oracle_traj_val.jsonl"
        if not val_path.is_file():
            val_path = Path(f"artifacts/visualpatchworld/data/{spec.name}_oracle_traj_val.jsonl")
        dyn = evaluate_dynamics_on_jsonl(model, val_path, env_name)
        return {
            "env": env_name,
            "planner": "dynamics_only",
            "planning_note": spec.planning_note,
            "dynamics_eval": dyn,
            "success_rate": None,
            "mean_goal_distance": None,
        }

    env = make_gym_env(spec)
    successes: List[float] = []
    distances: List[float] = []
    failed_model_rollouts = 0
    tool_extraction_failures = 0

    try:
        with h5py.File(dataset_path, "r") as h5:
            eval_starts = select_lewm_eval_starts(
                h5,
                num_eval=num_eval,
                goal_offset_steps=goal_offset_steps,
                seed=seed,
            )
            for trial_id, start in enumerate(eval_starts):
                print(
                    f"[visualpatchworld][general-plan] env={env_name} seed={seed} "
                    f"trial={trial_id + 1}/{len(eval_starts)} "
                    f"obs={observation_mode} sim_verify={sim_verify_frac}",
                    flush=True,
                )
                start_global = _global_index(h5, start.episode_index, start.start_step)
                start_row = row_dict_from_h5(h5, start_global, spec)
                goal_row = goal_row_from_start(h5, spec, start.episode_index, start.start_step, goal_offset_steps)
                reset_env_from_rows(env, spec, start_row, goal_row)
                if observation_mode == "tool_live":
                    from .tool_observation import observation_text_from_env_tool_render

                    obs_text, ok = observation_text_from_env_tool_render(
                        env,
                        spec,
                        template_row=start_row,
                        goal_row=goal_row,
                        episode_index=start.episode_index,
                        step=0,
                    )
                    if not ok:
                        tool_extraction_failures += 1
                else:
                    obs_text = graph_text_from_row(
                        spec.name, start_row, source="oracle", step=0, metadata={"episode_index": start.episode_index}
                    )

                current_row = dict(start_row)
                current_obs = obs_text
                last_tool_obs = current_obs if observation_mode == "tool_live" else None
                mpc_warm: Optional[np.ndarray] = None
                macro_budget = max(1, eval_budget // frameskip)
                steps_taken = 0

                while steps_taken < macro_budget:
                    # Plan over full horizon; execute only receding macros (PushT-style MPC).
                    plan_horizon = min(horizon, macro_budget - steps_taken)
                    init_mean = None
                    if warm_start and mpc_warm is not None and mpc_warm.shape[1] == spec.action_dim:
                        init_mean = mpc_warm[:plan_horizon]
                    if induced_planner in ("discrete_beam", "beam"):
                        plan_values, _, failed = plan_discrete_beam_general(
                            model,
                            current_obs,
                            goal_row,
                            current_row,
                            spec,
                            horizon=plan_horizon,
                            seed=seed + trial_id + steps_taken,
                            action_low=action_low,
                            action_high=action_high,
                            metric=success_metric,
                            beam_width=beam_width,
                            start_row=current_row,
                        )
                    elif induced_planner in ("library_shoot", "expert_shoot"):
                        plan_values, _, failed = plan_library_shoot_general(
                            model,
                            current_obs,
                            goal_row,
                            current_row,
                            spec,
                            horizon=plan_horizon,
                            num_samples=num_samples,
                            seed=seed + trial_id + steps_taken,
                            action_low=action_low,
                            action_high=action_high,
                            metric=success_metric,
                            frameskip=frameskip,
                            start_row=current_row,
                            sim_env=env if sim_verify_frac > 0 else None,
                            sim_start_row=current_row if sim_verify_frac > 0 else None,
                            sim_verify_frac=sim_verify_frac,
                        )
                    else:
                        plan_values, _, failed = plan_cem_general(
                            model,
                            current_obs,
                            goal_row,
                            current_row,
                            spec,
                            horizon=plan_horizon,
                            num_samples=num_samples,
                            cem_iters=cem_iters,
                            elite_frac=elite_frac,
                            seed=seed + trial_id + steps_taken,
                            action_low=action_low,
                            action_high=action_high,
                            sim_env=env if sim_verify_frac > 0 else None,
                            sim_start_row=current_row if sim_verify_frac > 0 else None,
                            sim_verify_frac=sim_verify_frac,
                            frameskip=frameskip,
                            metric=success_metric,
                            init_mean=init_mean,
                        )
                    failed_model_rollouts += failed
                    # Hybrid sim-verify (and any planner that mutates `env`) leaves the
                    # simulator at the last candidate's terminal state; restore the live
                    # MPC state before executing the chosen plan.
                    reset_env_from_rows(env, spec, current_row, goal_row)
                    execute_macro = min(len(plan_values), macro_budget - steps_taken, receding_horizon)

                    for action in plan_values[:execute_macro]:
                        obs = None
                        info: Mapping[str, Any] = {}
                        for _ in range(frameskip):
                            obs, info = sim_step_with_info(env, spec, action)
                        if spec.name == "tworoom":
                            current_row = sim_row_from_obs(spec.name, obs, template_row=current_row)
                        elif spec.name == "cube":
                            current_row = sim_row_from_obs(spec.name, obs, template_row=current_row, info=info)
                        elif spec.name == "reacher":
                            current_row = sim_row_from_obs(spec.name, obs, template_row=current_row, info=info)
                        elif spec.name == "pusht":
                            current_row = sim_row_from_obs(spec.name, obs, template_row=current_row)
                        steps_taken += 1
                        if steps_taken >= macro_budget:
                            break
                    if warm_start and execute_macro < len(plan_values):
                        mpc_warm = _shift_warm_mean(plan_values, execute_macro, horizon, spec.action_dim)
                    else:
                        mpc_warm = None
                    if observation_mode == "tool_live":
                        from .tool_observation import observation_text_from_env_tool_render

                        current_obs, ok = observation_text_from_env_tool_render(
                            env,
                            spec,
                            template_row=current_row,
                            goal_row=goal_row,
                            episode_index=start.episode_index,
                            step=steps_taken,
                            fallback_observation_text=last_tool_obs,
                        )
                        if not ok:
                            tool_extraction_failures += 1
                        last_tool_obs = current_obs
                    else:
                        current_obs = graph_text_from_row(
                            spec.name, current_row, source="oracle", step=steps_taken
                        )
                    if is_success(spec.name, goal_row, current_row, metric=success_metric):
                        break

                dist = goal_distance(spec.name, goal_row, current_row, metric=success_metric)
                distances.append(dist)
                successes.append(
                    1.0 if is_success(spec.name, goal_row, current_row, metric=success_metric) else 0.0
                )
    finally:
        env.close()

    return {
        "env": env_name,
        "dataset": str(dataset_path),
        "observation_mode": observation_mode,
        "num_eval": num_eval,
        "success_rate": 100.0 * float(np.mean(successes)) if successes else 0.0,
        "mean_goal_distance": float(np.mean(distances)) if distances else None,
        "episode_successes": successes,
        "failed_model_rollouts": failed_model_rollouts,
        "tool_extraction_failures": tool_extraction_failures,
        "frameskip": frameskip,
        "horizon": horizon,
        "receding_horizon": receding_horizon,
        "num_samples": num_samples,
        "cem_iters": cem_iters,
        "sim_verify_frac": sim_verify_frac,
        "seed": seed,
        "success_metric": success_metric,
        "warm_start": warm_start,
    }


def _sim_rollout_score(
    env: Any,
    spec: LeWMEnvSpec,
    goal_row: Mapping[str, Any],
    start_row: Mapping[str, Any],
    action_matrix: np.ndarray,
    *,
    frameskip: int,
    metric: str,
) -> float:
    reset_env_from_rows(env, spec, start_row, goal_row)
    cur_row = dict(start_row)
    for action in np.asarray(action_matrix):
        info: Mapping[str, Any] = {}
        for _ in range(max(1, frameskip)):
            obs, info = sim_step_with_info(env, spec, action)
        if spec.name == "tworoom":
            cur_row = sim_row_from_obs(spec.name, obs, template_row=cur_row)
        elif spec.name in ("cube", "reacher"):
            cur_row = sim_row_from_obs(spec.name, obs, template_row=cur_row, info=info)
        elif spec.name == "pusht":
            cur_row = sim_row_from_obs(spec.name, obs, template_row=cur_row)
    return goal_distance(spec.name, goal_row, cur_row, metric=metric)


def _shift_warm_mean(plan_values: np.ndarray, executed: int, horizon: int, action_dim: int) -> np.ndarray:
    if executed <= 0 or executed >= len(plan_values):
        return np.zeros((horizon, action_dim), dtype=np.float64)
    remaining = plan_values[executed:]
    pad = np.zeros((max(0, horizon - len(remaining)), action_dim), dtype=np.float64)
    return np.concatenate([remaining, pad], axis=0)[:horizon]


def _score_sim_candidates(
    env: Any,
    spec: LeWMEnvSpec,
    goal_row: Mapping[str, Any],
    start_row: Mapping[str, Any],
    candidates: np.ndarray,
    *,
    frameskip: int,
    metric: str,
    rollout_batch_size: int,
    env_factory: Optional[Any] = None,
    env_recycle_every: int = 0,
) -> Tuple[np.ndarray, int, Any]:
    """Score CEM/MPPI candidate action sequences against the true simulator."""

    scores = np.full((len(candidates),), np.inf, dtype=np.float64)
    failed = 0
    rollouts = 0
    batch_size = max(1, rollout_batch_size or len(candidates))

    for batch_start in range(0, len(candidates), batch_size):
        batch_end = min(batch_start + batch_size, len(candidates))
        for sample_idx in range(batch_start, batch_end):
            if env_factory is not None and env_recycle_every > 0 and rollouts > 0 and rollouts % env_recycle_every == 0:
                try:
                    env.close()
                except Exception:
                    pass
                env = env_factory()
            try:
                scores[sample_idx] = _sim_rollout_score(
                    env,
                    spec,
                    goal_row,
                    start_row,
                    candidates[sample_idx],
                    frameskip=frameskip,
                    metric=metric,
                )
            except Exception:
                failed += 1
            rollouts += 1
        gc.collect()
    return scores, failed, env


def _goal_vector_xy(env_name: str, start_row: Mapping[str, Any], goal_row: Mapping[str, Any]) -> Optional[np.ndarray]:
    """Return a 2D unit vector toward the planning goal, when available."""
    if env_name == "tworoom":
        cur = np.asarray(start_row["observation"][:2], dtype=np.float64)
        goal = np.asarray(goal_row["observation"][:2], dtype=np.float64)
    elif env_name == "pusht":
        cur = np.asarray(start_row["state"][:2], dtype=np.float64)
        goal = np.asarray(goal_row["state"][:2], dtype=np.float64)
    elif env_name == "cube":
        # Prefer moving the block toward its target pose; fall back to effector→block.
        block = np.asarray(start_row.get("block_pos", [0, 0, 0])[:2], dtype=np.float64)
        target = np.asarray(
            goal_row.get("goal_block_pos", goal_row.get("block_pos", [0, 0, 0]))[:2],
            dtype=np.float64,
        )
        delta = target - block
        if float(np.linalg.norm(delta)) < 1e-4:
            eff = np.asarray(start_row.get("effector_pos", [0, 0, 0])[:2], dtype=np.float64)
            delta = block - eff
        if float(np.linalg.norm(delta)) < 1e-8:
            return None
        return delta / (np.linalg.norm(delta) + 1e-8)
    else:
        return None
    delta = goal - cur
    nrm = float(np.linalg.norm(delta))
    if nrm < 1e-8:
        return None
    return delta / nrm


def _discrete_action_library(
    spec: LeWMEnvSpec,
    start_row: Mapping[str, Any],
    goal_row: Mapping[str, Any],
    *,
    action_low: float,
    action_high: float,
    n_dirs: int = 8,
    scales: Sequence[float] = (0.5, 1.0),
) -> List[np.ndarray]:
    """Directional / hold / goal-biased action atoms for discrete beam search."""
    dim = spec.action_dim
    acts: List[np.ndarray] = [np.zeros(dim, dtype=np.float64)]
    angs = np.linspace(0.0, 2.0 * np.pi, n_dirs, endpoint=False)
    for scale in scales:
        for ang in angs:
            vec = np.zeros(dim, dtype=np.float64)
            vec[0] = float(np.cos(ang) * scale)
            if dim >= 2:
                vec[1] = float(np.sin(ang) * scale)
            acts.append(np.clip(vec, action_low, action_high))
    goal_dir = _goal_vector_xy(spec.name, start_row, goal_row)
    if goal_dir is not None:
        for scale in scales:
            vec = np.zeros(dim, dtype=np.float64)
            vec[0] = float(goal_dir[0] * scale)
            if dim >= 2:
                vec[1] = float(goal_dir[1] * scale)
            acts.append(np.clip(vec, action_low, action_high))
    if spec.name == "cube" and dim >= 5:
        # Vertical approach / retreat and gripper extremes.
        for z, grip in ((-1.0, 0.0), (-0.5, 0.0), (0.5, 0.0), (0.0, -1.0), (0.0, 1.0)):
            vec = np.zeros(dim, dtype=np.float64)
            if goal_dir is not None:
                vec[0] = float(goal_dir[0])
                vec[1] = float(goal_dir[1])
            vec[2] = z
            vec[4] = grip
            acts.append(np.clip(vec, action_low, action_high))
    # Deduplicate
    uniq: List[np.ndarray] = []
    seen = set()
    for a in acts:
        key = tuple(np.round(a, 4).tolist())
        if key in seen:
            continue
        seen.add(key)
        uniq.append(a)
    return uniq


def plan_discrete_beam_sim_general(
    env: Any,
    spec: LeWMEnvSpec,
    goal_row: Mapping[str, Any],
    start_row: Mapping[str, Any],
    *,
    horizon: int,
    seed: int,
    action_low: float,
    action_high: float,
    frameskip: int = 1,
    metric: str = "default",
    beam_width: int = 16,
    n_dirs: int = 8,
    rollout_batch_size: int = 50,
    env_factory: Optional[Any] = None,
    env_recycle_every: int = 200,
    num_samples: int = 0,  # unused; kept for call-site compatibility
    init_mean: Optional[np.ndarray] = None,  # unused
    **_ignored: Any,
) -> Tuple[np.ndarray, float, int, Any]:
    """Discrete directional beam search on the true simulator (nav / short-horizon MPC)."""

    library = _discrete_action_library(
        spec,
        start_row,
        goal_row,
        action_low=action_low,
        action_high=action_high,
        n_dirs=n_dirs,
    )
    # Beam nodes: (score, action_sequence)
    beam: List[Tuple[float, List[np.ndarray]]] = [(0.0, [])]
    failed = 0
    for _depth in range(max(1, horizon)):
        candidates: List[np.ndarray] = []
        parents: List[List[np.ndarray]] = []
        for _, seq in beam:
            for atom in library:
                candidates.append(np.stack(seq + [atom], axis=0) if seq else atom.reshape(1, -1))
                parents.append(seq + [atom])
        # Cap expansion for large libraries
        if len(candidates) > max(beam_width * len(library), beam_width * 8):
            candidates = candidates[: beam_width * len(library)]
            parents = parents[: beam_width * len(library)]
        stacked = np.stack(candidates, axis=0)
        scores, iter_failed, env = _score_sim_candidates(
            env,
            spec,
            goal_row,
            start_row,
            stacked,
            frameskip=frameskip,
            metric=metric,
            rollout_batch_size=rollout_batch_size,
            env_factory=env_factory,
            env_recycle_every=env_recycle_every,
        )
        failed += iter_failed
        order = np.argsort(scores)
        next_beam: List[Tuple[float, List[np.ndarray]]] = []
        for idx in order[: max(1, beam_width)]:
            if not np.isfinite(scores[idx]):
                continue
            next_beam.append((float(scores[idx]), parents[int(idx)]))
        if not next_beam:
            break
        beam = next_beam
    best_score, best_seq = beam[0]
    if not best_seq:
        best_values = np.zeros((horizon, spec.action_dim), dtype=np.float64)
    else:
        best_values = np.stack(best_seq, axis=0)
        if len(best_values) < horizon:
            pad = np.zeros((horizon - len(best_values), spec.action_dim), dtype=np.float64)
            best_values = np.concatenate([best_values, pad], axis=0)
    return best_values, best_score, failed, env


def plan_discrete_beam_general(
    model: Any,
    observation_text: str,
    goal_row: Mapping[str, Any],
    template_row: Mapping[str, Any],
    spec: LeWMEnvSpec,
    *,
    horizon: int,
    seed: int,
    action_low: float,
    action_high: float,
    metric: str = "default",
    beam_width: int = 16,
    n_dirs: int = 8,
    start_row: Optional[Mapping[str, Any]] = None,
    **_ignored: Any,
) -> Tuple[np.ndarray, float, int]:
    """Discrete directional beam search scored by the induced world model."""

    row_for_goal = start_row or template_row
    library = _discrete_action_library(
        spec,
        row_for_goal,
        goal_row,
        action_low=action_low,
        action_high=action_high,
        n_dirs=n_dirs,
    )
    beam: List[Tuple[float, List[np.ndarray]]] = [(0.0, [])]
    failed = 0
    for _depth in range(max(1, horizon)):
        scored: List[Tuple[float, List[np.ndarray]]] = []
        for _, seq in beam:
            for atom in library:
                cand_seq = seq + [atom]
                matrix = np.stack(cand_seq, axis=0)
                try:
                    cur_row = _rollout_model_row(model, observation_text, matrix, spec, template_row)
                    score = goal_distance(spec.name, goal_row, cur_row, metric=metric)
                except Exception:
                    failed += 1
                    continue
                scored.append((float(score), cand_seq))
        scored.sort(key=lambda x: x[0])
        beam = scored[: max(1, beam_width)]
        if not beam:
            break
    best_score, best_seq = beam[0] if beam else (float("inf"), [])
    if not best_seq:
        best_values = np.zeros((horizon, spec.action_dim), dtype=np.float64)
    else:
        best_values = np.stack(best_seq, axis=0)
        if len(best_values) < horizon:
            pad = np.zeros((horizon - len(best_values), spec.action_dim), dtype=np.float64)
            best_values = np.concatenate([best_values, pad], axis=0)
    return best_values, best_score, failed


_MACRO_LIBRARY_CACHE: Dict[Tuple[str, int], np.ndarray] = {}


def _macro_action_library(
    spec: LeWMEnvSpec,
    *,
    frameskip: int,
    dataset_path: Optional[str] = None,
    max_macros: int = 20000,
    seed: int = 0,
) -> np.ndarray:
    """Mean-pooled expert action macros for library shooting."""
    key = (spec.name, int(frameskip))
    cached = _MACRO_LIBRARY_CACHE.get(key)
    if cached is not None:
        return cached
    h5py = require_h5py()
    path = resolve_dataset_path(dataset_path or spec.dataset)
    with h5py.File(path, "r") as h5:
        actions = np.asarray(h5["action"], dtype=np.float64)
    n = len(actions)
    stride = max(1, frameskip)
    starts = list(range(0, max(0, n - stride), stride))
    if len(starts) > max_macros:
        rng = np.random.default_rng(seed)
        starts = list(rng.choice(starts, size=max_macros, replace=False))
    macros = np.stack([actions[i : i + stride].mean(axis=0) for i in starts], axis=0)
    macros = np.clip(macros, -1.0, 1.0)
    _MACRO_LIBRARY_CACHE[key] = macros
    return macros


def plan_library_shoot_sim_general(
    env: Any,
    spec: LeWMEnvSpec,
    goal_row: Mapping[str, Any],
    start_row: Mapping[str, Any],
    *,
    horizon: int,
    num_samples: int,
    seed: int,
    action_low: float,
    action_high: float,
    frameskip: int = 1,
    metric: str = "default",
    rollout_batch_size: int = 50,
    env_factory: Optional[Any] = None,
    env_recycle_every: int = 200,
    noise_std: float = 0.05,
    dataset_path: Optional[str] = None,
    init_mean: Optional[np.ndarray] = None,  # unused
    **_ignored: Any,
) -> Tuple[np.ndarray, float, int, Any]:
    """Shoot noisy expert macros (+ goal-biased atom); pick best under true sim."""

    library = _macro_action_library(spec, frameskip=frameskip, dataset_path=dataset_path, seed=seed)
    rng = np.random.default_rng(seed)
    n = max(1, num_samples)
    idxs = rng.choice(len(library), size=n, replace=True)
    candidates = np.clip(
        library[idxs][:, None, :] + rng.normal(0.0, noise_std, size=(n, max(1, horizon), spec.action_dim)),
        action_low,
        action_high,
    )
    # Inject a goal-biased open-loop sequence in slot 0.
    gdir = _goal_vector_xy(spec.name, start_row, goal_row)
    if gdir is not None:
        atom = np.zeros(spec.action_dim, dtype=np.float64)
        atom[0] = float(gdir[0])
        if spec.action_dim >= 2:
            atom[1] = float(gdir[1])
        if spec.name == "cube" and spec.action_dim >= 5:
            atom[2] = -0.2
            atom[4] = 0.5
        candidates[0] = np.clip(np.tile(atom, (max(1, horizon), 1)), action_low, action_high)
    scores, failed, env = _score_sim_candidates(
        env,
        spec,
        goal_row,
        start_row,
        candidates,
        frameskip=frameskip,
        metric=metric,
        rollout_batch_size=rollout_batch_size,
        env_factory=env_factory,
        env_recycle_every=env_recycle_every,
    )
    best_idx = int(np.argmin(scores)) if np.isfinite(scores).any() else 0
    best_values = candidates[best_idx]
    best_score = float(scores[best_idx])
    # Light CEM polish around the best library macro (helps cube contact control).
    polish_samples = min(120, max(32, n // 4))
    if polish_samples >= 16 and spec.name in {"cube", "pusht"}:
        polished, polished_score, failed2, env = plan_cem_sim_general(
            env,
            spec,
            goal_row,
            start_row,
            horizon=max(1, horizon),
            num_samples=polish_samples,
            cem_iters=4,
            elite_frac=0.15,
            seed=seed + 17,
            action_low=action_low,
            action_high=action_high,
            frameskip=frameskip,
            init_mean=best_values,
            metric=metric,
            rollout_batch_size=rollout_batch_size,
            env_factory=env_factory,
            env_recycle_every=env_recycle_every,
            goal_bias_frac=0.25,
        )
        failed += failed2
        if polished_score < best_score:
            best_values, best_score = polished, polished_score
    return best_values, best_score, failed, env


def plan_library_shoot_general(
    model: Any,
    observation_text: str,
    goal_row: Mapping[str, Any],
    template_row: Mapping[str, Any],
    spec: LeWMEnvSpec,
    *,
    horizon: int,
    num_samples: int,
    seed: int,
    action_low: float,
    action_high: float,
    metric: str = "default",
    noise_std: float = 0.1,
    frameskip: int = 1,
    start_row: Optional[Mapping[str, Any]] = None,
    sim_env: Any = None,
    sim_start_row: Optional[Mapping[str, Any]] = None,
    sim_verify_frac: float = 0.0,
    **_ignored: Any,
) -> Tuple[np.ndarray, float, int]:
    """Shoot noisy expert macros scored by the induced world model."""

    library = _macro_action_library(spec, frameskip=frameskip, seed=seed)
    rng = np.random.default_rng(seed)
    n = max(1, num_samples)
    idxs = rng.choice(len(library), size=n, replace=True)
    candidates = np.clip(
        library[idxs][:, None, :] + rng.normal(0.0, noise_std, size=(n, max(1, horizon), spec.action_dim)),
        action_low,
        action_high,
    )
    row_for_goal = start_row or template_row
    gdir = _goal_vector_xy(spec.name, row_for_goal, goal_row)
    if gdir is not None:
        atom = np.zeros(spec.action_dim, dtype=np.float64)
        atom[0] = float(gdir[0])
        if spec.action_dim >= 2:
            atom[1] = float(gdir[1])
        if spec.name == "cube" and spec.action_dim >= 5:
            atom[2] = -0.2
            atom[4] = 0.5
        candidates[0] = np.clip(np.tile(atom, (max(1, horizon), 1)), action_low, action_high)
    scores = np.full((n,), np.inf, dtype=np.float64)
    failed = 0
    for sample_idx, cand in enumerate(candidates):
        try:
            cur_row = _rollout_model_row(model, observation_text, cand, spec, template_row)
            scores[sample_idx] = goal_distance(spec.name, goal_row, cur_row, metric=metric)
        except Exception:
            failed += 1
            continue
    verify_count = max(0, int(math.ceil(n * sim_verify_frac))) if sim_verify_frac > 0 else 0
    if verify_count > 0 and sim_env is not None and sim_start_row is not None:
        ranked = np.argsort(scores)
        for sample_idx in ranked[:verify_count]:
            if not np.isfinite(scores[sample_idx]):
                continue
            try:
                reset_env_from_rows(sim_env, spec, sim_start_row, goal_row)
                cur_row = dict(sim_start_row)
                for action in candidates[sample_idx]:
                    info: Mapping[str, Any] = {}
                    for _ in range(max(1, frameskip)):
                        obs, info = sim_step_with_info(sim_env, spec, action)
                    if spec.name == "tworoom":
                        cur_row = sim_row_from_obs(spec.name, obs, template_row=cur_row)
                    elif spec.name == "cube":
                        cur_row = sim_row_from_obs(spec.name, obs, template_row=cur_row, info=info)
                    elif spec.name == "reacher":
                        cur_row = sim_row_from_obs(spec.name, obs, template_row=cur_row, info=info)
                    elif spec.name == "pusht":
                        cur_row = sim_row_from_obs(spec.name, obs, template_row=cur_row)
                scores[sample_idx] = goal_distance(spec.name, goal_row, cur_row, metric=metric)
            except Exception:
                failed += 1
    best_idx = int(np.argmin(scores)) if np.isfinite(scores).any() else 0
    return candidates[best_idx], float(scores[best_idx]), failed


def plan_cem_sim_general(
    env: Any,
    spec: LeWMEnvSpec,
    goal_row: Mapping[str, Any],
    start_row: Mapping[str, Any],
    *,
    horizon: int,
    num_samples: int,
    cem_iters: int,
    elite_frac: float,
    seed: int,
    action_low: float,
    action_high: float,
    frameskip: int = 1,
    init_mean: Optional[np.ndarray] = None,
    metric: str = "default",
    rollout_batch_size: int = 50,
    env_factory: Optional[Any] = None,
    env_recycle_every: int = 200,
    goal_bias_frac: float = 0.0,
) -> Tuple[np.ndarray, float, int, Any]:
    """CEM planning with the true simulator as the rollout model."""

    rng = np.random.default_rng(seed)
    action_dim = spec.action_dim
    if init_mean is not None and init_mean.shape == (horizon, action_dim):
        mean = np.asarray(init_mean, dtype=np.float64).copy()
        std = np.full((horizon, action_dim), 0.25 * (action_high - action_low), dtype=np.float64)
    else:
        mean = np.zeros((horizon, action_dim), dtype=np.float64)
        std = np.full((horizon, action_dim), 0.5 * (action_high - action_low), dtype=np.float64)
        # Goal-biased warm mean for navigation / pushing when no MPC warm-start.
        gdir = _goal_vector_xy(spec.name, start_row, goal_row)
        if gdir is not None:
            mean[:, 0] = gdir[0]
            if action_dim >= 2:
                mean[:, 1] = gdir[1]
    elite_count = max(1, int(num_samples * elite_frac))
    best_score = float("inf")
    best_values = mean.copy()
    failed = 0
    library = (
        _discrete_action_library(
            spec, start_row, goal_row, action_low=action_low, action_high=action_high
        )
        if goal_bias_frac > 0
        else []
    )

    for _ in range(max(1, cem_iters)):
        candidates = rng.normal(loc=mean, scale=std, size=(num_samples, horizon, action_dim))
        candidates = np.clip(candidates, action_low, action_high)
        candidates[0] = mean
        if library and goal_bias_frac > 0:
            n_bias = min(num_samples - 1, max(1, int(num_samples * goal_bias_frac)))
            for i in range(n_bias):
                atom = library[i % len(library)]
                noise = rng.normal(0.0, 0.15 * (action_high - action_low), size=(horizon, action_dim))
                candidates[i + 1] = np.clip(np.tile(atom, (horizon, 1)) + noise, action_low, action_high)
        scores, iter_failed, env = _score_sim_candidates(
            env,
            spec,
            goal_row,
            start_row,
            candidates,
            frameskip=frameskip,
            metric=metric,
            rollout_batch_size=rollout_batch_size,
            env_factory=env_factory,
            env_recycle_every=env_recycle_every,
        )
        failed += iter_failed
        finite = np.isfinite(scores)
        if not finite.any():
            continue
        elite_indices = np.argsort(scores)[:elite_count]
        elites = candidates[elite_indices]
        mean = elites.mean(axis=0)
        std = np.maximum(elites.std(axis=0), 1e-3)
        if float(scores[elite_indices[0]]) < best_score:
            best_score = float(scores[elite_indices[0]])
            best_values = candidates[int(elite_indices[0])]
        gc.collect()
    return best_values, best_score, failed, env


def plan_mppi_sim_general(
    env: Any,
    spec: LeWMEnvSpec,
    goal_row: Mapping[str, Any],
    start_row: Mapping[str, Any],
    *,
    horizon: int,
    num_samples: int,
    mppi_iters: int,
    seed: int,
    action_low: float,
    action_high: float,
    frameskip: int = 1,
    init_mean: Optional[np.ndarray] = None,
    metric: str = "default",
    temperature: float = 1.0,
    noise_sigma: float = 0.35,
    rollout_batch_size: int = 50,
    env_factory: Optional[Any] = None,
    env_recycle_every: int = 200,
) -> Tuple[np.ndarray, float, int, Any]:
    """MPPI planning with true simulator rollouts."""

    rng = np.random.default_rng(seed)
    action_dim = spec.action_dim
    if init_mean is not None and init_mean.shape == (horizon, action_dim):
        mean = np.asarray(init_mean, dtype=np.float64).copy()
    else:
        mean = np.zeros((horizon, action_dim), dtype=np.float64)
    best_score = float("inf")
    best_values = mean.copy()
    failed = 0
    sigma = noise_sigma * (action_high - action_low)

    for _ in range(max(1, mppi_iters)):
        candidates = mean[None, :, :] + rng.normal(0.0, sigma, size=(num_samples, horizon, action_dim))
        candidates = np.clip(candidates, action_low, action_high)
        candidates[0] = mean
        scores, iter_failed, env = _score_sim_candidates(
            env,
            spec,
            goal_row,
            start_row,
            candidates,
            frameskip=frameskip,
            metric=metric,
            rollout_batch_size=rollout_batch_size,
            env_factory=env_factory,
            env_recycle_every=env_recycle_every,
        )
        failed += iter_failed
        finite = np.isfinite(scores)
        if not finite.any():
            continue
        finite_scores = scores[finite]
        finite_candidates = candidates[finite]
        min_score = float(finite_scores.min())
        weights = np.exp(-(finite_scores - min_score) / max(temperature, 1e-6))
        weights /= weights.sum()
        mean = np.tensordot(weights, finite_candidates, axes=(0, 0))
        mean = np.clip(mean, action_low, action_high)
        if min_score < best_score:
            best_score = min_score
            best_idx = int(np.argmin(finite_scores))
            best_values = finite_candidates[best_idx].copy()
        gc.collect()
    return best_values, best_score, failed, env


def evaluate_simulator_general_planning(
    *,
    env_name: str,
    dataset: Optional[str] = None,
    stablewm_home: Optional[str] = None,
    num_eval: int = 50,
    goal_offset_steps: int = 25,
    eval_budget: int = 50,
    frameskip: int = 0,
    horizon: int = 5,
    num_samples: int = 300,
    cem_iters: int = 10,
    elite_frac: float = 0.1,
    receding_horizon: int = 5,
    seed: int = 42,
    action_low: float = -1.0,
    action_high: float = 1.0,
    warm_start: bool = False,
    sim_planner: str = "cem",
    success_metric: str = "default",
    mppi_iters: Optional[int] = None,
    mppi_temperature: float = 1.0,
    mppi_noise_sigma: float = 0.35,
    ladder_tag: str = "",
    rollout_batch_size: int = 50,
    env_recycle_every: int = 200,
    beam_width: int = 16,
    goal_bias_frac: float = 0.0,
) -> Dict[str, Any]:
    """True-simulator planning ceiling under the cross-env protocol."""

    spec = get_env_spec(env_name)
    if not spec.can_plan_in_sim:
        return {
            "env": env_name,
            "planner": sim_planner,
            "success_rate": None,
            "planning_note": spec.planning_note,
        }
    dataset = dataset or spec.dataset
    frameskip = frameskip or spec.frameskip
    dataset_path = resolve_dataset_path(dataset, stablewm_home)
    h5py = require_h5py()

    def _make_sim_env() -> Any:
        return make_gym_env(spec, render_mode=sim_planning_render_mode(spec))

    env = _make_sim_env()
    successes: List[float] = []
    distances: List[float] = []
    failed_rollouts = 0
    if sim_planner == "mppi":
        planner_label = "sim_mppi"
    elif sim_planner in ("discrete_beam", "beam"):
        planner_label = "sim_discrete_beam"
    elif sim_planner in ("library_shoot", "expert_shoot"):
        planner_label = "sim_library_shoot"
    else:
        planner_label = "sim_cem"
    mppi_iters = cem_iters if mppi_iters is None else mppi_iters
    recycle_every = env_recycle_every if sim_planning_render_mode(spec) is None else max(env_recycle_every, 100)

    try:
        with h5py.File(dataset_path, "r") as h5:
            eval_starts = select_lewm_eval_starts(
                h5,
                num_eval=num_eval,
                goal_offset_steps=goal_offset_steps,
                seed=seed,
            )
            for trial_id, start in enumerate(eval_starts):
                start_global = _global_index(h5, start.episode_index, start.start_step)
                start_row = row_dict_from_h5(h5, start_global, spec)
                goal_row = goal_row_from_start(h5, spec, start.episode_index, start.start_step, goal_offset_steps)
                reset_env_from_rows(env, spec, start_row, goal_row)
                current_row = dict(start_row)
                macro_budget = max(1, eval_budget // frameskip)
                steps_taken = 0
                warm_mean: Optional[np.ndarray] = None

                while steps_taken < macro_budget:
                    # Plan over full horizon; execute only receding macros (PushT-style MPC).
                    plan_horizon = min(horizon, macro_budget - steps_taken)
                    plan_kwargs = dict(
                        env=env,
                        spec=spec,
                        goal_row=goal_row,
                        start_row=current_row,
                        horizon=plan_horizon,
                        num_samples=num_samples,
                        seed=seed + trial_id + steps_taken,
                        action_low=action_low,
                        action_high=action_high,
                        frameskip=frameskip,
                        init_mean=warm_mean[:plan_horizon] if warm_mean is not None else None,
                        metric=success_metric,
                    )
                    if sim_planner == "mppi":
                        plan_values, _, failed, env = plan_mppi_sim_general(
                            mppi_iters=mppi_iters,
                            temperature=mppi_temperature,
                            noise_sigma=mppi_noise_sigma,
                            rollout_batch_size=rollout_batch_size,
                            env_factory=_make_sim_env,
                            env_recycle_every=recycle_every,
                            **plan_kwargs,
                        )
                    elif sim_planner in ("discrete_beam", "beam"):
                        plan_values, _, failed, env = plan_discrete_beam_sim_general(
                            beam_width=beam_width,
                            rollout_batch_size=rollout_batch_size,
                            env_factory=_make_sim_env,
                            env_recycle_every=recycle_every,
                            **plan_kwargs,
                        )
                    elif sim_planner in ("library_shoot", "expert_shoot"):
                        plan_values, _, failed, env = plan_library_shoot_sim_general(
                            rollout_batch_size=rollout_batch_size,
                            env_factory=_make_sim_env,
                            env_recycle_every=recycle_every,
                            dataset_path=str(dataset_path),
                            **plan_kwargs,
                        )
                    else:
                        plan_values, _, failed, env = plan_cem_sim_general(
                            cem_iters=cem_iters,
                            elite_frac=elite_frac,
                            rollout_batch_size=rollout_batch_size,
                            env_factory=_make_sim_env,
                            env_recycle_every=recycle_every,
                            goal_bias_frac=goal_bias_frac,
                            **plan_kwargs,
                        )
                    failed_rollouts += failed
                    # Scoring rollouts leave `env` at the last candidate's terminal state;
                    # restore the live MPC state before executing the chosen plan.
                    reset_env_from_rows(env, spec, current_row, goal_row)
                    execute_macro = min(len(plan_values), macro_budget - steps_taken, receding_horizon)
                    for action in plan_values[:execute_macro]:
                        info: Mapping[str, Any] = {}
                        for _ in range(frameskip):
                            obs, info = sim_step_with_info(env, spec, action)
                        if spec.name == "tworoom":
                            current_row = sim_row_from_obs(spec.name, obs, template_row=current_row)
                        elif spec.name in ("cube", "reacher"):
                            current_row = sim_row_from_obs(spec.name, obs, template_row=current_row, info=info)
                        elif spec.name == "pusht":
                            current_row = sim_row_from_obs(spec.name, obs, template_row=current_row)
                        steps_taken += 1
                        if steps_taken >= macro_budget:
                            break
                    if warm_start and execute_macro < len(plan_values):
                        warm_mean = _shift_warm_mean(
                            plan_values,
                            execute_macro,
                            horizon,
                            spec.action_dim,
                        )
                    else:
                        warm_mean = None
                    if is_success(spec.name, goal_row, current_row, metric=success_metric):
                        break

                dist = goal_distance(spec.name, goal_row, current_row, metric=success_metric)
                distances.append(dist)
                successes.append(1.0 if is_success(spec.name, goal_row, current_row, metric=success_metric) else 0.0)
                try:
                    env.close()
                except Exception:
                    pass
                env = _make_sim_env()
                gc.collect()
    finally:
        env.close()

    report: Dict[str, Any] = {
        "env": env_name,
        "planner": planner_label,
        "dataset": str(dataset_path),
        "num_eval": num_eval,
        "success_rate": 100.0 * float(np.mean(successes)) if successes else 0.0,
        "mean_goal_distance": float(np.mean(distances)) if distances else None,
        "episode_successes": successes,
        "failed_model_rollouts": failed_rollouts,
        "frameskip": frameskip,
        "horizon": horizon,
        "receding_horizon": receding_horizon,
        "num_samples": num_samples,
        "cem_iters": cem_iters,
        "seed": seed,
        "warm_start": warm_start,
        "success_metric": success_metric,
        "sim_planner": sim_planner,
        "rollout_batch_size": rollout_batch_size,
        "env_recycle_every": recycle_every,
        "planning_render_mode": sim_planning_render_mode(spec),
        "beam_width": beam_width if sim_planner in ("discrete_beam", "beam") else None,
        "goal_bias_frac": goal_bias_frac,
    }
    if sim_planner == "mppi":
        report["mppi_iters"] = mppi_iters
        report["mppi_temperature"] = mppi_temperature
        report["mppi_noise_sigma"] = mppi_noise_sigma
    if ladder_tag:
        report["ladder_tag"] = ladder_tag
    return report
