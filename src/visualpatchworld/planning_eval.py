"""LeWM-style planning evaluation for induced PatchWorld PushT models."""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .hdf5 import episode_slices, require_h5py, resolve_dataset_path
from .lewm_sampling import graph_text_from_state, state_vector_from_payload
from .pusht import discretize_action
from .sim_eval import _make_pusht_env, _predict_next_graph


_ACTION_RE = re.compile(
    r"^(?P<label>\w+)\[dx=(?P<dx>-?\d+(?:\.\d+)?),dy=(?P<dy>-?\d+(?:\.\d+)?),speed=(?P<speed>-?\d+(?:\.\d+)?)\]$"
)


@dataclass(frozen=True)
class EvalStart:
    episode_index: int
    start_step: int
    row_index: int


def pusht_success(goal_state: Sequence[float], current_state: Sequence[float]) -> bool:
    goal = np.asarray(goal_state, dtype=np.float64)
    current = np.asarray(current_state, dtype=np.float64)
    pos_diff = np.linalg.norm(goal[:4] - current[:4])
    angle_diff = abs(goal[4] - current[4])
    angle_diff = min(angle_diff, 2 * math.pi - angle_diff)
    return bool(pos_diff < 20.0 and angle_diff < math.pi / 9.0)


def pusht_goal_distance(goal_state: Sequence[float], current_state: Sequence[float]) -> float:
    goal = np.asarray(goal_state, dtype=np.float64)
    current = np.asarray(current_state, dtype=np.float64)
    pos_diff = np.linalg.norm(goal[:4] - current[:4])
    angle_diff = abs(goal[4] - current[4])
    angle_diff = min(angle_diff, 2 * math.pi - angle_diff)
    return float(pos_diff + 10.0 * angle_diff)


def parse_action_deltas(action: str) -> Tuple[float, float]:
    match = _ACTION_RE.match(action.strip())
    if not match:
        return 0.0, 0.0
    return float(match.group("dx")), float(match.group("dy"))


def action_string(dx: float, dy: float) -> str:
    return discretize_action([dx, dy])


def sample_action_strings(rng: np.random.Generator, count: int) -> List[str]:
    actions = []
    for _ in range(count):
        dx = float(rng.uniform(-0.6, 0.6))
        dy = float(rng.uniform(-0.6, 0.6))
        actions.append(action_string(dx, dy))
    return actions


def action_values_to_strings(actions: np.ndarray) -> List[str]:
    return [action_string(float(action[0]), float(action[1])) for action in np.asarray(actions)]


def rollout_model_state(
    model: Any,
    observation_text: str,
    actions: Sequence[str],
    *,
    action_repeat: int = 1,
) -> np.ndarray:
    """Roll out the induced model over a macro-action sequence.

    ``action_repeat`` steps the (per-control-step) model that many times per macro
    action, so a per-frame model matches a planner whose macro action spans
    ``frameskip`` environment frames. With ``action_repeat=1`` this is the
    original per-step behaviour.
    """
    obs = observation_text
    predicted = None
    for action in actions:
        for _ in range(max(1, action_repeat)):
            predicted = _predict_next_graph(model, obs, action)
            obs = model.readout_observation(predicted, action)
    return np.asarray(state_vector_from_payload(predicted), dtype=np.float64)


def rollout_sim_macro(
    env: Any,
    start_state: Sequence[float],
    actions: Sequence[str],
    *,
    frameskip: int,
) -> np.ndarray:
    env.reset(options={"state": np.asarray(start_state, dtype=np.float64)})
    for action in actions:
        dx, dy = parse_action_deltas(action)
        for _ in range(frameskip):
            env.step(np.asarray([dx, dy], dtype=np.float32))
    return env.unwrapped._get_obs()


def rollout_sim_macro_values(
    env: Any,
    start_state: Sequence[float],
    actions: np.ndarray,
    *,
    frameskip: int,
) -> np.ndarray:
    env.reset(options={"state": np.asarray(start_state, dtype=np.float64)})
    for action in np.asarray(actions, dtype=np.float32):
        for _ in range(frameskip):
            env.step(action)
    return env.unwrapped._get_obs()


def rollout_sim_action_values(
    env: Any,
    start_state: Sequence[float],
    actions: np.ndarray,
) -> np.ndarray:
    env.reset(options={"state": np.asarray(start_state, dtype=np.float64)})
    for action in np.asarray(actions, dtype=np.float32):
        env.step(action)
    return env.unwrapped._get_obs()


def _cem_plan_sim_action_values(
    env: Any,
    start_state: Sequence[float],
    goal_state: Sequence[float],
    *,
    plan_steps: int,
    num_samples: int,
    cem_iters: int,
    elite_frac: float,
    rng: np.random.Generator,
    action_low: float,
    action_high: float,
    init_mean: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float]:
    """CEM over per-environment-step actions using true simulator dynamics."""

    mean = (
        np.asarray(init_mean, dtype=np.float64).copy()
        if init_mean is not None and np.asarray(init_mean).shape == (plan_steps, 2)
        else np.zeros((plan_steps, 2), dtype=np.float64)
    )
    std = np.full((plan_steps, 2), 0.5 * (action_high - action_low), dtype=np.float64)
    elite_count = max(1, int(num_samples * elite_frac))
    best_score = float("inf")
    best_values = mean.copy()

    for _ in range(max(1, cem_iters)):
        candidates = rng.normal(loc=mean, scale=std, size=(num_samples, plan_steps, 2))
        candidates = np.clip(candidates, action_low, action_high)
        candidates[0] = mean
        scores = np.empty((num_samples,), dtype=np.float64)
        for sample_idx, candidate in enumerate(candidates):
            predicted_state = rollout_sim_action_values(env, start_state, candidate)
            scores[sample_idx] = pusht_goal_distance(goal_state, predicted_state)
        elite_indices = np.argsort(scores)[:elite_count]
        elites = candidates[elite_indices]
        mean = elites.mean(axis=0)
        std = np.maximum(elites.std(axis=0), 1e-3)
        if float(scores[elite_indices[0]]) < best_score:
            best_score = float(scores[elite_indices[0]])
            best_values = candidates[int(elite_indices[0])]

    return best_values, best_score


def plan_random_shooting(
    model: Any,
    observation_text: str,
    goal_state: Sequence[float],
    *,
    horizon: int,
    num_samples: int,
    seed: int,
) -> List[str]:
    rng = np.random.default_rng(seed)
    best_score = float("inf")
    best_plan: List[str] = [action_string(0.0, 0.0) for _ in range(horizon)]
    for sample_idx in range(num_samples):
        candidate = sample_action_strings(rng, horizon)
        try:
            predicted_state = rollout_model_state(model, observation_text, candidate)
        except Exception:
            continue
        score = pusht_goal_distance(goal_state, predicted_state)
        if score < best_score:
            best_score = score
            best_plan = candidate
    return best_plan


def observation_text_from_state(
    state: Sequence[float],
    *,
    episode_index: Optional[int] = None,
    step: int = 0,
    block_velocity: Optional[Sequence[float]] = None,
) -> str:
    """Build oracle scene-graph text from a simulator state vector."""

    return graph_text_from_state(
        state,
        step=step,
        metadata={"episode_index": episode_index} if episode_index is not None else {},
        block_velocity=block_velocity,
    )


def observation_text_from_tool_render(
    env: Any,
    *,
    episode_index: Optional[int] = None,
    step: int = 0,
    fallback_observation_text: Optional[str] = None,
) -> Tuple[str, bool]:
    """Build a scene-graph observation by rendering the live sim frame and running CV tools.

    Returns ``(observation_text, extracted_ok)``. When extraction fails and
    ``fallback_observation_text`` is provided, returns the fallback with
    ``extracted_ok=False`` (stale re-observation, still pixel-fair).
    """

    import tempfile

    from PIL import Image

    from .pusht import graph_from_vlm_payload
    from .schema import render_graph_text
    from .visual_tools import PushTHeuristicSceneGraphExtractor

    frame = np.asarray(env.render(), dtype=np.uint8)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
        path = Path(handle.name)
    try:
        Image.fromarray(frame).save(path)
        payload = PushTHeuristicSceneGraphExtractor().extract(path)
    except Exception:
        if fallback_observation_text is not None:
            return fallback_observation_text, False
        raise
    finally:
        path.unlink(missing_ok=True)
    metadata = {"episode_index": episode_index} if episode_index is not None else {}
    graph = graph_from_vlm_payload(payload, step=step, metadata=metadata, source="tool")
    return render_graph_text(graph), True


def _initial_observation_text(
    *,
    observation_mode: str,
    env: Any,
    start_state: Sequence[float],
    start_global: int,
    episode_index: int,
    traj_by_row: Mapping[int, Mapping[str, Any]],
    block_velocity: Optional[Sequence[float]],
) -> Tuple[str, int]:
    if observation_mode == "tool_live":
        env.reset(options={"state": np.asarray(start_state, dtype=np.float64)})
        obs_text, ok = observation_text_from_tool_render(env, episode_index=episode_index, step=0)
        return obs_text, 0 if ok else 1
    if observation_mode in ("tool_traj", "vlm_traj") and start_global in traj_by_row:
        record = traj_by_row[start_global]
        transitions = record.get("transitions", [])
        if transitions:
            return str(transitions[0]["observation"]), 0
    return graph_text_from_state(
        start_state,
        step=0,
        metadata={"episode_index": episode_index},
        block_velocity=block_velocity,
    ), 0


def _replan_observation_text(
    *,
    observation_mode: str,
    env: Any,
    current_state: Sequence[float],
    episode_index: int,
    step: int,
    block_velocity: Optional[Sequence[float]],
    last_tool_observation: Optional[str] = None,
) -> Tuple[str, int]:
    if observation_mode in ("tool_live", "tool_traj"):
        obs_text, ok = observation_text_from_tool_render(
            env,
            episode_index=episode_index,
            step=step,
            fallback_observation_text=last_tool_observation,
        )
        return obs_text, 0 if ok else 1
    return observation_text_from_state(
        current_state,
        episode_index=episode_index,
        step=step,
        block_velocity=block_velocity,
    ), 0


def _block_velocity_from_states(
    prev_state: Optional[Sequence[float]],
    cur_state: Sequence[float],
) -> Optional[List[float]]:
    """Finite-difference block velocity (vx, vy, w) with angle wrapping."""
    if prev_state is None:
        return [0.0, 0.0, 0.0]
    cur = np.asarray(cur_state, dtype=np.float64)
    prev = np.asarray(prev_state, dtype=np.float64)
    dvx = float(cur[2] - prev[2])
    dvy = float(cur[3] - prev[3])
    dtheta = float(cur[4] - prev[4])
    dtheta = (dtheta + math.pi) % (2.0 * math.pi) - math.pi
    return [dvx, dvy, dtheta]


def discrete_macro_library(
    *,
    n_dirs: int = 12,
    speeds: Sequence[float] = (0.4, 0.7, 1.0),
    include_hold: bool = True,
) -> np.ndarray:
    """Compact directional macro alphabet for search-based planning."""

    actions: List[List[float]] = []
    for speed in speeds:
        for idx in range(max(1, n_dirs)):
            angle = 2.0 * math.pi * idx / float(n_dirs)
            actions.append([float(speed * math.cos(angle)), float(speed * math.sin(angle))])
    if include_hold:
        actions.append([0.0, 0.0])
    return np.asarray(actions, dtype=np.float64)


def contact_aware_macros(
    current_state: Sequence[float],
    goal_state: Sequence[float],
    *,
    speeds: Sequence[float] = (0.5, 1.0),
) -> np.ndarray:
    """Geometry-informed macros: approach block, then push block toward goal."""

    cur = np.asarray(current_state, dtype=np.float64)
    goal = np.asarray(goal_state, dtype=np.float64)
    ax, ay = float(cur[0]), float(cur[1])
    bx, by = float(cur[2]), float(cur[3])
    gx, gy = float(goal[2]), float(goal[3])
    to_block = np.asarray([bx - ax, by - ay], dtype=np.float64)
    push = np.asarray([gx - bx, gy - by], dtype=np.float64)
    tang = np.asarray([-push[1], push[0]], dtype=np.float64)

    def _unit(vec: np.ndarray) -> Optional[np.ndarray]:
        norm = float(np.linalg.norm(vec))
        if norm < 1e-6:
            return None
        return vec / norm

    dirs: List[np.ndarray] = []
    for vec in (to_block, push, tang, -tang):
        unit = _unit(vec)
        if unit is not None:
            dirs.append(unit)
    actions: List[List[float]] = []
    for unit in dirs:
        for speed in speeds:
            actions.append([float(speed * unit[0]), float(speed * unit[1])])
    return np.asarray(actions, dtype=np.float64) if actions else np.zeros((0, 2), dtype=np.float64)


def plan_beam_search(
    model: Any,
    observation_text: str,
    goal_state: Sequence[float],
    *,
    horizon: int,
    beam_width: int = 32,
    model_action_repeat: int = 1,
    action_library: Optional[np.ndarray] = None,
    current_state: Optional[Sequence[float]] = None,
) -> Tuple[np.ndarray, float, int]:
    """Discrete-macro beam search scored by induced-model rollouts.

    More natural than CEM for contact-rich PushT: search over a small action
    alphabet (directions × speeds + contact-aware suggestions) instead of
    Gaussian sampling in continuous action space.

    Expansions are incremental (one macro-step from each beam tip) so cost is
    O(horizon * beam_width * |library|) model steps, not quadratic.
    """

    library = discrete_macro_library() if action_library is None else np.asarray(action_library, dtype=np.float64)
    if current_state is not None:
        extras = contact_aware_macros(current_state, goal_state)
        if len(extras):
            library = np.vstack([library, extras])
    beam_width = max(1, int(beam_width))
    horizon = max(1, int(horizon))

    # beams: (actions so far, tip observation text, tip state vector or None)
    beams: List[Tuple[np.ndarray, str, Optional[np.ndarray]]] = [
        (np.zeros((0, 2), dtype=np.float64), observation_text, None)
    ]
    best_score = float("inf")
    best_values = np.zeros((horizon, 2), dtype=np.float64)
    failed = 0

    for _depth in range(horizon):
        scored: List[Tuple[float, np.ndarray, str, np.ndarray]] = []
        for prefix, tip_obs, _tip_state in beams:
            for action in library:
                candidate = (
                    np.vstack([prefix, action.reshape(1, 2)])
                    if prefix.shape[0]
                    else action.reshape(1, 2)
                )
                try:
                    obs = tip_obs
                    predicted = None
                    for _ in range(max(1, model_action_repeat)):
                        predicted = _predict_next_graph(
                            model, obs, action_string(float(action[0]), float(action[1]))
                        )
                        obs = model.readout_observation(predicted, action_string(float(action[0]), float(action[1])))
                    tip_state = np.asarray(state_vector_from_payload(predicted), dtype=np.float64)
                    score = pusht_goal_distance(goal_state, tip_state)
                except Exception:
                    failed += 1
                    continue
                scored.append((float(score), candidate, obs, tip_state))
                if float(score) < best_score:
                    best_score = float(score)
                    best_values = candidate.copy()
        if not scored:
            break
        scored.sort(key=lambda item: item[0])
        beams = [(cand, tip_obs, tip_state) for _, cand, tip_obs, tip_state in scored[:beam_width]]

    if best_values.shape[0] < horizon:
        pad = np.zeros((horizon - best_values.shape[0], 2), dtype=np.float64)
        best_values = np.vstack([best_values, pad])
    return best_values[:horizon], best_score, failed


def plan_beam_search_sim(
    env: Any,
    start_state: Sequence[float],
    goal_state: Sequence[float],
    *,
    horizon: int,
    beam_width: int = 32,
    frameskip: int = 1,
    action_library: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float]:
    """Discrete-macro beam search scored by true simulator rollouts (incremental)."""

    library = discrete_macro_library() if action_library is None else np.asarray(action_library, dtype=np.float64)
    extras = contact_aware_macros(start_state, goal_state)
    if len(extras):
        library = np.vstack([library, extras])
    beam_width = max(1, int(beam_width))
    horizon = max(1, int(horizon))

    beams: List[Tuple[np.ndarray, np.ndarray]] = [
        (np.zeros((0, 2), dtype=np.float64), np.asarray(start_state, dtype=np.float64))
    ]
    best_score = float("inf")
    best_values = np.zeros((horizon, 2), dtype=np.float64)

    for _depth in range(horizon):
        scored: List[Tuple[float, np.ndarray, np.ndarray]] = []
        for prefix, tip_state in beams:
            for action in library:
                candidate = (
                    np.vstack([prefix, action.reshape(1, 2)])
                    if prefix.shape[0]
                    else action.reshape(1, 2)
                )
                # One-macro step from tip (avoid replaying the whole prefix).
                next_state = rollout_sim_macro_values(
                    env, tip_state, action.reshape(1, 2), frameskip=frameskip
                )
                score = pusht_goal_distance(goal_state, next_state)
                scored.append((float(score), candidate, next_state))
                if float(score) < best_score:
                    best_score = float(score)
                    best_values = candidate.copy()
        if not scored:
            break
        scored.sort(key=lambda item: item[0])
        beams = [(cand, tip) for _, cand, tip in scored[:beam_width]]

    if best_values.shape[0] < horizon:
        pad = np.zeros((horizon - best_values.shape[0], 2), dtype=np.float64)
        best_values = np.vstack([best_values, pad])
    return best_values[:horizon], best_score


def plan_cem(
    model: Any,
    observation_text: str,
    goal_state: Sequence[float],
    *,
    horizon: int,
    num_samples: int,
    cem_iters: int,
    elite_frac: float,
    seed: int,
    action_low: float = -1.0,
    action_high: float = 1.0,
    init_mean: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
    sim_env: Any = None,
    sim_start_state: Optional[Sequence[float]] = None,
    sim_verify_frac: float = 0.0,
    frameskip: int = 5,
    model_action_repeat: int = 1,
) -> Tuple[np.ndarray, float, int]:
    """Plan continuous macro-actions by CEM using the induced model for scoring."""

    rng = rng if rng is not None else np.random.default_rng(seed)
    mean = (
        np.asarray(init_mean, dtype=np.float64).copy()
        if init_mean is not None and np.asarray(init_mean).shape == (horizon, 2)
        else np.zeros((horizon, 2), dtype=np.float64)
    )
    std = np.full((horizon, 2), 0.5 * (action_high - action_low), dtype=np.float64)
    elite_count = max(1, int(num_samples * elite_frac))
    best_score = float("inf")
    best_values = mean.copy()
    failed_rollouts = 0
    verify_count = max(0, int(math.ceil(num_samples * sim_verify_frac))) if sim_verify_frac > 0 else 0
    verify_rng = np.random.default_rng(seed + 7919)

    for _ in range(max(1, cem_iters)):
        candidates = rng.normal(loc=mean, scale=std, size=(num_samples, horizon, 2))
        candidates = np.clip(candidates, action_low, action_high)
        candidates[0] = mean
        scores = np.full((num_samples,), np.inf, dtype=np.float64)
        for sample_idx, candidate in enumerate(candidates):
            try:
                predicted_state = rollout_model_state(
                    model,
                    observation_text,
                    action_values_to_strings(candidate),
                    action_repeat=model_action_repeat,
                )
                scores[sample_idx] = pusht_goal_distance(goal_state, predicted_state)
            except Exception:
                failed_rollouts += 1
        if verify_count > 0 and sim_env is not None and sim_start_state is not None:
            finite = np.isfinite(scores)
            if finite.any():
                random_order = os.environ.get("VPW_SIM_VERIFY_ORDER", "model") == "random"
                ranked = verify_rng.permutation(num_samples) if random_order else np.argsort(scores)
                if random_order:
                    unverified = np.ones(num_samples, dtype=bool)
                    unverified[ranked[:verify_count]] = False
                for sample_idx in ranked[:verify_count]:
                    if not finite[sample_idx]:
                        continue
                    try:
                        sim_state = rollout_sim_macro_values(
                            sim_env,
                            sim_start_state,
                            candidates[sample_idx],
                            frameskip=frameskip,
                        )
                        scores[sample_idx] = pusht_goal_distance(goal_state, sim_state)
                    except Exception:
                        failed_rollouts += 1
                if random_order:
                    scores[unverified] = np.inf
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

    return best_values, best_score, failed_rollouts


def select_lewm_eval_starts(
    h5: Mapping[str, Any],
    *,
    num_eval: int = 50,
    goal_offset_steps: int = 25,
    seed: int = 42,
) -> List[EvalStart]:
    col_name = "episode_idx" if "episode_idx" in h5 else "ep_idx"
    episode_ids = np.asarray(h5[col_name][:])
    step_ids = np.asarray(h5["step_idx"][:])
    slices = episode_slices(h5, fallback_length=len(episode_ids))
    episode_lengths = {sl.episode_index: sl.length for sl in slices}

    valid_rows: List[EvalStart] = []
    for row_index, (episode_index, step) in enumerate(zip(episode_ids, step_ids)):
        episode_index = int(episode_index)
        step = int(step)
        max_start = episode_lengths.get(episode_index, 0) - goal_offset_steps - 1
        if step <= max_start:
            valid_rows.append(EvalStart(episode_index=episode_index, start_step=step, row_index=row_index))

    if len(valid_rows) < num_eval:
        raise ValueError(f"Not enough valid eval starts: {len(valid_rows)} < {num_eval}")

    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(len(valid_rows), size=num_eval, replace=False))
    return [valid_rows[int(i)] for i in chosen]


def _global_index(h5: Mapping[str, Any], episode_index: int, local_step: int) -> int:
    offset = int(h5["ep_offset"][episode_index])
    return offset + local_step


def evaluate_planning(
    *,
    model: Any,
    dataset: str,
    stablewm_home: Optional[str],
    trajectories_jsonl: Optional[Path] = None,
    num_eval: int = 50,
    goal_offset_steps: int = 25,
    eval_budget: int = 50,
    frameskip: int = 5,
    horizon: int = 10,
    num_samples: int = 64,
    planner: str = "random_shooting",
    cem_iters: int = 5,
    elite_frac: float = 0.1,
    action_low: float = -1.0,
    action_high: float = 1.0,
    receding_horizon: int = 1,
    model_rollout_horizon: int = 1,
    sim_verify_frac: float = 0.0,
    warm_start: bool = True,
    seed: int = 42,
    observation_mode: str = "oracle",
    expose_block_velocity: bool = False,
    model_action_repeat: int = 1,
) -> Dict[str, Any]:
    dataset_path = resolve_dataset_path(dataset, stablewm_home)
    h5py = require_h5py()
    env = _make_pusht_env()
    starts = []
    successes = []
    final_distances = []
    model_distances = []
    failed_model_rollouts = 0
    tool_extraction_failures = 0

    try:
        with h5py.File(dataset_path, "r") as h5:
            states = h5["state"]
            eval_starts = select_lewm_eval_starts(
                h5,
                num_eval=num_eval,
                goal_offset_steps=goal_offset_steps,
                seed=seed,
            )
            traj_by_row: Dict[int, Mapping[str, Any]] = {}
            if trajectories_jsonl is not None and trajectories_jsonl.is_file():
                import json

                with trajectories_jsonl.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        record = json.loads(line)
                        meta = record.get("metadata", {})
                        if not isinstance(meta, Mapping):
                            continue
                        if "global_start" in meta:
                            traj_by_row[int(meta["global_start"])] = record
                            continue
                        transitions = record.get("transitions", [])
                        if transitions and isinstance(transitions[0], Mapping):
                            transition_meta = transitions[0].get("metadata", {})
                            if isinstance(transition_meta, Mapping) and "raw_index" in transition_meta:
                                traj_by_row[int(transition_meta["raw_index"])] = record

            for trial_id, start in enumerate(eval_starts):
                start_global = _global_index(h5, start.episode_index, start.start_step)
                goal_global = _global_index(h5, start.episode_index, start.start_step + goal_offset_steps)
                start_state = np.asarray(states[start_global], dtype=np.float64)
                goal_state = np.asarray(states[goal_global], dtype=np.float64)

                start_block_velocity: Optional[List[float]] = None
                if expose_block_velocity:
                    prev_start_state = None
                    if start.start_step > 0:
                        prev_start_state = np.asarray(states[start_global - 1], dtype=np.float64)
                    start_block_velocity = _block_velocity_from_states(prev_start_state, start_state)

                env.reset(options={"state": np.asarray(start_state, dtype=np.float64)})
                obs_text, init_tool_failures = _initial_observation_text(
                    observation_mode=observation_mode,
                    env=env,
                    start_state=start_state,
                    start_global=start_global,
                    episode_index=start.episode_index,
                    traj_by_row=traj_by_row,
                    block_velocity=start_block_velocity,
                )
                tool_extraction_failures += init_tool_failures

                macro_horizon = min(horizon, max(1, eval_budget // frameskip))
                if planner == "random_shooting":
                    plan = plan_random_shooting(
                        model,
                        obs_text,
                        goal_state,
                        horizon=macro_horizon,
                        num_samples=num_samples,
                        seed=seed + trial_id,
                    )
                    try:
                        predicted_state = rollout_model_state(model, obs_text, plan)
                        model_distances.append(pusht_goal_distance(goal_state, predicted_state))
                    except Exception:
                        failed_model_rollouts += 1
                    final_state = rollout_sim_macro(env, start_state, plan, frameskip=frameskip)
                elif planner == "cem":
                    plan_values, model_score, failures = plan_cem(
                        model,
                        obs_text,
                        goal_state,
                        horizon=macro_horizon,
                        num_samples=num_samples,
                        cem_iters=cem_iters,
                        elite_frac=elite_frac,
                        seed=seed + trial_id,
                        action_low=action_low,
                        action_high=action_high,
                        sim_env=env if sim_verify_frac > 0 else None,
                        sim_start_state=start_state if sim_verify_frac > 0 else None,
                        sim_verify_frac=sim_verify_frac,
                        frameskip=frameskip,
                        model_action_repeat=model_action_repeat,
                    )
                    failed_model_rollouts += failures
                    if math.isfinite(model_score):
                        model_distances.append(float(model_score))
                    final_state = rollout_sim_macro_values(env, start_state, plan_values, frameskip=frameskip)
                elif planner == "cem_mpc":
                    macro_budget = max(1, eval_budget // frameskip)
                    current_state = start_state.copy()
                    current_obs = obs_text
                    last_tool_observation = obs_text if observation_mode == "tool_live" else None
                    prev_block_state = current_state.copy()
                    steps_remaining = macro_budget
                    best_score = float("inf")
                    warm_mean: Optional[np.ndarray] = None
                    rng = np.random.default_rng(seed + trial_id)
                    while steps_remaining > 0:
                        plan_horizon = min(
                            horizon,
                            model_rollout_horizon,
                            steps_remaining,
                        )
                        plan_values, model_score, failures = plan_cem(
                            model,
                            current_obs,
                            goal_state,
                            horizon=plan_horizon,
                            num_samples=num_samples,
                            cem_iters=cem_iters,
                            elite_frac=elite_frac,
                            seed=seed + trial_id,
                            action_low=action_low,
                            action_high=action_high,
                            init_mean=warm_mean[:plan_horizon] if warm_mean is not None else None,
                            rng=rng,
                            sim_env=env if sim_verify_frac > 0 else None,
                            sim_start_state=current_state if sim_verify_frac > 0 else None,
                            sim_verify_frac=sim_verify_frac,
                            frameskip=frameskip,
                            model_action_repeat=model_action_repeat,
                        )
                        failed_model_rollouts += failures
                        if math.isfinite(model_score):
                            best_score = min(best_score, float(model_score))
                        execute_macro = min(receding_horizon, steps_remaining, plan_horizon)
                        executed = plan_values[:execute_macro]
                        current_state = rollout_sim_macro_values(
                            env,
                            current_state,
                            executed,
                            frameskip=frameskip,
                        )
                        steps_remaining -= execute_macro
                        step_block_velocity = (
                            _block_velocity_from_states(prev_block_state, current_state)
                            if expose_block_velocity
                            else None
                        )
                        current_obs, replan_tool_failures = _replan_observation_text(
                            observation_mode=observation_mode,
                            env=env,
                            current_state=current_state,
                            episode_index=start.episode_index,
                            step=macro_budget - steps_remaining,
                            block_velocity=step_block_velocity,
                            last_tool_observation=last_tool_observation,
                        )
                        tool_extraction_failures += replan_tool_failures
                        if observation_mode == "tool_live":
                            last_tool_observation = current_obs
                        prev_block_state = current_state.copy()
                        if warm_start and execute_macro < len(plan_values):
                            remaining = plan_values[execute_macro:]
                            pad_count = model_rollout_horizon - len(remaining)
                            pad = np.zeros((max(0, pad_count), 2), dtype=np.float64)
                            warm_mean = np.concatenate([remaining, pad], axis=0)[:model_rollout_horizon]
                        else:
                            warm_mean = None
                    if math.isfinite(best_score):
                        model_distances.append(float(best_score))
                    final_state = current_state
                elif planner == "beam_mpc":
                    macro_budget = max(1, eval_budget // frameskip)
                    current_state = start_state.copy()
                    current_obs = obs_text
                    last_tool_observation = obs_text if observation_mode == "tool_live" else None
                    prev_block_state = current_state.copy()
                    steps_remaining = macro_budget
                    best_score = float("inf")
                    beam_width = max(8, int(num_samples) if num_samples <= 128 else int(num_samples) // 12)
                    while steps_remaining > 0:
                        plan_horizon = min(horizon, model_rollout_horizon, steps_remaining)
                        plan_values, model_score, failures = plan_beam_search(
                            model,
                            current_obs,
                            goal_state,
                            horizon=plan_horizon,
                            beam_width=beam_width,
                            model_action_repeat=model_action_repeat,
                            current_state=current_state,
                        )
                        failed_model_rollouts += failures
                        if math.isfinite(model_score):
                            best_score = min(best_score, float(model_score))
                        execute_macro = min(receding_horizon, steps_remaining, plan_horizon)
                        executed = plan_values[:execute_macro]
                        current_state = rollout_sim_macro_values(
                            env,
                            current_state,
                            executed,
                            frameskip=frameskip,
                        )
                        steps_remaining -= execute_macro
                        step_block_velocity = (
                            _block_velocity_from_states(prev_block_state, current_state)
                            if expose_block_velocity
                            else None
                        )
                        current_obs, replan_tool_failures = _replan_observation_text(
                            observation_mode=observation_mode,
                            env=env,
                            current_state=current_state,
                            episode_index=start.episode_index,
                            step=macro_budget - steps_remaining,
                            block_velocity=step_block_velocity,
                            last_tool_observation=last_tool_observation,
                        )
                        tool_extraction_failures += replan_tool_failures
                        if observation_mode == "tool_live":
                            last_tool_observation = current_obs
                        prev_block_state = current_state.copy()
                    if math.isfinite(best_score):
                        model_distances.append(float(best_score))
                    final_state = current_state
                else:
                    raise ValueError(f"Unsupported induced-model planner: {planner}")
                success = pusht_success(goal_state, final_state)
                starts.append(
                    {
                        "episode_index": start.episode_index,
                        "start_step": start.start_step,
                        "row_index": start.row_index,
                    }
                )
                successes.append(float(success))
                final_distances.append(pusht_goal_distance(goal_state, final_state))
    finally:
        env.close()

    return {
        "dataset": str(dataset_path),
        "observation_mode": observation_mode,
        "num_eval": num_eval,
        "goal_offset_steps": goal_offset_steps,
        "eval_budget": eval_budget,
        "frameskip": frameskip,
        "model_action_repeat": model_action_repeat,
        "horizon": horizon,
        "num_samples": num_samples,
        "planner": planner,
        "cem_iters": cem_iters if planner in ("cem", "cem_mpc") else 0,
        "elite_frac": elite_frac if planner in ("cem", "cem_mpc") else None,
        "beam_width": max(8, int(num_samples) if num_samples <= 128 else int(num_samples) // 12)
        if planner == "beam_mpc"
        else None,
        "action_low": action_low,
        "action_high": action_high,
        "receding_horizon": receding_horizon if planner in ("cem_mpc", "beam_mpc") else None,
        "model_rollout_horizon": model_rollout_horizon if planner in ("cem_mpc", "beam_mpc") else None,
        "sim_verify_frac": sim_verify_frac if planner in ("cem", "cem_mpc") else None,
        "warm_start": warm_start if planner == "cem_mpc" else None,
        "seed": seed,
        "success_rate": 100.0 * sum(successes) / max(1, len(successes)),
        "mean_goal_distance": sum(final_distances) / max(1, len(final_distances)),
        "mean_planner_model_distance": sum(model_distances) / max(1, len(model_distances)) if model_distances else None,
        "failed_model_rollouts": failed_model_rollouts,
        "tool_extraction_failures": tool_extraction_failures,
        "episode_successes": successes,
        "starts": starts,
    }


def evaluate_simulator_planning(
    *,
    dataset: str,
    stablewm_home: Optional[str],
    num_eval: int = 50,
    goal_offset_steps: int = 25,
    eval_budget: int = 50,
    frameskip: int = 5,
    horizon: int = 10,
    num_samples: int = 64,
    planner: str = "random_shooting",
    cem_iters: int = 5,
    elite_frac: float = 0.1,
    action_block: int = 5,
    receding_horizon: int = 5,
    warm_start: bool = True,
    action_low: float = -1.0,
    action_high: float = 1.0,
    seed: int = 42,
) -> Dict[str, Any]:
    """Plan with the true PushT simulator dynamics as an upper-bound model.

    `random_shooting` uses the same symbolic action distribution as
    induced-model planning. `cem` optimizes repeated macro-actions with the
    true simulator as the model. `cem_mpc` matches LeWM's action-block style
    more closely: optimize per-step actions for `horizon * action_block`
    simulator steps, execute `receding_horizon * action_block` steps, then
    replan until the budget is exhausted.
    """

    dataset_path = resolve_dataset_path(dataset, stablewm_home)
    h5py = require_h5py()
    env = _make_pusht_env()
    starts = []
    successes = []
    final_distances = []
    best_model_distances = []

    try:
        with h5py.File(dataset_path, "r") as h5:
            states = h5["state"]
            eval_starts = select_lewm_eval_starts(
                h5,
                num_eval=num_eval,
                goal_offset_steps=goal_offset_steps,
                seed=seed,
            )

            for trial_id, start in enumerate(eval_starts):
                start_global = _global_index(h5, start.episode_index, start.start_step)
                goal_global = _global_index(h5, start.episode_index, start.start_step + goal_offset_steps)
                start_state = np.asarray(states[start_global], dtype=np.float64)
                goal_state = np.asarray(states[goal_global], dtype=np.float64)
                macro_horizon = min(horizon, max(1, eval_budget // frameskip))

                rng = np.random.default_rng(seed + trial_id)
                if planner == "random_shooting":
                    best_score = float("inf")
                    best_plan: List[str] = [action_string(0.0, 0.0) for _ in range(macro_horizon)]
                    for _ in range(num_samples):
                        candidate = sample_action_strings(rng, macro_horizon)
                        predicted_state = rollout_sim_macro(env, start_state, candidate, frameskip=frameskip)
                        score = pusht_goal_distance(goal_state, predicted_state)
                        if score < best_score:
                            best_score = score
                            best_plan = candidate
                    final_state = rollout_sim_macro(env, start_state, best_plan, frameskip=frameskip)
                elif planner == "cem":
                    mean = np.zeros((macro_horizon, 2), dtype=np.float64)
                    std = np.full((macro_horizon, 2), 0.5 * (action_high - action_low), dtype=np.float64)
                    best_score = float("inf")
                    best_values = np.zeros((macro_horizon, 2), dtype=np.float64)
                    elite_count = max(1, int(num_samples * elite_frac))
                    for _ in range(cem_iters):
                        candidates = rng.normal(loc=mean, scale=std, size=(num_samples, macro_horizon, 2))
                        candidates = np.clip(candidates, action_low, action_high)
                        scores = np.empty((num_samples,), dtype=np.float64)
                        for sample_idx, candidate in enumerate(candidates):
                            predicted_state = rollout_sim_macro_values(
                                env,
                                start_state,
                                candidate,
                                frameskip=frameskip,
                            )
                            scores[sample_idx] = pusht_goal_distance(goal_state, predicted_state)
                        elite_indices = np.argsort(scores)[:elite_count]
                        elites = candidates[elite_indices]
                        mean = elites.mean(axis=0)
                        std = np.maximum(elites.std(axis=0), 1e-3)
                        if float(scores[elite_indices[0]]) < best_score:
                            best_score = float(scores[elite_indices[0]])
                            best_values = candidates[int(elite_indices[0])]
                    final_state = rollout_sim_macro_values(env, start_state, best_values, frameskip=frameskip)
                elif planner == "cem_mpc":
                    plan_steps = max(1, horizon * action_block)
                    execute_steps = max(1, min(receding_horizon * action_block, plan_steps))
                    current_state = start_state.copy()
                    steps_remaining = int(eval_budget)
                    best_score = float("inf")
                    warm_mean: Optional[np.ndarray] = None
                    while steps_remaining > 0:
                        planned_actions, planned_score = _cem_plan_sim_action_values(
                            env,
                            current_state,
                            goal_state,
                            plan_steps=plan_steps,
                            num_samples=num_samples,
                            cem_iters=cem_iters,
                            elite_frac=elite_frac,
                            rng=rng,
                            action_low=action_low,
                            action_high=action_high,
                            init_mean=warm_mean,
                        )
                        best_score = min(best_score, planned_score)
                        n_execute = min(execute_steps, steps_remaining, len(planned_actions))
                        executed = planned_actions[:n_execute]
                        current_state = rollout_sim_action_values(env, current_state, executed)
                        steps_remaining -= n_execute
                        if warm_start and n_execute < len(planned_actions):
                            remaining = planned_actions[n_execute:]
                            pad_count = plan_steps - len(remaining)
                            pad = np.zeros((max(0, pad_count), 2), dtype=np.float64)
                            warm_mean = np.concatenate([remaining, pad], axis=0)[:plan_steps]
                        else:
                            warm_mean = None
                    final_state = current_state
                elif planner == "beam_mpc":
                    # Macro beam-MPC matched to induced-model protocol (frameskip macros).
                    macro_budget = max(1, eval_budget // max(1, frameskip))
                    current_state = start_state.copy()
                    steps_remaining = macro_budget
                    best_score = float("inf")
                    beam_width = max(8, int(num_samples) if num_samples <= 128 else int(num_samples) // 12)
                    while steps_remaining > 0:
                        plan_horizon = min(horizon, steps_remaining)
                        planned_actions, planned_score = plan_beam_search_sim(
                            env,
                            current_state,
                            goal_state,
                            horizon=plan_horizon,
                            beam_width=beam_width,
                            frameskip=frameskip,
                        )
                        best_score = min(best_score, planned_score)
                        n_execute = min(receding_horizon, steps_remaining, plan_horizon)
                        current_state = rollout_sim_macro_values(
                            env,
                            current_state,
                            planned_actions[:n_execute],
                            frameskip=frameskip,
                        )
                        steps_remaining -= n_execute
                    final_state = current_state
                else:
                    raise ValueError(f"Unsupported simulator planner: {planner}")
                success = pusht_success(goal_state, final_state)
                starts.append(
                    {
                        "episode_index": start.episode_index,
                        "start_step": start.start_step,
                        "row_index": start.row_index,
                    }
                )
                successes.append(float(success))
                final_distances.append(pusht_goal_distance(goal_state, final_state))
                best_model_distances.append(float(best_score))
    finally:
        env.close()

    return {
        "dataset": str(dataset_path),
        "model_type": "ground_truth_simulator",
        "planner": planner,
        "num_eval": num_eval,
        "goal_offset_steps": goal_offset_steps,
        "eval_budget": eval_budget,
        "frameskip": frameskip,
        "horizon": horizon,
        "num_samples": num_samples,
        "cem_iters": cem_iters if planner in ("cem", "cem_mpc") else 0,
        "elite_frac": elite_frac if planner in ("cem", "cem_mpc") else None,
        "beam_width": max(8, int(num_samples) if num_samples <= 128 else int(num_samples) // 12)
        if planner == "beam_mpc"
        else None,
        "action_block": action_block if planner == "cem_mpc" else None,
        "receding_horizon": receding_horizon if planner in ("cem_mpc", "beam_mpc") else None,
        "warm_start": warm_start if planner == "cem_mpc" else None,
        "action_low": action_low,
        "action_high": action_high,
        "seed": seed,
        "success_rate": 100.0 * sum(successes) / max(1, len(successes)),
        "mean_goal_distance": sum(final_distances) / max(1, len(final_distances)),
        "mean_planner_model_distance": sum(best_model_distances) / max(1, len(best_model_distances)),
        "episode_successes": successes,
        "starts": starts,
    }
