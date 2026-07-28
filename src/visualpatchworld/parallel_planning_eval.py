"""Episode-parallel PushT planning eval (MuJoCo GT + VPW program).

Splits trials across processes so CEM/beam MPC finishes in minutes on
multi-core machines. Each worker owns its own env / model instance.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .hdf5 import require_h5py, resolve_dataset_path
from .lewm_sampling import graph_text_from_state
from .planning_eval import (
    EvalStart,
    _cem_plan_sim_action_values,
    _global_index,
    plan_beam_search,
    plan_beam_search_sim,
    plan_cem,
    pusht_goal_distance,
    pusht_success,
    rollout_sim_action_values,
    rollout_sim_macro_values,
    select_lewm_eval_starts,
)
from .sim_eval import _make_pusht_env, load_induced_model


@dataclass(frozen=True)
class TrialSpec:
    trial_id: int
    episode_index: int
    start_step: int
    row_index: int
    start_state: Tuple[float, ...]
    goal_state: Tuple[float, ...]


@dataclass(frozen=True)
class PlannerConfig:
    planner: str  # cem_mpc | beam_mpc
    dynamics: str  # mujoco | vpw
    model_path: str = ""
    eval_budget: int = 50
    frameskip: int = 1
    horizon: int = 5
    receding_horizon: int = 1
    model_rollout_horizon: int = 5
    num_samples: int = 100
    cem_iters: int = 5
    elite_frac: float = 0.1
    action_block: int = 1
    action_low: float = -1.0
    action_high: float = 1.0
    seed: int = 42
    warm_start: bool = True


def _beam_width(num_samples: int) -> int:
    return max(8, int(num_samples) if num_samples <= 128 else int(num_samples) // 12)


def collect_trial_specs(
    *,
    dataset: str,
    stablewm_home: Optional[str],
    num_eval: int,
    goal_offset_steps: int,
    seed: int,
) -> Tuple[str, List[TrialSpec]]:
    dataset_path = str(resolve_dataset_path(dataset, stablewm_home))
    h5py = require_h5py()
    specs: List[TrialSpec] = []
    with h5py.File(dataset_path, "r") as h5:
        states = h5["state"]
        starts = select_lewm_eval_starts(
            h5, num_eval=num_eval, goal_offset_steps=goal_offset_steps, seed=seed
        )
        for trial_id, start in enumerate(starts):
            start_global = _global_index(h5, start.episode_index, start.start_step)
            goal_global = _global_index(
                h5, start.episode_index, start.start_step + goal_offset_steps
            )
            specs.append(
                TrialSpec(
                    trial_id=trial_id,
                    episode_index=int(start.episode_index),
                    start_step=int(start.start_step),
                    row_index=int(start.row_index),
                    start_state=tuple(np.asarray(states[start_global], dtype=np.float64).tolist()),
                    goal_state=tuple(np.asarray(states[goal_global], dtype=np.float64).tolist()),
                )
            )
    return dataset_path, specs


def _run_mujoco_trial(trial: TrialSpec, cfg: PlannerConfig) -> Dict[str, Any]:
    env = _make_pusht_env()
    try:
        start_state = np.asarray(trial.start_state, dtype=np.float64)
        goal_state = np.asarray(trial.goal_state, dtype=np.float64)
        rng = np.random.default_rng(cfg.seed + trial.trial_id)
        best_score = float("inf")

        if cfg.planner == "cem_mpc":
            plan_steps = max(1, cfg.horizon * cfg.action_block)
            execute_steps = max(1, min(cfg.receding_horizon * cfg.action_block, plan_steps))
            current = start_state.copy()
            steps_remaining = int(cfg.eval_budget)
            warm_mean = None
            while steps_remaining > 0:
                planned, score = _cem_plan_sim_action_values(
                    env,
                    current,
                    goal_state,
                    plan_steps=plan_steps,
                    num_samples=cfg.num_samples,
                    cem_iters=cfg.cem_iters,
                    elite_frac=cfg.elite_frac,
                    rng=rng,
                    action_low=cfg.action_low,
                    action_high=cfg.action_high,
                    init_mean=warm_mean,
                )
                best_score = min(best_score, float(score))
                n_exec = min(execute_steps, steps_remaining, len(planned))
                current = rollout_sim_action_values(env, current, planned[:n_exec])
                steps_remaining -= n_exec
                if cfg.warm_start and n_exec < len(planned):
                    remaining = planned[n_exec:]
                    pad = np.zeros((max(0, plan_steps - len(remaining)), 2), dtype=np.float64)
                    warm_mean = np.concatenate([remaining, pad], axis=0)[:plan_steps]
                else:
                    warm_mean = None
            final = current
        elif cfg.planner == "beam_mpc":
            macro_budget = max(1, cfg.eval_budget // max(1, cfg.frameskip))
            current = start_state.copy()
            steps_remaining = macro_budget
            bw = _beam_width(cfg.num_samples)
            while steps_remaining > 0:
                plan_h = min(cfg.horizon, steps_remaining)
                planned, score = plan_beam_search_sim(
                    env,
                    current,
                    goal_state,
                    horizon=plan_h,
                    beam_width=bw,
                    frameskip=cfg.frameskip,
                )
                best_score = min(best_score, float(score))
                n_exec = min(cfg.receding_horizon, steps_remaining, plan_h)
                current = rollout_sim_macro_values(
                    env, current, planned[:n_exec], frameskip=cfg.frameskip
                )
                steps_remaining -= n_exec
            final = current
        else:
            raise ValueError(f"unsupported mujoco planner {cfg.planner}")

        return {
            "trial_id": trial.trial_id,
            "success": float(pusht_success(goal_state, final)),
            "goal_distance": float(pusht_goal_distance(goal_state, final)),
            "model_distance": float(best_score),
            "failed_model_rollouts": 0,
            "start": {
                "episode_index": trial.episode_index,
                "start_step": trial.start_step,
                "row_index": trial.row_index,
            },
        }
    finally:
        env.close()


def _run_vpw_trial(trial: TrialSpec, cfg: PlannerConfig) -> Dict[str, Any]:
    model = load_induced_model(Path(cfg.model_path))
    env = _make_pusht_env()
    try:
        start_state = np.asarray(trial.start_state, dtype=np.float64)
        goal_state = np.asarray(trial.goal_state, dtype=np.float64)
        obs = graph_text_from_state(
            start_state, step=0, metadata={"episode_index": trial.episode_index}
        )
        best_score = float("inf")
        failed = 0
        current = start_state.copy()
        current_obs = obs
        warm_mean = None
        rng = np.random.default_rng(cfg.seed + trial.trial_id)
        bw = _beam_width(cfg.num_samples)

        if cfg.planner == "cem_mpc":
            # Match MuJoCo cem_mpc: optimize/execute per-step actions over
            # horizon*action_block steps (frameskip is unused in this path).
            plan_steps = max(1, cfg.horizon * cfg.action_block)
            execute_steps = max(1, min(cfg.receding_horizon * cfg.action_block, plan_steps))
            steps_remaining = int(cfg.eval_budget)
            while steps_remaining > 0:
                plan_h = min(plan_steps, steps_remaining)
                planned, score, fails = plan_cem(
                    model,
                    current_obs,
                    goal_state,
                    horizon=plan_h,
                    num_samples=cfg.num_samples,
                    cem_iters=cfg.cem_iters,
                    elite_frac=cfg.elite_frac,
                    seed=cfg.seed + trial.trial_id,
                    action_low=cfg.action_low,
                    action_high=cfg.action_high,
                    init_mean=warm_mean[:plan_h] if warm_mean is not None else None,
                    rng=rng,
                    frameskip=1,
                    model_action_repeat=1,
                )
                failed += fails
                if np.isfinite(score):
                    best_score = min(best_score, float(score))
                n_exec = min(execute_steps, steps_remaining, len(planned))
                current = rollout_sim_action_values(env, current, planned[:n_exec])
                steps_remaining -= n_exec
                current_obs = graph_text_from_state(
                    current,
                    step=int(cfg.eval_budget) - steps_remaining,
                    metadata={"episode_index": trial.episode_index},
                )
                if cfg.warm_start and n_exec < len(planned):
                    remaining = planned[n_exec:]
                    pad = np.zeros((max(0, plan_steps - len(remaining)), 2), dtype=np.float64)
                    warm_mean = np.concatenate([remaining, pad], axis=0)[:plan_steps]
                else:
                    warm_mean = None
        elif cfg.planner == "beam_mpc":
            macro_budget = max(1, cfg.eval_budget // max(1, cfg.frameskip))
            steps_remaining = macro_budget
            while steps_remaining > 0:
                plan_h = min(cfg.horizon, cfg.model_rollout_horizon, steps_remaining)
                planned, score, fails = plan_beam_search(
                    model,
                    current_obs,
                    goal_state,
                    horizon=plan_h,
                    beam_width=bw,
                    model_action_repeat=1,
                    current_state=current,
                )
                failed += fails
                if np.isfinite(score):
                    best_score = min(best_score, float(score))
                n_exec = min(cfg.receding_horizon, steps_remaining, plan_h)
                current = rollout_sim_macro_values(
                    env, current, planned[:n_exec], frameskip=cfg.frameskip
                )
                steps_remaining -= n_exec
                current_obs = graph_text_from_state(
                    current,
                    step=macro_budget - steps_remaining,
                    metadata={"episode_index": trial.episode_index},
                )
        else:
            raise ValueError(f"unsupported vpw planner {cfg.planner}")

        return {
            "trial_id": trial.trial_id,
            "success": float(pusht_success(goal_state, current)),
            "goal_distance": float(pusht_goal_distance(goal_state, current)),
            "model_distance": float(best_score) if np.isfinite(best_score) else None,
            "failed_model_rollouts": failed,
            "start": {
                "episode_index": trial.episode_index,
                "start_step": trial.start_step,
                "row_index": trial.row_index,
            },
        }
    finally:
        env.close()


def _worker(payload: Tuple[Dict[str, Any], Dict[str, Any]]) -> Dict[str, Any]:
    trial_dict, cfg_dict = payload
    # EGL in workers (headless MuJoCo).
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    trial = TrialSpec(**trial_dict)
    cfg = PlannerConfig(**cfg_dict)
    if cfg.dynamics == "mujoco":
        return _run_mujoco_trial(trial, cfg)
    if cfg.dynamics == "vpw":
        return _run_vpw_trial(trial, cfg)
    raise ValueError(f"unknown dynamics {cfg.dynamics}")


def evaluate_parallel(
    *,
    trials: Sequence[TrialSpec],
    cfg: PlannerConfig,
    num_workers: int = 16,
) -> Dict[str, Any]:
    workers = max(1, min(int(num_workers), len(trials)))
    cfg_dict = asdict(cfg)
    payloads = [(asdict(t), cfg_dict) for t in trials]

    results: List[Dict[str, Any]] = []
    if workers == 1:
        results = [_worker(p) for p in payloads]
    else:
        # spawn avoids forking after pygame/mujoco import in parent
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
            futures = [pool.submit(_worker, p) for p in payloads]
            for fut in as_completed(futures):
                results.append(fut.result())

    results.sort(key=lambda r: int(r["trial_id"]))
    successes = [float(r["success"]) for r in results]
    distances = [float(r["goal_distance"]) for r in results]
    model_dists = [r["model_distance"] for r in results if r.get("model_distance") is not None]
    return {
        "dynamics": cfg.dynamics,
        "planner": cfg.planner,
        "model_path": cfg.model_path or None,
        "model_type": "ground_truth_simulator" if cfg.dynamics == "mujoco" else "vpw_program",
        "num_eval": len(results),
        "num_workers": workers,
        "eval_budget": cfg.eval_budget,
        "frameskip": cfg.frameskip,
        "horizon": cfg.horizon,
        "receding_horizon": cfg.receding_horizon,
        "model_rollout_horizon": cfg.model_rollout_horizon if cfg.dynamics == "vpw" else None,
        "action_block": cfg.action_block if cfg.planner == "cem_mpc" else None,
        "num_samples": cfg.num_samples,
        "cem_iters": cfg.cem_iters if cfg.planner == "cem_mpc" else 0,
        "beam_width": _beam_width(cfg.num_samples) if cfg.planner == "beam_mpc" else None,
        "elite_frac": cfg.elite_frac if cfg.planner == "cem_mpc" else None,
        "seed": cfg.seed,
        "success_rate": 100.0 * float(np.mean(successes)) if successes else 0.0,
        "mean_goal_distance": float(np.mean(distances)) if distances else None,
        "mean_planner_model_distance": float(np.mean(model_dists)) if model_dists else None,
        "failed_model_rollouts": int(sum(int(r.get("failed_model_rollouts", 0)) for r in results)),
        "episode_successes": successes,
        "starts": [r["start"] for r in results],
    }
