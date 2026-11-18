"""Fit the free constants of a candidate program from offline traces."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
from scipy.optimize import least_squares

from .propose import Candidate
from .traces import Episode

_BAD = 1e3


@dataclass
class FitResult:
    params: Dict[str, float]
    objective: str
    rollout_k: int
    train_loss: float
    val_one_step: float
    val_rollout: float
    restarts: int
    nfev: int
    elapsed_s: float
    history: List[Dict[str, float]] = field(default_factory=list)

    def to_json(self) -> Dict[str, Any]:
        return {
            "params": dict(self.params),
            "objective": self.objective,
            "rollout_k": self.rollout_k,
            "train_loss": self.train_loss,
            "val_one_step_nrmse": self.val_one_step,
            "val_rollout_nrmse": self.val_rollout,
            "restarts": self.restarts,
            "nfev": self.nfev,
            "elapsed_s": self.elapsed_s,
            "restart_history": self.history,
        }


def _safe_step(cand: Candidate, state, action, params, meta) -> Optional[Dict[str, float]]:
    try:
        out = cand.step(dict(state), list(action), params, meta)
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(out, dict):
        return None
    return out


def _windows(episodes: Sequence[Episode], k: int) -> List[List[Any]]:
    wins = []
    for ep in episodes:
        ts = ep.transitions
        for s in range(0, max(1, len(ts) - k + 1)):
            wins.append(ts[s : s + k])
    return wins


def residuals(
    cand: Candidate,
    params: Mapping[str, float],
    episodes: Sequence[Episode],
    keys: Sequence[str],
    scales: Mapping[str, float],
    *,
    k: int,
) -> np.ndarray:
    """Normalized prediction residuals over all length-``k`` rollout windows."""
    p = dict(params)
    out: List[float] = []
    for win in _windows(episodes, k):
        state = dict(win[0].state)
        for t in win:
            pred = _safe_step(cand, state, t.action, p, t.meta)
            if pred is None:
                out.extend([_BAD] * len(keys))
                break
            for key in keys:
                if key not in t.next_state:
                    continue
                try:
                    v = float(pred.get(key, state.get(key, 0.0)))
                except (TypeError, ValueError):
                    v = float("nan")
                r = (v - t.next_state[key]) / scales[key]
                out.append(r if math.isfinite(r) else _BAD)
            state = {key: float(pred.get(key, state.get(key, 0.0))) for key in state}
    return np.asarray(out, dtype=np.float64)


def nrmse(cand, params, episodes, keys, scales, *, k: int) -> float:
    r = residuals(cand, params, episodes, keys, scales, k=k)
    return float(np.sqrt(np.mean(np.minimum(r * r, _BAD * _BAD)))) if r.size else float("nan")


def fit_candidate(
    cand: Candidate,
    train: Sequence[Episode],
    val: Sequence[Episode],
    keys: Sequence[str],
    scales: Mapping[str, float],
    *,
    objective: str = "multi_step",
    rollout_k: int = 5,
    restarts: int = 4,
    max_nfev: int = 400,
    seed: int = 0,
) -> FitResult:
    names = list(cand.params)
    x0 = np.asarray([cand.params[n] for n in names], dtype=np.float64)
    k_fit = 1 if objective == "one_step" else rollout_k
    rng = np.random.default_rng(seed)
    t0 = time.time()
    best = None
    history = []
    nfev = 0
    for r in range(max(1, restarts)):
        if r == 0:
            init = x0.copy()
        else:
            init = x0 * np.exp(rng.normal(0.0, 0.5, size=x0.shape)) + rng.normal(0.0, 0.1, size=x0.shape)

        def fun(x: np.ndarray) -> np.ndarray:
            return residuals(cand, dict(zip(names, x)), train, keys, scales, k=k_fit)

        if names:
            try:
                sol = least_squares(fun, init, method="trf", max_nfev=max_nfev)
                x, loss, used = sol.x, float(np.mean(sol.fun**2)), int(sol.nfev)
            except Exception:  # noqa: BLE001
                x, loss, used = init, float("inf"), 0
        else:
            x, loss, used = init, float(np.mean(fun(init) ** 2)), 1
        nfev += used
        params = dict(zip(names, map(float, x)))
        val_roll = nrmse(cand, params, val, keys, scales, k=rollout_k)
        history.append({"restart": r, "train_loss": loss, "val_rollout_nrmse": val_roll})
        if best is None or (math.isfinite(val_roll) and val_roll < best[2]):
            best = (params, loss, val_roll)
    params, loss, val_roll = best
    return FitResult(
        params=params,
        objective=objective,
        rollout_k=rollout_k,
        train_loss=loss,
        val_one_step=nrmse(cand, params, val, keys, scales, k=1),
        val_rollout=val_roll,
        restarts=restarts,
        nfev=nfev,
        elapsed_s=time.time() - t0,
        history=history,
    )
