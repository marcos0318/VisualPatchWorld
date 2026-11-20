"""Fit the free constants of a candidate program from offline traces."""

from __future__ import annotations

import contextlib
import math
import signal
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import least_squares

from .propose import Candidate
from .traces import Episode

_BAD = 1e3


class FitTimeout(BaseException):
    """Raised by :func:`time_budget`. A BaseException so the ``except Exception`` guards
    around optimizer calls do not swallow it."""


@contextlib.contextmanager
def time_budget(seconds: float):
    """Abort the enclosed block after ``seconds`` (main thread, Unix). Candidate programs
    may loop over a fitted constant (e.g. ``range(int(p["substeps"]))``) that the
    optimizer can drive arbitrarily large."""
    if not seconds or seconds <= 0:
        yield
        return

    def _raise(signum, frame):
        raise FitTimeout(f"exceeded {seconds:.0f}s")

    old = signal.signal(signal.SIGALRM, _raise)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


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
    weight_fn: Optional[Callable[[Sequence[Any]], float]] = None,
) -> np.ndarray:
    """Normalized prediction residuals over all length-``k`` rollout windows, each
    window scaled by sqrt(weight_fn(window)) when given."""
    p = dict(params)
    out: List[float] = []
    for win in _windows(episodes, k):
        state = dict(win[0].state)
        sw = math.sqrt(weight_fn(win)) if weight_fn is not None else 1.0
        for t in win:
            pred = _safe_step(cand, state, t.action, p, t.meta)
            if pred is None:
                out.extend([_BAD * sw] * len(keys))
                break
            for key in keys:
                if key not in t.next_state:
                    continue
                try:
                    v = float(pred.get(key, state.get(key, 0.0)))
                except (TypeError, ValueError):
                    v = float("nan")
                r = (v - t.next_state[key]) / scales[key]
                out.append(r * sw if math.isfinite(r) else _BAD * sw)
            state = {key: float(pred.get(key, state.get(key, 0.0))) for key in state}
    return np.asarray(out, dtype=np.float64)


def nrmse(cand, params, episodes, keys, scales, *, k: int, weight_fn=None) -> float:
    r = residuals(cand, params, episodes, keys, scales, k=k, weight_fn=weight_fn)
    return float(np.sqrt(np.mean(np.minimum(r * r, _BAD * _BAD)))) if r.size else float("nan")


def _objects(keys: Sequence[str]) -> List[str]:
    return sorted({k.split(".", 1)[0] for k in keys})


def coverage_subset(episodes: Sequence[Episode], keys: Sequence[str], scales: Mapping[str, float], n: int,
                    tol: float = 1e-3) -> List[Episode]:
    """The ``n`` episodes in which the most rarely moving object moves most often, so that
    interaction-driven constants are identifiable from a small fitting subset."""
    objs = _objects(keys)

    def moves(ep, o):
        return sum(any(abs(t.next_state[k] - t.state[k]) / scales[k] > tol for k in keys
                       if k.startswith(o + ".") and k in t.next_state) for t in ep.transitions)

    counts = {o: sum(moves(ep, o) for ep in episodes) for o in objs}
    moving = [o for o in objs if counts[o] > 0]
    if not moving:
        return list(episodes[:n])
    rare = min(moving, key=lambda o: counts[o])
    return sorted(episodes, key=lambda ep: moves(ep, rare), reverse=True)[:n]


def param_dependencies(cand: Candidate, params: Mapping[str, float], episodes: Sequence[Episode],
                       keys: Sequence[str], n: int = 64) -> Dict[str, List[str]]:
    """Object -> constants whose perturbation changes that object's one-step prediction."""
    ts = [t for ep in episodes for t in ep.transitions][:n]
    base = [_safe_step(cand, t.state, t.action, dict(params), t.meta) or {} for t in ts]
    deps = {o: [] for o in _objects(keys)}
    for name, v in params.items():
        q = dict(params)
        q[name] = v * 1.1 + (1e-2 if abs(v) < 1e-6 else 0.0)
        pert = [_safe_step(cand, t.state, t.action, q, t.meta) or {} for t in ts]
        for o in deps:
            ok = [k for k in keys if k.startswith(o + ".")]
            if any(abs(float(a.get(k, 0.0)) - float(b.get(k, 0.0))) > 1e-9 for a, b in zip(base, pert) for k in ok):
                deps[o].append(name)
    return deps


def _grid(v: float) -> List[float]:
    if abs(v - round(v)) < 1e-9 and 1 <= v <= 50:
        return sorted({1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0, float(v)})
    if abs(v) < 1e-6:
        return [-1.0, -0.1, 0.0, 0.1, 1.0]
    return [v * m for m in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0)]


def structured_refit(
    cand: Candidate,
    params: Mapping[str, float],
    train: Sequence[Episode],
    val: Sequence[Episode],
    keys: Sequence[str],
    scales: Mapping[str, float],
    *,
    rollout_k: int = 5,
    subset: int = 10,
    max_nfev: int = 60,
    balance_modes: bool = False,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Identification-style refit: coverage subset, per-object staging by dependency,
    1-D grid search per constant (non-smooth losses such as contact switching), local
    least squares per stage, then a joint polish."""
    sub = coverage_subset(train, keys, scales, subset)
    wf = None
    if balance_modes:
        from .modes import balanced_weights

        wf, _ = balanced_weights(sub, keys, scales, rollout_k)
    p = dict(params)
    deps = param_dependencies(cand, p, sub, keys)
    order = sorted(deps, key=lambda o: len(deps[o]))
    done: List[str] = []
    stages = []
    for o in order:
        names = [n for n in deps[o] if n not in done]
        ok = [k for k in keys if k.startswith(o + ".")]
        if not names or not ok:
            continue
        for n in names:
            vals = _grid(p[n])
            losses = [nrmse(cand, {**p, n: g}, sub, ok, scales, k=rollout_k, weight_fn=wf) for g in vals]
            if np.isfinite(min(losses)):
                p[n] = float(vals[int(np.nanargmin(losses))])

        def fun(x, names=names, ok=ok):
            return residuals(cand, {**p, **dict(zip(names, x))}, sub, ok, scales, k=rollout_k, weight_fn=wf)

        try:
            sol = least_squares(fun, np.asarray([p[n] for n in names]), method="trf", max_nfev=max_nfev)
            p.update(dict(zip(names, map(float, sol.x))))
        except Exception:  # noqa: BLE001
            pass
        done += names
        stages.append({"object": o, "params": names})
    names = list(p)
    try:
        sol = least_squares(lambda x: residuals(cand, dict(zip(names, x)), sub, keys, scales, k=rollout_k, weight_fn=wf),
                            np.asarray([p[n] for n in names]), method="trf", max_nfev=max_nfev)
        p = dict(zip(names, map(float, sol.x)))
    except Exception:  # noqa: BLE001
        pass
    return p, {"dependencies": deps, "stages": stages}


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
    structured: bool = False,
    balance_modes: bool = False,
) -> FitResult:
    """``structured``: also run :func:`structured_refit` from the best restart and keep it
    if it has lower held-out rollout error. ``balance_modes``: weight training and
    held-out windows so each contact mode (see :mod:`.modes`) counts comparably."""
    names = list(cand.params)
    x0 = np.asarray([cand.params[n] for n in names], dtype=np.float64)
    k_fit = 1 if objective == "one_step" else rollout_k
    rng = np.random.default_rng(seed)
    t0 = time.time()
    wtr = wva = None
    if balance_modes:
        from .modes import balanced_weights

        wtr, _ = balanced_weights(train, keys, scales, k_fit)
        wva, _ = balanced_weights(val, keys, scales, rollout_k)
    best = None
    history = []
    nfev = 0
    for r in range(max(1, restarts)):
        if r == 0:
            init = x0.copy()
        else:
            init = x0 * np.exp(rng.normal(0.0, 0.5, size=x0.shape)) + rng.normal(0.0, 0.1, size=x0.shape)

        def fun(x: np.ndarray) -> np.ndarray:
            return residuals(cand, dict(zip(names, x)), train, keys, scales, k=k_fit, weight_fn=wtr)

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
        val_roll = nrmse(cand, params, val, keys, scales, k=rollout_k, weight_fn=wva)
        history.append({"restart": r, "train_loss": loss, "val_rollout_nrmse": val_roll})
        if best is None or (math.isfinite(val_roll) and val_roll < best[2]):
            best = (params, loss, val_roll)
    params, loss, val_roll = best
    if structured and names:
        for start_name, start in (("fitted", params), ("initial", dict(cand.params))):
            try:
                sp, info = structured_refit(cand, start, train, val, keys, scales, rollout_k=rollout_k,
                                            balance_modes=balance_modes)
                sv = nrmse(cand, sp, val, keys, scales, k=rollout_k, weight_fn=wva)
                history.append({"restart": f"structured_from_{start_name}", "val_rollout_nrmse": sv, **info})
                if math.isfinite(sv) and sv < val_roll:
                    params, val_roll = sp, sv
                    loss = float(np.mean(residuals(cand, sp, train, keys, scales, k=k_fit, weight_fn=wtr) ** 2))
            except Exception as exc:  # noqa: BLE001
                history.append({"restart": f"structured_from_{start_name}", "error": f"{type(exc).__name__}: {exc}"})
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
