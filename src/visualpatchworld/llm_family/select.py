"""Select one fitted candidate: offline (held-out rollout) or active (disagreement probes)."""

from __future__ import annotations

import itertools
import math
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .fit import FitResult, _safe_step
from .propose import Candidate
from .traces import Episode


def select_offline(fits: Mapping[str, FitResult]) -> Tuple[str, Dict[str, Any]]:
    scored = {n: f.val_rollout for n, f in fits.items() if math.isfinite(f.val_rollout)}
    winner = min(scored, key=scored.get)
    return winner, {"criterion": "val_rollout_nrmse", "scores": scored}


def _predict(cand: Candidate, params, start, actions, meta) -> Optional[List[Dict[str, float]]]:
    traj = [dict(start)]
    state = dict(start)
    for a in actions:
        nxt = _safe_step(cand, state, a, dict(params), meta)
        if nxt is None:
            return None
        try:
            state = {k: float(nxt.get(k, state[k])) for k in state}
        except (TypeError, ValueError):
            return None
        traj.append(state)
    return traj


def _traj_dist(a, b, keys, scales) -> float:
    total, n = 0.0, 0
    for sa, sb in zip(a[1:], b[1:]):
        for k in keys:
            d = (sa[k] - sb[k]) / scales[k]
            total += d * d
            n += 1
    return math.sqrt(total / max(1, n))


def select_active(
    cands: Mapping[str, Candidate],
    fits: Mapping[str, FitResult],
    sim: Any,
    probe_pool: Sequence[Episode],
    keys: Sequence[str],
    scales: Mapping[str, float],
    *,
    action_low: Sequence[float],
    action_high: Sequence[float],
    num_proposals: int = 200,
    num_probes: int = 5,
    probe_len: int = 3,
    seed: int = 0,
) -> Tuple[str, Dict[str, Any]]:
    """Run the ``num_probes`` probes with the largest predicted disagreement."""
    rng = np.random.default_rng(seed)
    names = list(cands)
    starts = [t for ep in probe_pool for t in ep.transitions if "raw_index" in t.source]
    # Keys the data never changes (e.g. a static goal) carry no signal about the
    # transition law, and the simulator may re-sample them on reset.
    moving = {k for ep in probe_pool for t in ep.transitions for k in keys if abs(t.next_state.get(k, 0.0) - t.state.get(k, 0.0)) > 1e-9}
    keys = [k for k in keys if k in moving] or list(keys)
    low, high = np.asarray(action_low, float), np.asarray(action_high, float)
    t0 = time.time()

    proposals = []
    for _ in range(num_proposals):
        t = starts[int(rng.integers(len(starts)))]
        if rng.random() < 0.5:
            a = rng.uniform(low, high)
            actions = [a.tolist()] * probe_len
        else:
            actions = [rng.uniform(low, high).tolist() for _ in range(probe_len)]
        preds = {n: _predict(cands[n], fits[n].params, t.state, actions, t.meta) for n in names}
        valid = [n for n in names if preds[n] is not None]
        if len(valid) < 2:
            continue
        pair = [_traj_dist(preds[x], preds[y], keys, scales) for x, y in itertools.combinations(valid, 2)]
        proposals.append((float(np.mean(pair)), t, actions, preds))
    proposals.sort(key=lambda p: -p[0])

    errors = {n: [] for n in names}
    log = []
    for disagreement, t, actions, preds in proposals[:num_probes]:
        observed = sim.rollout(t.source, actions)
        start = {**t.state, **observed[0]}
        row = {"disagreement": disagreement, "actions": actions, "start": start, "errors": {}}
        for n in names:
            pred = _predict(cands[n], fits[n].params, start, actions, t.meta)
            e = _traj_dist(pred, observed, keys, scales) if pred is not None else float("inf")
            errors[n].append(e)
            row["errors"][n] = e
        log.append(row)
    mean_err = {n: float(np.mean(v)) if v else float("inf") for n, v in errors.items()}
    winner = min(mean_err, key=mean_err.get)
    return winner, {
        "criterion": "mean_probe_rollout_nrmse",
        "scored_keys": keys,
        "scores": mean_err,
        "probes": log,
        "sim_resets": getattr(sim, "resets", None),
        "sim_steps": getattr(sim, "steps", None),
        "elapsed_s": time.time() - t0,
    }
