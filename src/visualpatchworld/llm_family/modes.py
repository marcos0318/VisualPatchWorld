"""Contact-mode labels for transitions, from identified contact geometry.

A transition's mode is ``free`` when the body did not move, ``onset`` when the body
starts moving after being at rest, and otherwise the face of the identified (grown)
contact shape nearest to the pusher at the end of the transition. Without contact
geometry the label falls back to which objects moved. Labels let fitting weight rare
modes as heavily as common ones and let refinement report error per mode: one-step
error averaged over all contacts hides rare faces, which planners rely on to turn and
align a body and then exploit when the model is wrong there.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .traces import Episode

_FACES = ("u_min", "u_max", "v_min", "v_max")


def _shape(meta: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    s = meta.get("contact_shape")
    return s if isinstance(s, Mapping) and s.get("boxes_u_min_u_max_v_min_v_max") else None


def _segments(boxes: Sequence[Sequence[float]]) -> List[Tuple[str, Tuple[float, float, float, float]]]:
    """Exterior face segments (u0, v0, u1, v1) of a box union; faces whose midpoint lies
    strictly inside another box are interior and dropped."""
    segs = []
    for i, (u0, u1, v0, v1) in enumerate(boxes):
        cand = {"u_min": (u0, v0, u0, v1), "u_max": (u1, v0, u1, v1), "v_min": (u0, v0, u1, v0), "v_max": (u0, v1, u1, v1)}
        for f in _FACES:
            a, b, c, d = cand[f]
            mu, mv = (a + c) / 2, (b + d) / 2
            inside = any(j != i and o[0] + 1 < mu < o[1] - 1 and o[2] + 1 < mv < o[3] - 1 for j, o in enumerate(boxes))
            if not inside:
                segs.append((f"box{i}.{f}", (a, b, c, d)))
    return segs


def _seg_dist(p: Tuple[float, float], s: Tuple[float, float, float, float]) -> float:
    (pu, pv), (a, b, c, d) = p, s
    du, dv = c - a, d - b
    t = 0.0 if du * du + dv * dv == 0 else max(0.0, min(1.0, ((pu - a) * du + (pv - b) * dv) / (du * du + dv * dv)))
    return math.hypot(pu - a - t * du, pv - b - t * dv)


def _moved_body(s, n, body: str, tol: float) -> bool:
    da = (n[f"{body}.angle"] - s[f"{body}.angle"] + math.pi) % (2 * math.pi) - math.pi
    return math.hypot(n[f"{body}.x"] - s[f"{body}.x"], n[f"{body}.y"] - s[f"{body}.y"]) > tol or abs(da) > tol * 0.02


def label_episode(ep: Episode, keys: Sequence[str], scales: Mapping[str, float], *, move_tol: float = 0.05) -> List[str]:
    ts = ep.transitions
    if not ts:
        return []
    shape = _shape(ts[0].meta)
    if shape is None or f"{shape.get('body')}.angle" not in ts[0].state:
        objs = sorted({k.split(".", 1)[0] for k in keys})
        out = []
        for t in ts:
            moved = [o for o in objs if any(abs(t.next_state[k] - t.state[k]) / scales[k] > 1e-3
                                             for k in keys if k.startswith(o + ".") and k in t.next_state)]
            out.append("moved:" + "+".join(moved) if moved else "static")
        return out
    body, pusher = shape["body"], shape["pusher"]
    segs = _segments(shape["boxes_u_min_u_max_v_min_v_max"])
    out, prev = [], False
    for t in ts:
        n = t.next_state
        moved = _moved_body(t.state, n, body, move_tol)
        if not moved:
            out.append("free")
        elif not prev:
            out.append("onset")
        else:
            c, s = math.cos(n[f"{body}.angle"]), math.sin(n[f"{body}.angle"])
            dx, dy = n[f"{pusher}.x"] - n[f"{body}.x"], n[f"{pusher}.y"] - n[f"{body}.y"]
            p = (c * dx + s * dy, -s * dx + c * dy)
            out.append(min(segs, key=lambda sg: _seg_dist(p, sg[1]))[0])
        prev = moved
    return out


def label_episodes(episodes: Sequence[Episode], keys: Sequence[str], scales: Mapping[str, float]) -> Dict[int, str]:
    """id(transition) -> mode label."""
    out: Dict[int, str] = {}
    for ep in episodes:
        for t, m in zip(ep.transitions, label_episode(ep, keys, scales)):
            out[id(t)] = m
    return out


def covering_episodes(episodes: Sequence[Episode], keys: Sequence[str], scales: Mapping[str, float], n: int) -> List[Episode]:
    """Greedily pick ``n`` episodes so every mode is represented: each pick maximizes
    sum over modes of log(1 + count after) - log(1 + count before)."""
    per_ep = [Counter(label_episode(ep, keys, scales)) for ep in episodes]
    have: Counter = Counter()
    chosen: List[int] = []
    left = set(range(len(episodes)))
    for _ in range(min(n, len(episodes))):
        gain = lambda i: sum(math.log1p(have[m] + c) - math.log1p(have[m]) for m, c in per_ep[i].items())
        i = max(sorted(left), key=gain)
        chosen.append(i)
        left.discard(i)
        have.update(per_ep[i])
    return [episodes[i] for i in sorted(chosen)]


def window_mode(win: Sequence[Any], labels: Mapping[int, str], counts: Mapping[str, int]) -> str:
    """A window takes the rarest mode among its transitions."""
    ms = [labels.get(id(t), "unlabelled") for t in win]
    return min(ms, key=lambda m: counts.get(m, 0))


def balanced_weights(episodes: Sequence[Episode], keys: Sequence[str], scales: Mapping[str, float], k: int, *,
                     power: float = 0.5, cap: float = 30.0, min_count: int = 5):
    """Per-window residual weights so every mode with at least ``min_count`` windows
    contributes comparably: w_m = (N / (M * n_m)) ** power, capped, normalized to mean 1.
    Returns (weight_fn(win) -> float, summary)."""
    from .fit import _windows

    labels = label_episodes(episodes, keys, scales)
    counts = Counter(labels.values())
    wins = _windows(episodes, k)
    wm = Counter(window_mode(w, labels, counts) for w in wins)
    big = {m: c for m, c in wm.items() if c >= min_count}
    n, m_ = sum(big.values()), max(1, len(big))
    raw = {m: min(cap, (n / (m_ * c)) ** power) if m in big else 1.0 for m, c in wm.items()}
    mean = sum(raw[m] * c for m, c in wm.items()) / max(1, sum(wm.values()))
    w = {m: v / mean for m, v in raw.items()}

    def fn(win) -> float:
        return w.get(window_mode(win, labels, counts), 1.0)

    return fn, {"window_counts": dict(wm), "weights": {m: round(v, 3) for m, v in w.items()}}


def per_mode_errors(cand, params, episodes: Sequence[Episode], keys: Sequence[str], scales: Mapping[str, float], *,
                    horizons: Sequence[int] = (1, 5, 25), stride: int = 5) -> Dict[str, Dict[str, Any]]:
    """Per mode: count, and normalized RMSE of the rollout state after h steps for each h,
    over windows starting at a transition of that mode (start every ``stride``-th
    transition for h > 1)."""
    from .fit import _safe_step

    labels = label_episodes(episodes, keys, scales)
    acc: Dict[str, Dict[int, List[float]]] = {}
    hmax = max(horizons)
    for ep in episodes:
        ts = ep.transitions
        for s0, t0 in enumerate(ts):
            m = labels.get(id(t0), "unlabelled")
            hs = [h for h in horizons if s0 + h <= len(ts) and (h == 1 or s0 % stride == 0 or m not in ("free", "static"))]
            if not hs:
                continue
            state = dict(t0.state)
            for j in range(max(hs)):
                pred = _safe_step(cand, state, ts[s0 + j].action, dict(params), ts[s0 + j].meta)
                if pred is None:
                    for h in hs:
                        if h > j:
                            acc.setdefault(m, {}).setdefault(h, []).append(1e3)
                    break
                state = {key: float(pred.get(key, state.get(key, 0.0))) for key in state}
                if j + 1 in hs:
                    tn = ts[s0 + j].next_state
                    e = [(state[key] - tn[key]) / scales[key] for key in keys if key in tn]
                    e = [x if math.isfinite(x) else 1e3 for x in e]
                    acc.setdefault(m, {}).setdefault(j + 1, []).append(float(np.mean(np.square(e))))
                if j + 1 >= hmax:
                    break
    out = {}
    for m, d in acc.items():
        out[m] = {"n": len(d.get(1, [])), **{f"nrmse@{h}": float(np.sqrt(np.mean(v))) for h, v in sorted(d.items())}}
    return out


def balanced_score(per_mode: Mapping[str, Mapping[str, Any]], h: int, *, min_count: int = 5) -> float:
    """Mean over modes (with >= ``min_count`` samples) of the h-step nRMSE; when episodes are
    shorter than h, the longest horizon that any mode has."""
    hs = sorted({int(k.split("@")[1]) for d in per_mode.values() for k in d if k.startswith("nrmse@")}, reverse=True)
    for hh in [h] + [x for x in hs if x < h]:
        vals = [d[f"nrmse@{hh}"] for d in per_mode.values() if d.get("n", 0) >= min_count and f"nrmse@{hh}" in d]
        if vals:
            return float(np.mean(vals))
    return float("inf")
