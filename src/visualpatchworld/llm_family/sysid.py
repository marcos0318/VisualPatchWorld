"""Contact-geometry identification from state traces (no shape metadata).

For an object pair (pusher, body) where the body carries a pose (x, y, angle) and moves
only in some transitions, the pusher's position expressed in the body frame at the end
of transitions where the body moved lies on the boundary of the body's shape grown by
the pusher's radius (a Minkowski sum), and no pusher position ever lies inside it. We fit
that grown shape as a union of K body-frame boxes by robust least squares on signed
distance: contact points should have signed distance 0, all pusher positions must have
signed distance >= 0. K is chosen on held-out points. Only the grown shape is
identifiable; it is exactly what a point-pusher contact test needs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import least_squares

from .traces import Episode


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def _objects(keys: Sequence[str]) -> List[str]:
    return sorted({k.split(".", 1)[0] for k in keys})


def find_pairs(episodes: Sequence[Episode], *, move_tol: float = 0.05, max_moving_frac: float = 0.9) -> List[Tuple[str, str]]:
    """(pusher, body) pairs: bodies have x, y, angle and are at rest in a sizeable fraction
    of transitions; pushers are every other object with x, y."""
    keys = sorted(episodes[0].transitions[0].state)
    objs = _objects(keys)
    posed = [o for o in objs if {f"{o}.x", f"{o}.y", f"{o}.angle"} <= set(keys)]
    pairs = []
    for b in posed:
        moves = [_moved(t.state, t.next_state, b, move_tol) for ep in episodes for t in ep.transitions]
        if 0.02 < np.mean(moves) < max_moving_frac:
            pairs += [(a, b) for a in objs if a != b and {f"{a}.x", f"{a}.y"} <= set(keys)]
    return pairs


def _moved(s, n, b, tol) -> bool:
    return math.hypot(n[f"{b}.x"] - s[f"{b}.x"], n[f"{b}.y"] - s[f"{b}.y"]) > tol or abs(_wrap(n[f"{b}.angle"] - s[f"{b}.angle"])) > tol * 0.02


def body_frame_points(episodes: Sequence[Episode], pusher: str, body: str, *, move_tol: float = 0.05):
    contact, free = [], []
    for ep in episodes:
        for t in ep.transitions:
            for st, dst in ((t.next_state, None),):
                c, s = math.cos(st[f"{body}.angle"]), math.sin(st[f"{body}.angle"])
                dx, dy = st[f"{pusher}.x"] - st[f"{body}.x"], st[f"{pusher}.y"] - st[f"{body}.y"]
                p = (c * dx + s * dy, -s * dx + c * dy)
                (contact if _moved(t.state, t.next_state, body, move_tol) else free).append(p)
    return np.asarray(contact), np.asarray(free)


def box_sdf(P: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Signed distance from points (N,2) to a union of boxes (K,4: u0,u1,v0,v1)."""
    out = np.full(len(P), np.inf)
    for u0, u1, v0, v1 in boxes:
        cu, cv, hu, hv = (u0 + u1) / 2, (v0 + v1) / 2, abs(u1 - u0) / 2, abs(v1 - v0) / 2
        qu, qv = np.abs(P[:, 0] - cu) - hu, np.abs(P[:, 1] - cv) - hv
        d = np.hypot(np.maximum(qu, 0), np.maximum(qv, 0)) + np.minimum(np.maximum(qu, qv), 0)
        out = np.minimum(out, d)
    return out


@dataclass
class ShapeFit:
    pusher: str
    body: str
    boxes: List[List[float]]
    k: int
    heldout_contact_mad: float
    heldout_penetration_frac: float
    n_contact: int
    n_free: int

    def to_meta(self) -> Dict[str, object]:
        return {
            "type": "union_of_boxes_in_body_frame",
            "identified_from": "state traces (contact analysis)",
            "pusher": self.pusher,
            "note": "boxes are the body shape grown by the pusher radius: the pusher CENTRE touches the body when it reaches a box boundary",
            "pose_convention": f"a body-frame point (u, v) is at world ({self.body}.x + u*cos({self.body}.angle) - v*sin({self.body}.angle), "
                               f"{self.body}.y + u*sin({self.body}.angle) + v*cos({self.body}.angle))",
            "boxes_u_min_u_max_v_min_v_max": [[round(x, 1) for x in b] for b in self.boxes],
        }


def _fit_k(C: np.ndarray, F: np.ndarray, k: int, rng, restarts: int, pen_w: float) -> Tuple[np.ndarray, float]:
    lo, hi = np.percentile(C, 1, axis=0), np.percentile(C, 99, axis=0)
    Fs = F[rng.choice(len(F), min(len(F), 6000), replace=False)] if len(F) else F
    Cs = C[rng.choice(len(C), min(len(C), 6000), replace=False)]

    def res(x):
        boxes = x.reshape(k, 4)
        rc = box_sdf(Cs, boxes)
        rf = np.minimum(box_sdf(Fs, boxes), 0.0) * pen_w if len(Fs) else np.zeros(0)
        return np.concatenate([rc, rf])

    best = None
    for _ in range(restarts):
        x0 = []
        for _ in range(k):
            a, b = rng.uniform(lo, hi), rng.uniform(lo, hi)
            x0 += [min(a[0], b[0]), max(a[0], b[0]), min(a[1], b[1]), max(a[1], b[1])]
        sol = least_squares(res, np.asarray(x0), loss="soft_l1", f_scale=2.0, max_nfev=400)
        cost = float(sol.cost)
        if best is None or cost < best[1]:
            best = (sol.x.reshape(k, 4), cost)
    boxes = best[0]
    boxes = np.stack([np.minimum(boxes[:, 0], boxes[:, 1]), np.maximum(boxes[:, 0], boxes[:, 1]),
                      np.minimum(boxes[:, 2], boxes[:, 3]), np.maximum(boxes[:, 2], boxes[:, 3])], axis=1)
    return boxes, best[1]


def extend_touching_faces(boxes: Sequence[Sequence[float]], tol: float = 3.0) -> List[List[float]]:
    """Extend each box face that touches, and is fully covered by, another box into that
    box's centre. The grown
    union is unchanged, but shrinking every box by the pusher radius no longer opens gaps
    where boxes only touched."""
    out = [list(map(float, b)) for b in boxes]
    for i, a in enumerate(out):
        for j, b in enumerate(boxes):
            if i == j:
                continue
            u_overlap = a[0] >= b[0] - tol and a[1] <= b[1] + tol
            v_overlap = a[2] >= b[2] - tol and a[3] <= b[3] + tol
            cu, cv = (b[0] + b[1]) / 2, (b[2] + b[3]) / 2
            if u_overlap:
                if b[2] - tol <= a[2] <= b[3] + tol and cv < a[2]:
                    a[2] = min(a[2], cv)
                if b[2] - tol <= a[3] <= b[3] + tol and cv > a[3]:
                    a[3] = max(a[3], cv)
            if v_overlap:
                if b[0] - tol <= a[0] <= b[1] + tol and cu < a[0]:
                    a[0] = min(a[0], cu)
                if b[0] - tol <= a[1] <= b[1] + tol and cu > a[1]:
                    a[1] = max(a[1], cu)
    return out


def box_union_mass_properties(boxes: Sequence[Sequence[float]]) -> Tuple[float, float, float]:
    """Uniform-density centroid (u, v) and polar moment per unit mass of a union of
    (assumed non-overlapping after clipping) boxes. Used only as an initial guess."""
    areas, cents, moms = [], [], []
    for u0, u1, v0, v1 in boxes:
        w, h = max(u1 - u0, 1e-6), max(v1 - v0, 1e-6)
        areas.append(w * h)
        cents.append(((u0 + u1) / 2, (v0 + v1) / 2))
        moms.append((w * w + h * h) / 12.0)
    A = sum(areas)
    cu = sum(a * c[0] for a, c in zip(areas, cents)) / A
    cv = sum(a * c[1] for a, c in zip(areas, cents)) / A
    j = sum(a * (m + (c[0] - cu) ** 2 + (c[1] - cv) ** 2) for a, m, c in zip(areas, moms, cents)) / A
    return cu, cv, j


# A general pusher/body contact program: the pusher tracks a PD target; a passive rigid body
# whose (grown) shape comes from ``meta["contact_shape"]`` receives frictionless normal
# impulses at its boundary, rotating about a fitted centre of mass; no body momentum.
PUSHER_BODY_SOURCE = r'''# HYPOTHESIS: pd_pusher_impulse_body_identified_shape
# MECHANISM: PD-tracking point pusher; frictionless impulse contact with a quasi-static rigid body whose grown shape was identified from data; rotation about a fitted centre of mass; several substeps per control step.
PARAMS = {"action_scale": 10.0, "kp": 10.0, "kv": 1.0, "dt": 0.01, "substeps": 10,
          "cog_u": 0.0, "cog_v": 0.0, "inertia": 1000.0, "bias": 0.05, "radius": 5.0, "slop": 0.0, "iters": 4}

def step(state, action, p, meta):
    import math
    shape = meta.get("contact_shape", {})
    P, B = shape.get("pusher", "agent"), shape.get("body", "block")
    boxes = shape.get("boxes_u_min_u_max_v_min_v_max", [])
    ax, ay = state[P + ".x"], state[P + ".y"]
    vx, vy = state.get(P + ".vx", 0.0), state.get(P + ".vy", 0.0)
    bx, by, th = state[B + ".x"], state[B + ".y"], state[B + ".angle"]
    a0 = max(-1.0, min(1.0, action[0])) if len(action) > 0 else 0.0
    a1 = max(-1.0, min(1.0, action[1])) if len(action) > 1 else 0.0
    tx, ty = ax + p["action_scale"] * a0, ay + p["action_scale"] * a1
    n_sub = max(1, int(round(p["substeps"])))
    dt = p["dt"]
    inertia = max(abs(p["inertia"]), 1e-6)
    cu, cv = p["cog_u"], p["cog_v"]
    r = max(p["radius"], 0.0)
    for _ in range(n_sub):
        vx += (p["kp"] * (tx - ax) - p["kv"] * vx) * dt
        vy += (p["kp"] * (ty - ay) - p["kv"] * vy) * dt
        c, s = math.cos(th), math.sin(th)
        dx, dy = ax - bx, ay - by
        lu, lv = c * dx + s * dy, -s * dx + c * dy
        contacts = []
        gx, gy = bx + c * cu - s * cv, by + s * cu + c * cv
        for g0, g1, h0, h1 in boxes:
            u0, u1 = min(g0 + r, 0.5 * (g0 + g1)), max(g1 - r, 0.5 * (g0 + g1))
            v0, v1 = min(h0 + r, 0.5 * (h0 + h1)), max(h1 - r, 0.5 * (h0 + h1))
            qu, qv = min(max(lu, u0), u1), min(max(lv, v0), v1)
            eu, ev = lu - qu, lv - qv
            d = math.hypot(eu, ev)
            if d > 1e-9:
                if d >= r:
                    continue
                nu, nv, pen = eu / d, ev / d, r - d
            else:
                g, nu, nv = min((lu - u0, -1.0, 0.0), (u1 - lu, 1.0, 0.0), (lv - v0, 0.0, -1.0), (v1 - lv, 0.0, 1.0))
                qu, qv, pen = lu + nu * g, lv + nv * g, r + g
            wnx, wny = c * nu - s * nv, s * nu + c * nv
            px, py = bx + c * qu - s * qv, by + s * qu + c * qv
            contacts.append([px - gx, py - gy, wnx, wny, max(pen - p["slop"], 0.0), 0.0])
        vbx = vby = w = 0.0
        for _ in range(max(1, int(round(p["iters"])))):
            for ct in contacts:
                rx, ry, nx, ny, pen, acc = ct
                rn = rx * ny - ry * nx
                k = 1.0 + rn * rn / inertia
                vrel = (vx - (vbx - w * ry)) * nx + (vy - (vby + w * rx)) * ny
                new = max(acc + (p["bias"] * pen / dt - vrel) / k, 0.0)
                dj = new - acc
                ct[5] = new
                vbx -= dj * nx
                vby -= dj * ny
                w -= dj * rn / inertia
        gx, gy = bx + c * cu - s * cv, by + s * cu + c * cv
        gx, gy, th = gx + vbx * dt, gy + vby * dt, th + w * dt
        c2, s2 = math.cos(th), math.sin(th)
        bx, by = gx - (c2 * cu - s2 * cv), gy - (s2 * cu + c2 * cv)
        ax, ay = ax + vx * dt, ay + vy * dt
    out = dict(state)
    out.update({P + ".x": ax, P + ".y": ay, B + ".x": bx, B + ".y": by, B + ".angle": th % (2.0 * math.pi)})
    if P + ".vx" in out:
        out[P + ".vx"], out[P + ".vy"] = vx, vy
    return out
'''


def with_meta(episodes, meta):
    from types import SimpleNamespace

    return [SimpleNamespace(transitions=[SimpleNamespace(state=t.state, action=t.action, next_state=t.next_state,
                                                         meta={**t.meta, **meta}) for t in ep.transitions]) for ep in episodes]


def _fit_subset(cand, params, names, train, val, keys, scales, *, nfev, restarts=1, seed=0):
    from types import SimpleNamespace

    from .fit import fit_candidate

    fixed = {k: v for k, v in params.items() if k not in names}
    sub = SimpleNamespace(step=lambda s, a, q, m: cand.step(s, a, {**fixed, **q}, m), params={k: params[k] for k in names})
    fr = fit_candidate(sub, train, val, keys, scales, objective="multi_step", rollout_k=5, restarts=restarts, max_nfev=nfev, seed=seed)
    return {**params, **fr.params}, fr


def identify_pusher_body(
    train_full: Sequence[Episode],
    val: Sequence[Episode],
    keys: Sequence[str],
    scales: Dict[str, float],
    *,
    substeps: Sequence[int] = (10, 5, 1),
    fit_episodes: int = 10,
    log=print,
) -> Optional[Dict[str, object]]:
    """Staged identification of the pusher/body contact program from traces only:
    shape (box union), control period, pusher constants, radius (grid), body constants,
    joint polish; the substep count is chosen on held-out 5-step nRMSE.
    Returns None when no pusher/body pair is found."""
    import time
    from types import SimpleNamespace

    from .fit import nrmse
    from .propose import compile_candidate

    t0 = time.time()
    pairs = find_pairs(train_full)
    log(f"[sysid] candidate pairs: {pairs}")
    if not pairs:
        return None
    pusher, body = pairs[0]
    shape = identify_shape(train_full, pusher, body)
    if shape is None:
        return None
    meta = {"contact_shape": {**shape.to_meta(), "body": body}}
    log(f"[sysid] shape K={shape.k} boxes={meta['contact_shape']['boxes_u_min_u_max_v_min_v_max']} "
        f"held-out contact err={shape.heldout_contact_mad:.2f} ({shape.n_contact} contacts)")

    def motion(ep):
        return sum(_moved(t.state, t.next_state, body, 0.05) for t in ep.transitions)

    fit_eps = sorted(train_full, key=motion, reverse=True)[:fit_episodes]
    tr, va = with_meta(fit_eps, meta), with_meta(val, meta)
    dxs, vs = [], []
    for ep in fit_eps:
        for t in ep.transitions:
            for ax in ("x", "y"):
                if f"{pusher}.v{ax}" in t.next_state:
                    dxs.append(t.next_state[f"{pusher}.{ax}"] - t.state[f"{pusher}.{ax}"])
                    vs.append(0.5 * (t.state[f"{pusher}.v{ax}"] + t.next_state[f"{pusher}.v{ax}"]))
    T = float(np.dot(dxs, vs) / max(np.dot(vs, vs), 1e-9)) if vs else 0.1
    cu, cv, j = box_union_mass_properties(shape.boxes)
    log(f"[sysid] control period from data T={T:.4f}; shape centroid=({cu:.1f},{cv:.1f}) inertia/m={j:.0f}")

    cand = compile_candidate(PUSHER_BODY_SOURCE)
    c = SimpleNamespace(step=cand.step)
    pk = [k for k in keys if k.startswith(pusher + ".")]
    bk = [k for k in keys if k.startswith(body + ".")]
    r_max = 0.5 * min(min(b[1] - b[0], b[3] - b[2]) for b in shape.boxes)

    def grid_radius(params):
        # Contact switching makes the loss non-smooth in the radius; search it on a grid (training loss).
        grid = np.arange(0.0, r_max, 1.25)
        losses = [nrmse(c, {**params, "radius": float(g)}, tr, bk, scales, k=5) for g in grid]
        return {**params, "radius": float(grid[int(np.argmin(losses))])}

    best, runs = None, []
    for n in substeps:
        params = {**cand.params, "substeps": float(n), "dt": T / n, "cog_u": cu, "cog_v": cv, "inertia": j}
        params, fr_a = _fit_subset(cand, params, ["action_scale", "kp", "kv", "dt"], tr, va, pk, scales, nfev=80, restarts=3)
        params = grid_radius(params)
        params, fr_b = _fit_subset(cand, params, ["cog_u", "cog_v", "inertia", "bias"], tr, va, bk, scales, nfev=60)
        params = grid_radius(params)
        params, _ = _fit_subset(cand, params, ["action_scale", "kp", "kv", "dt", "cog_u", "cog_v", "inertia", "bias"], tr, va, keys, scales, nfev=60)
        row = {"substeps": n, "pusher_val5": fr_a.val_rollout, "body_val5": fr_b.val_rollout,
               "val1": nrmse(c, params, va, keys, scales, k=1), "val5": nrmse(c, params, va, keys, scales, k=5), "params": params}
        runs.append(row)
        log(f"[sysid] substeps={n:2d} val1={row['val1']:.4f} val5={row['val5']:.4f} "
            f"params={ {k: round(v, 3) for k, v in params.items()} } ({time.time() - t0:.0f}s)")
        if best is None or row["val5"] < best["val5"]:
            best = row
    return {"candidate": cand, "params": best["params"], "meta": meta, "best": best, "runs": runs,
            "shape_fit": shape.__dict__, "control_period": T, "init_mass_props": [cu, cv, j], "elapsed_s": time.time() - t0}


def identify_shape(
    episodes: Sequence[Episode],
    pusher: str,
    body: str,
    *,
    max_k: int = 3,
    restarts: int = 12,
    pen_w: float = 3.0,
    seed: int = 0,
    k_tol: float = 0.15,
) -> Optional[ShapeFit]:
    C, F = body_frame_points(episodes, pusher, body)
    if len(C) < 50:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(C))
    Ctr, Cte = C[idx[: len(C) * 4 // 5]], C[idx[len(C) * 4 // 5 :]]
    fidx = rng.permutation(len(F))
    Ftr, Fte = F[fidx[: len(F) * 4 // 5]], F[fidx[len(F) * 4 // 5 :]]
    fits = []
    for k in range(1, max_k + 1):
        boxes, _ = _fit_k(Ctr, Ftr, k, rng, restarts, pen_w)
        mad = float(np.mean(np.minimum(np.abs(box_sdf(Cte, boxes)), 20.0)))
        pen = float(np.mean(box_sdf(Fte, boxes) < -2.0)) if len(Fte) else 0.0
        fits.append((k, boxes, mad, pen))
    # Smallest K whose held-out contact error is within k_tol (relative) of the best.
    best_mad = min(f[2] for f in fits)
    k, boxes, mad, pen = next(f for f in fits if f[2] <= best_mad * (1 + k_tol) + 0.25)
    return ShapeFit(pusher, body, extend_touching_faces(boxes.tolist()), k, mad, pen, len(C), len(F))
