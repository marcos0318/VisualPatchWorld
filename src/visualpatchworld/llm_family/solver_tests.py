"""Environment-agnostic physics tests for LLM-written contact solvers.

A solver has the signature ``solve(disc, body, dt, p)``: a kinematic disc
``{x, y, vx, vy, r}`` moves by ``(vx, vy) * dt`` during the step and may push a free
planar rigid body ``{x, y, ang, vx, vy, w, shape, com, mass, inertia}``, where ``(x, y)``
is the body-frame origin, ``shape`` is a union of body-frame boxes ``(u0, u1, v0, v1)`` and
``com`` the body-frame centre of mass. It returns the body's new
``{x, y, ang, vx, vy, w}``. The scenarios are synthetic shapes, never an evaluation
environment.
"""

from __future__ import annotations

import math
import random
import re
import signal
from typing import Any, Callable, Dict, List, Tuple

INTERFACE_DOC = """def solve(disc, body, dt, p):
    # disc: dict x, y, vx, vy, r. A kinematic pusher: (x, y) is its position at the START of
    #       the step; it moves by (vx, vy)*dt during the step whatever it touches, and is not
    #       affected by the body. r may be 0 (a point pusher against boxes already grown by
    #       the pusher radius); the pusher centre can then lie inside a box.
    # body: dict x, y, ang (pose of the body-frame origin), vx, vy, w (velocity of the centre
    #       of mass and angular velocity), shape (list of body-frame boxes (u0, u1, v0, v1)),
    #       com (body-frame centre of mass (cu, cv)), mass, inertia (about the centre of mass).
    #       Body frame: world = (x, y) + R(ang) (u, v).
    # dt:   step length. p: dict of numeric parameters with defaults you choose.
    # Return a dict with the body's new x, y, ang, vx, vy, w after the step."""

Shape = List[Tuple[float, float, float, float]]


def _R(a: float, u: float, v: float) -> Tuple[float, float]:
    c, s = math.cos(a), math.sin(a)
    return c * u - s * v, s * u + c * v


def _to_body(b: Dict[str, Any], px: float, py: float) -> Tuple[float, float]:
    c, s = math.cos(b["ang"]), math.sin(b["ang"])
    dx, dy = px - b["x"], py - b["y"]
    return c * dx + s * dy, -s * dx + c * dy


def _sdf(shape: Shape, u: float, v: float) -> float:
    best = float("inf")
    for u0, u1, v0, v1 in shape:
        du = max(u0 - u, 0.0, u - u1)
        dv = max(v0 - v, 0.0, v - v1)
        out = math.hypot(du, dv)
        inside = min(u - u0, u1 - u, v - v0, v1 - v)
        best = min(best, out if out > 0 else -inside)
    return best


def _com_world(b: Dict[str, Any]) -> Tuple[float, float]:
    du, dv = _R(b["ang"], *b["com"])
    return b["x"] + du, b["y"] + dv


def _body(shape: Shape, com, ang=0.0, x=0.0, y=0.0, mass=1.0, inertia=None) -> Dict[str, Any]:
    if inertia is None:
        us = [(u0 + u1) / 2 for u0, u1, _, _ in shape]
        inertia = mass * (max(u1 for _, u1, _, _ in shape) - min(u0 for u0, _, _, _ in shape)) ** 2 / 6.0
    return {"x": x, "y": y, "ang": ang, "vx": 0.0, "vy": 0.0, "w": 0.0, "shape": list(shape),
            "com": tuple(com), "mass": mass, "inertia": inertia}


class _Timeout(BaseException):
    pass


def _call(fn: Callable, disc, body, dt, p, timeout: float = 2.0) -> Dict[str, Any]:
    def _alarm(*_):
        raise _Timeout()

    old = signal.signal(signal.SIGALRM, _alarm)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        out = fn(dict(disc), {**body, "shape": list(body["shape"])}, dt, dict(p))
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)
    if not isinstance(out, dict):
        raise TypeError("solve must return a dict")
    return {k: float(out.get(k, body[k])) for k in ("x", "y", "ang", "vx", "vy", "w")}


BAR: Shape = [(-40.0, 40.0, -10.0, 10.0)]
ELL: Shape = [(-40.0, 40.0, -10.0, 10.0), (20.0, 40.0, 10.0, 70.0)]
DT = 0.1


def _push(body, side_dir, offset_u, speed, r=8.0, gap=2.0):
    """Disc next to the bar face in direction -side_dir (body frame), moving into it."""
    nu, nv = side_dir
    su = offset_u
    sv = (10.0 + r + gap) * (-nv) if nv else 0.0
    if nu:
        su, sv = (40.0 + r + gap) * (-nu), offset_u
    wx, wy = _R(body["ang"], su, sv)
    vx, vy = _R(body["ang"], nu * speed, nv * speed)
    return {"x": body["x"] + wx, "y": body["y"] + wy, "vx": vx, "vy": vy, "r": r}


def run_tests(fn: Callable, p: Dict[str, float] | None = None, *, seed: int = 0) -> List[Dict[str, Any]]:
    """Return a list of {name, passed, detail}."""
    p = dict(p or {})
    rng = random.Random(seed)
    results: List[Dict[str, Any]] = []

    def record(name, passed, detail):
        results.append({"name": name, "passed": bool(passed), "detail": detail})

    def safe(name, f):
        try:
            f()
        except _Timeout:
            record(name, False, "timed out (> 2 s for one call)")
        except Exception as e:  # noqa: BLE001
            record(name, False, f"raised {type(e).__name__}: {e}")

    def t_no_contact():
        worst = 0.0
        for _ in range(10):
            b = _body(ELL, (5.0, 15.0), ang=rng.uniform(-3, 3), x=rng.uniform(-50, 50), y=rng.uniform(-50, 50))
            ax, ay = _R(rng.uniform(0, 6.3), 1.0, 0.0)
            disc = {"x": b["x"] + 150 * ax, "y": b["y"] + 150 * ay, "vx": 100 * ax, "vy": 100 * ay, "r": 8.0}
            o = _call(fn, disc, b, DT, p)
            worst = max(worst, math.hypot(o["x"] - b["x"], o["y"] - b["y"]), 50 * abs(o["ang"] - b["ang"]))
        record("no_contact_no_motion", worst < 0.05,
               f"a body at rest with the disc far away and moving away moved by {worst:.3g} (expected 0)")

    def t_shape_aware():
        # disc in the notch of the L: inside the bounding box/circle, outside the boxes
        worst = 0.0
        for k in range(6):
            b = _body(ELL, (5.0, 15.0), ang=0.3 * k)
            u, v = -10.0 + 4 * k, 40.0
            wx, wy = _R(b["ang"], u, v)
            vx, vy = _R(b["ang"], 0.0, 20.0)
            disc = {"x": b["x"] + wx, "y": b["y"] + wy, "vx": vx, "vy": vy, "r": 6.0}
            o = _call(fn, disc, b, DT, p)
            worst = max(worst, math.hypot(o["x"] - b["x"], o["y"] - b["y"]), 50 * abs(o["ang"] - b["ang"]))
        record("shape_aware_detection", worst < 0.05,
               "a disc inside the notch of an L-shaped body (inside its bounding box and bounding "
               f"circle, but 4+ units from every box) moved the body by {worst:.3g}; detection must use "
               "the boxes, not a bounding circle or the distance between centres")

    def t_non_penetration():
        worst = -1e9
        for k in range(8):
            b = _body(BAR, (0.0, 0.0), ang=rng.uniform(-3, 3), x=rng.uniform(-30, 30), y=rng.uniform(-30, 30))
            side = [(0, 1), (0, -1), (1, 0), (-1, 0)][k % 4]
            disc = _push(b, side, rng.uniform(-25, 25) if side[1] else rng.uniform(-6, 6), speed=150.0)
            o = _call(fn, disc, b, DT, p)
            nb = {**b, **o}
            fx, fy = disc["x"] + disc["vx"] * DT, disc["y"] + disc["vy"] * DT
            depth = disc["r"] - _sdf(BAR, *_to_body(nb, fx, fy))
            worst = max(worst, depth)
        record("non_penetration", worst < 1.5,
               f"after pushing 15 units into the body, the disc's final position overlaps the body by "
               f"{worst:.3g} (allowed 1.5); the body must be moved out of the disc's way")

    def t_central_push():
        worst_rot, worst_dir = 0.0, 0.0
        for k in range(6):
            b = _body(BAR, (0.0, 0.0), ang=rng.uniform(-3, 3))
            side = [(0, 1), (0, -1), (1, 0), (-1, 0)][k % 4]
            disc = _push(b, side, 0.0, speed=150.0)
            o = _call(fn, disc, b, DT, p)
            dx, dy = o["x"] - b["x"], o["y"] - b["y"]
            d = math.hypot(dx, dy)
            worst_rot = max(worst_rot, abs(math.remainder(o["ang"] - b["ang"], 2 * math.pi)))
            if d < 1.0:
                worst_dir = 180.0
            else:
                c = (dx * disc["vx"] + dy * disc["vy"]) / (d * math.hypot(disc["vx"], disc["vy"]))
                worst_dir = max(worst_dir, math.degrees(math.acos(max(-1.0, min(1.0, c)))))
        record("central_push_translates", worst_rot < 0.01 and worst_dir < 10,
               f"a push through the centre of mass rotated the body by {worst_rot:.3g} rad (expected 0) "
               f"and moved it at {worst_dir:.3g} deg from the push direction (expected 0; 180 means it "
               "barely moved)")

    def t_offcentre_sign():
        bad = []
        for off in (-30.0, -20.0, 20.0, 30.0):
            for side in ((0, 1), (0, -1)):
                b = _body(BAR, (0.0, 0.0), ang=0.0)
                disc = _push(b, side, off, speed=150.0)
                o = _call(fn, disc, b, DT, p)
                da = math.remainder(o["ang"] - b["ang"], 2 * math.pi)
                expected = math.copysign(1.0, off * side[1])  # torque r x F, r=(off, -side*10), F=(0, side)
                if da * expected <= 1e-4:
                    bad.append((off, side[1], round(da, 4)))
        record("offcentre_rotation_sign", not bad,
               "a push at body-frame offset u (lever arm about the centre of mass) must rotate the body "
               f"with the sign of the torque r x F; wrong or zero for (offset, push v-direction, dang)={bad}")

    def t_rotation_about_com():
        worst = 0.0
        for cu in (-20.0, 20.0):
            b = _body(BAR, (cu, 0.0), ang=0.0)
            disc = _push(b, (0, 1), cu, speed=150.0)
            o = _call(fn, disc, b, DT, p)
            worst = max(worst, abs(math.remainder(o["ang"] - b["ang"], 2 * math.pi)))
        record("pivot_is_centre_of_mass", worst < 0.01,
               "with the centre of mass offset to u=cu, a push directly under it must not rotate the body "
               f"(rotated by {worst:.3g} rad); rotation and torque are about the centre of mass, not the "
               "body-frame origin")

    def t_com_along_normal():
        worst = 0.0
        for com in ((25.0, 6.0), (-20.0, -5.0)):
            for off in (-30.0, 10.0):
                b = _body(BAR, com, ang=0.4, x=5.0, y=-3.0)
                disc = _push(b, (0, 1), off, speed=80.0, gap=0.5)
                o = _call(fn, disc, b, 0.05, p)
                c0, c1 = _com_world(b), _com_world({**b, **o})
                dx, dy = c1[0] - c0[0], c1[1] - c0[1]
                d = math.hypot(dx, dy)
                if d < 1e-6:
                    worst = 90.0
                    continue
                c = (dx * disc["vx"] + dy * disc["vy"]) / (d * math.hypot(disc["vx"], disc["vy"]))
                worst = max(worst, math.degrees(math.acos(max(-1.0, min(1.0, c)))))
        record("com_moves_along_normal", worst < 3.0,
               "for a frictionless push on a flat face of a body whose centre of mass is offset from the "
               f"body-frame origin, the centre of mass moved at {worst:.3g} deg from the contact normal "
               "(expected 0): translate the centre of mass and rotate about it, then recover the origin as "
               "com_world - R(ang) com")

    def t_point_pusher():
        worst_pen, worst_dir = -1e9, 0.0
        for k in range(8):
            b = _body(BAR, (0.0, 0.0), ang=rng.uniform(-3, 3), x=rng.uniform(-30, 30), y=rng.uniform(-30, 30))
            nu, nv = [(0, 1), (0, -1), (1, 0), (-1, 0)][k % 4]
            # start 1 unit inside the face with outward normal -(nu, nv), moving inward
            su, sv = (rng.uniform(-25, 25), -nv * 9.0) if nv else (-nu * 39.0, rng.uniform(-5, 5))
            wx, wy = _R(b["ang"], su, sv)
            vx, vy = _R(b["ang"], nu * 60.0, nv * 60.0)
            disc = {"x": b["x"] + wx, "y": b["y"] + wy, "vx": vx, "vy": vy, "r": 0.0}
            o = _call(fn, disc, b, DT, p)
            nb = {**b, **o}
            fx, fy = disc["x"] + vx * DT, disc["y"] + vy * DT
            worst_pen = max(worst_pen, -_sdf(BAR, *_to_body(nb, fx, fy)))
            dx, dy = o["x"] - b["x"], o["y"] - b["y"]
            d = math.hypot(dx, dy)
            if d < 1.0:
                worst_dir = 180.0
            else:
                c = (dx * vx + dy * vy) / (d * math.hypot(vx, vy))
                worst_dir = max(worst_dir, math.degrees(math.acos(max(-1.0, min(1.0, c)))))
        record("point_pusher_inside_box", worst_pen < 1.5 and worst_dir < 15,
               "a point pusher (r=0) whose centre starts 1 unit inside a box face and moves 6 units further in "
               f"ended {worst_pen:.3g} inside the body (allowed 1.5) and moved it at {worst_dir:.3g} deg from the "
               "push (expected about 0; 180 means it barely moved). When the pusher centre is inside a box, "
               "the contact normal is that of the nearest face and the depth is r plus the distance to it")

    def t_effective_mass():
        errs = []
        for ratio in (300.0, 3000.0):
            for off in (15.0, 30.0):
                b = _body(BAR, (0.0, 0.0), ang=0.0, mass=1.0, inertia=ratio)
                disc = _push(b, (0, 1), off, speed=60.0, gap=0.5)
                o = _call(fn, disc, b, 0.05, p)
                c0 = _com_world(b)
                c1 = _com_world({**b, **o})
                dc = math.hypot(c1[0] - c0[0], c1[1] - c0[1])
                da = abs(math.remainder(o["ang"] - b["ang"], 2 * math.pi))
                if dc < 1e-6:
                    errs.append((ratio, off, "no translation"))
                    continue
                expected = off / ratio
                got = da / dc
                if not (0.6 * expected <= got <= 1.6 * expected):
                    errs.append((ratio, off, f"dang/dcom={got:.4g}, expected {expected:.4g}"))
        record("effective_mass_split", not errs,
               "for a short off-centre push with lever arm r_n, the change of angle per unit of centre-of-mass "
               "translation must be about r_n * mass / inertia (it follows from applying one impulse at the "
               f"contact point); failed for (inertia/mass, lever arm): {errs}")

    for name, f in [("no_contact_no_motion", t_no_contact), ("shape_aware_detection", t_shape_aware),
                    ("non_penetration", t_non_penetration), ("central_push_translates", t_central_push),
                    ("offcentre_rotation_sign", t_offcentre_sign), ("pivot_is_centre_of_mass", t_rotation_about_com),
                    ("com_moves_along_normal", t_com_along_normal), ("point_pusher_inside_box", t_point_pusher),
                    ("effective_mass_split", t_effective_mass)]:
        safe(name, f)
    return results


def solver_helpers(entries) -> Tuple[str, str]:
    """Source that defines each verified solver under its own name, and the prompt text for it."""
    from .mechanism_tests import FAMILIES, family_of

    srcs: List[str] = []
    docs: Dict[str, List[str]] = {}
    for e in entries:
        if e.source != "general_solver" or "def solve(" not in e.snippet:
            continue
        name = re.sub(r"\W", "_", e.name)
        fam = family_of(e)
        srcs.append("\n\n" + e.snippet.replace("def solve(", f"def {name}(", 1))
        keys = sorted(set(re.findall(r"p\.get\(\s*['\"](\w+)['\"]", e.snippet)) | set(re.findall(r"p\[['\"](\w+)['\"]\]", e.snippet)))
        params = ", ".join(keys) or "none"
        sig = FAMILIES[fam].interface.splitlines()[0].replace("def solve", name).rstrip(":")
        docs.setdefault(fam, []).append(f"  - {sig}: {e.mechanism.split(' Complete solver')[0]} Optional p keys: {params}.")
    if not srcs:
        return "", ""
    doc = (
        "\nVerified mechanism solvers. The functions below are ALREADY DEFINED in your program's namespace "
        "(they are appended to your code). Each is a complete solver for one kind of mechanism that passed "
        "generic physics checks on synthetic scenarios. If this environment contains a mechanism one of them "
        "covers, CALL it for that part instead of writing your own code for it; do not copy or re-implement "
        "them, and do not define functions with these names. If none applies, ignore them. Your step() maps "
        "this environment's state to their arguments, models everything else, and maps the result back; "
        "unknown geometry and constants should be fitted constants in PARAMS.\n"
    )
    for fam, lines in docs.items():
        doc += f"\n* {FAMILIES[fam].intro}. Interface:\n" + FAMILIES[fam].interface + "\n" + "\n".join(lines) + "\n"
        if fam == "contact":
            doc += ("  The body's centre-of-mass offset, mass and inertia are fitted constants. If the body keeps no "
                    "momentum between steps, pass zero velocities. Pass the pusher's position at the start of the step "
                    "and its velocity during the step; the solver moves it. With an identified contact shape whose "
                    "boxes are already grown by the pusher radius, pass those boxes with r=0.\n")
    return "".join(srcs) + "\n", doc


def report(results: List[Dict[str, Any]]) -> str:
    lines = []
    for r in results:
        lines.append(f"[{'PASS' if r['passed'] else 'FAIL'}] {r['name']}: {r['detail']}")
    return "\n".join(lines)
