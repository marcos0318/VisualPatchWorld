"""Mechanism families for LLM-written, test-verified solvers.

Each family has a generic interface, the physical properties its solvers are checked for, the
methods the LLM is asked to implement, and environment-agnostic tests on synthetic scenarios
(sizes and constants differ from every evaluation environment). ``contact`` is the planar
pusher / rigid-body family of ``solver_tests``.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

from . import solver_tests as _contact
from .solver_tests import _call, _Timeout  # noqa: F401


@dataclass
class Family:
    name: str
    intro: str
    interface: str
    properties: str
    methods: List[str]
    run_tests: Callable[..., List[Dict[str, Any]]]
    tags: List[str] = field(default_factory=list)


def _runner(tests):
    def run(fn: Callable, p: Dict[str, float] | None = None, *, seed: int = 0) -> List[Dict[str, Any]]:
        p = dict(p or {})
        rng = random.Random(seed)
        results: List[Dict[str, Any]] = []

        def record(name, passed, detail):
            results.append({"name": name, "passed": bool(passed), "detail": detail})

        for name, t in tests:
            try:
                t(fn, p, rng, lambda ok, detail, _n=name: record(_n, ok, detail))
            except _Timeout:
                record(name, False, "timed out (> 2 s for one call)")
            except Exception as e:  # noqa: BLE001
                record(name, False, f"raised {type(e).__name__}: {e}")
        return results
    return run


def _timed(fn, *args, timeout: float = 2.0):
    import signal

    def _alarm(*_):
        raise _Timeout()

    old = signal.signal(signal.SIGALRM, _alarm)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        return fn(*args)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


# ---------------------------------------------------------------- obstacle

OBSTACLE_INTERFACE = """def solve(agent, delta, walls, p):
    # agent: dict x, y, r. A disc (r may be 0) that moves on its own, e.g. a point robot.
    # delta: (dx, dy), the displacement the agent would make this step in free space.
    # walls: list of static axis-aligned boxes (x0, x1, y0, y1) the agent cannot overlap.
    # p:     dict of numeric parameters with defaults you choose.
    # Return a dict with the agent's new x, y."""

OBSTACLE_PROPERTIES = """  - free motion: with no wall in the way the agent moves by exactly delta;
  - non-penetration: the agent's centre ends at distance >= r from every wall;
  - no tunnelling: a step longer than a wall is thick still stops at the wall;
  - contact: a blocked agent ends touching the wall (within 1 unit), not short of it;
  - frictionless walls: the component of delta along the wall is kept (the agent slides);
  - openings: the agent passes through a gap between walls that is wider than its diameter."""


def _box_dist(x, y, b):
    x0, x1, y0, y1 = b
    dx = max(x0 - x, 0.0, x - x1)
    dy = max(y0 - y, 0.0, y - y1)
    if dx == 0.0 and dy == 0.0:
        return -min(x - x0, x1 - x, y - y0, y1 - y)
    return math.hypot(dx, dy)


def _obs_call(fn, agent, delta, walls, p):
    out = _timed(fn, dict(agent), tuple(delta), [tuple(w) for w in walls], dict(p))
    return float(out["x"]), float(out["y"])


def _t_obs_free(fn, p, rng, rec):
    worst = 0.0
    walls = [(300.0, 310.0, -400.0, 400.0)]
    for _ in range(10):
        a = {"x": rng.uniform(-50, 50), "y": rng.uniform(-50, 50), "r": rng.choice([0.0, 3.0])}
        d = (rng.uniform(-20, 20), rng.uniform(-20, 20))
        x, y = _obs_call(fn, a, d, walls, p)
        worst = max(worst, math.hypot(x - a["x"] - d[0], y - a["y"] - d[1]))
    rec(worst < 1e-3, f"with every wall far away the agent ended {worst:.3g} from start + delta (expected 0)")


def _t_obs_nonpen(fn, p, rng, rec):
    worst = -1e9
    for k in range(8):
        r = [0.0, 2.5][k % 2]
        w = (40.0, 44.0, -100.0, 100.0) if k < 4 else (-100.0, 100.0, 40.0, 44.0)
        a = {"x": 20.0, "y": rng.uniform(-30, 30), "r": r} if k < 4 else {"x": rng.uniform(-30, 30), "y": 20.0, "r": r}
        d = (rng.uniform(25, 60), rng.uniform(-5, 5)) if k < 4 else (rng.uniform(-5, 5), rng.uniform(25, 60))
        x, y = _obs_call(fn, a, d, [w], p)
        worst = max(worst, r - _box_dist(x, y, w))
    rec(worst < 0.25, f"after moving into a wall the agent overlaps it by {worst:.3g} (allowed 0.25); its centre must "
        "stay at least r from every wall")


def _t_obs_tunnel(fn, p, rng, rec):
    bad = []
    for r in (0.0, 1.5):
        a = {"x": 0.0, "y": 0.0, "r": r}
        x, _ = _obs_call(fn, a, (60.0, 0.0), [(20.0, 22.0, -50.0, 50.0)], p)
        if x > 20.0:
            bad.append((r, round(x, 2)))
    rec(not bad, "a step of 60 toward a wall 2 thick at x=20 must stop before it; the agent ended past it for "
        f"(r, x)={bad}. Check the swept path, not only the end position")


def _t_obs_contact(fn, p, rng, rec):
    worst = 0.0
    for r in (0.0, 2.0):
        a = {"x": 0.0, "y": 5.0, "r": r}
        x, _ = _obs_call(fn, a, (35.0, 0.0), [(20.0, 26.0, -50.0, 50.0)], p)
        worst = max(worst, abs((20.0 - r) - x))
    rec(worst < 1.0, f"a blocked agent stopped {worst:.3g} away from the contact position x = wall - r (allowed 1)")


def _t_obs_slide(fn, p, rng, rec):
    worst = 0.0
    for r, dy in ((0.0, 12.0), (2.0, -9.0)):
        a = {"x": 0.0, "y": 0.0, "r": r}
        x, y = _obs_call(fn, a, (30.0, dy), [(20.0, 26.0, -80.0, 80.0)], p)
        worst = max(worst, abs(y - dy))
    rec(worst < 0.5, f"pushing diagonally into a frictionless wall lost {worst:.3g} of the motion along the wall "
        "(expected 0: keep the tangential component)")


def _t_obs_gap(fn, p, rng, rec):
    walls = [(20.0, 26.0, -80.0, -5.0), (20.0, 26.0, 5.0, 80.0)]
    bad = []
    for r in (0.0, 3.0):
        a = {"x": 0.0, "y": 0.0, "r": r}
        x, y = _obs_call(fn, a, (40.0, 0.0), walls, p)
        if abs(x - 40.0) > 0.5 or abs(y) > 0.5:
            bad.append((r, round(x, 2), round(y, 2)))
    rec(not bad, "moving straight through a gap of width 10 in a wall must not be blocked; ended at (r, x, y)="
        f"{bad}, expected (40, 0)")


OBSTACLE = Family(
    name="obstacle",
    intro="a self-propelled disc (e.g. a point robot) moving among static walls",
    interface=OBSTACLE_INTERFACE,
    properties=OBSTACLE_PROPERTIES,
    methods=["projection: move, then push the agent out of every overlapping wall along the axis of least "
             "penetration, repeated a few times",
             "swept: substep the motion and stop the normal component at first contact, keeping the "
             "tangential component"],
    run_tests=_runner([("free_motion", _t_obs_free), ("non_penetration", _t_obs_nonpen),
                       ("no_tunnelling", _t_obs_tunnel), ("stops_at_contact", _t_obs_contact),
                       ("slides_along_wall", _t_obs_slide), ("passes_through_gap", _t_obs_gap)]),
    tags=["walls", "obstacle", "collision", "navigation", "solver", "complete"],
)

# ---------------------------------------------------------------- chain

CHAIN_INTERFACE = """def solve(q, dq, links, p):
    # q:     list of joint angles of a planar serial chain (radians, NOT wrapped).
    # dq:    list of commanded joint increments for this step.
    # links: list of dicts, one per joint: length (of the link after the joint), lo, hi
    #        (joint limits in radians; None means unlimited). Joint i's angle is relative to
    #        link i-1; the base is at (0, 0) and joint 0 is measured from the +x axis.
    # p:     dict of numeric parameters with defaults you choose.
    # Return a dict: q (new joint angles) and points (list of (x, y), the end of every link)."""

CHAIN_PROPERTIES = """  - forward kinematics: points are the exact link ends for the new angles;
  - joint limits: a joint never leaves [lo, hi]; a command beyond a limit stops AT the limit;
  - unlimited joints are not wrapped or clamped (angles may exceed pi);
  - joints are independent: a limit on one joint does not change the others' motion."""


def _fk(q, links):
    x = y = a = 0.0
    pts = []
    for qi, lk in zip(q, links):
        a += qi
        x += lk["length"] * math.cos(a)
        y += lk["length"] * math.sin(a)
        pts.append((x, y))
    return pts


def _chain_call(fn, q, dq, links, p):
    out = _timed(fn, list(q), list(dq), [dict(lk) for lk in links], dict(p))
    return [float(v) for v in out["q"]], [(float(a), float(b)) for a, b in out["points"]]


def _rand_chain(rng, n, limits=False):
    return [{"length": rng.uniform(0.3, 2.0), "lo": (-1.9 if limits else None), "hi": (1.9 if limits else None)}
            for _ in range(n)]


def _t_chain_fk(fn, p, rng, rec):
    worst = 0.0
    for n in (1, 2, 3, 3):
        links = _rand_chain(rng, n)
        q = [rng.uniform(-1, 1) for _ in range(n)]
        dq = [rng.uniform(-0.2, 0.2) for _ in range(n)]
        q2, pts = _chain_call(fn, q, dq, links, p)
        ref = _fk([a + b for a, b in zip(q, dq)], links)
        worst = max([worst] + [abs(a - b) for a, b in zip(q2, [a + b for a, b in zip(q, dq)])]
                    + [math.hypot(a[0] - b[0], a[1] - b[1]) for a, b in zip(pts, ref)])
        if len(pts) != n:
            worst = max(worst, 1e3)
    rec(worst < 1e-6, f"without limits, new angles must be q + dq and points the exact link ends; max error {worst:.3g}")


def _t_chain_limits(fn, p, rng, rec):
    bad = []
    links = [{"length": 1.0, "lo": None, "hi": None}, {"length": 0.7, "lo": -1.2, "hi": 1.5}]
    for q1, d1, expect in ((1.4, 0.4, 1.5), (-1.0, -0.5, -1.2), (1.5, 0.3, 1.5), (0.0, 0.3, 0.3)):
        q2, pts = _chain_call(fn, [0.2, q1], [0.1, d1], links, p)
        if abs(q2[1] - expect) > 1e-3:
            bad.append((q1, d1, round(q2[1], 4), expect))
        ref = _fk(q2, links)
        if math.hypot(pts[-1][0] - ref[-1][0], pts[-1][1] - ref[-1][1]) > 1e-4:
            bad.append(("points do not match q", q1, d1))
    rec(not bad, "a limited joint must end at clip(q + dq, lo, hi) (stop AT the limit, no bounce); failed for "
        f"(q, dq, got, expected)={bad}")


def _t_chain_nowrap(fn, p, rng, rec):
    links = [{"length": 1.0, "lo": None, "hi": None}, {"length": 1.0, "lo": None, "hi": None}]
    q2, _ = _chain_call(fn, [3.0, -3.0], [0.4, -0.4], links, p)
    ok = abs(q2[0] - 3.4) < 1e-6 and abs(q2[1] + 3.4) < 1e-6
    rec(ok, f"unlimited joints at (3.0, -3.0) moved by (0.4, -0.4) ended at {[round(v, 4) for v in q2]}, expected "
        "(3.4, -3.4): do not wrap angles to [-pi, pi]")


def _t_chain_indep(fn, p, rng, rec):
    links = [{"length": 1.0, "lo": None, "hi": None}, {"length": 1.0, "lo": -0.5, "hi": 0.5}]
    q2, _ = _chain_call(fn, [0.0, 0.5], [0.25, 0.3], links, p)
    rec(abs(q2[0] - 0.25) < 1e-6, f"joint 0 (unlimited) must move by its own command 0.25 while joint 1 is at its "
        f"limit; it moved by {q2[0]:.4g}")


CHAIN = Family(
    name="chain",
    intro="a planar serial chain of rigid links (e.g. a robot arm) with optional joint limits",
    interface=CHAIN_INTERFACE,
    properties=CHAIN_PROPERTIES,
    methods=["kinematic: integrate the commanded increments, clip limited joints, then forward kinematics"],
    run_tests=_runner([("forward_kinematics", _t_chain_fk), ("joint_limits", _t_chain_limits),
                       ("no_angle_wrapping", _t_chain_nowrap), ("joints_independent", _t_chain_indep)]),
    tags=["arm", "joints", "kinematics", "limits", "solver", "complete"],
)

# ---------------------------------------------------------------- grasp

GRASP_INTERFACE = """def solve(eff, eff_next, grip, obj, world, p):
    # eff, eff_next: (x, y, z) of a gripper before and after this step (it moves on its own).
    # grip:  gripper command for this step; grip > world["grip_threshold"] means closed.
    # obj:   dict x, y, z: centre of a cube of half-size world["half"] resting on or held above
    #        a floor at height world["floor"].
    # world: dict half, floor, grip_threshold, grasp_radius (the gripper holds the cube when it
    #        is closed and within grasp_radius of the cube centre), eff_radius (the gripper's
    #        collision radius).
    # p:     dict of numeric parameters with defaults you choose.
    # Return a dict with the cube's new x, y, z."""

GRASP_PROPERTIES = """  - an untouched cube on the floor does not move;
  - a held cube (gripper closed and within grasp_radius) moves rigidly with the gripper: its offset to the
    gripper is kept;
  - an open gripper does not carry the cube; a cube that is not held drops to the floor (z = floor + half)
    within the step, keeping x, y;
  - the cube never goes below the floor, even when the gripper does;
  - an open gripper moving sideways into the cube at its height pushes it out of the way (no overlap in x, y)."""


def _grasp_call(fn, eff, eff_next, grip, obj, world, p):
    out = _timed(fn, tuple(eff), tuple(eff_next), float(grip), dict(obj), dict(world), dict(p))
    return float(out["x"]), float(out["y"]), float(out["z"])


def _world(rng):
    return {"half": 0.04, "floor": 0.1, "grip_threshold": 0.2, "grasp_radius": 0.03, "eff_radius": 0.015}


def _t_grasp_rest(fn, p, rng, rec):
    w = _world(rng)
    worst = 0.0
    for _ in range(6):
        obj = {"x": rng.uniform(-0.3, 0.3), "y": rng.uniform(-0.3, 0.3), "z": w["floor"] + w["half"]}
        e = (obj["x"] + 0.25, obj["y"] - 0.2, 0.4)
        e2 = (e[0] + rng.uniform(-0.05, 0.05), e[1] + rng.uniform(-0.05, 0.05), e[2] + rng.uniform(-0.05, 0.05))
        x, y, z = _grasp_call(fn, e, e2, rng.choice([-1.0, 1.0]), obj, w, p)
        worst = max(worst, abs(x - obj["x"]), abs(y - obj["y"]), abs(z - obj["z"]))
    rec(worst < 1e-4, f"a cube on the floor far from the gripper moved by {worst:.3g} (expected 0)")


def _t_grasp_carry(fn, p, rng, rec):
    w = _world(rng)
    worst = 0.0
    for _ in range(6):
        obj = {"x": rng.uniform(-0.2, 0.2), "y": rng.uniform(-0.2, 0.2), "z": w["floor"] + w["half"] + rng.uniform(0, 0.1)}
        off = (rng.uniform(-0.01, 0.01), rng.uniform(-0.01, 0.01), rng.uniform(-0.01, 0.01))
        e = (obj["x"] + off[0], obj["y"] + off[1], obj["z"] + off[2])
        d = (rng.uniform(-0.04, 0.04), rng.uniform(-0.04, 0.04), rng.uniform(0.0, 0.06))
        e2 = (e[0] + d[0], e[1] + d[1], e[2] + d[2])
        x, y, z = _grasp_call(fn, e, e2, 0.9, obj, w, p)
        worst = max(worst, abs(x - obj["x"] - d[0]), abs(y - obj["y"] - d[1]), abs(z - obj["z"] - d[2]))
    rec(worst < 1e-3, f"a cube held by a closed gripper must move by the gripper's displacement; error {worst:.3g}")


def _t_grasp_release(fn, p, rng, rec):
    w = _world(rng)
    worst = 0.0
    for grip in (-1.0, 0.1):
        obj = {"x": 0.05, "y": -0.02, "z": w["floor"] + w["half"] + 0.12}
        e = (obj["x"], obj["y"], obj["z"])
        x, y, z = _grasp_call(fn, e, (e[0], e[1], e[2] + 0.03), grip, obj, w, p)
        worst = max(worst, abs(x - obj["x"]), abs(y - obj["y"]), abs(z - (w["floor"] + w["half"])))
    rec(worst < 2e-3, f"with the gripper open (grip below threshold), a lifted cube must drop to z = floor + half "
        f"keeping x, y and not follow the gripper up; error {worst:.3g}")


def _t_grasp_floor(fn, p, rng, rec):
    w = _world(rng)
    obj = {"x": 0.0, "y": 0.0, "z": w["floor"] + w["half"] + 0.01}
    e = (0.0, 0.0, obj["z"])
    _, _, z = _grasp_call(fn, e, (0.0, 0.0, obj["z"] - 0.08), 0.9, obj, w, p)
    rec(z >= w["floor"] + w["half"] - 1e-3, f"a held cube pushed down by the gripper ended at z={z:.4g}; it must not go "
        f"below the floor contact height floor + half = {w['floor'] + w['half']:.4g}")


def _t_grasp_push(fn, p, rng, rec):
    w = _world(rng)
    bad = []
    for sgn in (1.0, -1.0):
        obj = {"x": 0.0, "y": 0.0, "z": w["floor"] + w["half"]}
        reach = w["half"] + w["eff_radius"]
        e = (-sgn * (reach + 0.01), 0.0, obj["z"])
        e2 = (-sgn * (reach - 0.03), 0.0, obj["z"])
        x, y, z = _grasp_call(fn, e, e2, -1.0, obj, w, p)
        gap = abs(x - e2[0]) - reach
        if gap < -2e-3 or abs(y) > 2e-3 or abs(z - obj["z"]) > 2e-3:
            bad.append((sgn, round(x, 4), round(gap, 4)))
    rec(not bad, "an open gripper moving sideways into a cube on the floor must push it so that they do not "
        f"overlap in x, y (centre distance >= half + eff_radius) and not lift it; failed for (direction, x, gap)={bad}")


GRASP = Family(
    name="grasp",
    intro="a 3D gripper that can push, grasp, carry and release a cube above a floor",
    interface=GRASP_INTERFACE,
    properties=GRASP_PROPERTIES,
    methods=["kinematic attachment: decide held / not held, carry rigidly when held, otherwise resolve pushing "
             "in x, y and drop to the floor"],
    run_tests=_runner([("rest_untouched", _t_grasp_rest), ("carry_rigidly", _t_grasp_carry),
                       ("release_drops", _t_grasp_release), ("floor_is_solid", _t_grasp_floor),
                       ("open_gripper_pushes", _t_grasp_push)]),
    tags=["gripper", "grasp", "pick", "place", "carry", "solver", "complete"],
)

CONTACT = Family(
    name="contact",
    intro="planar contact between a kinematic disc (a pusher, e.g. a robot finger) and a free rigid body "
          "whose shape is a union of boxes",
    interface=_contact.INTERFACE_DOC,
    properties="",
    methods=[],
    run_tests=_contact.run_tests,
    tags=["contact", "pushing", "rigid_body", "rotation", "impulse", "solver", "complete"],
)

FAMILIES: Dict[str, Family] = {f.name: f for f in (CONTACT, OBSTACLE, CHAIN, GRASP)}


def family_of(entry) -> str:
    for t in getattr(entry, "tags", []) or []:
        if t.startswith("family:"):
            return t.split(":", 1)[1]
    return "contact"
