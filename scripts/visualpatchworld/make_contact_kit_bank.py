#!/usr/bin/env python
"""Append a hand-written, environment-agnostic contact-mechanics kit to a general bank."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

KIT = [
    {
        "name": "body_frame_transform",
        "mechanism": "Express a world point in a rigid body's frame (and back) from the body pose (x, y, angle).",
        "equations": "u = c*(px-x) + s*(py-y); v = -s*(px-x) + c*(py-y); world = (x + c*u - s*v, y + s*u + c*v), c=cos(angle), s=sin(angle)",
        "snippet": "def to_body(px, py, x, y, ang):\n    c, s = math.cos(ang), math.sin(ang)\n    dx, dy = px - x, py - y\n    return c*dx + s*dy, -s*dx + c*dy\n\ndef to_world(u, v, x, y, ang):\n    c, s = math.cos(ang), math.sin(ang)\n    return x + c*u - s*v, y + s*u + c*v",
        "params": [],
        "tags": ["geometry", "rigid_body", "pose", "frame", "rotation"],
    },
    {
        "name": "disc_rectangle_contact",
        "mechanism": "Contact between a disc and an axis-aligned rectangle in the rectangle's body frame: closest point, outward normal, penetration depth.",
        "equations": "q = clamp(center, rect); d = |center - q|; contact if d < radius; n = (center - q)/d; depth = radius - d",
        "snippet": "def disc_rect(u, v, radius, u0, u1, v0, v1):\n    qu, qv = min(max(u, u0), u1), min(max(v, v0), v1)\n    eu, ev = u - qu, v - qv\n    d = math.hypot(eu, ev)\n    if d >= radius or d < 1e-9:\n        return None\n    return qu, qv, eu / d, ev / d, radius - d",
        "params": [{"name": "radius", "meaning": "disc radius"}],
        "tags": ["contact", "collision", "geometry", "polygon", "shape", "disc"],
    },
    {
        "name": "rigid_body_normal_impulse",
        "mechanism": "Frictionless inelastic contact impulse on a planar rigid body hit by a kinematic pusher; the impulse acts at the contact point and also produces rotation about the centre of mass.",
        "equations": "r = contact - com; rn = r_x n_y - r_y n_x; k = 1/m + rn^2/I; vrel = (v_pusher - (v_body + w x r)).n; j = max(0, (bias - vrel)/k); v_body -= j n/m; w -= j rn/I",
        "snippet": "def normal_impulse(vb, w, v_push, r, n, m, I, bias=0.0):\n    rn = r[0]*n[1] - r[1]*n[0]\n    k = 1.0/m + rn*rn/I\n    vp = (vb[0] - w*r[1], vb[1] + w*r[0])\n    vrel = (v_push[0]-vp[0])*n[0] + (v_push[1]-vp[1])*n[1]\n    j = max(0.0, (bias - vrel)/k)\n    return (vb[0] - j*n[0]/m, vb[1] - j*n[1]/m), w - j*rn/I",
        "params": [{"name": "m", "meaning": "body mass"}, {"name": "I", "meaning": "moment of inertia about the centre of mass"}],
        "tags": ["contact", "impulse", "rigid_body", "rotation", "torque", "pushing"],
    },
    {
        "name": "pd_tracking_substeps",
        "mechanism": "An actuator tracks a target set by the action with a PD law, integrated with several small substeps per control step.",
        "equations": "target = pos + scale*action; for each substep: vel += (kp*(target-pos) - kv*vel)*dt; pos += vel*dt",
        "snippet": "def pd_track(pos, vel, target, kp, kv, dt, n):\n    for _ in range(n):\n        vel += (kp*(target - pos) - kv*vel)*dt\n        pos += vel*dt\n    return pos, vel",
        "params": [{"name": "kp", "meaning": "stiffness"}, {"name": "kv", "meaning": "damping"}, {"name": "dt", "meaning": "substep"}, {"name": "n", "meaning": "substeps per control step"}],
        "tags": ["actuation", "control", "pd", "integration", "substeps"],
    },
    {
        "name": "quasi_static_pushing",
        "mechanism": "A pushed object with high damping has no momentum: it moves only during contact, by the contact impulse of that substep, and stops when contact ends.",
        "equations": "each substep: v_body = 0; apply contact impulses; pos += v_body*dt; angle += w*dt",
        "snippet": "def quasi_static_update(com, ang, impulses_v, impulses_w, dt):\n    return (com[0] + impulses_v[0]*dt, com[1] + impulses_v[1]*dt), ang + impulses_w*dt",
        "params": [],
        "tags": ["contact", "pushing", "quasi_static", "damping", "manipulation"],
    },
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    base = json.loads(Path(args.base).read_text())
    kit = [{**e, "topic": "contact_mechanics_kit", "source": "general"} for e in KIT]
    Path(args.out).write_text(json.dumps(base + kit, indent=2), encoding="utf-8")
    print(f"wrote {args.out}: {len(base)} + {len(kit)} entries")


if __name__ == "__main__":
    main()
