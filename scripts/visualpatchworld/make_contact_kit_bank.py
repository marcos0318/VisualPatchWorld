#!/usr/bin/env python
"""Append a hand-written, environment-agnostic contact-mechanics kit to a general bank."""
from __future__ import annotations

import argparse
import json
import sys
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


SKILLS = [
    {
        "name": "rotation_about_centre_of_mass",
        "mechanism": "A rigid body reported by a reference point (x, y, angle) rotates about its centre of mass, which is offset from the reference point by an unknown body-frame vector (fit it); update the centre of mass and angle, then recover the reference point.",
        "equations": "g = (x, y) + R(angle) c; g' = g + v dt; angle' = angle + w dt; (x', y') = g' - R(angle') c",
        "snippet": "def move_about_com(x, y, ang, cu, cv, vx, vy, w, dt):\n    c, s = math.cos(ang), math.sin(ang)\n    gx, gy = x + c*cu - s*cv + vx*dt, y + s*cu + c*cv + vy*dt\n    ang2 = ang + w*dt\n    c2, s2 = math.cos(ang2), math.sin(ang2)\n    return gx - (c2*cu - s2*cv), gy - (s2*cu + c2*cv), ang2",
        "params": [{"name": "cu", "meaning": "centre-of-mass body-frame u offset"}, {"name": "cv", "meaning": "centre-of-mass body-frame v offset"}],
        "tags": ["rigid_body", "rotation", "centre_of_mass", "pivot", "pose"],
    },
    {
        "name": "substep_integration",
        "mechanism": "One recorded transition spans a control period much longer than the dynamics' time scale (stiff PD tracking, contact). Integrate the whole step function with n small substeps (n ~ 5-10) holding the action fixed; explicit single-step updates of stiff contact are unstable.",
        "equations": "for i in 1..n: (state) <- substep(state, action, dt = T/n)",
        "snippet": "for _ in range(n):\n    vel += (kp*(target - pos) - kv*vel)*dt\n    # ... contact detection and impulses at the current positions ...\n    pos += vel*dt",
        "params": [{"name": "n", "meaning": "substeps"}, {"name": "dt", "meaning": "substep length"}],
        "tags": ["integration", "substeps", "stiff", "contact", "stability"],
    },
    {
        "name": "point_pusher_vs_identified_shape",
        "mechanism": "If meta carries an identified contact shape (union of body-frame boxes grown by the pusher radius), the pusher touches the body when its centre enters a box; push it out along the nearest face normal with penetration = distance to that face.",
        "equations": "inside if u0<u<u1 and v0<v<v1; (g, n) = min over faces of (distance, outward normal)",
        "snippet": "for u0, u1, v0, v1 in meta['contact_shape']['boxes_u_min_u_max_v_min_v_max']:\n    if u0 < u < u1 and v0 < v < v1:\n        g, nu, nv = min((u-u0,-1.0,0.0),(u1-u,1.0,0.0),(v-v0,0.0,-1.0),(v1-v,0.0,1.0))",
        "params": [],
        "tags": ["contact", "geometry", "identified_shape", "pushing"],
    },
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--skills", action="store_true", help="also add the rigid-body skills")
    p.add_argument("--library", action="store_true", help="also add the complete pusher/body library program")
    args = p.parse_args()
    base = json.loads(Path(args.base).read_text())
    kit = [{**e, "topic": "contact_mechanics_kit", "source": "general"} for e in KIT]
    if args.skills:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
        from visualpatchworld.llm_family.sysid import PUSHER_BODY_SOURCE

        lib = {
            "name": "pd_pusher_impulse_body_program",
            "mechanism": "Complete program: PD-tracking disc pusher, frictionless sequential normal impulses against a quasi-static rigid body given as a union of body-frame boxes in meta['contact_shape'], rotation about a fitted centre of mass, n substeps.",
            "equations": "see snippet",
            "snippet": PUSHER_BODY_SOURCE,
            "params": [],
            "tags": ["contact", "pushing", "rigid_body", "program", "library"],
        }
        kit += [{**e, "topic": "rigid_body_skills", "source": "general"} for e in SKILLS + ([lib] if args.library else [])]
    Path(args.out).write_text(json.dumps(base + kit, indent=2), encoding="utf-8")
    print(f"wrote {args.out}: {len(base)} + {len(kit)} entries")


if __name__ == "__main__":
    main()
