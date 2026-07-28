"""Generalizable collision-geometry primitive for object-centric world models.

This is the *shared substrate* described in
``docs/scientific_discovery_dynamics.md`` §8: rather than hand-coding contact
behaviour per environment, we provide (a) object collision boundaries, (b) an
env-agnostic collision-detection primitive (closest point / contact normal /
penetration between a circle and a polygon), and (c) helpers that turn a detected
contact into a small, *geometry-aware* feature set on which a tiny contact-response
law can be induced. Only the per-object geometry is environment-specific (and is
perceivable); the detection math here is general.

PushT geometry is provided as a concrete instance (read directly from the
simulator shapes): agent = circle r=15; T-block = two polygons in the block's
local frame (origin = recorded block (x, y); centre of gravity at local (0, 45)).
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

Vec = Tuple[float, float]

# --- PushT collision geometry, ORACLE (read from sim source) ---
# NOTE: these constants come from the simulator's env.py and are therefore
# "cheating" (ground-truth code access). They are kept ONLY as a validation
# reference. The legitimate pipeline perceives geometry from pixels via
# ``visual_geometry.detect_pusht_geometry`` and loads it with
# ``load_perceived_geometry``; see docs/scientific_discovery_dynamics.md §8.
PUSHT_AGENT_RADIUS = 15.0
PUSHT_TEE_POLYGONS: Tuple[Tuple[Vec, ...], ...] = (
    ((-60.0, 0.0), (60.0, 0.0), (60.0, 30.0), (-60.0, 30.0)),       # bar
    ((-15.0, 30.0), (15.0, 30.0), (15.0, 120.0), (-15.0, 120.0)),   # stem
)
PUSHT_TEE_COG: Vec = (0.0, 45.0)


def load_perceived_geometry(path) -> dict:
    """Load perceived (pixel-detected, body-frame) geometry produced by the
    geometry-detection CLI: {"polygons": [[[x,y],...], ...], "agent_radius": r,
    "cog": [x, y]}."""
    import json
    from pathlib import Path

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        "polygons": [[(float(x), float(y)) for x, y in poly] for poly in data["polygons"]],
        "agent_radius": float(data["agent_radius"]),
        "cog": (float(data["cog"][0]), float(data["cog"][1])),
    }


def oracle_geometry() -> dict:
    """The source-derived geometry (validation only)."""
    return {
        "polygons": [list(poly) for poly in PUSHT_TEE_POLYGONS],
        "agent_radius": PUSHT_AGENT_RADIUS,
        "cog": PUSHT_TEE_COG,
    }


def rotate(v: Vec, angle: float) -> Vec:
    c, s = math.cos(angle), math.sin(angle)
    return (c * v[0] - s * v[1], s * v[0] + c * v[1])


def to_world(local: Vec, origin: Vec, angle: float) -> Vec:
    r = rotate(local, angle)
    return (origin[0] + r[0], origin[1] + r[1])


def polygon_to_world(poly: Sequence[Vec], origin: Vec, angle: float) -> List[Vec]:
    return [to_world(v, origin, angle) for v in poly]


def _closest_point_on_segment(p: Vec, a: Vec, b: Vec) -> Tuple[Vec, float]:
    abx, aby = b[0] - a[0], b[1] - a[1]
    apx, apy = p[0] - a[0], p[1] - a[1]
    denom = abx * abx + aby * aby
    t = 0.0 if denom <= 1e-12 else (apx * abx + apy * aby) / denom
    t = max(0.0, min(1.0, t))
    q = (a[0] + t * abx, a[1] + t * aby)
    return q, math.hypot(p[0] - q[0], p[1] - q[1])


def _point_in_polygon(p: Vec, poly: Sequence[Vec]) -> bool:
    """Ray-casting point-in-polygon (handles convex *and* non-convex shapes).

    Needed because perceived boundaries are a single non-convex contour (a T),
    not a union of convex pieces.
    """
    x, y = p
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def closest_point_on_polygon(p: Vec, poly: Sequence[Vec]) -> Tuple[Vec, float, bool]:
    """Return (closest boundary point, distance, inside?) for a convex polygon."""
    best_q, best_d = poly[0], float("inf")
    n = len(poly)
    for i in range(n):
        q, d = _closest_point_on_segment(p, poly[i], poly[(i + 1) % n])
        if d < best_d:
            best_d, best_q = d, q
    return best_q, best_d, _point_in_polygon(p, poly)


def circle_polygons_contact(
    center: Vec,
    radius: float,
    polygons_world: Sequence[Sequence[Vec]],
) -> Optional[dict]:
    """Collision query between a circle and a union of convex polygons.

    Returns the contact with the smallest surface distance:
      contact_point  -- closest boundary point (world)
      normal         -- unit outward normal (surface -> circle centre)
      distance       -- centre-to-boundary distance (>=0)
      penetration    -- radius - distance (positive if overlapping)
      inside         -- whether the circle centre is inside a polygon
    Returns None if there are no polygons.
    """
    best = None
    for poly in polygons_world:
        q, d, inside = closest_point_on_polygon(center, poly)
        score = -1e9 if inside else d
        if best is None or score < best[0]:
            best = (score, q, d, inside)
    if best is None:
        return None
    _, q, d, inside = best
    if d > 1e-9:
        nx, ny = (center[0] - q[0]) / d, (center[1] - q[1]) / d
    else:
        nx, ny = 0.0, 0.0
    if inside:
        nx, ny = -nx, -ny  # centre inside: outward normal points back out
        penetration = radius + d
    else:
        penetration = radius - d
    return {
        "contact_point": q,
        "normal": (nx, ny),
        "distance": d,
        "penetration": penetration,
        "inside": inside,
    }


def pusht_contact(
    agent_xy: Vec,
    block_xy: Vec,
    block_angle: float,
    *,
    agent_radius: float = PUSHT_AGENT_RADIUS,
    polygons: Optional[Sequence[Sequence[Vec]]] = None,
    cog: Vec = PUSHT_TEE_COG,
) -> dict:
    """Circle-vs-polygons collision query in world coords, plus CoG lever arm.

    ``polygons`` / ``cog`` default to the oracle T geometry but should be supplied
    from perceived geometry for the legitimate (no-source) pipeline.
    """
    polys = polygons if polygons is not None else PUSHT_TEE_POLYGONS
    polys_world = [polygon_to_world(p, block_xy, block_angle) for p in polys]
    contact = circle_polygons_contact(agent_xy, agent_radius, polys_world)
    cog_world = to_world(cog, block_xy, block_angle)
    if contact is None:
        contact = {"contact_point": block_xy, "normal": (0.0, 0.0), "distance": 1e9, "penetration": -1e9, "inside": False}
    cp = contact["contact_point"]
    contact["cog_world"] = cog_world
    contact["lever"] = (cp[0] - cog_world[0], cp[1] - cog_world[1])
    return contact


def pusht_contact_features(
    agent_xy: Vec,
    block_xy: Vec,
    block_angle: float,
    agent_disp: Vec,
    *,
    agent_radius: float = PUSHT_AGENT_RADIUS,
    polygons: Optional[Sequence[Sequence[Vec]]] = None,
    cog: Vec = PUSHT_TEE_COG,
) -> List[float]:
    """Geometry-aware contact-response basis for (dbx, dby, dtheta).

    Uses the *actual* contact point on the boundary, the local surface normal,
    and the lever arm about the centre of gravity -- the inputs the proxy
    (agent->centre) models lack.
    """
    c = pusht_contact(agent_xy, block_xy, block_angle, agent_radius=agent_radius, polygons=polygons, cog=cog)
    nx, ny = c["normal"]
    dax, day = agent_disp
    # push = component of agent motion into the block surface (-normal direction)
    push = -(dax * nx + day * ny)
    if push < 0.0:
        push = 0.0
    lx, ly = c["lever"]
    # torque about CoG from the applied displacement at the contact point
    torque = lx * day - ly * dax
    pen = c["penetration"]
    return [dax, day, push * (-nx), push * (-ny), torque, push, pen, 1.0]


N_PUSHT_CONTACT_FEATS = 8
