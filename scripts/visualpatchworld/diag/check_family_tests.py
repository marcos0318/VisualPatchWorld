#!/usr/bin/env python
"""Sanity check of the mechanism-family tests: hand reference solvers must pass, and typical
wrong solvers must fail the intended tests."""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from visualpatchworld.llm_family.mechanism_tests import FAMILIES  # noqa: E402
from visualpatchworld.llm_family.solver_tests import report  # noqa: E402


def obstacle_ref(agent, delta, walls, p):
    x, y, r = agent["x"], agent["y"], agent["r"]
    n = 50
    for _ in range(n):
        for ax in (0, 1):
            nx, ny = (x + delta[0] / n, y) if ax == 0 else (x, y + delta[1] / n)
            hit = False
            for x0, x1, y0, y1 in walls:
                dx = max(x0 - nx, 0.0, nx - x1)
                dy = max(y0 - ny, 0.0, ny - y1)
                inside = x0 < nx < x1 and y0 < ny < y1
                if math.hypot(dx, dy) < r - 1e-9 or inside:
                    hit = True
                    if ax == 0:
                        nx = (x0 - r) if delta[0] > 0 else (x1 + r)
                    else:
                        ny = (y0 - r) if delta[1] > 0 else (y1 + r)
            x, y = nx, ny
            _ = hit
    return {"x": x, "y": y}


def obstacle_endpoint_only(agent, delta, walls, p):
    x, y = agent["x"] + delta[0], agent["y"] + delta[1]
    for x0, x1, y0, y1 in walls:
        if x0 - agent["r"] < x < x1 + agent["r"] and y0 - agent["r"] < y < y1 + agent["r"]:
            return {"x": agent["x"], "y": agent["y"]}
    return {"x": x, "y": y}


def chain_ref(q, dq, links, p):
    import math as m
    out = []
    for qi, di, lk in zip(q, dq, links):
        v = qi + di
        if lk.get("lo") is not None:
            v = max(lk["lo"], v)
        if lk.get("hi") is not None:
            v = min(lk["hi"], v)
        out.append(v)
    x = y = a = 0.0
    pts = []
    for v, lk in zip(out, links):
        a += v
        x += lk["length"] * m.cos(a)
        y += lk["length"] * m.sin(a)
        pts.append((x, y))
    return {"q": out, "points": pts}


def chain_wrapping(q, dq, links, p):
    r = chain_ref(q, dq, links, p)
    r["q"] = [math.remainder(v, 2 * math.pi) for v in r["q"]]
    return r


def grasp_ref(eff, eff_next, grip, obj, world, p):
    h, f = world["half"], world["floor"]
    x, y, z = obj["x"], obj["y"], obj["z"]
    held = grip > world["grip_threshold"] and math.dist(eff, (x, y, z)) <= world["grasp_radius"]
    if held:
        x, y, z = x + eff_next[0] - eff[0], y + eff_next[1] - eff[1], z + eff_next[2] - eff[2]
        return {"x": x, "y": y, "z": max(z, f + h)}
    z = f + h
    reach = h + world["eff_radius"]
    if abs(eff_next[2] - z) < reach:
        dx, dy = x - eff_next[0], y - eff_next[1]
        if abs(dx) < reach and abs(dy) < reach:
            if abs(dx) >= abs(dy):
                x = eff_next[0] + math.copysign(reach, dx if dx else eff_next[0] - eff[0])
            else:
                y = eff_next[1] + math.copysign(reach, dy if dy else eff_next[1] - eff[1])
    return {"x": x, "y": y, "z": z}


def grasp_no_drop(eff, eff_next, grip, obj, world, p):
    r = grasp_ref(eff, eff_next, grip, obj, world, p)
    if not (grip > world["grip_threshold"]):
        r["z"] = obj["z"]
    return r


def main() -> None:
    for fam, fn in (("obstacle", obstacle_ref), ("obstacle", obstacle_endpoint_only), ("chain", chain_ref),
                    ("chain", chain_wrapping), ("grasp", grasp_ref), ("grasp", grasp_no_drop)):
        res = FAMILIES[fam].run_tests(fn)
        print(f"== {fam} {fn.__name__}: {sum(r['passed'] for r in res)}/{len(res)}")
        print(report(res))


if __name__ == "__main__":
    main()
