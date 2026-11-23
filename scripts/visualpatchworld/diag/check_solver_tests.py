"""Sanity-check solver_tests with a correct reference solver and the LLM's unverified ones."""
import json, math, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from visualpatchworld.llm_family.solver_tests import run_tests, report

def reference(disc, body, dt, p):
    n_sub = int(p.get("n_sub", 10))
    x, y, a = body["x"], body["y"], body["ang"]
    cu, cv = body["com"]
    k = body["inertia"] / body["mass"]
    c, s = math.cos(a), math.sin(a)
    gx, gy = x + c*cu - s*cv, y + s*cu + c*cv
    px, py = disc["x"], disc["y"]
    for i in range(n_sub):
        px += disc["vx"]*dt/n_sub; py += disc["vy"]*dt/n_sub
        for _ in range(4):
            c, s = math.cos(a), math.sin(a)
            ox, oy = gx - (c*cu - s*cv), gy - (s*cu + c*cv)
            du, dv = px-ox, py-oy
            u, v = c*du + s*dv, -s*du + c*dv
            best = None
            for u0,u1,v0,v1 in body["shape"]:
                qu, qv = min(max(u,u0),u1), min(max(v,v0),v1)
                eu, ev = u-qu, v-qv; d = math.hypot(eu, ev)
                if d < 1e-9:
                    dd = [(u-u0,(-1,0)),(u1-u,(1,0)),(v-v0,(0,-1)),(v1-v,(0,1))]
                    m, nn = min(dd); depth = disc["r"] + m; nb=( -nn[0], -nn[1])
                else:
                    depth = disc["r"] - d; nb = (-eu/d, -ev/d)
                if depth > 0 and (best is None or depth > best[0]):
                    best = (depth, nb, qu, qv)
            if best is None: break
            depth, (nu, nv), qu, qv = best
            nx, ny = c*nu - s*nv, s*nu + c*nv
            qx, qy = ox + c*qu - s*qv, oy + s*qu + c*qv
            rx, ry = qx-gx, qy-gy
            rn = rx*ny - ry*nx
            lam = depth/(1 + rn*rn/k)
            gx += lam*nx; gy += lam*ny; a += lam*rn/k
    c, s = math.cos(a), math.sin(a)
    return {"x": gx-(c*cu-s*cv), "y": gy-(s*cu+c*cv), "ang": a, "vx": 0.0, "vy": 0.0, "w": 0.0}

print("reference:"); print(report(run_tests(reference)))
