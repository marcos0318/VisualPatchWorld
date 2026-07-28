#!/usr/bin/env python
"""Validate that the active-probing structure identifier genuinely discriminates.

A discovery procedure is only meaningful if it can be *refuted*: the probes must
return different structures for different ground-truth dynamics, not always emit
"PD + quasi-static". This script runs the same probes against two synthetic
black-box sims with *known, deliberately different* structure:

* ``FakeLinearQuasiStatic`` -- a memoryless (first-order) agent + quasi-static
  block. The identifier must report ``agent_law = linear``.
* ``FakePDMomentum``        -- a PD (second-order) agent + a block that carries
  momentum (no damping). The identifier must report ``block_law = momentum``.

Both synthetic sims share the *collision primitive* with the real pipeline, so
only the structural choices differ. We then optionally confirm the real PushT sim
is identified as (pd, quasi_static, contact_driven).
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
PATCHWORLD_ROOT = ROOT  # vendored package at repo root
for p in (str(SRC), str(PATCHWORLD_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("MUJOCO_GL", "egl")

from visualpatchworld import collision, program_induction as pi  # noqa: E402

_GEO = collision.oracle_geometry()
_POLYS = _GEO["polygons"]
_RADIUS = _GEO["agent_radius"]
_COG = _GEO["cog"]


def _contact(ax, ay, bx, by, th):
    c = collision.pusht_contact((ax, ay), (bx, by), th, agent_radius=_RADIUS, polygons=_POLYS, cog=_COG)
    return c["penetration"], c["normal"]


class FakeLinearQuasiStatic:
    """Memoryless agent (pos += gain*action; no velocity) + quasi-static block."""

    GAIN = 28.0

    def reset(self, state):
        s = [float(v) for v in state]
        self.ax, self.ay, self.bx, self.by, self.th = s[0], s[1], s[2], s[3], s[4]
        return self._obs()

    def step(self, action):
        self.ax += self.GAIN * float(action[0])
        self.ay += self.GAIN * float(action[1])
        pen, (nx, ny) = _contact(self.ax, self.ay, self.bx, self.by, self.th)
        if pen > 0:  # quasi-static: block displaces to resolve overlap, keeps no momentum
            self.bx -= nx * pen
            self.by -= ny * pen
        return self._obs()

    def _obs(self):
        return [self.ax, self.ay, self.bx, self.by, self.th, 0.0, 0.0]

    def close(self):
        pass


class FakePDMomentum:
    """PD agent (velocity state) + block that carries momentum (no damping)."""

    SCALE = 100.0
    KP = 100.0
    KV = 20.0
    SUB = 10
    DT = 0.01
    KN = 0.6

    def reset(self, state):
        s = [float(v) for v in state]
        self.ax, self.ay, self.bx, self.by, self.th = s[0], s[1], s[2], s[3], s[4]
        self.avx, self.avy = (s[5], s[6]) if len(s) >= 7 else (0.0, 0.0)
        self.bvx = self.bvy = 0.0  # hidden block momentum, not exposed in obs
        return self._obs()

    def step(self, action):
        tx = self.ax + self.SCALE * float(action[0])
        ty = self.ay + self.SCALE * float(action[1])
        for _ in range(self.SUB):
            accx = self.KP * (tx - self.ax) - self.KV * self.avx
            accy = self.KP * (ty - self.ay) - self.KV * self.avy
            self.avx += accx * self.DT
            self.avy += accy * self.DT
            self.ax += self.avx * self.DT
            self.ay += self.avy * self.DT
            self.bx += self.bvx * self.DT  # momentum: damping = 0, velocity persists
            self.by += self.bvy * self.DT
            pen, (nx, ny) = _contact(self.ax, self.ay, self.bx, self.by, self.th)
            if pen > 0:
                self.bvx += -nx * pen * self.KN / self.DT
                self.bvy += -ny * pen * self.KN / self.DT
                self.bx -= nx * pen
                self.by -= ny * pen
        return self._obs()

    def _obs(self):
        return [self.ax, self.ay, self.bx, self.by, self.th, self.avx, self.avy]

    def close(self):
        pass


def _check(name, report, *, expect):
    ok = True
    for field, want in expect.items():
        got = getattr(report, field).choice
        status = "OK " if got == want else "FAIL"
        if got != want:
            ok = False
        d = getattr(report, field)
        print(f"  [{status}] {field}: got={got!r} want={want!r} (stat={d.statistic:.3f} margin={d.margin:.2f})")
    print(f"  => {name}: {'PASS' if ok else 'FAIL'} sketch={report.sketch}\n")
    return ok


def main() -> None:
    all_ok = True

    print("Synthetic control 1: linear/memoryless agent + quasi-static block")
    rep = pi.identify_structure(FakeLinearQuasiStatic())  # type: ignore[arg-type]
    all_ok &= _check("linear+quasi-static", rep, expect={
        "agent_law": "linear", "block_law": "quasi_static", "contact_law": "contact_driven"})

    print("Synthetic control 2: PD agent + momentum block")
    rep = pi.identify_structure(FakePDMomentum())  # type: ignore[arg-type]
    all_ok &= _check("pd+momentum", rep, expect={
        "agent_law": "pd", "block_law": "momentum", "contact_law": "contact_driven"})

    if "--no-sim" not in sys.argv:
        print("Real PushT simulator (ground truth: PD + quasi-static)")
        sim = pi.BlackBoxSim()
        try:
            rep = pi.identify_structure(sim)
        finally:
            sim.close()
        all_ok &= _check("real-pusht", rep, expect={
            "agent_law": "pd", "block_law": "quasi_static", "contact_law": "contact_driven"})

    print("ALL PASS" if all_ok else "SOME CHECKS FAILED")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
