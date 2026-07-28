#!/usr/bin/env python
"""Detect PushT collision geometry from rendered frames (no source access).

Renders frames from the simulator (rendering = allowed visual extraction),
segments the block/agent by colour, detects the block boundary contour and the
agent radius purely from pixels, calibrates the boundary into the scene-graph
block reference frame, and writes a perceived-geometry JSON consumed by the
``pusht_pushing_mechanics_collision`` strategy.

This replaces the cheating practice of reading the T polygon / agent radius from
the simulator's env.py.
"""

from __future__ import annotations

import argparse
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

import numpy as np  # noqa: E402

from visualpatchworld import visual_geometry as vg  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--num_frames", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default=str(ROOT / "artifacts" / "visualpatchworld" / "perceived_geometry" / "pusht_geometry.json"))
    args = ap.parse_args()

    import gymnasium as gym
    import stable_worldmodel.envs  # noqa: F401

    env = gym.make("swm/PushT-v1", render_mode="rgb_array")
    rng = np.random.default_rng(args.seed)
    frames = []
    poses = []
    for _ in range(args.num_frames):
        bx, by = rng.uniform(170, 340, 2)
        ang = rng.uniform(0, 2 * math.pi)
        ax, ay = rng.uniform(45, 160, 2)  # agent away from block to avoid occlusion
        env.reset(options={"state": np.array([ax, ay, bx, by, ang], dtype=float)})
        frames.append(np.asarray(env.render()).astype(np.float32))
        poses.append((float(bx), float(by), float(ang)))

    geom = vg.detect_and_calibrate(frames, poses)
    vg.save_geometry(geom, args.output)
    poly = np.array(geom["polygons"][0])
    print(f"perceived agent_radius={geom['agent_radius']:.2f}")
    print(f"perceived block polygon: {len(poly)} verts, "
          f"bbox x[{poly[:,0].min():.1f},{poly[:,0].max():.1f}] y[{poly[:,1].min():.1f},{poly[:,1].max():.1f}]")
    print(f"reference cog/offset={geom['cog']}")
    print(f"wrote perceived geometry -> {args.output}")


if __name__ == "__main__":
    main()
