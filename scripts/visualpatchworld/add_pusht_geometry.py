#!/usr/bin/env python
"""Add static PushT shape metadata to scene-graph traces: either the measured geometry
(disc radius, T as two body-frame rectangles, pose convention) or, with --meta_json,
the contact shape identified from the traces themselves. No mass, inertia, centre of
mass or controller constants are added."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

GEOMETRY = {
    "agent_shape": {"type": "disc", "radius": 15.0},
    "block_shape": {
        "type": "union_of_rectangles_in_block_frame",
        "pose_convention": "a block-frame point (u, v) is at world (block.x + u*cos(block.angle) - v*sin(block.angle), "
                           "block.y + u*sin(block.angle) + v*cos(block.angle))",
        "rectangles_u_min_u_max_v_min_v_max": [[-60.0, 60.0, 0.0, 30.0], [-15.0, 15.0, 30.0, 120.0]],
    },
    "walls": {"x_min": 5.0, "x_max": 506.0, "y_min": 5.0, "y_max": 506.0},
}


def _augment_text(text: str, extra: dict) -> str:
    out = []
    for line in str(text).splitlines():
        if line.startswith("Metadata:"):
            md = json.loads(line.split(":", 1)[1].strip() or "{}")
            md.update(extra)
            line = "Metadata: " + json.dumps(md, sort_keys=True, separators=(",", ":"))
        out.append(line)
    return "\n".join(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", default=str(Path.home() / "le-wm-code/artifacts/visualpatchworld/data_stride1"))
    p.add_argument("--dst", default=str(Path.home() / "le-wm-code/artifacts/visualpatchworld/data_stride1_geom"))
    p.add_argument("--meta_json", default=None,
                   help="JSON file whose 'shape' entry replaces the measured geometry (e.g. identify_contact_physics output)")
    args = p.parse_args()
    extra = json.loads(Path(args.meta_json).read_text())["shape"] if args.meta_json else GEOMETRY
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        src = Path(args.src) / f"pusht_oracle_traj_{split}.jsonl"
        with src.open() as fi, (dst / src.name).open("w") as fo:
            for line in fi:
                ep = json.loads(line)
                for t in ep["transitions"]:
                    for key in ("observation", "next_observation"):
                        if key in t:
                            t[key] = _augment_text(t[key], extra)
                fo.write(json.dumps(ep) + "\n")
        print("wrote", dst / src.name)


if __name__ == "__main__":
    main()
