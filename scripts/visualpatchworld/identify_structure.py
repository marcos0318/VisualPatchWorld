#!/usr/bin/env python
"""Discover dynamics *structure* by active probing, then (optionally) induce.

Level-1 of the two-level program-induction pipeline. Works for all four LeWM
environments:

* ``pusht``   — agent law / block law / contact law (program_induction)
* ``tworoom`` — target static / action-aligned / hold-still → linear scaffold
* ``cube``    — goal static / grip-gated contact / far-decoupled → contact scaffold
* ``reacher`` — joint actuation / target static / first-order damping → joint+FK

With ``--trajectories`` it runs Level 2: the discovered sketch selects a program
template whose constants are fit from data.
"""

from __future__ import annotations

import argparse
import json
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--env",
        default="pusht",
        choices=["pusht", "tworoom", "cube", "reacher"],
        help="LeWM environment to probe (default: pusht).",
    )
    ap.add_argument("--dataset", default=None, help="Optional HDF5 dataset override for probe starts.")
    ap.add_argument("--geometry_json", default=None, help="Perceived geometry JSON (PushT contact probes).")
    ap.add_argument("--report_out", default=None, help="Write the structure report JSON here.")
    ap.add_argument("--trajectories", default=None, help="Glob of training trajectories for Level-2 fitting.")
    ap.add_argument("--val_trajectories", default=None, help="Glob of held-out trajectories for restart selection.")
    ap.add_argument("--out", default=None, help="Write the induced model code here (requires --trajectories).")
    ap.add_argument("--rollout_k", type=int, default=6)
    ap.add_argument("--rollout_fit_steps", type=int, default=400)
    ap.add_argument("--fit_restarts", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from visualpatchworld import collision
    from visualpatchworld import structure_induction as si

    geometry = collision.load_perceived_geometry(args.geometry_json) if args.geometry_json else None

    print(f"[identify] env={args.env} running Level-1 active probing ...", flush=True)
    report = si.identify_env_structure(
        args.env,
        dataset=args.dataset,
        geometry=geometry,
        seed=args.seed,
    )
    report_json = report.to_json()
    print(json.dumps(report_json, indent=2))
    print(
        f"\n[identify] discovered summary={report.sketch.get('summary')} "
        f"template={report.template} canonical={report.canonical}"
    )

    if args.report_out:
        Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report_out).write_text(json.dumps(report_json, indent=2), encoding="utf-8")
        print(f"[identify] wrote structure report -> {args.report_out}")

    if args.trajectories:
        from patchworld.worldmodel_data import load_trajectories

        paths = sorted(Path().glob(args.trajectories)) or [Path(args.trajectories)]
        print(f"[identify] loading trajectories for Level-2 fit: {args.trajectories}")
        trajs = load_trajectories(paths)
        if not trajs:
            raise SystemExit(f"No trajectories loaded from: {args.trajectories}")
        val_trajs = None
        if args.val_trajectories:
            vpaths = sorted(Path().glob(args.val_trajectories)) or [Path(args.val_trajectories)]
            val_trajs = load_trajectories(vpaths)
            print(f"[identify] loaded {len(val_trajs)} val trajectories for restart selection")
        result = si.induce_from_structure(
            args.env,
            trajs,
            report=report,
            geometry_json=args.geometry_json,
            dataset=args.dataset,
            val_trajectories=val_trajs,
            rollout_k=args.rollout_k,
            steps=args.rollout_fit_steps,
            restarts=args.fit_restarts,
        )
        out = args.out or str(
            ROOT / "artifacts" / "visualpatchworld" / "induced" / f"{args.env}_program_induced_model.py"
        )
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(result["code"], encoding="utf-8")
        print(f"[identify] Level-2 fit complete via template '{result['template']}'")
        print(f"[identify] wrote induced model -> {out}")


if __name__ == "__main__":
    main()
