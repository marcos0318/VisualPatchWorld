#!/usr/bin/env python
"""Evaluate an induced PushT world model against simulator ground truth."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
PATCHWORLD_ROOT = ROOT  # vendored package at repo root
pythonpath_parts = [str(PATCHWORLD_ROOT), str(SRC)]
if os.environ.get("PYTHONPATH"):
    pythonpath_parts.append(os.environ["PYTHONPATH"])
os.environ["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(PATCHWORLD_ROOT) not in sys.path:
    sys.path.insert(0, str(PATCHWORLD_ROOT))

try:
    import hdf5plugin  # noqa: F401
except ModuleNotFoundError:
    pass

from visualpatchworld.hdf5 import require_h5py, resolve_dataset_path
from visualpatchworld.sim_eval import (
    compare_model_to_oracle_graphs,
    evaluate_sim_steps,
    load_induced_model,
    resolve_generated_model_path,
)


def _log(message: str) -> None:
    print(f"[visualpatchworld][{datetime.now().isoformat(timespec='seconds')}][sim-eval] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["oracle", "vlm", "tool"], default="oracle")
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument(
        "--trajectories_jsonl",
        default=None,
        help="Override trajectory file. Defaults to lewm-aligned oracle JSONL when present.",
    )
    parser.add_argument("--dataset", default="pusht_expert_train.h5")
    parser.add_argument("--stablewm_home", default=os.environ.get("STABLEWM_HOME"))
    parser.add_argument("--results_dir", default=str(ROOT / "artifacts" / "visualpatchworld" / "results"))
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--data_dir", default=str(ROOT / "artifacts" / "visualpatchworld" / "lewm_aligned"))
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--max_transitions", type=int, default=0, help="<=0 evaluates all transitions.")
    parser.add_argument("--position_tolerance", type=float, default=5.0)
    parser.add_argument("--velocity_tolerance", type=float, default=25.0)
    parser.add_argument(
        "--mode",
        choices=["sim", "oracle_graph", "both"],
        default="both",
        help="sim rolls swm/PushT-v1; oracle_graph compares against stored oracle next graphs.",
    )
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def _default_trajectory_path(args: argparse.Namespace) -> Path:
    if args.trajectories_jsonl:
        return Path(args.trajectories_jsonl)
    data_dir = Path(args.data_dir)
    lewm_path = data_dir / f"pusht_lewm_oracle_traj_{args.split}.jsonl"
    if lewm_path.is_file() and lewm_path.stat().st_size > 0:
        return lewm_path
    fallback = ROOT / "artifacts" / "visualpatchworld" / "data" / f"pusht_{args.source}_traj_{args.split}.jsonl"
    if fallback.is_file() and fallback.stat().st_size > 0:
        return fallback
    raise SystemExit(
        f"No trajectory JSONL found. Expected {lewm_path} or {fallback}. "
        "Run sample_lewm_aligned_trajectories.py first."
    )


def main() -> None:
    args = parse_args()
    if not (PATCHWORLD_ROOT / "patchworld").exists():
        raise SystemExit(f"PatchWorld dependency not found: {PATCHWORLD_ROOT}")

    traj_path = _default_trajectory_path(args)
    model_path = resolve_generated_model_path(Path(args.results_dir), args.source, args.model_path)
    _log(f"Loading induced model from {model_path}")
    model = load_induced_model(model_path)

    summary = {
        "model_path": str(model_path),
        "trajectories_jsonl": str(traj_path),
        "source": args.source,
        "split": args.split,
        "mode": args.mode,
    }

    if args.mode in ("sim", "both"):
        dataset_path = resolve_dataset_path(args.dataset, args.stablewm_home)
        h5py = require_h5py()
        _log(f"Rolling simulator ground truth from {dataset_path}")
        with h5py.File(dataset_path, "r") as h5:
            if "state" not in h5:
                raise SystemExit("Simulator evaluation requires `state` in the HDF5 dataset.")
            summary["sim_ground_truth"] = evaluate_sim_steps(
                model=model,
                states_h5=h5["state"],
                actions_h5=h5["action"],
                trajectories_jsonl=traj_path,
                frameskip=args.frameskip,
                max_transitions=args.max_transitions,
                position_tolerance=args.position_tolerance,
                velocity_tolerance=args.velocity_tolerance,
            )

    if args.mode in ("oracle_graph", "both"):
        _log("Comparing model predictions against stored oracle next graphs")
        summary["oracle_graph"] = compare_model_to_oracle_graphs(
            model=model,
            trajectories_jsonl=traj_path,
            max_transitions=args.max_transitions,
            position_tolerance=args.position_tolerance,
        )

    output = (
        Path(args.output)
        if args.output
        else ROOT / "artifacts" / "visualpatchworld" / "reports" / f"pusht_sim_steps_{args.source}_{args.split}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _log(f"Wrote sim-step evaluation to {output}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
