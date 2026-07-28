#!/usr/bin/env python
"""Unified transition acquisition for modular dynamics refitting."""

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

from visualpatchworld.sim_eval import load_induced_model, resolve_generated_model_path
from visualpatchworld.transition_acquisition import (
    AcquisitionConfig,
    acquire_transitions,
    write_acquired_jsonl,
)


def _log(message: str) -> None:
    print(
        f"[visualpatchworld][{datetime.now().isoformat(timespec='seconds')}][acquire] {message}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["oracle", "vlm", "tool"], default="tool")
    parser.add_argument(
        "--results_dir",
        default=str(ROOT / "artifacts" / "visualpatchworld" / "lewm_tool_modular_offline_results"),
    )
    parser.add_argument("--model_path", default=None)
    parser.add_argument(
        "--trajectories_jsonl",
        default=str(ROOT / "artifacts" / "visualpatchworld" / "lewm_tool_data" / "pusht_tool_traj_train.jsonl"),
    )
    parser.add_argument("--dataset", default="pusht_expert_train.h5")
    parser.add_argument("--stablewm_home", default=os.environ.get("STABLEWM_HOME"))
    parser.add_argument(
        "--output",
        default=str(ROOT / "artifacts" / "visualpatchworld" / "lewm_tool_data" / "unified_acquired_transitions.jsonl"),
    )
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--top_k", type=int, default=200)
    parser.add_argument("--num_explore", type=int, default=96)
    parser.add_argument("--num_supervision", type=int, default=400)
    parser.add_argument("--planning_num_eval", type=int, default=16)
    parser.add_argument("--planning_num_samples", type=int, default=64)
    parser.add_argument("--planning_cem_iters", type=int, default=3)
    parser.add_argument("--error_threshold", type=float, default=5.0)
    parser.add_argument("--regret_threshold", type=float, default=20.0)
    parser.add_argument("--sim_error_weight", type=float, default=1.0)
    parser.add_argument("--regret_weight", type=float, default=1.5)
    parser.add_argument("--near_contact_weight", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_counterexamples", action="store_true")
    parser.add_argument("--skip_planning_failures", action="store_true")
    parser.add_argument("--skip_supervision", action="store_true")
    parser.add_argument(
        "--diagnostics_output",
        default=None,
        help="Optional path to write acquisition diagnostics JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = resolve_generated_model_path(Path(args.results_dir), args.source, args.model_path)
    _log(f"Loading model from {model_path}")
    model = load_induced_model(model_path)

    config = AcquisitionConfig(
        top_k=args.top_k,
        num_explore=args.num_explore,
        num_supervision=args.num_supervision,
        planning_num_eval=args.planning_num_eval,
        planning_num_samples=args.planning_num_samples,
        planning_cem_iters=args.planning_cem_iters,
        error_threshold=args.error_threshold,
        regret_threshold=args.regret_threshold,
        sim_error_weight=args.sim_error_weight,
        regret_weight=args.regret_weight,
        near_contact_weight=args.near_contact_weight,
        seed=args.seed,
    )
    rows, diagnostics = acquire_transitions(
        model=model,
        config=config,
        trajectories_jsonl=Path(args.trajectories_jsonl) if args.trajectories_jsonl else None,
        dataset=args.dataset,
        stablewm_home=args.stablewm_home,
        frameskip=args.frameskip,
        include_counterexamples=not args.skip_counterexamples,
        include_planning_failures=not args.skip_planning_failures,
        include_supervision=not args.skip_supervision,
    )
    diagnostics["model_path"] = str(model_path)
    diagnostics["output"] = args.output
    diagnostics["acquired_count"] = len(rows)

    write_acquired_jsonl(rows, Path(args.output), item_id="unified-acquired-transitions")
    _log(f"Wrote {len(rows)} acquired transitions to {args.output}")
    if args.diagnostics_output:
        Path(args.diagnostics_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.diagnostics_output).write_text(
            json.dumps(diagnostics, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(diagnostics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
