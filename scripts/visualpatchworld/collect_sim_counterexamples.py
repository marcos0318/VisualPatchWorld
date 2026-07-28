#!/usr/bin/env python
"""Collect simulator-labeled counterexamples for active dynamics induction."""

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

from visualpatchworld.active_dynamics_induction import (
    collect_sim_counterexamples,
    summarize_counterexamples,
    write_counterexample_jsonl,
)
from visualpatchworld.sim_eval import load_induced_model, resolve_generated_model_path


def _log(message: str) -> None:
    print(
        f"[visualpatchworld][{datetime.now().isoformat(timespec='seconds')}][sim-ce] {message}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["oracle", "vlm", "tool"], default="tool")
    parser.add_argument(
        "--results_dir",
        default=str(ROOT / "artifacts" / "visualpatchworld" / "lewm_tool_scaffold_fit_gated_mlp_weighted_results"),
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
        default=str(ROOT / "artifacts" / "visualpatchworld" / "lewm_tool_data" / "sim_counterexample_transitions.jsonl"),
    )
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--max_transitions", type=int, default=0)
    parser.add_argument("--num_explore", type=int, default=96)
    parser.add_argument("--error_threshold", type=float, default=5.0)
    parser.add_argument("--top_k", type=int, default=160)
    parser.add_argument("--base_weight", type=float, default=6.0)
    parser.add_argument("--error_scale", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = resolve_generated_model_path(Path(args.results_dir), args.source, args.model_path)
    _log(f"Loading model from {model_path}")
    model = load_induced_model(model_path)

    rows = collect_sim_counterexamples(
        model=model,
        trajectories_jsonl=Path(args.trajectories_jsonl) if args.trajectories_jsonl else None,
        dataset=args.dataset,
        stablewm_home=args.stablewm_home,
        frameskip=args.frameskip,
        max_transitions=args.max_transitions,
        num_explore=args.num_explore,
        error_threshold=args.error_threshold,
        top_k=args.top_k,
        base_weight=args.base_weight,
        error_scale=args.error_scale,
        seed=args.seed,
    )
    summary = summarize_counterexamples(rows)
    summary["model_path"] = str(model_path)
    summary["output"] = args.output
    summary["weighted_transition_count"] = len(rows)

    write_counterexample_jsonl(rows, Path(args.output), item_id="sim-counterexamples")
    _log(f"Wrote {len(rows)} sim counterexamples to {args.output}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
