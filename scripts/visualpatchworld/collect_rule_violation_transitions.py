#!/usr/bin/env python
"""Collect transitions where model predictions violate key PushT physics rules."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from glob import glob
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

from patchworld.worldmodel_data import load_trajectories
from visualpatchworld.pusht_physics_rules import (
  collect_rule_violation_weighted_transitions,
  write_weighted_transitions_jsonl,
)
from visualpatchworld.sim_eval import load_induced_model, resolve_generated_model_path


def _log(message: str) -> None:
  print(
    f"[visualpatchworld][{datetime.now().isoformat(timespec='seconds')}][rule-violations] {message}",
    flush=True,
  )


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--trajectories_glob", required=True)
  parser.add_argument("--source", choices=["oracle", "vlm", "tool"], default="tool")
  parser.add_argument("--results_dir", default=str(ROOT / "artifacts" / "visualpatchworld" / "lewm_tool_scaffold_fit_results"))
  parser.add_argument("--model_path", default=None)
  parser.add_argument("--output", default=str(ROOT / "artifacts" / "visualpatchworld" / "lewm_tool_data" / "rule_violation_transitions.jsonl"))
  parser.add_argument("--max_trajectories", type=int, default=0)
  parser.add_argument("--weight", type=float, default=3.0)
  parser.add_argument(
    "--target_rules",
    nargs="+",
    default=["R1_no_block_motion_when_far", "R2_contact_push_alignment", "R5_block_speed_limit"],
  )
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  paths = sorted(Path(p) for p in glob(args.trajectories_glob))
  if not paths:
    raise SystemExit(f"No trajectories matched: {args.trajectories_glob}")
  trajectories = load_trajectories(paths)
  if args.max_trajectories > 0 and len(trajectories) > args.max_trajectories:
    trajectories = trajectories[: args.max_trajectories]

  model_path = resolve_generated_model_path(Path(args.results_dir), args.source, args.model_path)
  _log(f"Loading model from {model_path}")
  model = load_induced_model(model_path)

  rows = collect_rule_violation_weighted_transitions(
    model,
    trajectories,
    weight=args.weight,
    target_rules=args.target_rules,
  )
  write_weighted_transitions_jsonl(
    rows,
    Path(args.output),
    source="rule_violation_mining",
    item_id=f"{args.source}_rule_violations",
  )

  summary = {
    "model_path": str(model_path),
    "trajectories_glob": args.trajectories_glob,
    "n_trajectories": len(trajectories),
    "weighted_transition_count": len(rows),
    "weight": args.weight,
    "target_rules": list(args.target_rules),
    "output": args.output,
  }
  _log(f"Wrote {len(rows)} rule-violation transitions to {args.output}")
  print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
