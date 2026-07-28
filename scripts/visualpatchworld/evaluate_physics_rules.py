#!/usr/bin/env python
"""Evaluate PushT physics-rule compliance on dataset transitions and model predictions."""

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
  evaluate_dataset_rule_compliance,
  evaluate_model_rule_compliance,
  stratified_position_errors,
)
from visualpatchworld.sim_eval import load_induced_model, resolve_generated_model_path


def _log(message: str) -> None:
  print(f"[visualpatchworld][{datetime.now().isoformat(timespec='seconds')}][physics-rules] {message}", flush=True)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--trajectories_glob", required=True)
  parser.add_argument("--source", choices=["oracle", "vlm", "tool"], default="tool")
  parser.add_argument("--results_dir", default=str(ROOT / "artifacts" / "visualpatchworld" / "lewm_tool_scaffold_fit_results"))
  parser.add_argument("--model_path", default=None)
  parser.add_argument("--output", default=None)
  parser.add_argument("--max_trajectories", type=int, default=0)
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  paths = sorted(Path(p) for p in glob(args.trajectories_glob))
  if not paths:
    raise SystemExit(f"No trajectories matched: {args.trajectories_glob}")
  trajectories = load_trajectories(paths)
  if args.max_trajectories > 0 and len(trajectories) > args.max_trajectories:
    trajectories = trajectories[: args.max_trajectories]

  report: dict = {
    "trajectories_glob": args.trajectories_glob,
    "n_trajectories": len(trajectories),
    "dataset_rules": evaluate_dataset_rule_compliance(trajectories),
    "stratified_position_errors": stratified_position_errors(trajectories),
  }

  model_path = resolve_generated_model_path(Path(args.results_dir), args.source, args.model_path)
  _log(f"Loading model from {model_path}")
  model = load_induced_model(model_path)
  report["model_path"] = str(model_path)
  report["model_rules"] = evaluate_model_rule_compliance(model, trajectories)
  report["model_rules"]["stratified_prediction_errors"] = _model_stratified_errors(model, trajectories)

  output = (
    Path(args.output)
    if args.output
    else ROOT / "artifacts" / "visualpatchworld" / "reports" / "pusht_physics_rules_report.json"
  )
  output.parent.mkdir(parents=True, exist_ok=True)
  output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
  _log(f"Wrote physics rule report to {output}")
  print(json.dumps(report, indent=2, sort_keys=True))


def _model_stratified_errors(model, trajectories):
  import math

  from visualpatchworld.lewm_sampling import state_field_errors
  from visualpatchworld.pusht_local_induction import _parse_action, _parse_observation_text, _state_from_graph
  from visualpatchworld.sim_eval import _predict_next_graph

  near_block = []
  far_block = []
  for traj in trajectories:
    for transition in traj.transitions:
      obs = str(transition.observation)
      action = str(transition.action)
      state = _state_from_graph(_parse_observation_text(obs))
      sx, sy, bx, by, _ = state
      dist = math.hypot(bx - sx, by - sy)
      predicted = _predict_next_graph(model, obs, action)
      target = _state_from_graph(_parse_observation_text(str(transition.next_observation)))
      errors = state_field_errors(predicted, target)
      if dist <= 70.0:
        near_block.append(errors["block_position_error"])
      else:
        far_block.append(errors["block_position_error"])
  return {
    "near_contact_block_error_mean": sum(near_block) / len(near_block) if near_block else None,
    "far_contact_block_error_mean": sum(far_block) / len(far_block) if far_block else None,
  }


if __name__ == "__main__":
  main()
