#!/usr/bin/env python
"""Mine high-regret planning transitions for weighted dynamics refitting."""

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

from visualpatchworld.planning_failure_mining import (
  failures_to_weighted_transitions,
  mine_planning_failures,
  summarize_failures,
  write_failure_jsonl,
)
from visualpatchworld.sim_eval import load_induced_model, resolve_generated_model_path


def _log(message: str) -> None:
  print(f"[visualpatchworld][{datetime.now().isoformat(timespec='seconds')}][mine-failures] {message}", flush=True)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--source", choices=["oracle", "vlm", "tool"], default="tool")
  parser.add_argument("--results_dir", default=str(ROOT / "artifacts" / "visualpatchworld" / "lewm_tool_scaffold_fit_results"))
  parser.add_argument("--model_path", default=None)
  parser.add_argument("--dataset", default="pusht_expert_train.h5")
  parser.add_argument("--stablewm_home", default=os.environ.get("STABLEWM_HOME"))
  parser.add_argument("--output", default=str(ROOT / "artifacts" / "visualpatchworld" / "lewm_tool_data" / "planning_failure_transitions.jsonl"))
  parser.add_argument("--num_eval", type=int, default=16)
  parser.add_argument("--failure_weight", type=float, default=5.0)
  parser.add_argument("--regret_threshold", type=float, default=20.0)
  parser.add_argument("--num_samples", type=int, default=64)
  parser.add_argument("--cem_iters", type=int, default=3)
  parser.add_argument("--seed", type=int, default=42)
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  model_path = resolve_generated_model_path(Path(args.results_dir), args.source, args.model_path)
  _log(f"Loading model from {model_path}")
  model = load_induced_model(model_path)

  failures = mine_planning_failures(
    model=model,
    dataset=args.dataset,
    stablewm_home=args.stablewm_home,
    num_eval=args.num_eval,
    failure_weight=args.failure_weight,
    regret_threshold=args.regret_threshold,
    num_samples=args.num_samples,
    cem_iters=args.cem_iters,
    seed=args.seed,
  )
  summary = summarize_failures(failures)
  summary["model_path"] = str(model_path)
  summary["output"] = args.output
  summary["weighted_transition_count"] = len(failures_to_weighted_transitions(failures))

  write_failure_jsonl(failures, Path(args.output))
  _log(f"Wrote {len(failures)} mined failures to {args.output}")
  print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
