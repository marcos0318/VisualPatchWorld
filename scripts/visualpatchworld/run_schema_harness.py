#!/usr/bin/env python
"""Run Schema-style coding harness on a LeWM environment (Two-room pilot)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from visualpatchworld.code_baselines.runner import DEFAULT_MODEL
from visualpatchworld.schema_harness import SchemaHarnessConfig, run_schema_harness_eval


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env", default="tworoom", choices=["tworoom", "pusht", "cube", "reacher"])
    p.add_argument(
        "--mode",
        default="sticky",
        choices=["scaffold", "llm", "free", "init", "sticky"],
        help="sticky/init=Schema-adopted sticky accepted+soft void; free=no void; scaffold/llm=induce",
    )
    p.add_argument("--no_soft_void", action="store_true", help="Disable soft void-on-surprise")
    p.add_argument("--revise_every_surprises", type=int, default=4)
    p.add_argument("--revision_mode", default="scaffold", choices=["scaffold", "llm", "modular"])
    p.add_argument("--no_structure_probes", action="store_true")
    p.add_argument("--no_revise_at_end", action="store_true")
    p.add_argument("--num_eval", type=int, default=5)
    p.add_argument("--eval_budget", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--residual_epsilon", type=float, default=0.5)
    p.add_argument("--pass_threshold", type=float, default=0.8)
    p.add_argument(
        "--init_model",
        default="",
        help="Optional path to a pre-induced world_model.py (use with --mode init or free)",
    )
    p.add_argument("--seed_timeline_transitions", type=int, default=40)
    p.add_argument("--discovery_macros", type=int, default=4)
    p.add_argument("--max_deliberation_rounds", type=int, default=3)
    p.add_argument("--num_samples", type=int, default=100)
    p.add_argument("--cem_iters", type=int, default=5)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument(
        "--train_glob",
        default="artifacts/visualpatchworld/data/tworoom_oracle_traj_train.jsonl",
    )
    p.add_argument(
        "--output_dir",
        default="artifacts/visualpatchworld/schema_harness/smoke",
    )
    p.add_argument(
        "--no_certify",
        action="store_true",
        help="Disable certify/void even for scaffold/llm (ablation helper)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = SchemaHarnessConfig(
        env=args.env,
        mode=args.mode,
        seed=args.seed,
        num_eval=args.num_eval,
        eval_budget=args.eval_budget,
        residual_epsilon=args.residual_epsilon,
        pass_threshold=args.pass_threshold,
        seed_timeline_transitions=args.seed_timeline_transitions,
        discovery_macros=args.discovery_macros,
        max_deliberation_rounds=args.max_deliberation_rounds,
        model_name=args.model,
        num_samples=args.num_samples,
        cem_iters=args.cem_iters,
        train_glob=args.train_glob,
        output_dir=args.output_dir,
        require_certify=not args.no_certify and args.mode != "free",
        init_model_path=args.init_model,
        soft_void=not args.no_soft_void,
        revise_every_surprises=args.revise_every_surprises,
        revision_mode=args.revision_mode,
        run_structure_probes=not args.no_structure_probes,
        revise_at_end=not args.no_revise_at_end,
    )
    summary = run_schema_harness_eval(cfg)
    print(json.dumps({k: summary[k] for k in summary if k != "episodes"}, indent=2))
    print(f"[schema_harness] wrote {Path(args.output_dir) / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
