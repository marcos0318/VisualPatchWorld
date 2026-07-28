#!/usr/bin/env python
"""Bootstrap a strong PushT accepted model for Schema harness Phase F.

Default: copy/refit ``program_induced_best.py`` (PD physics) which is the
paper's strong induced planner (~66% Oracle+Induced; much higher with hybrid).
Also runs structure probes and writes Schema artifacts.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from visualpatchworld.code_baselines.common import (
    load_trajectory_glob,
    sanitize_generated_code,
    subsample_trajectories,
)
from visualpatchworld.physics_sim_pd_dynamics import generate_pd_physics_model_code
from visualpatchworld.program_induction import induce_program
from visualpatchworld.pushing_mechanics_rollout import generate_collision_rollout_model_code
from visualpatchworld.schema_harness.discovery import probe_pusht_structure, structure_sketch_from_probes
from visualpatchworld.schema_harness.timeline import Timeline
from visualpatchworld.schema_harness.vpw_adoption import write_vpw_schema_artifacts


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--template",
        default="pd_best",
        choices=["pd_best", "pd_refit", "induce_program", "collision_rollout"],
    )
    p.add_argument(
        "--best_model",
        default="artifacts/visualpatchworld/induced/program_induced_best.py",
    )
    p.add_argument("--train_glob", default="artifacts/visualpatchworld/data/pusht_oracle_traj_train.jsonl")
    p.add_argument("--max_train", type=int, default=48)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--output_dir",
        default="artifacts/visualpatchworld/schema_harness/phase_f/pusht_strong_accepted",
    )
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    train = subsample_trajectories(load_trajectory_glob(args.train_glob), args.max_train, args.seed)

    if args.template == "pd_best":
        code = Path(args.best_model).read_text(encoding="utf-8")
        source = f"copy:{args.best_model}"
    elif args.template == "pd_refit":
        code = sanitize_generated_code(generate_pd_physics_model_code(train))
        source = "pd_refit"
    elif args.template == "collision_rollout":
        code = sanitize_generated_code(
            generate_collision_rollout_model_code(train, rollout_k=6, reg_lambda=20.0)
        )
        source = "collision_rollout"
    else:
        result = induce_program(train)
        code = sanitize_generated_code(result["code"])
        source = f"induce_program:{result['template']}"
        (out / "structure_l1.json").write_text(json.dumps(result["structure"], indent=2), encoding="utf-8")

    print(f"[bootstrap] template={args.template} source={source}", flush=True)
    probes = probe_pusht_structure()
    sketch = structure_sketch_from_probes(probes)
    timeline = Timeline(env="pusht")
    timeline.extend_from_trajectories(train, source="seed")
    timeline.add_note(f"bootstrap_source={source}")
    for r in probes:
        timeline.add_note(f"probe[{r.name}] winner={r.winner}")
    timeline.add_note(f"structure_sketch={sketch}")

    # PushT residuals are large in absolute pose units; adoption report is diagnostic.
    report = write_vpw_schema_artifacts(
        output_dir=out,
        env_name="pusht",
        model_code=code,
        train_trajectories=train,
        notes=timeline.notes,
        residual_epsilon=40.0,
        pass_threshold=0.2,
    )
    (out / "structure_sketch.json").write_text(json.dumps(sketch, indent=2), encoding="utf-8")
    (out / "probes.json").write_text(json.dumps([r.to_dict() for r in probes], indent=2), encoding="utf-8")
    (out / "bootstrap_meta.json").write_text(
        json.dumps({"template": args.template, "source": source, "backtest": report["backtest"]}, indent=2),
        encoding="utf-8",
    )
    # Keep an explicit alias for harness init.
    shutil.copyfile(out / "world_model.accepted.py", out / "world_model.pd.py")
    print(json.dumps({"sketch": sketch, "out": str(out), "median_residual": report["backtest"]["median_residual"]}, indent=2))


if __name__ == "__main__":
    main()
