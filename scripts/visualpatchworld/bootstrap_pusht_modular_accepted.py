#!/usr/bin/env python
"""Fit a PushT modular gated-MLP accepted model + run structure probes (Phase E bootstrap)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from visualpatchworld.code_baselines.common import load_trajectory_glob, subsample_trajectories, sanitize_generated_code
from visualpatchworld.modular_dynamics import (
    generate_modular_offline_gated_mlp_model_code,
    load_gated_mlp_params_from_code,
)
from visualpatchworld.schema_harness.discovery import probe_pusht_structure, structure_sketch_from_probes
from visualpatchworld.schema_harness.timeline import Timeline
from visualpatchworld.schema_harness.vpw_adoption import write_vpw_schema_artifacts


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train_glob", default="artifacts/visualpatchworld/data/pusht_oracle_traj_train.jsonl")
    p.add_argument("--max_train", type=int, default=24)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", default="artifacts/visualpatchworld/schema_harness/phase_e/pusht_modular_accepted")
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train = load_trajectory_glob(args.train_glob)
    train = subsample_trajectories(train, args.max_train, args.seed)
    print(f"[bootstrap] fitting modular gated-MLP on {len(train)} trajs", flush=True)
    code = sanitize_generated_code(generate_modular_offline_gated_mlp_model_code(train, seed=args.seed))
    params = load_gated_mlp_params_from_code(code)
    (out / "base_params.json").write_text(json.dumps(params, indent=2), encoding="utf-8")

    print("[bootstrap] running PushT structure probes", flush=True)
    probes = probe_pusht_structure()
    sketch = structure_sketch_from_probes(probes)
    timeline = Timeline(env="pusht")
    timeline.extend_from_trajectories(train, source="seed")
    for r in probes:
        timeline.add_note(f"probe[{r.name}] winner={r.winner}")
        for n in r.notes:
            timeline.add_note(f"  {n}")
    timeline.add_note(f"structure_sketch={sketch}")

    report = write_vpw_schema_artifacts(
        output_dir=out,
        env_name="pusht",
        model_code=code,
        train_trajectories=train,
        notes=timeline.notes,
        residual_epsilon=5.0,
        pass_threshold=0.5,
    )
    (out / "structure_sketch.json").write_text(json.dumps(sketch, indent=2), encoding="utf-8")
    (out / "probes.json").write_text(json.dumps([r.to_dict() for r in probes], indent=2), encoding="utf-8")
    print(json.dumps({"sketch": sketch, "adoption": report["backtest"], "out": str(out)}, indent=2))


if __name__ == "__main__":
    main()
