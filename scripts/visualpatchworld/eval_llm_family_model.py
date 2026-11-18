#!/usr/bin/env python
"""Plan with one or more model files under the frozen GT>=90 protocol of the paper."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_fair_wm_ranking_gt90 as gt90  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env", required=True)
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--seeds", default="42")
    p.add_argument("--num_eval", type=int, default=50)
    p.add_argument("--out", required=True, help="JSON file to write (appends per model/seed)")
    p.add_argument("--episode_workers", type=int, default=8)
    args = p.parse_args()

    proto = gt90.PROTOCOLS[args.env]
    out = Path(args.out)
    results = json.loads(out.read_text()) if out.is_file() else {}
    for model in args.models:
        for seed in [int(s) for s in args.seeds.split(",")]:
            key = f"{Path(model).name}@seed{seed}"
            if key in results:
                continue
            if proto["backend"] == "pusht_parallel":
                rep = gt90._run_pusht(proto, Path(model), args.num_eval, seed, args.episode_workers)
            else:
                rep = gt90._run_model_general(args.env, proto, Path(model), args.num_eval, seed)
            results[key] = {
                "model": str(model),
                "seed": seed,
                "success_rate": rep["success_rate"],
                "mean_goal_distance": rep.get("mean_goal_distance"),
                "episode_successes": rep.get("episode_successes"),
                "elapsed_s": rep.get("elapsed_s"),
                "protocol": proto["tag"],
            }
            out.write_text(json.dumps(results, indent=2), encoding="utf-8")
            print(f"[eval] {key}: {rep['success_rate']:.1f}%", flush=True)


if __name__ == "__main__":
    main()
