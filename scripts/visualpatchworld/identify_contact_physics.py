#!/usr/bin/env python
"""Identify a pusher/body contact world model from state traces alone (see
``visualpatchworld.llm_family.sysid.identify_pusher_body``) and render it as a
planner-loadable model. No simulator access and no privileged metadata are used."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from visualpatchworld.llm_family import sysid  # noqa: E402
from visualpatchworld.llm_family.render import render_model  # noqa: E402
from visualpatchworld.llm_family.traces import load_episodes, schema_summary, state_keys, state_scales  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env", default="pusht")
    p.add_argument("--data", default=str(Path.home() / "le-wm-code/artifacts/visualpatchworld/data_stride1"))
    p.add_argument("--shape_episodes", type=int, default=None, help="episodes for identification (default all)")
    p.add_argument("--fit_episodes", type=int, default=10)
    p.add_argument("--substeps", default="10,5,1")
    p.add_argument("--out", default=str(ROOT / "artifacts/visualpatchworld/sysid_models/pusht_sysid.py"))
    args = p.parse_args()
    data = Path(args.data)
    full = load_episodes(data / f"{args.env}_oracle_traj_train.jsonl", max_episodes=args.shape_episodes)
    train40 = load_episodes(data / f"{args.env}_oracle_traj_train.jsonl", max_episodes=40)
    val = load_episodes(data / f"{args.env}_oracle_traj_val.jsonl", max_episodes=10)
    keys = state_keys(train40)
    scales = state_scales(train40, keys)
    res = sysid.identify_pusher_body(full, val, keys, scales, substeps=[int(x) for x in args.substeps.split(",")],
                                     fit_episodes=args.fit_episodes, log=lambda m: print(m, flush=True))
    if res is None:
        raise SystemExit("no pusher/body pair found")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    code = render_model(res["candidate"], res["params"], env=args.env, action_dim=schema_summary(train40)["action_dim"],
                        state_keys=keys, objective="multi_step", static_meta=res["meta"])
    out.write_text(code, encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps({"shape": res["meta"], **{k: v for k, v in res.items() if k not in ("candidate", "meta")}},
                                                   indent=2, default=str), encoding="utf-8")
    print(f"[sysid] best substeps={res['best']['substeps']} val5={res['best']['val5']:.4f} -> {out}", flush=True)


if __name__ == "__main__":
    main()
