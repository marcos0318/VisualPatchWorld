#!/usr/bin/env python
"""LLM-proposed hypothesis family -> fit every candidate -> offline/active selection.

Writes, under ``--output_dir``:
  family.json            proposed programs, fits, both selections, probe costs
  llm_log.json           raw prompts/replies (proposal + repairs)
  models/<sel>_<obj>.py  planner-loadable model for each selection x objective
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from visualpatchworld.llm_family.fit import fit_candidate  # noqa: E402
from visualpatchworld.llm_family.propose import propose_family  # noqa: E402
from visualpatchworld.llm_family.render import render_model  # noqa: E402
from visualpatchworld.llm_family.select import select_active, select_offline  # noqa: E402
from visualpatchworld.llm_family.traces import load_episodes, schema_summary, state_keys, state_scales  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env", required=True)
    p.add_argument("--data_dir", default=str(ROOT.parent / "le-wm-code/artifacts/visualpatchworld/data"))
    p.add_argument("--output_dir", required=True)
    p.add_argument("--num_hypotheses", type=int, default=6)
    p.add_argument("--model", default=None, help="LLM id (default $PATCHWORLD_MODEL)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--objectives", default="one_step,multi_step")
    p.add_argument("--rollout_k", type=int, default=5)
    p.add_argument("--restarts", type=int, default=4)
    p.add_argument("--max_nfev", type=int, default=400)
    p.add_argument("--num_probes", type=int, default=5)
    p.add_argument("--probe_len", type=int, default=3)
    p.add_argument("--no_active", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    (out / "models").mkdir(parents=True, exist_ok=True)
    data = Path(args.data_dir)
    train = load_episodes(data / f"{args.env}_oracle_traj_train.jsonl")
    val = load_episodes(data / f"{args.env}_oracle_traj_val.jsonl")
    keys = state_keys(train)
    scales = state_scales(train, keys)
    schema = schema_summary(train)
    print(f"[llm-family] {args.env}: {schema['num_episodes']} train eps, {schema['num_transitions']} transitions, keys={keys}", flush=True)

    t0 = time.time()
    llm_log: list = []
    family = propose_family(train, num_hypotheses=args.num_hypotheses, model=args.model, seed=args.seed, log=llm_log)
    (out / "llm_log.json").write_text(json.dumps(llm_log, indent=2), encoding="utf-8")
    print(f"[llm-family] {len(family)} valid candidates: {[c.name for c in family]}", flush=True)
    if not family:
        raise SystemExit("LLM produced no valid candidates")
    cands = {c.name: c for c in family}

    result = {
        "env": args.env,
        "llm_model": llm_log[0]["model"],
        "schema": schema,
        "scales": scales,
        "candidates": [c.to_json() for c in family],
        "fits": {},
        "selections": {},
        "args": vars(args),
    }
    sim = None
    if not args.no_active:
        from visualpatchworld.llm_family.sim_adapter import BlackBoxEnv

        sim = BlackBoxEnv(args.env)
    try:
        for objective in [o.strip() for o in args.objectives.split(",") if o.strip()]:
            fits = {}
            for name, cand in cands.items():
                fr = fit_candidate(
                    cand, train, val, keys, scales,
                    objective=objective, rollout_k=args.rollout_k,
                    restarts=args.restarts, max_nfev=args.max_nfev, seed=args.seed,
                )
                fits[name] = fr
                print(f"[llm-family] {objective:10s} {name:40s} val1={fr.val_one_step:.4f} val{args.rollout_k}={fr.val_rollout:.4f} ({fr.elapsed_s:.1f}s)", flush=True)
            result["fits"][objective] = {n: f.to_json() for n, f in fits.items()}
            selections = {"offline": select_offline(fits)}
            if sim is not None:
                before = (sim.resets, sim.steps)
                winner, info = select_active(
                    cands, fits, sim, val, keys, scales,
                    action_low=schema["action_min"], action_high=schema["action_max"],
                    num_probes=args.num_probes, probe_len=args.probe_len, seed=args.seed,
                )
                info["sim_resets"] = sim.resets - before[0]
                info["sim_steps"] = sim.steps - before[1]
                selections["active"] = (winner, info)
            for how, (winner, info) in selections.items():
                result["selections"][f"{how}_{objective}"] = {"winner": winner, **info}
                code = render_model(
                    cands[winner], fits[winner].params,
                    env=args.env, action_dim=schema["action_dim"], state_keys=keys, objective=objective,
                )
                path = out / "models" / f"{how}_{objective}.py"
                path.write_text(code, encoding="utf-8")
                print(f"[llm-family] {how}/{objective}: {winner} -> {path}", flush=True)
            for name, cand in cands.items():
                code = render_model(cand, fits[name].params, env=args.env, action_dim=schema["action_dim"], state_keys=keys, objective=objective)
                (out / "models" / f"cand_{name}_{objective}.py").write_text(code, encoding="utf-8")
    finally:
        if sim is not None:
            sim.close()
    result["elapsed_s"] = time.time() - t0
    (out / "family.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"[llm-family] wrote {out / 'family.json'} ({result['elapsed_s']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
