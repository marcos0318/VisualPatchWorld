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

from visualpatchworld.llm_family.bank import experience_entries, load_bank, render_bank  # noqa: E402
from visualpatchworld.llm_family.fit import FitResult, fit_candidate  # noqa: E402
from visualpatchworld.llm_family.propose import Candidate, propose_family  # noqa: E402
from visualpatchworld.llm_family.render import render_model  # noqa: E402
from visualpatchworld.llm_family.select import select_active, select_offline  # noqa: E402
from visualpatchworld.llm_family.traces import load_episodes, schema_summary, state_keys, state_scales  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env", required=True)
    p.add_argument("--data_dir", default=str(ROOT.parent / "le-wm-code/artifacts/visualpatchworld/data"))
    p.add_argument("--train_file", default=None, help="override <data_dir>/<env>_oracle_traj_train.jsonl")
    p.add_argument("--val_file", default=None, help="override <data_dir>/<env>_oracle_traj_val.jsonl")
    p.add_argument("--max_train_episodes", type=int, default=None)
    p.add_argument("--max_val_episodes", type=int, default=None)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--num_hypotheses", type=int, default=6)
    p.add_argument("--model", default=None, help="LLM id (default $PATCHWORLD_MODEL)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--prompt_version", default="scene_graph", choices=("schema", "scene_graph"))
    p.add_argument("--objectives", default="one_step,multi_step")
    p.add_argument("--rollout_k", type=int, default=5)
    p.add_argument("--restarts", type=int, default=4)
    p.add_argument("--max_nfev", type=int, default=400)
    p.add_argument("--num_probes", type=int, default=5)
    p.add_argument("--probe_len", type=int, default=3)
    p.add_argument("--no_active", action="store_true")
    p.add_argument("--bank_file", default=None, help="general mechanism bank JSON (from build_mechanism_bank.py)")
    p.add_argument("--experience_runs", nargs="*", default=[],
                   help="run dirs of OTHER environments whose selected programs join the bank")
    p.add_argument("--bank_no_snippets", action="store_true")
    p.add_argument("--analyze", action="store_true",
                   help="add metadata derived from the full training split (contact shapes, control period) to every transition")
    p.add_argument("--structured_fit", action="store_true", help="identification-style refit of every candidate")
    p.add_argument("--sysid", action="store_true",
                   help="identify a pusher/body contact program from the full training split and add it to the family")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    (out / "models").mkdir(parents=True, exist_ok=True)
    data = Path(args.data_dir)
    train = load_episodes(Path(args.train_file or data / f"{args.env}_oracle_traj_train.jsonl"), max_episodes=args.max_train_episodes)
    val = load_episodes(Path(args.val_file or data / f"{args.env}_oracle_traj_val.jsonl"), max_episodes=args.max_val_episodes)
    stride = train[0].transitions[0].stride
    keys = state_keys(train)
    scales = state_scales(train, keys)
    if args.analyze:
        from visualpatchworld.llm_family import sysid as _sysid

        derived = _sysid.analyze_interactions(load_episodes(Path(args.train_file or data / f"{args.env}_oracle_traj_train.jsonl")),
                                              log=lambda m: print(m, flush=True))
        for ep in list(train) + list(val):
            for t in ep.transitions:
                t.meta.update(derived)
        print(f"[llm-family] derived metadata: {sorted(derived)}", flush=True)
    schema = schema_summary(train)
    metas = [t.meta for ep in train for t in ep.transitions]
    static_meta = {k: v for k, v in (metas[0] if metas else {}).items() if all(m.get(k) == v for m in metas)}
    print(f"[llm-family] {args.env}: {schema['num_episodes']} train eps, {schema['num_transitions']} transitions, stride={stride}, keys={keys}", flush=True)

    bank = load_bank(Path(args.bank_file)) if args.bank_file else []
    experience = experience_entries([Path(d) for d in args.experience_runs])
    if any(e.source == f"experience:{args.env}" for e in experience):
        raise SystemExit("experience runs must come from other environments")
    bank_text = render_bank(bank + experience, with_snippet=not args.bank_no_snippets)
    print(f"[llm-family] bank: {len(bank)} general + {len(experience)} experience entries", flush=True)

    t0 = time.time()
    llm_log: list = []
    family = propose_family(
        train, num_hypotheses=args.num_hypotheses, model=args.model, seed=args.seed,
        prompt_version=args.prompt_version, bank_text=bank_text, log=llm_log,
    )
    (out / "llm_log.json").write_text(json.dumps(llm_log, indent=2), encoding="utf-8")
    print(f"[llm-family] {len(family)} valid candidates: {[c.name for c in family]}", flush=True)
    if not family:
        raise SystemExit("LLM produced no valid candidates")
    cands = {c.name: c for c in family}
    library_fits = {}
    if args.sysid:
        from visualpatchworld.llm_family import sysid

        full = load_episodes(Path(args.train_file or data / f"{args.env}_oracle_traj_train.jsonl"))
        ident = sysid.identify_pusher_body(full, val, keys, scales, log=lambda m: print(m, flush=True))
        if ident is not None:
            src, imeta = ident["candidate"], ident["meta"]
            lib = Candidate(name="sysid_pusher_body", mechanism=src.mechanism, code=src.code, params=dict(src.params),
                            step=lambda s, a, q, m, _f=src.step, _m=imeta: _f(s, a, q, {**_m, **m}))
            cands[lib.name] = lib
            family.append(lib)
            static_meta.update(imeta)
            b = ident["best"]
            library_fits[lib.name] = lambda objective: FitResult(
                params=b["params"], objective=objective, rollout_k=args.rollout_k, train_loss=float("nan"),
                val_one_step=b["val1"], val_rollout=b["val5"], restarts=1, nfev=0, elapsed_s=ident["elapsed_s"],
                history=[{"sysid_runs": [{k: v for k, v in r.items() if k != "params"} for r in ident["runs"]]}])

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

        sim = BlackBoxEnv(args.env, frames_per_step=stride)
    try:
        for objective in [o.strip() for o in args.objectives.split(",") if o.strip()]:
            fits = {}
            for name, cand in cands.items():
                if name in library_fits:
                    fits[name] = library_fits[name](objective)
                    print(f"[llm-family] {objective:10s} {name:40s} val1={fits[name].val_one_step:.4f} val{args.rollout_k}={fits[name].val_rollout:.4f} (identified)", flush=True)
                    continue
                fr = fit_candidate(
                    cand, train, val, keys, scales,
                    objective=objective, rollout_k=args.rollout_k,
                    restarts=args.restarts, max_nfev=args.max_nfev, seed=args.seed, structured=args.structured_fit,
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
                    static_meta=static_meta,
                )
                path = out / "models" / f"{how}_{objective}.py"
                path.write_text(code, encoding="utf-8")
                print(f"[llm-family] {how}/{objective}: {winner} -> {path}", flush=True)
            for name, cand in cands.items():
                code = render_model(cand, fits[name].params, env=args.env, action_dim=schema["action_dim"], state_keys=keys, objective=objective,
                                    static_meta=static_meta)
                (out / "models" / f"cand_{name}_{objective}.py").write_text(code, encoding="utf-8")
    finally:
        if sim is not None:
            sim.close()
    result["elapsed_s"] = time.time() - t0
    (out / "family.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"[llm-family] wrote {out / 'family.json'} ({result['elapsed_s']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
