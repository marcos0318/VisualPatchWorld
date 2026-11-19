#!/usr/bin/env python
"""Residual-driven refinement of an induced LLM family.

Starts from the family's offline-selected program, then for ``--rounds`` rounds shows
the LLM the current best program, its fitted constants and its training-set residuals,
asks for revisions, fits each with the same least squares, and keeps the best by
held-out k-step nRMSE (the same selection rule as induction). No simulator access.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from visualpatchworld.llm_family.bank import load_bank, render_bank  # noqa: E402
from visualpatchworld.llm_family.fit import fit_candidate  # noqa: E402
from visualpatchworld.llm_family.propose import compile_candidate  # noqa: E402
from visualpatchworld.llm_family.refine import finite, propose_revisions, residual_report  # noqa: E402
from visualpatchworld.llm_family.render import render_model  # noqa: E402
from visualpatchworld.llm_family.traces import load_episodes, schema_summary, state_keys, state_scales  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run_dir", required=True, help="induced family directory (family.json)")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--model", default=None, help="LLM id (default: the family's)")
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--num_revisions", type=int, default=3)
    p.add_argument("--objective", default="multi_step")
    p.add_argument("--bank_file", default=None)
    p.add_argument("--feedback_episodes", type=int, default=10)
    p.add_argument("--restarts", type=int, default=2)
    p.add_argument("--max_nfev", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--structured_fit", action="store_true")
    args = p.parse_args()

    fam = json.loads((Path(args.run_dir) / "family.json").read_text())
    fa = fam["args"]
    model = args.model or fam["llm_model"]
    data = Path(fa["data_dir"])
    train = load_episodes(Path(fa["train_file"] or data / f"{fa['env']}_oracle_traj_train.jsonl"), max_episodes=fa["max_train_episodes"])
    val = load_episodes(Path(fa["val_file"] or data / f"{fa['env']}_oracle_traj_val.jsonl"), max_episodes=fa["max_val_episodes"])
    if fa.get("analyze"):
        from visualpatchworld.llm_family import sysid

        derived = sysid.analyze_interactions(load_episodes(Path(fa["train_file"] or data / f"{fa['env']}_oracle_traj_train.jsonl")), log=lambda m: None)
        for ep in list(train) + list(val):
            for t in ep.transitions:
                t.meta.update(derived)
    keys = state_keys(train)
    scales = state_scales(train, keys)
    schema = schema_summary(train)
    metas = [t.meta for ep in train for t in ep.transitions]
    static_meta = {k: v for k, v in (metas[0] if metas else {}).items() if all(m.get(k) == v for m in metas)}
    bank_text = render_bank(load_bank(Path(args.bank_file))) if args.bank_file else ""
    check_set = [t for ep in train[:4] for t in ep.transitions][:32]
    feedback = train[: args.feedback_episodes]
    k = int(fa.get("rollout_k", 5))

    sel = fam["selections"][f"offline_{args.objective}"]["winner"]
    start = next(c for c in fam["candidates"] if c["name"] == sel)
    best_cand = compile_candidate(start["code"])
    fr0 = fam["fits"][args.objective][sel]
    best = {"cand": best_cand, "params": fr0["params"], "val": fr0["val_rollout_nrmse"], "round": 0}
    out = Path(args.output_dir)
    (out / "models").mkdir(parents=True, exist_ok=True)
    log: list = []
    history = [{"round": 0, "name": sel, "val_rollout_nrmse": best["val"], "code": start["code"]}]
    print(f"[refine] start {sel} val{k}={best['val']:.4f} ({model})", flush=True)
    t0 = time.time()
    for rnd in range(1, args.rounds + 1):
        report, stats = residual_report(best["cand"], best["params"], feedback, keys, scales)
        revs = propose_revisions(best["cand"], best["params"], best["val"], report, static_meta, check_set,
                                 num_revisions=args.num_revisions, model=model, bank_text=bank_text, log=log)
        print(f"[refine] round {rnd}: {len(revs)} valid revisions", flush=True)
        for c in revs:
            fr = fit_candidate(c, train, val, keys, scales, objective=args.objective, rollout_k=k,
                               restarts=args.restarts, max_nfev=args.max_nfev, seed=args.seed,
                               structured=args.structured_fit)
            v = finite(fr.val_rollout)
            history.append({"round": rnd, "name": c.name, "mechanism": c.mechanism, "val_rollout_nrmse": v,
                            "val_one_step_nrmse": fr.val_one_step, "params": fr.params, "code": c.code,
                            "feedback_stats": stats})
            print(f"[refine]   {c.name:44s} val1={fr.val_one_step:.4f} val{k}={v:.4f} ({fr.elapsed_s:.0f}s)", flush=True)
            if v < best["val"]:
                best = {"cand": c, "params": fr.params, "val": v, "round": rnd}
        print(f"[refine] best after round {rnd}: {best['cand'].name} val{k}={best['val']:.4f}", flush=True)
        (out / "llm_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
        (out / "refine.json").write_text(json.dumps({"args": vars(args), "model": model, "start": sel, "history": history,
                                                     "best": {"name": best["cand"].name, "round": best["round"], "val_rollout_nrmse": best["val"],
                                                              "params": best["params"], "code": best["cand"].code},
                                                     "elapsed_s": time.time() - t0}, indent=2, default=str), encoding="utf-8")
        code = render_model(best["cand"], best["params"], env=fa["env"], action_dim=schema["action_dim"], state_keys=keys,
                            objective=args.objective, static_meta=static_meta)
        (out / "models" / f"offline_{args.objective}.py").write_text(code, encoding="utf-8")
    print(f"[refine] done: {best['cand'].name} val{k}={best['val']:.4f} round={best['round']} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
