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
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from visualpatchworld.llm_family.bank import load_bank, render_bank  # noqa: E402
from visualpatchworld.llm_family.fit import FitTimeout, fit_candidate, nrmse, time_budget  # noqa: E402
from visualpatchworld.llm_family.modes import balanced_score, per_mode_errors  # noqa: E402
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
    p.add_argument("--balance_modes", action="store_true",
                   help="weight fitting residuals so each contact mode counts comparably")
    p.add_argument("--mode_feedback", action="store_true",
                   help="report per-contact-mode h-step error and the worst transition of each mode")
    p.add_argument("--select", choices=("rollout", "balanced"), default="rollout",
                   help="keep revisions by held-out k-step nRMSE, or by its mean over contact modes at --select_h")
    p.add_argument("--select_h", type=int, default=25)
    p.add_argument("--cover_modes", action="store_true",
                   help="choose the fitting/validation episodes from the full files so every contact mode is represented")
    p.add_argument("--val_episodes", type=int, default=0, help="with --cover_modes: number of validation episodes")
    p.add_argument("--fit_timeout", type=float, default=1800, help="seconds per revision (fit + score); 0 = none")
    p.add_argument("--constants_feedback", action="store_true",
                   help="flag fitted constants that collapsed, flipped sign or moved >10x from the program's initial guess")
    p.add_argument("--beam", type=int, default=1, help="keep the best N programs and revise them in turn")
    p.add_argument("--pusher_dynamics", action="store_true",
                   help="with an --analyze family: also identify the pusher's free-space dynamics into meta")
    p.add_argument("--state_limits", action="store_true",
                   help="identify hard limits of action-driven state variables into meta['state_limits'] "
                        "and tell the LLM to enforce them")
    p.add_argument("--pusher_helper", action="store_true",
                   help="with --pusher_dynamics: append a pusher_path() helper implementing the identified pusher "
                        "model to every revision and tell the LLM to use it")
    args = p.parse_args()

    fam = json.loads((Path(args.run_dir) / "family.json").read_text())
    fa = fam["args"]
    model = args.model or fam["llm_model"]
    data = Path(fa["data_dir"])
    train = load_episodes(Path(fa["train_file"] or data / f"{fa['env']}_oracle_traj_train.jsonl"), max_episodes=fa["max_train_episodes"])
    val = load_episodes(Path(fa["val_file"] or data / f"{fa['env']}_oracle_traj_val.jsonl"), max_episodes=fa["max_val_episodes"])
    if fa.get("analyze"):
        from visualpatchworld.llm_family import sysid

        derived = sysid.analyze_interactions(load_episodes(Path(fa["train_file"] or data / f"{fa['env']}_oracle_traj_train.jsonl")),
                                             log=lambda m: None, pusher_dynamics=args.pusher_dynamics)
        for ep in list(train) + list(val):
            for t in ep.transitions:
                t.meta.update(derived)
    limits = {}
    if args.state_limits:
        from visualpatchworld.llm_family import sysid

        limits = sysid.identify_state_limits(load_episodes(Path(fa["train_file"] or data / f"{fa['env']}_oracle_traj_train.jsonl")))
        print(f"[refine] state limits: {json.dumps(limits)}", flush=True)
        for ep in list(train) + list(val):
            for t in ep.transitions:
                t.meta["state_limits"] = limits
    keys = state_keys(train)
    scales = state_scales(train, keys)
    if args.cover_modes:
        from visualpatchworld.llm_family.modes import covering_episodes, label_episodes

        pool_tr = load_episodes(Path(fa["train_file"] or data / f"{fa['env']}_oracle_traj_train.jsonl"))
        pool_va = load_episodes(Path(fa["val_file"] or data / f"{fa['env']}_oracle_traj_val.jsonl"))
        if fa.get("analyze"):
            for ep in list(pool_tr) + list(pool_va):
                for t in ep.transitions:
                    t.meta.update(derived)
        if args.state_limits:
            for ep in list(pool_tr) + list(pool_va):
                for t in ep.transitions:
                    t.meta["state_limits"] = limits
        train = covering_episodes(pool_tr, keys, scales, len(train))
        val = covering_episodes(pool_va, keys, scales, args.val_episodes or len(val))
        from collections import Counter

        for name, eps in (("train", train), ("val", val)):
            print(f"[refine] mode-covering {name} ({len(eps)} episodes): "
                  f"{dict(Counter(label_episodes(eps, keys, scales).values()))}", flush=True)
    schema = schema_summary(train)
    metas = [t.meta for ep in train for t in ep.transitions]
    static_meta = {k: v for k, v in (metas[0] if metas else {}).items() if all(m.get(k) == v for m in metas)}
    bank_text = render_bank(load_bank(Path(args.bank_file))) if args.bank_file else ""
    helper_src = helper_doc = ""
    if args.pusher_helper:
        from visualpatchworld.llm_family import sysid as _sysid

        if "pusher_free_dynamics" not in static_meta:
            raise SystemExit("--pusher_helper needs --pusher_dynamics on an --analyze family")
        helper_src, helper_doc = _sysid.PUSHER_HELPER_SOURCE, _sysid.PUSHER_HELPER_DOC
    if limits:
        from visualpatchworld.llm_family import sysid as _sysid

        helper_doc += _sysid.STATE_LIMITS_DOC
    check_set = [t for ep in train[:4] for t in ep.transitions][:32]
    feedback = covering_episodes(train, keys, scales, args.feedback_episodes) if args.cover_modes else train[: args.feedback_episodes]
    k = int(fa.get("rollout_k", 5))

    sel = fam["selections"][f"offline_{args.objective}"]["winner"]
    start = next(c for c in fam["candidates"] if c["name"] == sel)
    best_cand = compile_candidate(start["code"])
    fr0 = fam["fits"][args.objective][sel]
    out = Path(args.output_dir)
    (out / "models").mkdir(parents=True, exist_ok=True)
    log: list = []
    t0 = time.time()
    tag = f"val{k}" if args.select == "rollout" else f"balanced@{args.select_h}"
    score_label = (f"held-out {k}-step normalized RMSE" if args.select == "rollout" else
                   f"held-out {args.select_h}-step normalized RMSE averaged over interaction modes")

    def score(c, params, fr_val):
        if args.select == "rollout":
            # Unweighted, on this run's validation episodes (fit scores may be mode-weighted).
            return finite(nrmse(c, params, val, keys, scales, k=k)), None
        pm = per_mode_errors(c, params, val, keys, scales, horizons=(1, 5, args.select_h))
        return finite(balanced_score(pm, args.select_h)), pm

    s0, pm0 = score(best_cand, fr0["params"], fr0["val_rollout_nrmse"])
    best = {"cand": best_cand, "params": fr0["params"], "val": s0, "round": 0}
    beam = [best]
    history = [{"round": 0, "name": sel, "val_rollout_nrmse": fr0["val_rollout_nrmse"], "score": s0, "val_per_mode": pm0,
                "code": start["code"]}]
    print(f"[refine] start {sel} {tag}={best['val']:.4f} ({model})", flush=True)
    for rnd in range(1, args.rounds + 1):
        parent = beam[(rnd - 1) % len(beam)]
        report, stats = residual_report(parent["cand"], parent["params"], feedback, keys, scales, per_mode=args.mode_feedback,
                                        horizons=(1, 5, args.select_h))
        revs = propose_revisions(parent["cand"], parent["params"], parent["val"], report, static_meta, check_set,
                                 num_revisions=args.num_revisions, model=model, bank_text=bank_text + helper_doc, log=log,
                                 score_label=score_label, constants_feedback=args.constants_feedback,
                                 code_suffix=helper_src)
        print(f"[refine] round {rnd}: {len(revs)} valid revisions of {parent['cand'].name}", flush=True)
        print(f"[refine] round {rnd}: {len(revs)} valid revisions", flush=True)
        for c in revs:
            try:
                with time_budget(args.fit_timeout):
                    fr = fit_candidate(c, train, val, keys, scales, objective=args.objective, rollout_k=k,
                                       restarts=args.restarts, max_nfev=args.max_nfev, seed=args.seed,
                                       structured=args.structured_fit, balance_modes=args.balance_modes)
                    v, pm = score(c, fr.params, fr.val_rollout)
            except FitTimeout as exc:
                history.append({"round": rnd, "name": c.name, "mechanism": c.mechanism, "timeout": str(exc), "code": c.code})
                print(f"[refine]   {c.name:44s} skipped: fit {exc}", flush=True)
                continue
            history.append({"round": rnd, "name": c.name, "mechanism": c.mechanism, "val_rollout_nrmse": finite(fr.val_rollout),
                            "score": v, "val_per_mode": pm, "val_one_step_nrmse": fr.val_one_step, "params": fr.params,
                            "code": c.code, "feedback_stats": stats})
            print(f"[refine]   {c.name:44s} val1={fr.val_one_step:.4f} val{k}={fr.val_rollout:.4f} {tag}={v:.4f} "
                  f"({fr.elapsed_s:.0f}s)", flush=True)
            if math.isfinite(v):
                beam = sorted(beam + [{"cand": c, "params": fr.params, "val": v, "round": rnd}], key=lambda e: e["val"])[: args.beam]
            best = beam[0]
        print(f"[refine] best after round {rnd}: {best['cand'].name} {tag}={best['val']:.4f}", flush=True)
        (out / "llm_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
        (out / "refine.json").write_text(json.dumps({"args": vars(args), "model": model, "start": sel, "history": history,
                                                     "best": {"name": best["cand"].name, "round": best["round"], "val_rollout_nrmse": best["val"],
                                                              "params": best["params"], "code": best["cand"].code},
                                                     "elapsed_s": time.time() - t0}, indent=2, default=str), encoding="utf-8")
        code = render_model(best["cand"], best["params"], env=fa["env"], action_dim=schema["action_dim"], state_keys=keys,
                            objective=args.objective, static_meta=static_meta)
        (out / "models" / f"offline_{args.objective}.py").write_text(code, encoding="utf-8")
    print(f"[refine] done: {best['cand'].name} {tag}={best['val']:.4f} round={best['round']} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
