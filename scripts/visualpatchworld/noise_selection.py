#!/usr/bin/env python
"""Refit an existing LLM-proposed family on noisy traces and check whether offline selection changes.

Gaussian noise with std sigma * (per-key one-step delta std) is added to every observed state,
consistently across consecutive transitions. No LLM calls are made.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from visualpatchworld.llm_family.fit import fit_candidate, nrmse  # noqa: E402
from visualpatchworld.llm_family.propose import compile_candidate  # noqa: E402
from visualpatchworld.llm_family.traces import load_episodes, state_keys, state_scales  # noqa: E402

FAMILY_ROOT = ROOT / "artifacts/visualpatchworld/llm_family"


def _noisy(episodes, keys, scales, sigma, rng):
    out = copy.deepcopy(episodes)
    for ep in out:
        ts = ep.transitions
        obs = [ts[0].state] + [t.next_state for t in ts]
        noisy = [{k: v + (rng.normal(0.0, sigma * scales[k]) if k in keys else 0.0) for k, v in s.items()} for s in obs]
        for i, t in enumerate(ts):
            t.state, t.next_state = noisy[i], dict(noisy[i + 1])
    return out


def _job(spec):
    run, sigma, noise_seed, objective = spec
    fam = json.loads((FAMILY_ROOT / run / "family.json").read_text())
    a = fam["args"]
    data = Path(a["data_dir"])
    train = load_episodes(Path(a["train_file"]) if a.get("train_file") else data / f"{fam['env']}_oracle_traj_train.jsonl")
    val = load_episodes(Path(a["val_file"]) if a.get("val_file") else data / f"{fam['env']}_oracle_traj_val.jsonl")
    keys = list(fam["schema"].get("state_keys") or state_keys(train))
    scales = fam.get("scales") or state_scales(train, keys)
    moving = [k for k in keys if np.std([t.next_state[k] - t.state[k] for ep in train for t in ep.transitions]) > 1e-6]
    rng = np.random.default_rng(1000 + noise_seed)
    ntrain, nval = _noisy(train, moving, scales, sigma, rng), _noisy(val, moving, scales, sigma, rng)
    rows = {}
    for c in fam["candidates"]:
        cand = compile_candidate(c["code"])
        cand.params = dict(c["init_params"])
        fit = fit_candidate(cand, ntrain, nval, keys, scales, objective=objective, rollout_k=a["rollout_k"],
                            restarts=a["restarts"], max_nfev=a["max_nfev"], seed=a["seed"])
        rows[c["name"]] = {
            "params": fit.params,
            "noisy_val_rollout": fit.val_rollout,
            "clean_val_rollout": nrmse(cand, fit.params, val, keys, scales, k=a["rollout_k"]),
        }
    winner = min(rows, key=lambda n: rows[n]["noisy_val_rollout"] if np.isfinite(rows[n]["noisy_val_rollout"]) else np.inf)
    return {
        "run": run, "sigma": sigma, "noise_seed": noise_seed, "objective": objective,
        "winner": winner, "clean_winner": fam["selections"][f"offline_{objective}"]["winner"],
        "fits": rows,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--sigmas", default="0.1,0.3,1.0")
    p.add_argument("--noise_seeds", default="0,1,2")
    p.add_argument("--objective", default="multi_step")
    p.add_argument("--workers", type=int, default=9)
    p.add_argument("--out", default=str(FAMILY_ROOT / "noise_selection.json"))
    args = p.parse_args()
    specs = [(r, float(s), int(n), args.objective) for r in args.runs
             for s in args.sigmas.split(",") for n in args.noise_seeds.split(",")]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(_job, specs))
    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    for r in results:
        w = r["fits"][r["winner"]]
        print(f"{r['run']:24s} sigma={r['sigma']:<4} seed={r['noise_seed']} winner={r['winner']:32s} "
              f"same={r['winner'] == r['clean_winner']} clean_val={w['clean_val_rollout']:.3f}")


if __name__ == "__main__":
    main()
