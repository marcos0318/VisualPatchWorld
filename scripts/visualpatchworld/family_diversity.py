#!/usr/bin/env python
"""Count behaviourally distinct mechanisms in each proposed family.

Two fitted candidates (multi-step fits) are treated as the same mechanism when their 5-step
held-out predictions differ, in the selection's normalized RMSE, by less than ``--rel`` times the
smaller of their two held-out errors, i.e. when the held-out data could not tell them apart.
Distinct mechanisms are the connected components of that relation.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from visualpatchworld.llm_family.propose import compile_candidate  # noqa: E402
from visualpatchworld.llm_family.select import _predict, _traj_dist  # noqa: E402
from visualpatchworld.llm_family.traces import load_episodes  # noqa: E402

FAMILY_ROOT = ROOT / "artifacts/visualpatchworld/llm_family"


def windows(episodes, k, stride):
    for ep in episodes:
        ts = ep.transitions
        for i in range(0, len(ts) - k + 1, stride):
            yield ts[i].state, [t.action for t in ts[i:i + k]], ts[i].meta, [ts[i].state] + [t.next_state for t in ts[i:i + k]]


def analyse(run: str, rel: float, stride: int):
    fam = json.loads((FAMILY_ROOT / run / "family.json").read_text())
    a = fam["args"]
    data = Path(a["data_dir"])
    val = load_episodes(Path(a["val_file"]) if a.get("val_file") else data / f"{fam['env']}_oracle_traj_val.jsonl",
                        max_episodes=a.get("max_val_episodes"))
    keys, scales = list(fam["schema"]["state_keys"]), fam["scales"]
    fits = fam["fits"]["multi_step"]
    cands = {}
    for c in fam["candidates"]:
        if c["name"] in fits and math.isfinite(fits[c["name"]]["val_rollout_nrmse"]):
            cand = compile_candidate(c["code"])
            cands[c["name"]] = (cand, fits[c["name"]]["params"])
    wins = list(windows(val, a["rollout_k"], stride))
    preds = {n: [_predict(c, p, s, acts, m) for s, acts, m, _ in wins] for n, (c, p) in cands.items()}
    names = list(cands)
    err = {n: fits[n]["val_rollout_nrmse"] for n in names}
    parent = {n: n for n in names}

    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x

    pair = {}
    for x, y in itertools.combinations(names, 2):
        ds = [_traj_dist(px, py, keys, scales) for px, py in zip(preds[x], preds[y]) if px and py]
        d = math.sqrt(sum(v * v for v in ds) / len(ds)) if ds else float("inf")
        pair[f"{x}|{y}"] = d
        if d < rel * min(err[x], err[y]):
            parent[find(x)] = find(y)
    groups = {}
    for n in names:
        groups.setdefault(find(n), []).append(n)
    return {"run": run, "env": fam["env"], "n_candidates": len(fam["candidates"]), "n_fitted": len(names),
            "n_distinct": len(groups), "groups": list(groups.values()), "val_err": err, "pair_dist": pair}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--rel", type=float, default=0.1)
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    rows = []
    for run in args.runs:
        r = analyse(run, args.rel, args.stride)
        rows.append(r)
        print(f"{run:34s} distinct={r['n_distinct']}/{r['n_fitted']}  groups={[len(g) for g in r['groups']]}")
    Path(args.out).write_text(json.dumps({"rel": args.rel, "rows": rows}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
