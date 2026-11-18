#!/usr/bin/env python
"""Paired comparisons of an LLM-family selection against same-LLM baselines.

Episodes are matched by (evaluation seed, start index); the planner's start states depend
only on the seed. Reports the success difference with a paired bootstrap 95% CI and an
exact McNemar p-value, Holm-corrected across baselines within each environment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts/visualpatchworld"
ENVS = ("tworoom", "reacher", "cube", "pusht")
METHODS = ("worldcoder", "gif_mcts", "poeworld", "pomdp_coder", "cwm_game", "tracepatch")
SEEDS = (42, 43, 44, 45, 46)


def ours(run: str, selection: str):
    d = ART / "llm_family" / run / "eval"
    out = {}
    if not (d / "aliases.json").is_file():
        return out
    aliases = json.loads((d / "aliases.json").read_text())
    for s in SEEDS:
        for fname, sels in aliases.items():
            if selection in sels and fname.endswith(f"seed{s}.json") and (d / fname).is_file():
                out[s] = next(iter(json.loads((d / fname).read_text()).values()))["episode_successes"]
    return out


def baseline(report_dir: Path, method: str, env: str):
    out = {}
    for s in SEEDS:
        f = report_dir / f"{method}_{env}_n50_seed{s}.json"
        if f.is_file():
            out[s] = json.loads(f.read_text())["episode_successes"]
    return out


def compare(a, b, rng):
    seeds = sorted(set(a) & set(b))
    x = np.concatenate([np.asarray(a[s], float) for s in seeds])
    y = np.concatenate([np.asarray(b[s], float) for s in seeds])
    diff = x - y
    boots = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(5000)]
    n01, n10 = int(((x == 1) & (y == 0)).sum()), int(((x == 0) & (y == 1)).sum())
    p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    return {"seeds": seeds, "n": len(diff), "diff": 100 * diff.mean(),
            "ci": (100 * np.percentile(boots, 2.5), 100 * np.percentile(boots, 97.5)), "p": p}


def holm(ps):
    order = np.argsort(ps)
    adj, running = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = running
    return adj


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run_tag", required=True, help="e.g. v1_{env}_qwen35_s0 or bank_general_{env}_coder_s0")
    p.add_argument("--selection", default="offline_multi_step")
    p.add_argument("--report_dir", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    rng = np.random.default_rng(0)
    results = {}
    for env in ENVS:
        mine = ours(args.run_tag.format(env=env), args.selection)
        rows = {}
        for m in METHODS:
            other = baseline(Path(args.report_dir), m, env)
            if mine and other and set(mine) & set(other):
                rows[m] = compare(mine, other, rng)
        if rows:
            for m, adj in zip(rows, holm([r["p"] for r in rows.values()])):
                rows[m]["p_holm"] = adj
        results[env] = rows
        for m, r in rows.items():
            print(f"{env:8s} vs {m:12s} n={r['n']:3d} seeds={len(r['seeds'])} diff={r['diff']:+6.1f} "
                  f"CI=[{r['ci'][0]:+6.1f},{r['ci'][1]:+6.1f}] p_holm={r['p_holm']:.2g}")
    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
