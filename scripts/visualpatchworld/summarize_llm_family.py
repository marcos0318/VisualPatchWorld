#!/usr/bin/env python
"""Collect LLM-family runs into one table: winner, held-out error, planning success."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FAMILY_ROOT = ROOT / "artifacts/visualpatchworld/llm_family"
SELECTIONS = ("offline_one_step", "offline_multi_step", "active_one_step", "active_multi_step")


def _success_by_selection(run_dir: Path):
    out = {}
    aliases = json.loads((run_dir / "eval" / "aliases.json").read_text()) if (run_dir / "eval" / "aliases.json").is_file() else {}
    for f in (run_dir / "eval").glob("*_seed*.json"):
        sels = aliases.get(f.name) or [f.name.rsplit("_seed", 1)[0]]
        for v in json.loads(f.read_text()).values():
            for sel in sels:
                out.setdefault(sel, {})[v["seed"]] = v["success_rate"]
    for f in (run_dir / "eval").glob("*.json"):
        if "_seed" in f.name or f.name == "aliases.json":
            continue
        for v in json.loads(f.read_text()).values():
            out.setdefault(f.stem, {})[v["seed"]] = v["success_rate"]
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=str(FAMILY_ROOT / "summary.json"))
    args = p.parse_args()
    rows = []
    for d in sorted(FAMILY_ROOT.iterdir()):
        fam = d / "family.json"
        if not fam.is_file():
            continue
        data = json.loads(fam.read_text())
        succ = _success_by_selection(d)
        prompt = data["args"].get("prompt_version", "schema")
        for sel in SELECTIONS:
            s = data["selections"].get(sel)
            if not s:
                continue
            objective = sel.split("_", 1)[1]
            fit = data["fits"][objective][s["winner"]]
            seeds = succ.get(sel, {})
            vals = list(seeds.values())
            rows.append({
                "run": d.name, "env": data["env"], "llm": data["llm_model"].split("/")[-1],
                "prompt": prompt, "selection": sel, "winner": s["winner"],
                "val_rollout": round(fit["val_rollout_nrmse"], 4),
                "best_val_rollout": round(min(v["val_rollout_nrmse"] for v in data["fits"][objective].values()), 4),
                "sim_resets": s.get("sim_resets"), "sim_steps": s.get("sim_steps"),
                "success_by_seed": seeds,
                "success_mean": round(float(np.mean(vals)), 2) if vals else None,
                "success_sd": round(float(np.std(vals, ddof=1)), 2) if len(vals) > 1 else None,
            })
    Path(args.out).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"{'run':24s} {'selection':19s} {'winner':32s} {'val':>6s} {'best':>6s} {'succ':>12s}")
    for r in rows:
        succ = "-" if r["success_mean"] is None else f"{r['success_mean']}" + (f"±{r['success_sd']}" if r["success_sd"] is not None else f" (n={len(r['success_by_seed'])})")
        print(f"{r['run']:24s} {r['selection']:19s} {r['winner'][:32]:32s} {r['val_rollout']:6.3f} {r['best_val_rollout']:6.3f} {succ:>12s}")


if __name__ == "__main__":
    main()
