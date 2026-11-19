#!/usr/bin/env python
"""Five-seed summaries for the revision experiments (floors, random-shortlist
control, proposal seeds, bank without the articulated-chain topic, tool traces)."""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts/visualpatchworld"
MODES = ART / "reports/fair_wm_ranking_gt90/vpw_modes"
FAM = ART / "llm_family"
SEEDS = (42, 43, 44, 45, 46)
ENVS = ("tworoom", "reacher", "cube", "pusht")


def _ms(vals):
    vals = [v for v in vals if v is not None]
    if len(vals) < len(SEEDS):
        return {"n": len(vals), "mean": None, "sd": None, "per_seed": vals}
    return {"n": len(vals), "mean": round(st.mean(vals), 1), "sd": round(st.stdev(vals), 1), "per_seed": vals}


def _mode(prefix, env):
    out = []
    for s in SEEDS:
        f = MODES / f"{prefix}_{env}_n50_seed{s}.json"
        out.append(json.loads(f.read_text()).get("success_rate") if f.is_file() else None)
    return _ms(out)


def _family(run, sel):
    out = []
    for s in SEEDS:
        f = FAM / run / "eval" / f"{sel}_seed{s}.json"
        if not f.is_file():
            aliases = FAM / run / "eval" / "aliases.json"
            if aliases.is_file():
                for name, sels in json.loads(aliases.read_text()).items():
                    if sel in sels and name.endswith(f"_seed{s}.json"):
                        f = FAM / run / "eval" / name
        out.append(next(iter(json.loads(f.read_text()).values())).get("success_rate") if f.is_file() else None)
    return _ms(out)


def _winner(run, sel="offline_multi_step"):
    p = FAM / run / "family.json"
    if not p.is_file():
        return None
    d = json.loads(p.read_text())
    w = d["selections"][sel]["winner"]
    obj = "multi_step" if "multi" in sel else "one_step"
    return {"winner": w, "params": d["fits"][obj][w]["params"], "val5": d["selections"][sel]["scores"][w]}


def main() -> None:
    rep = {
        "null_floor": {e: _mode("null_oracle_induced", e) for e in ENVS},
        "random_shortlist": {e: _mode("null_oracle_hybrid_random", e) for e in ENVS},
        "tooltrace_oracle_replan": {e: _family(f"tooltrace_{e}_qwen35_s0", "offline_multi_step") for e in ENVS[:3]},
        "tooltrace_tool_replan": {e: _mode("qwen35_tooltrace_tool_induced", e) for e in ENVS[:3]},
        "tooltrace_winner": {e: _winner(f"tooltrace_{e}_qwen35_s0") for e in ENVS[:3]},
        "bank_nochain": {},
        "proposal_seeds": {},
    }
    for llm in ("coder", "gptoss"):
        run = f"bank_nochain_reacher_{llm}_s0"
        rep["bank_nochain"][llm] = {sel: _family(run, sel) for sel in ("offline_multi_step", "offline_one_step")}
        rep["bank_nochain"][llm]["winner"] = _winner(run)
    for e in ENVS:
        rep["proposal_seeds"][e] = {}
        for s in (0, 1, 2):
            run = f"v1_{e}_qwen35_s{s}"
            rep["proposal_seeds"][e][f"s{s}"] = {
                sel: _family(run, sel) for sel in ("offline_multi_step", "offline_one_step")
            }
            w = _winner(run)
            rep["proposal_seeds"][e][f"s{s}"]["winner"] = w["winner"] if w else None
    out = ART / "reports/revision_summary.json"
    out.write_text(json.dumps(rep, indent=2, default=str))
    print(json.dumps(rep, indent=1, default=str))


if __name__ == "__main__":
    main()
