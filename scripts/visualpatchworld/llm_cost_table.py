#!/usr/bin/env python
"""LLM calls, tokens and induction wall-clock per method and environment, for one LLM."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts/visualpatchworld"
ENVS = ("tworoom", "reacher", "cube", "pusht")
METHODS = ("worldcoder", "gif_mcts", "poeworld", "pomdp_coder", "cwm_game", "tracepatch")


def baseline_row(root: Path, method: str, env: str):
    reports = sorted((root / method / env).glob("benchmark_*.json"))
    if not reports:
        return None
    usage = json.loads(reports[-1].read_text())["induction"]["llm_usage"]
    elapsed = None
    log = root / "logs" / f"driver_{method}.log"
    if not log.is_file():
        log = root.parent / f"{root.name}_driver_{method}.log"
    if log.is_file():
        text = log.read_text()
        s = re.search(rf"\[(\S+)\] START {method}/{env}\n", text)
        e = re.search(rf"\[(\S+)\] DONE {method}/{env} ", text)
        if s and e:
            elapsed = (datetime.fromisoformat(e.group(1)) - datetime.fromisoformat(s.group(1))).total_seconds()
    return {"calls": usage["calls"], "tokens": usage["total_tokens"], "elapsed_s": elapsed}


def ours_row(run: Path):
    if not (run / "family.json").is_file():
        return None
    log = json.loads((run / "llm_log.json").read_text())
    fam = json.loads((run / "family.json").read_text())
    tokens = sum((e.get("prompt_tokens") or 0) + (e.get("completion_tokens") or 0) for e in log)
    return {"calls": len(log), "tokens": tokens, "elapsed_s": fam.get("elapsed_s")}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--llm_tag", default="qwen35")
    p.add_argument("--baseline_root", default=str(ART / "code_baselines_qwen35"))
    p.add_argument("--out", default=None)
    args = p.parse_args()
    root = Path(args.baseline_root)
    table = {}
    for m in METHODS:
        table[m] = {e: baseline_row(root, m, e) for e in ENVS}
    fam = ART / "llm_family"
    table["ours"] = {e: ours_row(fam / f"v1_{e}_{args.llm_tag}_s0") for e in ENVS}
    table["ours_bank"] = {e: ours_row(fam / f"bank_general_{e}_{args.llm_tag}_s0") for e in ENVS}
    out = Path(args.out or ART / f"llm_family/cost_{args.llm_tag}.json")
    out.write_text(json.dumps(table, indent=2), encoding="utf-8")
    for m, rows in table.items():
        cells = []
        for e in ENVS:
            r = rows[e]
            cells.append("        -         " if r is None else
                         f"{r['calls']:3d}c {r['tokens']/1000:6.1f}k {('%5.0fs' % r['elapsed_s']) if r['elapsed_s'] else '   - '}")
        print(f"{m:12s} " + " | ".join(cells))


if __name__ == "__main__":
    main()
