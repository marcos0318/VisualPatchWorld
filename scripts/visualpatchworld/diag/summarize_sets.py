#!/usr/bin/env python
"""Mean planning success per model set and per (tag, env) from run_vpw_modes reports."""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[3] / "artifacts/visualpatchworld/reports/fair_wm_ranking_gt90/vpw_modes"


def main() -> None:
    pattern = re.compile(sys.argv[1] if len(sys.argv) > 1 else ".*")
    mode = sys.argv[2] if len(sys.argv) > 2 else "oracle_induced"
    sets = defaultdict(list)
    for f in OUT.glob(f"*_{mode}_*_n50_seed*.json"):
        r = json.loads(f.read_text())
        if r.get("success") and pattern.search(r["model_set"]):
            sets[(r["model_set"], r["env"])].append(r["success_rate"])
    groups = defaultdict(list)
    for (s, env), v in sorted(sets.items()):
        print(f"{s:45s} {env:8s} n={len(v)} mean={np.mean(v):5.1f}  {[round(x, 1) for x in v]}")
        groups[(re.sub(r"_s\d+$", "", s), env)].append(np.mean(v))
    print()
    for (g, env), v in sorted(groups.items()):
        print(f"{g:45s} {env:8s} seeds={len(v)} mean={np.mean(v):5.1f} sd={np.std(v):4.1f}  {[round(x, 1) for x in v]}")


if __name__ == "__main__":
    main()
