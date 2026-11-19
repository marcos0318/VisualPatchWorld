#!/usr/bin/env python
"""Print the LeWM-suite table rows (mean +- SD over seeds 42-46) from run_vpw_modes outputs.

Oracle+Induced cells come from the main-table result files. Also writes a JSON summary.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paired_tests import ART, SEEDS, ours  # noqa: E402

MODES_DIR = ART / "reports/fair_wm_ranking_gt90/vpw_modes"
ALL_METHODS = ART / "reports/fair_wm_ranking_gt90/all_methods"
ENVS = ("tworoom", "reacher", "cube", "pusht")


def _load(path: Path):
    if not path.is_file():
        return None
    d = json.loads(path.read_text())
    return d.get("episode_successes") if d.get("success", True) else None


def cell(model_set: str, mode: str, env: str):
    rates = []
    for s in SEEDS:
        if mode == "oracle_induced":
            if model_set == "qwen35":
                eps = ours(f"v1_{env}_qwen35_s0", "offline_multi_step").get(s)
            else:
                eps = _load(ALL_METHODS / f"vpw_oracle_induced_{env}_n50_seed{s}.json")
        elif model_set == "mujoco":
            eps = _load(MODES_DIR / f"mujoco_{env}_n50_seed{s}.json")
        else:
            eps = _load(MODES_DIR / f"{model_set}_{mode}_{env}_n50_seed{s}.json")
        if eps is None:
            return None
        rates.append(100 * float(np.mean(eps)))
    return float(np.mean(rates)), float(np.std(rates, ddof=1))


def main() -> None:
    rows = [("qwen35", m) for m in ("oracle_induced", "tool_induced", "oracle_hybrid", "tool_hybrid")]
    rows += [("hand", m) for m in ("oracle_induced", "tool_induced", "oracle_hybrid", "tool_hybrid")]
    rows += [("mujoco", "simulator")]
    summary = {}
    for model_set, mode in rows:
        cells = [cell(model_set, mode, e) for e in ENVS]
        summary[f"{model_set}/{mode}"] = dict(zip(ENVS, cells))
        txt = [f"{m:.1f}\\,$\\pm$\\,{sd:.1f}" if c else "--" for c in cells for m, sd in [c or (0, 0)]]
        mean = f"{np.mean([c[0] for c in cells]):.1f}" if all(cells) else "--"
        print(f"{model_set:7s} {mode:15s} & " + " & ".join(txt) + f" & {mean} \\\\")
    (MODES_DIR / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
