#!/usr/bin/env python
"""Check every cell of the paper's main table against the five-seed result files.

For each cell, recomputes mean and sample SD (ddof=1) over seeds 42-46 and checks the
printed values to within 0.05. A dagger cell must have no result file (program failed to
load). A bold cell must beat every same-LLM baseline under the paired test (exact McNemar,
Holm-corrected, p < 0.01); dagger baselines enter the test as failing every episode.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paired_tests import ART, ENVS, METHODS, SEEDS, baseline, compare, holm, ours  # noqa: E402

REPORTS = ART / "reports/fair_wm_ranking_gt90"
LLMS = {
    "Qwen3-Coder": ("all_methods", "coder"),
    "Qwen3.5": ("qwen35_baselines", "qwen35"),
    "gpt-oss-120b": ("gptoss_baselines", "gptoss"),
}
ROWS = {
    "WorldCoder": "worldcoder", "GIF-MCTS": "gif_mcts", "PoE-World": "poeworld",
    "POMDP-Coder": "pomdp_coder", "CWM": "cwm_game", "PatchWorld": "tracepatch",
    "Ours, one-step fit": ("v1", "offline_one_step"),
    "Ours, multi-step fit": ("v1", "offline_multi_step"),
    "+ mechanism bank": ("bank_general", "offline_multi_step"),
}
CELL = re.compile(r"^(\\textbf\{)?([\d.]+)\}?(?:\\,\$\\pm\$\\,([\d.]+))?(\$\^\\dagger\$)?$")


def series(llm: str, row: str, env: str):
    report_dir, tag = LLMS[llm]
    key = ROWS[row]
    if isinstance(key, str):
        return baseline(REPORTS / report_dir, key, env)
    prefix, selection = key
    return ours(f"{prefix}_{env}_{tag}_s0", selection)


def table_rows(tex: str):
    body = tex.split(r"\label{tab:main}")[0].rsplit(r"\begin{tabular}", 1)[1]
    llm = None
    for line in body.splitlines():
        m = re.match(r"\\multirow\{\d+\}\{\*\}\{(.+?)\}", line.strip())
        if m:
            llm = m.group(1)
            continue
        if line.strip().startswith("&") and llm:
            cells = [c.strip().rstrip("\\").strip() for c in line.split("&")[1:]]
            name = cells[0].replace(r"\quad", "").strip()
            yield llm, name, cells[1:]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tex", required=True)
    args = p.parse_args()
    tex = Path(args.tex).read_text(encoding="utf-8")
    rng = np.random.default_rng(0)
    problems, checked = [], 0
    for llm, name, cells in table_rows(tex):
        for env, cell in zip(ENVS, cells):
            m = CELL.match(cell)
            where = f"{llm} / {name} / {env}"
            if not m:
                problems.append(f"{where}: unparsable cell {cell!r}")
                continue
            bold, mean_s, sd_s, dagger = m.groups()
            data = series(llm, name, env)
            checked += 1
            if dagger:
                if data:
                    problems.append(f"{where}: dagger but {len(data)} result files exist")
                continue
            if sorted(data) != list(SEEDS):
                problems.append(f"{where}: seeds {sorted(data)}")
                continue
            rates = [100 * float(np.mean(data[s])) for s in SEEDS]
            mean, sd = float(np.mean(rates)), float(np.std(rates, ddof=1))
            if abs(mean - float(mean_s)) > 0.051 or sd_s is None or abs(sd - float(sd_s)) > 0.051:
                problems.append(f"{where}: printed {mean_s}+-{sd_s}, files give {mean:.2f}+-{sd:.2f}")
            if isinstance(ROWS[name], str):
                if bold:
                    problems.append(f"{where}: baseline cell is bold")
                continue
            tests = {}
            for method in METHODS:
                other = baseline(REPORTS / LLMS[llm][0], method, env)
                if not other:
                    other = {s: [0.0] * len(data[s]) for s in SEEDS}
                tests[method] = compare(data, other, rng)
            adj = holm([t["p"] for t in tests.values()])
            significant = all(t["diff"] > 0 and a < 0.01 for t, a in zip(tests.values(), adj))
            if bool(bold) != significant:
                problems.append(f"{where}: bold={bool(bold)} but paired test significant={significant}")
    print(f"checked {checked} cells")
    for line in problems:
        print("PROBLEM:", line)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
