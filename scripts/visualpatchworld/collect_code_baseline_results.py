#!/usr/bin/env python
"""Collect planning success rates from code baseline runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional


METHODS = ("tracepatch", "worldcoder", "poeworld", "gif_mcts", "pomdp_coder", "cwm_game")
ENVS = ("tworoom", "cube", "reacher", "pusht")
ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "artifacts/visualpatchworld/code_baselines"


def _latest_planning_report(method: str, env: str) -> Optional[Path]:
    env_dir = BASE / method / env
    if not env_dir.is_dir():
        return None
    candidates = sorted(env_dir.glob("planning_*_n50.json"), key=lambda p: p.stat().st_mtime)
    if candidates:
        return candidates[-1]
    candidates = sorted(env_dir.glob("planning_*.json"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def collect() -> Dict[str, Any]:
    table: Dict[str, Dict[str, Any]] = {}
    for method in METHODS:
        table[method] = {}
        for env in ENVS:
            report_path = _latest_planning_report(method, env)
            if report_path is None:
                table[method][env] = {"status": "missing"}
                continue
            data = json.loads(report_path.read_text(encoding="utf-8"))
            table[method][env] = {
                "status": "ok",
                "success_rate": data.get("success_rate"),
                "report": str(report_path),
            }
    return {"methods": list(METHODS), "envs": list(ENVS), "results": table}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--write", default=None, help="Optional path to write summary JSON")
    args = p.parse_args()
    summary = collect()
    text = json.dumps(summary, indent=2)
    print(text)
    if args.write:
        out = Path(args.write)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
