#!/usr/bin/env python
"""Build a general mechanism bank with an LLM, without access to any evaluation environment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from visualpatchworld.llm_family.bank import (  # noqa: E402
    SOLVER_TOPICS,
    TOPICS,
    build_general_bank,
    build_solver_bank,
    load_bank,
    save_bank,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--drop_topics", default="", help="comma-separated 0-based indices into TOPICS to omit")
    p.add_argument("--solvers", action="store_true", help="write complete interaction solvers (SOLVER_TOPICS)")
    p.add_argument("--base", default="", help="existing bank to prepend (e.g. the general bank)")
    args = p.parse_args()
    log: list = []
    if args.solvers:
        entries = build_solver_bank(args.model, log=log)
    else:
        drop = {int(i) for i in args.drop_topics.split(",") if i.strip()}
        topics = [t for i, t in enumerate(TOPICS) if i not in drop]
        entries = build_general_bank(args.model, log=log, topics=topics)
    if args.base:
        entries = load_bank(Path(args.base)) + entries
    save_bank(entries, Path(args.out))
    Path(args.out).with_suffix(".log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(f"[bank] {len(entries)} entries -> {args.out}")
    for e in entries:
        print(f"  {e.topic[:28]:28s} {e.name}")


if __name__ == "__main__":
    main()
