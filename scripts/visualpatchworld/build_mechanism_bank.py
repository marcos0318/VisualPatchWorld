#!/usr/bin/env python
"""Build a general mechanism bank with an LLM, without access to any evaluation environment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from visualpatchworld.llm_family.bank import TOPICS, build_general_bank, save_bank  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--drop_topics", default="", help="comma-separated 0-based indices into TOPICS to omit")
    args = p.parse_args()
    drop = {int(i) for i in args.drop_topics.split(",") if i.strip()}
    topics = [t for i, t in enumerate(TOPICS) if i not in drop]
    log: list = []
    entries = build_general_bank(args.model, log=log, topics=topics)
    save_bank(entries, Path(args.out))
    Path(args.out).with_suffix(".log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(f"[bank] {len(entries)} entries -> {args.out}")
    for e in entries:
        print(f"  {e.topic[:28]:28s} {e.name}")


if __name__ == "__main__":
    main()
