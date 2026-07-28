#!/usr/bin/env python
"""Evaluate VLM scene graphs against oracle PushT scene graphs."""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _log(message: str) -> None:
    print(f"[visualpatchworld][{datetime.now().isoformat(timespec='seconds')}][perception] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate_source", default="vlm")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--data_dir", default=str(ROOT / "artifacts" / "visualpatchworld" / "data"))
    parser.add_argument("--report_dir", default=str(ROOT / "artifacts" / "visualpatchworld" / "reports"))
    parser.add_argument("--oracle_jsonl", default=None)
    parser.add_argument("--candidate_jsonl", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    oracle_jsonl = Path(args.oracle_jsonl) if args.oracle_jsonl else data_dir / f"pusht_oracle_traj_{args.split}.jsonl"
    candidate_jsonl = (
        Path(args.candidate_jsonl)
        if args.candidate_jsonl
        else data_dir / f"pusht_{args.candidate_source}_traj_{args.split}.jsonl"
    )
    output = (
        Path(args.output)
        if args.output
        else Path(args.report_dir) / f"pusht_{args.candidate_source}_vs_oracle_{args.split}.json"
    )

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "visualpatchworld" / "evaluate_scene_graphs.py"),
        "--oracle_jsonl",
        str(oracle_jsonl),
        "--candidate_jsonl",
        str(candidate_jsonl),
        "--output",
        str(output),
    ]
    _log("Command: " + " ".join(cmd))
    if not args.dry_run:
        subprocess.run(cmd, cwd=str(ROOT), check=True)


if __name__ == "__main__":
    main()
