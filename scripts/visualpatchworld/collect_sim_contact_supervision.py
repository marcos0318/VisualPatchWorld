#!/usr/bin/env python
"""Collect simulator ground-truth near-contact transitions for dynamics distillation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
PATCHWORLD_ROOT = ROOT  # vendored package at repo root
pythonpath_parts = [str(PATCHWORLD_ROOT), str(SRC)]
if os.environ.get("PYTHONPATH"):
    pythonpath_parts.append(os.environ["PYTHONPATH"])
os.environ["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(PATCHWORLD_ROOT) not in sys.path:
    sys.path.insert(0, str(PATCHWORLD_ROOT))

from visualpatchworld.sim_contact_supervision import (
    collect_sim_contact_supervision,
    summarize_sim_contact_supervision,
    write_sim_contact_jsonl,
)


def _log(message: str) -> None:
    print(
        f"[visualpatchworld][{datetime.now().isoformat(timespec='seconds')}][sim-contact] {message}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="pusht_expert_train.h5")
    parser.add_argument("--stablewm_home", default=os.environ.get("STABLEWM_HOME"))
    parser.add_argument(
        "--output",
        default=str(ROOT / "artifacts" / "visualpatchworld" / "lewm_tool_data" / "sim_contact_supervision.jsonl"),
    )
    parser.add_argument("--num_samples", type=int, default=800)
    parser.add_argument("--contact_threshold", type=float, default=70.0)
    parser.add_argument("--contact_bias", type=float, default=0.92)
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top_k", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = collect_sim_contact_supervision(
        dataset=args.dataset,
        stablewm_home=args.stablewm_home,
        frameskip=args.frameskip,
        num_samples=args.num_samples,
        contact_threshold=args.contact_threshold,
        contact_bias=args.contact_bias,
        seed=args.seed,
        top_k=args.top_k,
    )
    summary = summarize_sim_contact_supervision(rows)
    output = Path(args.output)
    write_sim_contact_jsonl(rows, output, item_id="sim-contact-supervision")
    _log(f"Wrote {len(rows)} sim contact transitions to {output}")
    print(json.dumps({"output": str(output), **summary}, indent=2))


if __name__ == "__main__":
    main()
