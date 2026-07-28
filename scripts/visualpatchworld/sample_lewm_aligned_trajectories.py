#!/usr/bin/env python
"""Sample LeWM-aligned PushT trajectories for fair VisualPatchWorld baselines."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    import hdf5plugin  # noqa: F401
except ModuleNotFoundError:
    pass

from visualpatchworld.lewm_sampling import LeWMSamplingConfig, sample_lewm_aligned_dataset


def _log(message: str) -> None:
    print(f"[visualpatchworld][{datetime.now().isoformat(timespec='seconds')}][sample-lewm] {message}", flush=True)


def _normalize_sources(values: list[str]) -> list[str]:
    sources: list[str] = []
    for value in values:
        if value == "both":
            sources.extend(["image_action", "oracle"])
        else:
            sources.append(value)
    return list(dict.fromkeys(sources))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", nargs="+", default=["both"], choices=["image_action", "oracle", "both"])
    parser.add_argument("--dataset", default="pusht_expert_train.h5")
    parser.add_argument("--stablewm_home", default=os.environ.get("STABLEWM_HOME"))
    parser.add_argument(
        "--output_dir",
        default=str(ROOT / "artifacts" / "visualpatchworld" / "lewm_aligned"),
    )
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--num_steps", type=int, default=4)
    parser.add_argument("--train_fraction", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=3072)
    parser.add_argument("--max_clips", type=int, default=0, help="<=0 exports all LeWM clips.")
    parser.add_argument("--overwrite_images", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = LeWMSamplingConfig(
        frameskip=args.frameskip,
        num_steps=args.num_steps,
        train_fraction=args.train_fraction,
        seed=args.seed,
    )
    _log(
        "Sampling LeWM-aligned trajectories "
        f"frameskip={config.frameskip} num_steps={config.num_steps} span={config.span} "
        f"train_fraction={config.train_fraction} seed={config.seed}"
    )
    summary = sample_lewm_aligned_dataset(
        dataset=args.dataset,
        stablewm_home=args.stablewm_home,
        output_dir=Path(args.output_dir),
        sources=_normalize_sources(args.sources),
        config=config,
        max_clips=args.max_clips,
        overwrite_images=args.overwrite_images,
    )
    _log(f"Wrote LeWM-aligned summary to {Path(args.output_dir) / 'pusht_lewm_sampling_summary.json'}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
