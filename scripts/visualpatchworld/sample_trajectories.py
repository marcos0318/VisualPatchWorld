#!/usr/bin/env python
"""Sample/export PushT trajectories for VisualPatchWorld.

For the VLM source this script only consumes pre-extracted VLM cache files.
Use `extract_vlm_scene_graphs.py` to produce those cache files from images.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, List


ROOT = Path(__file__).resolve().parents[2]


def _log(message: str) -> None:
    print(f"[visualpatchworld][{datetime.now().isoformat(timespec='seconds')}][sample] {message}", flush=True)


def _normalize_sources(values: Iterable[str]) -> List[str]:
    sources: List[str] = []
    for value in values:
        if value == "both":
            sources.extend(["oracle", "vlm"])
        else:
            sources.append(value)
    return list(dict.fromkeys(sources))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", nargs="+", default=["both"], choices=["oracle", "vlm", "both"])
    parser.add_argument("--dataset", default="pusht_expert_train.h5")
    parser.add_argument("--stablewm_home", default=os.environ.get("STABLEWM_HOME"))
    parser.add_argument("--output_dir", default=str(ROOT / "artifacts" / "visualpatchworld" / "data"))
    parser.add_argument("--max_episodes", type=int, default=0, help="<=0 exports all episodes.")
    parser.add_argument("--max_steps_per_episode", type=int, default=0, help="<=0 exports all steps.")
    parser.add_argument("--step_stride", type=int, default=5)
    parser.add_argument("--train_fraction", type=float, default=0.8)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--vlm_cache_dir", default=str(ROOT / "artifacts" / "visualpatchworld" / "vlm_cache"))
    parser.add_argument(
        "--vlm_fallback_oracle",
        action="store_true",
        help="Debug-only: fill missing VLM cache entries from privileged state.",
    )
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = os.environ.copy()
    pythonpath = [str(ROOT / "src")]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)

    for source in _normalize_sources(args.sources):
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "visualpatchworld" / "export_pusht.py"),
            "--source",
            source,
            "--dataset",
            args.dataset,
            "--output_dir",
            args.output_dir,
            "--max_episodes",
            str(args.max_episodes),
            "--max_steps_per_episode",
            str(args.max_steps_per_episode),
            "--step_stride",
            str(args.step_stride),
            "--train_fraction",
            str(args.train_fraction),
            "--val_fraction",
            str(args.val_fraction),
        ]
        if args.stablewm_home:
            cmd.extend(["--stablewm_home", args.stablewm_home])
        if source == "vlm":
            cmd.extend(
                [
                    "--vlm_provider",
                    "cache",
                    "--vlm_cache_dir",
                    args.vlm_cache_dir,
                ]
            )
            if args.vlm_fallback_oracle:
                cmd.append("--vlm_fallback_oracle")

        _log("Command: " + " ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, cwd=str(ROOT), env=env, check=True)


if __name__ == "__main__":
    main()
