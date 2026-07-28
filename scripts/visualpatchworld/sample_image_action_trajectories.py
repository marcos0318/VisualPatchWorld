#!/usr/bin/env python
"""Sample PushT trajectories as image frames plus actions only."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    import hdf5plugin  # noqa: F401
except ModuleNotFoundError:
    pass

from visualpatchworld.hdf5 import episode_slices, iter_split_slices, require_h5py, resolve_dataset_path
from visualpatchworld.pusht import aggregate_action, discretize_action


def _log(message: str) -> None:
    print(f"[visualpatchworld][{datetime.now().isoformat(timespec='seconds')}][sample-raw] {message}", flush=True)


def _encode_png(pixels: Any, path: Path) -> None:
    try:
        import numpy as np
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise RuntimeError("Sampling image trajectories requires numpy and Pillow") from exc

    arr = np.asarray(pixels)
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
        arr = arr.transpose(1, 2, 0)
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0:
            arr = arr * 255.0
        arr = arr.clip(0, 255).astype("uint8")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def _read_scalar(dataset: Any, idx: int, default: float = 0.0) -> float:
    if dataset is None:
        return default
    try:
        return float(dataset[idx])
    except Exception:
        return default


def _write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
            count += 1
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="pusht_expert_train.h5")
    parser.add_argument("--stablewm_home", default=os.environ.get("STABLEWM_HOME"))
    parser.add_argument("--output_dir", default=str(ROOT / "artifacts" / "visualpatchworld" / "image_action_data"))
    parser.add_argument("--max_episodes", type=int, default=0, help="<=0 exports all episodes.")
    parser.add_argument("--max_steps_per_episode", type=int, default=0, help="<=0 exports all sampled transitions.")
    parser.add_argument("--step_stride", type=int, default=5)
    parser.add_argument("--train_fraction", type=float, default=0.8)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--overwrite_images", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = resolve_dataset_path(args.dataset, args.stablewm_home)
    output_dir = Path(args.output_dir)
    image_dir = output_dir / "images"
    h5py = require_h5py()
    split_records: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}
    summary: Dict[str, Any] = {
        "dataset": str(dataset_path),
        "source": "image_action",
        "step_stride": args.step_stride,
        "image_dir": str(image_dir),
        "splits": {},
    }

    _log(f"Sampling image/action trajectories from {dataset_path}")
    with h5py.File(dataset_path, "r") as h5:
        if "pixels" not in h5 or "action" not in h5:
            raise SystemExit("Image/action sampling requires `pixels` and `action` datasets.")
        pixels = h5["pixels"]
        actions = h5["action"]
        rewards = h5.get("reward") or h5.get("rewards")
        dones = h5.get("done") or h5.get("dones") or h5.get("terminated") or h5.get("terminals")
        slices = episode_slices(h5, fallback_length=len(pixels))
        if args.max_episodes > 0:
            slices = slices[: args.max_episodes]

        for split, episode in iter_split_slices(
            slices,
            train_fraction=args.train_fraction,
            val_fraction=args.val_fraction,
        ):
            transitions = []
            max_stop = episode.stop - args.step_stride
            raw_indices = range(episode.start, max_stop, args.step_stride)
            if args.max_steps_per_episode > 0:
                raw_indices = list(raw_indices)[: args.max_steps_per_episode]
            _log(
                f"Sampling split={split} episode={episode.episode_index} "
                f"transitions={len(raw_indices)}"
            )
            for local_step, idx in enumerate(raw_indices):
                next_idx = idx + args.step_stride
                obs_image = image_dir / f"pusht_ep{episode.episode_index:06d}_step{local_step:06d}.png"
                next_image = image_dir / f"pusht_ep{episode.episode_index:06d}_step{local_step + 1:06d}.png"
                if args.overwrite_images or not obs_image.exists():
                    _encode_png(pixels[idx], obs_image)
                if args.overwrite_images or not next_image.exists():
                    _encode_png(pixels[next_idx], next_image)

                action_values = aggregate_action(actions[idx:next_idx])
                transitions.append(
                    {
                        "observation_image": str(obs_image),
                        "action": discretize_action(action_values),
                        "action_values": [float(v) for v in action_values],
                        "next_observation_image": str(next_image),
                        "reward": _read_scalar(rewards, next_idx),
                        "done": bool(_read_scalar(dones, next_idx, 0.0)),
                        "metadata": {
                            "episode_index": episode.episode_index,
                            "step": local_step,
                            "raw_index": int(idx),
                            "next_step": local_step + 1,
                            "next_raw_index": int(next_idx),
                        },
                    }
                )
            if transitions:
                split_records[split].append(
                    {
                        "metadata": {
                            "env": "pusht",
                            "item_id": f"pusht-image-action-episode-{episode.episode_index}",
                            "task_idx": int(episode.episode_index),
                            "rollout_index": 0,
                            "split": split,
                            "source": "image_action",
                            "dataset": str(dataset_path),
                        },
                        "transitions": transitions,
                    }
                )

    for split, records in split_records.items():
        path = output_dir / f"pusht_image_action_traj_{split}.jsonl"
        summary["splits"][split] = {
            "path": str(path),
            "episodes": _write_jsonl(path, records),
            "transitions": sum(len(record["transitions"]) for record in records),
        }

    summary_path = output_dir / "pusht_image_action_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _log(f"Wrote image/action summary to {summary_path}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
