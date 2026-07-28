#!/usr/bin/env python
"""Export LeWM HDF5 trajectories as PatchWorld JSONL for any supported env."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from visualpatchworld.hdf5 import episode_slices, iter_split_slices, require_h5py, resolve_dataset_path
from visualpatchworld.lewm_env_specs import get_env_spec
from visualpatchworld.lewm_graph import (
    aggregate_action_window,
    graph_from_row,
    row_dict_from_h5,
)
from visualpatchworld.lewm_trajectory import make_env_record, make_env_transition, record_to_jsonl


def _log(message: str) -> None:
    print(f"[visualpatchworld][{datetime.now().isoformat(timespec='seconds')}][export] {message}", flush=True)


def _write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record_to_jsonl(record) + "\n")
            count += 1
    return count


def _read_scalar(dataset: Any, idx: int, default: float = 0.0) -> float:
    if dataset is None:
        return default
    try:
        return float(dataset[idx])
    except Exception:
        return default


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", required=True, choices=["pusht", "tworoom", "cube", "reacher"])
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--stablewm_home", default=os.environ.get("STABLEWM_HOME"))
    parser.add_argument("--output_dir", default=str(ROOT / "artifacts" / "visualpatchworld" / "data"))
    parser.add_argument("--source", choices=["oracle"], default="oracle")
    parser.add_argument("--max_episodes", type=int, default=128)
    parser.add_argument("--max_steps_per_episode", type=int, default=0)
    parser.add_argument("--step_stride", type=int, default=0, help="0 = env default frameskip")
    parser.add_argument("--train_fraction", type=float, default=0.8)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = get_env_spec(args.env)
    dataset = args.dataset or spec.dataset
    step_stride = args.step_stride or spec.frameskip
    dataset_path = resolve_dataset_path(dataset, args.stablewm_home)
    output_dir = Path(args.output_dir)
    _log(f"Export env={spec.name} dataset={dataset_path} stride={step_stride}")

    h5py = require_h5py()
    summaries: Dict[str, Any] = {
        "env": spec.name,
        "dataset": str(dataset_path),
        "source": args.source,
        "step_stride": step_stride,
        "splits": {},
    }
    split_records: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}

    with h5py.File(dataset_path, "r") as h5:
        if spec.name == "pusht" and "state" not in h5:
            raise SystemExit("PushT export requires `state` in HDF5.")
        if spec.name != "pusht" and "action" not in h5:
            raise SystemExit("Export requires `action` in HDF5.")

        length_key = "state" if spec.name == "pusht" else "observation" if "observation" in h5 else "action"
        fallback_len = len(h5[length_key])
        slices = episode_slices(h5, fallback_length=fallback_len)
        if args.max_episodes > 0:
            slices = slices[: args.max_episodes]

        rewards = h5.get("reward") or h5.get("rewards")
        dones = h5.get("done") or h5.get("dones") or h5.get("terminated") or h5.get("terminals")

        for split, episode in iter_split_slices(
            slices,
            train_fraction=args.train_fraction,
            val_fraction=args.val_fraction,
        ):
            transitions = []
            max_stop = episode.stop - step_stride
            raw_indices = range(episode.start, max_stop, step_stride)
            if args.max_steps_per_episode > 0:
                raw_indices = list(raw_indices)[: args.max_steps_per_episode]

            for local_step, idx in enumerate(raw_indices):
                next_idx = idx + step_stride
                start_row = row_dict_from_h5(h5, idx, spec)
                next_row = row_dict_from_h5(h5, next_idx, spec)
                meta = {"episode_index": episode.episode_index, "raw_index": int(idx), "next_raw_index": int(next_idx)}
                graph = graph_from_row(spec.name, start_row, source=args.source, step=local_step, metadata=meta)
                next_graph = graph_from_row(spec.name, next_row, source=args.source, step=local_step + 1, metadata=meta)
                action_vec = aggregate_action_window(h5, idx, step_stride)
                transition = make_env_transition(
                    graph,
                    [float(v) for v in action_vec.tolist()],
                    next_graph,
                    env_name=spec.name,
                    reward=_read_scalar(rewards, next_idx),
                    done=bool(_read_scalar(dones, next_idx, 0.0)),
                    metadata={**meta, "source": args.source},
                )
                transitions.append(transition)

            if transitions:
                split_records[split].append(
                    make_env_record(
                        transitions,
                        env_name=spec.name,
                        split=split,
                        episode_index=episode.episode_index,
                        source=args.source,
                        dataset=str(dataset_path),
                    )
                )

    for split, records in split_records.items():
        path = output_dir / f"{spec.name}_{args.source}_traj_{split}.jsonl"
        summaries["splits"][split] = {
            "path": str(path),
            "episodes": _write_jsonl(path, records),
            "transitions": sum(len(record["transitions"]) for record in records),
        }
        _log(f"Wrote {split}: {summaries['splits'][split]}")

    summary_path = output_dir / f"{spec.name}_{args.source}_summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summaries, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
