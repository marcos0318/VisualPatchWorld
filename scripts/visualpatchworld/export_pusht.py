#!/usr/bin/env python
"""Export LeWM PushT trajectories as PatchWorld JSONL."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from visualpatchworld.hdf5 import episode_slices, iter_split_slices, require_h5py, resolve_dataset_path
from visualpatchworld.pusht import (
    aggregate_action,
    graph_from_vlm_payload,
    make_patchworld_record,
    make_transition,
    record_to_jsonl,
)
from visualpatchworld.schema import graph_from_pusht_state
from visualpatchworld.vlm import VLMConfig, VLMSceneGraphExtractor


def _write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record_to_jsonl(record) + "\n")
            count += 1
    return count


def _log(message: str) -> None:
    print(f"[visualpatchworld][{datetime.now().isoformat(timespec='seconds')}][export] {message}", flush=True)


def _image_bytes_from_pixels(pixels: Any) -> Optional[bytes]:
    if pixels is None:
        return None
    try:
        import numpy as np
        from PIL import Image
    except ModuleNotFoundError:
        return None

    arr = np.asarray(pixels)
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
        arr = arr.transpose(1, 2, 0)
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0:
            arr = arr * 255.0
        arr = arr.clip(0, 255).astype("uint8")
    from io import BytesIO

    buffer = BytesIO()
    Image.fromarray(arr).save(buffer, format="PNG")
    return buffer.getvalue()


def _read_scalar(dataset: Any, idx: int, default: float = 0.0) -> float:
    if dataset is None:
        return default
    try:
        return float(dataset[idx])
    except Exception:
        return default


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="pusht_expert_train.h5")
    parser.add_argument("--stablewm_home", default=os.environ.get("STABLEWM_HOME"))
    parser.add_argument("--output_dir", default=str(ROOT / "artifacts" / "visualpatchworld" / "data"))
    parser.add_argument("--source", choices=["oracle", "vlm"], required=True)
    parser.add_argument("--max_episodes", type=int, default=0, help="<=0 exports all episodes.")
    parser.add_argument("--max_steps_per_episode", type=int, default=0, help="<=0 exports all steps.")
    parser.add_argument("--step_stride", type=int, default=5, help="One symbolic transition spans this many raw rows.")
    parser.add_argument("--train_fraction", type=float, default=0.8)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--vlm_cache_dir", default=str(ROOT / "artifacts" / "visualpatchworld" / "vlm_cache"))
    parser.add_argument("--vlm_provider", choices=["cache", "openai"], default=os.environ.get("VLM_PROVIDER", "cache"))
    parser.add_argument("--vlm_model", default=os.environ.get("VLM_MODEL", "Qwen/Qwen3.5-397B-A17B"))
    parser.add_argument(
        "--vlm_base_url",
        default=os.environ.get("VLM_BASE_URL", "https://api.deepinfra.com/v1/openai"),
    )
    parser.add_argument("--vlm_api_key_env", default=os.environ.get("VLM_API_KEY_ENV", "DEEPINFRA_API_KEY"))
    parser.add_argument(
        "--vlm_fallback_oracle",
        action="store_true",
        help="Use oracle graph if a VLM cache entry is missing; useful for pipeline smoke tests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _log(f"Resolving dataset={args.dataset!r} source={args.source}")
    dataset_path = resolve_dataset_path(args.dataset, args.stablewm_home)
    output_dir = Path(args.output_dir)
    source = args.source
    _log(f"Using dataset path: {dataset_path}")

    h5py = require_h5py()
    summaries: Dict[str, Any] = {
        "dataset": str(dataset_path),
        "source": source,
        "step_stride": args.step_stride,
        "splits": {},
    }

    vlm_extractor: Optional[VLMSceneGraphExtractor] = None
    if source == "vlm":
        _log(
            "Configuring VLM extractor "
            f"provider={args.vlm_provider} model={args.vlm_model} cache={args.vlm_cache_dir}"
        )
        vlm_extractor = VLMSceneGraphExtractor(
            VLMConfig(
                cache_dir=Path(args.vlm_cache_dir),
                provider=args.vlm_provider,
                model=args.vlm_model,
                base_url=args.vlm_base_url,
                api_key_env=args.vlm_api_key_env,
            )
        )

    split_records: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}
    with h5py.File(dataset_path, "r") as h5:
        if "state" not in h5 or "action" not in h5:
            raise SystemExit("PushT export requires `state` and `action` datasets in the HDF5 file.")
        states = h5["state"]
        actions = h5["action"]
        rewards = h5.get("reward") or h5.get("rewards")
        dones = h5.get("done") or h5.get("dones") or h5.get("terminated") or h5.get("terminals")
        pixels = h5.get("pixels")
        slices = episode_slices(h5, fallback_length=len(states))
        if args.max_episodes > 0:
            slices = slices[: args.max_episodes]
        _log(
            f"Loaded {len(slices)} episode slices; "
            f"max_steps_per_episode={args.max_steps_per_episode or 'all'} step_stride={args.step_stride}"
        )

        processed_episodes = 0
        for split, episode in iter_split_slices(
            slices,
            train_fraction=args.train_fraction,
            val_fraction=args.val_fraction,
        ):
            processed_episodes += 1
            transitions = []
            max_stop = episode.stop - args.step_stride
            raw_indices = range(episode.start, max_stop, args.step_stride)
            if args.max_steps_per_episode > 0:
                raw_indices = list(raw_indices)[: args.max_steps_per_episode]
            raw_count = len(raw_indices)
            _log(
                f"Processing {source} split={split} episode={episode.episode_index} "
                f"raw_rows={episode.length} transitions={raw_count}"
            )
            for local_step, idx in enumerate(raw_indices):
                next_idx = idx + args.step_stride
                oracle_graph = graph_from_pusht_state(
                    states[idx],
                    source=source if source == "oracle" else "oracle-reference",
                    step=local_step,
                    metadata={"episode_index": episode.episode_index, "raw_index": int(idx)},
                )
                oracle_next_graph = graph_from_pusht_state(
                    states[next_idx],
                    source=source if source == "oracle" else "oracle-reference",
                    step=local_step + 1,
                    metadata={"episode_index": episode.episode_index, "raw_index": int(next_idx)},
                )

                if source == "oracle":
                    graph = oracle_graph
                    next_graph = oracle_next_graph
                else:
                    assert vlm_extractor is not None
                    try:
                        image_bytes = _image_bytes_from_pixels(pixels[idx]) if pixels is not None else None
                        next_image_bytes = (
                            _image_bytes_from_pixels(pixels[next_idx]) if pixels is not None else None
                        )
                        payload = vlm_extractor.load_or_extract(
                            episode_index=episode.episode_index,
                            step=local_step,
                            image_bytes=image_bytes,
                        )
                        next_payload = vlm_extractor.load_or_extract(
                            episode_index=episode.episode_index,
                            step=local_step + 1,
                            image_bytes=next_image_bytes,
                        )
                        graph = graph_from_vlm_payload(
                            payload,
                            step=local_step,
                            metadata={"episode_index": episode.episode_index, "raw_index": int(idx)},
                        )
                        next_graph = graph_from_vlm_payload(
                            next_payload,
                            step=local_step + 1,
                            metadata={"episode_index": episode.episode_index, "raw_index": int(next_idx)},
                        )
                    except Exception:
                        if not args.vlm_fallback_oracle:
                            raise
                        if local_step == 0:
                            _log(
                                f"VLM unavailable for episode={episode.episode_index}; "
                                "using oracle fallback for this episode"
                            )
                        graph = graph_from_pusht_state(
                            states[idx],
                            source="vlm-fallback-oracle",
                            step=local_step,
                            metadata={"episode_index": episode.episode_index, "raw_index": int(idx)},
                        )
                        next_graph = graph_from_pusht_state(
                            states[next_idx],
                            source="vlm-fallback-oracle",
                            step=local_step + 1,
                            metadata={"episode_index": episode.episode_index, "raw_index": int(next_idx)},
                        )

                action_values = aggregate_action(actions[idx:next_idx])
                transition = make_transition(
                    graph,
                    action_values,
                    next_graph,
                    reward=_read_scalar(rewards, next_idx),
                    done=bool(_read_scalar(dones, next_idx, 0.0)),
                    metadata={
                        "episode_index": episode.episode_index,
                        "raw_index": int(idx),
                        "next_raw_index": int(next_idx),
                        "source": source,
                    },
                )
                transitions.append(transition)

            if transitions:
                split_records[split].append(
                    make_patchworld_record(
                        transitions,
                        split=split,
                        episode_index=episode.episode_index,
                        source=source,
                        dataset=str(dataset_path),
                    )
                )
                _log(
                    f"Finished split={split} episode={episode.episode_index} "
                    f"transitions={len(transitions)} ({processed_episodes}/{len(slices)} episodes)"
                )

    for split, records in split_records.items():
        path = output_dir / f"pusht_{source}_traj_{split}.jsonl"
        _log(f"Writing {len(records)} {split} records to {path}")
        summaries["splits"][split] = {
            "path": str(path),
            "episodes": _write_jsonl(path, records),
            "transitions": sum(len(record["transitions"]) for record in records),
        }

    summary_path = output_dir / f"pusht_{source}_summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2, sort_keys=True), encoding="utf-8")
    _log(f"Wrote summary to {summary_path}")
    print(json.dumps(summaries, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
