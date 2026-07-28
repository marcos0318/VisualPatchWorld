#!/usr/bin/env python
"""Extract PushT scene graphs from sampled image frames into the VLM cache."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Tuple


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    import hdf5plugin  # noqa: F401
except ModuleNotFoundError:
    pass

from visualpatchworld.hdf5 import episode_slices, iter_split_slices, require_h5py, resolve_dataset_path
from visualpatchworld.paths import resolve_sample_jsonl, vlm_cache_key
from visualpatchworld.vlm import VLMConfig, VLMSceneGraphExtractor


def _log(message: str) -> None:
    print(f"[visualpatchworld][{datetime.now().isoformat(timespec='seconds')}][vlm-extract] {message}", flush=True)


def _image_bytes_from_pixels(pixels: Any) -> Optional[bytes]:
    if pixels is None:
        return None
    try:
        import numpy as np
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise RuntimeError("VLM extraction from HDF5 pixels requires numpy and Pillow") from exc

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


def _load_jsonl(path: Path) -> List[Mapping[str, Any]]:
    records: List[Mapping[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _iter_sampled_frames(sample_dir: Path, splits: Iterable[str]) -> Iterable[Tuple[str, Mapping[str, Any], Path]]:
    for split in splits:
        try:
            path = resolve_sample_jsonl(sample_dir, split)
        except FileNotFoundError:
            _log(f"Skipping missing split={split} under {sample_dir}")
            continue
        for record in _load_jsonl(path):
            record_meta = record.get("metadata", {})
            for transition in record.get("transitions", []):
                if not isinstance(transition, Mapping):
                    continue
                metadata = transition.get("metadata", {})
                if not isinstance(metadata, Mapping):
                    continue
                obs_meta = dict(metadata)
                if "clip_index" not in obs_meta and isinstance(record_meta, Mapping):
                    if "clip_index" in record_meta:
                        obs_meta["clip_index"] = record_meta["clip_index"]
                    if "lewm_aligned" in record_meta:
                        obs_meta["lewm_aligned"] = record_meta["lewm_aligned"]
                next_meta = dict(obs_meta)
                next_meta["step"] = int(metadata["next_step"])
                yield split, obs_meta, Path(str(transition["observation_image"]))
                yield split, next_meta, Path(str(transition["next_observation_image"]))


def _extract_from_samples(args: argparse.Namespace, extractor: VLMSceneGraphExtractor) -> Dict[str, Any]:
    sample_dir = Path(args.sample_dir)
    seen: Set[str] = set()
    summary: Dict[str, Any] = {
        "sample_dir": str(sample_dir),
        "cache_dir": str(args.cache_dir),
        "provider": args.provider,
        "model": args.model,
        "splits": {split: {"frames": 0} for split in ("train", "val", "test")},
        "written": 0,
        "skipped_existing": 0,
    }
    for split, frame_meta, image_path in _iter_sampled_frames(sample_dir, args.splits):
        cache_key = vlm_cache_key(frame_meta)
        if cache_key in seen:
            continue
        seen.add(cache_key)
        cache_path = extractor.cache_path_for_key(cache_key)
        summary["splits"][split]["frames"] += 1
        if cache_path.exists() and not args.overwrite:
            summary["skipped_existing"] += 1
            continue
        if args.dry_run:
            _log(f"Would extract image={image_path} -> {cache_path}")
            continue
        if not image_path.exists():
            raise FileNotFoundError(f"Sampled frame does not exist: {image_path}")
        if cache_path.exists() and args.overwrite:
            cache_path.unlink()
        try:
            extractor.load_or_extract(
                episode_index=int(frame_meta["episode_index"]),
                step=int(frame_meta["step"]),
                image_bytes=image_path.read_bytes(),
                metadata=frame_meta,
            )
            summary["written"] += 1
            if summary["written"] % 10 == 0:
                _log(
                    f"Progress: written={summary['written']} "
                    f"skipped={summary['skipped_existing']} cache_key={cache_key}"
                )
            if args.max_frames > 0 and summary["written"] >= args.max_frames:
                _log(f"Reached max_frames={args.max_frames}; stopping extraction")
                return summary
        except Exception as exc:
            summary.setdefault("errors", []).append(
                {"cache_key": cache_key, "image": str(image_path), "error": str(exc)}
            )
            _log(f"Failed cache_key={cache_key}: {exc}")
    if summary.get("errors"):
        _log(f"Completed with {len(summary['errors'])} extraction errors")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="pusht_expert_train.h5")
    parser.add_argument("--stablewm_home", default=os.environ.get("STABLEWM_HOME"))
    parser.add_argument(
        "--sample_dir",
        default=None,
        help="Directory from sample_image_action_trajectories.py. If set, extraction reads sampled PNGs instead of HDF5.",
    )
    parser.add_argument("--cache_dir", default=str(ROOT / "artifacts" / "visualpatchworld" / "vlm_cache"))
    parser.add_argument("--provider", choices=["openai"], default="openai")
    parser.add_argument("--model", default=os.environ.get("VLM_MODEL", "Qwen/Qwen3.5-397B-A17B"))
    parser.add_argument("--base_url", default=os.environ.get("VLM_BASE_URL", "https://api.deepinfra.com/v1/openai"))
    parser.add_argument("--api_key_env", default=os.environ.get("VLM_API_KEY_ENV", "DEEPINFRA_API_KEY"))
    parser.add_argument("--max_episodes", type=int, default=0, help="<=0 visits all episodes.")
    parser.add_argument("--max_steps_per_episode", type=int, default=0, help="<=0 visits all sampled transitions.")
    parser.add_argument("--step_stride", type=int, default=5)
    parser.add_argument("--train_fraction", type=float, default=0.8)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--splits", nargs="+", default=["train", "val"], choices=["train", "val", "test"])
    parser.add_argument(
        "--max_frames",
        type=int,
        default=0,
        help="Stop after writing this many new cache files (<=0 means no limit).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Regenerate cache files that already exist.")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_dir = Path(args.cache_dir)
    extractor = VLMSceneGraphExtractor(
        VLMConfig(
            cache_dir=cache_dir,
            provider=args.provider,
            model=args.model,
            base_url=args.base_url,
            api_key_env=args.api_key_env,
        )
    )
    if args.sample_dir:
        summary = _extract_from_samples(args, extractor)
        summary_path = cache_dir / "pusht_vlm_extraction_summary.json"
        if not args.dry_run:
            summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        _log(f"Extraction summary: written={summary['written']} skipped_existing={summary['skipped_existing']}")
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    dataset_path = resolve_dataset_path(args.dataset, args.stablewm_home)
    h5py = require_h5py()
    split_filter = set(args.splits)
    seen: Set[Tuple[int, int]] = set()
    summary: Dict[str, Any] = {
        "dataset": str(dataset_path),
        "cache_dir": str(cache_dir),
        "provider": args.provider,
        "model": args.model,
        "step_stride": args.step_stride,
        "splits": {split: {"frames": 0, "episodes": 0} for split in ("train", "val", "test")},
        "written": 0,
        "skipped_existing": 0,
    }

    _log(f"Reading image frames from {dataset_path}")
    with h5py.File(dataset_path, "r") as h5:
        if "pixels" not in h5:
            raise SystemExit("Fair VLM extraction requires a `pixels` dataset; no state fallback is used.")
        pixels = h5["pixels"]
        fallback_length = len(pixels)
        slices = episode_slices(h5, fallback_length=fallback_length)
        if args.max_episodes > 0:
            slices = slices[: args.max_episodes]

        for split, episode in iter_split_slices(
            slices,
            train_fraction=args.train_fraction,
            val_fraction=args.val_fraction,
        ):
            if split not in split_filter:
                continue
            summary["splits"][split]["episodes"] += 1
            max_stop = episode.stop - args.step_stride
            raw_indices = range(episode.start, max_stop, args.step_stride)
            if args.max_steps_per_episode > 0:
                raw_indices = list(raw_indices)[: args.max_steps_per_episode]
            frame_pairs: List[Tuple[int, int]] = []
            for local_step, idx in enumerate(raw_indices):
                frame_pairs.append((local_step, idx))
                frame_pairs.append((local_step + 1, idx + args.step_stride))

            _log(
                f"Extracting split={split} episode={episode.episode_index} "
                f"unique_frames={len(set(frame_pairs))}"
            )
            for local_step, raw_idx in frame_pairs:
                cache_key = (episode.episode_index, local_step)
                if cache_key in seen:
                    continue
                seen.add(cache_key)
                cache_path = extractor.cache_path(episode.episode_index, local_step)
                summary["splits"][split]["frames"] += 1
                if cache_path.exists() and not args.overwrite:
                    summary["skipped_existing"] += 1
                    continue
                if args.dry_run:
                    _log(f"Would extract frame raw_index={raw_idx} -> {cache_path}")
                    continue
                if cache_path.exists() and args.overwrite:
                    cache_path.unlink()
                image_bytes = _image_bytes_from_pixels(pixels[raw_idx])
                if image_bytes is None:
                    raise RuntimeError(f"Could not encode pixels for raw_index={raw_idx}")
                extractor.load_or_extract(
                    episode_index=episode.episode_index,
                    step=local_step,
                    image_bytes=image_bytes,
                )
                summary["written"] += 1
                if args.max_frames > 0 and summary["written"] >= args.max_frames:
                    _log(f"Reached max_frames={args.max_frames}; stopping extraction")
                    break
            if args.max_frames > 0 and summary["written"] >= args.max_frames:
                break

    summary_path = cache_dir / "pusht_vlm_extraction_summary.json"
    if not args.dry_run:
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _log(f"Extraction summary: written={summary['written']} skipped_existing={summary['skipped_existing']}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
