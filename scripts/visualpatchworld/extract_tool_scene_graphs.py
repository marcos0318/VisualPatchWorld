#!/usr/bin/env python
"""Extract PushT scene graphs from sampled images with low-level vision tools."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Set, Tuple


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from visualpatchworld.paths import resolve_sample_jsonl, vlm_cache_key
from visualpatchworld.visual_tools import extractor_for_provider, write_payload


def _log(message: str) -> None:
    print(f"[visualpatchworld][{datetime.now().isoformat(timespec='seconds')}][tool-extract] {message}", flush=True)


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample_dir", default=str(ROOT / "artifacts" / "visualpatchworld" / "image_action_data"))
    parser.add_argument("--cache_dir", default=str(ROOT / "artifacts" / "visualpatchworld" / "tool_cache" / "pusht_heuristic"))
    parser.add_argument("--provider", default="pusht_heuristic", choices=["pusht_heuristic"])
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
    sample_dir = Path(args.sample_dir)
    cache_dir = Path(args.cache_dir)
    extractor = extractor_for_provider(args.provider)
    seen: Set[str] = set()
    summary: Dict[str, Any] = {
        "sample_dir": str(sample_dir),
        "cache_dir": str(cache_dir),
        "provider": args.provider,
        "splits": {split: {"frames": 0} for split in ("train", "val", "test")},
        "written": 0,
        "skipped_existing": 0,
        "errors": [],
    }

    for split, frame_meta, image_path in _iter_sampled_frames(sample_dir, args.splits):
        cache_key = vlm_cache_key(frame_meta)
        if cache_key in seen:
            continue
        seen.add(cache_key)
        cache_path = cache_dir / f"{cache_key}.json"
        summary["splits"][split]["frames"] += 1
        if cache_path.exists() and not args.overwrite:
            summary["skipped_existing"] += 1
            continue
        if args.dry_run:
            _log(f"Would extract image={image_path} -> {cache_path}")
            continue
        if not image_path.exists():
            raise FileNotFoundError(f"Sampled frame does not exist: {image_path}")
        try:
            payload = extractor.extract(image_path, metadata=frame_meta)
            write_payload(cache_path, payload)
            summary["written"] += 1
            if summary["written"] % 25 == 0:
                _log(f"Progress: written={summary['written']} skipped={summary['skipped_existing']}")
            if args.max_frames > 0 and summary["written"] >= args.max_frames:
                _log(f"Reached max_frames={args.max_frames}; stopping extraction")
                break
        except Exception as exc:
            error = {"cache_key": cache_key, "image": str(image_path), "error": str(exc)}
            summary["errors"].append(error)
            _log(f"Failed cache_key={cache_key}: {exc}")

    if not args.dry_run:
        cache_dir.mkdir(parents=True, exist_ok=True)
        summary_path = cache_dir / "pusht_tool_extraction_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        _log(f"Wrote extraction summary to {summary_path}")
    _log(
        f"Extraction summary: written={summary['written']} "
        f"skipped_existing={summary['skipped_existing']} errors={len(summary['errors'])}"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
