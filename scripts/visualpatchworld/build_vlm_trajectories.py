#!/usr/bin/env python
"""Build PatchWorld visual trajectories from image/action samples and visual cache."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from visualpatchworld.paths import resolve_sample_jsonl
from visualpatchworld.pusht import aggregate_action, graph_from_vlm_payload, make_patchworld_record, make_transition, record_to_jsonl
from visualpatchworld.vlm import VLMConfig, VLMSceneGraphExtractor


def _log(message: str) -> None:
    print(f"[visualpatchworld][{datetime.now().isoformat(timespec='seconds')}][build-visual] {message}", flush=True)


def _load_jsonl(path: Path) -> List[Mapping[str, Any]]:
    records: List[Mapping[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record_to_jsonl(record) + "\n")
            count += 1
    return count


def _ensure_visual_payload(payload: Mapping[str, Any], *, episode_index: int, step: int) -> None:
    if "state" in payload:
        raise ValueError(
            "VLM cache payload contains privileged `state`; "
            f"refusing fair build for episode={episode_index} step={step}"
        )


def _aggregate_transition_action(transition: Mapping[str, Any]) -> List[float]:
    if "action_block" in transition:
        return [float(v) for v in aggregate_action(transition["action_block"])]
    values = transition.get("action_values", [])
    if isinstance(values, list) and len(values) > 2 and len(values) % 2 == 0:
        pairs = [[float(values[i]), float(values[i + 1])] for i in range(0, len(values), 2)]
        return [float(v) for v in aggregate_action(pairs)]
    return [float(v) for v in values]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample_dir", default=str(ROOT / "artifacts" / "visualpatchworld" / "image_action_data"))
    parser.add_argument("--cache_dir", default=str(ROOT / "artifacts" / "visualpatchworld" / "vlm_cache"))
    parser.add_argument("--output_dir", default=str(ROOT / "artifacts" / "visualpatchworld" / "data"))
    parser.add_argument(
        "--source_name",
        default="vlm",
        help="Scene-graph source label and output prefix, e.g. vlm or tool.",
    )
    parser.add_argument("--splits", nargs="+", default=["train", "val"], choices=["train", "val", "test"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample_dir = Path(args.sample_dir)
    output_dir = Path(args.output_dir)
    extractor = VLMSceneGraphExtractor(VLMConfig(cache_dir=Path(args.cache_dir), provider="cache"))
    summary: Dict[str, Any] = {
        "sample_dir": str(sample_dir),
        "cache_dir": str(args.cache_dir),
        "source": args.source_name,
        "splits": {},
    }

    for split in args.splits:
        try:
            sample_path = resolve_sample_jsonl(sample_dir, split)
        except FileNotFoundError:
            _log(f"Skipping missing split={split} under {sample_dir}")
            continue
        _log(f"Building split={split} from {sample_path}")
        patchworld_records: List[Dict[str, Any]] = []
        for sample_record in _load_jsonl(sample_path):
            sample_meta = sample_record.get("metadata", {})
            transitions = []
            episode_index = int(sample_meta.get("task_idx", sample_meta.get("episode_index", 0)))
            for transition in sample_record.get("transitions", []):
                if not isinstance(transition, Mapping):
                    continue
                metadata = transition.get("metadata", {})
                if not isinstance(metadata, Mapping):
                    raise ValueError(f"Transition missing metadata in {sample_path}")
                episode_index = int(metadata["episode_index"])
                step = int(metadata["step"])
                next_step = int(metadata["next_step"])
                obs_meta = dict(metadata)
                if "clip_index" not in obs_meta and isinstance(sample_meta, Mapping):
                    if "clip_index" in sample_meta:
                        obs_meta["clip_index"] = sample_meta["clip_index"]
                    if "lewm_aligned" in sample_meta:
                        obs_meta["lewm_aligned"] = sample_meta["lewm_aligned"]
                next_meta = dict(obs_meta)
                next_meta["step"] = next_step
                try:
                    payload = extractor.load_or_extract(
                        episode_index=episode_index, step=step, metadata=obs_meta
                    )
                    next_payload = extractor.load_or_extract(
                        episode_index=episode_index, step=next_step, metadata=next_meta
                    )
                except FileNotFoundError as exc:
                    _log(f"Skipping transition with missing VLM cache: {exc}")
                    continue
                _ensure_visual_payload(payload, episode_index=episode_index, step=step)
                _ensure_visual_payload(next_payload, episode_index=episode_index, step=next_step)
                graph = graph_from_vlm_payload(
                    payload,
                    step=step,
                    source=args.source_name,
                    metadata={
                        "episode_index": episode_index,
                        "raw_index": int(metadata["raw_index"]),
                        "observation_image": transition["observation_image"],
                    },
                )
                next_graph = graph_from_vlm_payload(
                    next_payload,
                    step=next_step,
                    source=args.source_name,
                    metadata={
                        "episode_index": episode_index,
                        "raw_index": int(metadata["next_raw_index"]),
                        "observation_image": transition["next_observation_image"],
                    },
                )
                action_values = _aggregate_transition_action(transition)
                transition_metadata = {
                    "episode_index": episode_index,
                    "raw_index": int(metadata["raw_index"]),
                    "next_raw_index": int(metadata["next_raw_index"]),
                    "step": step,
                    "next_step": next_step,
                    "source": args.source_name,
                    "observation_image": transition["observation_image"],
                    "next_observation_image": transition["next_observation_image"],
                }
                for key in ("action_block", "frameskip", "clip_index", "lewm_aligned", "global_start", "local_start", "num_steps", "span"):
                    if key in transition:
                        transition_metadata[key] = transition[key]
                    elif key in metadata:
                        transition_metadata[key] = metadata[key]
                transitions.append(
                    make_transition(
                        graph,
                        action_values,
                        next_graph,
                        reward=float(transition.get("reward", 0.0)),
                        done=bool(transition.get("done", False)),
                        metadata=transition_metadata,
                        action_text=str(transition.get("action") or ""),
                    )
                )
            if transitions:
                patchworld_records.append(
                    make_patchworld_record(
                        transitions,
                        split=split,
                        episode_index=episode_index,
                        source=args.source_name,
                        dataset=str(sample_meta.get("dataset", sample_dir)),
                    )
                )

        output_path = output_dir / f"pusht_{args.source_name}_traj_{split}.jsonl"
        episodes = _write_jsonl(output_path, patchworld_records)
        summary["splits"][split] = {
            "path": str(output_path),
            "episodes": episodes,
            "transitions": sum(len(record["transitions"]) for record in patchworld_records),
        }

    summary_path = output_dir / f"pusht_{args.source_name}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _log(f"Wrote visual trajectory summary to {summary_path}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
