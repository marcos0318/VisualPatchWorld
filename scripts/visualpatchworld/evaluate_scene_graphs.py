#!/usr/bin/env python
"""Compare oracle and VLM PushT scene-graph JSONL exports."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


def _log(message: str) -> None:
    print(f"[visualpatchworld][{datetime.now().isoformat(timespec='seconds')}][compare] {message}", flush=True)


def _load_jsonl(path: Path) -> List[Mapping[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _objects(graph: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    return {str(obj.get("id")): obj for obj in graph.get("objects", []) if isinstance(obj, Mapping)}


def _position(obj: Mapping[str, Any]) -> tuple[float, float]:
    pos = obj.get("position", {})
    return float(pos.get("x", 0.0)), float(pos.get("y", 0.0))


def _graph_from_transition(transition: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    metadata = transition.get("metadata", {})
    if isinstance(metadata, Mapping) and key in metadata:
        return metadata[key]
    raise KeyError(f"Transition missing metadata.{key}")


def _compare_graphs(oracle: Mapping[str, Any], candidate: Mapping[str, Any]) -> Dict[str, float]:
    oracle_objects = _objects(oracle)
    candidate_objects = _objects(candidate)
    ids = sorted(set(oracle_objects) | set(candidate_objects))
    matched = [obj_id for obj_id in ids if obj_id in oracle_objects and obj_id in candidate_objects]
    position_errors = []
    angle_errors = []
    for obj_id in matched:
        ox, oy = _position(oracle_objects[obj_id])
        cx, cy = _position(candidate_objects[obj_id])
        position_errors.append(math.hypot(cx - ox, cy - oy))
        if "angle" in oracle_objects[obj_id] and "angle" in candidate_objects[obj_id]:
            angle_errors.append(abs(float(candidate_objects[obj_id]["angle"]) - float(oracle_objects[obj_id]["angle"])))
    return {
        "object_recall": len(matched) / max(1, len(oracle_objects)),
        "object_precision": len(matched) / max(1, len(candidate_objects)),
        "position_rmse": math.sqrt(sum(v * v for v in position_errors) / max(1, len(position_errors))),
        "angle_mae": sum(angle_errors) / max(1, len(angle_errors)) if angle_errors else 0.0,
    }


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / max(1, len(values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle_jsonl", required=True)
    parser.add_argument("--candidate_jsonl", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _log(f"Loading oracle JSONL: {args.oracle_jsonl}")
    oracle_records = _load_jsonl(Path(args.oracle_jsonl))
    _log(f"Loading candidate JSONL: {args.candidate_jsonl}")
    candidate_records = _load_jsonl(Path(args.candidate_jsonl))
    _log(f"Comparing {len(oracle_records)} oracle records with {len(candidate_records)} candidate records")
    comparisons = []
    for record_idx, (oracle_record, candidate_record) in enumerate(zip(oracle_records, candidate_records), start=1):
        if record_idx == 1 or record_idx == len(oracle_records) or record_idx % 10 == 0:
            _log(f"Comparing record {record_idx}/{len(oracle_records)}")
        for oracle_t, candidate_t in zip(oracle_record["transitions"], candidate_record["transitions"]):
            comparisons.append(
                _compare_graphs(
                    _graph_from_transition(oracle_t, "observation_graph"),
                    _graph_from_transition(candidate_t, "observation_graph"),
                )
            )
            comparisons.append(
                _compare_graphs(
                    _graph_from_transition(oracle_t, "next_observation_graph"),
                    _graph_from_transition(candidate_t, "next_observation_graph"),
                )
            )

    report = {
        "oracle_jsonl": args.oracle_jsonl,
        "candidate_jsonl": args.candidate_jsonl,
        "frames_compared": len(comparisons),
        "metrics": {
            key: _mean(item[key] for item in comparisons)
            for key in ("object_recall", "object_precision", "position_rmse", "angle_mae")
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    _log(f"Wrote comparison report to {output} with frames_compared={len(comparisons)}")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
