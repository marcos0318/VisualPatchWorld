#!/usr/bin/env python
"""Compare live tool extractors against oracle scene graphs from HDF5 states."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
PATCHWORLD_ROOT = ROOT  # vendored package at repo root
for part in (str(PATCHWORLD_ROOT), str(SRC)):
    if part not in sys.path:
        sys.path.insert(0, part)
os.environ["PYTHONPATH"] = os.pathsep.join([str(PATCHWORLD_ROOT), str(SRC), os.environ.get("PYTHONPATH", "")])

try:
    import hdf5plugin  # noqa: F401
except ModuleNotFoundError:
    pass

from visualpatchworld.hdf5 import require_h5py, resolve_dataset_path
from visualpatchworld.lewm_env_specs import get_env_spec, make_gym_env
from visualpatchworld.lewm_graph import graph_from_row, reset_env_from_rows, row_dict_from_h5
from visualpatchworld.visual_tools import extractor_for_env


def _log(msg: str) -> None:
    print(f"[visualpatchworld][{datetime.now().isoformat(timespec='seconds')}][tool-perception] {msg}", flush=True)


def _position_from_object(obj: Mapping[str, Any]) -> Tuple[float, float, float]:
    pos = obj.get("position", obj)
    if not isinstance(pos, Mapping):
        return 0.0, 0.0, 0.0
    return float(pos.get("x", 0.0)), float(pos.get("y", 0.0)), float(pos.get("z", 0.0))


def _oracle_positions(row: Mapping[str, Any], env_name: str) -> Dict[str, Tuple[float, float, float]]:
    graph = graph_from_row(env_name, row, source="oracle", step=0)
    out: Dict[str, Tuple[float, float, float]] = {}
    for obj in graph.objects:
        z = float(obj.attributes.get("z", 0.0)) if obj.attributes else 0.0
        out[obj.object_id] = (float(obj.x), float(obj.y), z)
    return out


def _tool_positions(payload: Mapping[str, Any]) -> Dict[str, Tuple[float, float, float]]:
    out: Dict[str, Tuple[float, float, float]] = {}
    for obj in payload.get("objects", []):
        if not isinstance(obj, Mapping):
            continue
        obj_id = str(obj.get("id", ""))
        if not obj_id:
            continue
        out[obj_id] = _position_from_object(obj)
    return out


def _compare_positions(
    oracle: Mapping[str, Tuple[float, float, float]],
    tool: Mapping[str, Tuple[float, float, float]],
) -> Dict[str, Any]:
    ids = sorted(set(oracle) | set(tool))
    matched = [obj_id for obj_id in ids if obj_id in oracle and obj_id in tool]
    per_object: Dict[str, float] = {}
    errors_xy: List[float] = []
    errors_xyz: List[float] = []
    for obj_id in matched:
        ox, oy, oz = oracle[obj_id]
        tx, ty, tz = tool[obj_id]
        err_xy = math.hypot(tx - ox, ty - oy)
        err_xyz = math.sqrt((tx - ox) ** 2 + (ty - oy) ** 2 + (tz - oz) ** 2)
        per_object[obj_id] = err_xy
        errors_xy.append(err_xy)
        errors_xyz.append(err_xyz)
    return {
        "object_recall": len(matched) / max(1, len(oracle)),
        "object_precision": len(matched) / max(1, len(tool)),
        "position_rmse_xy": math.sqrt(sum(v * v for v in errors_xy) / max(1, len(errors_xy))),
        "position_rmse_xyz": math.sqrt(sum(v * v for v in errors_xyz) / max(1, len(errors_xyz))),
        "per_object_rmse_xy": per_object,
        "matched_objects": matched,
    }


def _fit_affine(img_xy: np.ndarray, world_xy: np.ndarray) -> Tuple[float, float, float]:
    """Least-squares fit world = a*img_x + b*img_y + c for one coordinate."""

    design = np.column_stack([img_xy[:, 0], img_xy[:, 1], np.ones(len(img_xy))])
    coef, _, _, _ = np.linalg.lstsq(design, world_xy, rcond=None)
    return float(coef[0]), float(coef[1]), float(coef[2])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env", required=True, choices=["tworoom", "cube", "reacher", "pusht"])
    p.add_argument("--num_samples", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default=None)
    p.add_argument("--fit_affine", action="store_true", help="Refit affine img->world calibration from samples.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    spec = get_env_spec(args.env)
    h5py = require_h5py()
    dataset_path = resolve_dataset_path(spec.dataset, None)
    rng = np.random.default_rng(args.seed)
    env = make_gym_env(spec)
    comparisons: List[Dict[str, Any]] = []
    img_points: Dict[str, List[Tuple[float, float]]] = {}
    world_points: Dict[str, List[Tuple[float, float]]] = {}

    try:
        with h5py.File(dataset_path, "r") as h5:
            if spec.name == "pusht":
                total = int(h5["state"].shape[0])
            elif spec.name == "tworoom":
                total = int(h5["observation"].shape[0])
            else:
                total = int(h5["qpos"].shape[0])
            indices = [int(i) for i in rng.choice(total, size=min(args.num_samples, total), replace=False)]

            from PIL import Image

            extractor = extractor_for_env(spec.name)
            for sample_idx, global_idx in enumerate(indices):
                row = row_dict_from_h5(h5, global_idx, spec)
                goal_row = row
                reset_env_from_rows(env, spec, row, goal_row)
                frame = np.asarray(env.render(), dtype=np.uint8)
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
                    path = Path(handle.name)
                try:
                    Image.fromarray(frame).save(path)
                    payload = extractor.extract(path, metadata={"global_idx": global_idx})
                except Exception:
                    path.unlink(missing_ok=True)
                    continue
                finally:
                    path.unlink(missing_ok=True)

                oracle_pos = _oracle_positions(row, spec.name)
                tool_pos = _tool_positions(payload)
                metrics = _compare_positions(oracle_pos, tool_pos)
                metrics["global_idx"] = global_idx
                comparisons.append(metrics)

                if args.fit_affine and spec.name == "cube":
                    arr = np.asarray(frame)
                    for obj_id in ("block", "effector"):
                        if obj_id not in oracle_pos or obj_id not in tool_pos:
                            continue
                        # Recover img centroid from inverse of current affine (approximate via payload metadata)
                        tx, ty, _ = tool_pos[obj_id]
                        img_points.setdefault(obj_id, []).append((tx, ty))
                        ox, oy, _ = oracle_pos[obj_id]
                        world_points.setdefault(obj_id, []).append((ox, oy))

                if sample_idx == 0 or (sample_idx + 1) % 50 == 0:
                    _log(f"sample {sample_idx + 1}/{len(indices)} rmse_xy={metrics['position_rmse_xy']:.4f}")
    finally:
        env.close()

    def _mean(key: str) -> float:
        vals = [float(item[key]) for item in comparisons if key in item]
        return sum(vals) / max(1, len(vals))

    per_object_agg: Dict[str, float] = {}
    for obj_id in sorted({k for item in comparisons for k in item.get("per_object_rmse_xy", {})}):
        vals = [float(item["per_object_rmse_xy"][obj_id]) for item in comparisons if obj_id in item.get("per_object_rmse_xy", {})]
        per_object_agg[obj_id] = math.sqrt(sum(v * v for v in vals) / max(1, len(vals)))

    report: Dict[str, Any] = {
        "env": args.env,
        "dataset": str(dataset_path),
        "num_samples": len(comparisons),
        "metrics": {
            "object_recall": _mean("object_recall"),
            "object_precision": _mean("object_precision"),
            "position_rmse_xy": _mean("position_rmse_xy"),
            "position_rmse_xyz": _mean("position_rmse_xyz"),
            "per_object_rmse_xy": per_object_agg,
        },
    }
    if args.fit_affine and spec.name == "cube":
        affine: Dict[str, Any] = {}
        for obj_id in sorted(world_points):
            if len(world_points[obj_id]) < 10:
                continue
            img = np.asarray(img_points[obj_id], dtype=np.float64)
            world = np.asarray(world_points[obj_id], dtype=np.float64)
            ax, bx, cx = _fit_affine(img, world[:, 0])
            ay, by, cy = _fit_affine(img, world[:, 1])
            affine[obj_id] = {"x_coef": [ax, bx, cx], "y_coef": [ay, by, cy], "n": len(world_points[obj_id])}
        report["affine_refit"] = affine

    out = Path(args.output) if args.output else ROOT / "artifacts" / "visualpatchworld" / "reports" / f"{args.env}_tool_perception_n{len(comparisons)}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    _log(f"Wrote {out}")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
