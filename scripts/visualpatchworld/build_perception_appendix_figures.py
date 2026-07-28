#!/usr/bin/env python3
"""Build appendix figures for tool (pixel-fair) and VLM scene-graph extraction."""

from __future__ import annotations

import json
import math
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from appendix_figure_layout import (  # noqa: E402
    ACM_TEXTWIDTH_IN,
    chars_for_panel_inches,
    compact_scene_graph,
    draw_footer,
    draw_image_panel,
    draw_text_panel,
    draw_title_bar,
    outer_grid,
    save_figure,
    setup_matplotlib,
)

os.environ.setdefault("MUJOCO_GL", "egl")

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
PAPER_FIGURES = REPO_ROOT / "artifacts" / "visualpatchworld" / "figures"

sys.path.insert(0, str(SRC_ROOT))

from visualpatchworld.hdf5 import require_h5py, resolve_dataset_path  # noqa: E402
from visualpatchworld.lewm_env_specs import get_env_spec, make_gym_env  # noqa: E402
from visualpatchworld.lewm_graph import graph_from_row, graph_text_from_row, reset_env_from_rows, row_dict_from_h5  # noqa: E402
from visualpatchworld.pusht import graph_from_vlm_payload  # noqa: E402
from visualpatchworld.schema import render_graph_text  # noqa: E402
from visualpatchworld.tool_observation import graph_text_from_tool_payload  # noqa: E402
from visualpatchworld.visual_tools import PushTHeuristicSceneGraphExtractor, extractor_for_env  # noqa: E402

ENV_ORDER = ("pusht", "tworoom", "cube", "reacher")
ENV_LABELS = {
    "pusht": "PushT",
    "tworoom": "Two-room",
    "cube": "Cube",
    "reacher": "Reacher",
}
FRAME_INDICES = {"pusht": 120, "tworoom": 500, "cube": 800, "reacher": 300}
PANEL_WIDTH_IN = ACM_TEXTWIDTH_IN * 0.92 / 2.0
GRAPH_ATTR_WRAP = chars_for_panel_inches(PANEL_WIDTH_IN * 0.70, fontsize=5.4, mono=True)
FULL_GRAPH_ATTR_WRAP = chars_for_panel_inches(PANEL_WIDTH_IN, fontsize=5.4, mono=True)
FOOTER_WRAP = chars_for_panel_inches(PANEL_WIDTH_IN, fontsize=5.6, mono=False)

METRIC_REPORTS = {
    "pusht": REPO_ROOT / "artifacts/visualpatchworld/reports/pusht_tool_perception_n100.json",
    "tworoom": REPO_ROOT / "artifacts/visualpatchworld/reports/tworoom_tool_perception_n200.json",
    "cube": REPO_ROOT / "artifacts/visualpatchworld/reports/cube_tool_perception_n200_v2.json",
    "reacher": REPO_ROOT / "artifacts/visualpatchworld/reports/reacher_tool_perception_n200.json",
}

VLM_CACHE_DIR = REPO_ROOT / "artifacts/visualpatchworld/lewm_fair_vlm_cache"
PUSHT_VLM_CLIPS = (
    "pusht_clip0000031_obs00",
    "pusht_clip0000033_obs00",
    "pusht_clip0000078_obs00",
    "pusht_clip0000098_obs00",
)


def _position_rmse(oracle_row: Mapping[str, Any], payload: Mapping[str, Any], env: str) -> float:
    oracle_graph = graph_from_row(env, oracle_row, source="oracle", step=0)
    oracle = {obj.object_id: (obj.x, obj.y) for obj in oracle_graph.objects}
    tool = {}
    for obj in payload.get("objects", []):
        if not isinstance(obj, Mapping):
            continue
        pos = obj.get("position", obj)
        tool[str(obj.get("id", ""))] = (float(pos.get("x", 0.0)), float(pos.get("y", 0.0)))
    errs = []
    for obj_id, (ox, oy) in oracle.items():
        if obj_id not in tool:
            continue
        tx, ty = tool[obj_id]
        errs.append((tx - ox) ** 2 + (ty - oy) ** 2)
    return math.sqrt(sum(errs) / max(1, len(errs)))


def _load_report_metrics(env: str) -> str:
    path = METRIC_REPORTS.get(env)
    if path is None or not path.exists():
        return "Aggregate metrics unavailable."
    data = json.loads(path.read_text(encoding="utf-8"))
    metrics = data.get("metrics", {})
    rmse = metrics.get("position_rmse_xy", metrics.get("position_rmse"))
    recall = metrics.get("object_recall")
    precision = metrics.get("object_precision")
    n = data.get("num_samples", data.get("frames_compared"))
    parts = [f"n={n}"]
    if rmse is not None:
        parts.append(f"RMSE={float(rmse):.3f}")
    if recall is not None:
        parts.append(f"recall={float(recall):.2f}")
    if precision is not None:
        parts.append(f"precision={float(precision):.2f}")
    per_obj = metrics.get("per_object_rmse_xy", {})
    if per_obj:
        obj_bits = ", ".join(f"{k}={float(v):.3f}" for k, v in sorted(per_obj.items()))
        parts.append(f"per-object {obj_bits}")
    return "Aggregate (tool vs oracle): " + "; ".join(parts)


def _capture_tool_example(env: str, idx: int) -> Dict[str, Any]:
    spec = get_env_spec(env)
    h5py = require_h5py()
    dataset = resolve_dataset_path(spec.dataset)
    with h5py.File(dataset, "r") as h5:
        row = row_dict_from_h5(h5, idx, spec)
        goal_row = row_dict_from_h5(h5, min(idx + 25 * spec.frameskip, len(h5["action"]) - 1), spec)

    env_obj = make_gym_env(spec, render_mode="rgb_array")
    try:
        reset_env_from_rows(env_obj, spec, row, goal_row)
        frame = np.asarray(env_obj.render(), dtype=np.uint8)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
            path = Path(handle.name)
        try:
            Image.fromarray(frame).save(path)
            if env == "pusht":
                payload = PushTHeuristicSceneGraphExtractor().extract(path, metadata={"global_idx": idx})
                tool_text = render_graph_text(
                    graph_from_vlm_payload(payload, step=0, metadata={"global_idx": idx}, source="tool")
                )
            else:
                payload = extractor_for_env(spec.name).extract(path, metadata={"global_idx": idx})
                tool_text = graph_text_from_tool_payload(
                    env,
                    payload,
                    template_row=row,
                    goal_row=goal_row,
                    step=0,
                    metadata={"global_idx": idx},
                )
        finally:
            path.unlink(missing_ok=True)
    finally:
        env_obj.close()

    oracle_text = graph_text_from_row(env, row, source="oracle", step=0, metadata={"raw_index": idx})
    return {
        "title": ENV_LABELS[env],
        "frame": frame,
        "tool_text": tool_text,
        "oracle_text": oracle_text,
        "example_rmse": _position_rmse(row, payload, env),
        "aggregate_metrics": _load_report_metrics(env),
    }


def _raw_index_for_vlm_clip(stem: str) -> Optional[int]:
    tool_cache = REPO_ROOT / "artifacts/visualpatchworld/lewm_tool_cache/pusht_heuristic" / f"{stem}.json"
    if tool_cache.exists():
        meta = json.loads(tool_cache.read_text(encoding="utf-8")).get("metadata", {})
        sample_meta = meta.get("sample_metadata", {})
        if "raw_index" in sample_meta:
            return int(sample_meta["raw_index"])
    match = re.search(r"clip(\d+)", stem)
    if match:
        return int(match.group(1))
    return None


def _capture_vlm_example(stem: str) -> Dict[str, Any]:
    cache_path = VLM_CACHE_DIR / f"{stem}.json"
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    raw_index = _raw_index_for_vlm_clip(stem)
    if raw_index is None:
        raise RuntimeError(f"Could not resolve raw index for {stem}")

    spec = get_env_spec("pusht")
    h5py = require_h5py()
    dataset = resolve_dataset_path(spec.dataset)
    with h5py.File(dataset, "r") as h5:
        row = row_dict_from_h5(h5, raw_index, spec)
        goal_row = row_dict_from_h5(h5, min(raw_index + 25 * spec.frameskip, len(h5["action"]) - 1), spec)

    env_obj = make_gym_env(spec, render_mode="rgb_array")
    try:
        reset_env_from_rows(env_obj, spec, row, goal_row)
        frame = np.asarray(env_obj.render(), dtype=np.uint8)
    finally:
        env_obj.close()

    vlm_text = render_graph_text(
        graph_from_vlm_payload(payload, step=0, metadata={"raw_index": raw_index}, source="vlm")
    )
    oracle_text = graph_text_from_row("pusht", row, source="oracle", step=0, metadata={"raw_index": raw_index})
    rmse = _position_rmse(row, payload, "pusht")
    return {
        "title": f"PushT clip {stem.split('_')[-2][-4:]}",
        "frame": frame,
        "extracted_text": vlm_text,
        "oracle_text": oracle_text,
        "aggregate_metrics": f"Example RMSE={rmse:.1f}px vs oracle. VLM coordinates are not metrically calibrated.",
    }


def _draw_triple_panel(
    fig: plt.Figure,
    cell: plt.GridSpec,
    *,
    title: str,
    frame,
    left_label: str,
    left_text: str,
    right_label: str,
    right_text: str,
    footer: str,
) -> None:
    inner = cell.subgridspec(
        4,
        2,
        height_ratios=[0.07, 0.39, 0.43, 0.11],
        width_ratios=[0.30, 0.70],
        wspace=0.10,
        hspace=0.05,
    )
    ax_title = fig.add_subplot(inner[0, :])
    ax_img = fig.add_subplot(inner[1, 0])
    ax_left = fig.add_subplot(inner[1, 1])
    ax_right = fig.add_subplot(inner[2, :])
    ax_meta = fig.add_subplot(inner[3, :])

    draw_title_bar(ax_title, title, fontsize=8.0)
    draw_image_panel(ax_img, frame, square=True)
    draw_text_panel(
        ax_left,
        left_label,
        compact_scene_graph(left_text, attr_wrap=GRAPH_ATTR_WRAP, max_relations=3),
        body_size=5.4,
        max_lines=12,
        preserve_lines=True,
    )
    draw_text_panel(
        ax_right,
        right_label,
        compact_scene_graph(right_text, attr_wrap=FULL_GRAPH_ATTR_WRAP, max_relations=3),
        body_size=5.4,
        max_lines=12,
        preserve_lines=True,
        facecolor="#ffffff",
    )
    draw_footer(ax_meta, footer, fontsize=5.6, wrap_width=FOOTER_WRAP)


def build_tool_figure(output_path: Path = PAPER_FIGURES / "tool_scene_graph_examples.pdf") -> Path:
    setup_matplotlib()
    fig = plt.figure(figsize=(ACM_TEXTWIDTH_IN, 7.35), dpi=300)
    grid = outer_grid(fig, height=7.35)

    for idx, env in enumerate(ENV_ORDER):
        example = _capture_tool_example(env, FRAME_INDICES[env])
        row, col = divmod(idx, 2)
        _draw_triple_panel(
            fig,
            grid[row, col],
            title=f"{example['title']}  ·  example RMSE={example['example_rmse']:.2f}",
            frame=example["frame"],
            left_label="Tool",
            left_text=example["tool_text"],
            right_label="Oracle",
            right_text=example["oracle_text"],
            footer=(
                f"{example['aggregate_metrics']}\n"
                "Pipeline: color/mask heuristics, PCA orientation, temporal smoothing."
            ),
        )

    save_figure(fig, output_path)
    return output_path


def build_vlm_figure(output_path: Path = PAPER_FIGURES / "vlm_scene_graph_examples.pdf") -> Path:
    setup_matplotlib()
    fig = plt.figure(figsize=(ACM_TEXTWIDTH_IN, 7.35), dpi=300)
    grid = outer_grid(fig, height=7.35)

    for idx, stem in enumerate(PUSHT_VLM_CLIPS):
        example = _capture_vlm_example(stem)
        row, col = divmod(idx, 2)
        _draw_triple_panel(
            fig,
            grid[row, col],
            title=example["title"],
            frame=example["frame"],
            left_label="VLM",
            left_text=example["extracted_text"],
            right_label="Oracle",
            right_text=example["oracle_text"],
            footer=example["aggregate_metrics"],
        )

    save_figure(fig, output_path)
    return output_path


def main() -> None:
    tool_out = build_tool_figure()
    vlm_out = build_vlm_figure()
    print(f"Wrote {tool_out}")
    print(f"Wrote {tool_out.with_suffix('.png')}")
    print(f"Wrote {vlm_out}")
    print(f"Wrote {vlm_out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
