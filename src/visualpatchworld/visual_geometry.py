"""Perceived collision geometry from pixels (no shape prior, no source access).

This replaces source-derived shape constants (the T polygon / agent radius read
from the simulator code, which is cheating) with **boundary detection from
rendered frames**: colour segmentation -> largest contour -> canonical local
frame (area centroid + PCA orientation, asymmetry-disambiguated) -> aggregated
local occupancy -> simplified boundary polygon. The agent radius is the enclosing
circle of its mask. Only pixels and object colours are used; the object *shape*
is discovered, not assumed.

The output (a local-frame polygon + agent radius) feeds the general collision
primitive in ``collision.py``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

WORLD = 512.0
ROYAL_BLUE = (65.0, 105.0, 225.0)
LIGHT_SLATE_GRAY = (119.0, 136.0, 153.0)
LIGHT_GREEN = (144.0, 238.0, 144.0)

Vec = Tuple[float, float]


def _load_rgb(path) -> np.ndarray:
    from PIL import Image

    if isinstance(path, np.ndarray):
        arr = path
    else:
        arr = np.asarray(Image.open(path).convert("RGB"))
    return arr.astype(np.float32)


def _color_mask(arr: np.ndarray, rgb, thresh: float) -> np.ndarray:
    target = np.asarray(rgb, dtype=np.float32).reshape(1, 1, 3)
    dist = np.linalg.norm(arr - target, axis=2)
    return dist <= thresh


def _largest_contour(mask: np.ndarray):
    import cv2

    m = (mask.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def _block_pose_from_mask(mask: np.ndarray, width: int, height: int) -> Optional[Tuple[Vec, float, np.ndarray]]:
    """Return (centroid_world, canonical_angle, contour_world) from a block mask.

    canonical_angle orients the major (stem) axis to +y, disambiguated by which
    end is farther from the area centroid (the stem tip), so the local frame is
    consistent across poses without knowing the shape.
    """
    contour = _largest_contour(mask)
    if contour is None or len(contour) < 5:
        return None
    pts_img = contour.reshape(-1, 2).astype(np.float64)  # (x, y) in pixels
    sx, sy = WORLD / width, WORLD / height
    pts = np.column_stack([pts_img[:, 0] * sx, pts_img[:, 1] * sy])
    # area centroid from the filled mask (more stable than contour centroid)
    ys, xs = np.nonzero(mask)
    cx = float(xs.mean()) * sx
    cy = float(ys.mean()) * sy
    fill = np.column_stack([xs.astype(np.float64) * sx, ys.astype(np.float64) * sy])
    coords = fill - np.array([cx, cy])
    cov = coords.T @ coords / max(1, len(coords))
    vals, vecs = np.linalg.eigh(cov)
    axis = vecs[:, int(np.argmax(vals))]
    angle = math.atan2(float(axis[1]), float(axis[0]))
    # project fill onto major axis; the end farther from centroid is the stem tip
    proj = coords @ axis
    if abs(float(proj.max())) < abs(float(proj.min())):
        angle += math.pi  # flip so +axis points toward the far (stem) end
    # canonical: we want stem along +y, so rotate the world by (pi/2 - angle)
    canonical = angle - math.pi / 2.0
    return (cx, cy), canonical, pts


def _to_local(pts_world: np.ndarray, centroid: Vec, angle: float) -> np.ndarray:
    c, s = math.cos(-angle), math.sin(-angle)
    rel = pts_world - np.array(centroid)
    lx = c * rel[:, 0] - s * rel[:, 1]
    ly = s * rel[:, 0] + c * rel[:, 1]
    return np.column_stack([lx, ly])


def detect_pusht_geometry(
    image_paths: Sequence[Any],
    *,
    block_color=LIGHT_SLATE_GRAY,
    agent_color=ROYAL_BLUE,
    block_thresh: float = 90.0,
    agent_thresh: float = 95.0,
    grid_half: float = 130.0,
    grid_res: float = 2.0,
    vote_frac: float = 0.5,
) -> Dict[str, Any]:
    """Detect the block boundary polygon (local frame) and agent radius from frames."""
    import cv2

    n_bins = int(2 * grid_half / grid_res)
    accum = np.zeros((n_bins, n_bins), dtype=np.float64)
    n_block = 0
    agent_radii: List[float] = []
    for path in image_paths:
        arr = _load_rgb(path)
        h, w = arr.shape[:2]
        bmask = _color_mask(arr, block_color, block_thresh)
        amask = _color_mask(arr, agent_color, agent_thresh) & (arr[:, :, 2] > arr[:, :, 0] + 30.0)
        bmask = bmask & ~amask
        pose = _block_pose_from_mask(bmask, w, h)
        if pose is not None:
            centroid, angle, contour_world = pose
            # rasterize this frame's block fill into the canonical local grid
            ys, xs = np.nonzero(bmask)
            fill_world = np.column_stack([xs * (WORLD / w), ys * (WORLD / h)])
            local = _to_local(fill_world, centroid, angle)
            gi = ((local[:, 0] + grid_half) / grid_res).astype(int)
            gj = ((local[:, 1] + grid_half) / grid_res).astype(int)
            ok = (gi >= 0) & (gi < n_bins) & (gj >= 0) & (gj < n_bins)
            accum[gj[ok], gi[ok]] += 1.0
            n_block += 1
        ac = _largest_contour(amask)
        if ac is not None and len(ac) >= 5:
            (_, _), r = cv2.minEnclosingCircle(ac)
            agent_radii.append(float(r) * (WORLD / w))

    if n_block == 0:
        raise ValueError("No block detected in any frame")
    occ = (accum >= max(1.0, vote_frac * n_block)).astype(np.uint8) * 255
    contours, _ = cv2.findContours(occ, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.012 * peri, True).reshape(-1, 2).astype(np.float64)
    # grid index -> local world coords (centroid frame)
    poly = [((px * grid_res) - grid_half, (py * grid_res) - grid_half) for px, py in approx]
    agent_radius = float(np.median(agent_radii)) if agent_radii else 15.0
    return {
        "block_polygon_local": [[float(x), float(y)] for x, y in poly],
        "agent_radius": agent_radius,
        "reference": "area_centroid",
        "angle_convention": "pca_stem_+y",
        "n_frames": int(n_block),
    }


def detect_and_calibrate(
    frames: Sequence[Any],
    ref_poses: Sequence[Tuple[float, float, float]],
    **detect_kwargs: Any,
) -> Dict[str, Any]:
    """Perceive geometry and express it in the scene-graph block reference frame.

    The detector yields the boundary in the area-centroid frame. To use it with a
    scene graph whose block (x, y) is some fixed body reference, we calibrate the
    centroid's offset in that reference frame from (frame, pose) pairs -- using
    only perception and the (allowed) reported poses, never the simulator source.
    Returns geometry ready for the collision model:
      {"polygons": [[[x, y], ...]], "agent_radius": r, "cog": [ox, oy], ...}
    """
    geom = detect_pusht_geometry(frames, **detect_kwargs)
    offs = []
    block_thresh = detect_kwargs.get("block_thresh", 90.0)
    agent_thresh = detect_kwargs.get("agent_thresh", 95.0)
    for frame, (bx, by, ang) in zip(frames, ref_poses):
        arr = _load_rgb(frame)
        h, w = arr.shape[:2]
        bmask = _color_mask(arr, LIGHT_SLATE_GRAY, block_thresh)
        amask = _color_mask(arr, ROYAL_BLUE, agent_thresh) & (arr[:, :, 2] > arr[:, :, 0] + 30.0)
        pose = _block_pose_from_mask(bmask & ~amask, w, h)
        if pose is None:
            continue
        (cx, cy), _, _ = pose
        c, s = math.cos(-ang), math.sin(-ang)
        offs.append([c * (cx - bx) - s * (cy - by), s * (cx - bx) + c * (cy - by)])
    off = np.median(np.array(offs), axis=0) if offs else np.array([0.0, 0.0])
    poly_centroid = geom["block_polygon_local"]
    poly_body = [[float(x + off[0]), float(y + off[1])] for x, y in poly_centroid]
    return {
        "polygons": [poly_body],
        "agent_radius": geom["agent_radius"],
        "cog": [float(off[0]), float(off[1])],
        "centroid_offset_in_ref_frame": [float(off[0]), float(off[1])],
        "n_frames": geom["n_frames"],
        "source": "perceived_pixels",
    }


def save_geometry(geom: Dict[str, Any], path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(geom, indent=2, sort_keys=True), encoding="utf-8")


def load_geometry(path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
