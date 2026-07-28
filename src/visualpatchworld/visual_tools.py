"""Low-level visual abstraction tools for VisualPatchWorld."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None  # type: ignore


PUSHT_WORLD_SIZE = 512.0
ROYAL_BLUE = (65.0, 105.0, 225.0)
LIGHT_SLATE_GRAY = (119.0, 136.0, 153.0)
LIGHT_GREEN = (144.0, 238.0, 144.0)
T_BLOCK_SCALE = 30.0
T_BLOCK_LENGTH = 4.0
T_BLOCK_LOCAL_CENTROID = (0.0, 40.714285714285715)


@dataclass(frozen=True)
class MaskComponent:
    """A connected component in image coordinates."""

    ys: Any
    xs: Any

    @property
    def area(self) -> int:
        return int(len(self.xs))

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return int(self.xs.min()), int(self.ys.min()), int(self.xs.max()), int(self.ys.max())

    @property
    def centroid(self) -> Tuple[float, float]:
        return float(self.xs.mean()), float(self.ys.mean())

    def touches_border(self, width: int, height: int, margin: int = 2) -> bool:
        x1, y1, x2, y2 = self.bbox
        return x1 <= margin or y1 <= margin or x2 >= width - margin - 1 or y2 >= height - margin - 1


def _load_rgb(path: Path) -> Any:
    try:
        import numpy as np
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise RuntimeError("Tool visual abstraction requires numpy and Pillow") from exc

    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)


def _color_distance(arr: Any, rgb: Tuple[float, float, float]) -> Any:
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise RuntimeError("Tool visual abstraction requires numpy") from exc

    target = np.asarray(rgb, dtype=np.float32)
    return np.linalg.norm(arr - target.reshape(1, 1, 3), axis=2)


def _connected_components(mask: Any) -> List[MaskComponent]:
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise RuntimeError("Tool visual abstraction requires numpy") from exc

    height, width = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    components: List[MaskComponent] = []
    ys0, xs0 = np.nonzero(mask)
    for start_y, start_x in zip(ys0.tolist(), xs0.tolist()):
        if seen[start_y, start_x]:
            continue
        stack = [(start_y, start_x)]
        seen[start_y, start_x] = True
        ys: List[int] = []
        xs: List[int] = []
        while stack:
            y, x = stack.pop()
            ys.append(y)
            xs.append(x)
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny = y + dy
                nx = x + dx
                if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        components.append(MaskComponent(ys=np.asarray(ys), xs=np.asarray(xs)))
    components.sort(key=lambda component: component.area, reverse=True)
    return components


def _select_component(mask: Any, *, min_area: int, reject_border: bool = True) -> MaskComponent:
    height, width = mask.shape
    for component in _connected_components(mask):
        if component.area < min_area:
            continue
        if reject_border and component.touches_border(width, height):
            continue
        return component
    raise ValueError(f"No connected component found with min_area={min_area}")


def _select_component_robust(mask: Any, *, min_area: int, reject_border: bool = True) -> MaskComponent:
    """Select the largest valid component, relaxing constraints on failure."""

    min_areas = [min_area]
    if min_area > 1:
        min_areas.append(max(1, min_area // 2))
    for area in min_areas:
        for border in (reject_border, False):
            try:
                return _select_component(mask, min_area=area, reject_border=border)
            except ValueError:
                continue
    raise ValueError(f"No connected component found with min_area={min_area}")


def _to_world_xy(x: float, y: float, *, width: int, height: int) -> Tuple[float, float]:
    return x * PUSHT_WORLD_SIZE / float(width), y * PUSHT_WORLD_SIZE / float(height)


def _normalize_angle(angle: float) -> float:
    value = float(angle) % (2.0 * math.pi)
    return value if value >= 0.0 else value + 2.0 * math.pi


def _pca_angle_world(component: MaskComponent, *, width: int, height: int) -> float:
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise RuntimeError("Tool visual abstraction requires numpy") from exc

    world_x = component.xs.astype(np.float64) * PUSHT_WORLD_SIZE / float(width)
    world_y = component.ys.astype(np.float64) * PUSHT_WORLD_SIZE / float(height)
    coords = np.column_stack([world_x - world_x.mean(), world_y - world_y.mean()])
    cov = coords.T @ coords / max(1, len(coords))
    values, vectors = np.linalg.eigh(cov)
    axis = vectors[:, int(np.argmax(values))]
    return math.atan2(float(axis[1]), float(axis[0]))


def _t_template_mask(
    *,
    width: int,
    height: int,
    origin_x: float,
    origin_y: float,
    angle: float,
) -> Any:
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise RuntimeError("Tool visual abstraction requires numpy") from exc

    ys, xs = np.indices((height, width), dtype=np.float64)
    world_x = xs * PUSHT_WORLD_SIZE / float(width)
    world_y = ys * PUSHT_WORLD_SIZE / float(height)
    dx = world_x - origin_x
    dy = world_y - origin_y
    c = math.cos(angle)
    s = math.sin(angle)
    local_x = c * dx + s * dy
    local_y = -s * dx + c * dy

    scale = T_BLOCK_SCALE
    length = T_BLOCK_LENGTH
    bar = (
        (local_x >= -length * scale / 2.0)
        & (local_x <= length * scale / 2.0)
        & (local_y >= 0.0)
        & (local_y <= scale)
    )
    stem = (
        (local_x >= -scale / 2.0)
        & (local_x <= scale / 2.0)
        & (local_y >= scale)
        & (local_y <= length * scale)
    )
    return bar | stem


def _fit_t_block_pose(component: MaskComponent, *, width: int, height: int) -> Tuple[float, float, float, float]:
    """Fit PushT's fixed T-block template and return x, y, angle, IoU."""

    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise RuntimeError("Tool visual abstraction requires numpy") from exc

    mask = np.zeros((height, width), dtype=bool)
    mask[component.ys, component.xs] = True
    centroid_x_img, centroid_y_img = component.centroid
    centroid_x, centroid_y = _to_world_xy(centroid_x_img, centroid_y_img, width=width, height=height)
    local_cx, local_cy = T_BLOCK_LOCAL_CENTROID

    pca_angle = _pca_angle_world(component, width=width, height=height)
    candidate_centers = [pca_angle + math.pi / 2.0, pca_angle - math.pi / 2.0]
    offsets = [math.radians(v) for v in (-20, -12, -6, 0, 6, 12, 20)]
    best: Optional[Tuple[float, float, float, float]] = None
    for center in candidate_centers:
        for base in (center, center + math.pi):
            for offset in offsets:
                angle = _normalize_angle(base + offset)
                c = math.cos(angle)
                s = math.sin(angle)
                origin_x = centroid_x - (c * local_cx - s * local_cy)
                origin_y = centroid_y - (s * local_cx + c * local_cy)
                template = _t_template_mask(
                    width=width,
                    height=height,
                    origin_x=origin_x,
                    origin_y=origin_y,
                    angle=angle,
                )
                intersection = int(np.logical_and(mask, template).sum())
                union = int(np.logical_or(mask, template).sum())
                iou = float(intersection / union) if union else 0.0
                if best is None or iou > best[3]:
                    best = (origin_x, origin_y, angle, iou)

    if best is None:
        raise ValueError("Could not fit T-block template")
    return best


class PushTHeuristicSceneGraphExtractor:
    """Extract PushT scene graphs with deterministic color/mask geometry."""

    provider = "pusht_heuristic"

    def __init__(self, *, block_distance_threshold: float = 90.0, agent_distance_threshold: float = 95.0):
        self.block_distance_threshold = float(block_distance_threshold)
        self.agent_distance_threshold = float(agent_distance_threshold)

    def extract(self, image_path: Path, *, metadata: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        arr = _load_rgb(image_path)
        height, width = int(arr.shape[0]), int(arr.shape[1])
        dist_agent = _color_distance(arr, ROYAL_BLUE)
        dist_block = _color_distance(arr, LIGHT_SLATE_GRAY)
        dist_goal = _color_distance(arr, LIGHT_GREEN)

        agent_mask = (dist_agent <= self.agent_distance_threshold) & (arr[:, :, 2] > arr[:, :, 0] + 45.0)
        goal_mask = dist_goal <= 85.0
        block_mask = (dist_block <= self.block_distance_threshold) & ~agent_mask & ~goal_mask

        agent = _select_component_robust(agent_mask, min_area=20, reject_border=True)
        block = _select_component_robust(block_mask, min_area=80, reject_border=True)

        agent_x_img, agent_y_img = agent.centroid
        agent_x, agent_y = _to_world_xy(agent_x_img, agent_y_img, width=width, height=height)
        block_x, block_y, block_angle, block_iou = _fit_t_block_pose(block, width=width, height=height)

        payload: Dict[str, Any] = {
            "objects": [
                {
                    "id": "agent",
                    "category": "circular_pusher",
                    "position": {"x": round(agent_x, 3), "y": round(agent_y, 3)},
                    "confidence": round(min(1.0, agent.area / 100.0), 3),
                },
                {
                    "id": "block",
                    "category": "t_block",
                    "position": {"x": round(block_x, 3), "y": round(block_y, 3)},
                    "angle": round(block_angle, 3),
                    "confidence": round(block_iou, 3),
                },
            ],
            "relations": [],
            "metadata": {
                "provider": self.provider,
                "image": str(image_path),
                "image_size": {"width": width, "height": height},
                "world_size": PUSHT_WORLD_SIZE,
                "agent_area": agent.area,
                "block_area": block.area,
                "block_template_iou": round(block_iou, 4),
            },
        }
        if metadata:
            payload["metadata"]["sample_metadata"] = dict(metadata)
        return payload


def _centroid_from_mask(mask: Any) -> Tuple[float, float]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError("Empty mask")
    return float(xs.mean()), float(ys.mean())


def _img_to_world_xy(x_img: float, y_img: float, *, width: int, height: int, world_size: float = 224.0) -> Tuple[float, float]:
    return x_img * world_size / float(width), y_img * world_size / float(height)


def _affine_img_to_xy(x_img: float, y_img: float, coef: Sequence[float]) -> Tuple[float, float]:
    a, b, c = coef
    return a * x_img + b * y_img + c, 0.0


class TwoRoomHeuristicSceneGraphExtractor:
    """Extract agent (red) and target (green) from Two-room renders."""

    provider = "tworoom_heuristic"

    def extract(self, image_path: Path, *, metadata: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        arr = _load_rgb(image_path)
        height, width = int(arr.shape[0]), int(arr.shape[1])
        agent_mask = (arr[:, :, 0] > 180) & (arr[:, :, 1] < 120) & (arr[:, :, 2] < 120)
        target_mask = (arr[:, :, 1] > 180) & (arr[:, :, 0] < 120) & (arr[:, :, 2] < 120)
        agent_x, agent_y = _centroid_from_mask(agent_mask)
        target_x, target_y = _centroid_from_mask(target_mask)
        agent_wx, agent_wy = _img_to_world_xy(agent_x, agent_y, width=width, height=height)
        target_wx, target_wy = _img_to_world_xy(target_x, target_y, width=width, height=height)
        return {
            "objects": [
                {"id": "agent", "category": "navigator", "position": {"x": round(agent_wx, 3), "y": round(agent_wy, 3)}},
                {"id": "target", "category": "goal", "position": {"x": round(target_wx, 3), "y": round(target_wy, 3)}},
            ],
            "metadata": {"provider": self.provider, "image": str(image_path)},
        }


# Calibrated on LeWM cube renders (224x224, n=499): img centroid -> normalized world xy.
_CUBE_BLOCK_COEF = (1.2862648927496306e-05, 0.001528247034721575, 0.22835724534411167)
_CUBE_BLOCK_COEF_Y = (0.003052360538958905, 3.835876187656406e-05, -0.359123808438115)
_CUBE_EFF_COEF = (9.093740352257716e-05, 0.0002992037244077426, 0.3878788815882162)
_CUBE_EFF_COEF_Y = (0.0023346959820842047, 0.000912836260674667, -0.3338841815386129)
_CUBE_EFF_Z_COEF = (-0.0021, 0.0034, 0.31)
_CUBE_BLOCK_Z = 0.02


class CubeHeuristicSceneGraphExtractor:
    """Extract effector and block positions from OGB Cube renders."""

    provider = "cube_heuristic"

    def extract(self, image_path: Path, *, metadata: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        arr = _load_rgb(image_path)
        height, width = int(arr.shape[0]), int(arr.shape[1])
        block_mask = (arr[:, :, 0] > 130) & (arr[:, :, 1] < 100) & (arr[:, :, 2] < 100)
        eff_mask = (arr[:, :, 0] > 80) & (arr[:, :, 0] < 220) & (arr[:, :, 2] > 80) & (arr[:, :, 1] < 130)
        block = _select_component_robust(block_mask, min_area=2, reject_border=False)
        effector = _select_component_robust(eff_mask, min_area=15, reject_border=False)
        bx, by = block.centroid
        ex, ey = effector.centroid

        def map_block(x: float, y: float) -> Tuple[float, float, float]:
            wx = _CUBE_BLOCK_COEF[0] * x + _CUBE_BLOCK_COEF[1] * y + _CUBE_BLOCK_COEF[2]
            wy = _CUBE_BLOCK_COEF_Y[0] * x + _CUBE_BLOCK_COEF_Y[1] * y + _CUBE_BLOCK_COEF_Y[2]
            return wx, wy, _CUBE_BLOCK_Z

        def map_eff(x: float, y: float) -> Tuple[float, float, float]:
            wx = _CUBE_EFF_COEF[0] * x + _CUBE_EFF_COEF[1] * y + _CUBE_EFF_COEF[2]
            wy = _CUBE_EFF_COEF_Y[0] * x + _CUBE_EFF_COEF_Y[1] * y + _CUBE_EFF_COEF_Y[2]
            wz = _CUBE_EFF_Z_COEF[0] * x + _CUBE_EFF_Z_COEF[1] * y + _CUBE_EFF_Z_COEF[2]
            return wx, wy, wz

        bwx, bwy, bwz = map_block(bx, by)
        ewx, ewy, ewz = map_eff(ex, ey)
        return {
            "objects": [
                {"id": "effector", "category": "gripper", "position": {"x": round(ewx, 4), "y": round(ewy, 4), "z": round(ewz, 4)}},
                {"id": "block", "category": "cube", "position": {"x": round(bwx, 4), "y": round(bwy, 4), "z": round(bwz, 4)}},
            ],
            "metadata": {"provider": self.provider, "image": str(image_path)},
        }


def _reacher_finger_from_img(x_img: float, y_img: float) -> Tuple[float, float]:
    fx = 0.004288861334446997 * x_img + 0.00010537191377629849 * y_img - 0.4925000356417166
    fy = -1.6525411858629526e-05 * x_img - 0.004135426071057984 * y_img + 0.45714778882808144
    return fx, fy


def _reacher_ik_candidates(fx: float, fy: float, *, l1: float = 0.12, l2: float = 0.12) -> List[Tuple[float, float]]:
    r2 = fx * fx + fy * fy
    r = math.sqrt(max(r2, 1e-8))
    r = min(r, l1 + l2 - 1e-5)
    cos_q2 = (r2 - l1 * l1 - l2 * l2) / (2.0 * l1 * l2)
    cos_q2 = max(-1.0, min(1.0, cos_q2))
    q2_abs = math.acos(cos_q2)
    out: List[Tuple[float, float]] = []
    for q2 in (q2_abs, -q2_abs):
        q1 = math.atan2(fy, fx) - math.atan2(l2 * math.sin(q2), l1 + l2 * math.cos(q2))
        out.append((q1, q2))
    return out


def _reacher_ik(fx: float, fy: float, *, l1: float = 0.12, l2: float = 0.12, qpos_hint: Optional[Sequence[float]] = None) -> Tuple[float, float]:
    candidates = _reacher_ik_candidates(fx, fy, l1=l1, l2=l2)
    if qpos_hint is None:
        return candidates[0]
    hint = np.asarray(qpos_hint, dtype=np.float64)
    best = min(candidates, key=lambda q: float(np.sum((np.asarray(q, dtype=np.float64) - hint) ** 2)))
    return best


class ReacherHeuristicSceneGraphExtractor:
    """Extract finger position from Reacher renders and infer joint angles."""

    provider = "reacher_heuristic"

    def extract(self, image_path: Path, *, metadata: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        arr = _load_rgb(image_path)
        finger_mask = (arr[:, :, 0] > 150) & (arr[:, :, 1] > 80) & (arr[:, :, 2] < 150)
        finger = _select_component_robust(finger_mask, min_area=20, reject_border=False)
        fx_img, fy_img = finger.centroid
        fx, fy = _reacher_finger_from_img(fx_img, fy_img)
        qpos_hint = None
        if metadata and isinstance(metadata.get("template_row"), Mapping):
            template = metadata["template_row"]
            if "qpos" in template:
                qpos_hint = np.asarray(template["qpos"], dtype=np.float64)
        q1, q2 = _reacher_ik(fx, fy, qpos_hint=qpos_hint)
        target = {"x": 0.0, "y": 0.0}
        if metadata and isinstance(metadata.get("goal_row"), Mapping):
            goal = metadata["goal_row"]
            if "target_pos" in goal:
                tp = np.asarray(goal["target_pos"], dtype=np.float64)
                target = {"x": float(tp[0]), "y": float(tp[1])}
        return {
            "objects": [
                {"id": "arm", "category": "joint_state", "qpos": [round(q1, 4), round(q2, 4)]},
                {"id": "finger", "category": "end_effector", "position": {"x": round(fx, 4), "y": round(fy, 4)}},
                {"id": "target", "category": "goal", "position": target},
            ],
            "metadata": {"provider": self.provider, "image": str(image_path)},
        }


def extractor_for_provider(provider: str):
    if provider == "pusht_heuristic":
        return PushTHeuristicSceneGraphExtractor()
    if provider == "tworoom_heuristic":
        return TwoRoomHeuristicSceneGraphExtractor()
    if provider == "cube_heuristic":
        return CubeHeuristicSceneGraphExtractor()
    if provider == "reacher_heuristic":
        return ReacherHeuristicSceneGraphExtractor()
    raise ValueError(
        f"Unsupported visual abstraction provider {provider!r}. "
        "Available: pusht_heuristic, tworoom_heuristic, cube_heuristic, reacher_heuristic"
    )


def extractor_for_env(env_name: str):
    from .lewm_env_specs import get_env_spec

    spec = get_env_spec(env_name)
    return extractor_for_provider(f"{spec.name}_heuristic")


def write_payload(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
