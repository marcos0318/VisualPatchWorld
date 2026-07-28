"""Local, non-LLM PushT world-model induction baselines."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union


@dataclass(frozen=True)
class WeightedTransition:
  observation: str
  action: str
  next_observation: str
  weight: float = 1.0


_ACTION_RE = re.compile(
    r"^(?P<label>\w+)\[dx=(?P<dx>-?\d+(?:\.\d+)?),dy=(?P<dy>-?\d+(?:\.\d+)?),speed=(?P<speed>-?\d+(?:\.\d+)?)\]$"
)


def _parse_action(action: str) -> Tuple[float, float]:
    match = _ACTION_RE.match(str(action).strip())
    if not match:
        return 0.0, 0.0
    return float(match.group("dx")), float(match.group("dy"))


def _parse_observation_text(obs_text: str) -> Dict[str, Any]:
    graph: Dict[str, Any] = {
        "env": "pusht",
        "source": "",
        "step": 0,
        "goal": "move block to the target T pose",
        "objects": {},
        "relations": {},
        "metadata": {},
    }
    section = ""
    for raw_line in str(obs_text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("Environment:"):
            graph["env"] = line.split(":", 1)[1].strip()
            section = ""
        elif line.startswith("Source:"):
            graph["source"] = line.split(":", 1)[1].strip()
            section = ""
        elif line.startswith("Step:"):
            try:
                graph["step"] = int(float(line.split(":", 1)[1].strip()))
            except ValueError:
                graph["step"] = 0
            section = ""
        elif line.startswith("Goal:"):
            graph["goal"] = line.split(":", 1)[1].strip()
            section = ""
        elif line.startswith("Objects:"):
            section = "objects"
        elif line.startswith("Relations:"):
            section = "relations"
        elif line.startswith("Metadata:"):
            try:
                graph["metadata"] = json.loads(line.split(":", 1)[1].strip())
            except Exception:
                graph["metadata"] = {}
            section = ""
        elif section == "objects" and line.startswith("-"):
            obj = _parse_object_line(line)
            if obj:
                graph["objects"][obj["id"]] = obj
        elif section == "relations" and line.startswith("-"):
            rel = _parse_relation_line(line)
            if rel:
                graph["relations"][rel["key"]] = rel
    return graph


def _parse_object_line(line: str) -> Optional[Dict[str, Any]]:
    match = re.match(r"^-\s*(?P<id>\w+):(?P<category>\w+)\s*(?P<rest>.*)$", line)
    if not match:
        return None
    obj: Dict[str, Any] = {
        "id": match.group("id"),
        "category": match.group("category"),
        "x": 0.0,
        "y": 0.0,
        "vx": 0.0,
        "vy": 0.0,
        "conf": 1.0,
    }
    for key, value in re.findall(r"(\w+)=(-?\d+(?:\.\d+)?)", match.group("rest")):
        obj[key] = float(value)
    return obj


def _parse_relation_line(line: str) -> Optional[Dict[str, Any]]:
    match = re.match(r"^-\s*(\w+)\s+(\w+(?:_\w+)*)\s+(\w+)\s*=\s*(.+)$", line)
    if not match:
        return None
    value: Any = match.group(4).strip()
    try:
        value = float(value)
    except ValueError:
        pass
    return {
        "subject": match.group(1),
        "predicate": match.group(2),
        "object": match.group(3),
        "value": value,
        "key": f"{match.group(1)}_{match.group(2)}_{match.group(3)}",
    }


def _state_from_graph(graph: Mapping[str, Any]) -> Tuple[float, float, float, float, float]:
    objects = graph.get("objects", {})
    agent = objects.get("agent", {}) if isinstance(objects, Mapping) else {}
    block = objects.get("block", {}) if isinstance(objects, Mapping) else {}
    return (
        float(agent.get("x", 0.0)),
        float(agent.get("y", 0.0)),
        float(block.get("x", 0.0)),
        float(block.get("y", 0.0)),
        float(block.get("angle", 0.0)),
    )


def _fit_scale(numerators: List[float], denominators: List[float], default: float) -> float:
    ratios = [n / d for n, d in zip(numerators, denominators) if abs(d) > 1e-6 and math.isfinite(n) and math.isfinite(d)]
    if not ratios:
        return default
    ratios = [value for value in ratios if abs(value) < 1000.0]
    return float(median(ratios)) if ratios else default


def fit_pusht_kinematic_params(trajectories: Sequence[Any]) -> Dict[str, float]:
    """Estimate simple kinematic parameters from PatchWorld trajectories."""

    agent_dx: List[float] = []
    agent_ax: List[float] = []
    agent_dy: List[float] = []
    agent_ay: List[float] = []
    push_ratios: List[float] = []
    contact_distances: List[float] = []
    angle_deltas: List[float] = []

    for traj in trajectories:
        for transition in getattr(traj, "transitions", []):
            obs = _parse_observation_text(getattr(transition, "observation", ""))
            nxt = _parse_observation_text(getattr(transition, "next_observation", ""))
            ax, ay = _parse_action(getattr(transition, "action", ""))
            sx, sy, bx, by, ba = _state_from_graph(obs)
            nsx, nsy, nbx, nby, nba = _state_from_graph(nxt)
            dax = nsx - sx
            day = nsy - sy
            dbx = nbx - bx
            dby = nby - by
            agent_dx.append(dax)
            agent_ax.append(ax)
            agent_dy.append(day)
            agent_ay.append(ay)
            action_mag = math.hypot(dax, day)
            block_mag = math.hypot(dbx, dby)
            if action_mag > 1e-3 and block_mag > 0.25:
                push_ratios.append(max(-2.0, min(2.0, (dbx * dax + dby * day) / (action_mag * action_mag))))
                contact_distances.append(math.hypot(bx - sx, by - sy))
                delta_angle = abs(nba - ba)
                delta_angle = min(delta_angle, abs(2.0 * math.pi - delta_angle))
                angle_deltas.append(delta_angle / max(1e-6, block_mag))

    scale_x = _fit_scale(agent_dx, agent_ax, 100.0)
    scale_y = _fit_scale(agent_dy, agent_ay, 100.0)
    push_scale = float(median(push_ratios)) if push_ratios else 0.15
    contact_threshold = float(max(35.0, min(80.0, median(contact_distances) + 10.0))) if contact_distances else 45.0
    angle_scale = float(max(0.0, min(0.02, median(angle_deltas)))) if angle_deltas else 0.002
    return {
        "agent_scale_x": scale_x,
        "agent_scale_y": scale_y,
        "push_scale": push_scale,
        "contact_threshold": contact_threshold,
        "angle_scale": angle_scale,
    }


def generate_pusht_kinematic_model_code(trajectories: Sequence[Any]) -> str:
    params = fit_pusht_kinematic_params(trajectories)
    return _MODEL_TEMPLATE.format(**{key: repr(float(value)) for key, value in params.items()})


def _angle_delta(next_angle: float, angle: float) -> float:
    delta = (float(next_angle) - float(angle) + math.pi) % (2.0 * math.pi) - math.pi
    return delta


def _feature_vector(state: Tuple[float, float, float, float, float], action: Tuple[float, float]) -> List[float]:
    sx, sy, bx, by, ba = state
    ax, ay = action
    speed = math.hypot(ax, ay)
    rel_x = bx - sx
    rel_y = by - sy
    dist = math.hypot(rel_x, rel_y)
    near = 1.0 if dist <= 70.0 else 0.0
    inv_dist = 1.0 / max(dist, 1.0)
    return [
        1.0,
        ax,
        ay,
        speed,
        sx / 512.0,
        sy / 512.0,
        bx / 512.0,
        by / 512.0,
        math.cos(ba),
        math.sin(ba),
        rel_x / 512.0,
        rel_y / 512.0,
        dist / 512.0,
        near,
        ax * near,
        ay * near,
        ax * rel_x * inv_dist,
        ay * rel_y * inv_dist,
    ]


def _fit_weighted_ridge_matrix(
    features: Sequence[Sequence[float]],
    targets: Sequence[Sequence[float]],
    weights: Sequence[float],
    *,
    ridge: float,
    default_dim: int = 18,
) -> List[List[float]]:
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise RuntimeError("pusht scaffold fitting requires numpy") from exc

    if not features:
        return [[0.0] * default_dim for _ in range(5)]

    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if w.shape[0] != x.shape[0]:
        raise ValueError("Weight count must match feature row count")
    w = np.clip(w, 1e-6, None)
    sw = np.sqrt(w)[:, None]
    xw = x * sw
    yw = y * sw
    xtx = xw.T @ xw
    reg = ridge * np.eye(xtx.shape[0], dtype=np.float64)
    coef = np.linalg.solve(xtx + reg, xw.T @ yw).T
    return [[float(v) for v in row] for row in coef.tolist()]


def iter_weighted_transitions(
    trajectories: Sequence[Any],
    *,
    default_weight: float = 1.0,
) -> List[WeightedTransition]:
    rows: List[WeightedTransition] = []
    for traj in trajectories:
        for transition in getattr(traj, "transitions", []):
            rows.append(
                WeightedTransition(
                    observation=str(getattr(transition, "observation", "")),
                    action=str(getattr(transition, "action", "")),
                    next_observation=str(getattr(transition, "next_observation", "")),
                    weight=default_weight,
                )
            )
    return rows


def load_weighted_transitions_jsonl(path: Path | str) -> List[WeightedTransition]:
    import json
    from pathlib import Path as PathType

    rows: List[WeightedTransition] = []
    with PathType(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            for transition in record.get("transitions", []):
                metadata = transition.get("metadata", {})
                weight = float(metadata.get("weight", transition.get("weight", 1.0)))
                rows.append(
                    WeightedTransition(
                        observation=str(transition.get("observation", "")),
                        action=str(transition.get("action", "")),
                        next_observation=str(transition.get("next_observation", "")),
                        weight=weight,
                    )
                )
    return rows


def load_weighted_transitions_jsonl_paths(paths: Sequence[Union[Path, str]]) -> List[WeightedTransition]:
    rows: List[WeightedTransition] = []
    for path in paths:
        path = Path(path)
        if path.is_file():
            rows.extend(load_weighted_transitions_jsonl(path))
    return rows


def merge_weighted_transitions(
    base: Sequence[WeightedTransition],
    extra: Sequence[WeightedTransition],
) -> List[WeightedTransition]:
    return list(base) + list(extra)


def _fit_ridge_matrix(
    features: Sequence[Sequence[float]],
    targets: Sequence[Sequence[float]],
    *,
    ridge: float,
    default_dim: int = 18,
) -> List[List[float]]:
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise RuntimeError("pusht scaffold fitting requires numpy") from exc

    if not features:
        return [[0.0] * default_dim for _ in range(5)]

    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    xtx = x.T @ x
    reg = ridge * np.eye(xtx.shape[0], dtype=np.float64)
    coef = np.linalg.solve(xtx + reg, x.T @ y).T
    return [[float(v) for v in row] for row in coef.tolist()]


def fit_pusht_scaffold_params(trajectories: Sequence[Any], *, ridge: float = 1e-3) -> Dict[str, Any]:
    """Fit a fixed-feature linear dynamics module for PushT scene graphs."""

    features: List[List[float]] = []
    targets: List[List[float]] = []
    for traj in trajectories:
        for transition in getattr(traj, "transitions", []):
            obs = _parse_observation_text(getattr(transition, "observation", ""))
            nxt = _parse_observation_text(getattr(transition, "next_observation", ""))
            state = _state_from_graph(obs)
            next_state = _state_from_graph(nxt)
            action = _parse_action(getattr(transition, "action", ""))
            sx, sy, bx, by, ba = state
            nsx, nsy, nbx, nby, nba = next_state
            features.append(_feature_vector(state, action))
            targets.append([nsx - sx, nsy - sy, nbx - bx, nby - by, _angle_delta(nba, ba)])

    coef = _fit_ridge_matrix(features, targets, ridge=ridge)
    return {"coef": coef}


def fit_pusht_scaffold_contact_params(
    trajectories: Sequence[Any],
    *,
    ridge: float = 1e-3,
    contact_threshold: float = 70.0,
    min_bucket: int = 3,
    weighted_transitions: Optional[Sequence[WeightedTransition]] = None,
) -> Dict[str, Any]:
    """Fit separate free/contact linear dynamics heads for PushT scene graphs."""

    rows = (
        list(weighted_transitions)
        if weighted_transitions is not None
        else iter_weighted_transitions(trajectories, default_weight=1.0)
    )

    features: List[List[float]] = []
    targets: List[List[float]] = []
    sample_weights: List[float] = []
    contact_features: List[List[float]] = []
    contact_targets: List[List[float]] = []
    contact_weights: List[float] = []
    free_features: List[List[float]] = []
    free_targets: List[List[float]] = []
    free_weights: List[float] = []
    contact_distances: List[float] = []

    for row in rows:
        obs = _parse_observation_text(row.observation)
        nxt = _parse_observation_text(row.next_observation)
        state = _state_from_graph(obs)
        next_state = _state_from_graph(nxt)
        action = _parse_action(row.action)
        sx, sy, bx, by, ba = state
        nsx, nsy, nbx, nby, nba = next_state
        feature = _feature_vector(state, action)
        target = [nsx - sx, nsy - sy, nbx - bx, nby - by, _angle_delta(nba, ba)]
        weight = float(row.weight)
        dist = math.hypot(bx - sx, by - sy)
        block_move = math.hypot(nbx - bx, nby - by)
        features.append(feature)
        targets.append(target)
        sample_weights.append(weight)
        if dist <= contact_threshold:
            contact_features.append(feature)
            contact_targets.append(target)
            contact_weights.append(weight)
            if block_move > 0.25:
                contact_distances.append(dist)
        else:
            free_features.append(feature)
            free_targets.append(target)
            free_weights.append(weight)

    global_coef = _fit_weighted_ridge_matrix(features, targets, sample_weights, ridge=ridge)
    if len(contact_distances) >= min_bucket:
        contact_threshold = float(max(35.0, min(90.0, median(contact_distances) + 5.0)))

    if len(contact_features) >= min_bucket:
        contact_coef = _fit_weighted_ridge_matrix(
            contact_features,
            contact_targets,
            contact_weights,
            ridge=ridge,
        )
    else:
        contact_coef = global_coef

    if len(free_features) >= min_bucket:
        free_coef = _fit_weighted_ridge_matrix(
            free_features,
            free_targets,
            free_weights,
            ridge=ridge,
        )
    else:
        free_coef = global_coef

    return {
        "contact_threshold": contact_threshold,
        "coef_free": free_coef,
        "coef_contact": contact_coef,
    }


def generate_pusht_scaffold_fit_model_code(trajectories: Sequence[Any]) -> str:
    params = fit_pusht_scaffold_params(trajectories)
    return _SCAFFOLD_FIT_TEMPLATE.replace("__COEF_JSON__", json.dumps(params["coef"], sort_keys=True))


def generate_pusht_scaffold_fit_contact_model_code(
    trajectories: Sequence[Any],
    *,
    weighted_transitions: Optional[Sequence[WeightedTransition]] = None,
) -> str:
    params = fit_pusht_scaffold_contact_params(
        trajectories,
        weighted_transitions=weighted_transitions,
    )
    return (
        _SCAFFOLD_FIT_CONTACT_TEMPLATE.replace(
            "__CONTACT_THRESHOLD__",
            repr(float(params["contact_threshold"])),
        )
        .replace("__COEF_FREE_JSON__", json.dumps(params["coef_free"], sort_keys=True))
        .replace("__COEF_CONTACT_JSON__", json.dumps(params["coef_contact"], sort_keys=True))
    )


def generate_pusht_scaffold_fit_contact_weighted_model_code(
    trajectories: Sequence[Any],
    extra_weighted_transitions: Sequence[WeightedTransition],
) -> str:
    base_rows = iter_weighted_transitions(trajectories, default_weight=1.0)
    merged = merge_weighted_transitions(base_rows, extra_weighted_transitions)
    return generate_pusht_scaffold_fit_contact_model_code(trajectories, weighted_transitions=merged)


def _zero_coef_block_rows(coef: Sequence[Sequence[float]]) -> List[List[float]]:
    rows = [list(row) for row in coef]
    for row_idx in (2, 3, 4):
        if row_idx < len(rows):
            rows[row_idx] = [0.0] * len(rows[row_idx])
    return rows


def fit_pusht_scaffold_contact_gated_params(
    trajectories: Sequence[Any],
    *,
    ridge: float = 1e-3,
    contact_threshold: float = 70.0,
    min_bucket: int = 3,
    weighted_transitions: Optional[Sequence[WeightedTransition]] = None,
) -> Dict[str, Any]:
    """Contact-fit params with free-head block rows zeroed (R1 by construction)."""

    params = fit_pusht_scaffold_contact_params(
        trajectories,
        ridge=ridge,
        contact_threshold=contact_threshold,
        min_bucket=min_bucket,
        weighted_transitions=weighted_transitions,
    )
    params["coef_free"] = _zero_coef_block_rows(params["coef_free"])
    return params


def _compact_feature_vector(
    state: Tuple[float, float, float, float, float],
    action: Tuple[float, float],
) -> List[float]:
    sx, sy, bx, by, ba = state
    ax, ay = action
    dist = math.hypot(bx - sx, by - sy)
    return [
        sx / 512.0,
        sy / 512.0,
        bx / 512.0,
        by / 512.0,
        math.cos(ba),
        math.sin(ba),
        dist / 512.0,
        ax,
        ay,
        math.hypot(ax, ay),
    ]


def iter_contact_multistep_pairs(
    trajectories: Sequence[Any],
    *,
    contact_threshold: float = 70.0,
) -> List[Tuple[WeightedTransition, WeightedTransition]]:
    """Consecutive near-contact transitions for multi-step MLP fitting."""

    pairs: List[Tuple[WeightedTransition, WeightedTransition]] = []
    for traj in trajectories:
        transitions = list(getattr(traj, "transitions", []))
        weighted = [
            WeightedTransition(
                observation=str(getattr(t, "observation", "")),
                action=str(getattr(t, "action", "")),
                next_observation=str(getattr(t, "next_observation", "")),
                weight=1.0,
            )
            for t in transitions
        ]
        for idx in range(len(weighted) - 1):
            first, second = weighted[idx], weighted[idx + 1]
            state = _state_from_graph(_parse_observation_text(first.observation))
            mid_state = _state_from_graph(_parse_observation_text(first.next_observation))
            sx, sy, bx, by, _ = state
            msx, msy, mbx, mby, _ = mid_state
            if (
                math.hypot(bx - sx, by - sy) <= contact_threshold
                and math.hypot(mbx - msx, mby - msy) <= contact_threshold
            ):
                pairs.append((first, second))
    return pairs


def _mlp_forward_numpy(
    compact: Sequence[float],
    w1: Any,
    b1: Any,
    w2: Any,
    b2: Any,
) -> Any:
    import numpy as np

    h_pre = np.asarray(compact, dtype=np.float64) @ np.asarray(w1, dtype=np.float64).T + np.asarray(b1, dtype=np.float64)
    h = np.maximum(0.0, h_pre)
    return h @ np.asarray(w2, dtype=np.float64).T + np.asarray(b2, dtype=np.float64)


def _predict_block_delta_numpy(
    state: Tuple[float, float, float, float, float],
    action: Tuple[float, float],
    contact_coef: Sequence[Sequence[float]],
    *,
    w1: Any,
    b1: Any,
    w2: Any,
    b2: Any,
) -> Tuple[float, float, float]:
    feat_full = _feature_vector(state, action)
    compact = _compact_feature_vector(state, action)
    linear_block = [
        sum(c * f for c, f in zip(contact_coef[row_idx], feat_full))
        for row_idx in (2, 3, 4)
    ]
    residual = _mlp_forward_numpy(compact, w1, b1, w2, b2)
    return (
        linear_block[0] + float(residual[0]),
        linear_block[1] + float(residual[1]),
        linear_block[2] + float(residual[2]),
    )


def fit_contact_mlp_residual(
    rows: Sequence[WeightedTransition],
    contact_coef: Sequence[Sequence[float]],
    *,
    contact_threshold: float,
    hidden_dim: int = 32,
    ridge: float = 1e-4,
    steps: int = 2000,
    lr: float = 0.01,
    seed: int = 42,
    multistep_pairs: Optional[Sequence[Tuple[WeightedTransition, WeightedTransition]]] = None,
    multistep_weight: float = 0.5,
) -> Dict[str, Any]:
    """Fit a tiny MLP residual on block deltas for contact transitions only."""

    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise RuntimeError("contact MLP fitting requires numpy") from exc

    inputs: List[List[float]] = []
    targets: List[List[float]] = []
    for row in rows:
        state = _state_from_graph(_parse_observation_text(row.observation))
        next_state = _state_from_graph(_parse_observation_text(row.next_observation))
        action = _parse_action(row.action)
        sx, sy, bx, by, ba = state
        dist = math.hypot(bx - sx, by - sy)
        if dist > contact_threshold:
            continue
        feat_full = _feature_vector(state, action)
        compact = _compact_feature_vector(state, action)
        linear_block = [
            sum(c * f for c, f in zip(contact_coef[row_idx], feat_full))
            for row_idx in (2, 3, 4)
        ]
        true_block = [
            next_state[2] - bx,
            next_state[3] - by,
            _angle_delta(next_state[4], ba),
        ]
        residual = [true - pred for true, pred in zip(true_block, linear_block)]
        inputs.append(compact)
        targets.append(residual)

    multistep_data: List[Tuple[Tuple[float, float, float, float, float], Tuple[float, float], Tuple[float, float, float, float, float], Tuple[float, float], Tuple[float, float, float]]] = []
    if multistep_pairs:
        for first, second in multistep_pairs:
            state0 = _state_from_graph(_parse_observation_text(first.observation))
            state1 = _state_from_graph(_parse_observation_text(first.next_observation))
            state2 = _state_from_graph(_parse_observation_text(second.next_observation))
            sx, sy, bx, by, ba = state0
            dist0 = math.hypot(bx - sx, by - sy)
            if dist0 > contact_threshold:
                continue
            msx, msy, mbx, mby, _ = state1
            if math.hypot(mbx - msx, mby - msy) > contact_threshold:
                continue
            action0 = _parse_action(first.action)
            action1 = _parse_action(second.action)
            _, _, nbx, nby, nba = state2
            true_cum = (nbx - bx, nby - by, _angle_delta(nba, ba))
            multistep_data.append((state0, action0, state1, action1, true_cum))

    input_dim = 10
    output_dim = 3
    if not inputs and not multistep_data:
        return {
            "hidden_dim": hidden_dim,
            "input_dim": input_dim,
            "output_dim": output_dim,
            "w1": [[0.0] * input_dim for _ in range(hidden_dim)],
            "b1": [0.0] * hidden_dim,
            "w2": [[0.0] * hidden_dim for _ in range(output_dim)],
            "b2": [0.0] * output_dim,
            "n_contact_samples": 0,
            "n_multistep_pairs": 0,
        }

    rng = np.random.default_rng(seed)
    x = np.asarray(inputs, dtype=np.float64) if inputs else np.zeros((0, input_dim), dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64) if inputs else np.zeros((0, output_dim), dtype=np.float64)
    w1 = rng.normal(scale=0.05, size=(hidden_dim, input_dim))
    b1 = np.zeros(hidden_dim, dtype=np.float64)
    w2 = rng.normal(scale=0.05, size=(output_dim, hidden_dim))
    b2 = np.zeros(output_dim, dtype=np.float64)

    for _ in range(steps):
        grad_w2 = 2.0 * ridge * w2
        grad_b2 = np.zeros(output_dim, dtype=np.float64)
        grad_w1 = 2.0 * ridge * w1
        grad_b1 = np.zeros(hidden_dim, dtype=np.float64)

        if x.shape[0] > 0:
            h_pre = x @ w1.T + b1
            h = np.maximum(0.0, h_pre)
            pred = h @ w2.T + b2
            diff = pred - y
            loss_grad = (2.0 / x.shape[0]) * diff
            grad_w2 += loss_grad.T @ h
            grad_b2 += loss_grad.sum(axis=0)
            grad_h = loss_grad @ w2
            grad_h[h_pre <= 0.0] = 0.0
            grad_w1 += grad_h.T @ x
            grad_b1 += grad_h.sum(axis=0)

        if multistep_data:
            ms_grad_w2 = np.zeros_like(w2)
            ms_grad_b2 = np.zeros_like(b2)
            ms_grad_w1 = np.zeros_like(w1)
            ms_grad_b1 = np.zeros_like(b1)
            for state0, action0, state1, action1, true_cum in multistep_data:
                compact0 = _compact_feature_vector(state0, action0)
                h_pre0 = compact0 @ w1.T + b1
                h0 = np.maximum(0.0, h_pre0)
                res0 = h0 @ w2.T + b2
                feat0 = _feature_vector(state0, action0)
                linear0 = np.asarray(
                    [sum(c * f for c, f in zip(contact_coef[row_idx], feat0)) for row_idx in (2, 3, 4)],
                    dtype=np.float64,
                )
                delta0 = linear0 + res0

                compact1 = _compact_feature_vector(state1, action1)
                h_pre1 = compact1 @ w1.T + b1
                h1 = np.maximum(0.0, h_pre1)
                res1 = h1 @ w2.T + b2
                feat1 = _feature_vector(state1, action1)
                linear1 = np.asarray(
                    [sum(c * f for c, f in zip(contact_coef[row_idx], feat1)) for row_idx in (2, 3, 4)],
                    dtype=np.float64,
                )
                delta1 = linear1 + res1
                pred_cum = delta0 + delta1
                err = pred_cum - np.asarray(true_cum, dtype=np.float64)
                scale = (2.0 * multistep_weight) / max(1, len(multistep_data))
                loss_grad = scale * err
                ms_grad_w2 += np.outer(loss_grad, h0) + np.outer(loss_grad, h1)
                ms_grad_b2 += 2.0 * loss_grad
                grad_h1 = loss_grad @ w2
                grad_h1[h_pre1 <= 0.0] = 0.0
                grad_h0 = loss_grad @ w2
                grad_h0[h_pre0 <= 0.0] = 0.0
                ms_grad_w1 += np.outer(grad_h0, compact0) + np.outer(grad_h1, compact1)
                ms_grad_b1 += grad_h0 + grad_h1

            grad_w2 += ms_grad_w2
            grad_b2 += ms_grad_b2
            grad_w1 += ms_grad_w1
            grad_b1 += ms_grad_b1

        w2 -= lr * grad_w2
        b2 -= lr * grad_b2
        w1 -= lr * grad_w1
        b1 -= lr * grad_b1

    return {
        "hidden_dim": hidden_dim,
        "input_dim": input_dim,
        "output_dim": output_dim,
        "w1": w1.tolist(),
        "b1": b1.tolist(),
        "w2": w2.tolist(),
        "b2": b2.tolist(),
        "n_contact_samples": len(inputs),
        "n_multistep_pairs": len(multistep_data),
    }


def fit_pusht_scaffold_contact_gated_mlp_params(
    trajectories: Sequence[Any],
    *,
    ridge: float = 1e-3,
    contact_threshold: float = 70.0,
    min_bucket: int = 3,
    weighted_transitions: Optional[Sequence[WeightedTransition]] = None,
    mlp_hidden_dim: int = 32,
    mlp_steps: int = 2000,
    mlp_lr: float = 0.01,
    mlp_ridge: float = 1e-4,
    mlp_rollout_steps: int = 1,
    multistep_weight: float = 0.5,
    seed: int = 42,
) -> Dict[str, Any]:
    params = fit_pusht_scaffold_contact_gated_params(
        trajectories,
        ridge=ridge,
        contact_threshold=contact_threshold,
        min_bucket=min_bucket,
        weighted_transitions=weighted_transitions,
    )
    rows = (
        list(weighted_transitions)
        if weighted_transitions is not None
        else iter_weighted_transitions(trajectories, default_weight=1.0)
    )
    multistep_pairs = (
        iter_contact_multistep_pairs(trajectories, contact_threshold=float(params["contact_threshold"]))
        if mlp_rollout_steps >= 2
        else None
    )
    params["mlp"] = fit_contact_mlp_residual(
        rows,
        params["coef_contact"],
        contact_threshold=float(params["contact_threshold"]),
        hidden_dim=mlp_hidden_dim,
        ridge=mlp_ridge,
        steps=mlp_steps,
        lr=mlp_lr,
        seed=seed,
        multistep_pairs=multistep_pairs,
        multistep_weight=multistep_weight,
    )
    return params


def generate_pusht_scaffold_fit_contact_gated_model_code(
    trajectories: Sequence[Any],
    *,
    weighted_transitions: Optional[Sequence[WeightedTransition]] = None,
) -> str:
    params = fit_pusht_scaffold_contact_gated_params(
        trajectories,
        weighted_transitions=weighted_transitions,
    )
    return (
        _SCAFFOLD_FIT_CONTACT_GATED_TEMPLATE.replace(
            "__CONTACT_THRESHOLD__",
            repr(float(params["contact_threshold"])),
        )
        .replace("__COEF_FREE_JSON__", json.dumps(params["coef_free"], sort_keys=True))
        .replace("__COEF_CONTACT_JSON__", json.dumps(params["coef_contact"], sort_keys=True))
    )


def generate_pusht_scaffold_fit_contact_gated_mlp_model_code(
    trajectories: Sequence[Any],
    *,
    weighted_transitions: Optional[Sequence[WeightedTransition]] = None,
) -> str:
    params = fit_pusht_scaffold_contact_gated_mlp_params(
        trajectories,
        weighted_transitions=weighted_transitions,
    )
    return (
        _SCAFFOLD_FIT_CONTACT_GATED_MLP_TEMPLATE.replace(
            "__CONTACT_THRESHOLD__",
            repr(float(params["contact_threshold"])),
        )
        .replace("__COEF_FREE_JSON__", json.dumps(params["coef_free"], sort_keys=True))
        .replace("__COEF_CONTACT_JSON__", json.dumps(params["coef_contact"], sort_keys=True))
        .replace("__MLP_JSON__", json.dumps(params["mlp"], sort_keys=True))
    )


def generate_pusht_scaffold_fit_contact_gated_weighted_model_code(
    trajectories: Sequence[Any],
    extra_weighted_transitions: Sequence[WeightedTransition],
) -> str:
    base_rows = iter_weighted_transitions(trajectories, default_weight=1.0)
    merged = merge_weighted_transitions(base_rows, extra_weighted_transitions)
    return generate_pusht_scaffold_fit_contact_gated_model_code(trajectories, weighted_transitions=merged)


def generate_pusht_scaffold_fit_contact_gated_mlp_weighted_model_code(
    trajectories: Sequence[Any],
    extra_weighted_transitions: Sequence[WeightedTransition],
) -> str:
    base_rows = iter_weighted_transitions(trajectories, default_weight=1.0)
    merged = merge_weighted_transitions(base_rows, extra_weighted_transitions)
    return generate_pusht_scaffold_fit_contact_gated_mlp_model_code(trajectories, weighted_transitions=merged)


_MODEL_TEMPLATE = r'''
from patchworld.worldmodel_base import BaseWorldModel
import json
import math
import re
import copy


class PushtKinematicWorldModel(BaseWorldModel):
    def __init__(self):
        self.agent_scale_x = {agent_scale_x}
        self.agent_scale_y = {agent_scale_y}
        self.push_scale = {push_scale}
        self.contact_threshold = {contact_threshold}
        self.angle_scale = {angle_scale}

    def init_belief(self):
        return {{
            "env": "pusht",
            "source": "kinematic",
            "step": 0,
            "goal": "move block to the target T pose",
            "objects": {{}},
            "relations": {{}},
            "metadata": {{}},
        }}

    def init_belief_from_observation(self, obs_text: str):
        return self.correct_belief(self.init_belief(), obs_text)

    def extract_valid_action_forms(self):
        return {{
            "move_left": ["move_left[dx=<float>,dy=<float>,speed=<float>]"],
            "move_right": ["move_right[dx=<float>,dy=<float>,speed=<float>]"],
            "move_up": ["move_up[dx=<float>,dy=<float>,speed=<float>]"],
            "move_down": ["move_down[dx=<float>,dy=<float>,speed=<float>]"],
            "hold": ["hold[dx=<float>,dy=<float>,speed=<float>]"],
        }}

    def parse_observation(self, obs_text: str):
        graph = self.init_belief()
        section = ""
        for raw_line in str(obs_text or "").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("Environment:"):
                graph["env"] = line.split(":", 1)[1].strip()
                section = ""
            elif line.startswith("Source:"):
                graph["source"] = line.split(":", 1)[1].strip()
                section = ""
            elif line.startswith("Step:"):
                try:
                    graph["step"] = int(float(line.split(":", 1)[1].strip()))
                except Exception:
                    graph["step"] = 0
                section = ""
            elif line.startswith("Goal:"):
                graph["goal"] = line.split(":", 1)[1].strip()
                section = ""
            elif line.startswith("Objects:"):
                section = "objects"
            elif line.startswith("Relations:"):
                section = "relations"
            elif line.startswith("Metadata:"):
                try:
                    graph["metadata"] = json.loads(line.split(":", 1)[1].strip())
                except Exception:
                    graph["metadata"] = {{}}
                section = ""
            elif section == "objects" and line.startswith("-"):
                obj = self._parse_object(line)
                if obj:
                    graph["objects"][obj["id"]] = obj
            elif section == "relations" and line.startswith("-"):
                rel = self._parse_relation(line)
                if rel:
                    graph["relations"][rel["key"]] = rel
        self._recompute_relations(graph)
        return graph

    def _parse_object(self, line):
        match = re.match(r"^-\s*(\w+):(\w+)\s*(.*)$", line)
        if not match:
            return None
        obj = {{"id": match.group(1), "category": match.group(2), "x": 0.0, "y": 0.0, "vx": 0.0, "vy": 0.0, "conf": 1.0}}
        for key, value in re.findall(r"(\w+)=(-?\d+(?:\.\d+)?)", match.group(3)):
            obj[key] = float(value)
        return obj

    def _parse_relation(self, line):
        match = re.match(r"^-\s*(\w+)\s+(\w+(?:_\w+)*)\s+(\w+)\s*=\s*(.+)$", line)
        if not match:
            return None
        value = match.group(4).strip()
        try:
            value = float(value)
        except Exception:
            pass
        return {{"subject": match.group(1), "predicate": match.group(2), "object": match.group(3), "value": value, "key": f"{{match.group(1)}}_{{match.group(2)}}_{{match.group(3)}}"}}

    def correct_belief(self, belief_prior, obs_text: str):
        observed = self.parse_observation(obs_text)
        merged = copy.deepcopy(belief_prior) if isinstance(belief_prior, dict) else self.init_belief()
        merged.update({{key: observed.get(key, merged.get(key)) for key in ("env", "source", "step", "goal", "metadata")}})
        merged["objects"] = observed.get("objects", {{}})
        merged["relations"] = observed.get("relations", {{}})
        return merged

    def predict_belief(self, belief, action: str):
        b = copy.deepcopy(belief) if isinstance(belief, dict) else self.init_belief()
        objects = b.setdefault("objects", {{}})
        agent = objects.setdefault("agent", {{"id": "agent", "category": "circular_pusher", "x": 0.0, "y": 0.0, "vx": 0.0, "vy": 0.0, "conf": 1.0}})
        block = objects.setdefault("block", {{"id": "block", "category": "t_block", "x": 0.0, "y": 0.0, "angle": 0.0, "conf": 1.0}})
        dx, dy, speed = self._parse_action(action)
        agent_dx = dx * self.agent_scale_x
        agent_dy = dy * self.agent_scale_y
        old_ax = float(agent.get("x", 0.0))
        old_ay = float(agent.get("y", 0.0))
        old_bx = float(block.get("x", 0.0))
        old_by = float(block.get("y", 0.0))
        dist = math.hypot(old_bx - old_ax, old_by - old_ay)
        agent["x"] = self._clip(old_ax + agent_dx)
        agent["y"] = self._clip(old_ay + agent_dy)
        agent["vx"] = agent_dx
        agent["vy"] = agent_dy
        if dist <= self.contact_threshold or math.hypot(old_bx - agent["x"], old_by - agent["y"]) <= self.contact_threshold:
            block["x"] = self._clip(old_bx + agent_dx * self.push_scale)
            block["y"] = self._clip(old_by + agent_dy * self.push_scale)
            block["angle"] = (float(block.get("angle", 0.0)) + (dx * agent_dy - dy * agent_dx) * self.angle_scale) % (2.0 * math.pi)
        b["step"] = int(b.get("step", 0) or 0) + 1
        self._recompute_relations(b)
        return b

    def _parse_action(self, action):
        match = re.match(r"^\w+\[dx=(-?\d+(?:\.\d+)?),dy=(-?\d+(?:\.\d+)?),speed=(-?\d+(?:\.\d+)?)\]$", str(action or "").strip())
        if not match:
            return 0.0, 0.0, 0.0
        return float(match.group(1)), float(match.group(2)), float(match.group(3))

    def _clip(self, value):
        return max(0.0, min(512.0, float(value)))

    def _recompute_relations(self, graph):
        objects = graph.get("objects", {{}})
        agent = objects.get("agent")
        block = objects.get("block")
        rels = {{}}
        if agent and block:
            dx = float(block.get("x", 0.0)) - float(agent.get("x", 0.0))
            dy = float(block.get("y", 0.0)) - float(agent.get("y", 0.0))
            dist = math.hypot(dx, dy)
            direction = ("right" if dx >= 0 else "left") if abs(dx) >= abs(dy) else ("up" if dy >= 0 else "down")
            rels["agent_distance_to_block"] = {{"subject": "agent", "predicate": "distance_to", "object": "block", "value": dist, "key": "agent_distance_to_block"}}
            rels["agent_near_block"] = {{"subject": "agent", "predicate": "near", "object": "block", "value": float(dist <= 35.0), "key": "agent_near_block"}}
            rels["agent_relative_direction_to_block"] = {{"subject": "agent", "predicate": "relative_direction_to", "object": "block", "value": direction, "key": "agent_relative_direction_to_block"}}
        graph["relations"] = rels

    def readout_observation(self, belief, action: str = ""):
        b = belief if isinstance(belief, dict) else self.init_belief()
        objects = b.get("objects", {{}})
        agent = objects.get("agent", {{"id": "agent", "category": "circular_pusher", "x": 0.0, "y": 0.0, "vx": 0.0, "vy": 0.0, "conf": 1.0}})
        block = objects.get("block", {{"id": "block", "category": "t_block", "x": 0.0, "y": 0.0, "angle": 0.0, "conf": 1.0}})
        self._recompute_relations(b)
        lines = [
            "Environment: pusht",
            f"Source: {{b.get('source', 'kinematic')}}",
            f"Step: {{int(b.get('step', 0) or 0)}}",
            f"Goal: {{b.get('goal', 'move block to the target T pose')}}",
            "Objects:",
            f"- agent:circular_pusher x={{float(agent.get('x', 0.0)):.3f}} y={{float(agent.get('y', 0.0)):.3f}} vx={{float(agent.get('vx', 0.0)):.3f}} vy={{float(agent.get('vy', 0.0)):.3f}} conf={{float(agent.get('conf', 1.0)):.3f}}",
            f"- block:t_block x={{float(block.get('x', 0.0)):.3f}} y={{float(block.get('y', 0.0)):.3f}} angle={{float(block.get('angle', 0.0)):.3f}} conf={{float(block.get('conf', 1.0)):.3f}}",
            "Relations:",
        ]
        for key in sorted(b.get("relations", {{}})):
            rel = b["relations"][key]
            value = rel.get("value")
            value_text = f"{{float(value):.3f}}" if isinstance(value, (int, float)) else str(value)
            lines.append(f"- {{rel.get('subject')}} {{rel.get('predicate')}} {{rel.get('object')}} = {{value_text}}")
        lines.append("Metadata: {{}}")
        return "\n".join(lines)
'''.strip() + "\n"


_SCAFFOLD_FIT_TEMPLATE = r'''
from patchworld.worldmodel_base import BaseWorldModel
import json
import math
import re
import copy


class PushtScaffoldFitWorldModel(BaseWorldModel):
    COEF = __COEF_JSON__

    def init_belief(self):
        return {
            "env": "pusht",
            "source": "scaffold_fit",
            "step": 0,
            "goal": "move block to the target T pose",
            "objects": {},
            "relations": {},
            "metadata": {},
        }

    def init_belief_from_observation(self, obs_text: str):
        return self.correct_belief(self.init_belief(), obs_text)

    def extract_valid_action_forms(self):
        return {
            "move_left": ["move_left[dx=<float>,dy=<float>,speed=<float>]"],
            "move_right": ["move_right[dx=<float>,dy=<float>,speed=<float>]"],
            "move_up": ["move_up[dx=<float>,dy=<float>,speed=<float>]"],
            "move_down": ["move_down[dx=<float>,dy=<float>,speed=<float>]"],
            "hold": ["hold[dx=<float>,dy=<float>,speed=<float>]"],
        }

    def parse_observation(self, obs_text: str):
        graph = self.init_belief()
        section = ""
        for raw_line in str(obs_text or "").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("Environment:"):
                graph["env"] = line.split(":", 1)[1].strip()
                section = ""
            elif line.startswith("Source:"):
                graph["source"] = line.split(":", 1)[1].strip()
                section = ""
            elif line.startswith("Step:"):
                try:
                    graph["step"] = int(float(line.split(":", 1)[1].strip()))
                except Exception:
                    graph["step"] = 0
                section = ""
            elif line.startswith("Goal:"):
                graph["goal"] = line.split(":", 1)[1].strip()
                section = ""
            elif line.startswith("Objects:"):
                section = "objects"
            elif line.startswith("Relations:"):
                section = "relations"
            elif line.startswith("Metadata:"):
                try:
                    graph["metadata"] = json.loads(line.split(":", 1)[1].strip())
                except Exception:
                    graph["metadata"] = {}
                section = ""
            elif section == "objects" and line.startswith("-"):
                obj = self._parse_object(line)
                if obj:
                    graph["objects"][obj["id"]] = obj
            elif section == "relations" and line.startswith("-"):
                rel = self._parse_relation(line)
                if rel:
                    graph["relations"][rel["key"]] = rel
        self._recompute_relations(graph)
        return graph

    def _parse_object(self, line):
        match = re.match(r"^-\s*(\w+):(\w+)\s*(.*)$", line)
        if not match:
            return None
        obj = {"id": match.group(1), "category": match.group(2), "x": 0.0, "y": 0.0, "vx": 0.0, "vy": 0.0, "conf": 1.0}
        for key, value in re.findall(r"(\w+)=(-?\d+(?:\.\d+)?)", match.group(3)):
            obj[key] = float(value)
        return obj

    def _parse_relation(self, line):
        match = re.match(r"^-\s*(\w+)\s+(\w+(?:_\w+)*)\s+(\w+)\s*=\s*(.+)$", line)
        if not match:
            return None
        value = match.group(4).strip()
        try:
            value = float(value)
        except Exception:
            pass
        return {"subject": match.group(1), "predicate": match.group(2), "object": match.group(3), "value": value, "key": f"{match.group(1)}_{match.group(2)}_{match.group(3)}"}

    def correct_belief(self, belief_prior, obs_text: str):
        observed = self.parse_observation(obs_text)
        merged = copy.deepcopy(belief_prior) if isinstance(belief_prior, dict) else self.init_belief()
        for key in ("env", "source", "step", "goal", "metadata"):
            merged[key] = observed.get(key, merged.get(key))
        merged["objects"] = observed.get("objects", {})
        merged["relations"] = observed.get("relations", {})
        return merged

    def predict_belief(self, belief, action: str):
        b = copy.deepcopy(belief) if isinstance(belief, dict) else self.init_belief()
        objects = b.setdefault("objects", {})
        agent = objects.setdefault("agent", {"id": "agent", "category": "circular_pusher", "x": 0.0, "y": 0.0, "vx": 0.0, "vy": 0.0, "conf": 1.0})
        block = objects.setdefault("block", {"id": "block", "category": "t_block", "x": 0.0, "y": 0.0, "angle": 0.0, "conf": 1.0})
        dx, dy = self._parse_action(action)
        feats = self._features(agent, block, dx, dy)
        delta = [sum(c * f for c, f in zip(row, feats)) for row in self.COEF]
        agent["x"] = self._clip(float(agent.get("x", 0.0)) + delta[0])
        agent["y"] = self._clip(float(agent.get("y", 0.0)) + delta[1])
        agent["vx"] = delta[0]
        agent["vy"] = delta[1]
        block["x"] = self._clip(float(block.get("x", 0.0)) + delta[2])
        block["y"] = self._clip(float(block.get("y", 0.0)) + delta[3])
        block["angle"] = (float(block.get("angle", 0.0)) + delta[4]) % (2.0 * math.pi)
        b["step"] = int(b.get("step", 0) or 0) + 1
        self._recompute_relations(b)
        return b

    def _parse_action(self, action):
        match = re.match(r"^\w+\[dx=(-?\d+(?:\.\d+)?),dy=(-?\d+(?:\.\d+)?),speed=(-?\d+(?:\.\d+)?)\]$", str(action or "").strip())
        if not match:
            return 0.0, 0.0
        return float(match.group(1)), float(match.group(2))

    def _features(self, agent, block, dx, dy):
        sx = float(agent.get("x", 0.0))
        sy = float(agent.get("y", 0.0))
        bx = float(block.get("x", 0.0))
        by = float(block.get("y", 0.0))
        ba = float(block.get("angle", 0.0))
        speed = math.hypot(dx, dy)
        rel_x = bx - sx
        rel_y = by - sy
        dist = math.hypot(rel_x, rel_y)
        near = 1.0 if dist <= 70.0 else 0.0
        inv_dist = 1.0 / max(dist, 1.0)
        return [
            1.0,
            dx,
            dy,
            speed,
            sx / 512.0,
            sy / 512.0,
            bx / 512.0,
            by / 512.0,
            math.cos(ba),
            math.sin(ba),
            rel_x / 512.0,
            rel_y / 512.0,
            dist / 512.0,
            near,
            dx * near,
            dy * near,
            dx * rel_x * inv_dist,
            dy * rel_y * inv_dist,
        ]

    def _clip(self, value):
        return max(0.0, min(512.0, float(value)))

    def _recompute_relations(self, graph):
        objects = graph.get("objects", {})
        agent = objects.get("agent")
        block = objects.get("block")
        rels = {}
        if agent and block:
            dx = float(block.get("x", 0.0)) - float(agent.get("x", 0.0))
            dy = float(block.get("y", 0.0)) - float(agent.get("y", 0.0))
            dist = math.hypot(dx, dy)
            direction = ("right" if dx >= 0 else "left") if abs(dx) >= abs(dy) else ("up" if dy >= 0 else "down")
            rels["agent_distance_to_block"] = {"subject": "agent", "predicate": "distance_to", "object": "block", "value": dist, "key": "agent_distance_to_block"}
            rels["agent_near_block"] = {"subject": "agent", "predicate": "near", "object": "block", "value": float(dist <= 35.0), "key": "agent_near_block"}
            rels["agent_relative_direction_to_block"] = {"subject": "agent", "predicate": "relative_direction_to", "object": "block", "value": direction, "key": "agent_relative_direction_to_block"}
        graph["relations"] = rels

    def readout_observation(self, belief, action: str = ""):
        b = belief if isinstance(belief, dict) else self.init_belief()
        objects = b.get("objects", {})
        agent = objects.get("agent", {"id": "agent", "category": "circular_pusher", "x": 0.0, "y": 0.0, "vx": 0.0, "vy": 0.0, "conf": 1.0})
        block = objects.get("block", {"id": "block", "category": "t_block", "x": 0.0, "y": 0.0, "angle": 0.0, "conf": 1.0})
        self._recompute_relations(b)
        lines = [
            "Environment: pusht",
            f"Source: {b.get('source', 'scaffold_fit')}",
            f"Step: {int(b.get('step', 0) or 0)}",
            f"Goal: {b.get('goal', 'move block to the target T pose')}",
            "Objects:",
            f"- agent:circular_pusher x={float(agent.get('x', 0.0)):.3f} y={float(agent.get('y', 0.0)):.3f} vx={float(agent.get('vx', 0.0)):.3f} vy={float(agent.get('vy', 0.0)):.3f} conf={float(agent.get('conf', 1.0)):.3f}",
            f"- block:t_block x={float(block.get('x', 0.0)):.3f} y={float(block.get('y', 0.0)):.3f} angle={float(block.get('angle', 0.0)):.3f} conf={float(block.get('conf', 1.0)):.3f}",
            "Relations:",
        ]
        for key in sorted(b.get("relations", {})):
            rel = b["relations"][key]
            value = rel.get("value")
            value_text = f"{float(value):.3f}" if isinstance(value, (int, float)) else str(value)
            lines.append(f"- {rel.get('subject')} {rel.get('predicate')} {rel.get('object')} = {value_text}")
        lines.append("Metadata: {}")
        return "\n".join(lines)
'''.strip() + "\n"


_SCAFFOLD_FIT_CONTACT_TEMPLATE = r'''
from patchworld.worldmodel_base import BaseWorldModel
import json
import math
import re
import copy


class PushtScaffoldFitContactWorldModel(BaseWorldModel):
    CONTACT_THRESHOLD = __CONTACT_THRESHOLD__
    COEF_FREE = __COEF_FREE_JSON__
    COEF_CONTACT = __COEF_CONTACT_JSON__

    def init_belief(self):
        return {
            "env": "pusht",
            "source": "scaffold_fit_contact",
            "step": 0,
            "goal": "move block to the target T pose",
            "objects": {},
            "relations": {},
            "metadata": {},
        }

    def init_belief_from_observation(self, obs_text: str):
        return self.correct_belief(self.init_belief(), obs_text)

    def extract_valid_action_forms(self):
        return {
            "move_left": ["move_left[dx=<float>,dy=<float>,speed=<float>]"],
            "move_right": ["move_right[dx=<float>,dy=<float>,speed=<float>]"],
            "move_up": ["move_up[dx=<float>,dy=<float>,speed=<float>]"],
            "move_down": ["move_down[dx=<float>,dy=<float>,speed=<float>]"],
            "hold": ["hold[dx=<float>,dy=<float>,speed=<float>]"],
        }

    def parse_observation(self, obs_text: str):
        graph = self.init_belief()
        section = ""
        for raw_line in str(obs_text or "").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("Environment:"):
                graph["env"] = line.split(":", 1)[1].strip()
                section = ""
            elif line.startswith("Source:"):
                graph["source"] = line.split(":", 1)[1].strip()
                section = ""
            elif line.startswith("Step:"):
                try:
                    graph["step"] = int(float(line.split(":", 1)[1].strip()))
                except Exception:
                    graph["step"] = 0
                section = ""
            elif line.startswith("Goal:"):
                graph["goal"] = line.split(":", 1)[1].strip()
                section = ""
            elif line.startswith("Objects:"):
                section = "objects"
            elif line.startswith("Relations:"):
                section = "relations"
            elif line.startswith("Metadata:"):
                try:
                    graph["metadata"] = json.loads(line.split(":", 1)[1].strip())
                except Exception:
                    graph["metadata"] = {}
                section = ""
            elif section == "objects" and line.startswith("-"):
                obj = self._parse_object(line)
                if obj:
                    graph["objects"][obj["id"]] = obj
            elif section == "relations" and line.startswith("-"):
                rel = self._parse_relation(line)
                if rel:
                    graph["relations"][rel["key"]] = rel
        self._recompute_relations(graph)
        return graph

    def _parse_object(self, line):
        match = re.match(r"^-\s*(\w+):(\w+)\s*(.*)$", line)
        if not match:
            return None
        obj = {"id": match.group(1), "category": match.group(2), "x": 0.0, "y": 0.0, "vx": 0.0, "vy": 0.0, "conf": 1.0}
        for key, value in re.findall(r"(\w+)=(-?\d+(?:\.\d+)?)", match.group(3)):
            obj[key] = float(value)
        return obj

    def _parse_relation(self, line):
        match = re.match(r"^-\s*(\w+)\s+(\w+(?:_\w+)*)\s+(\w+)\s*=\s*(.+)$", line)
        if not match:
            return None
        value = match.group(4).strip()
        try:
            value = float(value)
        except Exception:
            pass
        return {"subject": match.group(1), "predicate": match.group(2), "object": match.group(3), "value": value, "key": f"{match.group(1)}_{match.group(2)}_{match.group(3)}"}

    def correct_belief(self, belief_prior, obs_text: str):
        observed = self.parse_observation(obs_text)
        merged = copy.deepcopy(belief_prior) if isinstance(belief_prior, dict) else self.init_belief()
        for key in ("env", "source", "step", "goal", "metadata"):
            merged[key] = observed.get(key, merged.get(key))
        merged["objects"] = observed.get("objects", {})
        merged["relations"] = observed.get("relations", {})
        return merged

    def predict_belief(self, belief, action: str):
        b = copy.deepcopy(belief) if isinstance(belief, dict) else self.init_belief()
        objects = b.setdefault("objects", {})
        agent = objects.setdefault("agent", {"id": "agent", "category": "circular_pusher", "x": 0.0, "y": 0.0, "vx": 0.0, "vy": 0.0, "conf": 1.0})
        block = objects.setdefault("block", {"id": "block", "category": "t_block", "x": 0.0, "y": 0.0, "angle": 0.0, "conf": 1.0})
        dx, dy = self._parse_action(action)
        feats = self._features(agent, block, dx, dy)
        dist = math.hypot(float(block.get("x", 0.0)) - float(agent.get("x", 0.0)), float(block.get("y", 0.0)) - float(agent.get("y", 0.0)))
        coef = self.COEF_CONTACT if dist <= self.CONTACT_THRESHOLD else self.COEF_FREE
        delta = [sum(c * f for c, f in zip(row, feats)) for row in coef]
        agent["x"] = self._clip(float(agent.get("x", 0.0)) + delta[0])
        agent["y"] = self._clip(float(agent.get("y", 0.0)) + delta[1])
        agent["vx"] = delta[0]
        agent["vy"] = delta[1]
        block["x"] = self._clip(float(block.get("x", 0.0)) + delta[2])
        block["y"] = self._clip(float(block.get("y", 0.0)) + delta[3])
        block["angle"] = (float(block.get("angle", 0.0)) + delta[4]) % (2.0 * math.pi)
        b["step"] = int(b.get("step", 0) or 0) + 1
        self._recompute_relations(b)
        return b

    def _parse_action(self, action):
        match = re.match(r"^\w+\[dx=(-?\d+(?:\.\d+)?),dy=(-?\d+(?:\.\d+)?),speed=(-?\d+(?:\.\d+)?)\]$", str(action or "").strip())
        if not match:
            return 0.0, 0.0
        return float(match.group(1)), float(match.group(2))

    def _features(self, agent, block, dx, dy):
        sx = float(agent.get("x", 0.0))
        sy = float(agent.get("y", 0.0))
        bx = float(block.get("x", 0.0))
        by = float(block.get("y", 0.0))
        ba = float(block.get("angle", 0.0))
        speed = math.hypot(dx, dy)
        rel_x = bx - sx
        rel_y = by - sy
        dist = math.hypot(rel_x, rel_y)
        near = 1.0 if dist <= 70.0 else 0.0
        inv_dist = 1.0 / max(dist, 1.0)
        return [
            1.0,
            dx,
            dy,
            speed,
            sx / 512.0,
            sy / 512.0,
            bx / 512.0,
            by / 512.0,
            math.cos(ba),
            math.sin(ba),
            rel_x / 512.0,
            rel_y / 512.0,
            dist / 512.0,
            near,
            dx * near,
            dy * near,
            dx * rel_x * inv_dist,
            dy * rel_y * inv_dist,
        ]

    def _clip(self, value):
        return max(0.0, min(512.0, float(value)))

    def _recompute_relations(self, graph):
        objects = graph.get("objects", {})
        agent = objects.get("agent")
        block = objects.get("block")
        rels = {}
        if agent and block:
            dx = float(block.get("x", 0.0)) - float(agent.get("x", 0.0))
            dy = float(block.get("y", 0.0)) - float(agent.get("y", 0.0))
            dist = math.hypot(dx, dy)
            direction = ("right" if dx >= 0 else "left") if abs(dx) >= abs(dy) else ("up" if dy >= 0 else "down")
            rels["agent_distance_to_block"] = {"subject": "agent", "predicate": "distance_to", "object": "block", "value": dist, "key": "agent_distance_to_block"}
            rels["agent_near_block"] = {"subject": "agent", "predicate": "near", "object": "block", "value": float(dist <= 35.0), "key": "agent_near_block"}
            rels["agent_relative_direction_to_block"] = {"subject": "agent", "predicate": "relative_direction_to", "object": "block", "value": direction, "key": "agent_relative_direction_to_block"}
        graph["relations"] = rels

    def readout_observation(self, belief, action: str = ""):
        b = belief if isinstance(belief, dict) else self.init_belief()
        objects = b.get("objects", {})
        agent = objects.get("agent", {"id": "agent", "category": "circular_pusher", "x": 0.0, "y": 0.0, "vx": 0.0, "vy": 0.0, "conf": 1.0})
        block = objects.get("block", {"id": "block", "category": "t_block", "x": 0.0, "y": 0.0, "angle": 0.0, "conf": 1.0})
        self._recompute_relations(b)
        lines = [
            "Environment: pusht",
            f"Source: {b.get('source', 'scaffold_fit_contact')}",
            f"Step: {int(b.get('step', 0) or 0)}",
            f"Goal: {b.get('goal', 'move block to the target T pose')}",
            "Objects:",
            f"- agent:circular_pusher x={float(agent.get('x', 0.0)):.3f} y={float(agent.get('y', 0.0)):.3f} vx={float(agent.get('vx', 0.0)):.3f} vy={float(agent.get('vy', 0.0)):.3f} conf={float(agent.get('conf', 1.0)):.3f}",
            f"- block:t_block x={float(block.get('x', 0.0)):.3f} y={float(block.get('y', 0.0)):.3f} angle={float(block.get('angle', 0.0)):.3f} conf={float(block.get('conf', 1.0)):.3f}",
            "Relations:",
        ]
        for key in sorted(b.get("relations", {})):
            rel = b["relations"][key]
            value = rel.get("value")
            value_text = f"{float(value):.3f}" if isinstance(value, (int, float)) else str(value)
            lines.append(f"- {rel.get('subject')} {rel.get('predicate')} {rel.get('object')} = {value_text}")
        lines.append("Metadata: {}")
        return "\n".join(lines)
'''.strip() + "\n"


_SCAFFOLD_FIT_CONTACT_GATED_TEMPLATE = r'''
from patchworld.worldmodel_base import BaseWorldModel
import json
import math
import re
import copy


class PushtScaffoldFitContactGatedWorldModel(BaseWorldModel):
    CONTACT_THRESHOLD = __CONTACT_THRESHOLD__
    COEF_FREE = __COEF_FREE_JSON__
    COEF_CONTACT = __COEF_CONTACT_JSON__

    def init_belief(self):
        return {
            "env": "pusht",
            "source": "scaffold_fit_contact_gated",
            "step": 0,
            "goal": "move block to the target T pose",
            "objects": {},
            "relations": {},
            "metadata": {},
        }

    def init_belief_from_observation(self, obs_text: str):
        return self.correct_belief(self.init_belief(), obs_text)

    def extract_valid_action_forms(self):
        return {
            "move_left": ["move_left[dx=<float>,dy=<float>,speed=<float>]"],
            "move_right": ["move_right[dx=<float>,dy=<float>,speed=<float>]"],
            "move_up": ["move_up[dx=<float>,dy=<float>,speed=<float>]"],
            "move_down": ["move_down[dx=<float>,dy=<float>,speed=<float>]"],
            "hold": ["hold[dx=<float>,dy=<float>,speed=<float>]"],
        }

    def parse_observation(self, obs_text: str):
        graph = self.init_belief()
        section = ""
        for raw_line in str(obs_text or "").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("Environment:"):
                graph["env"] = line.split(":", 1)[1].strip()
                section = ""
            elif line.startswith("Source:"):
                graph["source"] = line.split(":", 1)[1].strip()
                section = ""
            elif line.startswith("Step:"):
                try:
                    graph["step"] = int(float(line.split(":", 1)[1].strip()))
                except Exception:
                    graph["step"] = 0
                section = ""
            elif line.startswith("Goal:"):
                graph["goal"] = line.split(":", 1)[1].strip()
                section = ""
            elif line.startswith("Objects:"):
                section = "objects"
            elif line.startswith("Relations:"):
                section = "relations"
            elif line.startswith("Metadata:"):
                try:
                    graph["metadata"] = json.loads(line.split(":", 1)[1].strip())
                except Exception:
                    graph["metadata"] = {}
                section = ""
            elif section == "objects" and line.startswith("-"):
                obj = self._parse_object(line)
                if obj:
                    graph["objects"][obj["id"]] = obj
            elif section == "relations" and line.startswith("-"):
                rel = self._parse_relation(line)
                if rel:
                    graph["relations"][rel["key"]] = rel
        self._recompute_relations(graph)
        return graph

    def _parse_object(self, line):
        match = re.match(r"^-\s*(\w+):(\w+)\s*(.*)$", line)
        if not match:
            return None
        obj = {"id": match.group(1), "category": match.group(2), "x": 0.0, "y": 0.0, "vx": 0.0, "vy": 0.0, "conf": 1.0}
        for key, value in re.findall(r"(\w+)=(-?\d+(?:\.\d+)?)", match.group(3)):
            obj[key] = float(value)
        return obj

    def _parse_relation(self, line):
        match = re.match(r"^-\s*(\w+)\s+(\w+(?:_\w+)*)\s+(\w+)\s*=\s*(.+)$", line)
        if not match:
            return None
        value = match.group(4).strip()
        try:
            value = float(value)
        except Exception:
            pass
        return {"subject": match.group(1), "predicate": match.group(2), "object": match.group(3), "value": value, "key": f"{match.group(1)}_{match.group(2)}_{match.group(3)}"}

    def correct_belief(self, belief_prior, obs_text: str):
        observed = self.parse_observation(obs_text)
        merged = copy.deepcopy(belief_prior) if isinstance(belief_prior, dict) else self.init_belief()
        for key in ("env", "source", "step", "goal", "metadata"):
            merged[key] = observed.get(key, merged.get(key))
        merged["objects"] = observed.get("objects", {})
        merged["relations"] = observed.get("relations", {})
        return merged

    def predict_belief(self, belief, action: str):
        b = copy.deepcopy(belief) if isinstance(belief, dict) else self.init_belief()
        objects = b.setdefault("objects", {})
        agent = objects.setdefault("agent", {"id": "agent", "category": "circular_pusher", "x": 0.0, "y": 0.0, "vx": 0.0, "vy": 0.0, "conf": 1.0})
        block = objects.setdefault("block", {"id": "block", "category": "t_block", "x": 0.0, "y": 0.0, "angle": 0.0, "conf": 1.0})
        dx, dy = self._parse_action(action)
        feats = self._features(agent, block, dx, dy)
        dist = math.hypot(float(block.get("x", 0.0)) - float(agent.get("x", 0.0)), float(block.get("y", 0.0)) - float(agent.get("y", 0.0)))
        in_contact = dist <= self.CONTACT_THRESHOLD
        coef = self.COEF_CONTACT if in_contact else self.COEF_FREE
        delta = [sum(c * f for c, f in zip(row, feats)) for row in coef]
        if not in_contact:
            delta[2] = 0.0
            delta[3] = 0.0
            delta[4] = 0.0
        agent["x"] = self._clip(float(agent.get("x", 0.0)) + delta[0])
        agent["y"] = self._clip(float(agent.get("y", 0.0)) + delta[1])
        agent["vx"] = delta[0]
        agent["vy"] = delta[1]
        block["x"] = self._clip(float(block.get("x", 0.0)) + delta[2])
        block["y"] = self._clip(float(block.get("y", 0.0)) + delta[3])
        block["angle"] = (float(block.get("angle", 0.0)) + delta[4]) % (2.0 * math.pi)
        b["step"] = int(b.get("step", 0) or 0) + 1
        self._recompute_relations(b)
        return b

    def _parse_action(self, action):
        match = re.match(r"^\w+\[dx=(-?\d+(?:\.\d+)?),dy=(-?\d+(?:\.\d+)?),speed=(-?\d+(?:\.\d+)?)\]$", str(action or "").strip())
        if not match:
            return 0.0, 0.0
        return float(match.group(1)), float(match.group(2))

    def _features(self, agent, block, dx, dy):
        sx = float(agent.get("x", 0.0))
        sy = float(agent.get("y", 0.0))
        bx = float(block.get("x", 0.0))
        by = float(block.get("y", 0.0))
        ba = float(block.get("angle", 0.0))
        speed = math.hypot(dx, dy)
        rel_x = bx - sx
        rel_y = by - sy
        dist = math.hypot(rel_x, rel_y)
        near = 1.0 if dist <= 70.0 else 0.0
        inv_dist = 1.0 / max(dist, 1.0)
        return [
            1.0, dx, dy, speed,
            sx / 512.0, sy / 512.0, bx / 512.0, by / 512.0,
            math.cos(ba), math.sin(ba),
            rel_x / 512.0, rel_y / 512.0, dist / 512.0,
            near, dx * near, dy * near,
            dx * rel_x * inv_dist, dy * rel_y * inv_dist,
        ]

    def _clip(self, value):
        return max(0.0, min(512.0, float(value)))

    def _recompute_relations(self, graph):
        objects = graph.get("objects", {})
        agent = objects.get("agent")
        block = objects.get("block")
        rels = {}
        if agent and block:
            dx = float(block.get("x", 0.0)) - float(agent.get("x", 0.0))
            dy = float(block.get("y", 0.0)) - float(agent.get("y", 0.0))
            dist = math.hypot(dx, dy)
            direction = ("right" if dx >= 0 else "left") if abs(dx) >= abs(dy) else ("up" if dy >= 0 else "down")
            rels["agent_distance_to_block"] = {"subject": "agent", "predicate": "distance_to", "object": "block", "value": dist, "key": "agent_distance_to_block"}
            rels["agent_near_block"] = {"subject": "agent", "predicate": "near", "object": "block", "value": float(dist <= 35.0), "key": "agent_near_block"}
            rels["agent_relative_direction_to_block"] = {"subject": "agent", "predicate": "relative_direction_to", "object": "block", "value": direction, "key": "agent_relative_direction_to_block"}
        graph["relations"] = rels

    def readout_observation(self, belief, action: str = ""):
        b = belief if isinstance(belief, dict) else self.init_belief()
        objects = b.get("objects", {})
        agent = objects.get("agent", {"id": "agent", "category": "circular_pusher", "x": 0.0, "y": 0.0, "vx": 0.0, "vy": 0.0, "conf": 1.0})
        block = objects.get("block", {"id": "block", "category": "t_block", "x": 0.0, "y": 0.0, "angle": 0.0, "conf": 1.0})
        self._recompute_relations(b)
        lines = [
            "Environment: pusht",
            f"Source: {b.get('source', 'scaffold_fit_contact_gated')}",
            f"Step: {int(b.get('step', 0) or 0)}",
            f"Goal: {b.get('goal', 'move block to the target T pose')}",
            "Objects:",
            f"- agent:circular_pusher x={float(agent.get('x', 0.0)):.3f} y={float(agent.get('y', 0.0)):.3f} vx={float(agent.get('vx', 0.0)):.3f} vy={float(agent.get('vy', 0.0)):.3f} conf={float(agent.get('conf', 1.0)):.3f}",
            f"- block:t_block x={float(block.get('x', 0.0)):.3f} y={float(block.get('y', 0.0)):.3f} angle={float(block.get('angle', 0.0)):.3f} conf={float(block.get('conf', 1.0)):.3f}",
            "Relations:",
        ]
        for key in sorted(b.get("relations", {})):
            rel = b["relations"][key]
            value = rel.get("value")
            value_text = f"{float(value):.3f}" if isinstance(value, (int, float)) else str(value)
            lines.append(f"- {rel.get('subject')} {rel.get('predicate')} {rel.get('object')} = {value_text}")
        lines.append("Metadata: {}")
        return "\n".join(lines)
'''.strip() + "\n"


_SCAFFOLD_FIT_CONTACT_GATED_MLP_TEMPLATE = r'''
from patchworld.worldmodel_base import BaseWorldModel
import json
import math
import re
import copy


class PushtScaffoldFitContactGatedMlpWorldModel(BaseWorldModel):
    CONTACT_THRESHOLD = __CONTACT_THRESHOLD__
    COEF_FREE = __COEF_FREE_JSON__
    COEF_CONTACT = __COEF_CONTACT_JSON__
    MLP = __MLP_JSON__

    def init_belief(self):
        return {
            "env": "pusht",
            "source": "scaffold_fit_contact_gated_mlp",
            "step": 0,
            "goal": "move block to the target T pose",
            "objects": {},
            "relations": {},
            "metadata": {},
        }

    def init_belief_from_observation(self, obs_text: str):
        return self.correct_belief(self.init_belief(), obs_text)

    def extract_valid_action_forms(self):
        return {
            "move_left": ["move_left[dx=<float>,dy=<float>,speed=<float>]"],
            "move_right": ["move_right[dx=<float>,dy=<float>,speed=<float>]"],
            "move_up": ["move_up[dx=<float>,dy=<float>,speed=<float>]"],
            "move_down": ["move_down[dx=<float>,dy=<float>,speed=<float>]"],
            "hold": ["hold[dx=<float>,dy=<float>,speed=<float>]"],
        }

    def parse_observation(self, obs_text: str):
        graph = self.init_belief()
        section = ""
        for raw_line in str(obs_text or "").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("Environment:"):
                graph["env"] = line.split(":", 1)[1].strip()
                section = ""
            elif line.startswith("Source:"):
                graph["source"] = line.split(":", 1)[1].strip()
                section = ""
            elif line.startswith("Step:"):
                try:
                    graph["step"] = int(float(line.split(":", 1)[1].strip()))
                except Exception:
                    graph["step"] = 0
                section = ""
            elif line.startswith("Goal:"):
                graph["goal"] = line.split(":", 1)[1].strip()
                section = ""
            elif line.startswith("Objects:"):
                section = "objects"
            elif line.startswith("Relations:"):
                section = "relations"
            elif line.startswith("Metadata:"):
                try:
                    graph["metadata"] = json.loads(line.split(":", 1)[1].strip())
                except Exception:
                    graph["metadata"] = {}
                section = ""
            elif section == "objects" and line.startswith("-"):
                obj = self._parse_object(line)
                if obj:
                    graph["objects"][obj["id"]] = obj
            elif section == "relations" and line.startswith("-"):
                rel = self._parse_relation(line)
                if rel:
                    graph["relations"][rel["key"]] = rel
        self._recompute_relations(graph)
        return graph

    def _parse_object(self, line):
        match = re.match(r"^-\s*(\w+):(\w+)\s*(.*)$", line)
        if not match:
            return None
        obj = {"id": match.group(1), "category": match.group(2), "x": 0.0, "y": 0.0, "vx": 0.0, "vy": 0.0, "conf": 1.0}
        for key, value in re.findall(r"(\w+)=(-?\d+(?:\.\d+)?)", match.group(3)):
            obj[key] = float(value)
        return obj

    def _parse_relation(self, line):
        match = re.match(r"^-\s*(\w+)\s+(\w+(?:_\w+)*)\s+(\w+)\s*=\s*(.+)$", line)
        if not match:
            return None
        value = match.group(4).strip()
        try:
            value = float(value)
        except Exception:
            pass
        return {"subject": match.group(1), "predicate": match.group(2), "object": match.group(3), "value": value, "key": f"{match.group(1)}_{match.group(2)}_{match.group(3)}"}

    def correct_belief(self, belief_prior, obs_text: str):
        observed = self.parse_observation(obs_text)
        merged = copy.deepcopy(belief_prior) if isinstance(belief_prior, dict) else self.init_belief()
        for key in ("env", "source", "step", "goal", "metadata"):
            merged[key] = observed.get(key, merged.get(key))
        merged["objects"] = observed.get("objects", {})
        merged["relations"] = observed.get("relations", {})
        return merged

    def predict_belief(self, belief, action: str):
        b = copy.deepcopy(belief) if isinstance(belief, dict) else self.init_belief()
        objects = b.setdefault("objects", {})
        agent = objects.setdefault("agent", {"id": "agent", "category": "circular_pusher", "x": 0.0, "y": 0.0, "vx": 0.0, "vy": 0.0, "conf": 1.0})
        block = objects.setdefault("block", {"id": "block", "category": "t_block", "x": 0.0, "y": 0.0, "angle": 0.0, "conf": 1.0})
        dx, dy = self._parse_action(action)
        feats = self._features(agent, block, dx, dy)
        dist = math.hypot(float(block.get("x", 0.0)) - float(agent.get("x", 0.0)), float(block.get("y", 0.0)) - float(agent.get("y", 0.0)))
        in_contact = dist <= self.CONTACT_THRESHOLD
        coef = self.COEF_CONTACT if in_contact else self.COEF_FREE
        delta = [sum(c * f for c, f in zip(row, feats)) for row in coef]
        if in_contact:
            residual = self._mlp_block_residual(agent, block, dx, dy)
            delta[2] += residual[0]
            delta[3] += residual[1]
            delta[4] += residual[2]
        else:
            delta[2] = 0.0
            delta[3] = 0.0
            delta[4] = 0.0
        agent["x"] = self._clip(float(agent.get("x", 0.0)) + delta[0])
        agent["y"] = self._clip(float(agent.get("y", 0.0)) + delta[1])
        agent["vx"] = delta[0]
        agent["vy"] = delta[1]
        block["x"] = self._clip(float(block.get("x", 0.0)) + delta[2])
        block["y"] = self._clip(float(block.get("y", 0.0)) + delta[3])
        block["angle"] = (float(block.get("angle", 0.0)) + delta[4]) % (2.0 * math.pi)
        b["step"] = int(b.get("step", 0) or 0) + 1
        self._recompute_relations(b)
        return b

    def _mlp_block_residual(self, agent, block, dx, dy):
        compact = self._compact_features(agent, block, dx, dy)
        hidden = []
        for row, bias in zip(self.MLP["w1"], self.MLP["b1"]):
            hidden.append(max(0.0, sum(w * x for w, x in zip(row, compact)) + bias))
        outputs = []
        for row, bias in zip(self.MLP["w2"], self.MLP["b2"]):
            outputs.append(sum(w * h for w, h in zip(row, hidden)) + bias)
        return outputs

    def _compact_features(self, agent, block, dx, dy):
        sx = float(agent.get("x", 0.0))
        sy = float(agent.get("y", 0.0))
        bx = float(block.get("x", 0.0))
        by = float(block.get("y", 0.0))
        ba = float(block.get("angle", 0.0))
        speed = math.hypot(dx, dy)
        dist = math.hypot(bx - sx, by - sy)
        return [sx / 512.0, sy / 512.0, bx / 512.0, by / 512.0, math.cos(ba), math.sin(ba), dist / 512.0, dx, dy, speed]

    def _parse_action(self, action):
        match = re.match(r"^\w+\[dx=(-?\d+(?:\.\d+)?),dy=(-?\d+(?:\.\d+)?),speed=(-?\d+(?:\.\d+)?)\]$", str(action or "").strip())
        if not match:
            return 0.0, 0.0
        return float(match.group(1)), float(match.group(2))

    def _features(self, agent, block, dx, dy):
        sx = float(agent.get("x", 0.0))
        sy = float(agent.get("y", 0.0))
        bx = float(block.get("x", 0.0))
        by = float(block.get("y", 0.0))
        ba = float(block.get("angle", 0.0))
        speed = math.hypot(dx, dy)
        rel_x = bx - sx
        rel_y = by - sy
        dist = math.hypot(rel_x, rel_y)
        near = 1.0 if dist <= 70.0 else 0.0
        inv_dist = 1.0 / max(dist, 1.0)
        return [
            1.0, dx, dy, speed,
            sx / 512.0, sy / 512.0, bx / 512.0, by / 512.0,
            math.cos(ba), math.sin(ba),
            rel_x / 512.0, rel_y / 512.0, dist / 512.0,
            near, dx * near, dy * near,
            dx * rel_x * inv_dist, dy * rel_y * inv_dist,
        ]

    def _clip(self, value):
        return max(0.0, min(512.0, float(value)))

    def _recompute_relations(self, graph):
        objects = graph.get("objects", {})
        agent = objects.get("agent")
        block = objects.get("block")
        rels = {}
        if agent and block:
            dx = float(block.get("x", 0.0)) - float(agent.get("x", 0.0))
            dy = float(block.get("y", 0.0)) - float(agent.get("y", 0.0))
            dist = math.hypot(dx, dy)
            direction = ("right" if dx >= 0 else "left") if abs(dx) >= abs(dy) else ("up" if dy >= 0 else "down")
            rels["agent_distance_to_block"] = {"subject": "agent", "predicate": "distance_to", "object": "block", "value": dist, "key": "agent_distance_to_block"}
            rels["agent_near_block"] = {"subject": "agent", "predicate": "near", "object": "block", "value": float(dist <= 35.0), "key": "agent_near_block"}
            rels["agent_relative_direction_to_block"] = {"subject": "agent", "predicate": "relative_direction_to", "object": "block", "value": direction, "key": "agent_relative_direction_to_block"}
        graph["relations"] = rels

    def readout_observation(self, belief, action: str = ""):
        b = belief if isinstance(belief, dict) else self.init_belief()
        objects = b.get("objects", {})
        agent = objects.get("agent", {"id": "agent", "category": "circular_pusher", "x": 0.0, "y": 0.0, "vx": 0.0, "vy": 0.0, "conf": 1.0})
        block = objects.get("block", {"id": "block", "category": "t_block", "x": 0.0, "y": 0.0, "angle": 0.0, "conf": 1.0})
        self._recompute_relations(b)
        lines = [
            "Environment: pusht",
            f"Source: {b.get('source', 'scaffold_fit_contact_gated_mlp')}",
            f"Step: {int(b.get('step', 0) or 0)}",
            f"Goal: {b.get('goal', 'move block to the target T pose')}",
            "Objects:",
            f"- agent:circular_pusher x={float(agent.get('x', 0.0)):.3f} y={float(agent.get('y', 0.0)):.3f} vx={float(agent.get('vx', 0.0)):.3f} vy={float(agent.get('vy', 0.0)):.3f} conf={float(agent.get('conf', 1.0)):.3f}",
            f"- block:t_block x={float(block.get('x', 0.0)):.3f} y={float(block.get('y', 0.0)):.3f} angle={float(block.get('angle', 0.0)):.3f} conf={float(block.get('conf', 1.0)):.3f}",
            "Relations:",
        ]
        for key in sorted(b.get("relations", {})):
            rel = b["relations"][key]
            value = rel.get("value")
            value_text = f"{float(value):.3f}" if isinstance(value, (int, float)) else str(value)
            lines.append(f"- {rel.get('subject')} {rel.get('predicate')} {rel.get('object')} = {value_text}")
        lines.append("Metadata: {}")
        return "\n".join(lines)
'''.strip() + "\n"
