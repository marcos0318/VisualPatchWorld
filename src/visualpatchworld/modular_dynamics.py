"""Modular gated+MLP dynamics fitting with per-module refit."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .pusht_local_induction import (
    WeightedTransition,
    _SCAFFOLD_FIT_CONTACT_GATED_MLP_TEMPLATE,
    _angle_delta,
    _feature_vector,
    _fit_weighted_ridge_matrix,
    _parse_action,
    _parse_observation_text,
    _state_from_graph,
    _zero_coef_block_rows,
    fit_contact_mlp_residual,
    fit_pusht_scaffold_contact_gated_mlp_params,
    iter_contact_multistep_pairs,
    iter_weighted_transitions,
)
from .pusht_physics_rules import evaluate_model_rule_compliance


def _fit_weighted_ridge_rows(
    features: Sequence[Sequence[float]],
    targets: Sequence[Sequence[float]],
    weights: Sequence[float],
    row_indices: Sequence[int],
    base_coef: Sequence[Sequence[float]],
    *,
    ridge: float,
) -> List[List[float]]:
    coef = [list(row) for row in base_coef]
    for row_idx in row_indices:
        if row_idx >= len(coef):
            continue
        scalar_targets = [[target[row_idx]] for target in targets]
        fitted = _fit_weighted_ridge_matrix(features, scalar_targets, weights, ridge=ridge)
        coef[row_idx] = fitted[0]
    return coef


def _contact_distance(observation: str) -> float:
    state = _state_from_graph(_parse_observation_text(observation))
    return float(math.hypot(state[2] - state[0], state[3] - state[1]))


def _rows_for_contact_bucket(
    rows: Sequence[WeightedTransition],
    contact_threshold: float,
) -> List[WeightedTransition]:
    return [row for row in rows if _contact_distance(row.observation) <= contact_threshold]


def _rows_for_free_bucket(
    rows: Sequence[WeightedTransition],
    contact_threshold: float,
) -> List[WeightedTransition]:
    return [row for row in rows if _contact_distance(row.observation) > contact_threshold]


def _rows_to_fit_inputs(
    rows: Sequence[WeightedTransition],
) -> Tuple[List[List[float]], List[List[float]], List[float]]:
    features: List[List[float]] = []
    targets: List[List[float]] = []
    weights: List[float] = []
    for row in rows:
        state = _state_from_graph(_parse_observation_text(row.observation))
        next_state = _state_from_graph(_parse_observation_text(row.next_observation))
        action = _parse_action(row.action)
        sx, sy, bx, by, ba = state
        nsx, nsy, nbx, nby, nba = next_state
        features.append(_feature_vector(state, action))
        targets.append([nsx - sx, nsy - sy, nbx - bx, nby - by, _angle_delta(nba, ba)])
        weights.append(float(row.weight))
    return features, targets, weights


def load_gated_mlp_params_from_results_dir(results_dir: Path | str) -> Dict[str, Any]:
    """Load CONTACT_THRESHOLD / COEF_* / MLP from the newest generated model."""

    root = Path(results_dir)
    candidates: List[Path] = []
    for subdir in ("", "tool", "oracle"):
        model_dir = root / subdir / "generated_models" if subdir else root / "generated_models"
        if model_dir.is_dir():
            candidates.extend(model_dir.glob("*.py"))
    if not candidates:
        raise FileNotFoundError(f"No generated model .py files under {root} (or {root}/tool)")
    model_path = sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)[0]
    code = model_path.read_text(encoding="utf-8")
    return load_gated_mlp_params_from_code(code)


def load_gated_mlp_params_from_code(code: str) -> Dict[str, Any]:
    from visualpatchworld.patchworld_adapter import apply_pusht_patchworld_adapter, sanitize_pusht_model_code

    apply_pusht_patchworld_adapter()
    from patchworld.worldmodel_validator import load_model_from_code

    model, error = load_model_from_code(sanitize_pusht_model_code(code))
    if error or model is None:
        raise ValueError(f"Could not load model to extract params: {error}")
    for attr in ("CONTACT_THRESHOLD", "COEF_FREE", "COEF_CONTACT", "MLP"):
        if not hasattr(model, attr):
            raise ValueError(f"Model is missing required attribute {attr}")
    return {
        "contact_threshold": float(model.CONTACT_THRESHOLD),
        "coef_free": model.COEF_FREE,
        "coef_contact": model.COEF_CONTACT,
        "mlp": model.MLP,
    }


def fit_modular_offline_gated_mlp_params(
    trajectories: Sequence[Any],
    *,
    ridge: float = 1e-3,
    contact_threshold: float = 70.0,
    min_bucket: int = 3,
    mlp_hidden_dim: int = 64,
    mlp_steps: int = 3000,
    mlp_lr: float = 0.01,
    mlp_ridge: float = 1e-4,
    mlp_rollout_steps: int = 2,
    multistep_weight: float = 0.5,
    seed: int = 42,
) -> Dict[str, Any]:
    """Offline gated+MLP fit from trajectories only (no counterexamples)."""

    params = fit_pusht_scaffold_contact_gated_mlp_params(
        trajectories,
        ridge=ridge,
        contact_threshold=contact_threshold,
        min_bucket=min_bucket,
        weighted_transitions=None,
        mlp_hidden_dim=mlp_hidden_dim,
        mlp_steps=mlp_steps,
        mlp_lr=mlp_lr,
        mlp_ridge=mlp_ridge,
        mlp_rollout_steps=mlp_rollout_steps,
        multistep_weight=multistep_weight,
        seed=seed,
    )
    params["mlp_rollout_steps"] = mlp_rollout_steps
    return params


def refit_contact_module_params(
    base_params: Mapping[str, Any],
    trajectories: Sequence[Any],
    extra_transitions: Sequence[WeightedTransition],
    *,
    ridge: float = 1e-3,
    mlp_hidden_dim: int = 64,
    mlp_steps: int = 3000,
    mlp_lr: float = 0.01,
    mlp_ridge: float = 1e-4,
    mlp_rollout_steps: int = 2,
    multistep_weight: float = 0.5,
    seed: int = 42,
) -> Dict[str, Any]:
    """Refit contact block rows + MLP; freeze free head, gate, and agent rows."""

    contact_threshold = float(base_params["contact_threshold"])
    coef_free = _zero_coef_block_rows(base_params["coef_free"])
    coef_contact = [list(row) for row in base_params["coef_contact"]]

    base_rows = iter_weighted_transitions(trajectories, default_weight=1.0)
    merged = list(base_rows) + list(extra_transitions)
    contact_rows = _rows_for_contact_bucket(merged, contact_threshold)
    if not contact_rows:
        contact_rows = list(base_rows)

    features, targets, weights = _rows_to_fit_inputs(contact_rows)
    if features:
        coef_contact = _fit_weighted_ridge_rows(
            features,
            targets,
            weights,
            row_indices=(2, 3, 4),
            base_coef=coef_contact,
            ridge=ridge,
        )

    multistep_pairs = (
        iter_contact_multistep_pairs(trajectories, contact_threshold=contact_threshold)
        if mlp_rollout_steps >= 2
        else None
    )
    mlp = fit_contact_mlp_residual(
        contact_rows,
        coef_contact,
        contact_threshold=contact_threshold,
        hidden_dim=mlp_hidden_dim,
        ridge=mlp_ridge,
        steps=mlp_steps,
        lr=mlp_lr,
        seed=seed,
        multistep_pairs=multistep_pairs,
        multistep_weight=multistep_weight,
    )
    return {
        "contact_threshold": contact_threshold,
        "coef_free": coef_free,
        "coef_contact": coef_contact,
        "mlp": mlp,
        "n_base_rows": len(base_rows),
        "n_extra_rows": len(extra_transitions),
        "n_contact_rows": len(contact_rows),
        "mlp_rollout_steps": mlp_rollout_steps,
    }


def refit_agent_module_params(
    base_params: Mapping[str, Any],
    trajectories: Sequence[Any],
    extra_transitions: Sequence[WeightedTransition],
    *,
    ridge: float = 1e-3,
) -> Dict[str, Any]:
    """Refit agent rows of free/contact heads; freeze block rows, gate, and MLP."""

    contact_threshold = float(base_params["contact_threshold"])
    coef_free = _zero_coef_block_rows(base_params["coef_free"])
    coef_contact = [list(row) for row in base_params["coef_contact"]]
    mlp = dict(base_params["mlp"])

    base_rows = iter_weighted_transitions(trajectories, default_weight=1.0)
    merged = list(base_rows) + list(extra_transitions)

    contact_rows = _rows_for_contact_bucket(merged, contact_threshold)
    free_rows = _rows_for_free_bucket(merged, contact_threshold)
    if not free_rows:
        free_rows = list(base_rows)

    free_features, free_targets, free_weights = _rows_to_fit_inputs(free_rows)
    if free_features:
        coef_free = _fit_weighted_ridge_rows(
            free_features,
            free_targets,
            free_weights,
            row_indices=(0, 1),
            base_coef=coef_free,
            ridge=ridge,
        )

    contact_features, contact_targets, contact_weights = _rows_to_fit_inputs(contact_rows)
    if contact_features:
        coef_contact = _fit_weighted_ridge_rows(
            contact_features,
            contact_targets,
            contact_weights,
            row_indices=(0, 1),
            base_coef=coef_contact,
            ridge=ridge,
        )

    return {
        "contact_threshold": contact_threshold,
        "coef_free": coef_free,
        "coef_contact": coef_contact,
        "mlp": mlp,
        "n_base_rows": len(base_rows),
        "n_extra_rows": len(extra_transitions),
        "n_free_rows": len(free_rows),
        "n_contact_rows": len(contact_rows),
    }


def render_gated_mlp_model_code(params: Mapping[str, Any]) -> str:
    return (
        _SCAFFOLD_FIT_CONTACT_GATED_MLP_TEMPLATE.replace(
            "__CONTACT_THRESHOLD__",
            repr(float(params["contact_threshold"])),
        )
        .replace("__COEF_FREE_JSON__", json.dumps(params["coef_free"], sort_keys=True))
        .replace("__COEF_CONTACT_JSON__", json.dumps(params["coef_contact"], sort_keys=True))
        .replace("__MLP_JSON__", json.dumps(params["mlp"], sort_keys=True))
    )


def generate_modular_offline_gated_mlp_model_code(trajectories: Sequence[Any], **kwargs: Any) -> str:
    params = fit_modular_offline_gated_mlp_params(trajectories, **kwargs)
    return render_gated_mlp_model_code(params)


def generate_modular_contact_refit_model_code(
    trajectories: Sequence[Any],
    *,
    base_params: Mapping[str, Any],
    extra_transitions: Sequence[WeightedTransition],
    **kwargs: Any,
) -> str:
    params = refit_contact_module_params(
        base_params,
        trajectories,
        extra_transitions,
        **kwargs,
    )
    return render_gated_mlp_model_code(params)


def generate_modular_agent_refit_model_code(
    trajectories: Sequence[Any],
    *,
    base_params: Mapping[str, Any],
    extra_transitions: Sequence[WeightedTransition],
    **kwargs: Any,
) -> str:
    params = refit_agent_module_params(
        base_params,
        trajectories,
        extra_transitions,
        **kwargs,
    )
    return render_gated_mlp_model_code(params)


def evaluate_module_acceptance(
    model: Any,
    trajectories: Sequence[Any],
    *,
    contact_threshold: float = 70.0,
) -> Dict[str, Any]:
    """Module-level metrics before full planning evaluation."""

    from .pusht_physics_rules import stratified_position_errors

    rules = evaluate_model_rule_compliance(model, trajectories)
    stratified = stratified_position_errors(trajectories, contact_threshold=contact_threshold)
    r1_stats = rules.get("rules", {}).get("R1_no_block_motion_when_far", {})
    r1_rate = float(r1_stats.get("compliance_rate", 0.0)) if r1_stats else None
    near = stratified.get("near_contact", {})
    return {
        "r1_pass_rate": r1_rate,
        "near_contact_block_error_mean": near.get("block_position_error_mean"),
        "near_contact_agent_error_mean": near.get("agent_position_error_mean"),
        "near_contact_count": near.get("count", 0),
        "physics_summary": rules,
        "stratified_errors": stratified,
    }


def base_transition_mass(trajectories: Sequence[Any]) -> float:
    return float(len(iter_weighted_transitions(trajectories, default_weight=1.0)))
