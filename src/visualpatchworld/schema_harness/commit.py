"""commit_actions with soft void-on-surprise (Schema: reality outranks the model)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

from visualpatchworld.lewm_env_specs import LeWMEnvSpec
from visualpatchworld.lewm_graph import (
    action_string_from_vector,
    graph_text_from_row,
    sim_row_from_obs,
    sim_step_with_info,
)

from .backtest import step_residual
from .timeline import Timeline


@dataclass
class CommitResult:
    actions_executed: int
    voided: bool
    surprise_residuals: List[float]
    final_obs_text: str
    final_row: Dict[str, Any]
    steps_taken_macros: int
    surprise_count: int = 0


def commit_actions(
    *,
    env: Any,
    spec: LeWMEnvSpec,
    model: Any,
    timeline: Timeline,
    current_obs: str,
    current_row: Mapping[str, Any],
    action_matrix: np.ndarray,
    frameskip: int,
    residual_epsilon: float,
    soft_void: bool,
    source: str = "live",
) -> CommitResult:
    """Execute planned macros.

    Soft void (Schema-adopted): on first residual > epsilon, stop remaining macros
    in this chunk and tag the transition as a surprise. Does NOT revoke planning
    permission for the accepted model — caller should replan with last_accepted.
    """
    row = dict(current_row)
    obs_text = str(current_obs)
    residuals: List[float] = []
    executed = 0
    voided = False
    surprise_count = 0

    for action in np.asarray(action_matrix):
        action_str = action_string_from_vector(action, spec)
        obs = None
        info: Mapping[str, Any] = {}
        for _ in range(max(1, frameskip)):
            obs, info = sim_step_with_info(env, spec, action)
        next_row = sim_row_from_obs(spec.name, obs, template_row=row, info=info)
        next_obs = graph_text_from_row(spec.name, next_row, source="oracle", step=executed + 1)

        resid = step_residual(model, obs_text, action_str, next_obs) if model is not None else 0.0
        residuals.append(float(resid))
        is_surprise = bool(model is not None and resid > residual_epsilon)
        tag = "surprise" if is_surprise else source
        weight = float(min(10.0, 1.0 + resid)) if is_surprise else 1.0
        timeline.append(
            obs_text,
            action_str,
            next_obs,
            source=tag,
            residual=float(resid),
            weight=weight,
        )
        executed += 1
        row = next_row
        obs_text = next_obs

        if is_surprise:
            surprise_count += 1
            if soft_void:
                voided = True
                timeline.add_note(f"soft_void residual={resid:.4f} > eps={residual_epsilon}")
                break

    return CommitResult(
        actions_executed=executed,
        voided=voided,
        surprise_residuals=residuals,
        final_obs_text=obs_text,
        final_row=row,
        steps_taken_macros=executed,
        surprise_count=surprise_count,
    )


def discovery_probe_actions(
    spec: LeWMEnvSpec,
    rng: np.random.Generator,
    *,
    n_macros: int = 4,
) -> np.ndarray:
    """Orthogonal probes for structure discovery (Schema: action for discovery)."""
    actions = rng.uniform(-1.0, 1.0, size=(n_macros, spec.action_dim))
    if spec.action_dim == 2 and n_macros >= 4:
        actions[:4] = np.array(
            [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]],
            dtype=np.float64,
        )
    return np.clip(actions, -1.0, 1.0)
