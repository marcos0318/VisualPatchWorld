"""Residual backtest / certification for continuous LeWM world models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
from patchworld.worldmodel_data import Trajectory
from patchworld.worldmodel_validator import load_model_from_code

from visualpatchworld.code_baselines.common import sanitize_generated_code, state_vector_from_belief


@dataclass
class BacktestReport:
    certified: bool
    n_checked: int
    n_hard_fail: int
    n_residual_fail: int
    pass_rate: float
    mean_residual: float
    median_residual: float
    max_residual: float
    load_error: Optional[str] = None
    hard_errors: List[Dict[str, Any]] = field(default_factory=list)
    residual_epsilon: float = 0.5
    pass_threshold: float = 0.8

    def to_dict(self) -> Dict[str, Any]:
        return {
            "certified": self.certified,
            "n_checked": self.n_checked,
            "n_hard_fail": self.n_hard_fail,
            "n_residual_fail": self.n_residual_fail,
            "pass_rate": self.pass_rate,
            "mean_residual": self.mean_residual,
            "median_residual": self.median_residual,
            "max_residual": self.max_residual,
            "load_error": self.load_error,
            "residual_epsilon": self.residual_epsilon,
            "pass_threshold": self.pass_threshold,
            "hard_errors": self.hard_errors[:8],
        }


def _pose_residual(pred: Mapping[str, Any], tgt: Mapping[str, Any]) -> float:
    _, pred_vals = state_vector_from_belief(pred)
    _, tgt_vals = state_vector_from_belief(tgt)
    if not pred_vals or not tgt_vals or len(pred_vals) != len(tgt_vals):
        return 1e3
    return float(np.linalg.norm(np.asarray(pred_vals) - np.asarray(tgt_vals)))


def residual_backtest(
    code: str,
    trajectories: Sequence[Trajectory],
    *,
    epsilon: float = 0.5,
    pass_threshold: float = 0.8,
    max_transitions: int = 0,
) -> BacktestReport:
    """Certify a world-model program against Timeline history.

    Hard fails: load / parse / predict exceptions.
    Soft fails: pose residual > epsilon.
    Certified iff no hard fails and soft pass_rate >= pass_threshold.

    Continuous LeWM residuals are heavy-tailed, so defaults use a moderate
    epsilon with an 80% pass rate rather than ARC-style exact match.
    """
    sanitized = sanitize_generated_code(code)
    model, load_error = load_model_from_code(sanitized)
    if model is None or load_error:
        return BacktestReport(
            certified=False,
            n_checked=0,
            n_hard_fail=1,
            n_residual_fail=0,
            pass_rate=0.0,
            mean_residual=float("inf"),
            median_residual=float("inf"),
            max_residual=float("inf"),
            load_error=load_error or "load failed",
            hard_errors=[{"error_type": "load_error", "error": load_error}],
            residual_epsilon=epsilon,
            pass_threshold=pass_threshold,
        )

    hard_errors: List[Dict[str, Any]] = []
    residuals: List[float] = []
    n_checked = 0
    n_hard = 0
    n_soft = 0

    for traj in trajectories:
        for step in traj.transitions:
            if max_transitions > 0 and n_checked >= max_transitions:
                break
            n_checked += 1
            try:
                belief = model.init_belief_from_observation(step.observation)
                predicted = model.predict_belief(belief, step.action)
                if not isinstance(predicted, dict) or not predicted.get("objects"):
                    readout = model.readout_observation(predicted, step.action)
                    predicted = model.parse_observation(readout)
                actual = model.parse_observation(step.next_observation)
            except Exception as exc:
                n_hard += 1
                hard_errors.append(
                    {
                        "error_type": "predict_or_parse_error",
                        "error": str(exc),
                        "action": step.action,
                    }
                )
                residuals.append(1e3)
                continue

            resid = _pose_residual(predicted, actual)
            residuals.append(resid)
            if resid > epsilon:
                n_soft += 1
        if max_transitions > 0 and n_checked >= max_transitions:
            break

    if n_checked == 0:
        return BacktestReport(
            certified=False,
            n_checked=0,
            n_hard_fail=0,
            n_residual_fail=0,
            pass_rate=0.0,
            mean_residual=float("inf"),
            median_residual=float("inf"),
            max_residual=float("inf"),
            load_error="empty timeline",
            residual_epsilon=epsilon,
            pass_threshold=pass_threshold,
        )

    n_pass = sum(1 for r in residuals if r <= epsilon)
    pass_rate = float(n_pass) / float(n_checked)
    certified = n_hard == 0 and pass_rate >= pass_threshold
    return BacktestReport(
        certified=certified,
        n_checked=n_checked,
        n_hard_fail=n_hard,
        n_residual_fail=n_soft,
        pass_rate=pass_rate,
        mean_residual=float(np.mean(residuals)),
        median_residual=float(np.median(residuals)),
        max_residual=float(np.max(residuals)),
        hard_errors=hard_errors,
        residual_epsilon=epsilon,
        pass_threshold=pass_threshold,
    )


def step_residual(model: Any, observation: str, action: str, next_observation: str) -> float:
    """One-step pose residual for commit-time surprise checks."""
    try:
        belief = model.init_belief_from_observation(observation)
        predicted = model.predict_belief(belief, action)
        if not isinstance(predicted, dict) or not predicted.get("objects"):
            readout = model.readout_observation(predicted, action)
            predicted = model.parse_observation(readout)
        actual = model.parse_observation(next_observation)
        return _pose_residual(predicted, actual)
    except Exception:
        return 1e3


def should_promote(
    candidate: BacktestReport,
    accepted: Optional[BacktestReport],
    *,
    min_pass_rate: float = 0.8,
    require_hard_clean: bool = True,
) -> bool:
    """Promotion-only certification (Schema certify ≠ plan permission).

    A candidate may replace accepted only if it is hard-clean and improves
    median residual (primary) / pass_rate (tie-break). Never used to disable
    planning with the sticky accepted model.
    """
    if require_hard_clean and candidate.n_hard_fail > 0:
        return False
    if candidate.n_checked <= 0:
        return False
    if accepted is None:
        return candidate.pass_rate >= min_pass_rate or candidate.n_hard_fail == 0
    better_median = candidate.median_residual < accepted.median_residual - 1e-9
    better_pass = candidate.pass_rate > accepted.pass_rate + 1e-9
    return bool(better_median or (abs(candidate.median_residual - accepted.median_residual) <= 1e-9 and better_pass))


def seed_holdout_ok(
    candidate_seed: BacktestReport,
    accepted_seed: BacktestReport,
    *,
    median_slack: float = 5.0,
    pass_slack: float = 0.05,
) -> bool:
    """Block promotions that overfit curated surprises and regress on seed Timeline.

    Continuous analogue of Schema's full Timeline re-certify: improving the
    surprise-heavy curated slice is not enough if seed residuals blow up.
    """
    if candidate_seed.n_checked <= 0 or accepted_seed.n_checked <= 0:
        return False
    if candidate_seed.n_hard_fail > accepted_seed.n_hard_fail:
        return False
    median_ok = candidate_seed.median_residual <= accepted_seed.median_residual + max(0.0, median_slack)
    pass_ok = candidate_seed.pass_rate + 1e-9 >= accepted_seed.pass_rate - max(0.0, pass_slack)
    return bool(median_ok and pass_ok)
