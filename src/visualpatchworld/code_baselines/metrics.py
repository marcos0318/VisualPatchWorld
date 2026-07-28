"""Transition-level metrics shared by code WM baselines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .common import load_model, sanitize_generated_code, state_vector_from_belief


@dataclass
class TransitionEval:
    accuracy: float
    passed: int
    total: int
    load_error: bool
    errors: List[Dict[str, Any]]
    mse: float


def iter_transitions(trajectories: Sequence[Any], *, max_transitions: int = 0) -> List[Any]:
    out: List[Any] = []
    for traj in trajectories:
        out.extend(getattr(traj, "transitions", []))
        if max_transitions > 0 and len(out) >= max_transitions:
            return out[:max_transitions]
    return out


def default_transition_tolerance(env_name: str = "") -> float:
    """Coordinate-scale-aware MSE gate for REx / code-WM accuracy.

    PushT / Two-room use pixel-like units (~1e2), so tolerance=5 is reasonable.
    Cube / Reacher are meter-scale (~1e-1); the same gate marks almost any model
    as 100% accurate and skips repair (WorldCoder Cube historically exited with
    rounds_run=0 despite PatchWorld still reporting hundreds of errors).
    """

    key = (env_name or "").strip().lower().replace("-", "").replace("_", "")
    if key in {"cube", "ogb", "reacher", "dmc"}:
        return 0.05
    return 5.0


def transition_matches(model: Any, transition: Any, *, tolerance: float = 5.0) -> Tuple[bool, float]:
    obs = getattr(transition, "observation", "") or ""
    nxt = getattr(transition, "next_observation", "") or ""
    action = getattr(transition, "action", "") or ""
    try:
        belief = model.init_belief_from_observation(obs)
        predicted = model.predict_belief(belief, action)
        if not isinstance(predicted, Mapping) or not predicted.get("objects"):
            readout = model.readout_observation(predicted, action)
            predicted = model.parse_observation(readout)
        pred_keys, pred_vals = state_vector_from_belief(predicted)
        tgt = model.parse_observation(nxt)
        tgt_keys, tgt_vals = state_vector_from_belief(tgt)
        if not pred_keys or pred_keys != tgt_keys:
            return False, 1e4
        mse = float(sum((a - b) ** 2 for a, b in zip(pred_vals, tgt_vals)))
        return mse <= tolerance * tolerance * max(len(pred_vals), 1), mse
    except Exception:
        return False, 1e4


def evaluate_transition_accuracy(
    code: str,
    trajectories: Sequence[Any],
    *,
    max_transitions: int = 0,
    tolerance: Optional[float] = None,
    env_name: str = "",
) -> TransitionEval:
    if tolerance is None:
        tolerance = default_transition_tolerance(env_name)
    model, err = load_model(code)
    if model is None:
        return TransitionEval(
            accuracy=0.0,
            passed=0,
            total=0,
            load_error=True,
            errors=[{"error_type": "load_error", "message": err or "load failed"}],
            mse=1e6,
        )

    transitions = iter_transitions(trajectories, max_transitions=max_transitions)
    passed = 0
    mse_total = 0.0
    errors: List[Dict[str, Any]] = []
    for transition in transitions:
        ok, mse = transition_matches(model, transition, tolerance=tolerance)
        mse_total += mse
        if ok:
            passed += 1
        elif len(errors) < 12:
            errors.append(
                {
                    "error_type": "observation_mismatch",
                    "action": getattr(transition, "action", ""),
                    "observation": (getattr(transition, "observation", "") or "")[:700],
                    "next_observation": (getattr(transition, "next_observation", "") or "")[:700],
                    "mse": mse,
                }
            )
    total = len(transitions)
    accuracy = passed / total if total else 0.0
    return TransitionEval(
        accuracy=accuracy,
        passed=passed,
        total=total,
        load_error=False,
        errors=errors,
        mse=mse_total / max(total, 1),
    )


def reward_from_eval(eval_result: TransitionEval) -> float:
    if eval_result.load_error:
        return 0.0
    return eval_result.accuracy
