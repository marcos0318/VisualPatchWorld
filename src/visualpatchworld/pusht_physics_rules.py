"""PushT physics-rule compliance checks for induced dynamics."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .pusht_local_induction import (
    _parse_action,
    _parse_observation_text,
    _state_from_graph,
)


@dataclass(frozen=True)
class PhysicsRuleConfig:
  contact_threshold: float = 70.0
  free_block_move_tol: float = 0.5
  max_block_move: float = 80.0
  max_agent_move: float = 160.0
  workspace_min: float = 0.0
  workspace_max: float = 512.0
  contact_push_cos_tol: float = 0.0


@dataclass
class RuleCheckResult:
  rule_id: str
  passed: bool
  detail: str = ""


@dataclass
class TransitionRuleReport:
  observation: str = ""
  action: str = ""
  state: Tuple[float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0)
  next_state: Tuple[float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0)
  checks: List[RuleCheckResult] = field(default_factory=list)

  @property
  def passed(self) -> bool:
    return all(check.passed for check in self.checks)

  @property
  def violations(self) -> List[str]:
    return [check.rule_id for check in self.checks if not check.passed]


def _delta(state: Sequence[float], next_state: Sequence[float]) -> Tuple[float, float, float, float, float]:
  return (
    float(next_state[0]) - float(state[0]),
    float(next_state[1]) - float(state[1]),
    float(next_state[2]) - float(state[2]),
    float(next_state[3]) - float(state[3]),
    float(next_state[4]) - float(state[4]),
  )


def _cosine_similarity(ax: float, ay: float, bx: float, by: float) -> float:
  denom = math.hypot(ax, ay) * math.hypot(bx, by)
  if denom <= 1e-8:
    return 1.0
  return (ax * bx + ay * by) / denom


def check_physics_rules(
  state: Sequence[float],
  action: Sequence[float],
  next_state: Sequence[float],
  *,
  config: Optional[PhysicsRuleConfig] = None,
) -> List[RuleCheckResult]:
  """Evaluate symbolic PushT physics rules on one macro-step transition."""

  cfg = config or PhysicsRuleConfig()
  sx, sy, bx, by, ba = (float(v) for v in state[:5])
  nsx, nsy, nbx, nby, nba = (float(v) for v in next_state[:5])
  ax, ay = float(action[0]), float(action[1])
  dax, day, dbx, dby, _ = _delta(state[:5], next_state[:5])
  dist = math.hypot(bx - sx, by - sy)
  block_move = math.hypot(dbx, dby)
  agent_move = math.hypot(dax, day)

  checks: List[RuleCheckResult] = []

  far = dist > cfg.contact_threshold
  checks.append(
    RuleCheckResult(
      rule_id="R1_no_block_motion_when_far",
      passed=not (far and block_move > cfg.free_block_move_tol),
      detail=f"dist={dist:.2f}, block_move={block_move:.2f}",
    )
  )

  in_contact = dist <= cfg.contact_threshold
  aligned = _cosine_similarity(dax, day, dbx, dby) >= cfg.contact_push_cos_tol
  checks.append(
    RuleCheckResult(
      rule_id="R2_contact_push_alignment",
      passed=not (in_contact and block_move > 0.25 and agent_move > 0.25 and not aligned),
      detail=f"cos={_cosine_similarity(dax, day, dbx, dby):.2f}",
    )
  )

  checks.append(
    RuleCheckResult(
      rule_id="R3_agent_motion_bounded",
      passed=agent_move <= cfg.max_agent_move,
      detail=f"agent_move={agent_move:.2f}",
    )
  )

  in_bounds = all(
    cfg.workspace_min <= value <= cfg.workspace_max
    for value in (nsx, nsy, nbx, nby)
  )
  checks.append(
    RuleCheckResult(
      rule_id="R4_workspace_bounds",
      passed=in_bounds,
      detail=f"next=({nsx:.1f},{nsy:.1f},{nbx:.1f},{nby:.1f})",
    )
  )

  checks.append(
    RuleCheckResult(
      rule_id="R5_block_speed_limit",
      passed=block_move <= cfg.max_block_move,
      detail=f"block_move={block_move:.2f}",
    )
  )

  angle_delta = abs(nba - ba)
  angle_delta = min(angle_delta, abs(2.0 * math.pi - angle_delta))
  checks.append(
    RuleCheckResult(
      rule_id="R6_angle_wrap_reasonable",
      passed=angle_delta <= math.pi,
      detail=f"angle_delta={angle_delta:.3f}",
    )
  )

  return checks


def check_transition_text(
  observation: str,
  action: str,
  next_observation: str,
  *,
  config: Optional[PhysicsRuleConfig] = None,
) -> TransitionRuleReport:
  state = _state_from_graph(_parse_observation_text(observation))
  next_state = _state_from_graph(_parse_observation_text(next_observation))
  action_xy = _parse_action(action)
  checks = check_physics_rules(state, action_xy, next_state, config=config)
  return TransitionRuleReport(
    observation=observation,
    action=action,
    state=state,
    next_state=next_state,
    checks=checks,
  )


def summarize_rule_reports(reports: Sequence[TransitionRuleReport]) -> Dict[str, Any]:
  if not reports:
    return {"n_transitions": 0, "compliance_rate": 0.0, "rules": {}}

  rule_ids = sorted({check.rule_id for report in reports for check in report.checks})
  rule_stats: Dict[str, Dict[str, float]] = {}
  for report in reports:
    for check in report.checks:
      if check.rule_id not in rule_stats:
        rule_stats[check.rule_id] = {"passed": 0.0, "total": 0.0}
      rule_stats[check.rule_id]["total"] += 1.0
      if check.passed:
        rule_stats[check.rule_id]["passed"] += 1.0

  for rule_id in rule_ids:
    total = rule_stats[rule_id]["total"]
    passed = rule_stats[rule_id]["passed"]
    rule_stats[rule_id]["compliance_rate"] = passed / max(1.0, total)

  fully_passed = sum(1 for report in reports if report.passed)
  return {
    "n_transitions": len(reports),
    "compliance_rate": fully_passed / len(reports),
    "rules": rule_stats,
  }


def evaluate_dataset_rule_compliance(
  trajectories: Sequence[Any],
  *,
  config: Optional[PhysicsRuleConfig] = None,
) -> Dict[str, Any]:
  reports: List[TransitionRuleReport] = []
  for traj in trajectories:
    for transition in getattr(traj, "transitions", []):
      reports.append(
        check_transition_text(
          str(getattr(transition, "observation", "")),
          str(getattr(transition, "action", "")),
          str(getattr(transition, "next_observation", "")),
          config=config,
        )
      )
  summary = summarize_rule_reports(reports)
  summary["source"] = "dataset_transitions"
  return summary


def evaluate_model_rule_compliance(
  model: Any,
  trajectories: Sequence[Any],
  *,
  config: Optional[PhysicsRuleConfig] = None,
) -> Dict[str, Any]:
  from .sim_eval import _predict_next_graph

  reports: List[TransitionRuleReport] = []
  for traj in trajectories:
    for transition in getattr(traj, "transitions", []):
      obs = str(getattr(transition, "observation", ""))
      action = str(getattr(transition, "action", ""))
      state = _state_from_graph(_parse_observation_text(obs))
      predicted = _predict_next_graph(model, obs, action)
      next_state = _state_from_graph(predicted)
      checks = check_physics_rules(state, _parse_action(action), next_state, config=config)
      reports.append(
        TransitionRuleReport(
          observation=obs,
          action=action,
          state=state,
          next_state=next_state,
          checks=checks,
        )
      )
  summary = summarize_rule_reports(reports)
  summary["source"] = "model_predictions"
  return summary


def stratified_position_errors(
  trajectories: Sequence[Any],
  *,
  contact_threshold: float = 70.0,
) -> Dict[str, Any]:
  """Observation/dynamics errors split by contact regime."""

  from .lewm_sampling import state_field_errors

  near_agent: List[float] = []
  near_block: List[float] = []
  far_agent: List[float] = []
  far_block: List[float] = []

  for traj in trajectories:
    for transition in getattr(traj, "transitions", []):
      obs = _parse_observation_text(getattr(transition, "observation", ""))
      nxt = _parse_observation_text(getattr(transition, "next_observation", ""))
      sx, sy, bx, by, _ = _state_from_graph(obs)
      nsx, nsy, nbx, nby, nba = _state_from_graph(nxt)
      dist = math.hypot(bx - sx, by - sy)
      errors = state_field_errors(nxt, [nsx, nsy, nbx, nby, nba])
      if dist <= contact_threshold:
        near_agent.append(errors["agent_position_error"])
        near_block.append(errors["block_position_error"])
      else:
        far_agent.append(errors["agent_position_error"])
        far_block.append(errors["block_position_error"])

  def _mean(values: List[float]) -> Optional[float]:
    return float(sum(values) / len(values)) if values else None

  return {
    "contact_threshold": contact_threshold,
    "near_contact": {
      "count": len(near_agent),
      "agent_position_error_mean": _mean(near_agent),
      "block_position_error_mean": _mean(near_block),
    },
    "far_contact": {
      "count": len(far_agent),
      "agent_position_error_mean": _mean(far_agent),
      "block_position_error_mean": _mean(far_block),
    },
  }


def collect_rule_violation_weighted_transitions(
  model: Any,
  trajectories: Sequence[Any],
  *,
  config: Optional[PhysicsRuleConfig] = None,
  weight: float = 3.0,
  target_rules: Optional[Sequence[str]] = None,
) -> List["WeightedTransition"]:
  """Upsample transitions where model predictions violate key physics rules."""

  from .pusht_local_induction import WeightedTransition
  from .sim_eval import _predict_next_graph

  cfg = config or PhysicsRuleConfig()
  focus = set(target_rules or ("R1_no_block_motion_when_far", "R2_contact_push_alignment", "R5_block_speed_limit"))
  rows: List[WeightedTransition] = []

  for traj in trajectories:
    for transition in getattr(traj, "transitions", []):
      obs = str(getattr(transition, "observation", ""))
      action = str(getattr(transition, "action", ""))
      truth_next = str(getattr(transition, "next_observation", ""))
      state = _state_from_graph(_parse_observation_text(obs))
      predicted = _predict_next_graph(model, obs, action)
      pred_state = _state_from_graph(predicted)
      checks = check_physics_rules(state, _parse_action(action), pred_state, config=cfg)
      violated = {check.rule_id for check in checks if not check.passed}
      if violated & focus:
        rows.append(
          WeightedTransition(
            observation=obs,
            action=action,
            next_observation=truth_next,
            weight=weight,
          )
        )
  return rows


def write_weighted_transitions_jsonl(
  rows: Sequence["WeightedTransition"],
  path: Path | str,
  *,
  source: str,
  item_id: str,
) -> None:
  from pathlib import Path as PathType

  from .pusht_local_induction import WeightedTransition

  record = {
    "metadata": {
      "env": "pusht",
      "source": source,
      "item_id": item_id,
      "split": "train",
    },
    "transitions": [
      {
        "observation": row.observation,
        "action": row.action,
        "next_observation": row.next_observation,
        "reward": 0.0,
        "done": False,
        "metadata": {
          "weight": row.weight,
          "source": source,
        },
      }
      for row in rows
    ],
  }
  out = PathType(path)
  out.parent.mkdir(parents=True, exist_ok=True)
  import json

  out.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")

