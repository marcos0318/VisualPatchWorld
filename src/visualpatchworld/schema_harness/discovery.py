"""Discriminating experiments (Schema: action for discovery).

Beyond PushT structure ID: Two-room probes that separate competing hypotheses
about what the state *is* and how actions move it. Probes only use gym
reset/step — no simulator source.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from visualpatchworld.lewm_env_specs import LeWMEnvSpec, get_env_spec, make_gym_env
from visualpatchworld.lewm_graph import (
    action_string_from_vector,
    graph_text_from_row,
    reset_env_from_rows,
    sim_row_from_obs,
    sim_step_with_info,
)

from .timeline import Timeline


@dataclass
class ProbeResult:
    name: str
    hypothesis_a: str
    hypothesis_b: str
    winner: str
    evidence: Dict[str, float]
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "hypothesis_a": self.hypothesis_a,
            "hypothesis_b": self.hypothesis_b,
            "winner": self.winner,
            "evidence": dict(self.evidence),
            "notes": list(self.notes),
        }


def _agent_xy(row: Mapping[str, Any], env_name: str) -> np.ndarray:
    if env_name == "tworoom":
        obs = np.asarray(row["observation"], dtype=np.float64)
        return obs[:2].copy()
    if env_name == "pusht":
        state = np.asarray(row["state"], dtype=np.float64)
        return state[:2].copy()
    if env_name == "cube":
        return np.asarray(row["effector_pos"], dtype=np.float64).copy()
    if env_name == "reacher":
        return np.asarray(row["finger_pos"], dtype=np.float64).copy()
    raise KeyError(env_name)


def _target_xy(row: Mapping[str, Any], env_name: str) -> np.ndarray:
    if env_name == "tworoom":
        obs = np.asarray(row["observation"], dtype=np.float64)
        return obs[2:4].copy()
    if env_name == "cube":
        return np.asarray(row["goal_block_pos"], dtype=np.float64).copy()
    if env_name == "reacher":
        return np.asarray(row["target_pos"], dtype=np.float64).copy()
    raise KeyError(env_name)


def _cube_block_xy(row: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(row["block_pos"], dtype=np.float64).copy()


def _reacher_qpos(row: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(row["qpos"], dtype=np.float64).copy()


def _reacher_qvel(row: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(row["qvel"], dtype=np.float64).copy()


def _macro_step(
    env: Any,
    spec: LeWMEnvSpec,
    row: Dict[str, Any],
    action: np.ndarray,
    frameskip: int,
) -> Dict[str, Any]:
    obs = None
    info: Mapping[str, Any] = {}
    for _ in range(max(1, frameskip)):
        obs, info = sim_step_with_info(env, spec, action)
    return sim_row_from_obs(spec.name, obs, template_row=row, info=info)


def _sync_row_after_reset(
    env: Any,
    spec: LeWMEnvSpec,
    start_row: Mapping[str, Any],
) -> Dict[str, Any]:
    """Refresh privileged fields from a zero-action step after reset.

    Dataset start rows can disagree with the live env target/goal after
    ``reset(options=...)`` (especially Reacher target_pos).
    """
    row = dict(start_row)
    action = np.zeros(spec.action_dim, dtype=np.float64)
    obs, info = sim_step_with_info(env, spec, action)
    return sim_row_from_obs(spec.name, obs, template_row=row, info=info)


def probe_tworoom_target_static(
    env: Any,
    spec: LeWMEnvSpec,
    start_row: Mapping[str, Any],
    goal_row: Mapping[str, Any],
) -> ProbeResult:
    """H_a: target static under agent motion; H_b: target co-moves with agent."""
    reset_env_from_rows(env, spec, start_row, goal_row)
    row = dict(start_row)
    t0 = _target_xy(row, "tworoom")
    a0 = _agent_xy(row, "tworoom")
    action = np.array([1.0, 0.0], dtype=np.float64)
    row = _macro_step(env, spec, row, action, spec.frameskip)
    t1 = _target_xy(row, "tworoom")
    a1 = _agent_xy(row, "tworoom")
    target_move = float(np.linalg.norm(t1 - t0))
    agent_move = float(np.linalg.norm(a1 - a0))
    winner = "target_static" if target_move < 1e-3 and agent_move > 1e-3 else "target_comoves_or_stuck"
    return ProbeResult(
        name="tworoom_target_static",
        hypothesis_a="target_static",
        hypothesis_b="target_comoves_or_stuck",
        winner=winner,
        evidence={"target_move": target_move, "agent_move": agent_move},
        notes=[f"target_move={target_move:.4f} agent_move={agent_move:.4f} → {winner}"],
    )


def probe_tworoom_action_direction(
    env: Any,
    spec: LeWMEnvSpec,
    start_row: Mapping[str, Any],
    goal_row: Mapping[str, Any],
) -> ProbeResult:
    """H_a: agent displaces along action; H_b: action-independent / noop dynamics."""
    reset_env_from_rows(env, spec, start_row, goal_row)
    row = dict(start_row)
    a0 = _agent_xy(row, "tworoom")
    action = np.array([0.0, 1.0], dtype=np.float64)
    row = _macro_step(env, spec, row, action, spec.frameskip)
    a1 = _agent_xy(row, "tworoom")
    delta = a1 - a0
    aligned = float(delta[1])  # expect +y for action [0,1]
    orthogonal = abs(float(delta[0]))
    winner = "action_aligned" if aligned > 0.5 and aligned > orthogonal else "misaligned_or_weak"
    return ProbeResult(
        name="tworoom_action_direction",
        hypothesis_a="action_aligned",
        hypothesis_b="misaligned_or_weak",
        winner=winner,
        evidence={"delta_y": float(delta[1]), "delta_x": float(delta[0]), "aligned": aligned, "orthogonal": orthogonal},
        notes=[f"delta=({delta[0]:.3f},{delta[1]:.3f}) → {winner}"],
    )


def probe_tworoom_hold_still(
    env: Any,
    spec: LeWMEnvSpec,
    start_row: Mapping[str, Any],
    goal_row: Mapping[str, Any],
) -> ProbeResult:
    """H_a: zero action ≈ no motion; H_b: drift / autonomous dynamics."""
    reset_env_from_rows(env, spec, start_row, goal_row)
    row = dict(start_row)
    a0 = _agent_xy(row, "tworoom")
    action = np.zeros(spec.action_dim, dtype=np.float64)
    row = _macro_step(env, spec, row, action, spec.frameskip)
    a1 = _agent_xy(row, "tworoom")
    drift = float(np.linalg.norm(a1 - a0))
    winner = "hold_still" if drift < 0.5 else "autonomous_drift"
    return ProbeResult(
        name="tworoom_hold_still",
        hypothesis_a="hold_still",
        hypothesis_b="autonomous_drift",
        winner=winner,
        evidence={"drift": drift},
        notes=[f"hold_drift={drift:.4f} → {winner}"],
    )


def probe_cube_goal_static(
    env: Any,
    spec: LeWMEnvSpec,
    start_row: Mapping[str, Any],
    goal_row: Mapping[str, Any],
) -> ProbeResult:
    """H_a: goal pose is static under effector motion; H_b: goal co-moves."""
    reset_env_from_rows(env, spec, start_row, goal_row)
    row = _sync_row_after_reset(env, spec, start_row)
    g0 = _target_xy(row, "cube")
    e0 = _agent_xy(row, "cube")
    # Translate effector in +x without grip.
    action = np.array([0.4, 0.0, 0.0, 0.0, -1.0], dtype=np.float64)
    row = _macro_step(env, spec, row, action, spec.frameskip)
    g1 = _target_xy(row, "cube")
    e1 = _agent_xy(row, "cube")
    goal_move = float(np.linalg.norm(g1 - g0))
    eff_move = float(np.linalg.norm(e1 - e0))
    # Goal may be missing from info; fall back to start goal field.
    if "goal_block_pos" not in row and "goal_block_pos" in start_row:
        g1 = np.asarray(start_row["goal_block_pos"], dtype=np.float64)
        goal_move = float(np.linalg.norm(g1 - g0))
    winner = "goal_static" if goal_move < 1e-3 and eff_move > 1e-3 else "goal_comoves_or_stuck"
    return ProbeResult(
        name="cube_goal_static",
        hypothesis_a="goal_static",
        hypothesis_b="goal_comoves_or_stuck",
        winner=winner,
        evidence={"goal_move": goal_move, "effector_move": eff_move},
        notes=[f"goal_move={goal_move:.4f} effector_move={eff_move:.4f} → {winner}"],
    )


def probe_cube_grip_gated_contact(
    env: Any,
    spec: LeWMEnvSpec,
    start_row: Mapping[str, Any],
    goal_row: Mapping[str, Any],
) -> ProbeResult:
    """H_a: block tracks effector only under near+grip; H_b: always/never coupled."""
    reset_env_from_rows(env, spec, start_row, goal_row)
    row = _sync_row_after_reset(env, spec, start_row)

    # Near + grip-open trial: a4 > 0 should latch/track when already near.
    eff0 = _agent_xy(row, "cube")
    blk0 = _cube_block_xy(row)
    dist0 = float(np.linalg.norm(eff0 - blk0))
    grip_action = np.array([0.2, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    row_grip = row
    for _ in range(3):
        row_grip = _macro_step(env, spec, row_grip, grip_action, spec.frameskip)
    eff1 = _agent_xy(row_grip, "cube")
    blk1 = _cube_block_xy(row_grip)
    track_err_grip = float(np.linalg.norm(blk1 - eff1))
    block_move_grip = float(np.linalg.norm(blk1 - blk0))

    # Reset and try near + no-grip: a4 < 0 should not latch as strongly.
    reset_env_from_rows(env, spec, start_row, goal_row)
    row = _sync_row_after_reset(env, spec, start_row)
    blk0b = _cube_block_xy(row)
    nogrip_action = np.array([0.2, 0.0, 0.0, 0.0, -1.0], dtype=np.float64)
    row_nogrip = row
    for _ in range(3):
        row_nogrip = _macro_step(env, spec, row_nogrip, nogrip_action, spec.frameskip)
    blk1b = _cube_block_xy(row_nogrip)
    block_move_nogrip = float(np.linalg.norm(blk1b - blk0b))

    track_err_nogrip = float(
        np.linalg.norm(_cube_block_xy(row_nogrip) - _agent_xy(row_nogrip, "cube"))
    )
    # Latch signature: grip keeps block near effector; no-grip does not.
    gated = dist0 < 0.05 and track_err_grip < 0.03 and track_err_nogrip > 0.05
    always = track_err_grip < 0.03 and track_err_nogrip < 0.03
    never = track_err_grip > 0.05 and block_move_grip < 0.01
    if gated:
        winner = "contact_grasp_gated"
    elif always:
        winner = "always_coupled"
    elif never:
        winner = "block_static"
    else:
        winner = "contact_uncertain"
    return ProbeResult(
        name="cube_grip_gated_contact",
        hypothesis_a="contact_grasp_gated",
        hypothesis_b="always_coupled_or_static",
        winner=winner,
        evidence={
            "dist0": dist0,
            "block_move_grip": block_move_grip,
            "block_move_nogrip": block_move_nogrip,
            "track_err_grip": track_err_grip,
            "track_err_nogrip": track_err_nogrip,
        },
        notes=[
            f"dist0={dist0:.4f} track_grip={track_err_grip:.4f} "
            f"track_nogrip={track_err_nogrip:.4f} → {winner}"
        ],
    )


def probe_cube_far_decoupled(
    env: Any,
    spec: LeWMEnvSpec,
    start_row: Mapping[str, Any],
    goal_row: Mapping[str, Any],
) -> ProbeResult:
    """H_a: far effector motion leaves block still; H_b: action-driven block."""
    reset_env_from_rows(env, spec, start_row, goal_row)
    row = _sync_row_after_reset(env, spec, start_row)
    # Move away from block.
    for _ in range(6):
        eff = _agent_xy(row, "cube")
        blk = _cube_block_xy(row)
        away = eff - blk
        n = float(np.linalg.norm(away))
        step = 0.5 * away / max(n, 1e-6)
        action = np.array([step[0], step[1], step[2] if away.size > 2 else 0.0, 0.0, -1.0], dtype=np.float64)
        action = np.pad(action, (0, max(0, 5 - action.size)))[:5]
        row = _macro_step(env, spec, row, action, spec.frameskip)
    blk0 = _cube_block_xy(row)
    eff0 = _agent_xy(row, "cube")
    dist = float(np.linalg.norm(eff0 - blk0))
    action = np.array([0.3, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    row = _macro_step(env, spec, row, action, spec.frameskip)
    blk1 = _cube_block_xy(row)
    block_move = float(np.linalg.norm(blk1 - blk0))
    winner = "far_decoupled" if dist > 0.08 and block_move < 0.02 else "far_coupled_or_close"
    return ProbeResult(
        name="cube_far_decoupled",
        hypothesis_a="far_decoupled",
        hypothesis_b="far_coupled_or_close",
        winner=winner,
        evidence={"dist": dist, "block_move": block_move},
        notes=[f"far_dist={dist:.4f} block_move={block_move:.4f} → {winner}"],
    )


def probe_reacher_joint_actuation(
    env: Any,
    spec: LeWMEnvSpec,
    start_row: Mapping[str, Any],
    goal_row: Mapping[str, Any],
) -> ProbeResult:
    """H_a: actions change joint angles; H_b: joints inert / Cartesian-only."""
    reset_env_from_rows(env, spec, start_row, goal_row)
    row = _sync_row_after_reset(env, spec, start_row)
    q0 = _reacher_qpos(row)
    action = np.array([1.0, 0.0], dtype=np.float64)
    row = _macro_step(env, spec, row, action, spec.frameskip)
    q1 = _reacher_qpos(row)
    dq = float(np.linalg.norm(q1 - q0))
    winner = "joint_actuated" if dq > 0.02 else "joint_inert"
    return ProbeResult(
        name="reacher_joint_actuation",
        hypothesis_a="joint_actuated",
        hypothesis_b="joint_inert",
        winner=winner,
        evidence={"dq": dq},
        notes=[f"dq={dq:.4f} → {winner}"],
    )


def probe_reacher_target_static(
    env: Any,
    spec: LeWMEnvSpec,
    start_row: Mapping[str, Any],
    goal_row: Mapping[str, Any],
) -> ProbeResult:
    """H_a: target pose static under arm motion; H_b: target co-moves."""
    reset_env_from_rows(env, spec, start_row, goal_row)
    row = _sync_row_after_reset(env, spec, start_row)
    action = np.array([0.0, 1.0], dtype=np.float64)
    # Warmup: Reacher may settle target_pos on the first non-trivial macro.
    row = _macro_step(env, spec, row, action, spec.frameskip)
    t0 = _target_xy(row, "reacher")
    q0 = _reacher_qpos(row)
    row = _macro_step(env, spec, row, action, spec.frameskip)
    t1 = _target_xy(row, "reacher")
    q1 = _reacher_qpos(row)
    target_move = float(np.linalg.norm(t1 - t0))
    joint_move = float(np.linalg.norm(q1 - q0))
    winner = "target_static" if target_move < 1e-3 and joint_move > 1e-3 else "target_comoves_or_stuck"
    return ProbeResult(
        name="reacher_target_static",
        hypothesis_a="target_static",
        hypothesis_b="target_comoves_or_stuck",
        winner=winner,
        evidence={"target_move": target_move, "joint_move": joint_move},
        notes=[f"target_move={target_move:.4f} joint_move={joint_move:.4f} → {winner}"],
    )


def probe_reacher_first_order_damping(
    env: Any,
    spec: LeWMEnvSpec,
    start_row: Mapping[str, Any],
    goal_row: Mapping[str, Any],
) -> ProbeResult:
    """H_a: hold damps velocity (first-order); H_b: coasts (second-order)."""
    reset_env_from_rows(env, spec, start_row, goal_row)
    row = _sync_row_after_reset(env, spec, start_row)
    # Excite joints.
    for _ in range(3):
        row = _macro_step(env, spec, row, np.array([1.0, -1.0], dtype=np.float64), spec.frameskip)
    v_after_push = float(np.linalg.norm(_reacher_qvel(row)))
    # Hold.
    for _ in range(4):
        row = _macro_step(env, spec, row, np.zeros(2, dtype=np.float64), spec.frameskip)
    v_after_hold = float(np.linalg.norm(_reacher_qvel(row)))
    ratio = v_after_hold / max(v_after_push, 1e-6)
    winner = "first_order_damped" if ratio < 0.5 or v_after_hold < 0.05 else "second_order_coast"
    return ProbeResult(
        name="reacher_first_order_damping",
        hypothesis_a="first_order_damped",
        hypothesis_b="second_order_coast",
        winner=winner,
        evidence={"v_after_push": v_after_push, "v_after_hold": v_after_hold, "ratio": ratio},
        notes=[f"v_push={v_after_push:.4f} v_hold={v_after_hold:.4f} ratio={ratio:.3f} → {winner}"],
    )


def probe_pusht_structure(*, geometry: Optional[Mapping[str, Any]] = None) -> List[ProbeResult]:
    """Wrap program_induction.identify_structure as Schema-style probe results."""
    from visualpatchworld.program_induction import identify_structure

    report = identify_structure(geometry=geometry)
    results: List[ProbeResult] = []
    for name, decision in (
        ("pusht_agent_law", report.agent_law),
        ("pusht_block_law", report.block_law),
        ("pusht_contact_law", report.contact_law),
    ):
        alts = list(decision.alternatives)
        other = alts[0] if alts else "alt"
        results.append(
            ProbeResult(
                name=name,
                hypothesis_a=str(decision.choice),
                hypothesis_b=str(other),
                winner=str(decision.choice),
                evidence={
                    "statistic": float(decision.statistic),
                    "threshold": float(decision.threshold),
                    "margin": float(decision.margin),
                    "confident": float(1.0 if decision.confident else 0.0),
                },
                notes=[
                    f"{decision.question}: choose={decision.choice} "
                    f"stat={decision.statistic:.4f} thr={decision.threshold:.4f} "
                    f"confident={decision.confident}"
                ],
            )
        )
    results.append(
        ProbeResult(
            name="pusht_sketch",
            hypothesis_a="pd_quasi_static_contact",
            hypothesis_b="other",
            winner="_".join(report.sketch),
            evidence={"all_confident": float(1.0 if report.all_confident else 0.0)},
            notes=[f"sketch={list(report.sketch)} all_confident={report.all_confident}"],
        )
    )
    return results


def run_structure_probes(
    *,
    env_name: str,
    start_row: Mapping[str, Any],
    goal_row: Mapping[str, Any],
    timeline: Optional[Timeline] = None,
    env: Any = None,
    geometry: Optional[Mapping[str, Any]] = None,
) -> List[ProbeResult]:
    """Run env-specific discriminating probes; append conclusions to Timeline notes."""
    spec = get_env_spec(env_name)
    owns = env is None
    if owns and spec.name != "pusht":
        env = make_gym_env(spec)
    results: List[ProbeResult] = []
    try:
        if spec.name == "tworoom":
            results = [
                probe_tworoom_target_static(env, spec, start_row, goal_row),
                probe_tworoom_action_direction(env, spec, start_row, goal_row),
                probe_tworoom_hold_still(env, spec, start_row, goal_row),
            ]
        elif spec.name == "cube":
            results = [
                probe_cube_goal_static(env, spec, start_row, goal_row),
                probe_cube_grip_gated_contact(env, spec, start_row, goal_row),
                probe_cube_far_decoupled(env, spec, start_row, goal_row),
            ]
        elif spec.name == "reacher":
            results = [
                probe_reacher_joint_actuation(env, spec, start_row, goal_row),
                probe_reacher_target_static(env, spec, start_row, goal_row),
                probe_reacher_first_order_damping(env, spec, start_row, goal_row),
            ]
        elif spec.name == "pusht":
            # Uses its own BlackBoxSim; do not require the planning env.
            results = probe_pusht_structure(geometry=geometry)
        else:
            results = [
                ProbeResult(
                    name=f"{spec.name}_probe_stub",
                    hypothesis_a="fixed_scaffold",
                    hypothesis_b="open",
                    winner="fixed_scaffold",
                    evidence={},
                    notes=[f"No general probes for {spec.name} yet."],
                )
            ]
    finally:
        if owns and env is not None:
            env.close()

    if timeline is not None:
        for r in results:
            timeline.add_note(f"probe[{r.name}] winner={r.winner} evidence={r.evidence}")
            for n in r.notes:
                timeline.add_note(f"  {n}")
    return results


def structure_sketch_from_probes(results: Sequence[ProbeResult]) -> Dict[str, str]:
    """Compress probe winners into a dynamical sketch for notes / revision."""
    sketch = {r.name: r.winner for r in results}
    if any(r.name.startswith("tworoom") for r in results):
        sketch["summary"] = (
            "tworoom_linear_scaffold"
            if sketch.get("tworoom_target_static") == "target_static"
            and sketch.get("tworoom_action_direction") == "action_aligned"
            and sketch.get("tworoom_hold_still") == "hold_still"
            else "tworoom_structure_uncertain"
        )
    if any(r.name.startswith("cube") for r in results):
        sketch["summary"] = (
            "cube_contact_grasp_gated"
            if sketch.get("cube_goal_static") == "goal_static"
            and sketch.get("cube_grip_gated_contact") == "contact_grasp_gated"
            and sketch.get("cube_far_decoupled") == "far_decoupled"
            else "cube_linear_or_uncertain"
        )
    if any(r.name.startswith("reacher") for r in results):
        sketch["summary"] = (
            "reacher_joint_fk"
            if sketch.get("reacher_joint_actuation") == "joint_actuated"
            and sketch.get("reacher_target_static") == "target_static"
            and sketch.get("reacher_first_order_damping") == "first_order_damped"
            else "reacher_linear_or_uncertain"
        )
    if any(r.name.startswith("pusht") for r in results):
        sketch["summary"] = sketch.get("pusht_sketch", "pusht_structure_unknown")
    return sketch


def record_probe_transitions(
    *,
    env: Any,
    spec: LeWMEnvSpec,
    start_row: Mapping[str, Any],
    goal_row: Mapping[str, Any],
    timeline: Timeline,
    actions: Sequence[np.ndarray],
) -> None:
    """Execute probe actions and append tagged probe transitions to Timeline."""
    reset_env_from_rows(env, spec, start_row, goal_row)
    row = dict(start_row)
    obs_text = graph_text_from_row(spec.name, row, source="oracle", step=0)
    for i, action in enumerate(actions):
        action = np.asarray(action, dtype=np.float64)
        action_str = action_string_from_vector(action, spec)
        next_row = _macro_step(env, spec, row, action, spec.frameskip)
        next_obs = graph_text_from_row(spec.name, next_row, source="oracle", step=i + 1)
        timeline.append(obs_text, action_str, next_obs, source="probe", residual=0.0, weight=1.0)
        row = next_row
        obs_text = next_obs
