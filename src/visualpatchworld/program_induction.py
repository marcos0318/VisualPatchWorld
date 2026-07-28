"""Two-level program induction for object-centric world models.

This module implements the discovery pipeline discussed in
``docs/inducing_better_model.md`` and ``docs/scientific_discovery_dynamics.md``:

* **Level 1 -- Program (structure) induction.** Rather than hand-asserting the
  dynamics structure (e.g. "the PushT agent is PD-controlled, the block is
  quasi-static"), an *active-probing identifier* **discovers** that structure by
  designing and executing discriminating experiments against a black-box
  simulator -- the interactive, hypothesis-refuting protocol that NewtonBench
  shows pure curve-fitters skip. Each probe separates two competing structural
  hypotheses (PD-vs-linear agent; quasi-static-vs-momentum block; contact-driven
  block motion), reading only ``reset(state)`` / ``step(action)`` observations
  (no source, no parameters).

* **Level 2 -- Parameter fitting.** The discovered structure selects a general
  Python program *template* whose constant "holes" (gains, friction, contact
  stiffness) are then identified from data by the existing differentiable
  rollout fitters (``physics_sim_pd_dynamics.fit_pd_physics_params`` etc.).

The two levels compose: ``induce_program`` first identifies the structure, then
dispatches to the matching fitter and renders runnable model code. The structure
identifier is environment-agnostic in spirit (it only assumes a resettable
black-box sim and an object-centric observation); the PushT instance is provided
concretely here and is validated against the real simulator in
``scripts/visualpatchworld/identify_structure.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

Vec = Tuple[float, float]


# ---------------------------------------------------------------------------
# Black-box simulator wrapper (the only access the identifier is allowed)
# ---------------------------------------------------------------------------


class BlackBoxSim:
    """Minimal black-box interface to a resettable PushT simulator.

    Exposes only ``reset(state) -> obs`` and ``step(action) -> obs``. It reads no
    simulator source, parameters, or internals -- everything the structure
    identifier learns must come from the observed outcomes of designed probes.

    State / observation layout (7-dim): ``[agent_x, agent_y, block_x, block_y,
    block_angle, agent_vx, agent_vy]``. Actions are normalized to ``[-1, 1]^2``.
    """

    def __init__(self, env: Any = None, *, make_env: Optional[Callable[[], Any]] = None):
        if env is None:
            if make_env is None:
                from .sim_eval import _make_pusht_env

                make_env = _make_pusht_env
            env = make_env()
            self._owns_env = True
        else:
            self._owns_env = False
        self.env = env

    def reset(self, state: Sequence[float]) -> List[float]:
        import numpy as np

        self.env.reset(options={"state": np.asarray(state, dtype=np.float64)})
        return [float(v) for v in self.env.unwrapped._get_obs()]

    def step(self, action: Sequence[float]) -> List[float]:
        import numpy as np

        self.env.step(np.asarray(action, dtype=np.float32))
        return [float(v) for v in self.env.unwrapped._get_obs()]

    def close(self) -> None:
        if self._owns_env:
            try:
                self.env.close()
            except Exception:
                pass

    def __enter__(self) -> "BlackBoxSim":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def _dist(a: Sequence[float], b: Sequence[float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


# ---------------------------------------------------------------------------
# Structure decisions / report
# ---------------------------------------------------------------------------


@dataclass
class Decision:
    """A single structural choice with the experiment that produced it."""

    question: str
    choice: str
    alternatives: Sequence[str]
    statistic: float
    threshold: float
    confident: bool
    evidence: Dict[str, Any] = field(default_factory=dict)

    @property
    def margin(self) -> float:
        """Signed distance of the statistic from the decision threshold,
        normalized by the threshold (positive = comfortably on the chosen side)."""
        scale = max(abs(self.threshold), 1e-6)
        return float((self.statistic - self.threshold) / scale)

    def to_json(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "choice": self.choice,
            "alternatives": list(self.alternatives),
            "statistic": float(self.statistic),
            "threshold": float(self.threshold),
            "margin": self.margin,
            "confident": bool(self.confident),
            "evidence": self.evidence,
        }


@dataclass
class StructureReport:
    agent_law: Decision
    block_law: Decision
    contact_law: Decision

    @property
    def sketch(self) -> Tuple[str, str, str]:
        return (self.agent_law.choice, self.block_law.choice, self.contact_law.choice)

    @property
    def all_confident(self) -> bool:
        return all(d.confident for d in (self.agent_law, self.block_law, self.contact_law))

    def to_json(self) -> Dict[str, Any]:
        return {
            "sketch": list(self.sketch),
            "all_confident": self.all_confident,
            "agent_law": self.agent_law.to_json(),
            "block_law": self.block_law.to_json(),
            "contact_law": self.contact_law.to_json(),
        }


# ---------------------------------------------------------------------------
# Level 1 -- active-probing structure identifier
# ---------------------------------------------------------------------------

# A free, in-bounds workspace pose where the block sits at the centre and is far
# enough from the walls that the T (which extends ~120 px) never touches them.
_BLOCK_CENTER: Vec = (256.0, 200.0)


def probe_agent_law(
    sim: BlackBoxSim,
    *,
    directions: Sequence[Vec] = ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)),
    coast_threshold: float = 0.08,
    vel_threshold: float = 30.0,
) -> Decision:
    """Distinguish a **PD (second-order, velocity-state) agent** from a
    **linear/memoryless (first-order) agent**.

    Experiment: from rest, with the block placed far away (no contact), apply a
    constant push for one control step, then command *hold*. A memoryless agent
    teleports by ``gain*action`` and stops instantly (no velocity state, zero
    coasting). A PD agent accumulates velocity, so it (a) reports a large
    residual velocity after the push and (b) keeps coasting during the hold.
    """
    coast_ratios: List[float] = []
    vel_mags: List[float] = []
    for dx, dy in directions:
        # block parked far from the agent so the probe is contact-free
        bx, by = 60.0, 440.0
        s0 = sim.reset([256.0, 256.0, bx, by, 0.0, 0.0, 0.0])
        p0 = (s0[0], s0[1])
        s1 = sim.step([dx, dy])
        p1 = (s1[0], s1[1])
        v1 = (s1[5], s1[6])
        s2 = sim.step([0.0, 0.0])
        p2 = (s2[0], s2[1])
        active = _dist(p1, p0)
        coast = _dist(p2, p1)
        if active < 1.0:
            continue
        coast_ratios.append(coast / active)
        vel_mags.append(math.hypot(v1[0], v1[1]))

    coast_ratio = sum(coast_ratios) / max(1, len(coast_ratios))
    vel_mag = sum(vel_mags) / max(1, len(vel_mags))
    is_pd = coast_ratio > coast_threshold and vel_mag > vel_threshold
    return Decision(
        question="agent control law (PD/second-order vs linear/memoryless)",
        choice="pd" if is_pd else "linear",
        alternatives=("pd", "linear"),
        statistic=coast_ratio,
        threshold=coast_threshold,
        confident=(coast_ratio > coast_threshold and vel_mag > vel_threshold)
        or (coast_ratio < coast_threshold * 0.5 and vel_mag < vel_threshold),
        evidence={
            "mean_coast_ratio": coast_ratio,
            "mean_residual_speed": vel_mag,
            "vel_threshold": vel_threshold,
            "n_directions": len(coast_ratios),
        },
    )


def _find_contact_config(
    sim: BlackBoxSim,
    *,
    block_center: Vec,
    geometry: Optional[Mapping[str, Any]] = None,
) -> Optional[Tuple[Vec, Vec, float]]:
    """Search for an (agent_offset, push_action) that actually moves the block.

    Returns (offset, action, push_step_disp) for the configuration with the
    largest per-step block displacement, or ``None`` if nothing made contact.
    Geometry, if supplied (perceived from pixels), seeds the search with offsets
    along the surface normals; otherwise a coarse cardinal scan is used.
    """
    bx, by = block_center
    candidates: List[Tuple[Vec, Vec]] = []
    radius = float(geometry["agent_radius"]) if geometry else 15.0
    reach = radius + 30.0
    for off, act in (
        ((-reach, 0.0), (1.0, 0.0)),
        ((reach, 0.0), (-1.0, 0.0)),
        ((0.0, -reach), (0.0, 1.0)),
        ((0.0, reach), (0.0, -1.0)),
        ((0.0, -90.0), (0.0, 1.0)),  # push the T stem from below
    ):
        candidates.append((off, act))

    best: Optional[Tuple[Vec, Vec, float]] = None
    for off, act in candidates:
        s0 = sim.reset([bx + off[0], by + off[1], bx, by, 0.0, 0.0, 0.0])
        b_before = (s0[2], s0[3])
        s = sim.step(act)
        s = sim.step(act)
        b_after = (s[2], s[3])
        push = _dist(b_after, b_before) / 2.0
        if best is None or push > best[2]:
            best = (off, act, push)
    if best is None or best[2] < 1.0:
        return None
    return best


def probe_block_law(
    sim: BlackBoxSim,
    *,
    block_center: Vec = _BLOCK_CENTER,
    geometry: Optional[Mapping[str, Any]] = None,
    residual_threshold: float = 0.15,
) -> Decision:
    """Distinguish a **quasi-static block** (no momentum -- stops the instant
    contact ends) from a **momentum block** (coasts at ~constant velocity after
    contact ends).

    Experiment: find a pushing configuration, get the block moving, then drive
    the agent perpendicular-and-away so contact breaks. Watch the *per-step*
    block displacement after the escape. A quasi-static block decays to ~0 within
    a step or two; a momentum block (zero damping) keeps translating at a roughly
    constant rate. The statistic is the steady-state residual displacement
    relative to the pushing displacement.
    """
    cfg = _find_contact_config(sim, block_center=block_center, geometry=geometry)
    if cfg is None:
        return Decision(
            question="block dynamics (quasi-static vs momentum)",
            choice="unknown",
            alternatives=("quasi_static", "momentum"),
            statistic=float("nan"),
            threshold=residual_threshold,
            confident=False,
            evidence={"error": "no contact configuration found"},
        )
    off, act, _ = cfg
    bx, by = block_center
    s0 = sim.reset([bx + off[0], by + off[1], bx, by, 0.0, 0.0, 0.0])
    b0 = (s0[2], s0[3])
    s = sim.step(act)
    s = sim.step(act)
    b_push = (s[2], s[3])
    push_step = _dist(b_push, b0) / 2.0

    # escape: one perpendicular+away push, then hold and watch the block settle
    perp = (act[1], -act[0])
    escape = (perp[0] - act[0], perp[1] - act[1])
    prev = b_push
    per_step: List[float] = []
    for a in (escape, (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)):
        s = sim.step(a)
        cur = (s[2], s[3])
        per_step.append(_dist(cur, prev))
        prev = cur

    steady = sum(per_step[-2:]) / 2.0  # last two hold steps = steady state
    residual_ratio = steady / max(push_step, 1e-6)
    is_quasi_static = residual_ratio < residual_threshold
    return Decision(
        question="block dynamics (quasi-static vs momentum)",
        choice="quasi_static" if is_quasi_static else "momentum",
        alternatives=("quasi_static", "momentum"),
        statistic=residual_ratio,
        threshold=residual_threshold,
        confident=residual_ratio < residual_threshold * 0.6
        or residual_ratio > residual_threshold * 2.0,
        evidence={
            "push_step_disp": push_step,
            "post_escape_per_step": per_step,
            "steady_state_disp": steady,
            "residual_ratio": residual_ratio,
            "config_offset": list(off),
            "config_action": list(act),
        },
    )


def probe_contact_law(
    sim: BlackBoxSim,
    *,
    block_center: Vec = _BLOCK_CENTER,
    move_threshold: float = 1.0,
) -> Decision:
    """Verify the block moves **only through contact** (no action directly drives
    it). Experiment: park the agent far from the block, move it around for
    several steps in free space, and confirm the block does not move.
    """
    bx, by = block_center
    s0 = sim.reset([bx - 180.0, by, bx, by, 0.0, 0.0, 0.0])
    b0 = (s0[2], s0[3])
    s = s0
    for _ in range(3):
        s = sim.step([-1.0, 0.0])  # drive further away from the block
    moved = _dist((s[2], s[3]), b0)
    is_contact_driven = moved < move_threshold
    return Decision(
        question="block actuation (contact-driven vs action-driven)",
        choice="contact_driven" if is_contact_driven else "action_driven",
        alternatives=("contact_driven", "action_driven"),
        statistic=moved,
        threshold=move_threshold,
        confident=moved < move_threshold * 0.5 or moved > move_threshold * 3.0,
        evidence={"block_disp_with_agent_far": moved},
    )


def identify_structure(
    sim: Optional[BlackBoxSim] = None,
    *,
    geometry: Optional[Mapping[str, Any]] = None,
) -> StructureReport:
    """Run the active-probing identifier and return the discovered structure.

    If ``sim`` is ``None`` a black-box PushT sim is created (and closed) here.
    """
    owns = sim is None
    if owns:
        sim = BlackBoxSim()
    try:
        agent = probe_agent_law(sim)
        block = probe_block_law(sim, geometry=geometry)
        contact = probe_contact_law(sim)
    finally:
        if owns:
            sim.close()
    return StructureReport(agent_law=agent, block_law=block, contact_law=contact)


# ---------------------------------------------------------------------------
# Level 2 -- dispatch discovered structure to a parameter fitter / template
# ---------------------------------------------------------------------------


def _induce_pd_quasistatic(trajectories, *, geometry_json=None, **fit_kwargs) -> str:
    from .physics_sim_pd_dynamics import generate_pd_physics_model_code

    return generate_pd_physics_model_code(trajectories, geometry_json=geometry_json, **fit_kwargs)


def _induce_generic_contact(trajectories, *, geometry_json=None, **fit_kwargs) -> str:
    # Fallback for structures we have not yet fit a faithful template to:
    # the smooth geometry-aware contact-response law (best generic CEM planner).
    from .pushing_mechanics_rollout import generate_collision_rollout_model_code

    allowed = {"rollout_k", "steps", "use_l34", "gate_width", "reg_lambda"}
    kwargs = {k: v for k, v in fit_kwargs.items() if k in allowed}
    return generate_collision_rollout_model_code(trajectories, geometry_json=geometry_json, **kwargs)


# Maps a discovered (agent, block, contact) sketch to a Level-2 template/fitter.
SKETCH_DISPATCH: Dict[Tuple[str, str, str], Tuple[str, Callable[..., str]]] = {
    ("pd", "quasi_static", "contact_driven"): ("pusht_pd_physics", _induce_pd_quasistatic),
}


def select_template(report: StructureReport) -> Tuple[str, Callable[..., str]]:
    """Pick the Level-2 template/fitter for a discovered structure.

    Falls back to the generic geometry-aware contact model when the discovered
    sketch has no faithful template yet (so induction always produces a model).
    """
    return SKETCH_DISPATCH.get(report.sketch, ("pusht_pushing_mechanics_collision_rollout", _induce_generic_contact))


def induce_program(
    trajectories: Sequence[Any],
    *,
    sim: Optional[BlackBoxSim] = None,
    geometry: Optional[Mapping[str, Any]] = None,
    geometry_json: Optional[str] = None,
    report: Optional[StructureReport] = None,
    **fit_kwargs: Any,
) -> Dict[str, Any]:
    """Full two-level induction: discover structure (L1), then fit params (L2).

    Returns ``{"structure": <json>, "template": <name>, "code": <python>}``.
    """
    if geometry is None and geometry_json:
        from . import collision

        geometry = collision.load_perceived_geometry(geometry_json)
    if report is None:
        report = identify_structure(sim, geometry=geometry)
    template_name, fitter = select_template(report)
    code = fitter(trajectories, geometry_json=geometry_json, **fit_kwargs)
    return {
        "structure": report.to_json(),
        "template": template_name,
        "code": code,
    }
