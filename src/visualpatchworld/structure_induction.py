"""Multi-environment Level-1 structure discovery + Level-2 template fit.

PushT keeps the original ``program_induction`` probes. Two-room, Cube, and
Reacher use black-box ``reset``/``step`` probes in ``schema_harness.discovery``
to select among a small sketch family, then dispatch to the matching Level-2
parameter-fit template (linear scaffold, cube contact/grasp, or reacher joint+FK).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from visualpatchworld.general_dynamics import (
    generate_cube_contact_scaffold_model_code,
    generate_general_scaffold_model_code,
)
from visualpatchworld.hdf5 import episode_slices, require_h5py, resolve_dataset_path
from visualpatchworld.lewm_env_specs import get_env_spec
from visualpatchworld.lewm_graph import goal_row_from_start, row_dict_from_h5
from visualpatchworld.reacher_joint_dynamics import generate_reacher_joint_model_code
from visualpatchworld.schema_harness.discovery import (
    ProbeResult,
    run_structure_probes,
    structure_sketch_from_probes,
)


def _first_valid_start(h5: Any, *, goal_offset_steps: int, seed: int = 42) -> Tuple[int, int]:
    """Return (episode_index, start_step) without importing planning_eval/torch."""
    import numpy as np

    col_name = "episode_idx" if "episode_idx" in h5 else "ep_idx"
    episode_ids = np.asarray(h5[col_name][:])
    step_ids = np.asarray(h5["step_idx"][:])
    slices = episode_slices(h5, fallback_length=len(episode_ids))
    episode_lengths = {sl.episode_index: sl.length for sl in slices}
    valid: List[Tuple[int, int]] = []
    for episode_index, step in zip(episode_ids, step_ids):
        episode_index = int(episode_index)
        step = int(step)
        max_start = episode_lengths.get(episode_index, 0) - goal_offset_steps - 1
        if step <= max_start:
            valid.append((episode_index, step))
    if not valid:
        raise RuntimeError("No valid probe starts in dataset")
    rng = np.random.default_rng(seed)
    return valid[int(rng.integers(0, len(valid)))]


@dataclass
class EnvStructureReport:
    env: str
    probes: List[ProbeResult] = field(default_factory=list)
    sketch: Dict[str, str] = field(default_factory=dict)
    template: str = ""
    canonical: bool = False

    def to_json(self) -> Dict[str, Any]:
        return {
            "env": self.env,
            "probes": [p.to_dict() for p in self.probes],
            "sketch": dict(self.sketch),
            "template": self.template,
            "canonical": self.canonical,
            "summary": self.sketch.get("summary", ""),
        }


def load_probe_rows(
    env_name: str,
    *,
    dataset: Optional[str] = None,
    seed: int = 42,
    prefer_near_contact: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load one (start, goal) pair from the LeWM HDF5 dataset for probing.

    For Cube, prefer a near-contact frame so grip-gate probes are informative.
    """
    import numpy as np

    spec = get_env_spec(env_name)
    path = resolve_dataset_path(dataset or spec.dataset)
    h5py = require_h5py()
    with h5py.File(path, "r") as h5:
        episode_index, start_step = _first_valid_start(
            h5, goal_offset_steps=spec.goal_offset_steps, seed=seed
        )
        if prefer_near_contact and spec.name == "cube":
            eff = np.asarray(h5["proprio_effector_pos"][:], dtype=np.float64)
            blk = np.asarray(h5["privileged_block_0_pos"][:], dtype=np.float64)
            dist = np.linalg.norm(eff - blk, axis=1)
            near = np.flatnonzero(dist < 0.03)
            if near.size > 0:
                rng = np.random.default_rng(seed)
                idx = int(near[int(rng.integers(0, near.size))])
                # Map global index → episode/local step via ep_offset.
                offsets = np.asarray(h5["ep_offset"][:], dtype=np.int64)
                episode_index = int(np.searchsorted(offsets, idx, side="right") - 1)
                start_step = int(idx - offsets[episode_index])
                # Keep a valid goal offset when possible.
                lengths = np.asarray(h5["ep_len"][:], dtype=np.int64) if "ep_len" in h5 else None
                if lengths is not None:
                    max_start = int(lengths[episode_index]) - spec.goal_offset_steps - 1
                    if start_step > max_start:
                        start_step = max(0, max_start)
                        idx = int(offsets[episode_index] + start_step)
                start_row = row_dict_from_h5(h5, idx, spec)
                goal_row = goal_row_from_start(
                    h5, spec, episode_index, start_step, spec.goal_offset_steps
                )
                return start_row, goal_row

        offset = int(h5["ep_offset"][episode_index])
        start_global = offset + int(start_step)
        start_row = row_dict_from_h5(h5, start_global, spec)
        goal_row = goal_row_from_start(
            h5, spec, episode_index, start_step, spec.goal_offset_steps
        )
    return start_row, goal_row


def select_template_name(env_name: str, sketch: Mapping[str, str]) -> Tuple[str, bool]:
    """Map a probe sketch to a Level-2 template name and whether it is canonical GT."""
    env = get_env_spec(env_name).name
    summary = str(sketch.get("summary", ""))

    if env == "pusht":
        canonical = summary == "pd_quasi_static_contact_driven" or summary.startswith("pd_quasi_static_contact")
        # Keep PushT's historical template names.
        if "pd" in summary and "quasi_static" in summary and "contact" in summary:
            return "pusht_pd_physics", True
        return "pusht_pushing_mechanics_collision_rollout", False

    if env == "tworoom":
        canonical = summary == "tworoom_linear_scaffold"
        return ("tworoom_linear_scaffold", canonical)

    if env == "cube":
        if summary == "cube_contact_grasp_gated":
            return "cube_contact_scaffold", True
        return "cube_linear_scaffold", False

    if env == "reacher":
        if summary == "reacher_joint_fk":
            return "reacher_joint_fit", True
        return "reacher_linear_scaffold", False

    raise ValueError(f"No template map for env={env_name!r}")


def _fit_cube_linear(trajectories: Sequence[Any]) -> str:
    return generate_general_scaffold_model_code(trajectories, "cube", force_linear=True)


def _fit_reacher_linear(trajectories: Sequence[Any]) -> str:
    return generate_general_scaffold_model_code(trajectories, "reacher")


def _fit_tworoom(trajectories: Sequence[Any]) -> str:
    return generate_general_scaffold_model_code(trajectories, "tworoom")


def _fit_cube_contact(trajectories: Sequence[Any]) -> str:
    return generate_cube_contact_scaffold_model_code(trajectories)


def _fit_reacher_joint(trajectories: Sequence[Any]) -> str:
    return generate_reacher_joint_model_code(trajectories)


TEMPLATE_FITTERS: Dict[str, Callable[[Sequence[Any]], str]] = {
    "tworoom_linear_scaffold": _fit_tworoom,
    "cube_contact_scaffold": _fit_cube_contact,
    "cube_linear_scaffold": _fit_cube_linear,
    "reacher_joint_fit": _fit_reacher_joint,
    "reacher_linear_scaffold": _fit_reacher_linear,
}


def identify_env_structure(
    env_name: str,
    *,
    start_row: Optional[Mapping[str, Any]] = None,
    goal_row: Optional[Mapping[str, Any]] = None,
    dataset: Optional[str] = None,
    geometry: Optional[Mapping[str, Any]] = None,
    seed: int = 42,
) -> EnvStructureReport:
    """Run Level-1 active probes and select the Level-2 template."""
    env = get_env_spec(env_name).name

    if env == "pusht":
        from visualpatchworld import program_induction as pi

        sim = pi.BlackBoxSim()
        try:
            report = pi.identify_structure(sim, geometry=geometry)
        finally:
            sim.close()
        probes = run_structure_probes(
            env_name="pusht",
            start_row=start_row or {},
            goal_row=goal_row or {},
            geometry=geometry,
        )
        sketch = structure_sketch_from_probes(probes)
        # Prefer the structured sketch tuple from program_induction.
        sketch["summary"] = "_".join(report.sketch)
        sketch["pusht_sketch"] = "_".join(report.sketch)
        template, canonical = select_template_name("pusht", sketch)
        return EnvStructureReport(
            env="pusht",
            probes=probes,
            sketch=sketch,
            template=template,
            canonical=canonical and report.all_confident,
        )

    if start_row is None or goal_row is None:
        start_row, goal_row = load_probe_rows(env, dataset=dataset, seed=seed)

    probes = run_structure_probes(
        env_name=env,
        start_row=start_row,
        goal_row=goal_row,
    )
    sketch = structure_sketch_from_probes(probes)
    template, canonical = select_template_name(env, sketch)
    return EnvStructureReport(
        env=env,
        probes=probes,
        sketch=sketch,
        template=template,
        canonical=canonical,
    )


def induce_from_structure(
    env_name: str,
    trajectories: Sequence[Any],
    *,
    report: Optional[EnvStructureReport] = None,
    geometry_json: Optional[str] = None,
    dataset: Optional[str] = None,
    rollout_k: int = 6,
    steps: int = 400,
    restarts: int = 4,
    val_trajectories: Optional[Sequence[Any]] = None,
) -> Dict[str, Any]:
    """Level-1 (if needed) + Level-2 fit for any LeWM environment."""
    env = get_env_spec(env_name).name
    if report is None:
        geometry = None
        if geometry_json:
            from visualpatchworld import collision

            geometry = collision.load_perceived_geometry(geometry_json)
        report = identify_env_structure(env, dataset=dataset, geometry=geometry)

    if env == "pusht":
        from visualpatchworld import program_induction as pi
        from visualpatchworld import collision as _collision

        # Rebuild a PushT StructureReport for the existing fitter path.
        sim = pi.BlackBoxSim()
        try:
            pusht_report = pi.identify_structure(sim, geometry=_collision.load_perceived_geometry(geometry_json) if geometry_json else None)
        finally:
            sim.close()
        result = pi.induce_program(
            trajectories,
            report=pusht_report,
            geometry_json=geometry_json,
            val_trajectories=val_trajectories,
            rollout_k=rollout_k,
            steps=steps,
            restarts=restarts,
        )
        result["structure"] = report.to_json()
        return result

    fitter = TEMPLATE_FITTERS.get(report.template)
    if fitter is None:
        raise RuntimeError(f"No Level-2 fitter for template {report.template!r}")
    code = fitter(trajectories)
    return {
        "code": code,
        "template": report.template,
        "structure": report.to_json(),
        "env": env,
    }
