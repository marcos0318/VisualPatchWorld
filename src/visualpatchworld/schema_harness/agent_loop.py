"""Schema-inspired VPW loop: sticky accepted theory + soft void + promotion-only certify.

Adopted from Schema:
  - append-only Timeline (ground truth)
  - deliberation separate from commit_actions
  - soft void when reality disagrees (replan; do not burn budget on random explore)
  - persistent world_model.accepted.py + notes.md
  - certify/backtest only for *promoting* a better theory

Rejected from naive Schema port (Phase C failure):
  - binary certify gate that disables planning
  - uncertified → random explore
  - fit/certify on polluted live Timeline without curation
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from visualpatchworld.code_baselines.common import load_model, load_trajectory_glob, subsample_trajectories
from visualpatchworld.code_baselines.runner import DEFAULT_MODEL
from visualpatchworld.general_planning_eval import plan_cem_general
from visualpatchworld.hdf5 import require_h5py, resolve_dataset_path
from visualpatchworld.lewm_env_specs import get_env_spec, make_gym_env
from visualpatchworld.lewm_graph import (
    goal_distance,
    goal_row_from_start,
    graph_text_from_row,
    is_success,
    reset_env_from_rows,
    row_dict_from_h5,
)
from visualpatchworld.planning_eval import select_lewm_eval_starts

from .backtest import residual_backtest
from .commit import commit_actions, discovery_probe_actions
from .discovery import run_structure_probes, structure_sketch_from_probes
from .revision import revise_theory
from .theorize import deliberate
from .timeline import Timeline


def _global_index(h5: Any, episode_index: int, local_step: int) -> int:
    offset = int(h5["ep_offset"][episode_index])
    return offset + local_step


@dataclass
class SchemaHarnessConfig:
    env: str = "tworoom"
    mode: str = "scaffold"  # scaffold | llm | free | init | sticky
    seed: int = 42
    num_eval: int = 5
    eval_budget: int = 50
    residual_epsilon: float = 0.5
    pass_threshold: float = 0.8
    seed_timeline_transitions: int = 40
    discovery_macros: int = 0
    max_deliberation_rounds: int = 3
    model_name: str = DEFAULT_MODEL
    num_samples: int = 100
    cem_iters: int = 5
    elite_frac: float = 0.1
    train_glob: str = "artifacts/visualpatchworld/data/tworoom_oracle_traj_train.jsonl"
    output_dir: str = "artifacts/visualpatchworld/schema_harness/smoke"
    require_certify: bool = True  # legacy flag; sticky mode ignores as plan gate
    init_model_path: str = ""
    soft_void: bool = True
    revise_every_surprises: int = 4  # end-of-episode revision if enough surprises
    max_surprise_frac: float = 0.3
    run_structure_probes: bool = True
    revision_mode: str = "scaffold"  # scaffold | llm | modular (PushT module-local)
    revise_at_end: bool = True  # one final promotion-gated revision after all episodes


@dataclass
class EpisodeLog:
    trial_id: int
    success: bool
    goal_distance: float
    env_macros: int
    surprises: int
    voids: int
    deliberations: int
    llm_calls: int
    used_accepted: bool
    promotions: int
    backtest: Dict[str, Any] = field(default_factory=dict)


def _save_accepted(code: str, out_dir: Path) -> Path:
    path = out_dir / "world_model.accepted.py"
    path.write_text(code, encoding="utf-8")
    shutil.copyfile(path, out_dir / "world_model.py")
    return path


def run_schema_harness_eval(cfg: SchemaHarnessConfig) -> Dict[str, Any]:
    spec = get_env_spec(cfg.env)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    notes_path = out_dir / "notes.md"
    timeline_path = out_dir / "timeline.json"
    accepted_path = out_dir / "world_model.accepted.py"

    timeline = Timeline(env=cfg.env)
    if cfg.seed_timeline_transitions > 0:
        train = load_trajectory_glob(cfg.train_glob)
        seeded = subsample_trajectories(train, min(8, len(train)), cfg.seed)
        timeline.extend_from_trajectories(
            seeded, max_transitions=cfg.seed_timeline_transitions, source="seed"
        )
        timeline.add_note(f"seeded {timeline.n_transitions} transitions from {cfg.train_glob}")

    free_mode = cfg.mode == "free"
    # sticky: Schema-adopted planning with accepted theory (init/scaffold/llm/sticky)
    sticky = not free_mode
    soft_void = bool(cfg.soft_void and sticky)

    frameskip = spec.frameskip
    horizon = spec.planning_horizon
    receding = spec.planning_receding
    macro_budget = max(1, cfg.eval_budget // frameskip)

    accepted_code = ""
    accepted_model = None
    accepted_report = None
    llm_calls_total = 0
    promotions_total = 0
    revision_reports: List[Dict[str, Any]] = []
    structure_sketch: Dict[str, str] = {}

    # Level-1 structure discovery before the first Level-2 fit when requested.
    if cfg.run_structure_probes:
        try:
            from visualpatchworld.structure_induction import identify_env_structure

            report = identify_env_structure(cfg.env, seed=cfg.seed)
            structure_sketch = dict(report.sketch)
            timeline.add_note(
                f"prefit_structure_sketch={structure_sketch} template={report.template}"
            )
        except Exception as exc:  # noqa: BLE001 — probing is best-effort before fit
            timeline.add_note(f"prefit_structure_probe_failed: {exc}")

    if cfg.init_model_path:
        accepted_code = Path(cfg.init_model_path).read_text(encoding="utf-8")
        _save_accepted(accepted_code, out_dir)
        accepted_model, err = load_model(accepted_code)
        if accepted_model is None:
            raise RuntimeError(f"failed to load init model: {err}")
        fit_trajs = timeline.curated_for_fit(max_surprise_frac=cfg.max_surprise_frac)
        accepted_report = residual_backtest(
            accepted_code,
            fit_trajs or timeline.as_trajectories(),
            epsilon=cfg.residual_epsilon,
            pass_threshold=cfg.pass_threshold,
        )
        timeline.add_note(
            f"accepted init model pass={accepted_report.pass_rate:.3f} "
            f"median={accepted_report.median_residual:.4f}"
        )
    elif timeline.n_transitions > 0:
        theorize_mode = "scaffold" if cfg.mode in ("free", "sticky", "init") else cfg.mode
        accepted_code, accepted_model, accepted_report, llm_n = deliberate(
            timeline,
            cfg.env,
            mode=theorize_mode,
            epsilon=cfg.residual_epsilon,
            pass_threshold=cfg.pass_threshold,
            model_name=cfg.model_name,
            max_deliberation_rounds=cfg.max_deliberation_rounds,
            world_model_path=accepted_path,
            notes_path=notes_path,
            structure_sketch=structure_sketch or None,
        )
        llm_calls_total += llm_n
        _save_accepted(accepted_code, out_dir)

    if accepted_model is None:
        raise RuntimeError("no accepted world model available for planning")

    notes_path.write_text("\n".join(timeline.notes) + "\n", encoding="utf-8")

    dataset_path = resolve_dataset_path(spec.dataset, None)
    h5py = require_h5py()
    env = make_gym_env(spec)
    episode_logs: List[EpisodeLog] = []
    rng = np.random.default_rng(cfg.seed)

    try:
        with h5py.File(dataset_path, "r") as h5:
            starts = select_lewm_eval_starts(
                h5,
                num_eval=cfg.num_eval,
                goal_offset_steps=spec.goal_offset_steps,
                seed=cfg.seed,
            )
            for trial_id, start in enumerate(starts):
                start_global = _global_index(h5, start.episode_index, start.start_step)
                start_row = row_dict_from_h5(h5, start_global, spec)
                goal_row = goal_row_from_start(
                    h5, spec, start.episode_index, start.start_step, spec.goal_offset_steps
                )
                reset_env_from_rows(env, spec, start_row, goal_row)
                current_row = dict(start_row)
                current_obs = graph_text_from_row(
                    spec.name, start_row, source="oracle", step=0, metadata={"episode_index": start.episode_index}
                )

                steps_taken = 0
                surprises = 0
                voids = 0
                deliberations = 0
                episode_promotions = 0

                # Schema-style discriminating experiments once at the first episode.
                if trial_id == 0 and cfg.run_structure_probes:
                    probe_results = run_structure_probes(
                        env_name=cfg.env,
                        start_row=start_row,
                        goal_row=goal_row,
                        timeline=timeline,
                        env=env,
                    )
                    structure_sketch = structure_sketch_from_probes(probe_results)
                    timeline.add_note(f"structure_sketch={structure_sketch}")
                    # Restore episode start after probes.
                    reset_env_from_rows(env, spec, start_row, goal_row)
                    current_row = dict(start_row)
                    current_obs = graph_text_from_row(
                        spec.name, start_row, source="oracle", step=0, metadata={"episode_index": start.episode_index}
                    )

                if cfg.discovery_macros > 0 and trial_id == 0:
                    probes = discovery_probe_actions(spec, rng, n_macros=cfg.discovery_macros)
                    probe_result = commit_actions(
                        env=env,
                        spec=spec,
                        model=accepted_model,
                        timeline=timeline,
                        current_obs=current_obs,
                        current_row=current_row,
                        action_matrix=probes,
                        frameskip=frameskip,
                        residual_epsilon=cfg.residual_epsilon,
                        soft_void=False,
                        source="probe",
                    )
                    current_obs = probe_result.final_obs_text
                    current_row = probe_result.final_row
                    steps_taken += probe_result.steps_taken_macros

                while steps_taken < macro_budget:
                    # Always plan with sticky accepted model (Schema plan-inside-theory,
                    # without Phase-C plan-permission gate).
                    macro_horizon = min(horizon, macro_budget - steps_taken, receding)
                    plan_values, _, _ = plan_cem_general(
                        accepted_model,
                        current_obs,
                        goal_row,
                        current_row,
                        spec,
                        horizon=macro_horizon,
                        num_samples=cfg.num_samples,
                        cem_iters=cfg.cem_iters,
                        elite_frac=cfg.elite_frac,
                        seed=cfg.seed + trial_id + steps_taken,
                        action_low=-1.0,
                        action_high=1.0,
                    )
                    result = commit_actions(
                        env=env,
                        spec=spec,
                        model=accepted_model,
                        timeline=timeline,
                        current_obs=current_obs,
                        current_row=current_row,
                        action_matrix=plan_values[:macro_horizon],
                        frameskip=frameskip,
                        residual_epsilon=cfg.residual_epsilon,
                        soft_void=soft_void,
                        source="live",
                    )
                    current_obs = result.final_obs_text
                    current_row = result.final_row
                    steps_taken += result.steps_taken_macros
                    surprises += result.surprise_count
                    if result.voided:
                        voids += 1
                        # Soft void: replan next iteration with same accepted model.
                    if is_success(spec.name, goal_row, current_row):
                        break

                # End-of-episode revision (Schema deliberate after surprises), sticky planning.
                if sticky and surprises >= cfg.revise_every_surprises:
                    deliberations += 1
                    rev_mode = cfg.revision_mode if cfg.mode != "llm" else "llm"
                    if cfg.env == "pusht" and rev_mode == "scaffold":
                        rev_mode = "modular"
                    accepted_code, accepted_model, accepted_report, rev, llm_n = revise_theory(
                        timeline,
                        cfg.env,
                        accepted_code=accepted_code,
                        accepted_report=accepted_report,
                        mode=rev_mode,
                        residual_epsilon=cfg.residual_epsilon,
                        pass_threshold=cfg.pass_threshold,
                        max_surprise_frac=cfg.max_surprise_frac,
                        min_surprises=cfg.revise_every_surprises,
                        model_name=cfg.model_name,
                        out_dir=out_dir,
                    )
                    llm_calls_total += llm_n
                    revision_reports.append(rev.to_dict())
                    notes_path.write_text("\n".join(timeline.notes) + "\n", encoding="utf-8")
                    if rev.promoted:
                        episode_promotions += 1
                        promotions_total += 1

                dist = goal_distance(spec.name, goal_row, current_row)
                ok = bool(is_success(spec.name, goal_row, current_row))
                episode_logs.append(
                    EpisodeLog(
                        trial_id=trial_id,
                        success=ok,
                        goal_distance=float(dist),
                        env_macros=steps_taken,
                        surprises=surprises,
                        voids=voids,
                        deliberations=deliberations,
                        llm_calls=llm_calls_total,
                        used_accepted=True,
                        promotions=episode_promotions,
                        backtest=accepted_report.to_dict() if accepted_report else {},
                    )
                )
                print(
                    f"[schema_harness] trial={trial_id+1}/{len(starts)} success={ok} "
                    f"dist={dist:.3f} macros={steps_taken} voids={voids} surprises={surprises} "
                    f"promotions={episode_promotions}",
                    flush=True,
                )
        # Final promotion-gated revision after the eval pass (uses accumulated surprises).
        if sticky and cfg.revise_at_end:
            end_mode = cfg.revision_mode if cfg.mode != "llm" else "llm"
            if cfg.env == "pusht" and end_mode == "scaffold":
                end_mode = "modular"
            accepted_code, accepted_model, accepted_report, rev, llm_n = revise_theory(
                timeline,
                cfg.env,
                accepted_code=accepted_code,
                accepted_report=accepted_report,
                mode=end_mode,
                residual_epsilon=cfg.residual_epsilon,
                pass_threshold=cfg.pass_threshold,
                max_surprise_frac=cfg.max_surprise_frac,
                min_surprises=max(1, cfg.revise_every_surprises),
                model_name=cfg.model_name,
                out_dir=out_dir,
            )
            llm_calls_total += llm_n
            revision_reports.append({"final": True, **rev.to_dict()})
            if rev.promoted:
                promotions_total += 1
    finally:
        env.close()
        timeline.save(timeline_path)
        notes_path.write_text("\n".join(timeline.notes) + "\n", encoding="utf-8")

    successes = [1.0 if e.success else 0.0 for e in episode_logs]
    summary = {
        "env": cfg.env,
        "mode": cfg.mode,
        "design": "sticky_soft_void_probes_revision_promotion",
        "soft_void": soft_void,
        "structure_sketch": structure_sketch,
        "revision_reports": revision_reports,
        "num_eval": cfg.num_eval,
        "success_rate": 100.0 * float(np.mean(successes)) if successes else 0.0,
        "mean_goal_distance": float(np.mean([e.goal_distance for e in episode_logs])) if episode_logs else None,
        "mean_env_macros": float(np.mean([e.env_macros for e in episode_logs])) if episode_logs else None,
        "total_surprises": int(sum(e.surprises for e in episode_logs)),
        "total_voids": int(sum(e.voids for e in episode_logs)),
        "total_deliberations": int(sum(e.deliberations for e in episode_logs)),
        "total_promotions": promotions_total,
        "llm_calls": llm_calls_total,
        "residual_epsilon": cfg.residual_epsilon,
        "pass_threshold": cfg.pass_threshold,
        "seed": cfg.seed,
        "final_backtest": accepted_report.to_dict() if accepted_report else {},
        "episodes": [asdict(e) for e in episode_logs],
        "phase_a_reference": {
            "tracepatch_oracle_n50": 98.0,
            "worldcoder_oracle_n50": 54.0,
            "vpw_oracle_induced_n50": 96.0,
            "phase_c_naive_schema": 88.0,
            "phase_c_free": 98.0,
        },
        "world_model_path": str(accepted_path),
        "timeline_path": str(timeline_path),
        "notes_path": str(notes_path),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
