"""write_code: induce / repair world-model programs from Timeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from patchworld.worldmodel_data import Trajectory

from visualpatchworld.code_baselines.common import (
    load_model,
    make_inducer,
    sanitize_generated_code,
)
from visualpatchworld.code_baselines.runner import DEFAULT_MODEL, apply_code_baseline_llm_env
from visualpatchworld.general_dynamics import (
    generate_cube_contact_scaffold_model_code,
    generate_general_scaffold_model_code,
)
from visualpatchworld.reacher_joint_dynamics import generate_reacher_joint_model_code

from .backtest import BacktestReport, residual_backtest
from .timeline import Timeline


def write_scaffold_code(
    timeline: Timeline,
    env_name: str,
    *,
    max_surprise_frac: float = 0.3,
    structure_sketch: Optional[Dict[str, str]] = None,
) -> str:
    trajs = timeline.curated_for_fit(max_surprise_frac=max_surprise_frac) or timeline.as_trajectories()
    if not trajs or not trajs[0].transitions:
        raise RuntimeError("Timeline empty; cannot fit scaffold")
    if structure_sketch:
        # Lazy import avoids circular dependency with structure_induction → discovery.
        from visualpatchworld.structure_induction import TEMPLATE_FITTERS, select_template_name

        template, _ = select_template_name(env_name, structure_sketch)
        fitter = TEMPLATE_FITTERS.get(template)
        if fitter is not None:
            return sanitize_generated_code(fitter(trajs))
        if template == "pusht_pd_physics":
            from visualpatchworld.physics_sim_pd_dynamics import generate_pd_physics_model_code

            return sanitize_generated_code(generate_pd_physics_model_code(trajs))
    # Fallback: env defaults (cube → contact scaffold, else linear).
    if env_name == "reacher":
        return sanitize_generated_code(generate_reacher_joint_model_code(trajs))
    if env_name == "cube":
        return sanitize_generated_code(generate_cube_contact_scaffold_model_code(trajs))
    return sanitize_generated_code(generate_general_scaffold_model_code(trajs, env_name))


def write_llm_code(
    timeline: Timeline,
    env_name: str,
    *,
    model_name: str = DEFAULT_MODEL,
    max_refine_rounds: int = 2,
    validation_trajectories: Optional[Sequence[Trajectory]] = None,
    max_surprise_frac: float = 0.3,
) -> str:
    apply_code_baseline_llm_env()
    from patchworld.inducer_llm_client import LLMClient

    trajs = timeline.curated_for_fit(max_surprise_frac=max_surprise_frac) or timeline.as_trajectories()
    if not trajs or not trajs[0].transitions:
        raise RuntimeError("Timeline empty; cannot induce with LLM")
    llm = LLMClient(model=model_name)
    inducer = make_inducer(llm, max_refine_rounds=max_refine_rounds, show_progress=True)
    train = trajs
    val = list(validation_trajectories or trajs)
    code = inducer.tracepatch_induce(env_name, train, validation_trajectories=val)
    return sanitize_generated_code(code)


def deliberate(
    timeline: Timeline,
    env_name: str,
    *,
    mode: str,
    epsilon: float,
    pass_threshold: float,
    model_name: str = DEFAULT_MODEL,
    max_deliberation_rounds: int = 3,
    world_model_path: Optional[Path] = None,
    notes_path: Optional[Path] = None,
    structure_sketch: Optional[Dict[str, str]] = None,
) -> Tuple[str, Any, BacktestReport, int]:
    """Schema inner loop: write_code → run_backtest until certified or rounds exhausted."""
    mode = mode.strip().lower()
    last_code = ""
    last_report = BacktestReport(
        certified=False,
        n_checked=0,
        n_hard_fail=0,
        n_residual_fail=0,
        pass_rate=0.0,
        mean_residual=float("inf"),
        median_residual=float("inf"),
        max_residual=float("inf"),
    )
    llm_calls = 0

    for round_idx in range(max(1, max_deliberation_rounds)):
        if mode == "scaffold":
            last_code = write_scaffold_code(
                timeline, env_name, structure_sketch=structure_sketch
            )
        elif mode in ("llm", "schema", "tracepatch"):
            last_code = write_llm_code(timeline, env_name, model_name=model_name)
            llm_calls += 1
        else:
            raise ValueError(f"Unknown theorize mode {mode!r}")

        fit_trajs = timeline.curated_for_fit() or timeline.as_trajectories()
        last_report = residual_backtest(
            last_code,
            fit_trajs,
            epsilon=epsilon,
            pass_threshold=pass_threshold,
        )
        timeline.add_note(
            f"deliberate round={round_idx} mode={mode} certified={last_report.certified} "
            f"pass_rate={last_report.pass_rate:.3f} mean_resid={last_report.mean_residual:.4f}"
        )
        if world_model_path is not None:
            world_model_path.parent.mkdir(parents=True, exist_ok=True)
            world_model_path.write_text(last_code, encoding="utf-8")
        if notes_path is not None:
            notes_path.parent.mkdir(parents=True, exist_ok=True)
            notes_path.write_text("\n".join(timeline.notes) + "\n", encoding="utf-8")
        if last_report.certified:
            break
        # Scaffold has no stochastic retries that help; stop early.
        if mode == "scaffold":
            break

    model, err = load_model(last_code)
    if model is None:
        raise RuntimeError(f"deliberate produced unloadable model: {err}")
    return last_code, model, last_report, llm_calls
