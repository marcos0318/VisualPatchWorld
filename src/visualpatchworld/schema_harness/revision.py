"""Surprise-driven theory revision with promotion-only acceptance.

Schema: when predictions fail, revise the rule (or representation), then
re-certify before trusting the new theory. VPW continuous form: revise on
curated Timeline, promote only if residual metrics improve vs sticky accepted.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from visualpatchworld.code_baselines.common import load_model, sanitize_generated_code
from visualpatchworld.code_baselines.runner import DEFAULT_MODEL
from visualpatchworld.general_dynamics import generate_general_scaffold_model_code

from .backtest import BacktestReport, residual_backtest, seed_holdout_ok, should_promote
from .timeline import Timeline


@dataclass
class RevisionReport:
    attempted: bool
    promoted: bool
    mode: str
    module_hint: str
    n_surprises: int
    accepted_median: float
    candidate_median: float
    notes: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attempted": self.attempted,
            "promoted": self.promoted,
            "mode": self.mode,
            "module_hint": self.module_hint,
            "n_surprises": self.n_surprises,
            "accepted_median": self.accepted_median,
            "candidate_median": self.candidate_median,
            "notes": list(self.notes),
        }


def diagnose_surprise_module(timeline: Timeline, *, env_name: str) -> str:
    """Cheap module hint from surprise residuals (continuous analogue of Schema bug localization)."""
    surprises = [r for r in timeline.records if r.source == "surprise"]
    if not surprises:
        return "none"
    mean_resid = sum(float(r.residual or 0.0) for r in surprises) / max(1, len(surprises))
    if env_name == "tworoom":
        if mean_resid > 5.0:
            return "agent_law"
        if mean_resid > 0.5:
            return "agent_gain"
        return "noise"
    if env_name == "pusht":
        # Prefer contact module when many high-residual surprises occur near contact.
        near = 0
        far = 0
        try:
            from visualpatchworld.pusht_local_induction import _parse_observation_text, _state_from_graph
            import math

            for r in surprises:
                state = _state_from_graph(_parse_observation_text(r.transition.observation))
                dist = float(math.hypot(state[2] - state[0], state[3] - state[1]))
                if dist <= 70.0:
                    near += 1
                else:
                    far += 1
        except Exception:
            return "contact" if mean_resid > 2.0 else "agent"
        if near >= far:
            return "contact"
        return "agent"
    return "dynamics"


def _surprises_as_weighted(timeline: Timeline) -> List[Any]:
    from visualpatchworld.pusht_local_induction import WeightedTransition

    rows: List[Any] = []
    for r in timeline.records:
        if r.source != "surprise":
            continue
        rows.append(
            WeightedTransition(
                observation=r.transition.observation,
                action=r.transition.action,
                next_observation=r.transition.next_observation,
                weight=float(max(1.0, r.weight)),
            )
        )
    return rows


def _is_pd_physics_code(code: str) -> bool:
    return "PushtPDPhysicsWorldModel" in code or ("KP =" in code and "KV =" in code and "K_N =" in code)


def _is_gated_mlp_code(code: str) -> bool:
    return "COEF_CONTACT" in code and "CONTACT_THRESHOLD" in code and "MLP" in code


def _write_pusht_pd_candidate(timeline: Timeline, *, max_surprise_frac: float) -> str:
    """Re-fit PD physics template on curated Timeline (keeps strong planner class)."""
    from visualpatchworld.physics_sim_pd_dynamics import generate_pd_physics_model_code

    curated = timeline.curated_for_fit(max_surprise_frac=max_surprise_frac)
    if not curated or not curated[0].transitions:
        raise RuntimeError("no curated transitions for PushT PD revision")
    return sanitize_generated_code(generate_pd_physics_model_code(curated))


def _write_pusht_modular_candidate(
    timeline: Timeline,
    *,
    accepted_code: str,
    module_hint: str,
    max_surprise_frac: float,
) -> str:
    """Module-local contact/agent refit under Schema promotion discipline.

    Never silently replace a strong PD physics accepted model with a weak
    modular fit — use PD re-fit instead.
    """
    if _is_pd_physics_code(accepted_code):
        return _write_pusht_pd_candidate(timeline, max_surprise_frac=max_surprise_frac)

    from visualpatchworld.modular_dynamics import (
        generate_modular_agent_refit_model_code,
        generate_modular_contact_refit_model_code,
        load_gated_mlp_params_from_code,
    )

    curated = timeline.curated_for_fit(max_surprise_frac=max_surprise_frac)
    if not curated or not curated[0].transitions:
        raise RuntimeError("no curated transitions for PushT modular revision")
    extras = _surprises_as_weighted(timeline)
    if not _is_gated_mlp_code(accepted_code):
        raise RuntimeError(
            "accepted PushT model is neither PD physics nor gated-MLP; "
            "refusing to replace it with a weak modular bootstrap"
        )
    base_params = load_gated_mlp_params_from_code(accepted_code)

    if module_hint == "agent":
        return sanitize_generated_code(
            generate_modular_agent_refit_model_code(
                curated,
                base_params=base_params,
                extra_transitions=extras,
            )
        )
    return sanitize_generated_code(
        generate_modular_contact_refit_model_code(
            curated,
            base_params=base_params,
            extra_transitions=extras,
            mlp_steps=800,
            mlp_rollout_steps=2,
        )
    )


def _write_candidate_code(
    timeline: Timeline,
    env_name: str,
    *,
    mode: str,
    model_name: str,
    max_surprise_frac: float,
    max_refine_rounds: int,
    accepted_code: str = "",
    module_hint: str = "dynamics",
) -> Tuple[str, int]:
    curated = timeline.curated_for_fit(max_surprise_frac=max_surprise_frac)
    if not curated or not curated[0].transitions:
        raise RuntimeError("no curated transitions for revision")
    llm_calls = 0
    if mode in ("modular", "scaffold") and env_name == "pusht":
        # Prefer module-local refit for PushT exactness.
        code = _write_pusht_modular_candidate(
            timeline,
            accepted_code=accepted_code,
            module_hint=module_hint,
            max_surprise_frac=max_surprise_frac,
        )
        return code, llm_calls
    if mode == "scaffold":
        code = sanitize_generated_code(generate_general_scaffold_model_code(curated, env_name))
        return code, llm_calls
    if mode in ("llm", "tracepatch"):
        from visualpatchworld.code_baselines.common import make_inducer
        from visualpatchworld.code_baselines.runner import apply_code_baseline_llm_env
        from patchworld.inducer_llm_client import LLMClient

        apply_code_baseline_llm_env()
        llm = LLMClient(model=model_name)
        inducer = make_inducer(llm, max_refine_rounds=max_refine_rounds, show_progress=True)
        code = inducer.tracepatch_induce(env_name, curated, validation_trajectories=curated)
        llm_calls = 1
        return sanitize_generated_code(code), llm_calls
    raise ValueError(f"unknown revision mode {mode!r}")


def revise_theory(
    timeline: Timeline,
    env_name: str,
    *,
    accepted_code: str,
    accepted_report: Optional[BacktestReport],
    mode: str = "scaffold",
    residual_epsilon: float = 0.5,
    pass_threshold: float = 0.8,
    max_surprise_frac: float = 0.3,
    min_surprises: int = 4,
    model_name: str = DEFAULT_MODEL,
    max_refine_rounds: int = 2,
    out_dir: Optional[Path] = None,
) -> Tuple[str, Any, BacktestReport, RevisionReport, int]:
    """Propose a revised world model; promote only if curated backtest improves."""
    n_surprises = sum(1 for r in timeline.records if r.source == "surprise")
    module_hint = diagnose_surprise_module(timeline, env_name=env_name)
    notes: List[str] = []
    accepted_model, err = load_model(accepted_code)
    if accepted_model is None:
        raise RuntimeError(f"accepted model unloadable: {err}")
    if accepted_report is None:
        accepted_report = residual_backtest(
            accepted_code,
            timeline.curated_for_fit(max_surprise_frac=max_surprise_frac) or timeline.as_trajectories(),
            epsilon=residual_epsilon,
            pass_threshold=pass_threshold,
        )

    if n_surprises < min_surprises:
        report = RevisionReport(
            attempted=False,
            promoted=False,
            mode=mode,
            module_hint=module_hint,
            n_surprises=n_surprises,
            accepted_median=accepted_report.median_residual,
            candidate_median=accepted_report.median_residual,
            notes=[f"skip_revision n_surprises={n_surprises} < {min_surprises}"],
        )
        return accepted_code, accepted_model, accepted_report, report, 0

    notes.append(f"revise module_hint={module_hint} n_surprises={n_surprises} mode={mode}")
    timeline.add_note(notes[-1])
    try:
        candidate_code, llm_calls = _write_candidate_code(
            timeline,
            env_name,
            mode=mode,
            model_name=model_name,
            max_surprise_frac=max_surprise_frac,
            max_refine_rounds=max_refine_rounds,
            accepted_code=accepted_code,
            module_hint=module_hint,
        )
    except Exception as exc:
        notes.append(f"revision_failed: {exc}")
        timeline.add_note(notes[-1])
        report = RevisionReport(
            attempted=True,
            promoted=False,
            mode=mode,
            module_hint=module_hint,
            n_surprises=n_surprises,
            accepted_median=accepted_report.median_residual,
            candidate_median=float("inf"),
            notes=notes,
        )
        return accepted_code, accepted_model, accepted_report, report, 0

    curated = timeline.curated_for_fit(max_surprise_frac=max_surprise_frac)
    cand_report = residual_backtest(
        candidate_code,
        curated,
        epsilon=residual_epsilon,
        pass_threshold=pass_threshold,
    )
    # Also score accepted on the same curated slice for fair promotion.
    accepted_on_curated = residual_backtest(
        accepted_code,
        curated,
        epsilon=residual_epsilon,
        pass_threshold=pass_threshold,
    )
    promote = should_promote(cand_report, accepted_on_curated, min_pass_rate=pass_threshold)
    # Seed holdout: reject curated-overfit refits that regress the sticky prior.
    seed_only = timeline.curated_for_fit(max_surprise_frac=0.0)
    cand_seed = residual_backtest(
        candidate_code,
        seed_only,
        epsilon=residual_epsilon,
        pass_threshold=pass_threshold,
    )
    accepted_seed = residual_backtest(
        accepted_code,
        seed_only,
        epsilon=residual_epsilon,
        pass_threshold=pass_threshold,
    )
    holdout = seed_holdout_ok(cand_seed, accepted_seed)
    if promote and not holdout:
        notes.append(
            f"seed_holdout_block cand_seed median={cand_seed.median_residual:.4f} "
            f"pass={cand_seed.pass_rate:.3f} accepted_seed median={accepted_seed.median_residual:.4f} "
            f"pass={accepted_seed.pass_rate:.3f}"
        )
        timeline.add_note(notes[-1])
        promote = False
    notes.append(
        f"candidate median={cand_report.median_residual:.4f} pass={cand_report.pass_rate:.3f} "
        f"accepted_curated median={accepted_on_curated.median_residual:.4f} "
        f"seed_holdout={holdout} promote={promote}"
    )
    timeline.add_note(notes[-1])

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "world_model.candidate.py").write_text(candidate_code, encoding="utf-8")
        (out_dir / "notes.md").write_text("\n".join(timeline.notes) + "\n", encoding="utf-8")

    if promote:
        if out_dir is not None:
            accepted_path = Path(out_dir) / "world_model.accepted.py"
            accepted_path.write_text(candidate_code, encoding="utf-8")
            shutil.copyfile(accepted_path, Path(out_dir) / "world_model.py")
        model, err = load_model(candidate_code)
        if model is None:
            notes.append(f"promote_load_failed: {err}")
            report = RevisionReport(
                attempted=True,
                promoted=False,
                mode=mode,
                module_hint=module_hint,
                n_surprises=n_surprises,
                accepted_median=accepted_on_curated.median_residual,
                candidate_median=cand_report.median_residual,
                notes=notes,
            )
            return accepted_code, accepted_model, accepted_report, report, llm_calls
        timeline.add_note("promoted candidate → accepted")
        report = RevisionReport(
            attempted=True,
            promoted=True,
            mode=mode,
            module_hint=module_hint,
            n_surprises=n_surprises,
            accepted_median=accepted_on_curated.median_residual,
            candidate_median=cand_report.median_residual,
            notes=notes,
        )
        return candidate_code, model, cand_report, report, llm_calls

    report = RevisionReport(
        attempted=True,
        promoted=False,
        mode=mode,
        module_hint=module_hint,
        n_surprises=n_surprises,
        accepted_median=accepted_on_curated.median_residual,
        candidate_median=cand_report.median_residual,
        notes=notes,
    )
    return accepted_code, accepted_model, accepted_report, report, llm_calls
