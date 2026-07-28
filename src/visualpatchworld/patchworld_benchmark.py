"""Local PatchWorld benchmark runner for VisualPatchWorld.

This is adapted from PatchWorld's benchmark CLI, with PushT registration kept
inside `le-wm-code` via `patchworld_adapter.apply_pusht_patchworld_adapter()`.
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Dict, List

from visualpatchworld.patchworld_adapter import apply_pusht_patchworld_adapter, sanitize_pusht_model_code
from visualpatchworld.pusht_local_induction import (
    generate_pusht_kinematic_model_code,
    generate_pusht_scaffold_fit_contact_model_code,
    generate_pusht_scaffold_fit_contact_gated_mlp_model_code,
    generate_pusht_scaffold_fit_contact_gated_mlp_weighted_model_code,
    generate_pusht_scaffold_fit_contact_gated_model_code,
    generate_pusht_scaffold_fit_contact_gated_weighted_model_code,
    generate_pusht_scaffold_fit_contact_weighted_model_code,
    generate_pusht_scaffold_fit_model_code,
    iter_weighted_transitions,
    load_weighted_transitions_jsonl_paths,
)
from visualpatchworld.counterexample_store import build_capped_extra_transitions, parse_module_filter
from visualpatchworld.modular_dynamics import (
    generate_modular_agent_refit_model_code,
    generate_modular_contact_refit_model_code,
    generate_modular_offline_gated_mlp_model_code,
    load_gated_mlp_params_from_results_dir,
)
from visualpatchworld.compositional_dynamics import (
    generate_compositional_contact_refit_model_code,
    generate_compositional_offline_model_code,
    load_compositional_params_from_results_dir,
)
from visualpatchworld.pushing_mechanics_dynamics import (
    generate_pushing_mechanics_model_code,
)
from visualpatchworld.pushing_mechanics_collision import (
    generate_collision_model_code,
)
from visualpatchworld.pushing_mechanics_rollout import (
    generate_collision_rollout_model_code,
)
from visualpatchworld.physics_sim_dynamics import (
    generate_physics_sim_model_code,
)
from visualpatchworld.physics_sim_pd_dynamics import (
    generate_pd_physics_model_code,
)
from visualpatchworld import program_induction as program_induction
from visualpatchworld.general_dynamics import generate_general_scaffold_model_code
from visualpatchworld.reacher_joint_dynamics import generate_reacher_joint_model_code

apply_pusht_patchworld_adapter()

from patchworld.inducer_llm_client import LLMClient
from patchworld.residual_memory import EpisodicResidualWorldModel
from patchworld.worldmodel_data import Trajectory, get_stats, load_trajectories
from patchworld.worldmodel_evaluator import (
    compute_autoregressive_rollout_token_f1,
    compute_experience_fit,
    compute_observation_fit,
)
from patchworld.worldmodel_inducer import Inducer
from patchworld.worldmodel_validator import load_model_from_code


DEFAULT_PATCHWORLD_MODEL = "Qwen/Qwen3-Coder-480B-A35B-Instruct-Turbo"


def _log(message: str) -> None:
    print(f"[visualpatchworld][{datetime.now().isoformat(timespec='seconds')}][induce] {message}", flush=True)


def _load_paths(pattern: str) -> List[Path]:
    paths = [Path(p) for p in glob(pattern)]
    return [p for p in sorted(paths) if p.is_file() and p.suffix == ".jsonl"]


def _load_trajs(pattern: str) -> List[Trajectory]:
    paths = _load_paths(pattern)
    if not paths:
        raise SystemExit(f"No trajectory files matched: {pattern}")
    trajs = load_trajectories(paths)
    if not trajs:
        raise SystemExit(f"No trajectories loaded from: {pattern}")
    return trajs


def _subsample(trajs: List[Trajectory], max_count: int, seed: int) -> List[Trajectory]:
    if max_count <= 0 or len(trajs) <= max_count:
        return trajs
    rnd = random.Random(seed)
    return rnd.sample(trajs, max_count)


def _evaluate_model(model: object, train: List[Trajectory], eval_: List[Trajectory], env: str) -> Dict[str, Dict]:
    return {
        "train": {
            "experience_fit": compute_experience_fit(model, train),
            "observation_fit": compute_observation_fit(model, train, env_name=env),
        },
        "eval": {
            "experience_fit": compute_experience_fit(model, eval_),
            "observation_fit": compute_observation_fit(model, eval_, env_name=env),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VisualPatchWorld-local PatchWorld benchmark runner.")
    parser.add_argument("--env", default="pusht")
    parser.add_argument("--train_glob", required=True)
    parser.add_argument("--eval_glob", required=True)
    parser.add_argument("--output_dir", default="artifacts/visualpatchworld/results")
    parser.add_argument("--max_train", type=int, default=0)
    parser.add_argument("--max_eval", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default=DEFAULT_PATCHWORLD_MODEL)
    parser.add_argument(
        "--induction_strategy",
        default="tracepatch",
        choices=[
            "tracepatch",
            "naive_induction",
            "pusht_kinematic",
            "pusht_scaffold_fit",
            "pusht_scaffold_fit_contact",
            "pusht_scaffold_fit_contact_weighted",
            "pusht_scaffold_fit_contact_gated",
            "pusht_scaffold_fit_contact_gated_weighted",
            "pusht_scaffold_fit_contact_gated_mlp",
            "pusht_scaffold_fit_contact_gated_mlp_weighted",
            "pusht_modular_gated_mlp_offline",
            "pusht_modular_gated_mlp_contact_refit",
            "pusht_modular_gated_mlp_agent_refit",
            "pusht_modular_gated_mlp_oracle_offline",
            "pusht_modular_gated_mlp_oracle_contact_refit",
            "pusht_compositional_gated_mlp_oracle_offline",
            "pusht_compositional_gated_mlp_oracle_contact_refit",
            "pusht_pushing_mechanics",
            "pusht_pushing_mechanics_collision",
            "pusht_pushing_mechanics_collision_rollout",
            "pusht_physics_sim",
            "pusht_pd_physics",
            "pusht_program_induction",
            "program_induction",
            "general_scaffold_fit",
            "reacher_joint_fit",
        ],
    )
    parser.add_argument(
        "--failure_transitions_jsonl",
        default=None,
        help="Weighted failure transitions for weighted contact/gated strategies.",
    )
    parser.add_argument(
        "--base_results_dir",
        default=None,
        help="Base induced model directory for modular contact/agent refit.",
    )
    parser.add_argument(
        "--counterexample_store",
        default=None,
        help="Comma-separated module-tagged counterexample JSONLs for modular refit.",
    )
    parser.add_argument(
        "--geometry_json",
        default=None,
        help="Perceived collision geometry JSON (from detect_geometry.py) for "
        "pusht_pushing_mechanics_collision. If omitted, oracle geometry is used (validation only).",
    )
    parser.add_argument("--rollout_k", type=int, default=6, help="K-step rollout horizon for rollout-loss fit.")
    parser.add_argument("--rollout_fit_steps", type=int, default=400, help="Adam steps for rollout-loss fit.")
    parser.add_argument("--rollout_gate_width", type=float, default=2.5, help="Soft contact-gate width (larger = smoother, more planner-friendly).")
    parser.add_argument("--rollout_reg_lambda", type=float, default=20.0, help="L2 toward the one-step solution in the rollout fit (larger = smoother).")
    parser.add_argument(
        "--use_l34",
        action="store_true",
        help="Enable L3 contact-resist + L4 non-penetration projection in the rollout fit. "
        "Off by default: on PushT they rarely trigger and perturb the fit to a worse solution.",
    )
    parser.add_argument("--max_extra_fraction", type=float, default=0.3)
    parser.add_argument("--counterexample_modules", default="contact,contact_angle")
    parser.add_argument("--exclude_counterexample_modules", default="contact_gate")
    parser.add_argument("--mlp_rollout_steps", type=int, default=0, help="0 = strategy default")
    parser.add_argument("--mlp_hidden_dim", type=int, default=0, help="0 = strategy default")
    parser.add_argument("--max_refine_rounds", type=int, default=4)
    parser.add_argument("--contrastive_max_per_pattern", type=int, default=5)
    parser.add_argument("--contrastive_max_transitions", type=int, default=60)
    parser.add_argument("--patch_candidates_per_round", type=int, default=4)
    parser.add_argument("--patch_beam_size", type=int, default=2)
    parser.add_argument("--patch_no_improve_patience", type=int, default=2)
    parser.add_argument("--initial_load_error_retries", type=int, default=5)
    parser.add_argument("--run_rollout", action="store_true")
    parser.add_argument("--rollout_horizon", type=int, default=15)
    parser.add_argument("--rollout_steps", default="1,2,3,5,8,10,15")
    parser.add_argument("--use_residual_memory", action="store_true")
    parser.add_argument("--residual_memory_max_entries", type=int, default=0)
    parser.add_argument("--residual_memory_min_count", type=int, default=1)
    parser.add_argument("--residual_memory_min_confidence", type=float, default=1.0)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    _log(f"Loading train trajectories: {args.train_glob}")
    train_all = _load_trajs(args.train_glob)
    _log(f"Loading eval trajectories: {args.eval_glob}")
    eval_all = _load_trajs(args.eval_glob)
    train_trajs = _subsample(train_all, args.max_train, args.seed)
    eval_trajs = _subsample(eval_all, args.max_eval, args.seed + 1)
    _log(
        f"Loaded train_pool={len(train_all)} eval_pool={len(eval_all)}; "
        f"using train={len(train_trajs)} eval={len(eval_trajs)}"
    )

    out_dir = Path(args.output_dir)
    model_dir = out_dir / "generated_models"
    model_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = model_dir / f"{args.env}_patchworld_{ts}.py"

    llm = None
    inducer = None
    if args.induction_strategy == "program_induction":
        _log(
            f"Starting two-level program induction for env={args.env}: "
            "Level-1 active-probing structure discovery + Level-2 parameter fit (no LLM)"
        )
        from visualpatchworld import structure_induction as si

        result = si.induce_from_structure(
            args.env,
            train_trajs,
            geometry_json=args.geometry_json,
            rollout_k=args.rollout_k,
            steps=args.rollout_fit_steps,
            restarts=max(1, int(getattr(args, "fit_restarts", 4) or 4)),
        )
        _log(
            f"Discovered structure summary={result.get('structure', {}).get('summary')} "
            f"template={result['template']}"
        )
        code = result["code"]
    elif args.induction_strategy == "general_scaffold_fit":
        _log(f"Starting general scaffold-fit induction for env={args.env} (no LLM)")
        code = generate_general_scaffold_model_code(train_trajs, args.env)
    elif args.induction_strategy == "reacher_joint_fit":
        if args.env != "reacher":
            raise SystemExit("reacher_joint_fit induction is only available for env=reacher")
        _log("Starting reacher joint-dynamics induction (no LLM)")
        code = generate_reacher_joint_model_code(train_trajs)
    elif args.induction_strategy in {
        "pusht_kinematic",
        "pusht_scaffold_fit",
        "pusht_scaffold_fit_contact",
        "pusht_scaffold_fit_contact_weighted",
        "pusht_scaffold_fit_contact_gated",
        "pusht_scaffold_fit_contact_gated_weighted",
        "pusht_scaffold_fit_contact_gated_mlp",
        "pusht_scaffold_fit_contact_gated_mlp_weighted",
        "pusht_modular_gated_mlp_offline",
        "pusht_modular_gated_mlp_contact_refit",
        "pusht_modular_gated_mlp_agent_refit",
        "pusht_modular_gated_mlp_oracle_offline",
        "pusht_modular_gated_mlp_oracle_contact_refit",
        "pusht_compositional_gated_mlp_oracle_offline",
        "pusht_compositional_gated_mlp_oracle_contact_refit",
        "pusht_pushing_mechanics",
        "pusht_pushing_mechanics_collision",
        "pusht_pushing_mechanics_collision_rollout",
        "pusht_physics_sim",
        "pusht_pd_physics",
        "pusht_program_induction",
    }:
        if args.env != "pusht":
            raise SystemExit(f"{args.induction_strategy} induction is only available for env=pusht")
        if args.induction_strategy == "pusht_kinematic":
            _log("Starting local PushT kinematic induction (no LLM)")
            code = generate_pusht_kinematic_model_code(train_trajs)
        elif args.induction_strategy == "pusht_scaffold_fit_contact_weighted":
            if not args.failure_transitions_jsonl:
                raise SystemExit(
                    "pusht_scaffold_fit_contact_weighted requires --failure_transitions_jsonl"
                )
            _log("Starting failure-weighted contact scaffold-fit induction (no LLM)")
            failure_paths = [p.strip() for p in str(args.failure_transitions_jsonl).split(",") if p.strip()]
            extra = load_weighted_transitions_jsonl_paths(failure_paths)
            _log(f"Loaded {len(extra)} weighted extra transitions from {len(failure_paths)} file(s)")
            code = generate_pusht_scaffold_fit_contact_weighted_model_code(train_trajs, extra)
        elif args.induction_strategy in {
            "pusht_scaffold_fit_contact_gated_weighted",
            "pusht_scaffold_fit_contact_gated_mlp_weighted",
        }:
            if not args.failure_transitions_jsonl:
                raise SystemExit(f"{args.induction_strategy} requires --failure_transitions_jsonl")
            _log(f"Starting {args.induction_strategy} induction (no LLM)")
            failure_paths = [p.strip() for p in str(args.failure_transitions_jsonl).split(",") if p.strip()]
            extra = load_weighted_transitions_jsonl_paths(failure_paths)
            _log(f"Loaded {len(extra)} weighted extra transitions from {len(failure_paths)} file(s)")
            if args.induction_strategy == "pusht_scaffold_fit_contact_gated_weighted":
                code = generate_pusht_scaffold_fit_contact_gated_weighted_model_code(train_trajs, extra)
            else:
                code = generate_pusht_scaffold_fit_contact_gated_mlp_weighted_model_code(train_trajs, extra)
        elif args.induction_strategy == "pusht_scaffold_fit_contact_gated_mlp":
            _log("Starting contact-gated scaffold-fit with MLP residual induction (no LLM)")
            code = generate_pusht_scaffold_fit_contact_gated_mlp_model_code(train_trajs)
        elif args.induction_strategy == "pusht_modular_gated_mlp_offline":
            _log("Starting modular offline gated+MLP induction (trajectories only, no LLM)")
            code = generate_modular_offline_gated_mlp_model_code(train_trajs, seed=args.seed)
        elif args.induction_strategy == "pusht_modular_gated_mlp_oracle_offline":
            _log("Starting oracle modular offline gated+MLP with multistep contact loss (no LLM)")
            mlp_rollout = args.mlp_rollout_steps if args.mlp_rollout_steps > 0 else 2
            mlp_hidden = args.mlp_hidden_dim if args.mlp_hidden_dim > 0 else 64
            code = generate_modular_offline_gated_mlp_model_code(
                train_trajs,
                seed=args.seed,
                mlp_rollout_steps=mlp_rollout,
                mlp_hidden_dim=mlp_hidden,
                mlp_steps=3000,
            )
        elif args.induction_strategy == "pusht_pushing_mechanics":
            _log("Starting physics-structured pushing-mechanics induction (quasi-static law + momentum, no LLM)")
            code = generate_pushing_mechanics_model_code(train_trajs)
        elif args.induction_strategy == "pusht_pushing_mechanics_collision":
            _log("Starting collision-geometry contact induction (shared collision primitive + response law, no LLM)")
            code = generate_collision_model_code(train_trajs, geometry_json=args.geometry_json)
        elif args.induction_strategy == "pusht_pushing_mechanics_collision_rollout":
            _log("Starting collision-geometry contact induction with multi-step rollout-loss fit + soft gate (no LLM)")
            code = generate_collision_rollout_model_code(
                train_trajs,
                geometry_json=args.geometry_json,
                rollout_k=args.rollout_k,
                steps=args.rollout_fit_steps,
                use_l34=args.use_l34,
                gate_width=args.rollout_gate_width,
                reg_lambda=args.rollout_reg_lambda,
            )
        elif args.induction_strategy == "pusht_physics_sim":
            _log("Starting physics-simulation induction (rigid body + penalty contact, identified params, no LLM)")
            code = generate_physics_sim_model_code(
                train_trajs,
                geometry_json=args.geometry_json,
                rollout_k=args.rollout_k,
                steps=args.rollout_fit_steps,
            )
        elif args.induction_strategy == "pusht_pd_physics":
            _log("Starting faithful PD-agent + quasi-static-block physics induction (no LLM)")
            code = generate_pd_physics_model_code(
                train_trajs,
                geometry_json=args.geometry_json,
                rollout_k=args.rollout_k,
                steps=args.rollout_fit_steps,
            )
        elif args.induction_strategy == "pusht_program_induction":
            _log("Starting two-level program induction: Level-1 active-probing structure discovery + Level-2 parameter fit (no LLM)")
            from visualpatchworld import collision as _collision

            geometry = _collision.load_perceived_geometry(args.geometry_json) if args.geometry_json else None
            sim = program_induction.BlackBoxSim()
            try:
                report = program_induction.identify_structure(sim, geometry=geometry)
            finally:
                sim.close()
            _log(f"Discovered dynamics structure: {report.sketch} (all_confident={report.all_confident})")
            result = program_induction.induce_program(
                train_trajs,
                report=report,
                geometry_json=args.geometry_json,
                rollout_k=args.rollout_k,
                steps=args.rollout_fit_steps,
            )
            _log(f"Level-2 fit complete via discovered template '{result['template']}'")
            code = result["code"]
        elif args.induction_strategy == "pusht_compositional_gated_mlp_oracle_offline":
            _log("Starting compositional oracle offline (AgentModule + ContactModule + Gate)")
            mlp_rollout = args.mlp_rollout_steps if args.mlp_rollout_steps > 0 else 2
            mlp_hidden = args.mlp_hidden_dim if args.mlp_hidden_dim > 0 else 64
            code = generate_compositional_offline_model_code(
                train_trajs,
                seed=args.seed,
                mlp_rollout_steps=mlp_rollout,
                mlp_hidden_dim=mlp_hidden,
                mlp_steps=3000,
            )
        elif args.induction_strategy in {
            "pusht_modular_gated_mlp_contact_refit",
            "pusht_modular_gated_mlp_oracle_contact_refit",
            "pusht_modular_gated_mlp_agent_refit",
            "pusht_compositional_gated_mlp_oracle_contact_refit",
        }:
            if not args.base_results_dir:
                raise SystemExit(f"{args.induction_strategy} requires --base_results_dir")
            _log(f"Starting {args.induction_strategy} (no LLM)")
            compositional = args.induction_strategy.startswith("pusht_compositional_")
            if compositional:
                base_params = load_compositional_params_from_results_dir(args.base_results_dir)
            else:
                base_params = load_gated_mlp_params_from_results_dir(args.base_results_dir)
            extra = []
            ce_paths = []
            if args.counterexample_store:
                ce_paths = [p.strip() for p in str(args.counterexample_store).split(",") if p.strip()]
            elif args.failure_transitions_jsonl:
                ce_paths = [p.strip() for p in str(args.failure_transitions_jsonl).split(",") if p.strip()]
            if ce_paths:
                base_mass = float(len(iter_weighted_transitions(train_trajs, default_weight=1.0)))
                extra, store_info = build_capped_extra_transitions(
                    base_mass=base_mass,
                    counterexample_paths=ce_paths,
                    modules=parse_module_filter(args.counterexample_modules),
                    exclude_modules=parse_module_filter(args.exclude_counterexample_modules),
                    max_extra_fraction=args.max_extra_fraction,
                )
                _log(
                    f"Loaded {len(extra)} capped counterexamples from {len(ce_paths)} file(s); "
                    f"store={store_info.get('capped', {})}"
                )
            if args.induction_strategy == "pusht_modular_gated_mlp_contact_refit":
                code = generate_modular_contact_refit_model_code(
                    train_trajs,
                    base_params=base_params,
                    extra_transitions=extra,
                    seed=args.seed,
                )
            elif args.induction_strategy == "pusht_compositional_gated_mlp_oracle_contact_refit":
                mlp_rollout = args.mlp_rollout_steps if args.mlp_rollout_steps > 0 else 2
                mlp_hidden = args.mlp_hidden_dim if args.mlp_hidden_dim > 0 else 64
                code = generate_compositional_contact_refit_model_code(
                    train_trajs,
                    base_params=base_params,
                    extra_transitions=extra,
                    seed=args.seed,
                    mlp_rollout_steps=mlp_rollout,
                    mlp_hidden_dim=mlp_hidden,
                    mlp_steps=3000,
                )
            elif args.induction_strategy == "pusht_modular_gated_mlp_oracle_contact_refit":
                mlp_rollout = args.mlp_rollout_steps if args.mlp_rollout_steps > 0 else 2
                mlp_hidden = args.mlp_hidden_dim if args.mlp_hidden_dim > 0 else 64
                code = generate_modular_contact_refit_model_code(
                    train_trajs,
                    base_params=base_params,
                    extra_transitions=extra,
                    seed=args.seed,
                    mlp_rollout_steps=mlp_rollout,
                    mlp_hidden_dim=mlp_hidden,
                    mlp_steps=3000,
                )
            else:
                code = generate_modular_agent_refit_model_code(
                    train_trajs,
                    base_params=base_params,
                    extra_transitions=extra,
                )
        elif args.induction_strategy == "pusht_scaffold_fit_contact_gated":
            _log("Starting contact-gated scaffold-fit induction (no LLM)")
            code = generate_pusht_scaffold_fit_contact_gated_model_code(train_trajs)
        elif args.induction_strategy == "pusht_scaffold_fit_contact":
            _log("Starting contact-conditioned scaffolded fitted PushT induction (no LLM)")
            code = generate_pusht_scaffold_fit_contact_model_code(train_trajs)
        else:
            _log("Starting scaffolded fitted PushT induction (no LLM)")
            code = generate_pusht_scaffold_fit_model_code(train_trajs)
        _log("Finished local induction; sanitizing and writing generated model")
    else:
        _log(
            f"Initializing LLM model={args.model} strategy={args.induction_strategy} "
            f"max_refine_rounds={args.max_refine_rounds}"
        )
        llm = LLMClient(model=args.model)
        inducer = Inducer(
            llm=llm,
            max_refine_rounds=args.max_refine_rounds,
            contrastive_max_per_pattern=args.contrastive_max_per_pattern,
            contrastive_max_transitions=args.contrastive_max_transitions,
            induction_strategy=args.induction_strategy,
            patch_candidates_per_round=args.patch_candidates_per_round,
            patch_beam_size=args.patch_beam_size,
            patch_no_improve_patience=args.patch_no_improve_patience,
            initial_load_error_retries=args.initial_load_error_retries,
            verbose=args.verbose,
            show_progress=args.verbose,
            induction_track="simple_state",
        )

        _log(f"Starting PatchWorld induction for env={args.env}")
        code = inducer.induce(args.env, train_trajs, validation_trajectories=eval_trajs)
        _log("Finished induction; sanitizing and writing generated model")
    code = sanitize_pusht_model_code(code)
    model_path.write_text(code, encoding="utf-8")

    # Schema-adopted artifacts: accepted model + curated residual notes (promotion log).
    try:
        from visualpatchworld.schema_harness import write_vpw_schema_artifacts

        schema_artifacts = write_vpw_schema_artifacts(
            output_dir=out_dir,
            env_name=args.env,
            model_code=code,
            train_trajectories=train_trajs,
            notes=[
                f"induced_via={args.induction_strategy}",
                f"model_path={model_path}",
            ],
        )
        _log(f"Wrote Schema-adopted VPW artifacts under {out_dir}")
    except Exception as exc:  # pragma: no cover - non-fatal provenance helper
        schema_artifacts = {"error": str(exc)}
        _log(f"Schema-adopted artifact write skipped: {exc}")

    _log(f"Loading generated model from {model_path}")
    model, load_error = load_model_from_code(code)
    if load_error or model is None:
        raise SystemExit(f"Generated model failed to load: {load_error}")

    _log("Evaluating generated model on train/eval trajectories")
    report: Dict[str, object] = {
        "timestamp": datetime.now().isoformat(),
        "env": args.env,
        "model": args.model,
        "train_glob": args.train_glob,
        "eval_glob": args.eval_glob,
        "train_stats_pool": get_stats(train_all),
        "eval_stats_pool": get_stats(eval_all),
        "train_size_used": len(train_trajs),
        "eval_size_used": len(eval_trajs),
        "induction": {
            "strategy": args.induction_strategy,
            "max_refine_rounds": args.max_refine_rounds,
            "stats": dict(inducer.last_induce_stats or {}) if inducer is not None else {
                "total_trajectories": len(train_trajs),
                "validation_trajectories": len(eval_trajs),
                "local_inducer": args.induction_strategy,
                "converged": None,
            },
            "llm_usage": llm.get_usage() if llm is not None else {},
            "adapter": "visualpatchworld.patchworld_adapter.apply_pusht_patchworld_adapter",
        },
        "symbolic_model_path": str(model_path),
        "schema_adoption": schema_artifacts,
        "simple_state": _evaluate_model(model, train_trajs, eval_trajs, args.env),
    }

    if args.use_residual_memory:
        _log("Evaluating residual-memory wrapper")
        residual = EpisodicResidualWorldModel(
            model,
            train_trajs,
            env_name=args.env,
            max_entries=args.residual_memory_max_entries,
            min_count=args.residual_memory_min_count,
            min_confidence=args.residual_memory_min_confidence,
        )
        report["simple_state_residual"] = _evaluate_model(residual, train_trajs, eval_trajs, args.env)
        report["residual_memory"] = {"stats": residual.residual_stats}

    if args.run_rollout:
        _log(f"Running autoregressive rollout metrics horizon={args.rollout_horizon}")
        rollout_steps = [int(s.strip()) for s in args.rollout_steps.split(",") if s.strip()]
        report["rq2_rollout"] = {
            "simple_state": compute_autoregressive_rollout_token_f1(
                model,
                eval_trajs,
                max_horizon=args.rollout_horizon,
                report_steps=rollout_steps,
            )
        }
        if args.use_residual_memory:
            residual_model = EpisodicResidualWorldModel(
                model,
                train_trajs,
                env_name=args.env,
                max_entries=args.residual_memory_max_entries,
                min_count=args.residual_memory_min_count,
                min_confidence=args.residual_memory_min_confidence,
            )
            report["rq2_rollout"]["simple_state_residual"] = compute_autoregressive_rollout_token_f1(
                residual_model,
                eval_trajs,
                max_horizon=args.rollout_horizon,
                report_steps=rollout_steps,
            )

    report_path = out_dir / f"benchmark_{args.env}_{ts}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    _log(f"Generated model: {model_path}")
    _log(f"Benchmark report: {report_path}")


if __name__ == "__main__":
    main()
