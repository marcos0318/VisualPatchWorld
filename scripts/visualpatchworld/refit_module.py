#!/usr/bin/env python
"""CLI for modular gated+MLP dynamics refit (offline / contact / agent)."""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import List

from visualpatchworld.counterexample_store import build_capped_extra_transitions, parse_module_filter
from visualpatchworld.modular_dynamics import (
    base_transition_mass,
    evaluate_module_acceptance,
    generate_modular_agent_refit_model_code,
    generate_modular_contact_refit_model_code,
    generate_modular_offline_gated_mlp_model_code,
    load_gated_mlp_params_from_results_dir,
    render_gated_mlp_model_code,
    refit_agent_module_params,
    refit_contact_module_params,
)
from visualpatchworld.patchworld_adapter import apply_pusht_patchworld_adapter, sanitize_pusht_model_code

apply_pusht_patchworld_adapter()

from patchworld.worldmodel_data import Trajectory, get_stats, load_trajectories
from patchworld.worldmodel_evaluator import compute_experience_fit, compute_observation_fit
from patchworld.worldmodel_validator import load_model_from_code


ROOT = Path(__file__).resolve().parents[2]


def _log(message: str) -> None:
    print(f"[visualpatchworld][{datetime.now().isoformat(timespec='seconds')}][refit] {message}", flush=True)


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--module",
        required=True,
        choices=["offline", "contact", "agent"],
        help="Which dynamics module to fit or refit.",
    )
    parser.add_argument("--train_glob", required=True)
    parser.add_argument("--eval_glob", default=None)
    parser.add_argument("--output_dir", default=str(ROOT / "artifacts" / "visualpatchworld" / "modular_results"))
    parser.add_argument("--base_results_dir", default=None, help="Required for contact/agent refit.")
    parser.add_argument(
        "--counterexample_store",
        default=None,
        help="Comma-separated JSONL paths with module-tagged counterexamples.",
    )
    parser.add_argument("--max_train", type=int, default=0)
    parser.add_argument("--max_eval", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_extra_fraction", type=float, default=0.3)
    parser.add_argument(
        "--counterexample_modules",
        default="contact,contact_angle",
        help="Comma-separated module tags to include from the store.",
    )
    parser.add_argument(
        "--exclude_counterexample_modules",
        default="contact_gate",
        help="Comma-separated module tags to exclude from the store.",
    )
    parser.add_argument("--write_acceptance_report", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    _log(f"Loading train trajectories: {args.train_glob}")
    train_all = _load_trajs(args.train_glob)
    eval_glob = args.eval_glob or args.train_glob
    _log(f"Loading eval trajectories: {eval_glob}")
    eval_all = _load_trajs(eval_glob)
    train_trajs = _subsample(train_all, args.max_train, args.seed)
    eval_trajs = _subsample(eval_all, args.max_eval, args.seed + 1)
    _log(
        f"Using train={len(train_trajs)} eval={len(eval_trajs)} "
        f"(pools {len(train_all)}/{len(eval_all)})"
    )

    out_dir = Path(args.output_dir)
    model_dir = out_dir / "generated_models"
    model_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = model_dir / f"pusht_modular_{args.module}_{ts}.py"

    refit_info: dict = {"module": args.module}
    if args.module == "offline":
        _log("Fitting offline gated+MLP from trajectories only")
        code = generate_modular_offline_gated_mlp_model_code(train_trajs, seed=args.seed)
    else:
        if not args.base_results_dir:
            raise SystemExit(f"--base_results_dir is required for module={args.module}")
        base_params = load_gated_mlp_params_from_results_dir(args.base_results_dir)
        _log(f"Loaded base params from {args.base_results_dir}")

        extra = []
        if args.counterexample_store:
            ce_paths = [p.strip() for p in str(args.counterexample_store).split(",") if p.strip()]
            base_mass = base_transition_mass(train_trajs)
            extra, refit_info["counterexample_store"] = build_capped_extra_transitions(
                base_mass=base_mass,
                counterexample_paths=ce_paths,
                modules=parse_module_filter(args.counterexample_modules),
                exclude_modules=parse_module_filter(args.exclude_counterexample_modules),
                max_extra_fraction=args.max_extra_fraction,
            )
            _log(
                f"Loaded {len(extra)} capped extra transitions "
                f"(max_extra_fraction={args.max_extra_fraction})"
            )

        if args.module == "contact":
            _log("Refitting contact block rows + MLP only")
            params = refit_contact_module_params(base_params, train_trajs, extra, seed=args.seed)
            code = render_gated_mlp_model_code(params)
            refit_info["fit_stats"] = {k: params[k] for k in ("n_base_rows", "n_extra_rows", "n_contact_rows")}
        else:
            _log("Refitting agent rows of free/contact heads only")
            params = refit_agent_module_params(base_params, train_trajs, extra)
            code = render_gated_mlp_model_code(params)
            refit_info["fit_stats"] = {
                k: params[k] for k in ("n_base_rows", "n_extra_rows", "n_free_rows", "n_contact_rows")
            }

    code = sanitize_pusht_model_code(code)
    model_path.write_text(code, encoding="utf-8")
    _log(f"Wrote model to {model_path}")

    model, load_error = load_model_from_code(code)
    if load_error or model is None:
        raise SystemExit(f"Generated model failed to load: {load_error}")

    report = {
        "timestamp": datetime.now().isoformat(),
        "module": args.module,
        "train_glob": args.train_glob,
        "eval_glob": eval_glob,
        "train_stats_pool": get_stats(train_all),
        "eval_stats_pool": get_stats(eval_all),
        "train_size_used": len(train_trajs),
        "eval_size_used": len(eval_trajs),
        "base_results_dir": args.base_results_dir,
        "counterexample_store": args.counterexample_store,
        "refit_info": refit_info,
        "symbolic_model_path": str(model_path),
        "simple_state": {
            "train": {
                "experience_fit": compute_experience_fit(model, train_trajs),
                "observation_fit": compute_observation_fit(model, train_trajs, env_name="pusht"),
            },
            "eval": {
                "experience_fit": compute_experience_fit(model, eval_trajs),
                "observation_fit": compute_observation_fit(model, eval_trajs, env_name="pusht"),
            },
        },
    }

    if args.write_acceptance_report:
        report["module_acceptance"] = evaluate_module_acceptance(
            model,
            eval_trajs,
            contact_threshold=float(getattr(model, "CONTACT_THRESHOLD", 70.0)),
        )

    report_path = out_dir / f"refit_{args.module}_{ts}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _log(f"Refit report: {report_path}")


if __name__ == "__main__":
    main()
