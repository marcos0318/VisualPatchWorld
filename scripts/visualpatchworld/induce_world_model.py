#!/usr/bin/env python
"""Induce a PushT world model from VisualPatchWorld trajectories.

For the fair VLM path, pass `--sample_dir` with image/action samples. This
script then performs the visual extraction and PatchWorld trajectory build
inside the induction adapter before launching PatchWorld.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _log(message: str) -> None:
    print(f"[visualpatchworld][{datetime.now().isoformat(timespec='seconds')}][induce] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["oracle", "vlm", "tool"], required=True)
    parser.add_argument("--data_dir", default=str(ROOT / "artifacts" / "visualpatchworld" / "data"))
    parser.add_argument("--output_dir", default=str(ROOT / "artifacts" / "visualpatchworld" / "results"))
    parser.add_argument(
        "--sample_dir",
        default=None,
        help="Image/action sample directory. For --source vlm, triggers image extraction and trajectory build.",
    )
    parser.add_argument("--vlm_cache_dir", default=str(ROOT / "artifacts" / "visualpatchworld" / "vlm_cache"))
    parser.add_argument(
        "--tool_cache_dir",
        default=str(ROOT / "artifacts" / "visualpatchworld" / "tool_cache" / "pusht_heuristic"),
    )
    parser.add_argument("--tool_provider", default="pusht_heuristic", choices=["pusht_heuristic"])
    parser.add_argument("--vlm_model", default=os.environ.get("VLM_MODEL", "Qwen/Qwen3.5-397B-A17B"))
    parser.add_argument("--vlm_base_url", default=os.environ.get("VLM_BASE_URL", "https://api.deepinfra.com/v1/openai"))
    parser.add_argument("--vlm_api_key_env", default=os.environ.get("VLM_API_KEY_ENV", "DEEPINFRA_API_KEY"))
    parser.add_argument("--vlm_overwrite_cache", action="store_true")
    parser.add_argument(
        "--vlm_max_frames",
        type=int,
        default=0,
        help="Stop VLM extraction after this many new cache writes (<=0 means no limit).",
    )
    parser.add_argument(
        "--vlm_splits",
        nargs="+",
        default=None,
        choices=["train", "val", "test"],
        help="Splits to extract when building VLM trajectories.",
    )
    parser.add_argument("--max_train", type=int, default=8)
    parser.add_argument("--max_eval", type=int, default=4)
    parser.add_argument("--model", default=None)
    parser.add_argument("--max_refine_rounds", type=int, default=4)
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
        ],
    )
    parser.add_argument("--failure_transitions_jsonl", default=None)
    parser.add_argument("--base_results_dir", default=None)
    parser.add_argument("--counterexample_store", default=None)
    parser.add_argument("--max_extra_fraction", type=float, default=0.3)
    parser.add_argument("--counterexample_modules", default="contact,contact_angle")
    parser.add_argument("--exclude_counterexample_modules", default="contact_gate")
    parser.add_argument("--mlp_rollout_steps", type=int, default=0)
    parser.add_argument("--mlp_hidden_dim", type=int, default=0)
    parser.add_argument("--patch_candidates_per_round", type=int, default=4)
    parser.add_argument("--patch_beam_size", type=int, default=2)
    parser.add_argument("--patch_no_improve_patience", type=int, default=2)
    parser.add_argument("--initial_load_error_retries", type=int, default=5)
    parser.add_argument("--llm_request_timeout_s", type=int, default=None)
    parser.add_argument("--llm_max_retries", type=int, default=None)
    parser.add_argument("--subprocess_timeout_s", type=int, default=0)
    parser.add_argument("--run_rollout", action="store_true")
    parser.add_argument("--use_residual_memory", action="store_true")
    parser.add_argument(
        "--prepare_only",
        action="store_true",
        help="Run integrated visual extraction/build, then stop before PatchWorld induction.",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("extra", nargs=argparse.REMAINDER)
    return parser.parse_args()


def _run(cmd: list[str], *, dry_run: bool) -> None:
    _log("Command: " + " ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, cwd=str(ROOT), check=True)


def _prepare_vlm_trajectories(args: argparse.Namespace) -> None:
    if args.source != "vlm" or not args.sample_dir:
        return

    extract_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "visualpatchworld" / "extract_vlm_scene_graphs.py"),
        "--sample_dir",
        args.sample_dir,
        "--cache_dir",
        args.vlm_cache_dir,
        "--model",
        args.vlm_model,
        "--base_url",
        args.vlm_base_url,
        "--api_key_env",
        args.vlm_api_key_env,
    ]
    if args.vlm_overwrite_cache:
        extract_cmd.append("--overwrite")
    if args.vlm_max_frames > 0:
        extract_cmd.extend(["--max_frames", str(args.vlm_max_frames)])
    if args.vlm_splits:
        extract_cmd.extend(["--splits", *args.vlm_splits])
    if args.dry_run:
        extract_cmd.append("--dry_run")
    _log("Extracting VLM scene graphs from image/action samples")
    _run(extract_cmd, dry_run=False)

    build_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "visualpatchworld" / "build_vlm_trajectories.py"),
        "--sample_dir",
        args.sample_dir,
        "--cache_dir",
        args.vlm_cache_dir,
        "--output_dir",
        args.data_dir,
        "--source_name",
        "vlm",
    ]
    _log("Building VLM PatchWorld trajectories from extracted scene graphs")
    _run(build_cmd, dry_run=args.dry_run)


def _prepare_tool_trajectories(args: argparse.Namespace) -> None:
    if args.source != "tool" or not args.sample_dir:
        return

    extract_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "visualpatchworld" / "extract_tool_scene_graphs.py"),
        "--sample_dir",
        args.sample_dir,
        "--cache_dir",
        args.tool_cache_dir,
        "--provider",
        args.tool_provider,
    ]
    if args.vlm_splits:
        extract_cmd.extend(["--splits", *args.vlm_splits])
    if args.vlm_overwrite_cache:
        extract_cmd.append("--overwrite")
    if args.vlm_max_frames > 0:
        extract_cmd.extend(["--max_frames", str(args.vlm_max_frames)])
    if args.dry_run:
        extract_cmd.append("--dry_run")
    _log("Extracting tool-assisted scene graphs from image/action samples")
    _run(extract_cmd, dry_run=False)

    build_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "visualpatchworld" / "build_vlm_trajectories.py"),
        "--sample_dir",
        args.sample_dir,
        "--cache_dir",
        args.tool_cache_dir,
        "--output_dir",
        args.data_dir,
        "--source_name",
        "tool",
    ]
    _log("Building tool PatchWorld trajectories from extracted scene graphs")
    _run(build_cmd, dry_run=args.dry_run)


def main() -> None:
    args = parse_args()
    _prepare_vlm_trajectories(args)
    _prepare_tool_trajectories(args)
    if args.dry_run:
        _log("Dry run requested; not launching PatchWorld induction")
        return
    if args.prepare_only:
        _log("Prepare-only requested; not launching PatchWorld induction")
        return

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "visualpatchworld" / "run_patchworld.py"),
        "--source",
        args.source,
        "--data_dir",
        args.data_dir,
        "--output_dir",
        args.output_dir,
        "--max_train",
        str(args.max_train),
        "--max_eval",
        str(args.max_eval),
        "--max_refine_rounds",
        str(args.max_refine_rounds),
        "--induction_strategy",
        args.induction_strategy,
        "--patch_candidates_per_round",
        str(args.patch_candidates_per_round),
        "--patch_beam_size",
        str(args.patch_beam_size),
        "--patch_no_improve_patience",
        str(args.patch_no_improve_patience),
        "--initial_load_error_retries",
        str(args.initial_load_error_retries),
    ]
    if args.llm_request_timeout_s is not None:
        cmd.extend(["--llm_request_timeout_s", str(args.llm_request_timeout_s)])
    if args.llm_max_retries is not None:
        cmd.extend(["--llm_max_retries", str(args.llm_max_retries)])
    if args.subprocess_timeout_s > 0:
        cmd.extend(["--subprocess_timeout_s", str(args.subprocess_timeout_s)])
    if args.failure_transitions_jsonl:
        cmd.extend(["--failure_transitions_jsonl", args.failure_transitions_jsonl])
    if args.base_results_dir:
        cmd.extend(["--base_results_dir", args.base_results_dir])
    if args.counterexample_store:
        cmd.extend(["--counterexample_store", args.counterexample_store])
    if args.max_extra_fraction != 0.3:
        cmd.extend(["--max_extra_fraction", str(args.max_extra_fraction)])
    if args.counterexample_modules != "contact,contact_angle":
        cmd.extend(["--counterexample_modules", args.counterexample_modules])
    if args.exclude_counterexample_modules != "contact_gate":
        cmd.extend(["--exclude_counterexample_modules", args.exclude_counterexample_modules])
    if args.mlp_rollout_steps > 0:
        cmd.extend(["--mlp_rollout_steps", str(args.mlp_rollout_steps)])
    if args.mlp_hidden_dim > 0:
        cmd.extend(["--mlp_hidden_dim", str(args.mlp_hidden_dim)])
    if args.model:
        cmd.extend(["--model", args.model])
    if args.run_rollout:
        cmd.append("--run_rollout")
    if args.use_residual_memory:
        cmd.append("--use_residual_memory")
    if args.verbose:
        cmd.append("--verbose")
    if args.dry_run:
        cmd.append("--dry_run")
    if args.extra:
        extras = args.extra[1:] if args.extra and args.extra[0] == "--" else args.extra
        cmd.extend(extras)

    _run(cmd, dry_run=False)


if __name__ == "__main__":
    main()
