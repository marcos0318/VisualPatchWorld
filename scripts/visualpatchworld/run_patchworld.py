#!/usr/bin/env python
"""Run PatchWorld induction/evaluation on VisualPatchWorld exports."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PATCHWORLD_ROOT = ROOT  # vendored package at repo root
DEFAULT_PATCHWORLD_MODEL = "Qwen/Qwen3-Coder-480B-A35B-Instruct-Turbo"
LEGACY_MULTIMODAL_MODEL = "Qwen/Qwen3.5-397B-A17B"


def _log(message: str) -> None:
    print(f"[visualpatchworld][{datetime.now().isoformat(timespec='seconds')}][runner] {message}", flush=True)


def _default_induction_model() -> str:
    explicit = os.environ.get("PATCHWORLD_INDUCTION_MODEL")
    if explicit:
        return explicit
    legacy = os.environ.get("PATCHWORLD_MODEL")
    if legacy and legacy != LEGACY_MULTIMODAL_MODEL:
        return legacy
    return DEFAULT_PATCHWORLD_MODEL


def _resolve_traj_jsonl(data_dir: Path, source: str, split: str) -> Path | None:
    candidates = [
        data_dir / f"pusht_lewm_{source}_traj_{split}.jsonl",
        data_dir / f"pusht_{source}_traj_{split}.jsonl",
    ]
    for candidate in candidates:
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    return None


def _resolve_eval_glob(data_dir: Path, source: str, split: str) -> str:
    preferred = _resolve_traj_jsonl(data_dir, source, split)
    if preferred is not None:
        return str(preferred)
    for fallback in ("val", "test", "train"):
        if fallback == split:
            continue
        candidate = _resolve_traj_jsonl(data_dir, source, fallback)
        if candidate is not None:
            _log(f"Eval split {split!r} is empty; using {fallback} instead: {candidate}")
            return str(candidate)
    raise SystemExit(
        f"No non-empty evaluation JSONL found for source={source!r} under {data_dir}. "
        "Export more episodes or pass a custom eval glob via extra args."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["oracle", "vlm", "tool"], required=True)
    parser.add_argument("--data_dir", default=str(ROOT / "artifacts" / "visualpatchworld" / "data"))
    parser.add_argument("--output_dir", default=str(ROOT / "artifacts" / "visualpatchworld" / "results"))
    parser.add_argument("--eval_split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--max_train", type=int, default=8)
    parser.add_argument("--max_eval", type=int, default=4)
    parser.add_argument("--model", default=_default_induction_model())
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
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("extra", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) / args.source
    train_path = _resolve_traj_jsonl(data_dir, args.source, "train")
    if train_path is None:
        raise SystemExit(
            f"No non-empty train JSONL found for source={args.source!r} under {data_dir}."
        )
    train_glob = str(train_path)
    eval_glob = _resolve_eval_glob(data_dir, args.source, args.eval_split)
    _log(f"Train trajectories: {train_glob}")
    _log(f"Eval trajectories: {eval_glob}")
    if not (PATCHWORLD_ROOT / "patchworld").exists():
        raise SystemExit(f"PatchWorld dependency not found: {PATCHWORLD_ROOT}")

    cmd = [
        sys.executable,
        "-m",
        "visualpatchworld.patchworld_benchmark",
        "--env",
        "pusht",
        "--train_glob",
        train_glob,
        "--eval_glob",
        eval_glob,
        "--output_dir",
        str(output_dir),
        "--max_train",
        str(args.max_train),
        "--max_eval",
        str(args.max_eval),
        "--model",
        args.model,
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
    if args.run_rollout:
        cmd.append("--run_rollout")
    if args.use_residual_memory:
        cmd.append("--use_residual_memory")
    if args.verbose:
        cmd.append("--verbose")
    if args.extra:
        extras = args.extra[1:] if args.extra and args.extra[0] == "--" else args.extra
        cmd.extend(extras)

    env = os.environ.copy()
    pythonpath_parts = [str(PATCHWORLD_ROOT), str(ROOT / "src")]
    if env.get("PYTHONPATH"):
        pythonpath_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    env.setdefault("PATCHWORLD_LLM_REQUEST_TIMEOUT_S", "600")
    env.setdefault("PATCHWORLD_LLM_MAX_RETRIES", "12")
    env.setdefault("PATCHWORLD_LLM_RETRY_BASE_SLEEP_S", "3")
    env.setdefault("PATCHWORLD_LLM_RETRY_MAX_SLEEP_S", "120")
    env.setdefault("PATCHWORLD_LLM_MAX_RETRY_WAIT_S", "1800")
    env.setdefault("PATCHWORLD_LLM_CACHE", "1")
    env.setdefault(
        "PATCHWORLD_LLM_CACHE_PATH",
        str(ROOT / "artifacts" / "visualpatchworld" / "llm_cache.sqlite3"),
    )
    if args.llm_request_timeout_s is not None:
        env["PATCHWORLD_LLM_REQUEST_TIMEOUT_S"] = str(args.llm_request_timeout_s)
    if args.llm_max_retries is not None:
        env["PATCHWORLD_LLM_MAX_RETRIES"] = str(args.llm_max_retries)

    _log(f"PatchWorld dependency: {PATCHWORLD_ROOT}")
    _log(f"Output directory: {output_dir}")
    _log(
        "LLM transport: "
        f"timeout={env['PATCHWORLD_LLM_REQUEST_TIMEOUT_S']}s "
        f"retries={env['PATCHWORLD_LLM_MAX_RETRIES']} "
        f"cache={env['PATCHWORLD_LLM_CACHE']} "
        f"cache_path={env['PATCHWORLD_LLM_CACHE_PATH']}"
    )
    _log("Command: " + " ".join(cmd))
    if args.dry_run:
        _log("Dry run requested; not launching induction")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    _log("Launching local VisualPatchWorld benchmark subprocess")
    subprocess.run(
        cmd,
        check=True,
        cwd=str(ROOT),
        env=env,
        timeout=args.subprocess_timeout_s if args.subprocess_timeout_s > 0 else None,
    )
    _log("PatchWorld subprocess finished successfully")


if __name__ == "__main__":
    main()
