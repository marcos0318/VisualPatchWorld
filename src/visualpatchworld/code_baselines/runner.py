"""Unified entry point for LeWM programmatic code baselines."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from patchworld.inducer_llm_client import LLMClient
from patchworld.worldmodel_data import Trajectory, get_stats

from visualpatchworld.patchworld_adapter import apply_pusht_patchworld_adapter, sanitize_pusht_model_code

from .common import (
    is_loadable_worldmodel,
    load_trajectory_glob,
    make_inducer,
    sanitize_generated_code,
    subsample_trajectories,
    validate_code,
)
from .cwm_game import cwm_game_induce
from .gif_mcts import gif_mcts_induce
from .pomdp_coder import pomdp_coder_induce
from .poeworld import poeworld_induce
from .worldcoder import worldcoder_induce

apply_pusht_patchworld_adapter()

DEFAULT_MODEL = "Qwen/Qwen3-Coder-480B-A35B-Instruct-Turbo"
METHODS = ("worldcoder", "poeworld", "gif_mcts", "tracepatch", "pomdp_coder", "cwm_game")

_CODE_BASELINE_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_LLM_CACHE = _CODE_BASELINE_ROOT / "artifacts" / "visualpatchworld" / "llm_cache.sqlite3"


def apply_code_baseline_llm_env(
    *,
    llm_request_timeout_s: int | None = None,
    llm_max_retries: int | None = None,
) -> Dict[str, str]:
    """Match PatchWorld induction transport defaults (480B codegen can exceed 120s)."""
    defaults = {
        "PATCHWORLD_LLM_REQUEST_TIMEOUT_S": "600",
        "PATCHWORLD_LLM_MAX_RETRIES": "12",
        "PATCHWORLD_LLM_RETRY_BASE_SLEEP_S": "3",
        "PATCHWORLD_LLM_RETRY_MAX_SLEEP_S": "120",
        "PATCHWORLD_LLM_MAX_RETRY_WAIT_S": "1800",
        "PATCHWORLD_LLM_CACHE": "1",
        "PATCHWORLD_LLM_CACHE_PATH": str(_DEFAULT_LLM_CACHE),
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)
    if llm_request_timeout_s is not None:
        os.environ["PATCHWORLD_LLM_REQUEST_TIMEOUT_S"] = str(llm_request_timeout_s)
    if llm_max_retries is not None:
        os.environ["PATCHWORLD_LLM_MAX_RETRIES"] = str(llm_max_retries)
    return {key: os.environ[key] for key in defaults}


def induce_code_baseline(
    *,
    method: str,
    env_name: str,
    train_glob: str,
    eval_glob: str,
    output_dir: str | Path,
    model: str = DEFAULT_MODEL,
    max_train: int = 24,
    max_eval: int = 12,
    seed: int = 42,
    max_refine_rounds: int = 3,
    worldcoder_programs: int = 2,
    poeworld_batches: int = 3,
    poeworld_experts_per_batch: int = 4,
    gif_mcts_iterations: int = 50,
    pomdp_coder_candidates: int = 3,
    cwm_retries: int = 50,
    show_progress: bool = True,
    llm_request_timeout_s: int | None = None,
    llm_max_retries: int | None = None,
) -> Dict[str, Any]:
    """Induce a programmatic baseline and write model + benchmark report."""
    llm_env = apply_code_baseline_llm_env(
        llm_request_timeout_s=llm_request_timeout_s,
        llm_max_retries=llm_max_retries,
    )
    method = method.strip().lower()
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}; choose from {METHODS}")

    train_all = load_trajectory_glob(train_glob)
    eval_all = load_trajectory_glob(eval_glob)
    train_trajs = subsample_trajectories(train_all, max_train, seed)
    eval_trajs = subsample_trajectories(eval_all, max_eval, seed + 1)

    out_dir = Path(output_dir)
    model_dir = out_dir / "generated_models"
    model_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = model_dir / f"{env_name}_{method}_{ts}.py"

    llm = LLMClient(model=model)
    inducer = make_inducer(llm, max_refine_rounds=max_refine_rounds, show_progress=show_progress)

    if method == "tracepatch":
        code = inducer.tracepatch_induce(env_name, train_trajs, validation_trajectories=eval_trajs)
    elif method == "worldcoder":
        code = worldcoder_induce(
            inducer,
            env_name,
            train_trajs,
            eval_trajs,
            initial_programs=worldcoder_programs,
            max_rounds=max_refine_rounds,
            seed=seed,
        )
    elif method == "poeworld":
        code = poeworld_induce(
            inducer,
            env_name,
            train_trajs,
            eval_trajs,
            experts_per_batch=poeworld_experts_per_batch,
            num_batches=poeworld_batches,
            seed=seed,
        )
    elif method == "gif_mcts":
        code = gif_mcts_induce(
            inducer,
            env_name,
            train_trajs,
            eval_trajs,
            iterations=gif_mcts_iterations,
            seed=seed,
        )
    elif method == "pomdp_coder":
        code = pomdp_coder_induce(
            inducer,
            env_name,
            train_trajs,
            eval_trajs,
            initial_candidates=pomdp_coder_candidates,
            max_rounds=max_refine_rounds,
            seed=seed,
        )
    else:
        code = cwm_game_induce(
            inducer,
            env_name,
            train_trajs,
            eval_trajs,
            retries=cwm_retries,
            seed=seed,
        )

    code = sanitize_generated_code(code)
    ok, load_err = is_loadable_worldmodel(code)
    if not ok:
        raise RuntimeError(f"{method} induction produced unloadable model: {load_err}")
    model_path.write_text(code, encoding="utf-8")
    val_errors, val_score = validate_code(code, eval_trajs, label=f"{method}_final")

    report: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "env": env_name,
        "method": method,
        "model": model,
        "llm_transport": llm_env,
        "train_glob": train_glob,
        "eval_glob": eval_glob,
        "train_stats_pool": get_stats(train_all),
        "eval_stats_pool": get_stats(eval_all),
        "train_size_used": len(train_trajs),
        "eval_size_used": len(eval_trajs),
        "induction": {
            "method": method,
            "stats": dict(inducer.last_induce_stats or {}),
            "llm_usage": llm.get_usage(),
        },
        "validation": {
            "score": val_score,
            "error_count": len(val_errors),
        },
        "symbolic_model_path": str(model_path),
    }
    report_path = out_dir / f"benchmark_{env_name}_{method}_{ts}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report
