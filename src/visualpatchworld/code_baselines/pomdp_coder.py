"""Curtis et al. POMDP-coder style induction for VPW oracle graphs."""

from __future__ import annotations

import random
from typing import Any, Dict, List, Sequence

from patchworld.worldmodel_data import Trajectory
from patchworld.worldmodel_inducer import Inducer

from .common import generate_loadable_code, is_loadable_worldmodel, sanitize_generated_code
from .metrics import TransitionEval, evaluate_transition_accuracy
from .prompts import (
    PATCHWORLD_API_REQUIREMENTS,
    POMDP_CODER_INITIAL_PROMPT,
    POMDP_CODER_LOAD_REPAIR_PROMPT,
    POMDP_CODER_REFINE_PROMPT,
    POMDP_CODER_RULES,
)


def _api_requirements(class_name: str) -> str:
    return PATCHWORLD_API_REQUIREMENTS.format(class_name=class_name)


def _candidate_score(eval_result: TransitionEval) -> tuple[int, float, int]:
    """Sort key: loadable first, then higher accuracy, then lower MSE."""
    return (
        0 if not eval_result.load_error else 1,
        -float(eval_result.accuracy),
        int(eval_result.mse * 1_000_000),
    )


def _best_candidate(candidates: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not candidates:
        raise RuntimeError("POMDP-coder did not produce any candidates")
    return sorted(candidates, key=lambda item: _candidate_score(item["eval"]))[0]


def pomdp_coder_induce(
    inducer: Inducer,
    env_name: str,
    train_trajectories: Sequence[Trajectory],
    validation_trajectories: Sequence[Trajectory],
    *,
    initial_candidates: int = 3,
    max_rounds: int = 3,
    seed: int = 42,
) -> str:
    """LLM-guided generate-test-refine port of Curtis et al. (CoRL 2025).

    The original method induces POMDP components as short probabilistic programs.
    This port keeps the same LLM-prior plus empirical-feedback loop, but constrains
    the emitted artifact to the local PatchWorld ``BaseWorldModel`` API used by
    the VPW CEM planner.
    """
    env = (env_name or "").strip().lower()
    class_name = f"{env.capitalize()}WorldModel"
    api_requirements = _api_requirements(class_name)
    rng = random.Random(seed)

    train_transitions = inducer._select_induction_transitions(list(train_trajectories))
    transition_text = inducer._format_transitions(train_transitions)

    candidates: List[Dict[str, Any]] = []
    round_records: List[Dict[str, Any]] = []
    llm_calls = 0

    def _generate_initial(candidate_idx: int) -> str:
        nonlocal llm_calls
        prompt = POMDP_CODER_INITIAL_PROMPT.format(
            env_name=env_name,
            candidate_idx=candidate_idx,
            rules=POMDP_CODER_RULES,
            api_requirements=api_requirements,
            transition_text=transition_text,
            class_name=class_name,
        )
        llm_calls += 1
        return sanitize_generated_code(inducer._extract_code(inducer.llm.generate(prompt)))

    for candidate_idx in range(1, max(1, initial_candidates) + 1):
        try:
            code = generate_loadable_code(lambda idx=candidate_idx: _generate_initial(idx), max_attempts=2)
        except RuntimeError:
            code = _generate_initial(candidate_idx)
        eval_result = evaluate_transition_accuracy(code, validation_trajectories)
        candidates.append(
            {
                "round": 0,
                "candidate_idx": candidate_idx,
                "code": code,
                "eval": eval_result,
                "action": "generate",
            }
        )

    for round_idx in range(1, max(0, max_rounds) + 1):
        parent = _best_candidate(candidates)
        parent_eval: TransitionEval = parent["eval"]
        if not parent_eval.load_error and parent_eval.accuracy >= 1.0:
            break

        if parent_eval.load_error:
            prompt = POMDP_CODER_LOAD_REPAIR_PROMPT.format(
                env_name=env_name,
                api_requirements=api_requirements,
                code=parent["code"],
                class_name=class_name,
            )
            action = "load_repair"
        else:
            feedback_transitions = inducer._select_induction_transitions(list(validation_trajectories))
            feedback_text = inducer._format_transitions(feedback_transitions)
            prompt = POMDP_CODER_REFINE_PROMPT.format(
                env_name=env_name,
                code=parent["code"],
                errors=inducer._format_errors(parent_eval.errors[:12]),
                transition_text=feedback_text,
                class_name=class_name,
            )
            action = "refine"

        llm_calls += 1
        try:
            code = sanitize_generated_code(inducer._extract_code(inducer.llm.generate(prompt)))
        except Exception:
            # Keep a fresh candidate path alive if a repair call fails transiently.
            code = _generate_initial(rng.randint(1000, 9999))
            action = "regenerate"

        eval_result = evaluate_transition_accuracy(code, validation_trajectories)
        record = {
            "round": round_idx,
            "action": action,
            "parent_accuracy": parent_eval.accuracy,
            "child_accuracy": eval_result.accuracy,
            "child_load_error": eval_result.load_error,
            "child_mse": eval_result.mse,
        }
        round_records.append(record)
        candidates.append(
            {
                "round": round_idx,
                "candidate_idx": len(candidates) + 1,
                "code": code,
                "eval": eval_result,
                "action": action,
            }
        )

    best = _best_candidate(candidates)
    best_code = best["code"]
    ok, err = is_loadable_worldmodel(best_code)
    if not ok:
        best_code = generate_loadable_code(lambda: _generate_initial(rng.randint(1000, 9999)), max_attempts=4)
        best_eval = evaluate_transition_accuracy(best_code, validation_trajectories)
    else:
        best_eval = best["eval"]

    inducer.last_induce_stats = {
        "method": "pomdp_coder",
        "initial_candidates": max(1, initial_candidates),
        "rounds_run": len(round_records),
        "candidate_count": len(candidates),
        "best_accuracy": best_eval.accuracy,
        "best_mse": best_eval.mse,
        "best_load_error": best_eval.load_error,
        "llm_calls": llm_calls,
        "rounds": round_records,
    }
    return sanitize_generated_code(best_code)
