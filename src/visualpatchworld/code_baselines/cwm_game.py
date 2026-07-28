"""Lehrach et al. CWM tree-search synthesis for VPW oracle graphs."""

from __future__ import annotations

import random
import traceback
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from patchworld.worldmodel_data import Trajectory
from patchworld.worldmodel_inducer import Inducer

from .common import generate_loadable_code, is_loadable_worldmodel, load_model, sanitize_generated_code, state_vector_from_belief
from .metrics import TransitionEval, iter_transitions
from .prompts import (
    CWM_GAME_INITIAL_PROMPT,
    CWM_GAME_LOAD_REPAIR_PROMPT,
    CWM_GAME_REFINE_PROMPT,
    CWM_GAME_RULES,
    PATCHWORLD_API_REQUIREMENTS,
)
from .rex import RExProgram


def _api_requirements(class_name: str) -> str:
    return PATCHWORLD_API_REQUIREMENTS.format(class_name=class_name)


def _transition_matches_with_trace(model: Any, transition: Any, *, tolerance: float = 5.0) -> Tuple[bool, float, Dict[str, Any]]:
    obs = getattr(transition, "observation", "") or ""
    nxt = getattr(transition, "next_observation", "") or ""
    action = getattr(transition, "action", "") or ""
    try:
        belief = model.init_belief_from_observation(obs)
        predicted = model.predict_belief(belief, action)
        if not isinstance(predicted, Mapping) or not predicted.get("objects"):
            readout = model.readout_observation(predicted, action)
            predicted = model.parse_observation(readout)
        pred_keys, pred_vals = state_vector_from_belief(predicted)
        tgt = model.parse_observation(nxt)
        tgt_keys, tgt_vals = state_vector_from_belief(tgt)
        if not pred_keys or pred_keys != tgt_keys:
            return (
                False,
                1e4,
                {
                    "error_type": "state_key_mismatch",
                    "action": action,
                    "predicted_keys": pred_keys,
                    "target_keys": tgt_keys,
                    "observation": obs[:700],
                    "next_observation": nxt[:700],
                },
            )
        mse = float(sum((a - b) ** 2 for a, b in zip(pred_vals, tgt_vals)))
        ok = mse <= tolerance * tolerance * max(len(pred_vals), 1)
        return (
            ok,
            mse,
            {
                "error_type": "observation_mismatch",
                "action": action,
                "observation": obs[:700],
                "next_observation": nxt[:700],
                "mse": mse,
            },
        )
    except Exception as exc:
        return (
            False,
            1e4,
            {
                "error_type": "runtime_error",
                "action": action,
                "message": str(exc),
                "traceback": traceback.format_exc()[-4000:],
                "observation": obs[:700],
                "next_observation": nxt[:700],
            },
        )


def evaluate_cwm_program(code: str, trajectories: Sequence[Any], *, max_transitions: int = 0) -> TransitionEval:
    model, err = load_model(code)
    if model is None:
        return TransitionEval(
            accuracy=0.0,
            passed=0,
            total=0,
            load_error=True,
            errors=[{"error_type": "load_error", "message": err or "load failed", "traceback": err or "load failed"}],
            mse=1e6,
        )

    transitions = iter_transitions(trajectories, max_transitions=max_transitions)
    passed = 0
    mse_total = 0.0
    errors: List[Dict[str, Any]] = []
    for transition in transitions:
        ok, mse, error = _transition_matches_with_trace(model, transition)
        mse_total += mse
        if ok:
            passed += 1
        elif len(errors) < 12:
            errors.append(error)
    total = len(transitions)
    return TransitionEval(
        accuracy=passed / total if total else 0.0,
        passed=passed,
        total=total,
        load_error=False,
        errors=errors,
        mse=mse_total / max(total, 1),
    )


def cwm_select_program(programs: Sequence[RExProgram], rng: random.Random, *, heuristic_weight: float = 5.0) -> RExProgram:
    """Lehrach-style tree selection from unit-test pass rate and refinement count."""
    if not programs:
        raise ValueError("empty CWM tree")
    samples: List[tuple[float, RExProgram]] = []
    for program in programs:
        h = max(0.0, min(1.0, float(program.accuracy)))
        alpha = max(1.0 + heuristic_weight * h, 1e-3)
        beta = max(1.0 + heuristic_weight * (1.0 - h), 1e-3)
        accuracy_draw = rng.betavariate(alpha, beta)
        explore_bonus = 0.1 / (1.0 + program.refinement_count)
        samples.append((accuracy_draw + explore_bonus, program))
    samples.sort(key=lambda item: item[0], reverse=True)
    return samples[0][1]


def _best_cwm(pool: Sequence[RExProgram]) -> RExProgram | None:
    loadable = [p for p in pool if not p.load_error]
    if not loadable:
        return None
    return max(loadable, key=lambda p: (p.accuracy, -p.refinement_count, -p.failures, -p.mse if hasattr(p, "mse") else 0.0))


def cwm_game_induce(
    inducer: Inducer,
    env_name: str,
    train_trajectories: Sequence[Trajectory],
    validation_trajectories: Sequence[Trajectory],
    *,
    retries: int = 50,
    initial_programs: int = 2,
    pool_size: int = 8,
    seed: int = 42,
) -> str:
    """Tree-search Code World Model synthesis port from Lehrach et al. (2025)."""
    env = (env_name or "").strip().lower()
    class_name = f"{env.capitalize()}WorldModel"
    rng = random.Random(seed)
    transitions = inducer._select_induction_transitions(list(train_trajectories))
    transition_text = inducer._format_transitions(transitions)
    api_requirements = _api_requirements(class_name)

    pool: List[RExProgram] = []
    llm_calls = 0
    records: List[Dict[str, Any]] = []

    for candidate_idx in range(1, initial_programs + 1):
        prompt = CWM_GAME_INITIAL_PROMPT.format(
            env_name=env_name,
            candidate_idx=candidate_idx,
            rules=CWM_GAME_RULES,
            api_requirements=api_requirements,
            transition_text=transition_text,
            class_name=class_name,
        )

        def _generate(prompt_text: str = prompt) -> str:
            nonlocal llm_calls
            llm_calls += 1
            return inducer._extract_code(inducer.llm.generate(prompt_text))

        try:
            code = generate_loadable_code(lambda: _generate(), max_attempts=2)
        except RuntimeError:
            code = sanitize_generated_code(_generate())
        eval_result = evaluate_cwm_program(code, validation_trajectories)
        pool.append(RExProgram.from_eval(code, eval_result))
        records.append(
            {
                "retry": 0,
                "candidate_idx": candidate_idx,
                "accuracy": eval_result.accuracy,
                "load_error": eval_result.load_error,
                "errors": eval_result.errors[:3],
            }
        )

    for retry_idx in range(1, retries + 1):
        best = _best_cwm(pool)
        if best and best.accuracy >= 1.0:
            break

        parent = cwm_select_program(pool, rng)
        candidate_idx = len(pool) + 1
        if parent.load_error:
            prompt = CWM_GAME_LOAD_REPAIR_PROMPT.format(
                env_name=env_name,
                retry_idx=retry_idx,
                candidate_idx=candidate_idx,
                api_requirements=api_requirements,
                code=parent.code,
                errors=inducer._format_errors(parent.errors[:12]),
                class_name=class_name,
            )
        elif parent.errors:
            prompt = CWM_GAME_REFINE_PROMPT.format(
                env_name=env_name,
                retry_idx=retry_idx,
                candidate_idx=candidate_idx,
                api_requirements=api_requirements,
                code=parent.code,
                errors=inducer._format_errors(parent.errors[:12]),
                transition_text=transition_text,
                class_name=class_name,
            )
        else:
            continue

        child_code = sanitize_generated_code(inducer._extract_code(inducer.llm.generate(prompt)))
        llm_calls += 1
        eval_result = evaluate_cwm_program(child_code, validation_trajectories)
        child = RExProgram.from_eval(child_code, eval_result, parent=parent)
        pool.append(child)
        records.append(
            {
                "retry": retry_idx,
                "parent_id": parent.program_id,
                "parent_accuracy": parent.accuracy,
                "child_id": child.program_id,
                "child_accuracy": child.accuracy,
                "child_load_error": child.load_error,
                "child_refinement_count": child.refinement_count,
                "errors": eval_result.errors[:3],
            }
        )
        pool.sort(key=lambda p: (p.load_error, -p.accuracy, p.refinement_count, p.failures))
        pool = pool[:pool_size]

    best = _best_cwm(pool)
    if best is None:
        raise RuntimeError("CWM-Game failed to produce a loadable model")

    inducer.last_induce_stats = {
        "method": "cwm_game_tree_search",
        "retries_budget": retries,
        "initial_programs": initial_programs,
        "pool_size": len(pool),
        "loadable_pool_size": len([p for p in pool if not p.load_error]),
        "best_accuracy": best.accuracy,
        "best_refinement_count": best.refinement_count,
        "llm_calls": llm_calls,
        "records": records,
    }
    ok, err = is_loadable_worldmodel(best.code)
    if not ok:
        raise RuntimeError(f"CWM-Game failed to produce a loadable model: {err}")
    return best.code
