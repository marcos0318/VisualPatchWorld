"""WorldCoder synthesis via REx program-pool refinement (Tang et al., arXiv:2402.12275)."""

from __future__ import annotations

import random
from typing import Any, Dict, List, Sequence

from patchworld.worldmodel_data import Trajectory
from patchworld.worldmodel_inducer import Inducer

from .common import generate_loadable_code, is_loadable_worldmodel, sanitize_generated_code
from .prompts import (
    PATCHWORLD_API_REQUIREMENTS,
    WORLDCODER_INITIAL_VARIANT_PROMPT,
    WORLDCODER_LOAD_REPAIR_PROMPT,
    WORLDCODER_REPAIR_PROMPT,
    WORLDCODER_RULES,
)
from .rex import RExProgram, evaluate_rex_program, rex_select_program


def _api_requirements(class_name: str) -> str:
    return PATCHWORLD_API_REQUIREMENTS.format(class_name=class_name)


def worldcoder_induce(
    inducer: Inducer,
    env_name: str,
    train_trajectories: Sequence[Trajectory],
    validation_trajectories: Sequence[Trajectory],
    *,
    initial_programs: int = 2,
    max_rounds: int = 3,
    pool_size: int = 8,
    seed: int = 42,
    llm_budget: int = 0,
) -> str:
    """Offline WorldCoder port: REx Thompson sampling + counterexample-guided repair."""
    env = (env_name or "").strip().lower()
    class_name = f"{env.capitalize()}WorldModel"
    from patchworld.worldmodel_inducer import SPATIAL_ENVS

    is_spatial = env in SPATIAL_ENVS
    rng = random.Random(seed)
    transitions = inducer._select_induction_transitions(list(train_trajectories))
    transition_text = inducer._format_transitions(transitions)
    rules = WORLDCODER_RULES
    api_requirements = _api_requirements(class_name)

    pool: List[RExProgram] = []
    base_prompt = inducer._build_loose_interface_prompt(
        env_name=env_name,
        class_name=class_name,
        rules=f"{rules}\n\n{api_requirements}",
        transition_text=transition_text,
    )
    llm_calls = 0

    initial_codes: List[str] = []
    for variant_idx in range(1, initial_programs + 1):
        if variant_idx == 1:
            prompt = base_prompt
        else:
            prompt = WORLDCODER_INITIAL_VARIANT_PROMPT.format(
                variant_idx=variant_idx,
                env_name=env_name,
                rules=rules,
                api_requirements=api_requirements,
                transition_text=transition_text,
                class_name=class_name,
            )

        def _generate(prompt_text: str = prompt) -> str:
            nonlocal llm_calls
            llm_calls += 1
            return inducer._extract_code(inducer.llm.generate(prompt_text))

        try:
            initial_codes.append(generate_loadable_code(lambda: _generate(), max_attempts=2))
        except RuntimeError:
            initial_codes.append(sanitize_generated_code(_generate()))

    for code in initial_codes:
        eval_result = evaluate_rex_program(code, validation_trajectories, env_name=env)
        pool.append(RExProgram.from_eval(code, eval_result))

    round_records: List[Dict[str, Any]] = []
    for round_idx in range(1, max_rounds + 1):
        if llm_budget and llm_calls >= llm_budget:
            break
        loadable_pool = [p for p in pool if not p.load_error]
        # Meter-scale envs rarely hit exact 1.0 under a tight tolerance; keep a
        # high-accuracy early-exit so PushT still short-circuits when saturated.
        if loadable_pool and max(p.accuracy for p in loadable_pool) >= 1.0:
            break

        round_records.append({"round": round_idx, "repairs": []})
        repairs_per_round = max(1, inducer.patch_candidates_per_round)
        for _ in range(repairs_per_round):
            if llm_budget and llm_calls >= llm_budget:
                break
            parent = rex_select_program(pool, rng)
            if parent.load_error:
                repair_prompt = WORLDCODER_LOAD_REPAIR_PROMPT.format(
                    round_idx=round_idx,
                    candidate_idx=len(round_records[-1]["repairs"]) + 1,
                    env_name=env_name,
                    api_requirements=api_requirements,
                    code=parent.code,
                    class_name=class_name,
                )
            elif not parent.errors:
                continue
            else:
                repair_prompt = WORLDCODER_REPAIR_PROMPT.format(
                    round_idx=round_idx,
                    candidate_idx=len(round_records[-1]["repairs"]) + 1,
                    env_name=env_name,
                    code=parent.code,
                    errors=inducer._format_errors(parent.errors[:12]),
                )
            child_code = sanitize_generated_code(inducer._extract_code(inducer.llm.generate(repair_prompt)))
            llm_calls += 1
            eval_result = evaluate_rex_program(child_code, validation_trajectories, env_name=env)
            child = RExProgram.from_eval(child_code, eval_result, parent=parent)
            pool.append(child)
            round_records[-1]["repairs"].append(
                {
                    "parent_id": parent.program_id,
                    "parent_accuracy": parent.accuracy,
                    "child_accuracy": child.accuracy,
                    "child_load_error": child.load_error,
                    "child_refinement_count": child.refinement_count,
                }
            )

        pool.sort(key=lambda p: (p.load_error, -p.accuracy, p.refinement_count, p.failures))
        pool = pool[:pool_size]

    loadable = [p for p in pool if not p.load_error]
    if loadable:
        best = max(loadable, key=lambda p: (p.accuracy, -p.refinement_count, -p.failures))
    else:
        fallback_code = generate_loadable_code(
            lambda: sanitize_generated_code(inducer._extract_code(inducer.llm.generate(base_prompt))),
            max_attempts=3,
        )
        llm_calls += 3
        eval_result = evaluate_rex_program(fallback_code, validation_trajectories, env_name=env)
        best = RExProgram.from_eval(fallback_code, eval_result)

    inducer.last_induce_stats = {
        "method": "worldcoder_rex",
        "pool_size": len(pool),
        "loadable_pool_size": len(loadable),
        "rounds_run": len(round_records),
        "best_accuracy": best.accuracy,
        "best_load_error": best.load_error,
        "best_refinement_count": best.refinement_count,
        "llm_calls": llm_calls,
        "rounds": round_records,
    }
    ok, err = is_loadable_worldmodel(best.code)
    if not ok:
        raise RuntimeError(f"WorldCoder failed to produce a loadable model: {err}")
    return best.code
