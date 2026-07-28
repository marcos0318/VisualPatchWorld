"""REx-style program selection used by WorldCoder."""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .metrics import TransitionEval, evaluate_transition_accuracy


@dataclass
class RExProgram:
    program_id: str
    code: str
    parent_id: Optional[str] = None
    refinement_count: int = 0
    successes: int = 0
    failures: int = 0
    accuracy: float = 0.0
    errors: List[Dict[str, Any]] = field(default_factory=list)
    load_error: bool = False

    @classmethod
    def from_eval(cls, code: str, eval_result: TransitionEval, *, parent: Optional["RExProgram"] = None) -> "RExProgram":
        parent_id = parent.program_id if parent else None
        refinement_count = parent.refinement_count + 1 if parent else 0
        return cls(
            program_id=uuid.uuid4().hex[:12],
            code=code,
            parent_id=parent_id,
            refinement_count=refinement_count,
            successes=eval_result.passed,
            failures=max(eval_result.total - eval_result.passed, 0),
            accuracy=eval_result.accuracy,
            errors=list(eval_result.errors),
            load_error=eval_result.load_error,
        )


def rex_select_program(programs: Sequence[RExProgram], rng: random.Random, *, explore_coef: float = 0.2) -> RExProgram:
    """Thompson sampling on replay accuracy plus REx exploration over refinement count."""
    if not programs:
        raise ValueError("empty program pool")
    samples: List[tuple[float, RExProgram]] = []
    for program in programs:
        alpha = program.successes + 1
        beta = program.failures + 1
        accuracy_draw = rng.betavariate(max(alpha, 1e-3), max(beta, 1e-3))
        explore_bonus = explore_coef / (1.0 + program.refinement_count)
        samples.append((accuracy_draw + explore_bonus, program))
    samples.sort(key=lambda item: item[0], reverse=True)
    return samples[0][1]


def evaluate_rex_program(
    code: str,
    trajectories: Sequence[Any],
    *,
    max_transitions: int = 0,
    env_name: str = "",
    tolerance: Optional[float] = None,
) -> TransitionEval:
    kwargs: Dict[str, Any] = {"max_transitions": max_transitions, "env_name": env_name}
    if tolerance is not None:
        kwargs["tolerance"] = tolerance
    return evaluate_transition_accuracy(code, trajectories, **kwargs)
