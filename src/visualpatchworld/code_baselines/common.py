"""Shared helpers for LeWM code baseline synthesis."""

from __future__ import annotations

import copy
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from patchworld.worldmodel_data import Trajectory, load_trajectories
from patchworld.worldmodel_inducer import Inducer
from patchworld.worldmodel_validator import load_model_from_code, validate_model

from visualpatchworld.patchworld_adapter import apply_pusht_patchworld_adapter, sanitize_pusht_model_code


apply_pusht_patchworld_adapter()

PATCHWORLD_REQUIRED_METHODS = (
    "init_belief",
    "init_belief_from_observation",
    "parse_observation",
    "correct_belief",
    "predict_belief",
    "readout_observation",
    "extract_valid_action_forms",
)

PATCHWORLD_IMPORT = "from patchworld.worldmodel_base import BaseWorldModel"


def _defines_local_baseworld(code: str) -> bool:
    return bool(re.search(r"^class\s+BaseWorldModel\s*[:\(]", code, flags=re.MULTILINE))


def strip_local_baseworld_stub(code: str) -> str:
    """Remove LLM-defined BaseWorldModel stubs that shadow PatchWorld's class."""
    if not _defines_local_baseworld(code):
        return code
    stripped = re.sub(
        r"^class\s+BaseWorldModel\s*[:\(][\s\S]*?(?=^class\s|\Z)",
        "",
        code,
        count=1,
        flags=re.MULTILINE,
    )
    return stripped.strip() + "\n"


def ensure_patchworld_import(code: str) -> str:
    cleaned = strip_local_baseworld_stub(code)
    if PATCHWORLD_IMPORT in cleaned:
        return cleaned
    return f"{PATCHWORLD_IMPORT}\n\n{cleaned}"


def has_patchworld_api_surface(code: str) -> bool:
    if not re.search(r"^class\s+\w+WorldModel\s*\(", code, flags=re.MULTILINE):
        return False
    if _defines_local_baseworld(code) and PATCHWORLD_IMPORT not in code:
        return False
    return all(f"def {method}(" in code for method in PATCHWORLD_REQUIRED_METHODS)


def is_loadable_worldmodel(code: str) -> Tuple[bool, Optional[str]]:
    sanitized = sanitize_generated_code(code)
    if not has_patchworld_api_surface(sanitized):
        return False, "missing PatchWorld BaseWorldModel API methods"
    model, err = load_model_from_code(sanitized)
    if model is None:
        return False, err or "load failed"
    return True, None


def generate_loadable_code(generate_fn, *, max_attempts: int = 3) -> str:
    """Retry LLM synthesis until PatchWorld can load the module."""
    last_err = "unknown"
    for _ in range(max_attempts):
        raw = generate_fn()
        ok, err = is_loadable_worldmodel(raw)
        if ok:
            return sanitize_generated_code(raw)
        last_err = err or "load failed"
    raise RuntimeError(f"failed to synthesize loadable model after {max_attempts} attempts: {last_err}")


def sanitize_generated_code_raw(code: str) -> str:
    """Strip patterns that break dynamic module loading (no PatchWorld rewrite)."""
    if not code:
        return code
    cleaned: List[str] = []
    for line in code.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("from .") or stripped.startswith("import ."):
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


def sanitize_generated_code(code: str) -> str:
    """Normalize generated code for PatchWorld dynamic loading."""
    cleaned = ensure_patchworld_import(sanitize_generated_code_raw(code))
    return sanitize_pusht_model_code(cleaned)


@dataclass
class ProgramCandidate:
    code: str
    score: float
    error_count: int
    errors: List[Dict[str, Any]]


def load_trajectory_glob(pattern: str) -> List[Trajectory]:
    from glob import glob

    paths = sorted(Path(p) for p in glob(pattern) if Path(p).is_file())
    if not paths:
        raise FileNotFoundError(f"No trajectory files matched: {pattern}")
    trajs = load_trajectories(paths)
    if not trajs:
        raise RuntimeError(f"No trajectories loaded from: {pattern}")
    return trajs


def subsample_trajectories(trajs: Sequence[Trajectory], max_count: int, seed: int) -> List[Trajectory]:
    items = list(trajs)
    if max_count <= 0 or len(items) <= max_count:
        return items
    rnd = random.Random(seed)
    return rnd.sample(items, max_count)


def make_inducer(
    llm: Any,
    *,
    max_refine_rounds: int = 3,
    contrastive_max_transitions: int = 40,
    patch_candidates_per_round: int = 3,
    patch_beam_size: int = 2,
    show_progress: bool = True,
) -> Inducer:
    return Inducer(
        llm=llm,
        max_refine_rounds=max_refine_rounds,
        contrastive_max_per_pattern=4,
        contrastive_max_transitions=contrastive_max_transitions,
        induction_strategy="tracepatch",
        patch_candidates_per_round=patch_candidates_per_round,
        patch_beam_size=patch_beam_size,
        patch_no_improve_patience=2,
        initial_load_error_retries=3,
        verbose=show_progress,
        show_progress=show_progress,
        induction_track="simple_state",
    )


def validate_code(
    code: str,
    trajectories: Sequence[Trajectory],
    *,
    max_trajectories: int = 0,
    label: str = "validate",
    show_progress: bool = False,
) -> Tuple[List[Dict[str, Any]], float]:
    limit = len(trajectories) if max_trajectories <= 0 else min(max_trajectories, len(trajectories))
    errors = validate_model(
        sanitize_generated_code(code),
        list(trajectories),
        max_trajectories=limit,
        show_progress=show_progress,
        progress_label=label,
        attach_step_text_to_first_n_errors=4,
    )
    score = _score_errors(errors)
    return errors, score


def _score_errors(errors: Sequence[Mapping[str, Any]]) -> float:
    if not errors:
        return 0.0
    if any(str(err.get("error_type", "")) == "load_error" for err in errors):
        return 1e6 + len(errors)
    score = 0.0
    for err in errors:
        score += 1.0 + (0.25 * _failure_priority(str(err.get("error_type", ""))))
    return score


def _failure_priority(error_type: str) -> int:
    priorities = {
        "load_error": 100,
        "parse_error": 90,
        "predict_error": 80,
        "readout_error": 70,
        "belief_mismatch": 60,
        "observation_mismatch": 50,
    }
    return priorities.get(error_type, 10)


def load_model(code: str) -> Tuple[Any, Optional[str]]:
    return load_model_from_code(sanitize_generated_code(code))


def state_vector_from_belief(belief: Mapping[str, Any]) -> Tuple[List[str], List[float]]:
    objects = belief.get("objects", {})
    if not isinstance(objects, Mapping):
        return [], []
    keys: List[str] = []
    values: List[float] = []
    for object_id in sorted(objects):
        obj = objects[object_id]
        if not isinstance(obj, Mapping):
            continue
        for field in ("x", "y", "z", "angle", "vx", "vy"):
            if field in obj:
                keys.append(f"{object_id}.{field}")
                try:
                    values.append(float(obj[field]))
                except (TypeError, ValueError):
                    values.append(0.0)
    return keys, values


def one_step_state_error(model: Any, transition: Any) -> float:
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
            return 1e3
        return float(sum((a - b) ** 2 for a, b in zip(pred_vals, tgt_vals)))
    except Exception:
        return 1e4


def fit_program_score(model_code: str, trajectories: Sequence[Trajectory], *, max_transitions: int = 80) -> float:
    model, err = load_model(model_code)
    if model is None:
        return 1e6
    total = 0.0
    count = 0
    for traj in trajectories:
        for transition in getattr(traj, "transitions", []):
            total += one_step_state_error(model, transition)
            count += 1
            if count >= max_transitions:
                return total / max(count, 1)
    return total / max(count, 1)


def thompson_sample_candidate(candidates: Sequence[ProgramCandidate], rng: random.Random) -> ProgramCandidate:
    if not candidates:
        raise ValueError("empty candidate pool")
    weights = []
    for cand in candidates:
        success = 1.0 / (1.0 + cand.score)
        weights.append(max(success, 1e-6))
    total = sum(weights)
    pick = rng.random() * total
    acc = 0.0
    for cand, weight in zip(candidates, weights):
        acc += weight
        if pick <= acc:
            return cand
    return candidates[-1]


def keep_top_candidates(candidates: Sequence[ProgramCandidate], beam_size: int) -> List[ProgramCandidate]:
    ordered = sorted(candidates, key=lambda c: (c.score, c.error_count))
    out: List[ProgramCandidate] = []
    seen: set[str] = set()
    for cand in ordered:
        key = cand.code.strip()
        if key in seen:
            continue
        seen.add(key)
        out.append(cand)
        if len(out) >= beam_size:
            break
    return out


def extract_worldmodel_classes(code: str) -> List[str]:
    return re.findall(r"^class\s+(\w+WorldModel)\b", code, flags=re.MULTILINE)


POE_WRAPPER_TEMPLATE = '''
from patchworld.worldmodel_base import BaseWorldModel
import copy
import json
import math
import re

__EXPERT_MODULES__

def _weighted_blend_beliefs(beliefs, weights):
    if not beliefs:
        return {}
    primary = copy.deepcopy(beliefs[0])
    if len(beliefs) == 1:
        return primary
    w = [max(0.0, float(x)) for x in weights]
    total = sum(w) or 1.0
    w = [x / total for x in w]
    objects = primary.setdefault("objects", {})
    numeric_keys = ("x", "y", "z", "angle", "vx", "vy", "conf")
    for obj_id, obj in list(objects.items()):
        if not isinstance(obj, dict):
            continue
        for key in numeric_keys:
            if key not in obj:
                continue
            acc = 0.0
            used = 0.0
            for belief, weight in zip(beliefs, w):
                other = belief.get("objects", {}).get(obj_id, {})
                if isinstance(other, dict) and key in other:
                    try:
                        acc += weight * float(other[key])
                        used += weight
                    except (TypeError, ValueError):
                        pass
            if used > 0:
                obj[key] = acc / used
    return primary


class PoeworldCombinedWorldModel(BaseWorldModel):
    WEIGHTS = __WEIGHTS_JSON__

    def __init__(self):
        self._experts = [__EXPERT_INSTANCES__]
        self._primary = self._experts[0]

    def init_belief(self):
        return self._primary.init_belief()

    def init_belief_from_observation(self, obs_text: str):
        return self._primary.init_belief_from_observation(obs_text)

    def parse_observation(self, obs_text: str):
        return self._primary.parse_observation(obs_text)

    def correct_belief(self, belief_prior, obs_text: str):
        return self._primary.correct_belief(belief_prior, obs_text)

    def extract_valid_action_forms(self):
        return self._primary.extract_valid_action_forms()

    def predict_belief(self, belief, action: str):
        preds = [expert.predict_belief(copy.deepcopy(belief), action) for expert in self._experts]
        return _weighted_blend_beliefs(preds, self.WEIGHTS)

    def readout_observation(self, belief, action: str = "") -> str:
        return self._primary.readout_observation(belief, action)
'''.strip()


def assemble_poeworld_model(expert_codes: Sequence[str], weights: Sequence[float]) -> str:
    expert_blocks: List[str] = []
    class_names: List[str] = []
    for idx, code in enumerate(expert_codes):
        names = extract_worldmodel_classes(code)
        if not names:
            raise RuntimeError(f"Expert {idx} did not define a *WorldModel class")
        class_names.append(names[0])
        expert_blocks.append(code.strip())

    instances = ", ".join(f"{name}()" for name in class_names)
    wrapper = (
        POE_WRAPPER_TEMPLATE.replace("__EXPERT_MODULES__", "\n\n".join(expert_blocks))
        .replace("__WEIGHTS_JSON__", json.dumps(list(weights)))
        .replace("__EXPERT_INSTANCES__", instances)
    )
    return sanitize_generated_code(wrapper)


def fit_poe_weights(
    expert_codes: Sequence[str],
    trajectories: Sequence[Trajectory],
    *,
    seed: int = 0,
    max_transitions: int = 120,
    weight_delta: float = 0.01,
    sigma: float = 2.0,
) -> List[float]:
    """MLE for exponentially-weighted product of experts (PoE-World eq. 5)."""
    from scipy import optimize

    from .metrics import iter_transitions, transition_matches

    n = len(expert_codes)
    if n == 0:
        return []
    if n == 1:
        return [1.0]

    models: List[Any] = []
    for code in expert_codes:
        model, err = load_model(code)
        if model is None:
            raise RuntimeError(f"Expert model failed to load: {err}")
        models.append(model)

    transitions = iter_transitions(trajectories, max_transitions=max_transitions)
    if not transitions:
        return [1.0 / n] * n

    log_liks: List[List[float]] = []
    for transition in transitions:
        row: List[float] = []
        for model in models:
            ok, mse = transition_matches(model, transition, tolerance=1e9)
            if not ok and mse >= 1e4:
                row.append(-50.0)
            else:
                row.append(-mse / (2.0 * sigma * sigma))
        log_liks.append(row)

    def nll(logits: Sequence[float]) -> float:
        weights = _softmax(list(logits))
        total = 0.0
        for row in log_liks:
            log_mix = _logsumexp([w + lp for w, lp in zip([math.log(max(w, 1e-12)) for w in weights], row)])
            total -= log_mix
        return total / len(log_liks)

    x0 = [0.0] * n
    result = optimize.minimize(nll, x0, method="L-BFGS-B")
    weights = _softmax(list(result.x))
    pruned = [w if w >= weight_delta else 0.0 for w in weights]
    total = sum(pruned) or 1.0
    return [w / total for w in pruned]


def _logsumexp(values: Sequence[float]) -> float:
    if not values:
        return float("-inf")
    max_v = max(values)
    return max_v + math.log(sum(math.exp(v - max_v) for v in values))


def _softmax(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    max_v = max(values)
    exps = [math.exp(v - max_v) for v in values]
    total = sum(exps) or 1.0
    return [v / total for v in exps]
