"""PoE-World synthesis: many programmatic experts + MLE weights (arXiv:2505.10819)."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence

from patchworld.worldmodel_data import Trajectory
from patchworld.worldmodel_inducer import Inducer

from .common import assemble_poeworld_model, fit_poe_weights, is_loadable_worldmodel, load_model, sanitize_generated_code, validate_code
from .metrics import iter_transitions
from .prompts import POE_ASSEMBLY_NOTE, POE_ENV_GUIDANCE, POE_EXPERT_BATCH_PROMPT, POE_EXPERT_FOCI

POE_ACTION_HELPERS = """
def _poe_parse_action_fields(action):
    action = str(action or "")
    fields = {}
    for key, value in re.findall(r"(\\w+)=([-?\\d.]+)", action):
        try:
            fields[key] = float(value)
        except (TypeError, ValueError):
            continue
    return fields
""".strip()

_ENV_ACTION_FORMS: Dict[str, Dict[str, List[str]]] = {
    "tworoom": {
        "move_up": ["move_up[dx=<float>,dy=<float>,speed=<float>]"],
        "move_down": ["move_down[dx=<float>,dy=<float>,speed=<float>]"],
        "move_left": ["move_left[dx=<float>,dy=<float>,speed=<float>]"],
        "move_right": ["move_right[dx=<float>,dy=<float>,speed=<float>]"],
        "hold": ["hold[dx=<float>,dy=<float>,speed=<float>]"],
    },
    "reacher": {
        "move_up": ["move_up[dx=<float>,dy=<float>,speed=<float>]"],
        "move_down": ["move_down[dx=<float>,dy=<float>,speed=<float>]"],
        "move_left": ["move_left[dx=<float>,dy=<float>,speed=<float>]"],
        "move_right": ["move_right[dx=<float>,dy=<float>,speed=<float>]"],
        "hold": ["hold[dx=<float>,dy=<float>,speed=<float>]"],
    },
    "cube": {
        "act": ["act[a0=<float>,a1=<float>,a2=<float>,a3=<float>,a4=<float>,speed=<float>]"],
    },
    "pusht": {
        "move_up": ["move_up[dx=<float>,dy=<float>,speed=<float>]"],
        "move_down": ["move_down[dx=<float>,dy=<float>,speed=<float>]"],
        "move_left": ["move_left[dx=<float>,dy=<float>,speed=<float>]"],
        "move_right": ["move_right[dx=<float>,dy=<float>,speed=<float>]"],
        "hold": ["hold[dx=<float>,dy=<float>,speed=<float>]"],
    },
}


def _env_action_forms(env_name: str) -> Dict[str, List[str]]:
    return _ENV_ACTION_FORMS.get((env_name or "").strip().lower(), _ENV_ACTION_FORMS["tworoom"])


def _fallback_expert_source(env_name: str) -> str:
    env = (env_name or "").strip().lower()
    if env == "cube":
        return """
def poe_expert_0(belief, action):
    fields = _poe_parse_action_fields(action)
    effector = belief.get("objects", {}).get("effector", {})
    if not effector:
        return {}
    updates = {}
    for idx in range(5):
        key = f"a{idx}"
        if key in fields:
            updates[key] = float(effector.get(key, 0.0)) + fields[key]
    if "speed" in fields:
        updates["speed"] = fields["speed"]
    return {"effector": updates} if updates else {}
""".strip()
    return """
def poe_expert_0(belief, action):
    fields = _poe_parse_action_fields(action)
    agent = belief.get("objects", {}).get("agent", {})
    if not agent:
        return {}
    dx = float(fields.get("dx", 0.0))
    dy = float(fields.get("dy", 0.0))
    return {
        "agent": {
            "x": float(agent.get("x", 0.0)) + dx,
            "y": float(agent.get("y", 0.0)) + dy,
        }
    }
""".strip()


def _extract_expert_modules(code: str) -> List[str]:
    blocks: List[str] = []
    pattern = re.compile(r"^def\s+poe_expert_\d+\(", re.MULTILINE)
    starts = [match.start() for match in pattern.finditer(code)]
    if not starts:
        return [code.strip()] if code.strip() else []
    for idx, start in enumerate(starts):
        end = starts[idx + 1] if idx + 1 < len(starts) else len(code)
        blocks.append(code[start:end].strip())
    return blocks


def _expert_module_to_worldmodel(module_code: str, env_name: str, expert_idx: int) -> str:
    class_name = f"PoeExpert{expert_idx}WorldModel"
    action_forms = _env_action_forms(env_name)
    return sanitize_generated_code(
        f"""
from patchworld.worldmodel_base import BaseWorldModel
import copy
import json
import re

{POE_ACTION_HELPERS}

{module_code}

class {class_name}(BaseWorldModel):
    ENV = {env_name!r}
    ACTION_FORMS = {action_forms!r}

    def init_belief(self):
        return {{"env": self.ENV, "source": "poeworld_expert", "step": 0, "goal": "", "objects": {{}}, "relations": {{}}, "metadata": {{}}}}

    def init_belief_from_observation(self, obs_text: str):
        return self.correct_belief(self.init_belief(), obs_text)

    def parse_observation(self, obs_text: str):
        belief = self.init_belief()
        section = ""
        for raw_line in str(obs_text or "").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("Environment:"):
                belief["env"] = line.split(":", 1)[1].strip()
            elif line.startswith("Source:"):
                belief["source"] = line.split(":", 1)[1].strip()
            elif line.startswith("Step:"):
                try:
                    belief["step"] = int(float(line.split(":", 1)[1].strip()))
                except ValueError:
                    belief["step"] = 0
            elif line.startswith("Goal:"):
                belief["goal"] = line.split(":", 1)[1].strip()
            elif line.startswith("Objects:"):
                section = "objects"
            elif line.startswith("Relations:"):
                section = "relations"
            elif line.startswith("Metadata:"):
                try:
                    belief["metadata"] = json.loads(line.split(":", 1)[1].strip())
                except Exception:
                    belief["metadata"] = {{}}
                section = ""
            elif section == "objects" and line.startswith("-"):
                match = re.match(r"^-\s*(\\w+):(\\w+)\\s*(.*)$", line)
                if match:
                    obj = {{"id": match.group(1), "category": match.group(2), "x": 0.0, "y": 0.0, "conf": 1.0}}
                    for key, value in re.findall(r"(\\w+)=(-?\\d+(?:\\.\\d+)?)", match.group(3)):
                        obj[key] = float(value)
                    belief["objects"][obj["id"]] = obj
            elif section == "relations" and line.startswith("-"):
                match = re.match(r"^-\s*(.+?)\\s*=\\s*(.+)$", line)
                if match:
                    belief["relations"][match.group(1).strip()] = {{"key": match.group(1).strip(), "value": match.group(2).strip()}}
        return belief

    def correct_belief(self, belief_prior, obs_text: str):
        parsed = self.parse_observation(obs_text)
        merged = copy.deepcopy(belief_prior if isinstance(belief_prior, dict) else self.init_belief())
        for key, value in parsed.items():
            if key in ("objects", "relations", "metadata") and isinstance(value, dict):
                merged.setdefault(key, {{}}).update(value)
            else:
                merged[key] = value
        return merged

    def extract_valid_action_forms(self):
        return self.ACTION_FORMS

    def predict_belief(self, belief, action: str):
        updated = copy.deepcopy(belief if isinstance(belief, dict) else self.init_belief())
        fn = globals().get("poe_expert_{expert_idx}")
        if callable(fn):
            patch = fn(updated, action)
            if isinstance(patch, dict):
                objs = patch.get("objects", patch)
                if isinstance(objs, dict):
                    for obj_id, fields in objs.items():
                        if obj_id == "objects" or not isinstance(fields, dict):
                            continue
                        updated.setdefault("objects", {{}}).setdefault(obj_id, {{}}).update(fields)
        updated["step"] = int(updated.get("step", 0)) + 1
        return updated

    def readout_observation(self, belief, action: str = "") -> str:
        return str(belief)
"""
    )


def _admit_expert(code: str, train_trajectories: Sequence[Trajectory]) -> bool:
    model, err = load_model(code)
    if model is None:
        return False
    probe = iter_transitions(train_trajectories, max_transitions=1)
    if not probe:
        return True
    try:
        transition = probe[0]
        obs = getattr(transition, "observation", "") or ""
        action = getattr(transition, "action", "") or ""
        belief = model.init_belief_from_observation(obs)
        model.predict_belief(belief, action)
    except Exception:
        return False
    return True


def poeworld_induce(
    inducer: Inducer,
    env_name: str,
    train_trajectories: Sequence[Trajectory],
    validation_trajectories: Sequence[Trajectory],
    *,
    experts_per_batch: int = 4,
    num_batches: int = 3,
    weight_delta: float = 0.01,
    seed: int = 42,
) -> str:
    """Synthesize many small experts, fit PoE weights with L-BFGS, prune low-weight experts."""
    env_key = (env_name or "").strip().lower()
    transitions = inducer._select_induction_transitions(list(train_trajectories))
    transition_text = inducer._format_transitions(transitions)
    foci = list(POE_EXPERT_FOCI)
    env_guidance = POE_ENV_GUIDANCE.get(env_key, POE_ENV_GUIDANCE["tworoom"])
    expert_codes: List[str] = []
    expert_idx = 0
    llm_calls = 0
    batch_attempts = 0
    max_batch_attempts = max(num_batches + 2, num_batches * 2)
    if env_key == "pusht":
        max_batch_attempts = max(max_batch_attempts, 8)

    while not expert_codes and batch_attempts < max_batch_attempts:
        for batch_idx in range(num_batches):
            focus = foci[(batch_attempts * num_batches + batch_idx) % len(foci)]
            prompt = POE_EXPERT_BATCH_PROMPT.format(
                env_name=env_name,
                batch_idx=batch_idx,
                experts_per_batch=experts_per_batch,
                focus=focus,
                env_guidance=env_guidance,
                transition_text=transition_text,
            )
            prompt = (
                f"{prompt}\n\n{POE_ASSEMBLY_NOTE}\n"
                "Return ONLY Python `def poe_expert_k(belief, action)` functions.\n"
                "Each function must be deterministic and small (one causal law).\n"
            )
            raw = sanitize_generated_code(inducer._extract_code(inducer.llm.generate(prompt)))
            llm_calls += 1
            for module in _extract_expert_modules(raw):
                code = _expert_module_to_worldmodel(module, env_name, expert_idx)
                if not _admit_expert(code, train_trajectories):
                    continue
                expert_codes.append(code)
                expert_idx += 1
        batch_attempts += 1

    if not expert_codes:
        fallback = _expert_module_to_worldmodel(_fallback_expert_source(env_name), env_name, 0)
        if _admit_expert(fallback, train_trajectories):
            expert_codes = [fallback]
            expert_idx = 1

    if not expert_codes:
        raise RuntimeError("PoE-World failed to synthesize any loadable experts")

    weights = fit_poe_weights(
        expert_codes,
        train_trajectories,
        seed=seed,
        weight_delta=weight_delta,
    )
    kept = [(code, weight) for code, weight in zip(expert_codes, weights) if weight >= weight_delta]
    if not kept:
        kept = [(expert_codes[0], 1.0)]
    expert_codes, weights = zip(*kept)
    combined = assemble_poeworld_model(expert_codes, weights)
    ok, err = is_loadable_worldmodel(combined)
    if not ok:
        raise RuntimeError(f"PoE-World combined model is not loadable: {err}")
    errors, score = validate_code(combined, validation_trajectories, label="poeworld_combined")
    inducer.last_induce_stats = {
        "method": "poeworld",
        "num_experts_synthesized": expert_idx,
        "num_experts_kept": len(expert_codes),
        "weights": list(weights),
        "weight_delta": weight_delta,
        "combined_score": score,
        "combined_errors": len(errors),
        "llm_calls": llm_calls,
        "batch_attempts": batch_attempts,
        "used_fallback_expert": expert_idx == 1 and len(expert_codes) == 1,
    }
    return combined
