"""Runtime PatchWorld adapter for VisualPatchWorld PushT experiments.

This module keeps PushT-specific integration local to `le-wm-code`. It does
not edit files in the sibling PatchWorld checkout; instead it augments imported
PatchWorld modules in memory before induction/evaluation starts.
"""

from __future__ import annotations

import re
from typing import Any, Dict


PUSHT_NUMERIC_COERCION_HELPERS = '''
def _vpw_as_float(value, default=0.0):
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
    return default


def _vpw_coerce_belief_numerics(belief):
    if not isinstance(belief, dict):
        return belief
    import copy

    b = copy.deepcopy(belief)
    numeric_obj_keys = ("x", "y", "vx", "vy", "angle", "conf")
    for obj in b.get("objects", {}).values():
        if isinstance(obj, dict):
            for key in numeric_obj_keys:
                if key in obj:
                    obj[key] = _vpw_as_float(obj[key])
    for rel in _VPW_RELATIONS_ITER:
        if isinstance(rel, dict) and "value" in rel:
            value = rel["value"]
            if not isinstance(value, str):
                rel["value"] = _vpw_as_float(value)
    step = b.get("step")
    if isinstance(step, dict) and "value" in step and not isinstance(step["value"], str):
        step["value"] = int(_vpw_as_float(step["value"]))
    return b
'''.strip()

# Generated models may store relations as a dict or as a list of raw strings.
_LEGACY_RELATIONS_ITER = 'b.get("relations", {}).values()'
_VPW_RELATIONS_ITER = (
    '(b.get("relations").values() if isinstance(b.get("relations"), dict) else [])'
)
PUSHT_NUMERIC_COERCION_HELPERS = PUSHT_NUMERIC_COERCION_HELPERS.replace(
    "_VPW_RELATIONS_ITER", _VPW_RELATIONS_ITER
)


LEWM_SCENE_GRAPH_GUIDANCE = """
- Observations are fixed-template text scene graphs: Environment, Source, Step, Goal, Objects, Relations, Metadata.
- Implement parse_observation / readout_observation so they round-trip the template exactly.
- Learn reusable transition rules from numeric object fields; do not memorize episode indices or raw HDF5 indices from Metadata.
- Parse actions from bracket notation and apply continuous control values to the appropriate objects.
- Recompute relation fields (distance, near, relative_direction) from predicted object geometry instead of copying stale text.
""".strip()

TWOROOM_PROMPT_GUIDANCE = LEWM_SCENE_GRAPH_GUIDANCE + """
- Object IDs: `agent` (navigator) and `target` (goal). Actions: move_left/right/up/down/hold with dx, dy, speed.
- The target is static; predict agent motion from action deltas and keep target pose unless evidence shows otherwise.
- Two-room navigation may involve walls/doors encoded in Metadata; generalize motion rather than hardcoding train coordinates.
""".strip()

CUBE_PROMPT_GUIDANCE = LEWM_SCENE_GRAPH_GUIDANCE + """
- Object IDs: `block`, `effector`, `goal`. Actions: act[a0=...,a1=...,a2=...,a3=...,a4=...,speed=...] (5-D control).
- Predict effector and block motion from the 5-D action vector; goal pose is static.
- Update block-effector and block-goal relations from predicted positions.
""".strip()

REACHER_PROMPT_GUIDANCE = LEWM_SCENE_GRAPH_GUIDANCE + """
- Object IDs: `arm` (joint_state x,y as proxy), `finger` (end effector), `target` (goal). Actions: move_* with dx, dy, speed.
- Target is static; predict finger/end-effector motion from actions. Arm joint proxy fields may co-move with finger.
""".strip()

PUSHT_PROMPT_GUIDANCE = """
- Treat PushT observations as structured scene graphs rendered in a fixed text template: Environment, Source, Step, Goal, Objects, Relations, Metadata.
- Parse object lines into a canonical belief with `agent` and `block` entries. Preserve numeric fields: x, y, block angle, agent velocity, confidence, and source.
- Parse action strings such as `move_left[dx=-0.1000,dy=0.0000,speed=0.1000]`; the bracket values are continuous deltas and should drive the predicted agent pose.
- Do not memorize episode indices, raw indices, or exact train coordinates. Learn reusable kinematic rules over the object fields.
- Preserve object IDs exactly as `agent` and `block`; VLM observations may vary in source but should normalize to the same IDs.
- Update the `agent relative_direction_to block`, `agent distance_to block`, and `agent near block` relations from the predicted object positions rather than copying stale relation text.
- If evidence is insufficient to move the block, keep the block pose unchanged and keep uncertainty in the belief instead of fabricating contact dynamics.
- Render observations in the same section order and line format used in the examples so `parse_observation(readout_observation(...))` round-trips.
- In `readout_observation`, coerce numeric object/relation fields with `float(...)` before using `:.3f` formatting; string coordinates cause runtime format errors.
""".strip()


def _looks_like_pusht_model_code(code: str) -> bool:
    lowered = code.lower()
    return "pusht" in lowered or "circular_pusher" in lowered


def sanitize_pusht_model_code(code: str) -> str:
    """Inject numeric coercion helpers into generated PushT world models."""
    if code and "_vpw_coerce_belief_numerics" in code:
        return code.replace(f"for rel in {_LEGACY_RELATIONS_ITER}:", f"for rel in {_VPW_RELATIONS_ITER}:")
    if not code:
        return code
    if not _looks_like_pusht_model_code(code):
        return code

    sanitized = code
    if "_vpw_as_float" not in sanitized:
        class_match = re.search(r"^class\s+\w+WorldModel\b", sanitized, flags=re.MULTILINE)
        if class_match:
            sanitized = (
                sanitized[: class_match.start()]
                + PUSHT_NUMERIC_COERCION_HELPERS
                + "\n\n"
                + sanitized[class_match.start() :]
            )
        else:
            sanitized = PUSHT_NUMERIC_COERCION_HELPERS + "\n\n" + sanitized

    readout_match = re.search(
        r"(def readout_observation\(self,\s*belief(?::\s*dict)?,\s*action:\s*str\s*=\s*\"\"\)\s*->\s*str:\n)([\t ]+)",
        sanitized,
    )
    if readout_match and "belief = _vpw_coerce_belief_numerics(belief)" not in sanitized:
        indent = readout_match.group(2)
        insert_at = readout_match.start(2)
        sanitized = (
            sanitized[:insert_at]
            + f"{indent}belief = _vpw_coerce_belief_numerics(belief)\n"
            + sanitized[insert_at:]
        )
    # Some code models occasionally drop indentation on the first local
    # statement after our readout injection (`lines = []`). Repair this narrow
    # pattern before dynamic loading so otherwise valid models can be evaluated.
    sanitized = sanitized.replace(
        "\n        belief = _vpw_coerce_belief_numerics(belief)\nlines = []\n",
        "\n        belief = _vpw_coerce_belief_numerics(belief)\n        lines = []\n",
    )

    return sanitized


LEWM_ENVS = frozenset({"pusht", "tworoom", "cube", "reacher"})


def apply_pusht_patchworld_adapter() -> None:
    """Register LeWM envs with imported PatchWorld modules in memory."""

    from patchworld import inducer_constants, inducer_prompts, worldmodel_inducer, worldmodel_validator

    spatial_envs = set(inducer_constants.SPATIAL_ENVS)
    spatial_envs.update(LEWM_ENVS)
    patched_spatial_envs = frozenset(spatial_envs)
    inducer_constants.SPATIAL_ENVS = patched_spatial_envs
    worldmodel_inducer.SPATIAL_ENVS = patched_spatial_envs
    inducer_prompts.ENV_PROMPT_GUIDANCE.update(
        {
            "pusht": PUSHT_PROMPT_GUIDANCE,
            "tworoom": TWOROOM_PROMPT_GUIDANCE,
            "cube": CUBE_PROMPT_GUIDANCE,
            "reacher": REACHER_PROMPT_GUIDANCE,
        }
    )

    original_load = worldmodel_validator.load_model_from_code

    def _load_with_pusht_safety(code: str):
        return original_load(sanitize_pusht_model_code(code))

    worldmodel_validator.load_model_from_code = _load_with_pusht_safety


def extract_pusht_fields(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """Best-effort structured fields for local reporting.

    PatchWorld's generic token metrics still run unchanged. This helper gives
    VisualPatchWorld reports a stable place to add PushT-aware metrics later.
    """

    if not isinstance(parsed, dict):
        return {}
    fields: Dict[str, Any] = {}
    for key in ("objects", "object_states", "scene_objects"):
        if key in parsed:
            fields["objects"] = parsed[key]
            break
    for key in ("relations", "scene_relations"):
        if key in parsed:
            fields["relations"] = parsed[key]
            break
    return fields
