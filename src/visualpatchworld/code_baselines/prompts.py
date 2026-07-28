"""Prompt templates for paper-faithful code WM baselines."""

from __future__ import annotations

WORLDCODER_RULES = """
WorldCoder protocol (Tang et al., arXiv:2402.12275):
- Synthesize a reusable Python transition program from the provided oracle trajectories.
- Prefer modular, inspectable code over memorizing coordinates or episode indices.
- Generalize across held-out trajectories; keep numeric parsing robust.
""".strip()

PATCHWORLD_API_REQUIREMENTS = """
CRITICAL loading requirements (PatchWorld benchmark):
- Start with: from patchworld.worldmodel_base import BaseWorldModel
- Do NOT define your own BaseWorldModel class.
- Implement class {class_name}(BaseWorldModel) with these exact methods:
  parse_observation(self, obs_text: str) -> dict
  init_belief(self)
  init_belief_from_observation(self, obs_text: str)
  correct_belief(self, belief_prior, obs_text: str)
  predict_belief(self, belief, action: str)
  readout_observation(self, belief, action: str = "") -> str
  extract_valid_action_forms(self) -> dict[str, list[str]]
- Observations are LeWM text scene graphs (Environment, Source, Step, Goal, Objects, Relations, Metadata).
- belief["objects"] is a dict keyed by object id, not a list.
""".strip()

POE_EXPERT_FOCI = (
    "agent/end-effector kinematics from actions",
    "object-object interaction and contact effects",
    "goal-relative relations and static-object constraints",
    "pusher-block contact and quasi-static block motion",
)

POE_EXPERT_BATCH_PROMPT = """
You are synthesizing a batch of programmatic experts for PoE-World (arXiv:2505.10819).

Environment: {env_name}
Batch: {batch_idx}
Experts to write: {experts_per_batch}
Expert focus for this batch: {focus}

{env_guidance}

Each expert must be a small deterministic Python function:

def poe_expert_k(belief, action):
    \"\"\"One causal law; return partial object-field updates.\"\"\"
    fields = _poe_parse_action_fields(action)
    ...
    return {{"agent": {{"x": ..., "y": ...}}, ...}}  # only fields this expert sets

Rules:
- `action` is a string like `move_left[dx=-0.1000,dy=0.0000,speed=0.1000]`, NOT a dict.
- Parse numeric fields with `_poe_parse_action_fields(action)` (provided by the wrapper).
- Use the LeWM text scene-graph `belief` dict. `belief["objects"]` is keyed by object id.
- Do not write classes or imports. Output ONLY `def poe_expert_k(...)` functions.

Example transitions:
{transition_text}
""".strip()

POE_ENV_GUIDANCE = {
    "tworoom": "Object ids: agent, target. Actions: move_left/right/up/down/hold with dx, dy, speed.",
    "reacher": "Object ids: arm, finger, target. Actions: move_* with dx, dy, speed; target is static.",
    "cube": "Object ids: effector, block, goal. Actions: act[a0=...,a1=...,a2=...,a3=...,a4=...,speed=...].",
    "pusht": "Object ids: agent, block. Actions: move_* with dx, dy, speed. Block may move on contact.",
}

POE_EXPERT_PROMPT = """
You are implementing one expert for a Product-of-Experts world model (PoE-World, arXiv:2505.10819).

Environment: {env_name}
Expert focus: {focus}

Write ONE small Python class named {class_name}(BaseWorldModel) that specializes in:
{focus}

Requirements:
- Use the same text scene-graph observation template as the examples.
- Keep the expert self-contained and loadable.
- Do not reference other experts.
- Output ONLY Python code.

Example transitions:
{transition_text}
""".strip()

POE_ASSEMBLY_NOTE = """
PoE-World will combine multiple experts with learned weights outside this module.
Each expert should implement full BaseWorldModel methods, but may simplify unrelated dynamics.
""".strip()

GIF_GENERATE_PROMPT = """
GIF-MCTS action: GENERATE (Dainese et al., arXiv:2405.15383)

Implement a complete world model for environment `{env_name}` from scratch.

{rules}

{api_requirements}

Example transitions:
{transition_text}

Output ONLY a complete Python module with one `{class_name}` class inheriting BaseWorldModel.
""".strip()

GIF_IMPROVE_PROMPT = """
GIF-MCTS action: IMPROVE

Extend the current world model to better fit unseen transitions while preserving working behavior.

Environment: {env_name}

Current code:
```python
{code}
```

Improvement goals:
- Add missing transition rules suggested by the examples below.
- Keep parse/readout formats unchanged.
- Do not rewrite unrelated correct logic.

New example transitions:
{transition_text}

Output ONLY the full improved Python module.
""".strip()

GIF_FIX_PROMPT = """
GIF-MCTS action: FIX

Repair replay failures in the current world model.

Environment: {env_name}

{api_requirements}

Current code:
```python
{code}
```

Validation failures:
{errors}

Output ONLY the full repaired Python module with class {class_name}(BaseWorldModel).
""".strip()

WORLDCODER_LOAD_REPAIR_PROMPT = """
WorldCoder load repair round {round_idx}, candidate {candidate_idx}.

The current program cannot be loaded by PatchWorld because it does not implement the
required BaseWorldModel API. Rewrite it as a loadable PatchWorld module.

Environment: {env_name}

{api_requirements}

Current program (broken):
```python
{code}
```

Output ONLY the full repaired Python module with class {class_name}(BaseWorldModel).
""".strip()

WORLDCODER_REPAIR_PROMPT = """
WorldCoder REx-style repair round {round_idx}, candidate {candidate_idx}.

You are refining one program from a candidate pool. Fix the prioritized replay failures while
preserving behavior that already passes validation.

Environment: {env_name}

Current program:
```python
{code}
```

Failures:
{errors}

Output ONLY the full repaired Python module.
""".strip()

WORLDCODER_INITIAL_VARIANT_PROMPT = """
WorldCoder initial synthesis variant {variant_idx}.

Generate an alternative world-model implementation for `{env_name}` that may use a different
internal representation from other candidates in the program pool.

{rules}

{api_requirements}

Example transitions:
{transition_text}

Output ONLY Python code for class {class_name}(BaseWorldModel).
""".strip()

CWM_GAME_RULES = """
Lehrach et al. Code World Models for General Game Playing protocol (arXiv:2510.04542):
- Translate rules and trajectory examples into an executable Python world model.
- Represent state explicitly enough to support transition, observation readout, valid-action enumeration, and terminal/goal checks.
- Treat trajectory-derived validation failures as unit tests and refine the candidate program from their stack traces.
- This VPW port targets fully observed LeWM oracle scene graphs, so inference functions for hidden-information games and value-function synthesis are omitted.
- The final artifact must be a deterministic PatchWorld BaseWorldModel because CEM planning consumes concrete rollouts.
""".strip()

CWM_GAME_INITIAL_PROMPT = """
You are adapting Lehrach et al. (2025), "Code World Models for General Game Playing",
to VisualPatchWorld oracle scene graphs.

Environment: {env_name}
Candidate: {candidate_idx}

Protocol:
1. Infer a compact state representation from the LeWM scene-graph observations.
2. Implement executable transition logic, observation reconstruction, and valid-action forms.
3. Prefer game-rule-like causal code over memorizing individual examples.
4. The generated code will be tested against trajectory-derived unit tests and refined from failures.

{rules}

{api_requirements}

Example transitions:
{transition_text}

Output ONLY a complete Python module with one `{class_name}` class inheriting BaseWorldModel.
""".strip()

CWM_GAME_REFINE_PROMPT = """
Lehrach et al. CWM tree-search refinement step.

Environment: {env_name}
Tree retry: {retry_idx}
Candidate: {candidate_idx}

The current CWM failed trajectory-derived unit tests. Refine the same executable
world model using the failure details below. Preserve behavior that already passes
validation, keep the PatchWorld API exactly, and output a complete replacement module.

{api_requirements}

Current candidate:
```python
{code}
```

Failed unit tests / stack traces:
{errors}

Additional empirical transitions:
{transition_text}

Output ONLY the full revised Python module with class {class_name}(BaseWorldModel).
""".strip()

CWM_GAME_LOAD_REPAIR_PROMPT = """
Lehrach et al. CWM load-repair step.

Environment: {env_name}
Tree retry: {retry_idx}
Candidate: {candidate_idx}

The current CWM cannot be loaded or does not implement the required PatchWorld API.
Rewrite it as a loadable executable world model while preserving the intended rules.

{api_requirements}

Current candidate:
```python
{code}
```

Load failure / stack trace:
{errors}

Output ONLY the full repaired Python module with class {class_name}(BaseWorldModel).
""".strip()

POMDP_CODER_RULES = """
Curtis et al. POMDP-coder protocol (CoRL 2025):
- Treat the LLM as a structural prior over short probabilistic programs.
- Propose low-complexity transition and observation components, then test them
  against empirical trajectories.
- Prefer explicit variables, causal branches, and simple noise/uncertainty
  summaries over memorizing episode-specific coordinates.
- In this VPW port, the final artifact must be a deterministic PatchWorld
  BaseWorldModel because CEM planning consumes concrete rollouts.
""".strip()

POMDP_CODER_INITIAL_PROMPT = """
You are adapting Curtis et al. (CoRL 2025), "LLM-Guided Probabilistic Program
Induction for POMDP Model Estimation", to VisualPatchWorld oracle scene graphs.

Environment: {env_name}
Candidate: {candidate_idx}

Protocol:
1. Infer a compact latent state representation from the scene-graph text.
2. Write short programmatic transition and observation functions.
3. Use empirical trajectory examples as tests for the proposed model.
4. Keep uncertainty/probabilistic assumptions explicit in comments or helper
   constants, but return deterministic rollouts for the planner.

{rules}

{api_requirements}

Example transitions:
{transition_text}

Output ONLY a complete Python module with one `{class_name}` class inheriting BaseWorldModel.
""".strip()

POMDP_CODER_REFINE_PROMPT = """
Curtis et al. generate-test-refine step for POMDP-coder.

Environment: {env_name}

The current candidate failed empirical validation. Revise the same compact
probabilistic-program-style model so that the transition/readout behavior better
matches held-out trajectories. Preserve the PatchWorld API exactly.

Current candidate:
```python
{code}
```

Validation failures:
{errors}

Additional empirical transitions:
{transition_text}

Output ONLY the full revised Python module with class {class_name}(BaseWorldModel).
""".strip()

POMDP_CODER_LOAD_REPAIR_PROMPT = """
POMDP-coder load repair.

Environment: {env_name}

The current candidate cannot be loaded by the PatchWorld evaluator. Rewrite it
as a valid compact world-model program while preserving the intended transition
logic.

{api_requirements}

Current candidate:
```python
{code}
```

Output ONLY the full repaired Python module with class {class_name}(BaseWorldModel).
""".strip()
