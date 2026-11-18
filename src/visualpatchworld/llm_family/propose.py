"""Ask a code LLM to propose a family of parametric transition programs."""

from __future__ import annotations

import json
import math
import os
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from .traces import Episode, Transition, schema_summary

PROGRAM_CONTRACT = '''\
Each hypothesis must be ONE self-contained Python code block of this exact shape:

```python
# HYPOTHESIS: <short_snake_case_name>
# MECHANISM: <one sentence on the qualitative dynamics this program assumes>
PARAMS = {"<param_name>": <initial float>, ...}

def step(state, action, p, meta):
    """state: dict mapping "object.field" -> float (do not mutate).
    action: list of floats. p: dict of fitted constants (same keys as PARAMS).
    meta: dict of static scene metadata (may be empty).
    Return a NEW dict with exactly the same keys as `state`."""
    ...
```

Rules:
- Use only the Python standard library `math` module (already imported); no numpy, no I/O.
- Every free constant must live in PARAMS; the values you give are only initial guesses
  and will be fit to data by numerical optimization, so keep the number of constants small (<= 12).
- The program must be deterministic and must not raise for any finite input.
'''


@dataclass
class Candidate:
    name: str
    mechanism: str
    code: str
    params: Dict[str, float]
    step: Callable[..., Dict[str, float]] = field(repr=False)
    repaired: bool = False

    def to_json(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "mechanism": self.mechanism,
            "code": self.code,
            "init_params": dict(self.params),
            "repaired": self.repaired,
        }


def _fmt_state(state: Mapping[str, float]) -> str:
    return "{" + ", ".join(f'"{k}": {v:.3f}' for k, v in sorted(state.items())) + "}"


def _example_lines(episodes: Sequence[Episode], n: int, seed: int) -> List[str]:
    rng = random.Random(seed)
    pool = [t for ep in episodes for t in ep.transitions]
    picks = rng.sample(pool, min(n, len(pool)))
    lines = []
    for t in picks:
        act = "[" + ", ".join(f"{a:.3f}" for a in t.action) + "]"
        lines.append(f"state={_fmt_state(t.state)}\n  action={act}\n  next_state={_fmt_state(t.next_state)}")
    return lines


def build_prompt(
    episodes: Sequence[Episode],
    *,
    num_hypotheses: int,
    num_examples: int = 16,
    seed: int = 0,
    example_meta: Optional[Mapping[str, Any]] = None,
) -> str:
    schema = schema_summary(episodes)
    ranges = "\n".join(
        f"  {k}: [{v['min']:.3f}, {v['max']:.3f}]" for k, v in schema["state_ranges"].items()
    )
    examples = "\n".join(f"- {line}" for line in _example_lines(episodes, num_examples, seed))
    meta = json.dumps(example_meta or {}, sort_keys=True)
    return f"""You are building an executable world model (a simulator written as code) for an
unknown control environment. You only see recorded transitions; you do not see the
environment's source code or its physical parameters.

State variables (object.field) and observed ranges:
{ranges}
Action: {schema['action_dim']} floats, observed range {schema['action_min']} to {schema['action_max']}.
Each transition is one control macro-step (the action may be held for several frames).
Static scene metadata available to `meta` (example): {meta}

Example transitions:
{examples}

Task: propose {num_hypotheses} hypotheses for the transition law. The hypotheses must differ
in QUALITATIVE STRUCTURE (which variables are affected, linear vs nonlinear response,
first-order vs momentum, presence of obstacles/contact/gating, saturation, coupling between
objects, ...), not merely in parameter values. Cover the plausible alternatives a scientist
would want to discriminate between with experiments. Include at least one simple baseline.

{PROGRAM_CONTRACT}
Output the {num_hypotheses} code blocks and nothing else."""


_BLOCK_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.S)


def _strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)


def extract_blocks(text: str) -> List[str]:
    return [b.strip() for b in _BLOCK_RE.findall(_strip_think(text)) if "def step" in b]


def _header(code: str, tag: str) -> str:
    m = re.search(rf"#\s*{tag}:\s*(.+)", code)
    return m.group(1).strip() if m else ""


def compile_candidate(code: str) -> Candidate:
    namespace: Dict[str, Any] = {"math": math, "__builtins__": __builtins__}
    exec(compile(code, "<llm_candidate>", "exec"), namespace)
    params = namespace.get("PARAMS")
    step = namespace.get("step")
    if not isinstance(params, dict) or not callable(step):
        raise ValueError("candidate must define PARAMS dict and step()")
    params = {str(k): float(v) for k, v in params.items()}
    name = _header(code, "HYPOTHESIS") or f"candidate_{abs(hash(code)) % 10_000}"
    return Candidate(name=name, mechanism=_header(code, "MECHANISM"), code=code, params=params, step=step)


def check_candidate(cand: Candidate, transitions: Sequence[Transition]) -> None:
    for t in transitions:
        out = cand.step(dict(t.state), list(t.action), dict(cand.params), dict(t.meta))
        if not isinstance(out, dict):
            raise TypeError(f"step returned {type(out).__name__}, expected dict")
        missing = set(t.state) - set(out)
        if missing:
            raise KeyError(f"step output missing keys {sorted(missing)}")
        for k in t.state:
            v = float(out[k])
            if not math.isfinite(v):
                raise ValueError(f"non-finite output for {k}")


def _client():
    from openai import OpenAI

    return OpenAI(
        api_key=os.environ.get("PATCHWORLD_LLM_API_KEY") or os.environ["DEEPINFRA_API_KEY"],
        base_url=os.environ.get("PATCHWORLD_LLM_BASE_URL", "https://api.deepinfra.com/v1/openai"),
    )


def chat(prompt: str, *, model: Optional[str] = None, temperature: float = 0.7, max_tokens: int = 16000) -> Dict[str, Any]:
    model = model or os.environ.get("PATCHWORLD_MODEL", "Qwen/Qwen3.5-397B-A17B")
    t0 = time.time()
    resp = _client().chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    usage = getattr(resp, "usage", None)
    return {
        "model": model,
        "text": resp.choices[0].message.content or "",
        "elapsed_s": time.time() - t0,
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
    }


def _repair(code: str, error: str, *, model: Optional[str]) -> Dict[str, Any]:
    prompt = (
        "This world-model hypothesis program failed validation.\n\n"
        f"```python\n{code}\n```\n\nError: {error}\n\n{PROGRAM_CONTRACT}\n"
        "Return the corrected program as a single code block, keeping the same hypothesis."
    )
    return chat(prompt, model=model, temperature=0.2)


def propose_family(
    episodes: Sequence[Episode],
    *,
    num_hypotheses: int = 6,
    model: Optional[str] = None,
    seed: int = 0,
    temperature: float = 0.7,
    log: Optional[List[Dict[str, Any]]] = None,
) -> List[Candidate]:
    """Query the LLM once for a family, then validate (and repair once) each program."""
    meta = next((t.meta for ep in episodes for t in ep.transitions if t.meta), {})
    meta = {k: v for k, v in meta.items() if k not in {"episode_index", "raw_index", "next_raw_index"}}
    prompt = build_prompt(episodes, num_hypotheses=num_hypotheses, seed=seed, example_meta=meta)
    reply = chat(prompt, model=model, temperature=temperature)
    if log is not None:
        log.append({"kind": "propose", "prompt": prompt, **reply})
    check_set = [t for ep in episodes[:4] for t in ep.transitions][:32]
    family: List[Candidate] = []
    for code in extract_blocks(reply["text"]):
        cand: Optional[Candidate] = None
        try:
            cand = compile_candidate(code)
            check_candidate(cand, check_set)
        except Exception as exc:  # noqa: BLE001
            fix = _repair(code, f"{type(exc).__name__}: {exc}", model=model)
            if log is not None:
                log.append({"kind": "repair", "error": str(exc), **fix})
            blocks = extract_blocks(fix["text"])
            cand = None
            if blocks:
                try:
                    cand = compile_candidate(blocks[0])
                    check_candidate(cand, check_set)
                    cand.repaired = True
                except Exception:  # noqa: BLE001
                    cand = None
        if cand is not None:
            family.append(cand)
    names = set()
    for cand in family:
        while cand.name in names:
            cand.name += "_b"
        names.add(cand.name)
    return family
