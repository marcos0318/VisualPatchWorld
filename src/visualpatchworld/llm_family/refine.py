"""Residual-driven refinement: show the LLM where its best fitted program fails and ask for revisions."""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .fit import _safe_step
from .propose import PROGRAM_CONTRACT, Candidate, chat, check_candidate, compile_candidate, extract_blocks
from .traces import Episode


def _objects(keys: Sequence[str]) -> List[str]:
    return sorted({k.split(".", 1)[0] for k in keys})


def residual_report(
    cand: Candidate,
    params: Mapping[str, float],
    episodes: Sequence[Episode],
    keys: Sequence[str],
    scales: Mapping[str, float],
    *,
    worst: int = 6,
    move_tol: float = 1e-3,
) -> Tuple[str, Dict[str, Any]]:
    """One-step residual summary on ``episodes``: per-field normalized RMSE, split by
    whether each object's state changed, plus the worst transitions."""
    rows = []
    for ep in episodes:
        for t in ep.transitions:
            pred = _safe_step(cand, t.state, t.action, dict(params), t.meta) or {}
            err = {k: (float(pred.get(k, t.state.get(k, 0.0))) - t.next_state[k]) / scales[k] for k in keys if k in t.next_state}
            moved = {o: any(abs(t.next_state[k] - t.state[k]) / scales[k] > move_tol for k in keys if k.startswith(o + ".") and k in t.next_state)
                     for o in _objects(keys)}
            rows.append((t, pred, err, moved))
    stats: Dict[str, Any] = {"per_field": {}, "by_motion": {}}
    for k in keys:
        stats["per_field"][k] = float(np.sqrt(np.mean([r[2].get(k, 0.0) ** 2 for r in rows])))
    for o in _objects(keys):
        for flag in (True, False):
            sub = [r for r in rows if r[3][o] == flag]
            if sub:
                stats["by_motion"][f"{o} {'moved' if flag else 'did not move'} ({len(sub)} transitions)"] = {
                    k: float(np.sqrt(np.mean([r[2].get(k, 0.0) ** 2 for r in sub]))) for k in keys
                }
    rows.sort(key=lambda r: -sum(v * v for v in r[2].values()))
    fmt = lambda d: "{" + ", ".join(f'"{k}": {float(d.get(k, 0.0)):.3f}' for k in keys) + "}"
    lines = ["Normalized one-step RMSE per field (error / field scale):",
             "  " + ", ".join(f"{k}={v:.3f}" for k, v in stats["per_field"].items()),
             "Split by whether each object's state changed in the transition:"]
    for name, d in stats["by_motion"].items():
        lines.append(f"  {name}: " + ", ".join(f"{k}={v:.3f}" for k, v in d.items()))
    lines.append(f"The {worst} worst transitions (state, action, true next_state, your predicted next_state):")
    for t, pred, _, _ in rows[:worst]:
        act = "[" + ", ".join(f"{a:.3f}" for a in t.action) + "]"
        lines.append(f"- state={fmt(t.state)}\n  action={act}\n  true_next={fmt(t.next_state)}\n  predicted={fmt(pred)}")
    return "\n".join(lines), stats


def build_refine_prompt(
    cand: Candidate,
    params: Mapping[str, float],
    val_rollout: float,
    report: str,
    meta: Mapping[str, Any],
    *,
    num_revisions: int,
    bank_text: str = "",
) -> str:
    return f"""You are improving an executable world model (a simulator written as code) for an
unknown control environment, from recorded transitions. Below is the current best program,
its constants after least-squares fitting, and where it fails.

Static scene metadata passed to `step` as `meta` (use it if relevant):
{json.dumps(dict(meta), sort_keys=True)}

Current program (held-out 5-step normalized RMSE = {val_rollout:.4f}):
```python
{cand.code}
```
Fitted constants: {json.dumps({k: round(float(v), 5) for k, v in params.items()})}

Diagnostics on training transitions:
{report}
{bank_text}

Task: diagnose which physical mechanism the program gets wrong or omits, then write
{num_revisions} revised programs, each fixing the diagnosed failure in a different way. A revision
may be substantially longer than the current program (e.g. explicit geometry, contact
resolution, several integration substeps) if the diagnostics call for it. Before each code
block, give a two-sentence diagnosis.

{PROGRAM_CONTRACT}"""


def propose_revisions(
    cand: Candidate,
    params: Mapping[str, float],
    val_rollout: float,
    report: str,
    meta: Mapping[str, Any],
    check_set,
    *,
    num_revisions: int = 3,
    model: Optional[str] = None,
    bank_text: str = "",
    temperature: float = 0.7,
    log: Optional[List[Dict[str, Any]]] = None,
) -> List[Candidate]:
    prompt = build_refine_prompt(cand, params, val_rollout, report, meta, num_revisions=num_revisions, bank_text=bank_text)
    reply = chat(prompt, model=model, temperature=temperature, max_tokens=24000)
    if log is not None:
        log.append({"kind": "refine", "prompt": prompt, **reply})
    out = []
    for code in extract_blocks(reply["text"]):
        try:
            c = compile_candidate(code)
            check_candidate(c, check_set)
        except Exception as exc:  # noqa: BLE001
            if log is not None:
                log.append({"kind": "refine_invalid", "error": f"{type(exc).__name__}: {exc}"})
            continue
        out.append(c)
    return out


def finite(x: float) -> float:
    return x if math.isfinite(x) else float("inf")
