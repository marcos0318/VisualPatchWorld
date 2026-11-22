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
    per_mode: bool = False,
    horizons: Sequence[int] = (1, 5, 25),
) -> Tuple[str, Dict[str, Any]]:
    """One-step residual summary on ``episodes``: per-field normalized RMSE, split by
    whether each object's state changed, plus the worst transitions. ``per_mode`` adds
    h-step error per contact mode and shows the worst transition of each mode."""
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
    picked = rows[:worst]
    if per_mode:
        from .modes import label_episodes, per_mode_errors

        pm = per_mode_errors(cand, params, episodes, keys, scales, horizons=horizons)
        stats["per_mode"] = pm
        hs = sorted({int(c.split("@")[1]) for d in pm.values() for c in d if c.startswith("nrmse@")})
        lines.append("Error by interaction mode. 'free' = the body did not move, 'onset' = first transition of a "
                     "contact, 'boxI.FACE' = sustained contact on that face of meta.contact_shape (u_min/u_max/"
                     "v_min/v_max). nrmse@h = normalized error of the state after rolling your program h steps "
                     "from a transition of that mode. Planning relies on every mode, including rare ones:")
        for m, d in sorted(pm.items(), key=lambda kv: -kv[1].get("n", 0)):
            lines.append(f"  {m:14s} n={d.get('n', 0):5d}  " + "  ".join(f"nrmse@{h}={d[f'nrmse@{h}']:.3f}" for h in hs if f"nrmse@{h}" in d))
        # Worst example per mode (worst modes first) instead of the globally worst, so
        # rare modes are shown even when common modes dominate the total error.
        labels = label_episodes(episodes, keys, scales)
        best_of: Dict[str, Any] = {}
        for r in rows:
            m = labels.get(id(r[0]), "unlabelled")
            best_of.setdefault(m, r)
        h0 = f"nrmse@{hs[-1]}" if hs else "nrmse@1"
        order = sorted(best_of, key=lambda m: -pm.get(m, {}).get(h0, 0.0))
        picked = [best_of[m] for m in order][:worst]
        picked_modes = order[:worst]
    lines.append(f"{len(picked)} worst transitions (state, action, true next_state, your predicted next_state):")
    for i, (t, pred, _, _) in enumerate(picked):
        act = "[" + ", ".join(f"{a:.3f}" for a in t.action) + "]"
        tag = f" [mode {picked_modes[i]}]" if per_mode else ""
        lines.append(f"-{tag} state={fmt(t.state)}\n  action={act}\n  true_next={fmt(t.next_state)}\n  predicted={fmt(pred)}")
    return "\n".join(lines), stats


def constant_report(initial: Mapping[str, float], fitted: Mapping[str, float], *, collapse: float = 0.02,
                    ratio: float = 10.0) -> str:
    """Flag fitted constants that collapsed, flipped sign or moved by more than ``ratio``x
    from the program's own initial guess. Such fits usually mean the term is compensating
    for a missing or wrong mechanism rather than measuring a physical quantity."""
    rows = []
    for k, v0 in initial.items():
        if k not in fitted:
            continue
        v0, v = float(v0), float(fitted[k])
        if abs(v0) < 1e-9:
            continue
        r = v / v0
        if abs(r) < collapse:
            rows.append(f"  {k}: {v0:g} -> {v:.4g} (collapsed to ~0: the data switched this term off)")
        elif r < 0:
            rows.append(f"  {k}: {v0:g} -> {v:.4g} (sign flipped)")
        elif r > ratio or r < 1 / ratio:
            rows.append(f"  {k}: {v0:g} -> {v:.4g} ({r:.3g}x the initial guess)")
    if not rows:
        return ""
    return ("Suspicious fitted constants (initial guess -> fitted). A constant that collapses, flips sign, or "
            "takes a physically implausible value (e.g. a negative mass, damping factor or restitution, a "
            "velocity multiplier outside [0, 1]) is usually absorbing the error of a missing or wrong "
            "mechanism; fix the mechanism instead of keeping the fudge term:\n" + "\n".join(rows))


def build_refine_prompt(
    cand: Candidate,
    params: Mapping[str, float],
    val_rollout: float,
    report: str,
    meta: Mapping[str, Any],
    *,
    num_revisions: int,
    bank_text: str = "",
    score_label: str = "held-out 5-step normalized RMSE",
    constants_text: str = "",
) -> str:
    return f"""You are improving an executable world model (a simulator written as code) for an
unknown control environment, from recorded transitions. Below is the current best program,
its constants after least-squares fitting, and where it fails.

Static scene metadata passed to `step` as `meta` (use it if relevant):
{json.dumps(dict(meta), sort_keys=True)}

Current program ({score_label} = {val_rollout:.4f}):
```python
{cand.code}
```
Fitted constants: {json.dumps({k: round(float(v), 5) for k, v in params.items()})}
{constants_text}

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
    score_label: str = "held-out 5-step normalized RMSE",
    constants_feedback: bool = False,
    code_suffix: str = "",
) -> List[Candidate]:
    ct = constant_report(cand.params, params) if constants_feedback else ""
    prompt = build_refine_prompt(cand, params, val_rollout, report, meta, num_revisions=num_revisions, bank_text=bank_text,
                                 score_label=score_label, constants_text=ct)
    reply = chat(prompt, model=model, temperature=temperature, max_tokens=24000)
    if log is not None:
        log.append({"kind": "refine", "prompt": prompt, **reply})
    out = []
    for code in extract_blocks(reply["text"]):
        try:
            c = compile_candidate(code + code_suffix)
            check_candidate(c, check_set)
        except Exception as exc:  # noqa: BLE001
            if log is not None:
                log.append({"kind": "refine_invalid", "error": f"{type(exc).__name__}: {exc}"})
            continue
        out.append(c)
    return out


def finite(x: float) -> float:
    return x if math.isfinite(x) else float("inf")
