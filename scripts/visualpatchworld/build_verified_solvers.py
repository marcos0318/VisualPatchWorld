#!/usr/bin/env python
"""Have an LLM write complete contact solvers, verified only by environment-agnostic tests.

The LLM sees a generic interface (a kinematic disc pushing a free planar rigid body made of
boxes) and a list of physical properties. Its solvers are run on synthetic scenarios
(`llm_family.solver_tests`); failures are returned for repair. Solvers that pass every test
are appended to a mechanism bank.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from visualpatchworld.llm_family.bank import BankEntry, _parse_json_list, load_bank, save_bank  # noqa: E402
from visualpatchworld.llm_family.propose import chat  # noqa: E402
from visualpatchworld.llm_family.solver_tests import INTERFACE_DOC, report, run_tests  # noqa: E402

PROMPT = """Write {n} distinct, complete Python solvers for planar contact between a kinematic disc
(a pusher, e.g. a robot finger) and a free rigid body whose shape is a union of boxes. They will
be added to a general library of mechanisms used to model unknown simulated environments, so
they must be physically correct and general. Use only `math` (import it inside the function).

Interface (exactly this signature):
{interface}

Your solvers are checked automatically on synthetic shapes for these properties:
  - no contact, no motion: a body at rest is not moved when the disc does not touch it;
  - detection uses the actual boxes (closest point on each box), not a bounding circle;
  - non-penetration: after the step the disc does not overlap the body;
  - a push through the centre of mass translates the body along the push without rotation;
  - an off-centre push rotates the body with the sign of the torque r x F;
  - rotation is about the centre of mass, not the body-frame origin;
  - a frictionless push on a flat face moves the centre of mass along the contact normal;
  - effective mass: an impulse J at contact offset r from the centre of mass changes the
    centre-of-mass velocity by J/m and the angular velocity by (r x J)/I.
Write one solver for each of these methods, in this order:
{methods}
Subdivide the step into substeps.

Return ONLY a JSON list of objects with keys "name" (snake_case), "mechanism" (one or two
sentences: the method and its assumptions), "params" (list of {{"name", "meaning",
"default"}}), and "code" (the full `def solve(disc, body, dt, p):` source, at most 70 lines)."""

REPAIR = """Your solver `{name}` failed some of the automatic physics checks.

Code:
```python
{code}
```

Test report:
{report}

Fix the solver so that it passes every check, keeping its method ({mechanism}). Return ONLY a
JSON object with keys "name", "mechanism", "params", "code" as before."""


DEFAULT_METHODS = [
    "dynamic, velocity-level: sequential impulses with position correction; the body keeps its momentum",
    "quasi-static, position-level: the body keeps no momentum and is moved directly out of penetration "
    "at every substep",
    "quasi-static, velocity-level: contact impulses at every substep, and the body's velocity is reset "
    "to zero after each substep",
]


def compile_solver(code: str):
    ns: dict = {"math": math}
    exec(compile(code, "<solver>", "exec"), ns)  # noqa: S102
    fn = ns.get("solve")
    if not callable(fn):
        raise ValueError("no function named solve")
    return fn


def evaluate(obj: dict) -> tuple[bool, str]:
    try:
        fn = compile_solver(obj["code"])
    except Exception as e:  # noqa: BLE001
        return False, f"[FAIL] compile: {type(e).__name__}: {e}"
    defaults = {}
    for prm in obj.get("params", []):
        try:
            defaults[str(prm["name"])] = float(prm["default"])
        except Exception:  # noqa: BLE001
            pass
    res = run_tests(fn, defaults)
    return all(r["passed"] for r in res), report(res)


def parse_obj(text: str) -> dict:
    try:
        lst = _parse_json_list(text)
        if lst and isinstance(lst[0], dict):
            return lst[0]
    except Exception:  # noqa: BLE001
        pass
    from visualpatchworld.llm_family.propose import _strip_think

    t = _strip_think(text)
    return json.loads(t[t.find("{") : t.rfind("}") + 1])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3.5-397B-A17B")
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--base", required=True, help="bank to append verified solvers to")
    p.add_argument("--out", required=True)
    p.add_argument("--start_from", default="", help="bank JSON whose general_solver entries are repaired instead of new ones")
    p.add_argument("--methods", default="", help="';'-separated method descriptions, one solver each")
    args = p.parse_args()
    methods = [m.strip() for m in args.methods.split(";") if m.strip()] or DEFAULT_METHODS
    args.n = len(methods)

    log: list = []
    if args.start_from:
        solvers = [{"name": e.name, "mechanism": e.mechanism.split(" Complete solver")[0], "code": e.snippet,
                    "params": []} for e in load_bank(Path(args.start_from)) if e.source == "general_solver"]
    else:
        prompt = PROMPT.format(n=args.n, interface=INTERFACE_DOC,
                               methods="\n".join(f"  {i + 1}. {m}" for i, m in enumerate(methods)))
        reply = chat(prompt, model=args.model, temperature=0.4, max_tokens=32000)
        log.append({"kind": "solvers", "prompt": prompt, **reply})
        solvers = [o for o in _parse_json_list(reply["text"]) if isinstance(o, dict) and o.get("code")]
    print(f"[solvers] {len(solvers)} proposed: {[s.get('name') for s in solvers]}", flush=True)

    verified = []
    for s in solvers:
        hist = []
        for r in range(args.rounds + 1):
            ok, rep = evaluate(s)
            npass = rep.count("[PASS]")
            hist.append({"round": r, "passed_all": ok, "n_pass": npass, "report": rep, "code": s["code"]})
            print(f"[solvers] {s.get('name')} round {r}: {npass}/9 pass", flush=True)
            if ok or r == args.rounds:
                break
            q = REPAIR.format(name=s.get("name"), code=s["code"], report=rep, mechanism=s.get("mechanism", ""))
            rr = chat(q, model=args.model, temperature=0.3, max_tokens=24000)
            log.append({"kind": "repair", "solver": s.get("name"), "round": r + 1, "prompt": q, **rr})
            try:
                s2 = parse_obj(rr["text"])
                if s2.get("code"):
                    s = {**s, **s2}
            except Exception as e:  # noqa: BLE001
                print(f"[solvers] repair parse failed: {e}", flush=True)
        s["history"] = hist
        if hist[-1]["passed_all"]:
            verified.append(s)

    entries = load_bank(Path(args.base))
    for s in verified:
        entries.append(BankEntry(
            name=str(s["name"]),
            mechanism=str(s.get("mechanism", "")) + " Complete solver with interface " +
            "solve(disc, body, dt, p); verified on generic physics checks.",
            equations="",
            snippet=s["code"],
            params=[{"name": str(q.get("name")), "meaning": f"{q.get('meaning')} (default {q.get('default')})"}
                    for q in s.get("params", []) if isinstance(q, dict)],
            tags=["contact", "pushing", "rigid_body", "rotation", "impulse", "solver", "complete"],
            topic="complete contact solvers",
            source="general_solver",
        ))
    save_bank(entries, Path(args.out))
    Path(args.out).with_suffix(".log.json").write_text(
        json.dumps({"log": log, "solvers": solvers}, indent=2, default=str), encoding="utf-8")
    print(f"[solvers] verified {len(verified)}/{len(solvers)}: {[s['name'] for s in verified]} -> {args.out}")


if __name__ == "__main__":
    main()
