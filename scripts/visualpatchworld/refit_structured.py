#!/usr/bin/env python
"""Apply the structured (identification-style) refit to every candidate of an induced
family, starting from its stored fitted constants, and re-run offline selection.

Equivalent to inducing with ``fit_candidate(structured=True)`` (the standard fit is kept
when the structured refit does not lower held-out error). Writes ``structured.json`` and
``models/structured_<objective>.py`` into the run directory.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from visualpatchworld.llm_family.fit import nrmse, structured_refit  # noqa: E402
from visualpatchworld.llm_family.propose import compile_candidate  # noqa: E402
from visualpatchworld.llm_family.render import render_model  # noqa: E402
from visualpatchworld.llm_family.traces import load_episodes, schema_summary, state_keys, state_scales  # noqa: E402


def _load(run_dir: Path):
    fam = json.loads((run_dir / "family.json").read_text())
    a = fam["args"]
    data = Path(a["data_dir"])
    train = load_episodes(Path(a["train_file"] or data / f"{a['env']}_oracle_traj_train.jsonl"), max_episodes=a["max_train_episodes"])
    val = load_episodes(Path(a["val_file"] or data / f"{a['env']}_oracle_traj_val.jsonl"), max_episodes=a["max_val_episodes"])
    return fam, train, val


def _job(spec):
    run_dir, name, objective = spec
    fam, train, val = _load(Path(run_dir))
    keys = state_keys(train)
    scales = state_scales(train, keys)
    k = int(fam["args"].get("rollout_k", 5))
    code = next(c["code"] for c in fam["candidates"] if c["name"] == name)
    cand = compile_candidate(code)
    old = fam["fits"][objective][name]
    t0 = time.time()
    if not cand.params:
        return name, objective, old["val_rollout_nrmse"], old["val_rollout_nrmse"], old["params"], {}, 0.0
    p, info = structured_refit(cand, old["params"], train, val, keys, scales, rollout_k=k)
    v = nrmse(cand, p, val, keys, scales, k=k)
    return name, objective, old["val_rollout_nrmse"], v, p, info, time.time() - t0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run_dirs", nargs="+", required=True)
    ap.add_argument("--objectives", default="multi_step")
    ap.add_argument("--max_parallel", type=int, default=32)
    args = ap.parse_args()
    specs = []
    for rd in args.run_dirs:
        fam = json.loads((Path(rd) / "family.json").read_text())
        for obj in args.objectives.split(","):
            for c in fam["candidates"]:
                if c["name"] in fam["fits"].get(obj, {}) and not c["name"].startswith("sysid_"):
                    specs.append((rd, c["name"], obj))
    results = {}
    with ProcessPoolExecutor(max_workers=args.max_parallel) as ex:
        for spec, res in zip(specs, ex.map(_job, specs)):
            results.setdefault(spec[0], []).append(res)
    for rd, rows in results.items():
        fam, train, val = _load(Path(rd))
        keys = state_keys(train)
        out = {"objectives": {}}
        for obj in args.objectives.split(","):
            sel = [r for r in rows if r[1] == obj]
            table = {r[0]: {"val5_standard": r[2], "val5_structured": r[3], "kept": "structured" if r[3] < r[2] else "standard",
                            "params": r[4] if r[3] < r[2] else fam["fits"][obj][r[0]]["params"], "info": r[5], "elapsed_s": r[6]} for r in sel}
            for n, f in fam["fits"].get(obj, {}).items():
                if n not in table:
                    table[n] = {"val5_standard": f["val_rollout_nrmse"], "val5_structured": f["val_rollout_nrmse"], "kept": "standard", "params": f["params"]}
            best = lambda key: min(table, key=lambda n: table[n][key] if math.isfinite(table[n][key]) else math.inf)
            w_old = fam["selections"][f"offline_{obj}"]["winner"]
            w_new = min(table, key=lambda n: min(table[n]["val5_standard"], table[n]["val5_structured"]))
            out["objectives"][obj] = {"winner_standard": w_old, "winner_structured": w_new,
                                      "val5_standard": table[w_old]["val5_standard"],
                                      "val5_structured": min(table[w_new]["val5_standard"], table[w_new]["val5_structured"]),
                                      "candidates": table}
            metas = [t.meta for ep in train for t in ep.transitions]
            static_meta = {k: v for k, v in (metas[0] if metas else {}).items() if all(m.get(k) == v for m in metas)}
            code = next(c["code"] for c in fam["candidates"] if c["name"] == w_new)
            model = render_model(compile_candidate(code), table[w_new]["params"], env=fam["env"],
                                 action_dim=schema_summary(train)["action_dim"], state_keys=keys, objective=obj, static_meta=static_meta)
            (Path(rd) / "models" / f"structured_{obj}.py").write_text(model, encoding="utf-8")
            o = out["objectives"][obj]
            print(f"[structured] {Path(rd).name} {obj}: {w_old} {o['val5_standard']:.4f} -> {w_new} {o['val5_structured']:.4f}", flush=True)
        (Path(rd) / "structured.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
