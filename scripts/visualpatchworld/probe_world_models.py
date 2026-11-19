#!/usr/bin/env python
"""Probe every baseline and selected program with one protocol, outside the planner.

Each program is loaded with the evaluator's loader and stepped the way that domain's
planner steps it: PushT rolls out through the program's own text readout
(``rollout_model_state``), the other domains carry the predicted belief forward
(``_rollout_model_row``). Actions are encoded with the planners' encoders. Over held-out
5-step windows with the recorded actions we report

* crash: fraction of windows in which the program raises;
* nrmse5: 5-step normalized RMSE with the same keys, scales and clamp as offline
  selection (``llm_family.fit.nrmse``); a crash scores as the clamp value;
* action_sensitive: fraction of first steps whose prediction changes when the action is
  negated or replaced by zero;
* block_moves (PushT): fraction of first steps with a recorded block displacement in
  which the program moves the block at all.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

ART = ROOT / "artifacts/visualpatchworld"
REPORTS = ART / "reports/fair_wm_ranking_gt90"
DATA = Path("/home/comp/baijiaxin/le-wm-code/artifacts/visualpatchworld/data")
ENVS = ("tworoom", "reacher", "cube", "pusht")
METHODS = ("worldcoder", "gif_mcts", "poeworld", "pomdp_coder", "cwm_game", "tracepatch")
LLMS = {"coder": "all_methods", "qwen35": "qwen35_baselines", "gptoss": "gptoss_baselines"}
OURS = {"ours": "v1_{env}_{llm}_s0", "ours_bank": "bank_general_{env}_{llm}_s0"}
FIELDS = ("x", "y", "z", "angle", "vx", "vy")
BAD = 1e3
K = 5


def held_out_file(env: str) -> Path:
    return DATA / (f"{env}_oracle_traj_test.jsonl" if env == "pusht" else f"{env}_oracle_traj_val.jsonl")


def windows(env: str, max_windows: int) -> List[List[Mapping[str, Any]]]:
    out = []
    with held_out_file(env).open(encoding="utf-8") as fh:
        for line in fh:
            ts = json.loads(line).get("transitions", [])
            for s in range(0, max(0, len(ts) - K + 1)):
                out.append(ts[s : s + K])
    if len(out) > max_windows:
        idx = np.linspace(0, len(out) - 1, max_windows).round().astype(int)
        out = [out[i] for i in idx]
    return out


def belief_state(belief: Any) -> Dict[str, float]:
    state: Dict[str, float] = {}
    objects = belief.get("objects", {}) if isinstance(belief, Mapping) else {}
    items = objects.items() if isinstance(objects, Mapping) else (
        (o.get("id"), o) for o in objects if isinstance(o, Mapping))
    for oid, obj in items:
        if not isinstance(obj, Mapping):
            continue
        flat = dict(obj)
        for nest, pre in (("position", ""), ("velocity", "v")):
            if isinstance(obj.get(nest), Mapping):
                for k, v in obj[nest].items():
                    flat.setdefault(pre + k, v)
        if "orientation" in flat and "angle" not in flat:
            flat["angle"] = flat["orientation"]
        for f in FIELDS:
            try:
                v = float(flat[f])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(v):
                state[f"{oid}.{f}"] = v
    return state


def probe(job: Dict[str, Any]) -> Dict[str, Any]:
    from visualpatchworld.llm_family.traces import parse_graph_text
    from visualpatchworld.parallel_planning_eval import load_induced_model
    from visualpatchworld.sim_eval import _predict_next_graph

    env, keys, scales = job["env"], job["keys"], job["scales"]
    out = dict(job["meta"])
    try:
        model = load_induced_model(Path(job["model_path"]))
    except Exception as exc:  # noqa: BLE001
        out.update(loadable=False, error=repr(exc)[:200])
        return out
    if env == "pusht":
        from visualpatchworld.planning_eval import action_values_to_strings

        def encode(v):
            return action_values_to_strings(np.asarray([v], dtype=np.float64))[0]

        def init(obs_text):
            return obs_text

        def step(obs_text, action):
            graph = _predict_next_graph(model, obs_text, action)
            return graph, model.readout_observation(graph, action)
    else:
        from visualpatchworld.general_planning_eval import get_env_spec
        from visualpatchworld.lewm_graph import action_string_from_vector

        spec = get_env_spec(env)

        def encode(v):
            return action_string_from_vector(np.asarray(v, dtype=np.float64), spec)

        def init(obs_text):
            return model.init_belief_from_observation(obs_text)

        def step(belief, action):
            predicted = model.predict_belief(belief, action)
            if isinstance(predicted, Mapping) and predicted.get("objects"):
                return predicted, predicted
            parsed = model.parse_observation(model.readout_observation(predicted, action))
            nxt = parsed if isinstance(parsed, Mapping) and parsed.get("objects") else predicted
            return nxt, nxt

    wins = windows(env, job["max_windows"])
    res: List[float] = []
    crashes = sensitive = moved_truth = moved_pred = 0
    for win in wins:
        state = parse_graph_text(win[0]["observation"])["state"]
        try:
            carrier = init(win[0]["observation"])
            for i, t in enumerate(win):
                values = np.asarray(t["metadata"]["action_values"], dtype=np.float64)
                if i == 0:
                    alts = []
                    for alt in (-values, 0.0 * values):
                        try:
                            alts.append(belief_state(step(copy.deepcopy(carrier), encode(alt))[0]))
                        except Exception:  # noqa: BLE001
                            pass
                graph, carrier = step(carrier, encode(values))
                pred = belief_state(graph)
                if i == 0:
                    sensitive += any(abs(a.get(k, 0.0) - pred.get(k, 0.0)) > 1e-3 * scales[k] for a in alts for k in keys)
                    if env == "pusht":
                        nxt = parse_graph_text(t["next_observation"])["state"]
                        bkeys = [k for k in keys if k.startswith("block.")]
                        if any(abs(nxt.get(k, 0) - state.get(k, 0)) > 0.5 * scales[k] for k in bkeys):
                            moved_truth += 1
                            moved_pred += any(abs(pred.get(k, state.get(k, 0)) - state.get(k, 0)) > 1e-6 for k in bkeys)
                truth = parse_graph_text(t["next_observation"])["state"]
                for k in keys:
                    if k in truth:
                        r = (pred.get(k, state.get(k, 0.0)) - truth[k]) / scales[k]
                        res.append(r if math.isfinite(r) else BAD)
                state = {k: pred.get(k, state.get(k, 0.0)) for k in state}
        except Exception:  # noqa: BLE001
            crashes += 1
            res.extend([BAD] * len(keys))
    r = np.asarray(res, dtype=np.float64)
    out.update(
        loadable=True,
        windows=len(wins),
        crash=crashes / max(1, len(wins)),
        nrmse5=float(np.sqrt(np.mean(np.minimum(r * r, BAD * BAD)))) if r.size else None,
        action_sensitive=sensitive / max(1, len(wins)),
        block_moves=(moved_pred / moved_truth) if env == "pusht" and moved_truth else None,
    )
    return out


def jobs(max_windows: int) -> List[Dict[str, Any]]:
    out = []
    for env in ENVS:
        fam = json.loads((ART / f"llm_family/v1_{env}_qwen35_s0/family.json").read_text())
        keys, scales = list(fam["schema"]["state_keys"]), fam["scales"]
        for llm, rep in LLMS.items():
            for m in METHODS:
                f = REPORTS / rep / f"{m}_{env}_n50_seed42.json"
                meta = {"env": env, "llm": llm, "method": m}
                if not f.is_file():
                    out.append({"meta": {**meta, "loadable": False, "error": "no loadable final program"}})
                    continue
                path = ROOT / json.loads(f.read_text())["model_path"]
                out.append({"meta": meta, "env": env, "keys": keys, "scales": scales,
                            "model_path": str(path), "max_windows": max_windows})
            for name, tag in OURS.items():
                path = ART / "llm_family" / tag.format(env=env, llm=llm) / "models/offline_multi_step.py"
                if path.is_file():
                    out.append({"meta": {"env": env, "llm": llm, "method": name}, "env": env, "keys": keys,
                                "scales": scales, "model_path": str(path), "max_windows": max_windows})
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max_windows", type=int, default=300)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--out", default=str(ART / "reports/world_model_probe.json"))
    args = p.parse_args()
    todo = jobs(args.max_windows)
    results = [j["meta"] for j in todo if "model_path" not in j]
    with ProcessPoolExecutor(args.workers) as ex:
        results += list(ex.map(probe, [j for j in todo if "model_path" in j]))
    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    for r in sorted(results, key=lambda r: (r["env"], r["llm"], r["method"])):
        print(f"{r['env']:8s} {r['llm']:7s} {r['method']:12s} load={r.get('loadable')!s:5s} "
              f"crash={r.get('crash', float('nan')):.2f} nrmse5={r.get('nrmse5') or float('nan'):8.3f} "
              f"act={r.get('action_sensitive', float('nan')):.2f} block={r.get('block_moves')}")


if __name__ == "__main__":
    main()
