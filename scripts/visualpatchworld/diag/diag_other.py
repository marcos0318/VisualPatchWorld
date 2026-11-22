import json, math, re, sys, collections, numpy as np
from pathlib import Path
from visualpatchworld.llm_family.traces import load_episodes, state_keys, state_scales
from visualpatchworld.llm_family.propose import compile_candidate
from visualpatchworld.llm_family.fit import nrmse, _safe_step

F = Path("artifacts/visualpatchworld/llm_family")
D = Path.home() / "le-wm-code/artifacts/visualpatchworld/data"
TASK = {"tworoom": "agent", "reacher": "finger", "cube": "block"}
ACT = {"tworoom": None, "reacher": ["arm.x", "arm.y"], "cube": ["effector.x", "effector.y", "effector.z"]}


def labels(env, ep, gain):
    out = []
    for t in ep.transitions:
        s, n = t.state, t.next_state
        if env == "tworoom":
            wx = json.loads(t.meta["door_coords"])[0] if isinstance(t.meta.get("door_coords"), str) else 112.0
            d = math.hypot(n["agent.x"] - s["agent.x"], n["agent.y"] - s["agent.y"])
            a = math.hypot(*t.action[:2])
            if (s["agent.x"] - wx) * (n["agent.x"] - wx) < 0:
                out.append("crossing")
            elif a > 0.2 and d < 0.6 * gain * a:
                out.append("blocked")
            else:
                out.append("free")
        elif env == "cube":
            mv = math.dist([s[f"block.{c}"] for c in "xyz"], [n[f"block.{c}"] for c in "xyz"])
            out.append("free" if mv < 2e-3 else ("lifted" if n["block.z"] > 0.03 else "block_pushed"))
        else:
            out.append("free")
    return out


def tworoom_gain(eps):
    r = [math.hypot(t.next_state["agent.x"] - t.state["agent.x"], t.next_state["agent.y"] - t.state["agent.y"])
         / math.hypot(*t.action[:2]) for e in eps for t in e.transitions if math.hypot(*t.action[:2]) > 0.2]
    return float(np.median(r))


def load_prog(path):
    txt = path.read_text()
    ns = {}
    for name in ("CANDIDATE_SOURCE", "FITTED_PARAMS"):
        m = re.search(rf"^{name} = (.*)$", txt, re.M)
        ns[name] = eval(m.group(1), {"true": True, "false": False, "null": None})
    c = compile_candidate(ns["CANDIDATE_SOURCE"])
    return c, ns["FITTED_PARAMS"]


def fudge(init, fit):
    n = 0
    for k, f in fit.items():
        i = init.get(k)
        if i is None or i == 0:
            continue
        if f * i < 0 or abs(f) < 0.02 * abs(i) or abs(f) > 10 * abs(i) or abs(f) < abs(i) / 10:
            n += 1
    return n, len(fit)


def rollout_err(c, p, eps, keys, scales, obj, h, force=None, modes=None):
    acc = collections.defaultdict(list)
    for ep in eps:
        ts, lab = ep.transitions, modes[id(ep)] if modes else None
        for s0 in range(0, len(ts) - h + 1):
            st = dict(ts[s0].state)
            ok = True
            for j in range(h):
                pr = _safe_step(c, st, ts[s0 + j].action, dict(p), ts[s0 + j].meta)
                if pr is None:
                    ok = False
                    break
                st = {k: float(pr.get(k, st[k])) for k in st}
                if force:
                    for k in force:
                        st[k] = ts[s0 + j].next_state[k]
            tn = ts[s0 + h - 1].next_state
            ks = [k for k in keys if k.startswith(obj + ".")]
            e = float(np.sqrt(np.mean([((st[k] - tn[k]) / scales[k]) ** 2 for k in ks]))) if ok else 10.0
            e = min(e, 10.0) if math.isfinite(e) else 10.0
            m = lab[s0] if lab else "all"
            acc[m].append(e)
            acc["ALL"].append(e)
    return {m: (round(float(np.mean(v)), 4), len(v)) for m, v in acc.items()}


env = sys.argv[1]
tr = load_episodes(D / f"{env}_oracle_traj_train.jsonl")
val = load_episodes(D / f"{env}_oracle_traj_val.jsonl")
keys = state_keys(tr)
scales = state_scales(tr, keys)
gain = tworoom_gain(tr) if env == "tworoom" else 0
modes_val = {id(e): labels(env, e, gain) for e in val}
cnt_all = collections.Counter(l for e in tr for l in labels(env, e, gain))
cnt_40 = collections.Counter(l for e in tr[:40] for l in labels(env, e, gain))
cnt_val = collections.Counter(l for e in val for l in modes_val[id(e)])
print(f"== {env}: train modes {dict(cnt_all)}; first-40 {dict(cnt_40)}; val {dict(cnt_val)}  free gain {gain:.2f}")
rows = {}
for d in sys.argv[2:]:
    path = F / d / "models/offline_multi_step.py"
    if not path.is_file():
        continue
    try:
        c, p = load_prog(path)
    except Exception as ex:
        print(d, "load fail", ex)
        continue
    r = {"name": c.name[:40]}
    r["val1"] = round(nrmse(c, p, val, keys, scales, k=1), 4)
    r["val5"] = round(nrmse(c, p, val, keys, scales, k=5), 4)
    obj = TASK[env]
    r["task_h1"] = rollout_err(c, p, val, keys, scales, obj, 1, modes=modes_val)
    r["task_h5"] = rollout_err(c, p, val, keys, scales, obj, 5, modes=modes_val)
    if ACT[env]:
        r["task_h5_actforced"] = rollout_err(c, p, val, keys, scales, obj, 5, force=ACT[env])["ALL"][0]
        aobj = ACT[env][0].split(".")[0]
        r["actuator_h5"] = rollout_err(c, p, val, keys, scales, aobj, 5)["ALL"][0]
    r["fudge"] = fudge(c.params, p)
    rows[d] = r
    print(d, json.dumps(r), flush=True)
Path(f"/tmp/diag_other_{env}.json").write_text(json.dumps({"modes": {"train": cnt_all, "first40": cnt_40, "val": cnt_val},
                                                            "rows": rows}, indent=1))
