import itertools, json, sys
from pathlib import Path
from visualpatchworld.llm_family.traces import load_episodes, state_keys, state_scales
from visualpatchworld.llm_family.propose import compile_candidate
from visualpatchworld.llm_family.fit import nrmse
from visualpatchworld.llm_family.render import render_model

SRC = '''# HYPOTHESIS: identified_effector_rigid_grasp
# MECHANISM: Effector moves linearly with the action (identified gains); the block rigidly follows the effector while grasped (gripper command above a threshold and effector at the block, with hysteresis once lifted); a released block drops to the table.
PARAMS = {"gx": 0.11, "gy": 0.109, "gz": 0.098, "d0": 0.02, "g_on": 0.3, "g_hold": 0.1, "z_floor": 0.02, "z_lift": 0.03, "e_zmin": 0.015}

def step(state, action, p, meta):
    import math
    s = dict(state)
    E = [state["effector.x"], state["effector.y"], state["effector.z"]]
    B = [state["block.x"], state["block.y"], state["block.z"]]
    a = [max(-1.0, min(1.0, v)) for v in action]
    En = [E[0] + p["gx"] * a[0], E[1] + p["gy"] * a[1], max(p["e_zmin"], E[2] + p["gz"] * a[2])]
    d = math.sqrt(sum((E[i] - B[i]) ** 2 for i in range(3)))
    grip = a[4] if len(a) > 4 else 0.0
    grasped = d < p["d0"] and (grip > p["g_on"] or (B[2] > p["z_lift"] and grip > p["g_hold"]))
    if grasped:
        Bn = [En[i] + (B[i] - E[i]) for i in range(3)]
        Bn[2] = max(p["z_floor"], Bn[2])
    else:
        Bn = [B[0], B[1], p["z_floor"] if B[2] > p["z_floor"] else B[2]]
    for i, c in enumerate("xyz"):
        s["effector." + c] = En[i]
        s["block." + c] = Bn[i]
    return s
'''

D = Path.home() / "le-wm-code/artifacts/visualpatchworld/data"
tr = load_episodes(D / "cube_oracle_traj_train.jsonl")
val = load_episodes(D / "cube_oracle_traj_val.jsonl")
keys = state_keys(tr)
scales = state_scales(tr, keys)
c = compile_candidate(SRC)
best = None
for d0, gon, ghold in itertools.product((0.01, 0.015, 0.02, 0.03, 0.05), (0.2, 0.3, 0.35, 0.4), (-0.1, 0.0, 0.1, 0.2, 0.3)):
    p = dict(c.params, d0=d0, g_on=gon, g_hold=ghold)
    e = nrmse(c, p, tr[:30], keys, scales, k=5)
    if best is None or e < best[0]:
        best = (e, p)
p = best[1]
print("train5", round(best[0], 4), "val1", round(nrmse(c, p, val, keys, scales, k=1), 4), "val5", round(nrmse(c, p, val, keys, scales, k=5), 4),
      {k: p[k] for k in ("d0", "g_on", "g_hold")})
out = Path("artifacts/visualpatchworld/llm_family/ident_cube_ref_s0/models")
out.mkdir(parents=True, exist_ok=True)
(out / "offline_multi_step.py").write_text(render_model(c, p, env="cube", action_dim=5, state_keys=keys, objective="multi_step"))
