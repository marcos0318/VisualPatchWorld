import json, math, sys, numpy as np
sys.argv=[sys.argv[0]]+sys.argv[1:]
exec(open("/tmp/diag_other.py").read().split("env = sys.argv[1]")[0])
env=sys.argv[1]; obj=TASK[env]
val=load_episodes(D/f"{env}_oracle_traj_val.jsonl")
out={}
for d in sys.argv[2:]:
    path=F/d/"models/offline_multi_step.py"
    if not path.is_file(): continue
    c,p=load_prog(path); cs=[]; rel=[]
    ks=[k for k in val[0].transitions[0].state if k.startswith(obj+".")]
    for ep in val:
        for t in ep.transitions:
            pr=_safe_step(c,dict(t.state),t.action,dict(p),t.meta)
            if pr is None: cs.append(-1); continue
            dp=np.array([pr.get(k,t.state[k])-t.state[k] for k in ks]); dt=np.array([t.next_state[k]-t.state[k] for k in ks])
            if np.linalg.norm(dt)<1e-6: continue
            cs.append(float(dp@dt/(np.linalg.norm(dp)*np.linalg.norm(dt)+1e-12))); rel.append(np.linalg.norm(dp)/np.linalg.norm(dt))
    out[d]={"cos":round(float(np.mean(cs)),4),"mag_ratio":round(float(np.median(rel)),3)}
    print(d,out[d],flush=True)
json.dump(out,open(f"/tmp/diag_dir_{env}.json","w"))
