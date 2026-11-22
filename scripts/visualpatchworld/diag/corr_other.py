import json, glob, re, collections, numpy as np
from scipy.stats import spearmanr
V = "/home/comp/baijiaxin/VisualPatchWorld-marcos0318/artifacts/visualpatchworld/reports/fair_wm_ranking_gt90/vpw_modes/"
for env in ("tworoom", "reacher", "cube"):
    rows = {}
    for line in open(f"/tmp/diag_other_{env}.log"):
        m = re.match(r"^(\S+) (\{.*\})$", line.strip())
        if m:
            rows[m[1]] = json.loads(m[2])
    for line in open("/tmp/diag_other_cube.log") if False else []:
        pass
    try:
        dirm = json.load(open(f"/tmp/diag_dir_{env}.json"))
    except FileNotFoundError:
        dirm = {}
    sr = {}
    for d in rows:
        setname = re.sub(rf"_{env}_", "_", d) if d.startswith("refine_v2") else d
        v = [json.load(open(f))["success_rate"] for f in glob.glob(V + f"{setname}_oracle_induced_{env}_n50_seed*.json")]
        v = [x for x in v if x is not None]
        if len(v) == 5:
            sr[d] = float(np.mean(v))
    ds = sorted(sr, key=lambda d: -sr[d])
    print(f"\n== {env} ({len(ds)} programs planned)")
    print(f"{'program':34s} {'SR':>5s} {'val1':>6s} {'val5':>6s} {'task1':>6s} {'task5':>6s} {'cos':>5s} {'mag':>5s} mode_task1")
    for d in ds:
        r = rows[d]
        modes = {k: v[0] for k, v in r["task_h1"].items() if k not in ("ALL",)}
        print(f"{d:34s} {sr[d]:5.1f} {r['val1']:6.3f} {r['val5']:6.3f} {r['task_h1']['ALL'][0]:6.3f} {r['task_h5']['ALL'][0]:6.3f} "
              f"{dirm.get(d, {}).get('cos', float('nan')):5.2f} {dirm.get(d, {}).get('mag_ratio', float('nan')):5.2f} {modes}")
    if len(ds) >= 5:
        y = [sr[d] for d in ds]
        mets = {"val1": lambda r, d: r["val1"], "val5": lambda r, d: r["val5"], "task_h1": lambda r, d: r["task_h1"]["ALL"][0],
                "task_h5": lambda r, d: r["task_h5"]["ALL"][0], "-cos": lambda r, d: -dirm.get(d, {}).get("cos", 0)}
        if env == "cube":
            mets["lifted_h1"] = lambda r, d: r["task_h1"]["lifted"][0]
        good = [d for d in ds if rows[d]["val5"] < 1.2] if env == "reacher" else [d for d in ds if rows[d]["val5"] < 0.8]
        print(f"Spearman within the {len(good)} better programs:", {k: round(spearmanr([f(rows[d], d) for d in good], [sr[d] for d in good]).correlation, 2) for k, f in mets.items()})
        print("Spearman(metric, SR):", {k: round(spearmanr([f(rows[d], d) for d in ds], y).correlation, 2) for k, f in mets.items()})
