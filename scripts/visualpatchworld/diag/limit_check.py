"""Functional state-limit check of rendered programs: python limit_check.py ENV DIR [DIR ...]"""
import re, sys
from pathlib import Path
from visualpatchworld.llm_family.traces import load_episodes
from visualpatchworld.llm_family.propose import compile_candidate
from visualpatchworld.llm_family.sysid import identify_state_limits, limit_violation

env = sys.argv[1]
tr = load_episodes(Path.home() / f"le-wm-code/artifacts/visualpatchworld/data/{env}_oracle_traj_train.jsonl")
lim = identify_state_limits(tr)
for t in (t for ep in tr for t in ep.transitions):
    t.meta["state_limits"] = lim
for d in sys.argv[2:]:
    p = Path("artifacts/visualpatchworld/llm_family") / d / "models/offline_multi_step.py"
    if not p.is_file():
        continue
    txt = p.read_text()
    c = compile_candidate(eval(re.search(r"^CANDIDATE_SOURCE = (.*)$", txt, re.M).group(1)))
    params = eval(re.search(r"^FITTED_PARAMS = (.*)$", txt, re.M).group(1))
    print(f"{d:34s} limit violation rate {limit_violation(c, params, tr, lim):.2f}", flush=True)
