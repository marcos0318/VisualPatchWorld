#!/usr/bin/env python
"""Five-seed replan-observation x scoring matrix for pinned world-model programs.

For each model set (hand-designed families, LLM-proposed families) and each
environment, runs Tool+Induced, Oracle+Hybrid and Tool+Hybrid under the frozen
GT>=90 planners, plus the MuJoCo+CEM ceiling. Oracle+Induced cells come from the
main-table result files. Skips cells whose JSON already exists unless --force.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
for path in (str(ROOT), str(ROOT / "src"), str(SCRIPT_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("STABLEWM_HOME", str(Path.home() / ".stable-wm"))

ART = ROOT / "artifacts/visualpatchworld"
OUT = ART / "reports/fair_wm_ranking_gt90/vpw_modes"
ENVS = ("tworoom", "reacher", "cube", "pusht")
SEEDS = (42, 43, 44, 45, 46)

MODEL_SETS: Dict[str, Dict[str, Path]] = {
    "hand": {
        "tworoom": ART / "lewm_tworoom_program_results/generated_models/tworoom_patchworld_20260722_165027.py",
        "reacher": ART / "lewm_reacher_program_results/generated_models/reacher_patchworld_20260722_165027.py",
        "cube": ART / "lewm_cube_program_results/generated_models/cube_patchworld_20260722_165027.py",
        "pusht": ART / "lewm_pusht_program_results/generated_models/pusht_patchworld_program_induced_best.py",
    },
    "qwen35": {e: ART / f"llm_family/v1_{e}_qwen35_s0/models/offline_multi_step.py" for e in ENVS},
    "null": {e: ART / f"null_models/{e}_null.py" for e in ENVS},
    "qwen35_tooltrace": {e: ART / f"llm_family/tooltrace_{e}_qwen35_s0/models/offline_multi_step.py" for e in ENVS},
    "contact": {"pusht": ART / "contact_models/pusht_contact_textbook.py"},
    "sysid": {"pusht": ART / "sysid_models/pusht_sysid.py"},
    **{f"{tag}_qwen35_s{s}": {e: ART / f"llm_family/{tag}_{e}_qwen35_s{s}/models/offline_multi_step.py" for e in ENVS}
       for tag in ("v2", "refine_v2") for s in (0, 1, 2)},
    **{f"contact_{a}": {"pusht": ART / f"contact_models/pusht_contact_{a}.py"}
       for a in ("one_substep", "no_rotation", "pivot_at_origin", "square_block")},
    **{f"{tag}_{m}": {"pusht": ART / f"llm_family/{tag}_pusht_{m}/models/offline_multi_step.py"}
       for tag in ("geom", "geombank", "geomkit", "kit", "refine_geomkit", "refine_geom", "refine_v1",
                   "sysidskill", "sysidlib", "refine_sysidskill", "refine_sysidlib", "sysidfam",
                   "refine_modes_v2", "refine_modes_sysidskill")
       for m in ("qwen35_s0", "qwen35_s1", "qwen35_s2", "coder_s0", "gptoss_s0")},
}
MODES = {
    "oracle_induced": ("oracle", 0.0),
    "tool_induced": ("tool_live", 0.0),
    "oracle_hybrid": ("oracle", 0.3),
    "tool_hybrid": ("tool_live", 0.3),
    # Simulator scores a uniformly random 30% of candidates; only those are eligible.
    "oracle_hybrid_random": ("oracle", 0.3),
}


def _job(payload: Dict[str, Any]) -> Dict[str, Any]:
    import run_fair_baseline_replan_gt90 as fr
    from run_fair_multiseed_selected import _run_mujoco
    from run_fair_wm_ranking_gt90 import PROTOCOLS

    out_path = Path(payload["out"])
    env, seed, mode = payload["env"], int(payload["seed"]), payload["mode"]
    t0 = time.time()
    os.environ["VPW_SIM_VERIFY_ORDER"] = "random" if mode.endswith("_random") else "model"
    try:
        if mode == "mujoco":
            report = _run_mujoco(env, 50, seed, int(payload["workers"]))
        else:
            obs, sv = MODES[mode]
            model = Path(payload["model"])
            proto = PROTOCOLS[env]
            if env == "pusht":
                if obs == "oracle" and sv == 0.0:
                    report = fr._run_pusht_induced(proto=proto, model_path=model, num_eval=50, seed=seed,
                                                   workers=int(payload["workers"]))
                else:
                    report = fr._run_pusht_hybrid(proto=proto, model_path=model, observation_mode=obs,
                                                  sim_verify_frac=sv, num_eval=50, seed=seed)
            else:
                report = fr._run_general(env=env, proto=proto, model_path=model, observation_mode=obs,
                                         sim_verify_frac=sv, num_eval=50, seed=seed)
        report.update({"mode": mode, "model_set": payload["set"], "env": env, "seed": seed, "success": True})
    except Exception as exc:  # noqa: BLE001
        report = {"mode": mode, "model_set": payload["set"], "env": env, "seed": seed, "success": False,
                  "error": f"{type(exc).__name__}: {exc}", "success_rate": None}
    report["wall_s"] = time.time() - t0
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return {"out": out_path.name, "sr": report.get("success_rate"), "err": report.get("error")}


def main() -> None:
    import multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sets", default="hand,qwen35")
    p.add_argument("--modes", default="tool_induced,oracle_hybrid,tool_hybrid,mujoco")
    p.add_argument("--envs", default=",".join(ENVS))
    p.add_argument("--max_parallel", type=int, default=48)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    jobs = []
    for env in args.envs.split(","):
        for mode in args.modes.split(","):
            sets = ["mujoco"] if mode == "mujoco" else args.sets.split(",")
            for s in sets:
                for seed in SEEDS:
                    out = OUT / f"{s}_{mode}_{env}_n50_seed{seed}.json" if mode != "mujoco" else OUT / f"mujoco_{env}_n50_seed{seed}.json"
                    if out.is_file() and not args.force and json.loads(out.read_text()).get("success"):
                        continue
                    model = "" if mode == "mujoco" else str(MODEL_SETS[s][env])
                    jobs.append({"set": s, "mode": mode, "env": env, "seed": seed, "model": model,
                                 "out": str(out), "workers": args.workers})
    # Longest cells first so the pool drains evenly.
    jobs.sort(key=lambda j: (j["env"] != "pusht", j["mode"] != "mujoco"))
    print(f"[vpw-modes] {len(jobs)} cells, parallel={args.max_parallel}", flush=True)
    with ProcessPoolExecutor(max_workers=args.max_parallel) as ex:
        futs = [ex.submit(_job, j) for j in jobs]
        for fut in as_completed(futs):
            r = fut.result()
            print(f"[vpw-modes] {r['out']} sr={r['sr']} err={r['err']}", flush=True)


if __name__ == "__main__":
    main()
