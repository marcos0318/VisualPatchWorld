#!/usr/bin/env python
"""Plan every distinct selected model of finished LLM-family runs, with bounded parallelism.

Selections that picked the same program with the same fitted constants render to
identical files; those are evaluated once and the result is shared.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FAMILY_ROOT = ROOT / "artifacts/visualpatchworld/llm_family"
SELECTIONS = ("offline_one_step", "offline_multi_step", "active_one_step", "active_multi_step")


def _digest(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()[:12]


def collect(runs, seeds):
    jobs = []
    for run in runs:
        d = FAMILY_ROOT / run
        if not (d / "family.json").is_file():
            print(f"[queue] skip unfinished {run}", flush=True)
            continue
        env = json.loads((d / "family.json").read_text())["env"]
        seen = {}
        for sel in SELECTIONS:
            m = d / "models" / f"{sel}.py"
            if not m.is_file():
                continue
            h = _digest(m)
            seen.setdefault(h, []).append(sel)
        for h, sels in seen.items():
            for seed in seeds:
                out = d / "eval" / f"{sels[0]}_seed{seed}.json"
                jobs.append({"run": run, "env": env, "model": d / "models" / f"{sels[0]}.py", "aliases": sels, "seed": seed, "out": out})
    return jobs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--seeds", default="42")
    p.add_argument("--max_parallel", type=int, default=8)
    args = p.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    jobs = [j for j in collect(args.runs, seeds) if not j["out"].is_file()]
    print(f"[queue] {len(jobs)} evaluations", flush=True)
    env = {**os.environ, "PYTHONPATH": f"{ROOT / 'src'}:{ROOT}", "MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl", "OMP_NUM_THREADS": "2"}
    running = []
    while jobs or running:
        while jobs and len(running) < args.max_parallel:
            j = jobs.pop(0)
            j["out"].parent.mkdir(parents=True, exist_ok=True)
            log = j["out"].with_suffix(".log")
            cmd = [sys.executable, "-W", "ignore", str(ROOT / "scripts/visualpatchworld/eval_llm_family_model.py"),
                   "--env", j["env"], "--models", str(j["model"]), "--seeds", str(j["seed"]), "--out", str(j["out"])]
            proc = subprocess.Popen(cmd, stdout=log.open("w"), stderr=subprocess.STDOUT, env=env, cwd=ROOT)
            running.append((proc, j))
            print(f"[queue] start {j['run']} {j['aliases']} seed={j['seed']}", flush=True)
        time.sleep(10)
        for proc, j in list(running):
            if proc.poll() is not None:
                running.remove((proc, j))
                res = json.loads(j["out"].read_text()) if j["out"].is_file() else {}
                sr = next(iter(res.values()), {}).get("success_rate") if res else None
                (j["out"].parent / "aliases.json").write_text(json.dumps({**(json.loads((j["out"].parent / "aliases.json").read_text()) if (j["out"].parent / "aliases.json").is_file() else {}), j["out"].name: j["aliases"]}, indent=2))
                print(f"[queue] done {j['run']} {j['aliases']} seed={j['seed']} rc={proc.returncode} success={sr}", flush=True)


if __name__ == "__main__":
    main()
