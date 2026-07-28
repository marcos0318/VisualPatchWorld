#!/usr/bin/env bash
# Reacher improvement ladder: waits for other planning jobs, then runs sequentially.
# Frame MPC (h=1,r=1,frameskip=1) + finger_match + joint model; sim warm-start / MPPI.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PY:-python3}"
export STABLEWM_HOME="${STABLEWM_HOME:-${HOME}/.stable-wm}"
export MUJOCO_GL=egl
export PYTHONPATH="${ROOT}/src:${ROOT}:${PYTHONPATH:-}"

REPORT="${ROOT}/artifacts/visualpatchworld/reports"
LOG="${ROOT}/logs/paper_experiments"
JOINT="${ROOT}/artifacts/visualpatchworld/lewm_reacher_joint_results"
EVAL="${ROOT}/scripts/visualpatchworld/evaluate_cross_env_planning.py"
NUM_EVAL="${NUM_EVAL:-50}"
NUM_SAMPLES="${NUM_SAMPLES:-300}"
CEM_ITERS="${CEM_ITERS:-10}"
WAIT_FOR_FAIR="${WAIT_FOR_FAIR:-1}"

# Frame MPC (PushT-style replan every control step)
FS=1
H=1
R=1
METRIC=finger_match

mkdir -p "$REPORT" "$LOG"
log() { echo "[reacher-queue][$(date +%H:%M:%S)] $*" | tee -a "$LOG/reacher_improve_queue_master.log"; }

wait_for_jobs() {
  if [[ "$WAIT_FOR_FAIR" == "1" ]]; then
    log "waiting for run_sim_cem_fair_missing.sh to finish..."
    while pgrep -f "run_sim_cem_fair_missing.sh" >/dev/null 2>&1; do sleep 30; done
  fi
  log "waiting for any evaluate_cross_env_planning.py to finish..."
  while pgrep -f "evaluate_cross_env_planning.py" >/dev/null 2>&1; do sleep 20; done
  log "GPU/CPU clear — starting Reacher ladder"
}

run_eval() {
  local tag="$1"; shift
  local out="$REPORT/reacher_improve_${tag}_n${NUM_EVAL}.json"
  if [[ -f "$out" && "${FORCE:-0}" != "1" ]]; then
    log "skip $tag (exists)"
    return 0
  fi
  log "run $tag -> $out"
  "$PY" "$EVAL" --env reacher --results_dir "$JOINT" \
    --num_eval "$NUM_EVAL" --num_samples "$NUM_SAMPLES" --cem_iters "$CEM_ITERS" \
    --seed 42 --success_metric "$METRIC" \
    --frameskip "$FS" --horizon "$H" --receding_horizon "$R" \
    "$@" --output "$out" \
    2>&1 | tee "$LOG/reacher_improve_${tag}.log"
}

wait_for_jobs

log "=== Reacher improvement ladder (frame MPC + finger_match) ==="

# Induced planning
run_eval "oracle_induced_frame" --observation_mode oracle
run_eval "tool_induced_frame" --observation_mode tool_live
run_eval "oracle_induced_frame_warm" --observation_mode oracle --warm_start
run_eval "tool_induced_frame_warm" --observation_mode tool_live --warm_start

# Sim ceilings with warm-start
run_eval "sim_cem_frame_warm" --planner sim_cem --warm_start --ladder_tag improve_frame_warm
run_eval "sim_mppi_frame_warm" --planner sim_mppi --warm_start --ladder_tag improve_frame_mppi

# Sim-verify with aligned finger metric (oracle graphs)
run_eval "oracle_simverify_frame" --observation_mode oracle --sim_verify_frac 0.3
run_eval "tool_simverify_frame" --observation_mode tool_live --sim_verify_frac 0.3

# Macro baseline for comparison (already have macro numbers; optional re-run)
run_eval "sim_cem_finger_frame_nowarm" --planner sim_cem --ladder_tag improve_frame_nowarm

log "=== summary ==="
"$PY" - <<PY
import json
from pathlib import Path
root = Path("${REPORT}")
for p in sorted(root.glob("reacher_improve_*_n${NUM_EVAL}.json")):
    d = json.loads(p.read_text())
    print(
        p.name,
        "succ=", d.get("success_rate"),
        "fs=", d.get("frameskip"),
        "h=", d.get("horizon"),
        "r=", d.get("receding_horizon"),
        "obs=", d.get("observation_mode", "-"),
        "planner=", d.get("planner", "induced"),
        "warm=", d.get("warm_start"),
        "sv=", d.get("sim_verify_frac", 0),
        "tool_fail=", d.get("tool_extraction_failures", "-"),
    )
PY
log "reacher_improve_queue complete"
