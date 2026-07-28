#!/usr/bin/env bash
# Re-run macro Reacher rows after tool-observation bugfix (correct failure counts + sim fallback).
# Waits for reacher_improve_queue to finish, then reruns key Table-1 configs.
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
FORCE=1

mkdir -p "$REPORT" "$LOG"
log() { echo "[reacher-rerun-fixed][$(date +%H:%M:%S)] $*" | tee -a "$LOG/reacher_rerun_macro_fixed.log"; }

while pgrep -f "run_reacher_improve_queue.sh" >/dev/null 2>&1; do sleep 30; done
while pgrep -f "evaluate_cross_env_planning.py" >/dev/null 2>&1; do sleep 20; done

run() {
  local out="$1"; shift
  log "run -> $out"
  "$PY" "$EVAL" --env reacher --results_dir "$JOINT" --num_eval "$NUM_EVAL" \
    --num_samples 300 --cem_iters 10 --seed 42 --success_metric finger_match \
    "$@" --output "$out" 2>&1 | tee -a "$LOG/reacher_rerun_macro_fixed.log"
}

log "=== macro finger_match reruns (bugfix) ==="
run "$REPORT/reacher_joint_tool_live_finger_n50.json" --observation_mode tool_live
run "$REPORT/reacher_joint_tool_hybrid_finger_n50.json" --observation_mode tool_live --sim_verify_frac 0.3
run "$REPORT/reacher_joint_oracle_hybrid_finger_n50.json" --observation_mode oracle --sim_verify_frac 0.3

log "done"
