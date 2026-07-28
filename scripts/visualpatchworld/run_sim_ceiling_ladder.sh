#!/usr/bin/env bash
# Sim-ceiling planning ladder (run one step at a time via STEP=1|2|3|4).
#
# Step 1: Fair CEM budget (300x10), matched to induced planning
# Step 2: Fair CEM + MPC warm-start
# Step 3: MPPI (300 samples x 10 iters) + warm-start
# Step 4: Reacher diagnostic — fingertip-to-target metric (all envs use default except reacher)
#
# Examples:
#   STEP=1 bash scripts/visualpatchworld/run_sim_ceiling_ladder.sh
#   STEP=2 ENV=tworoom bash scripts/visualpatchworld/run_sim_ceiling_ladder.sh
#   STEP=4 bash scripts/visualpatchworld/run_sim_ceiling_ladder.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PY:-python3}"
export STABLEWM_HOME="${STABLEWM_HOME:-${HOME}/.stable-wm}"
export MUJOCO_GL=egl
export PYTHONPATH="${ROOT}/src:${ROOT}:${PYTHONPATH:-}"

REPORT="${ROOT}/artifacts/visualpatchworld/reports"
LOG="${ROOT}/logs/paper_experiments"
NUM_EVAL="${NUM_EVAL:-50}"
NUM_SAMPLES="${NUM_SAMPLES:-300}"
CEM_ITERS="${CEM_ITERS:-10}"
SIM_ROLLOUT_BATCH="${SIM_ROLLOUT_BATCH:-50}"
SIM_ENV_RECYCLE_EVERY="${SIM_ENV_RECYCLE_EVERY:-150}"
STEP="${STEP:-1}"
ENV="${ENV:-all}"
FORCE="${FORCE:-0}"

mkdir -p "$REPORT" "$LOG"
log() { echo "[sim-ladder][step${STEP}][$(date +%H:%M:%S)] $*"; }

envs=()
if [[ "$ENV" == "all" ]]; then
  envs=(tworoom cube reacher pusht)
else
  envs=("$ENV")
fi

run_sim() {
  local env="$1"
  local tag="$2"
  local out="$3"
  shift 3
  if [[ -f "$out" && "$FORCE" != "1" ]]; then
    log "skip $env $tag (exists $out)"
    return 0
  fi
  local results_dir="${ROOT}/artifacts/visualpatchworld/lewm_${env}_general_results"
  if [[ "$env" == "pusht" ]]; then
    results_dir="${ROOT}/artifacts/visualpatchworld/induced"
  fi
  log "run $env $tag -> $out"
  local batch="$SIM_ROLLOUT_BATCH"
  local recycle="$SIM_ENV_RECYCLE_EVERY"
  if [[ "$env" == "cube" ]]; then
    batch=30
    recycle=100
  fi
  local mpc_args=()
  if [[ "$env" == "pusht" ]]; then
    mpc_args=(--frameskip 1 --horizon 1 --receding_horizon 1)
  fi
  "$PY" "$ROOT/scripts/visualpatchworld/evaluate_cross_env_planning.py" \
    --env "$env" \
    --results_dir "$results_dir" \
    --num_eval "$NUM_EVAL" \
    --num_samples "$NUM_SAMPLES" \
    --cem_iters "$CEM_ITERS" \
    --seed 42 \
    --ladder_tag "$tag" \
    --sim_rollout_batch "$batch" \
    --sim_env_recycle_every "$recycle" \
    "${mpc_args[@]}" \
    "$@" \
    --output "$out" \
    2>&1 | tee "$LOG/sim_ladder_${tag}_${env}.log"
}

case "$STEP" in
  1)
    log "Step 1: fair sim CEM (300x10, no warm-start)"
    for env in "${envs[@]}"; do
      run_sim "$env" "fair_cem" \
        "$REPORT/cross_env_${env}_sim_cem_fair_n${NUM_EVAL}.json" \
        --planner sim_cem
    done
    ;;
  2)
    log "Step 2: fair sim CEM + MPC warm-start"
    for env in "${envs[@]}"; do
      run_sim "$env" "fair_cem_warm" \
        "$REPORT/cross_env_${env}_sim_cem_warm_n${NUM_EVAL}.json" \
        --planner sim_cem \
        --warm_start
    done
    ;;
  3)
    log "Step 3: MPPI + warm-start"
    for env in "${envs[@]}"; do
      run_sim "$env" "fair_mppi_warm" \
        "$REPORT/cross_env_${env}_sim_mppi_warm_n${NUM_EVAL}.json" \
        --planner sim_mppi \
        --warm_start
    done
    ;;
  4)
    log "Step 4: Reacher fingertip metric diagnostic (sim CEM fair + warm)"
    run_sim reacher "fair_cem_finger" \
      "$REPORT/cross_env_reacher_sim_cem_finger_n${NUM_EVAL}.json" \
      --planner sim_cem \
      --warm_start \
      --success_metric finger_match
    ;;
  *)
    echo "Unknown STEP=$STEP (use 1, 2, 3, or 4)" >&2
    exit 1
    ;;
esac

log "summary (step $STEP)"
"$PY" - <<PY
import json
from pathlib import Path

step = "${STEP}"
patterns = {
    "1": "cross_env_*_sim_cem_fair_n${NUM_EVAL}.json",
    "2": "cross_env_*_sim_cem_warm_n${NUM_EVAL}.json",
    "3": "cross_env_*_sim_mppi_warm_n${NUM_EVAL}.json",
    "4": "cross_env_reacher_sim_cem_finger_n${NUM_EVAL}.json",
}
root = Path("${REPORT}")
for p in sorted(root.glob(patterns.get(step, "*.json"))):
    d = json.loads(p.read_text())
    print(
        p.name,
        "success=", d.get("success_rate"),
        "planner=", d.get("planner"),
        "warm=", d.get("warm_start"),
        "metric=", d.get("success_metric", "default"),
    )
PY
