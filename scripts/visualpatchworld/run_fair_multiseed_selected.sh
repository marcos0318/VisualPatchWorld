#!/usr/bin/env bash
# Fair-protocol multi-seed CI (cost-aware paper refresh).
#
# Batch A: tworoom tool/oracle hybrid + mujoco  (seeds 42-46)
# Batch B: pusht tool induced + tool hybrid     (seeds 42-46)
# Batch C: cube  tool induced + tool hybrid     (seeds 42-46)
#
# Existing seed-42 fair JSONs are skipped. Resume-safe.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PY:-python3}"
OUT="${ROOT}/artifacts/visualpatchworld/reports/fair_wm_ranking_gt90/all_methods"
LOG="${ROOT}/logs/paper_experiments"
QUEUE_LOG="${LOG}/fair_multiseed_selected_queue.log"

export STABLEWM_HOME="${STABLEWM_HOME:-${HOME}/.stable-wm}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export PYTHONPATH="${ROOT}/src:${ROOT}:${ROOT}/scripts/visualpatchworld:${PYTHONPATH:-}"

NUM_EVAL="${NUM_EVAL:-50}"
SEEDS="${SEEDS:-42 43 44 45 46}"
BATCHES="${BATCHES:-A,B,C}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
FORCE="${FORCE:-0}"
EPISODE_WORKERS="${EPISODE_WORKERS:-8}"

mkdir -p "$OUT" "$LOG"

log() {
  echo "[fair-ms][$(date +%Y-%m-%dT%H:%M:%S)] $*" | tee -a "$QUEUE_LOG"
}

# job line format: METHOD ENV
build_jobs() {
  local batches_upper
  batches_upper=$(echo "$BATCHES" | tr '[:lower:]' '[:upper:]')
  if [[ "$batches_upper" == *A* ]]; then
    echo "vpw_tool_hybrid tworoom"
    echo "vpw_oracle_hybrid tworoom"
    echo "mujoco_cem tworoom"
  fi
  if [[ "$batches_upper" == *B* ]]; then
    echo "vpw_tool_induced pusht"
    echo "vpw_tool_hybrid pusht"
  fi
  if [[ "$batches_upper" == *C* ]]; then
    echo "vpw_tool_induced cube"
    echo "vpw_tool_hybrid cube"
  fi
}

run_cell() {
  local method="$1"
  local env="$2"
  local seed="$3"
  local out="${OUT}/${method}_${env}_n${NUM_EVAL}_seed${seed}.json"
  local job_log="${LOG}/fair_ms_${method}_${env}_n${NUM_EVAL}_seed${seed}.log"

  if [[ -f "$out" && "$FORCE" != "1" ]]; then
    log "skip ${method}/${env}/seed${seed}"
    return 0
  fi

  log "run  ${method}/${env}/seed${seed} -> $(basename "$out")"
  if [[ "$method" == "mujoco_cem" ]]; then
    if "$PY" -u "${ROOT}/scripts/visualpatchworld/run_fair_multiseed_selected.py" \
        --batches none \
        --seeds "$seed" \
        --num_eval "$NUM_EVAL" \
        --max_parallel 1 \
        --episode_workers "$EPISODE_WORKERS" \
        --output_dir "$OUT" \
        >/dev/null 2>&1; then
      :
    fi
    # Direct one-cell mujoco via inline Python (avoids batch filter 'none').
    if ! "$PY" -u - <<PY >"$job_log" 2>&1
import json, sys
from pathlib import Path
sys.path[:0] = [
    "${ROOT}/src",
    "${ROOT}",
    "${ROOT}/scripts/visualpatchworld",
]
from run_fair_multiseed_selected import _run_mujoco, _out_path
out = _out_path(Path("${OUT}"), "mujoco_cem", "${env}", ${NUM_EVAL}, ${seed})
report = _run_mujoco("${env}", ${NUM_EVAL}, ${seed}, ${EPISODE_WORKERS})
out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
print("success_rate", report.get("success_rate"), flush=True)
PY
    then
      log "fail ${method}/${env}/seed${seed} (see ${job_log})"
      return 0
    fi
  else
    if ! "$PY" -u "${ROOT}/scripts/visualpatchworld/run_fair_baseline_replan_gt90.py" \
        --methods "$method" \
        --envs "$env" \
        --num_eval "$NUM_EVAL" \
        --seed "$seed" \
        --episode_workers "$EPISODE_WORKERS" \
        --output_dir "$OUT" \
        >"$job_log" 2>&1; then
      log "fail ${method}/${env}/seed${seed} (see ${job_log})"
      return 0
    fi
  fi
  log "done ${method}/${env}/seed${seed}"
}

# Simple parallel queue
pids=()
labels=()
wait_one() {
  if ((${#pids[@]} == 0)); then
    return 0
  fi
  local pid="${pids[0]}"
  local label="${labels[0]}"
  pids=("${pids[@]:1}")
  labels=("${labels[@]:1}")
  if ! wait "$pid"; then
    log "wait-fail ${label} pid=${pid}"
  fi
}

log "start batches=${BATCHES} seeds=${SEEDS} parallel=${MAX_PARALLEL}"
mapfile -t CELLS < <(build_jobs)
for seed in $SEEDS; do
  for cell in "${CELLS[@]}"; do
    method="${cell%% *}"
    env="${cell##* }"
    while ((${#pids[@]} >= MAX_PARALLEL)); do
      wait_one
    done
    run_cell "$method" "$env" "$seed" &
    pids+=("$!")
    labels+=("${method}/${env}/seed${seed}")
  done
done
while ((${#pids[@]} > 0)); do
  wait_one
done
log "all queued cells finished"
log "aggregate with: python ${ROOT}/../KDD2027VisualPatchWorld/scripts/aggregate_fair_multiseed_ci.py --format text"
