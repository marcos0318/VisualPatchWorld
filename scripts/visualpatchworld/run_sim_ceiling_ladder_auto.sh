#!/usr/bin/env bash
# Wait for Step 1 outputs, finish any missing envs, then run Steps 2–4 sequentially.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPORT="${ROOT}/artifacts/visualpatchworld/reports"
LOG="${ROOT}/logs/paper_experiments"
NUM_EVAL="${NUM_EVAL:-50}"
POLL_SEC="${POLL_SEC:-120}"
FROM_STEP="${FROM_STEP:-2}"

mkdir -p "$LOG"
log() { echo "[sim-ladder-auto][$(date +%H:%M:%S)] $*" | tee -a "$LOG/sim_ceiling_ladder_auto.log"; }

envs=(tworoom cube reacher pusht)

missing_step1() {
  local m=()
  for env in "${envs[@]}"; do
    [[ -f "$REPORT/cross_env_${env}_sim_cem_fair_n${NUM_EVAL}.json" ]] || m+=("$env")
  done
  echo "${m[@]}"
}

wait_for_step1() {
  log "polling Step 1 fair CEM every ${POLL_SEC}s"
  while true; do
    local missing
    missing=$(missing_step1)
    if [[ -z "$missing" ]]; then
      log "Step 1 complete for all envs"
      return 0
    fi
    log "Step 1 pending:${missing}"
    sleep "$POLL_SEC"
  done
}

finish_step1_missing() {
  local missing
  missing=$(missing_step1)
  for env in $missing; do
    if pgrep -f "cross_env_${env}_sim_cem_fair_n${NUM_EVAL}.json" >/dev/null 2>&1; then
      log "skip Step 1 $env (eval already running)"
      continue
    fi
    log "run Step 1 for $env"
    STEP=1 ENV="$env" FORCE=0 bash "$ROOT/scripts/visualpatchworld/run_sim_ceiling_ladder.sh" \
      >> "$LOG/sim_ceiling_ladder_master.log" 2>&1
  done
}

log "auto pipeline start (FROM_STEP=$FROM_STEP)"
while true; do
  finish_step1_missing
  missing=$(missing_step1)
  [[ -z "$missing" ]] && break
  log "still waiting on:${missing}"
  sleep "$POLL_SEC"
done

for step in 2 3 4; do
  if (( step < FROM_STEP )); then
    continue
  fi
  log "starting Step $step"
  STEP="$step" FORCE=0 bash "$ROOT/scripts/visualpatchworld/run_sim_ceiling_ladder.sh" \
    >> "$LOG/sim_ceiling_ladder_master.log" 2>&1
  log "Step $step done"
done

log "pipeline complete"
for env in "${envs[@]}"; do
  for tag in sim_cem_fair sim_cem_warm sim_mppi_warm; do
    f="$REPORT/cross_env_${env}_${tag}_n${NUM_EVAL}.json"
    [[ -f "$f" ]] || continue
    sr=$(python3 -c "import json; print(json.load(open('$f'))['success_rate'])")
    log "  $f success=$sr"
  done
done
finger="$REPORT/cross_env_reacher_sim_cem_finger_n${NUM_EVAL}.json"
if [[ -f "$finger" ]]; then
  sr=$(python3 -c "import json; print(json.load(open('$finger'))['success_rate'])")
  log "  $finger success=$sr"
fi
