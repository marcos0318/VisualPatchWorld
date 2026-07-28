#!/usr/bin/env bash
# Tier 2 tool hybrid (Two-room, Reacher) + Track B sim-ceiling ladder (sequential, one env at a time).
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
FORCE="${FORCE:-0}"

mkdir -p "$REPORT" "$LOG"
log() { echo "[paper-remaining][$(date +%H:%M:%S)] $*" | tee -a "$LOG/paper_remaining_master.log"; }

run_tool_hybrid() {
  local env="$1"
  local results_dir="$2"
  local out="$REPORT/cross_env_${env}_hybrid_tool_sv03_n${NUM_EVAL}.json"
  if [[ -f "$out" && "$FORCE" != "1" ]]; then
    log "skip Tier2 $env tool hybrid (exists $out)"
    return 0
  fi
  log "Tier2: $env tool_live + sim_verify=0.3 -> $out"
  "$PY" "$ROOT/scripts/visualpatchworld/evaluate_cross_env_planning.py" \
    --env "$env" \
    --results_dir "$results_dir" \
    --source oracle \
    --observation_mode tool_live \
    --sim_verify_frac 0.3 \
    --num_eval "$NUM_EVAL" \
    --num_samples "$NUM_SAMPLES" \
    --cem_iters "$CEM_ITERS" \
    --seed 42 \
    --output "$out" \
    2>&1 | tee "$LOG/tier2_${env}_tool_hybrid_sv03.log"
}

log "=== Tier 2: tool hybrid (Two-room, Reacher) ==="
run_tool_hybrid tworoom "${ROOT}/artifacts/visualpatchworld/lewm_tworoom_general_results"
run_tool_hybrid reacher "${ROOT}/artifacts/visualpatchworld/lewm_reacher_general_results"

log "=== Track B Step 1: fair sim CEM (missing envs, sequential) ==="
for env in cube reacher pusht; do
  out="$REPORT/cross_env_${env}_sim_cem_fair_n${NUM_EVAL}.json"
  if [[ -f "$out" && "$FORCE" != "1" ]]; then
    log "skip Step1 $env (exists)"
    continue
  fi
  log "Step1: $env fair CEM 300x10"
  STEP=1 ENV="$env" FORCE=1 bash "$ROOT/scripts/visualpatchworld/run_sim_ceiling_ladder.sh" \
    >> "$LOG/sim_ceiling_ladder_master.log" 2>&1
done

log "=== Track B Steps 2–4 (sequential) ==="
for step in 2 3 4; do
  log "starting Step $step"
  STEP="$step" FORCE=0 bash "$ROOT/scripts/visualpatchworld/run_sim_ceiling_ladder.sh" \
    >> "$LOG/sim_ceiling_ladder_master.log" 2>&1
  log "Step $step done"
done

log "=== summary ==="
"$PY" - <<PY
import json
from pathlib import Path

root = Path("${REPORT}")
print("--- Tier 2 tool hybrid ---")
for name in [
    "cross_env_tworoom_hybrid_tool_sv03_n${NUM_EVAL}.json",
    "cross_env_reacher_hybrid_tool_sv03_n${NUM_EVAL}.json",
    "pusht_tool_live_program_hybrid_sv03_n${NUM_EVAL}.json",
]:
    p = root / name
    if p.exists():
        d = json.loads(p.read_text())
        print(name, "success=", d.get("success_rate"), "obs=", d.get("observation_mode"), "sv=", d.get("sim_verify_frac"))
    else:
        print(name, "MISSING")

print("--- Track B ---")
for env in ("tworoom", "cube", "reacher", "pusht"):
    for tag in ("sim_cem_fair", "sim_cem_warm", "sim_mppi_warm"):
        p = root / f"cross_env_{env}_{tag}_n${NUM_EVAL}.json"
        if p.exists():
            d = json.loads(p.read_text())
            print(p.name, "success=", d.get("success_rate"), "planner=", d.get("planner"))
finger = root / "cross_env_reacher_sim_cem_finger_n${NUM_EVAL}.json"
if finger.exists():
    d = json.loads(finger.read_text())
    print(finger.name, "success=", d.get("success_rate"), "metric=", d.get("success_metric"))
PY

log "paper_remaining pipeline complete"
