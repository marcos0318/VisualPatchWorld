#!/usr/bin/env bash
# Pixel-fair cross-env planning: tool_live re-observation + general_scaffold_fit models.
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
log() { echo "[cross-env-pixel][$(date +%H:%M:%S)] $*"; }

run_env() {
  local env="$1"
  local out="$REPORT/cross_env_${env}_tool_live_n${NUM_EVAL}.json"
  if [[ -f "$out" && "$FORCE" != "1" ]]; then
    log "skip $env (exists)"
    return 0
  fi
  local results="${ROOT}/artifacts/visualpatchworld/lewm_${env}_general_results"
  log "run $env tool_live -> $out"
  "$PY" "$ROOT/scripts/visualpatchworld/evaluate_cross_env_planning.py" \
    --env "$env" \
    --results_dir "$results" \
    --source oracle \
    --observation_mode tool_live \
    --num_eval "$NUM_EVAL" \
    --num_samples "$NUM_SAMPLES" \
    --cem_iters "$CEM_ITERS" \
    --output "$out" \
    2>&1 | tee "$LOG/cross_env_${env}_tool_live.log"
}

run_cube_sim() {
  local out="$REPORT/cross_env_cube_sim_cem_n${NUM_EVAL}.json"
  if [[ -f "$out" && "$FORCE" != "1" ]]; then
    log "skip cube sim_cem (exists)"
    return 0
  fi
  log "run cube sim_cem (light budget)"
  "$PY" "$ROOT/scripts/visualpatchworld/evaluate_cross_env_planning.py" \
    --env cube \
    --results_dir "${ROOT}/artifacts/visualpatchworld/lewm_cube_general_results" \
    --planner sim_cem \
    --num_eval "$NUM_EVAL" \
    --num_samples "${SIM_SAMPLES:-50}" \
    --cem_iters "${SIM_ITERS:-3}" \
    --output "$out" \
    2>&1 | tee "$LOG/cross_env_cube_sim_cem.log"
}

for env in tworoom cube reacher; do
  run_env "$env"
done
run_cube_sim

log "summary"
"$PY" - <<PY
import json
from pathlib import Path
root = Path("$REPORT")
for p in sorted(root.glob("cross_env_*tool_live*.json")) + sorted(root.glob("cross_env_cube_sim_cem*.json")):
    d = json.loads(p.read_text())
    print(p.name, "success=", d.get("success_rate"), "tool_fail=", d.get("tool_extraction_failures"))
PY
