#!/usr/bin/env bash
# Cube step: tool_live + hybrid sim-verify (after CV recalibration).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PY:-python3}"
export STABLEWM_HOME="${STABLEWM_HOME:-${HOME}/.stable-wm}"
export MUJOCO_GL=egl
export PYTHONPATH="${ROOT}/src:${ROOT}:${PYTHONPATH:-}"

REPORT="${ROOT}/artifacts/visualpatchworld/reports"
LOG="${ROOT}/logs/paper_experiments"
RESULTS="${ROOT}/artifacts/visualpatchworld/lewm_cube_general_results"
NUM_EVAL="${NUM_EVAL:-50}"
NUM_SAMPLES="${NUM_SAMPLES:-300}"
CEM_ITERS="${CEM_ITERS:-10}"
FORCE="${FORCE:-0}"

mkdir -p "$REPORT" "$LOG"
log() { echo "[cube-improve][$(date +%H:%M:%S)] $*"; }

run_eval() {
  local tag="$1"
  local mode="$2"
  local sv="$3"
  local out="$REPORT/cross_env_cube_${tag}_n${NUM_EVAL}.json"
  if [[ -f "$out" && "$FORCE" != "1" ]]; then
    log "skip $tag (exists)"
    return 0
  fi
  log "run cube $tag (obs=$mode sim_verify=$sv)"
  extra=()
  if [[ "$sv" != "0" ]]; then
    extra+=(--sim_verify_frac "$sv")
  fi
  "$PY" "$ROOT/scripts/visualpatchworld/evaluate_cross_env_planning.py" \
    --env cube \
    --results_dir "$RESULTS" \
    --source oracle \
    --observation_mode "$mode" \
    --num_eval "$NUM_EVAL" \
    --num_samples "$NUM_SAMPLES" \
    --cem_iters "$CEM_ITERS" \
    "${extra[@]}" \
    --output "$out" \
    2>&1 | tee "$LOG/cube_${tag}.log"
}

run_eval "tool_live_v2" tool_live 0
run_eval "hybrid_tool_sv03" tool_live 0.3
run_eval "hybrid_oracle_sv03" oracle 0.3

log "summary"
"$PY" - <<PY
import json
from pathlib import Path
for p in sorted(Path("$REPORT").glob("cross_env_cube_*_n${NUM_EVAL}.json")):
    d = json.loads(p.read_text())
    print(p.name, "success=", d.get("success_rate"), "tool_fail=", d.get("tool_extraction_failures"), "sv=", d.get("sim_verify_frac"))
PY
