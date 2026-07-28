#!/usr/bin/env bash
# Run MuJoCo+CEM fair (300x10) for envs missing cross_env_*_sim_cem_fair_n50.json.
# One env at a time to reduce OOM risk.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PY:-python3}"
LOG="${ROOT}/logs/paper_experiments"
NUM_EVAL="${NUM_EVAL:-50}"
FORCE="${FORCE:-0}"

mkdir -p "$LOG"
log() { echo "[sim-fair-missing][$(date +%H:%M:%S)] $*" | tee -a "$LOG/sim_cem_fair_missing_master.log"; }

log "=== MuJoCo+CEM fair 300x10 (missing envs only) ==="
for env in tworoom cube reacher pusht; do
  out="${ROOT}/artifacts/visualpatchworld/reports/cross_env_${env}_sim_cem_fair_n${NUM_EVAL}.json"
  if [[ -f "$out" && "$FORCE" != "1" ]]; then
    log "skip $env (exists $out)"
    continue
  fi
  log "start $env"
  if STEP=1 ENV="$env" FORCE=1 bash "$ROOT/scripts/visualpatchworld/run_sim_ceiling_ladder.sh" \
    >> "$LOG/sim_cem_fair_missing_master.log" 2>&1; then
    log "done $env"
  else
    log "FAILED $env (exit $?)"
  fi
done

log "=== summary ==="
"$PY" - <<PY
import json
from pathlib import Path
root = Path("${ROOT}/artifacts/visualpatchworld/reports")
for env in ("tworoom", "cube", "reacher", "pusht"):
    p = root / f"cross_env_{env}_sim_cem_fair_n50.json"
    if p.exists():
        d = json.loads(p.read_text())
        print(env, "success=", d.get("success_rate"), "metric=", d.get("success_metric", "default"))
    else:
        print(env, "MISSING")
PY
log "sim_cem_fair_missing complete"
