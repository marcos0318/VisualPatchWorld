#!/usr/bin/env bash
# Rerun PushT MuJoCo+CEM fair (300x10, frame MPC) after sim_step PushT fix.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG="${ROOT}/logs/paper_experiments"
mkdir -p "$LOG"

log() { echo "[pusht-fair][$(date +%H:%M:%S)] $*" | tee -a "$LOG/pusht_sim_cem_fair.log"; }

while pgrep -f "evaluate_cross_env_planning.py" >/dev/null 2>&1; do
  log "waiting for other evaluate_cross_env_planning.py jobs..."
  sleep 30
done

log "starting PushT fair sim CEM (frame MPC h=1,r=1,fs=1)"
STEP=1 ENV=pusht FORCE=1 bash "$ROOT/scripts/visualpatchworld/run_sim_ceiling_ladder.sh" \
  >> "$LOG/pusht_sim_cem_fair.log" 2>&1
log "done"
