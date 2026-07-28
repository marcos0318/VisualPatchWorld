#!/usr/bin/env bash
# PushT step: hardened CV + program-induced tool_live (CEM 300x30).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PY:-python3}"
export STABLEWM_HOME="${STABLEWM_HOME:-${HOME}/.stable-wm}"
export MUJOCO_GL=egl
export PYTHONPATH="${ROOT}/src:${ROOT}:${PYTHONPATH:-}"

REPORT="${ROOT}/artifacts/visualpatchworld/reports"
LOG="${ROOT}/logs/paper_experiments"
MODEL="${ROOT}/artifacts/visualpatchworld/induced/program_induced_best.py"
NUM_EVAL="${NUM_EVAL:-50}"
NUM_SAMPLES="${NUM_SAMPLES:-300}"
CEM_ITERS="${CEM_ITERS:-30}"

mkdir -p "$REPORT" "$LOG"
log() { echo "[pusht-improve][$(date +%H:%M:%S)] $*"; }

# Kill stale eval if stuck >2h
old_pids=$(pgrep -f "evaluate_planning.py.*pusht_tool_live_program_n50_v2" || true)
if [[ -n "$old_pids" ]]; then
  log "stopping stale PushT eval: $old_pids"
  kill $old_pids 2>/dev/null || true
  sleep 2
fi

out="$REPORT/pusht_tool_live_program_n${NUM_EVAL}_v2.json"
log "run tool_live program CEM ${NUM_SAMPLES}x${CEM_ITERS}"
"$PY" "$ROOT/scripts/visualpatchworld/evaluate_planning.py" \
  --source oracle \
  --model_path "$MODEL" \
  --results_dir "${ROOT}/artifacts/visualpatchworld/induced" \
  --planner cem_mpc \
  --observation_mode tool_live \
  --expose_block_velocity \
  --num_eval "$NUM_EVAL" \
  --seed 42 \
  --goal_offset_steps 25 \
  --eval_budget 50 \
  --frameskip 1 \
  --horizon 5 \
  --model_rollout_horizon 5 \
  --receding_horizon 1 \
  --model_action_repeat 1 \
  --num_samples "$NUM_SAMPLES" \
  --cem_iters "$CEM_ITERS" \
  --output "$out" \
  2>&1 | tee "$LOG/pusht_tool_live_v2.log"

log "static perception audit"
"$PY" "$ROOT/scripts/visualpatchworld/evaluate_env_tool_perception.py" \
  --env pusht \
  --num_samples 100 \
  --output "${REPORT}/pusht_tool_perception_n100.json"

log "summary"
"$PY" - <<PY
import json
from pathlib import Path
for name in ["pusht_tool_live_program_n${NUM_EVAL}_v2.json", "pusht_tool_perception_n100.json"]:
    p = Path("$REPORT") / name
    if p.exists():
        d = json.loads(p.read_text())
        print(name, d.get("success_rate") or d.get("metrics", {}).get("position_rmse_xy"))
PY
