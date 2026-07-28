#!/usr/bin/env bash
# Step 3: Hardened PushT CV + rerun program-induced tool_live eval.
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
CEM_ITERS="${CEM_ITERS:-30}"
MODEL="${ROOT}/artifacts/visualpatchworld/induced/program_induced_best.py"
FORCE="${FORCE:-1}"

mkdir -p "$REPORT" "$LOG"
log() { echo "[step3-pusht][$(date +%H:%M:%S)] $*"; }

out="$REPORT/pusht_tool_live_program_n${NUM_EVAL}.json"
log "run tool_live program-induced CEM ${NUM_SAMPLES}x${CEM_ITERS} -> $out"
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
  2>&1 | tee "$LOG/pusht_tool_live_program_hardened.log"

log "cube perception re-audit after calibration"
"$PY" "$ROOT/scripts/visualpatchworld/evaluate_env_tool_perception.py" \
  --env cube \
  --num_samples 200 \
  --output "${REPORT}/cube_tool_perception_n200_v2.json"

log "cube tool_live replan with new calibration"
"$PY" "$ROOT/scripts/visualpatchworld/evaluate_cross_env_planning.py" \
  --env cube \
  --results_dir "${ROOT}/artifacts/visualpatchworld/lewm_cube_general_results" \
  --source oracle \
  --observation_mode tool_live \
  --num_eval "$NUM_EVAL" \
  --num_samples "$NUM_SAMPLES" \
  --cem_iters 10 \
  --output "${REPORT}/cross_env_cube_tool_live_n${NUM_EVAL}_v2.json" \
  2>&1 | tee "$LOG/cross_env_cube_tool_live_v2.log"

log "summary"
"$PY" - <<PY
import json
from pathlib import Path
for name in [
    "pusht_tool_live_program_n${NUM_EVAL}.json",
    "cube_tool_perception_n200_v2.json",
    "cross_env_cube_tool_live_n${NUM_EVAL}_v2.json",
]:
    p = Path("$REPORT") / name
    if p.exists():
        d = json.loads(p.read_text())
        print(name, d.get("success_rate") or d.get("metrics", {}).get("position_rmse_xy"))
PY
