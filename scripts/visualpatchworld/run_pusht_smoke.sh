#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export STABLEWM_HOME="${STABLEWM_HOME:-${HOME}/.stable-wm}"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

DATA_DIR="$ROOT/artifacts/visualpatchworld/data"
REPORT_DIR="$ROOT/artifacts/visualpatchworld/reports"

log() {
  echo "[visualpatchworld][$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

log "Starting PushT smoke pipeline"
log "STABLEWM_HOME=$STABLEWM_HOME"
log "MAX_EPISODES=${MAX_EPISODES:-6} MAX_STEPS=${MAX_STEPS:-12} RUN_PATCHWORLD=${RUN_PATCHWORLD:-0}"

log "Sampling oracle and VLM PushT trajectories"
python "$ROOT/scripts/visualpatchworld/sample_trajectories.py" \
  --sources both \
  --output_dir "$DATA_DIR" \
  --max_episodes "${MAX_EPISODES:-6}" \
  --max_steps_per_episode "${MAX_STEPS:-12}" \
  --vlm_cache_dir "${VLM_CACHE_DIR:-$ROOT/artifacts/visualpatchworld/vlm_cache}" \
  --vlm_fallback_oracle

log "Comparing VLM scene graphs against oracle validation graphs"
python "$ROOT/scripts/visualpatchworld/evaluate_perception.py" \
  --split val \
  --data_dir "$DATA_DIR" \
  --output "$REPORT_DIR/pusht_vlm_vs_oracle_smoke.json"

if [[ "${RUN_PATCHWORLD:-0}" == "1" ]]; then
  log "Starting PatchWorld induction/evaluation"
  python "$ROOT/scripts/visualpatchworld/induce_world_model.py" \
    --source oracle \
    --max_train "${PATCHWORLD_MAX_TRAIN:-4}" \
    --max_eval "${PATCHWORLD_MAX_EVAL:-2}" \
    --max_refine_rounds "${PATCHWORLD_REFINE_ROUNDS:-2}" \
    --use_residual_memory \
    --verbose
  python "$ROOT/scripts/visualpatchworld/evaluate_dynamics.py" --source oracle
  python "$ROOT/scripts/visualpatchworld/evaluate_rollout.py" --source oracle
  log "Finished PatchWorld induction/evaluation"
else
  log "Skipping PatchWorld induction because RUN_PATCHWORLD is not 1"
fi

log "Finished PushT smoke pipeline"
