#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export STABLEWM_HOME="${STABLEWM_HOME:-${HOME}/.stable-wm}"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

SAMPLE_DIR="${SAMPLE_DIR:-$ROOT/artifacts/visualpatchworld/fair_image_action_samples}"
DATA_DIR="${DATA_DIR:-$ROOT/artifacts/visualpatchworld/fair_data}"
CACHE_DIR="${VLM_CACHE_DIR:-$ROOT/artifacts/visualpatchworld/fair_vlm_cache}"
REPORT_DIR="${REPORT_DIR:-$ROOT/artifacts/visualpatchworld/fair_reports}"
RESULT_DIR="${RESULT_DIR:-$ROOT/artifacts/visualpatchworld/fair_results}"

DATASET="${DATASET:-pusht_expert_train.h5}"
MAX_EPISODES="${MAX_EPISODES:-20}"
MAX_STEPS="${MAX_STEPS:-12}"
STEP_STRIDE="${STEP_STRIDE:-5}"

log() {
  echo "[visualpatchworld-fair][$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

cd "$ROOT"
mkdir -p "$SAMPLE_DIR" "$DATA_DIR" "$CACHE_DIR" "$REPORT_DIR" "$RESULT_DIR"

log "Starting fair image-only PushT pipeline"
log "No --vlm_fallback_oracle is used. VLM cache must come from pixels only."
log "DATASET=$DATASET MAX_EPISODES=$MAX_EPISODES MAX_STEPS=$MAX_STEPS STEP_STRIDE=$STEP_STRIDE"

log "Sampling raw image/action trajectories"
python "$ROOT/scripts/visualpatchworld/sample_image_action_trajectories.py" \
  --dataset "$DATASET" \
  --output_dir "$SAMPLE_DIR" \
  --max_episodes "$MAX_EPISODES" \
  --max_steps_per_episode "$MAX_STEPS" \
  --step_stride "$STEP_STRIDE"

log "Exporting oracle labels for diagnostics only"
python "$ROOT/scripts/visualpatchworld/sample_trajectories.py" \
  --sources oracle \
  --dataset "$DATASET" \
  --output_dir "$DATA_DIR" \
  --max_episodes "$MAX_EPISODES" \
  --max_steps_per_episode "$MAX_STEPS" \
  --step_stride "$STEP_STRIDE"

log "Preparing VLM trajectories inside the world-model induction adapter"
induce_args=(
  "$ROOT/scripts/visualpatchworld/induce_world_model.py"
    --source vlm
    --sample_dir "$SAMPLE_DIR"
    --vlm_cache_dir "$CACHE_DIR"
    --data_dir "$DATA_DIR"
    --output_dir "$RESULT_DIR"
    --vlm_model "${VLM_MODEL:-Qwen/Qwen3.5-397B-A17B}"
    --vlm_base_url "${VLM_BASE_URL:-https://api.deepinfra.com/v1/openai}"
    --vlm_api_key_env "${VLM_API_KEY_ENV:-DEEPINFRA_API_KEY}"
    --max_train "${PATCHWORLD_MAX_TRAIN:-8}"
    --max_eval "${PATCHWORLD_MAX_EVAL:-4}"
    --max_refine_rounds "${PATCHWORLD_REFINE_ROUNDS:-4}"
)
if [[ "${VLM_OVERWRITE_CACHE:-0}" == "1" ]]; then
  induce_args+=(--vlm_overwrite_cache)
fi
if [[ "${RUN_ROLLOUT:-0}" == "1" ]]; then
  induce_args+=(--run_rollout)
fi
if [[ "${USE_RESIDUAL_MEMORY:-0}" == "1" ]]; then
  induce_args+=(--use_residual_memory)
fi

if [[ "${RUN_PATCHWORLD:-0}" == "1" ]]; then
  log "Inducing and evaluating image-only VLM PatchWorld model"
  python "${induce_args[@]}"
  python "$ROOT/scripts/visualpatchworld/evaluate_dynamics.py" \
    --source vlm \
    --results_dir "$RESULT_DIR" \
    --output "$REPORT_DIR/pusht_dynamics_vlm_fair.json"
  python "$ROOT/scripts/visualpatchworld/evaluate_rollout.py" \
    --source vlm \
    --results_dir "$RESULT_DIR" \
    --output "$REPORT_DIR/pusht_rollout_vlm_fair.json"
else
  log "Preparing VLM trajectories only because RUN_PATCHWORLD is not 1"
  python "${induce_args[@]}" --prepare_only
fi

log "Evaluating perception on val and test splits"
python "$ROOT/scripts/visualpatchworld/evaluate_perception.py" \
  --split val \
  --data_dir "$DATA_DIR" \
  --report_dir "$REPORT_DIR"
python "$ROOT/scripts/visualpatchworld/evaluate_perception.py" \
  --split test \
  --data_dir "$DATA_DIR" \
  --report_dir "$REPORT_DIR"

if [[ "${RUN_ORACLE_UPPER_BOUND:-0}" == "1" ]]; then
  log "Running oracle upper-bound induction for diagnostics only"
  oracle_args=(
    "$ROOT/scripts/visualpatchworld/induce_world_model.py"
    --source oracle
    --data_dir "$DATA_DIR"
    --output_dir "$RESULT_DIR"
    --max_train "${PATCHWORLD_MAX_TRAIN:-8}"
    --max_eval "${PATCHWORLD_MAX_EVAL:-4}"
    --max_refine_rounds "${PATCHWORLD_REFINE_ROUNDS:-4}"
  )
  if [[ "${RUN_ROLLOUT:-0}" == "1" ]]; then
    oracle_args+=(--run_rollout)
  fi
  if [[ "${USE_RESIDUAL_MEMORY:-0}" == "1" ]]; then
    oracle_args+=(--use_residual_memory)
  fi
  python "${oracle_args[@]}"

  python "$ROOT/scripts/visualpatchworld/evaluate_dynamics.py" \
    --source oracle \
    --results_dir "$RESULT_DIR" \
    --output "$REPORT_DIR/pusht_dynamics_oracle_upper_bound.json"
  python "$ROOT/scripts/visualpatchworld/evaluate_rollout.py" \
    --source oracle \
    --results_dir "$RESULT_DIR" \
    --output "$REPORT_DIR/pusht_rollout_oracle_upper_bound.json"
fi

log "Finished fair image-only PushT pipeline"
