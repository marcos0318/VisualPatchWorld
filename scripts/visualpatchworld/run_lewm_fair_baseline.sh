#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-${CONDA_PREFIX:-}/envs/lewm/bin/python3}"
export STABLEWM_HOME="${STABLEWM_HOME:-${HOME}/.stable-wm}"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

SAMPLE_DIR="${SAMPLE_DIR:-$ROOT/artifacts/visualpatchworld/lewm_fair_samples}"
ORACLE_DIR="${ORACLE_DIR:-$ROOT/artifacts/visualpatchworld/lewm_fair_oracle}"
DATA_DIR="${DATA_DIR:-$ROOT/artifacts/visualpatchworld/lewm_fair_data}"
CACHE_DIR="${VLM_CACHE_DIR:-$ROOT/artifacts/visualpatchworld/lewm_fair_vlm_cache}"
RESULT_DIR="${RESULT_DIR:-$ROOT/artifacts/visualpatchworld/lewm_fair_results}"
REPORT_DIR="${REPORT_DIR:-$ROOT/artifacts/visualpatchworld/lewm_fair_reports}"

DATASET="${DATASET:-pusht_expert_train.h5}"
MAX_CLIPS="${MAX_CLIPS:-500}"
MAX_TRAIN="${MAX_TRAIN:-8}"
MAX_EVAL="${MAX_EVAL:-4}"
MAX_REFINE_ROUNDS="${MAX_REFINE_ROUNDS:-4}"
RUN_PATCHWORLD="${RUN_PATCHWORLD:-1}"
VLM_MAX_FRAMES="${VLM_MAX_FRAMES:-0}"
VLM_SPLITS="${VLM_SPLITS:-}"

log() {
  echo "[lewm-fair][$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

step="${1:-all}"

run_sample() {
  log "Step 1: LeWM-aligned image/action sampling (max_clips=${MAX_CLIPS})"
  "$PYTHON" "$ROOT/scripts/visualpatchworld/sample_lewm_aligned_trajectories.py" \
    --sources image_action \
    --dataset "$DATASET" \
    --output_dir "$SAMPLE_DIR" \
    --max_clips "$MAX_CLIPS"

  log "Step 1b: Oracle labels for diagnostics only"
  "$PYTHON" "$ROOT/scripts/visualpatchworld/sample_lewm_aligned_trajectories.py" \
    --sources oracle \
    --dataset "$DATASET" \
    --output_dir "$ORACLE_DIR" \
    --max_clips "$MAX_CLIPS"
}

run_induce() {
  log "Step 2: Fair VLM induction (no oracle fallback)"
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
    --max_train "$MAX_TRAIN"
    --max_eval "$MAX_EVAL"
    --max_refine_rounds "$MAX_REFINE_ROUNDS"
  )
  if [[ "${VLM_OVERWRITE_CACHE:-0}" == "1" ]]; then
    induce_args+=(--vlm_overwrite_cache)
  fi
  if [[ "${VLM_MAX_FRAMES}" -gt 0 ]]; then
    induce_args+=(--vlm_max_frames "${VLM_MAX_FRAMES}")
  fi
  if [[ -n "${VLM_SPLITS}" ]]; then
    # shellcheck disable=SC2206
    induce_args+=(--vlm_splits ${VLM_SPLITS})
  fi
  if [[ "${RUN_PATCHWORLD:-0}" == "1" ]]; then
    "$PYTHON" "${induce_args[@]}" --verbose
  else
    log "RUN_PATCHWORLD!=1; preparing VLM trajectories only"
    "$PYTHON" "${induce_args[@]}" --prepare_only
  fi
}

run_perception() {
  log "Step 3: Perception eval (VLM vs oracle)"
  mkdir -p "$REPORT_DIR"
  "$PYTHON" "$ROOT/scripts/visualpatchworld/evaluate_perception.py" \
    --split val \
    --oracle_jsonl "$ORACLE_DIR/pusht_lewm_oracle_traj_val.jsonl" \
    --candidate_jsonl "$DATA_DIR/pusht_vlm_traj_val.jsonl" \
    --output "$REPORT_DIR/pusht_vlm_vs_oracle_val.json"
}

run_sim_eval() {
  log "Step 4: Sim-step eval vs ground-truth physics"
  "$PYTHON" "$ROOT/scripts/visualpatchworld/evaluate_sim_steps.py" \
    --source vlm \
    --split val \
    --data_dir "$DATA_DIR" \
    --trajectories_jsonl "$DATA_DIR/pusht_vlm_traj_val.jsonl" \
    --results_dir "$RESULT_DIR" \
    --output "$REPORT_DIR/pusht_sim_steps_vlm_fair_val.json"
}

run_planning() {
  log "Step 5a: Planning with oracle observations (model dynamics upper bound)"
  "$PYTHON" "$ROOT/scripts/visualpatchworld/evaluate_planning.py" \
    --source vlm \
    --results_dir "$RESULT_DIR" \
    --data_dir "$DATA_DIR" \
    --observation_mode oracle \
    --output "$REPORT_DIR/pusht_planning_vlm_oracle_obs.json"

  log "Step 5b: Planning with VLM trajectory observations (fairer)"
  "$PYTHON" "$ROOT/scripts/visualpatchworld/evaluate_planning.py" \
    --source vlm \
    --results_dir "$RESULT_DIR" \
    --data_dir "$DATA_DIR" \
    --observation_mode vlm_traj \
    --output "$REPORT_DIR/pusht_planning_vlm_fair.json"
}

cd "$ROOT"
mkdir -p "$SAMPLE_DIR" "$ORACLE_DIR" "$DATA_DIR" "$CACHE_DIR" "$RESULT_DIR" "$REPORT_DIR"

case "$step" in
  sample) run_sample ;;
  induce) run_induce ;;
  perception) run_perception ;;
  sim) run_sim_eval ;;
  planning) run_planning ;;
  all)
    run_sample
    run_induce
    run_perception
    run_sim_eval
    run_planning
    ;;
  *)
    echo "Usage: $0 [sample|induce|perception|sim|planning|all]" >&2
    exit 1
    ;;
esac

log "Done ($step)"
