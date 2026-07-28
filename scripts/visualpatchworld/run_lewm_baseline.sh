#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export STABLEWM_HOME="${STABLEWM_HOME:-${HOME}/.stable-wm}"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

DATA_DIR="${DATA_DIR:-$ROOT/artifacts/visualpatchworld/lewm_aligned}"
RESULT_DIR="${RESULT_DIR:-$ROOT/artifacts/visualpatchworld/lewm_results}"
REPORT_DIR="${REPORT_DIR:-$ROOT/artifacts/visualpatchworld/reports}"
MAX_CLIPS="${MAX_CLIPS:-500}"
MAX_TRAIN="${MAX_TRAIN:-8}"
MAX_EVAL="${MAX_EVAL:-4}"
MAX_REFINE_ROUNDS="${MAX_REFINE_ROUNDS:-2}"

log() {
  echo "[lewm-baseline][$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

step="${1:-all}"

run_sample() {
  log "Step 1/3: sampling LeWM-aligned trajectories (max_clips=${MAX_CLIPS})"
  python "$ROOT/scripts/visualpatchworld/sample_lewm_aligned_trajectories.py" \
    --sources both \
    --output_dir "$DATA_DIR" \
    --max_clips "$MAX_CLIPS"
}

run_induce() {
  log "Step 2/3: inducing oracle PatchWorld model"
  python "$ROOT/scripts/visualpatchworld/induce_world_model.py" \
    --source oracle \
    --data_dir "$DATA_DIR" \
    --output_dir "$RESULT_DIR" \
    --max_train "$MAX_TRAIN" \
    --max_eval "$MAX_EVAL" \
    --max_refine_rounds "$MAX_REFINE_ROUNDS" \
    --verbose
}

run_eval() {
  log "Step 3/3: evaluating induced model against simulator ground truth"
  python "$ROOT/scripts/visualpatchworld/evaluate_sim_steps.py" \
    --source oracle \
    --split val \
    --data_dir "$DATA_DIR" \
    --results_dir "$RESULT_DIR" \
    --output "$REPORT_DIR/pusht_sim_steps_lewm_oracle_val.json"
}

case "$step" in
  sample) run_sample ;;
  induce) run_induce ;;
  eval) run_eval ;;
  all)
    run_sample
    run_induce
    run_eval
    ;;
  *)
    echo "Usage: $0 [sample|induce|eval|all]" >&2
    exit 1
    ;;
esac

log "Done ($step)"
