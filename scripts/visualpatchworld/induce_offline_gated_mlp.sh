#!/usr/bin/env bash
# Phase 0: offline gated+MLP induction from trajectories only (no counterexamples).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-${CONDA_PREFIX:-}/envs/lewm/bin/python3}"
PATCHWORLD_ROOT="${PATCHWORLD_ROOT:-${ROOT}}"
export STABLEWM_HOME="${STABLEWM_HOME:-${HOME}/.stable-wm}"
export PYTHONPATH="$PATCHWORLD_ROOT:$ROOT/src:${PYTHONPATH:-}"

DATA_DIR="${DATA_DIR:-$ROOT/artifacts/visualpatchworld/lewm_tool_data}"
REPORT_DIR="${REPORT_DIR:-$ROOT/artifacts/visualpatchworld/reports}"
TRAIN_GLOB="${TRAIN_GLOB:-$DATA_DIR/pusht_tool_traj_train.jsonl}"

OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/artifacts/visualpatchworld/lewm_tool_modular_offline_results}"
MAX_TRAIN="${MAX_TRAIN:-16}"
MAX_EVAL="${MAX_EVAL:-4}"
RUN_PHYSICS="${RUN_PHYSICS:-1}"

PHYSICS_REPORT="${PHYSICS_REPORT:-$REPORT_DIR/pusht_physics_rules_modular_offline.json}"

log() {
  echo "[modular-offline][$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

log "Step 1: offline modular gated+MLP induction (trajectories only)"
"$PYTHON" "$ROOT/scripts/visualpatchworld/induce_world_model.py" \
  --source tool \
  --data_dir "$DATA_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --max_train "$MAX_TRAIN" \
  --max_eval "$MAX_EVAL" \
  --induction_strategy pusht_modular_gated_mlp_offline \
  --max_refine_rounds 0

if [[ "$RUN_PHYSICS" == "1" ]]; then
  log "Step 2: physics-rule audit"
  "$PYTHON" "$ROOT/scripts/visualpatchworld/evaluate_physics_rules.py" \
    --trajectories_glob "$TRAIN_GLOB" \
    --source tool \
    --results_dir "$OUTPUT_DIR" \
    --output "$PHYSICS_REPORT"
fi

log "Done. Results: $OUTPUT_DIR"
