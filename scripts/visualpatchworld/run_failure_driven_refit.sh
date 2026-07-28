#!/usr/bin/env bash
# Fast failure-driven refit pipeline:
# 1) mine high-regret MPC transitions
# 2) weighted contact-fit induction
# 3) optional fair planning comparison
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-${CONDA_PREFIX:-}/envs/lewm/bin/python3}"
PATCHWORLD_ROOT="${PATCHWORLD_ROOT:-${ROOT}}"
export STABLEWM_HOME="${STABLEWM_HOME:-${HOME}/.stable-wm}"
export PYTHONPATH="$PATCHWORLD_ROOT:$ROOT/src:${PYTHONPATH:-}"

BASE_RESULTS_DIR="${BASE_RESULTS_DIR:-$ROOT/artifacts/visualpatchworld/lewm_tool_scaffold_fit_results}"
DATA_DIR="${DATA_DIR:-$ROOT/artifacts/visualpatchworld/lewm_tool_data}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/artifacts/visualpatchworld/lewm_tool_scaffold_fit_weighted_results}"
FAILURE_JSONL="${FAILURE_JSONL:-$DATA_DIR/planning_failure_transitions.jsonl}"
NUM_EVAL_MINE="${NUM_EVAL_MINE:-16}"
MAX_TRAIN="${MAX_TRAIN:-16}"
MAX_EVAL="${MAX_EVAL:-4}"
RUN_PLANNING="${RUN_PLANNING:-1}"

log() {
  echo "[failure-refit][$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

log "Step 1: mine planning failures"
"$PYTHON" "$ROOT/scripts/visualpatchworld/mine_planning_failures.py" \
  --source tool \
  --results_dir "$BASE_RESULTS_DIR" \
  --output "$FAILURE_JSONL" \
  --num_eval "$NUM_EVAL_MINE"

log "Step 2: weighted contact-fit induction"
"$PYTHON" "$ROOT/scripts/visualpatchworld/induce_world_model.py" \
  --source tool \
  --data_dir "$DATA_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --max_train "$MAX_TRAIN" \
  --max_eval "$MAX_EVAL" \
  --induction_strategy pusht_scaffold_fit_contact_weighted \
  --failure_transitions_jsonl "$FAILURE_JSONL" \
  --max_refine_rounds 0

if [[ "$RUN_PLANNING" == "1" ]]; then
  log "Step 3: fair planning comparison"
  INDUCED_RESULTS_DIR="$OUTPUT_DIR" \
  INDUCED_REPORT="$ROOT/artifacts/visualpatchworld/reports/pusht_planning_tool_scaffold_fit_weighted_cem_mpc_fair_n50.json" \
  COMPARISON_REPORT="$ROOT/artifacts/visualpatchworld/reports/pusht_planning_fair_comparison_weighted_n50.json" \
  bash "$ROOT/scripts/visualpatchworld/run_fair_planning_comparison.sh"
fi

log "Done."
