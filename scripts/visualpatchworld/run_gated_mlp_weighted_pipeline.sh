#!/usr/bin/env bash
# Gated + MLP dynamics with round-2 merged failure mining:
# planning failures v1 + v2 + rule violations -> weighted refit -> fair planning.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-${CONDA_PREFIX:-}/envs/lewm/bin/python3}"
PATCHWORLD_ROOT="${PATCHWORLD_ROOT:-${ROOT}}"
export STABLEWM_HOME="${STABLEWM_HOME:-${HOME}/.stable-wm}"
export PYTHONPATH="$PATCHWORLD_ROOT:$ROOT/src:${PYTHONPATH:-}"

DATA_DIR="${DATA_DIR:-$ROOT/artifacts/visualpatchworld/lewm_tool_data}"
REPORT_DIR="${REPORT_DIR:-$ROOT/artifacts/visualpatchworld/reports}"
TRAIN_GLOB="${TRAIN_GLOB:-$DATA_DIR/pusht_tool_traj_train.jsonl}"

FAILURE_JSONL_V1="${FAILURE_JSONL_V1:-$DATA_DIR/planning_failure_transitions.jsonl}"
FAILURE_JSONL_V2="${FAILURE_JSONL_V2:-$DATA_DIR/planning_failure_transitions_v2.jsonl}"
RULE_VIOLATION_JSONL="${RULE_VIOLATION_JSONL:-$DATA_DIR/rule_violation_transitions.jsonl}"
MERGED_FAILURE_JSONL="${MERGED_FAILURE_JSONL:-$FAILURE_JSONL_V1,$FAILURE_JSONL_V2,$RULE_VIOLATION_JSONL}"

OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/artifacts/visualpatchworld/lewm_tool_scaffold_fit_gated_mlp_weighted_results}"
MAX_TRAIN="${MAX_TRAIN:-16}"
MAX_EVAL="${MAX_EVAL:-4}"
SIM_VERIFY_FRAC="${SIM_VERIFY_FRAC:-0.4}"
RUN_PLANNING="${RUN_PLANNING:-1}"

PHYSICS_REPORT="${PHYSICS_REPORT:-$REPORT_DIR/pusht_physics_rules_gated_mlp_weighted.json}"
INDUCED_REPORT="${INDUCED_REPORT:-$REPORT_DIR/pusht_planning_gated_mlp_weighted_simverify_cem_mpc_fair_n50.json}"
COMPARISON_REPORT="${COMPARISON_REPORT:-$REPORT_DIR/pusht_planning_fair_comparison_gated_mlp_weighted_n50.json}"

log() {
  echo "[gated-mlp-weighted][$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

log "Step 1: induce gated+MLP weighted model (merged failures)"
"$PYTHON" "$ROOT/scripts/visualpatchworld/induce_world_model.py" \
  --source tool \
  --data_dir "$DATA_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --max_train "$MAX_TRAIN" \
  --max_eval "$MAX_EVAL" \
  --induction_strategy pusht_scaffold_fit_contact_gated_mlp_weighted \
  --failure_transitions_jsonl "$MERGED_FAILURE_JSONL" \
  --max_refine_rounds 0

log "Step 2: physics-rule audit"
"$PYTHON" "$ROOT/scripts/visualpatchworld/evaluate_physics_rules.py" \
  --trajectories_glob "$TRAIN_GLOB" \
  --source tool \
  --results_dir "$OUTPUT_DIR" \
  --output "$PHYSICS_REPORT"

if [[ "$RUN_PLANNING" == "1" ]]; then
  log "Step 3: fair planning (sim_verify_frac=${SIM_VERIFY_FRAC})"
  INDUCED_RESULTS_DIR="$OUTPUT_DIR" \
  SIM_VERIFY_FRAC="$SIM_VERIFY_FRAC" \
  INDUCED_REPORT="$INDUCED_REPORT" \
  COMPARISON_REPORT="$COMPARISON_REPORT" \
  bash "$ROOT/scripts/visualpatchworld/run_fair_planning_comparison.sh"
fi

log "Done."
