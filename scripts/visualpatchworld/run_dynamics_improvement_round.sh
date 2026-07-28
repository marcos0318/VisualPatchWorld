#!/usr/bin/env bash
# Round-2 dynamics improvement:
# 1) physics-rule audit on input model
# 2) re-mine planning failures from that model
# 3) collect rule-violation transitions from train trajectories
# 4) weighted contact-fit refit with merged failure JSONLs
# 5) fair planning comparison with higher sim-verify fraction
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-${CONDA_PREFIX:-}/envs/lewm/bin/python3}"
PATCHWORLD_ROOT="${PATCHWORLD_ROOT:-${ROOT}}"
export STABLEWM_HOME="${STABLEWM_HOME:-${HOME}/.stable-wm}"
export PYTHONPATH="$PATCHWORLD_ROOT:$ROOT/src:${PYTHONPATH:-}"

DATA_DIR="${DATA_DIR:-$ROOT/artifacts/visualpatchworld/lewm_tool_data}"
REPORT_DIR="${REPORT_DIR:-$ROOT/artifacts/visualpatchworld/reports}"

INPUT_RESULTS_DIR="${INPUT_RESULTS_DIR:-$ROOT/artifacts/visualpatchworld/lewm_tool_scaffold_fit_weighted_results}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/artifacts/visualpatchworld/lewm_tool_scaffold_fit_weighted_v2_results}"

FAILURE_JSONL_V1="${FAILURE_JSONL_V1:-$DATA_DIR/planning_failure_transitions.jsonl}"
FAILURE_JSONL_V2="${FAILURE_JSONL_V2:-$DATA_DIR/planning_failure_transitions_v2.jsonl}"
RULE_VIOLATION_JSONL="${RULE_VIOLATION_JSONL:-$DATA_DIR/rule_violation_transitions.jsonl}"
MERGED_FAILURE_JSONL="${MERGED_FAILURE_JSONL:-$FAILURE_JSONL_V1,$FAILURE_JSONL_V2,$RULE_VIOLATION_JSONL}"

TRAIN_GLOB="${TRAIN_GLOB:-$DATA_DIR/pusht_tool_traj_train.jsonl}"
MAX_TRAIN="${MAX_TRAIN:-16}"
MAX_EVAL="${MAX_EVAL:-4}"

NUM_EVAL_MINE="${NUM_EVAL_MINE:-24}"
FAILURE_WEIGHT="${FAILURE_WEIGHT:-8.0}"
RULE_VIOLATION_WEIGHT="${RULE_VIOLATION_WEIGHT:-3.0}"

SIM_VERIFY_FRAC="${SIM_VERIFY_FRAC:-0.4}"
RUN_PLANNING="${RUN_PLANNING:-1}"

PHYSICS_REPORT="${PHYSICS_REPORT:-$REPORT_DIR/pusht_physics_rules_weighted_v1.json}"
INDUCED_REPORT="${INDUCED_REPORT:-$REPORT_DIR/pusht_planning_weighted_v2_simverify_cem_mpc_fair_n50.json}"
COMPARISON_REPORT="${COMPARISON_REPORT:-$REPORT_DIR/pusht_planning_fair_comparison_weighted_v2_simverify_n50.json}"

log() {
  echo "[dynamics-round2][$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

log "Step 1: physics-rule audit on input model ($INPUT_RESULTS_DIR)"
"$PYTHON" "$ROOT/scripts/visualpatchworld/evaluate_physics_rules.py" \
  --trajectories_glob "$TRAIN_GLOB" \
  --source tool \
  --results_dir "$INPUT_RESULTS_DIR" \
  --output "$PHYSICS_REPORT"

log "Step 2: re-mine planning failures (num_eval=$NUM_EVAL_MINE, weight=$FAILURE_WEIGHT)"
"$PYTHON" "$ROOT/scripts/visualpatchworld/mine_planning_failures.py" \
  --source tool \
  --results_dir "$INPUT_RESULTS_DIR" \
  --output "$FAILURE_JSONL_V2" \
  --num_eval "$NUM_EVAL_MINE" \
  --failure_weight "$FAILURE_WEIGHT"

log "Step 3: collect rule-violation transitions (weight=$RULE_VIOLATION_WEIGHT)"
"$PYTHON" "$ROOT/scripts/visualpatchworld/collect_rule_violation_transitions.py" \
  --trajectories_glob "$TRAIN_GLOB" \
  --source tool \
  --results_dir "$INPUT_RESULTS_DIR" \
  --output "$RULE_VIOLATION_JSONL" \
  --weight "$RULE_VIOLATION_WEIGHT"

log "Step 4: weighted contact-fit refit v2 (merged failures)"
"$PYTHON" "$ROOT/scripts/visualpatchworld/induce_world_model.py" \
  --source tool \
  --data_dir "$DATA_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --max_train "$MAX_TRAIN" \
  --max_eval "$MAX_EVAL" \
  --induction_strategy pusht_scaffold_fit_contact_weighted \
  --failure_transitions_jsonl "$MERGED_FAILURE_JSONL" \
  --max_refine_rounds 0

if [[ "$RUN_PLANNING" == "1" ]]; then
  log "Step 5: fair planning comparison (sim_verify_frac=$SIM_VERIFY_FRAC)"
  INDUCED_RESULTS_DIR="$OUTPUT_DIR" \
  SIM_VERIFY_FRAC="$SIM_VERIFY_FRAC" \
  INDUCED_REPORT="$INDUCED_REPORT" \
  COMPARISON_REPORT="$COMPARISON_REPORT" \
  bash "$ROOT/scripts/visualpatchworld/run_fair_planning_comparison.sh"
fi

log "Done."
