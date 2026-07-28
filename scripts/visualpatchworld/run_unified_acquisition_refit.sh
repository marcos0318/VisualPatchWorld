#!/usr/bin/env bash
# Unified acquisition + modular contact refit + planning-gated acceptance.
#
# Replaces separate sim-CE / planning-failure / supervision JSONL merges with one
# priority-ranked acquisition pool, then refits contact modules and optionally
# accepts the round only if fair planning improves.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-${CONDA_PREFIX:-}/envs/lewm/bin/python3}"
PATCHWORLD_ROOT="${PATCHWORLD_ROOT:-${ROOT}}"
export STABLEWM_HOME="${STABLEWM_HOME:-${HOME}/.stable-wm}"
export PYTHONPATH="$PATCHWORLD_ROOT:$ROOT/src:${PYTHONPATH:-}"

INDUCED_SOURCE="${INDUCED_SOURCE:-oracle}"
if [[ "$INDUCED_SOURCE" == "oracle" ]]; then
  DATA_DIR="${DATA_DIR:-$ROOT/artifacts/visualpatchworld/data}"
  TOOL_DATA_DIR="${TOOL_DATA_DIR:-$ROOT/artifacts/visualpatchworld/lewm_tool_data}"
  TRAJ_JSONL="${TRAJ_JSONL:-$DATA_DIR/pusht_oracle_traj_train.jsonl}"
  TRAIN_GLOB="${TRAIN_GLOB:-$DATA_DIR/pusht_oracle_traj_train.jsonl}"
  OFFLINE_RESULTS_DIR="${OFFLINE_RESULTS_DIR:-$ROOT/artifacts/visualpatchworld/lewm_oracle_modular_offline_results}"
  OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/artifacts/visualpatchworld/lewm_oracle_unified_acquisition_refit_results}"
  INDUCTION_OFFLINE="${INDUCTION_OFFLINE:-pusht_modular_gated_mlp_oracle_offline}"
  INDUCTION_REFIT="${INDUCTION_REFIT:-pusht_modular_gated_mlp_oracle_contact_refit}"
  BASELINE_REPORT="${BASELINE_REPORT:-$REPORT_DIR/pusht_planning_oracle_contact_refit_simverify_n50.json}"
  INDUCED_REPORT="${INDUCED_REPORT:-$REPORT_DIR/pusht_planning_unified_acquisition_refit_n50.json}"
  ACQUIRED_JSONL="${ACQUIRED_JSONL:-$TOOL_DATA_DIR/unified_acquired_oracle_transitions.jsonl}"
else
  DATA_DIR="${DATA_DIR:-$ROOT/artifacts/visualpatchworld/lewm_tool_data}"
  TRAJ_JSONL="${TRAJ_JSONL:-$DATA_DIR/pusht_tool_traj_train.jsonl}"
  TRAIN_GLOB="${TRAIN_GLOB:-$DATA_DIR/pusht_tool_traj_train.jsonl}"
  OFFLINE_RESULTS_DIR="${OFFLINE_RESULTS_DIR:-$ROOT/artifacts/visualpatchworld/lewm_tool_modular_offline_results}"
  OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/artifacts/visualpatchworld/lewm_tool_unified_acquisition_refit_results}"
  INDUCTION_OFFLINE="${INDUCTION_OFFLINE:-pusht_modular_gated_mlp_offline}"
  INDUCTION_REFIT="${INDUCTION_REFIT:-pusht_modular_gated_mlp_contact_refit}"
  BASELINE_REPORT="${BASELINE_REPORT:-$REPORT_DIR/pusht_planning_gated_mlp_weighted_simverify_cem_mpc_fair_n50.json}"
  INDUCED_REPORT="${INDUCED_REPORT:-$REPORT_DIR/pusht_planning_unified_acquisition_refit_n50.json}"
  ACQUIRED_JSONL="${ACQUIRED_JSONL:-$DATA_DIR/unified_acquired_transitions.jsonl}"
fi

REPORT_DIR="${REPORT_DIR:-$ROOT/artifacts/visualpatchworld/reports}"
BASE_RESULTS_DIR="${BASE_RESULTS_DIR:-$OFFLINE_RESULTS_DIR}"
FALLBACK_BASE_DIR="${FALLBACK_BASE_DIR:-$ROOT/artifacts/visualpatchworld/lewm_tool_scaffold_fit_gated_mlp_weighted_results}"

MAX_TRAIN="${MAX_TRAIN:-16}"
MAX_EVAL="${MAX_EVAL:-4}"
MAX_EXTRA_FRACTION="${MAX_EXTRA_FRACTION:-0.4}"
CE_MODULES="${CE_MODULES:-contact,contact_angle}"
EXCLUDE_CE_MODULES="${EXCLUDE_CE_MODULES:-contact_gate}"

TOP_K="${TOP_K:-200}"
NUM_EXPLORE="${NUM_EXPLORE:-96}"
NUM_SUPERVISION="${NUM_SUPERVISION:-400}"
PLANNING_NUM_EVAL="${PLANNING_NUM_EVAL:-16}"
REGRET_WEIGHT="${REGRET_WEIGHT:-1.5}"

RUN_OFFLINE="${RUN_OFFLINE:-0}"
RUN_PHYSICS="${RUN_PHYSICS:-1}"
RUN_PLANNING="${RUN_PLANNING:-1}"
SIM_VERIFY_FRAC="${SIM_VERIFY_FRAC:-0.6}"
MIN_SUCCESS_GAIN="${MIN_SUCCESS_GAIN:-0.0}"
MAX_GOAL_DISTANCE_REGRESSION="${MAX_GOAL_DISTANCE_REGRESSION:-3.0}"

PHYSICS_REPORT="${PHYSICS_REPORT:-$REPORT_DIR/pusht_physics_rules_unified_acquisition_refit.json}"
COMPARISON_REPORT="${COMPARISON_REPORT:-$REPORT_DIR/pusht_planning_fair_comparison_unified_acquisition_refit_n50.json}"
ACCEPTANCE_REPORT="${ACCEPTANCE_REPORT:-$REPORT_DIR/unified_acquisition_acceptance.json}"

log() {
  echo "[unified-acquisition][$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

resolve_base_dir() {
  local candidate="$1"
  if [[ -d "$candidate/tool/generated_models" ]] && compgen -G "$candidate/tool/generated_models/*.py" > /dev/null; then
    echo "$candidate"
    return 0
  fi
  if [[ -d "$candidate/oracle/generated_models" ]] && compgen -G "$candidate/oracle/generated_models/*.py" > /dev/null; then
    echo "$candidate"
    return 0
  fi
  if [[ -d "$candidate/generated_models" ]] && compgen -G "$candidate/generated_models/*.py" > /dev/null; then
    echo "$candidate"
    return 0
  fi
  return 1
}

if [[ "$RUN_OFFLINE" == "1" ]]; then
  log "Phase 0: offline base induction (${INDUCTION_OFFLINE})"
  "$PYTHON" "$ROOT/scripts/visualpatchworld/induce_world_model.py" \
    --source "$INDUCED_SOURCE" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OFFLINE_RESULTS_DIR" \
    --max_train "$MAX_TRAIN" \
    --max_eval "$MAX_EVAL" \
    --induction_strategy "$INDUCTION_OFFLINE" \
    --max_refine_rounds 0
  BASE_RESULTS_DIR="$OFFLINE_RESULTS_DIR"
elif ! resolve_base_dir "$BASE_RESULTS_DIR" >/dev/null; then
  log "Base model missing; falling back to $FALLBACK_BASE_DIR"
  BASE_RESULTS_DIR="$FALLBACK_BASE_DIR"
fi

BASE_RESULTS_DIR="$(resolve_base_dir "$BASE_RESULTS_DIR" || true)"
if [[ -z "${BASE_RESULTS_DIR:-}" ]]; then
  echo "No base model found under $BASE_RESULTS_DIR or fallback $FALLBACK_BASE_DIR" >&2
  exit 1
fi
log "Using base model from $BASE_RESULTS_DIR"

log "Phase 1: unified transition acquisition (top_k=${TOP_K})"
"$PYTHON" "$ROOT/scripts/visualpatchworld/acquire_transitions.py" \
  --source "$INDUCED_SOURCE" \
  --results_dir "$BASE_RESULTS_DIR" \
  --trajectories_jsonl "$TRAJ_JSONL" \
  --top_k "$TOP_K" \
  --num_explore "$NUM_EXPLORE" \
  --num_supervision "$NUM_SUPERVISION" \
  --planning_num_eval "$PLANNING_NUM_EVAL" \
  --regret_weight "$REGRET_WEIGHT" \
  --output "$ACQUIRED_JSONL"

log "Phase 2: modular contact refit (max_extra_fraction=${MAX_EXTRA_FRACTION})"
"$PYTHON" "$ROOT/scripts/visualpatchworld/induce_world_model.py" \
  --source "$INDUCED_SOURCE" \
  --data_dir "$DATA_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --max_train "$MAX_TRAIN" \
  --max_eval "$MAX_EVAL" \
  --induction_strategy "$INDUCTION_REFIT" \
  --base_results_dir "$BASE_RESULTS_DIR" \
  --counterexample_store "$ACQUIRED_JSONL" \
  --max_extra_fraction "$MAX_EXTRA_FRACTION" \
  --counterexample_modules "$CE_MODULES" \
  --exclude_counterexample_modules "$EXCLUDE_CE_MODULES" \
  --max_refine_rounds 0

if [[ "$RUN_PHYSICS" == "1" ]]; then
  log "Phase 3a: physics-rule audit"
  "$PYTHON" "$ROOT/scripts/visualpatchworld/evaluate_physics_rules.py" \
    --trajectories_glob "$TRAIN_GLOB" \
    --source "$INDUCED_SOURCE" \
    --results_dir "$OUTPUT_DIR" \
    --output "$PHYSICS_REPORT"
fi

if [[ "$RUN_PLANNING" == "1" ]]; then
  log "Phase 3b: fair planning eval (sim_verify_frac=${SIM_VERIFY_FRAC})"
  INDUCED_RESULTS_DIR="$OUTPUT_DIR" \
  INDUCED_SOURCE="$INDUCED_SOURCE" \
  SIM_VERIFY_FRAC="$SIM_VERIFY_FRAC" \
  INDUCED_REPORT="$INDUCED_REPORT" \
  COMPARISON_REPORT="$COMPARISON_REPORT" \
  bash "$ROOT/scripts/visualpatchworld/run_fair_planning_comparison.sh"

  log "Phase 3c: planning-gated acceptance"
  ACCEPT_JSON="$("$PYTHON" - <<'PY' "$BASELINE_REPORT" "$INDUCED_REPORT" "$MIN_SUCCESS_GAIN" "$MAX_GOAL_DISTANCE_REGRESSION" "$ACCEPTANCE_REPORT"
import json
import sys
from datetime import datetime
from pathlib import Path

from visualpatchworld.active_dynamics_induction import evaluate_acceptance

baseline_path, candidate_path, min_gain, max_regress, out_path = sys.argv[1:6]
baseline_raw = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
candidate_raw = json.loads(Path(candidate_path).read_text(encoding="utf-8"))
baseline = baseline_raw.get("induced") or baseline_raw
candidate = candidate_raw.get("induced") or candidate_raw
result = evaluate_acceptance(
    baseline_report=baseline,
    candidate_report=candidate,
    min_success_gain=float(min_gain),
    max_goal_distance_regression=float(max_regress),
)
payload = {
    "timestamp": datetime.now().isoformat(),
    "baseline_report": baseline_path,
    "candidate_report": candidate_path,
    "acceptance": result,
}
Path(out_path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2, sort_keys=True))
PY
)"
  log "Acceptance gate -> ${ACCEPT_JSON}"
fi

log "Done. Unified acquisition refit results: $OUTPUT_DIR"
