#!/usr/bin/env bash
# Modular active contact refit:
#   Phase 0: offline base (or reuse OFFLINE_RESULTS_DIR)
#   Phase 1: mine sim counterexamples from current model
#   Phase 2: contact-only refit with capped module-tagged weights
#   Phase 3: physics audit + optional fair planning with acceptance gate
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-${CONDA_PREFIX:-}/envs/lewm/bin/python3}"
PATCHWORLD_ROOT="${PATCHWORLD_ROOT:-${ROOT}}"
export STABLEWM_HOME="${STABLEWM_HOME:-${HOME}/.stable-wm}"
export PYTHONPATH="$PATCHWORLD_ROOT:$ROOT/src:${PYTHONPATH:-}"

DATA_DIR="${DATA_DIR:-$ROOT/artifacts/visualpatchworld/lewm_tool_data}"
REPORT_DIR="${REPORT_DIR:-$ROOT/artifacts/visualpatchworld/reports}"
TRAIN_GLOB="${TRAIN_GLOB:-$DATA_DIR/pusht_tool_traj_train.jsonl}"
TRAJ_JSONL="${TRAJ_JSONL:-$TRAIN_GLOB}"

OFFLINE_RESULTS_DIR="${OFFLINE_RESULTS_DIR:-$ROOT/artifacts/visualpatchworld/lewm_tool_modular_offline_results}"
BASE_RESULTS_DIR="${BASE_RESULTS_DIR:-$OFFLINE_RESULTS_DIR}"
FALLBACK_BASE_DIR="${FALLBACK_BASE_DIR:-$ROOT/artifacts/visualpatchworld/lewm_tool_scaffold_fit_gated_mlp_weighted_results}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/artifacts/visualpatchworld/lewm_tool_modular_contact_refit_results}"

MAX_TRAIN="${MAX_TRAIN:-16}"
MAX_EVAL="${MAX_EVAL:-4}"
MAX_EXTRA_FRACTION="${MAX_EXTRA_FRACTION:-0.3}"
CE_MODULES="${CE_MODULES:-contact,contact_angle}"
EXCLUDE_CE_MODULES="${EXCLUDE_CE_MODULES:-contact_gate}"

NUM_EXPLORE="${NUM_EXPLORE:-96}"
TOP_K_CE="${TOP_K_CE:-120}"
ERROR_THRESHOLD="${ERROR_THRESHOLD:-5.0}"
RUN_OFFLINE="${RUN_OFFLINE:-1}"
RUN_PHYSICS="${RUN_PHYSICS:-1}"
RUN_PLANNING="${RUN_PLANNING:-0}"
SIM_VERIFY_FRAC="${SIM_VERIFY_FRAC:-0.4}"

CE_JSONL="${CE_JSONL:-$DATA_DIR/modular_contact_ce_transitions.jsonl}"
PHYSICS_REPORT="${PHYSICS_REPORT:-$REPORT_DIR/pusht_physics_rules_modular_contact_refit.json}"
BASELINE_REPORT="${BASELINE_REPORT:-$REPORT_DIR/pusht_planning_gated_mlp_weighted_simverify_cem_mpc_fair_n50.json}"
INDUCED_REPORT="${INDUCED_REPORT:-$REPORT_DIR/pusht_planning_modular_contact_refit_simverify_cem_mpc_fair_n50.json}"

log() {
  echo "[modular-contact-refit][$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

resolve_base_dir() {
  local candidate="$1"
  if [[ -d "$candidate/tool/generated_models" ]] && compgen -G "$candidate/tool/generated_models/*.py" > /dev/null; then
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
  log "Phase 0: offline base induction"
  OUTPUT_DIR="$OFFLINE_RESULTS_DIR" REPORT_DIR="$REPORT_DIR" \
  OFFLINE_RESULTS_DIR="$OFFLINE_RESULTS_DIR" \
  MAX_TRAIN="$MAX_TRAIN" MAX_EVAL="$MAX_EVAL" RUN_PHYSICS=0 \
  bash "$ROOT/scripts/visualpatchworld/induce_offline_gated_mlp.sh"
  BASE_RESULTS_DIR="$(resolve_base_dir "$OFFLINE_RESULTS_DIR" || true)"
  if [[ -z "${BASE_RESULTS_DIR:-}" ]]; then
    BASE_RESULTS_DIR="$OFFLINE_RESULTS_DIR"
  fi
elif ! resolve_base_dir "$BASE_RESULTS_DIR" >/dev/null; then
  log "Offline base missing; falling back to $FALLBACK_BASE_DIR"
  BASE_RESULTS_DIR="$FALLBACK_BASE_DIR"
fi

BASE_RESULTS_DIR="$(resolve_base_dir "$BASE_RESULTS_DIR" || true)"
if [[ -z "${BASE_RESULTS_DIR:-}" ]]; then
  echo "No base model found (fallback $FALLBACK_BASE_DIR also missing)" >&2
  exit 1
fi
log "Using base model from $BASE_RESULTS_DIR"

USE_UNIFIED_ACQUISITION="${USE_UNIFIED_ACQUISITION:-0}"
if [[ "$USE_UNIFIED_ACQUISITION" == "1" ]]; then
  log "Phase 1: unified transition acquisition"
  ACQUIRED_JSONL="${ACQUIRED_JSONL:-$DATA_DIR/unified_acquired_transitions.jsonl}"
  "$PYTHON" "$ROOT/scripts/visualpatchworld/acquire_transitions.py" \
    --results_dir "$BASE_RESULTS_DIR" \
    --trajectories_jsonl "$TRAJ_JSONL" \
    --top_k "${TOP_K_CE:-120}" \
    --num_explore "$NUM_EXPLORE" \
    --output "$ACQUIRED_JSONL"
  CE_JSONL="$ACQUIRED_JSONL"
else
  log "Phase 1: mine sim counterexamples"
  "$PYTHON" "$ROOT/scripts/visualpatchworld/collect_sim_counterexamples.py" \
    --results_dir "$BASE_RESULTS_DIR" \
    --trajectories_jsonl "$TRAJ_JSONL" \
    --num_explore "$NUM_EXPLORE" \
    --top_k "$TOP_K_CE" \
    --error_threshold "$ERROR_THRESHOLD" \
    --output "$CE_JSONL"
fi

log "Phase 2: contact-only modular refit (max_extra_fraction=${MAX_EXTRA_FRACTION})"
"$PYTHON" "$ROOT/scripts/visualpatchworld/induce_world_model.py" \
  --source tool \
  --data_dir "$DATA_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --max_train "$MAX_TRAIN" \
  --max_eval "$MAX_EVAL" \
  --induction_strategy pusht_modular_gated_mlp_contact_refit \
  --base_results_dir "$BASE_RESULTS_DIR" \
  --counterexample_store "$CE_JSONL" \
  --max_extra_fraction "$MAX_EXTRA_FRACTION" \
  --counterexample_modules "$CE_MODULES" \
  --exclude_counterexample_modules "$EXCLUDE_CE_MODULES" \
  --max_refine_rounds 0

if [[ "$RUN_PHYSICS" == "1" ]]; then
  log "Phase 3a: physics-rule audit"
  "$PYTHON" "$ROOT/scripts/visualpatchworld/evaluate_physics_rules.py" \
    --trajectories_glob "$TRAIN_GLOB" \
    --source tool \
    --results_dir "$OUTPUT_DIR" \
    --output "$PHYSICS_REPORT"
fi

if [[ "$RUN_PLANNING" == "1" ]]; then
  log "Phase 3b: fair planning eval (sim_verify_frac=${SIM_VERIFY_FRAC})"
  INDUCED_RESULTS_DIR="$OUTPUT_DIR" \
  SIM_VERIFY_FRAC="$SIM_VERIFY_FRAC" \
  INDUCED_REPORT="$INDUCED_REPORT" \
  COMPARISON_REPORT="$REPORT_DIR/pusht_planning_fair_comparison_modular_contact_refit_n50.json" \
  bash "$ROOT/scripts/visualpatchworld/run_fair_planning_comparison.sh"
fi

log "Done. Contact refit results: $OUTPUT_DIR"
