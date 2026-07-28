#!/usr/bin/env bash
# Compositional dynamics: explicit AgentModule + ContactModule + ContactGate.
# Uses oracle trajectories + sim contact supervision (same data as oracle pipeline).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-${CONDA_PREFIX:-}/envs/lewm/bin/python3}"
PATCHWORLD_ROOT="${PATCHWORLD_ROOT:-${ROOT}}"
export STABLEWM_HOME="${STABLEWM_HOME:-${HOME}/.stable-wm}"
export PYTHONPATH="$PATCHWORLD_ROOT:$ROOT/src:${PYTHONPATH:-}"

ORACLE_DATA_DIR="${ORACLE_DATA_DIR:-$ROOT/artifacts/visualpatchworld/data}"
TOOL_DATA_DIR="${TOOL_DATA_DIR:-$ROOT/artifacts/visualpatchworld/lewm_tool_data}"
REPORT_DIR="${REPORT_DIR:-$ROOT/artifacts/visualpatchworld/reports}"

OFFLINE_RESULTS_DIR="${OFFLINE_RESULTS_DIR:-$ROOT/artifacts/visualpatchworld/lewm_compositional_oracle_offline_results}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/artifacts/visualpatchworld/lewm_compositional_oracle_contact_refit_results}"

MAX_TRAIN="${MAX_TRAIN:-16}"
MAX_EVAL="${MAX_EVAL:-4}"
NUM_CONTACT_SAMPLES="${NUM_CONTACT_SAMPLES:-800}"
MAX_EXTRA_FRACTION="${MAX_EXTRA_FRACTION:-0.5}"
SIM_VERIFY_FRAC="${SIM_VERIFY_FRAC:-0.6}"
RUN_PLANNING="${RUN_PLANNING:-1}"

SUPERVISION_JSONL="${SUPERVISION_JSONL:-$TOOL_DATA_DIR/sim_contact_supervision.jsonl}"
TRAIN_GLOB="${ORACLE_DATA_DIR}/pusht_oracle_traj_train.jsonl"

log() {
  echo "[compositional-oracle][$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

log "Phase 0: compositional oracle offline fit"
"$PYTHON" "$ROOT/scripts/visualpatchworld/induce_world_model.py" \
  --source oracle \
  --data_dir "$ORACLE_DATA_DIR" \
  --output_dir "$OFFLINE_RESULTS_DIR" \
  --max_train "$MAX_TRAIN" \
  --max_eval "$MAX_EVAL" \
  --induction_strategy pusht_compositional_gated_mlp_oracle_offline \
  --max_refine_rounds 0

if [[ ! -f "$SUPERVISION_JSONL" ]]; then
  log "Phase 1: collect sim contact supervision"
  "$PYTHON" "$ROOT/scripts/visualpatchworld/collect_sim_contact_supervision.py" \
    --num_samples "$NUM_CONTACT_SAMPLES" \
    --output "$SUPERVISION_JSONL"
else
  log "Phase 1: reusing $SUPERVISION_JSONL"
fi

log "Phase 2: compositional contact refit"
"$PYTHON" "$ROOT/scripts/visualpatchworld/induce_world_model.py" \
  --source oracle \
  --data_dir "$ORACLE_DATA_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --max_train "$MAX_TRAIN" \
  --max_eval "$MAX_EVAL" \
  --induction_strategy pusht_compositional_gated_mlp_oracle_contact_refit \
  --base_results_dir "$OFFLINE_RESULTS_DIR" \
  --counterexample_store "$SUPERVISION_JSONL" \
  --max_extra_fraction "$MAX_EXTRA_FRACTION" \
  --max_refine_rounds 0

log "Phase 3a: physics audit"
"$PYTHON" "$ROOT/scripts/visualpatchworld/evaluate_physics_rules.py" \
  --trajectories_glob "$TRAIN_GLOB" \
  --source oracle \
  --results_dir "$OUTPUT_DIR" \
  --output "$REPORT_DIR/pusht_physics_rules_compositional_oracle_contact_refit.json"

if [[ "$RUN_PLANNING" == "1" ]]; then
  log "Phase 3b: fair planning (sim_verify_frac=${SIM_VERIFY_FRAC})"
  INDUCED_RESULTS_DIR="$OUTPUT_DIR" \
  INDUCED_SOURCE=oracle \
  SIM_VERIFY_FRAC="$SIM_VERIFY_FRAC" \
  INDUCED_REPORT="$REPORT_DIR/pusht_planning_compositional_oracle_contact_refit_n50.json" \
  COMPARISON_REPORT="$REPORT_DIR/pusht_planning_fair_comparison_compositional_oracle_n50.json" \
  bash "$ROOT/scripts/visualpatchworld/run_fair_planning_comparison.sh"
fi

log "Done. Results: $OUTPUT_DIR"
