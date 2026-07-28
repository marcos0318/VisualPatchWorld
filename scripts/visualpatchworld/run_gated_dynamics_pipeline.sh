#!/usr/bin/env bash
# Induce contact-gated and gated+MLP dynamics, audit physics rules, and run fair planning.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-${CONDA_PREFIX:-}/envs/lewm/bin/python3}"
PATCHWORLD_ROOT="${PATCHWORLD_ROOT:-${ROOT}}"
export STABLEWM_HOME="${STABLEWM_HOME:-${HOME}/.stable-wm}"
export PYTHONPATH="$PATCHWORLD_ROOT:$ROOT/src:${PYTHONPATH:-}"

DATA_DIR="${DATA_DIR:-$ROOT/artifacts/visualpatchworld/lewm_tool_data}"
REPORT_DIR="${REPORT_DIR:-$ROOT/artifacts/visualpatchworld/reports}"
TRAIN_GLOB="${TRAIN_GLOB:-$DATA_DIR/pusht_tool_traj_train.jsonl}"
MAX_TRAIN="${MAX_TRAIN:-16}"
MAX_EVAL="${MAX_EVAL:-4}"
SIM_VERIFY_FRAC="${SIM_VERIFY_FRAC:-0.2}"
RUN_PLANNING="${RUN_PLANNING:-1}"

GATED_DIR="${GATED_DIR:-$ROOT/artifacts/visualpatchworld/lewm_tool_scaffold_fit_gated_results}"
GATED_MLP_DIR="${GATED_MLP_DIR:-$ROOT/artifacts/visualpatchworld/lewm_tool_scaffold_fit_gated_mlp_results}"

log() {
  echo "[gated-dynamics][$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

log "Step 1: induce contact-gated model"
"$PYTHON" "$ROOT/scripts/visualpatchworld/induce_world_model.py" \
  --source tool \
  --data_dir "$DATA_DIR" \
  --output_dir "$GATED_DIR" \
  --max_train "$MAX_TRAIN" \
  --max_eval "$MAX_EVAL" \
  --induction_strategy pusht_scaffold_fit_contact_gated \
  --max_refine_rounds 0

log "Step 2: induce contact-gated + MLP residual model"
"$PYTHON" "$ROOT/scripts/visualpatchworld/induce_world_model.py" \
  --source tool \
  --data_dir "$DATA_DIR" \
  --output_dir "$GATED_MLP_DIR" \
  --max_train "$MAX_TRAIN" \
  --max_eval "$MAX_EVAL" \
  --induction_strategy pusht_scaffold_fit_contact_gated_mlp \
  --max_refine_rounds 0

log "Step 3: physics-rule audit (gated)"
"$PYTHON" "$ROOT/scripts/visualpatchworld/evaluate_physics_rules.py" \
  --trajectories_glob "$TRAIN_GLOB" \
  --source tool \
  --results_dir "$GATED_DIR" \
  --output "$REPORT_DIR/pusht_physics_rules_gated.json"

log "Step 4: physics-rule audit (gated + MLP)"
"$PYTHON" "$ROOT/scripts/visualpatchworld/evaluate_physics_rules.py" \
  --trajectories_glob "$TRAIN_GLOB" \
  --source tool \
  --results_dir "$GATED_MLP_DIR" \
  --output "$REPORT_DIR/pusht_physics_rules_gated_mlp.json"

if [[ "$RUN_PLANNING" == "1" ]]; then
  log "Step 5: fair planning (gated, sim_verify=${SIM_VERIFY_FRAC})"
  INDUCED_RESULTS_DIR="$GATED_DIR" \
  SIM_VERIFY_FRAC="$SIM_VERIFY_FRAC" \
  INDUCED_REPORT="$REPORT_DIR/pusht_planning_gated_cem_mpc_fair_n50.json" \
  COMPARISON_REPORT="$REPORT_DIR/pusht_planning_fair_comparison_gated_n50.json" \
  bash "$ROOT/scripts/visualpatchworld/run_fair_planning_comparison.sh"

  log "Step 6: fair planning (gated + MLP, sim_verify=${SIM_VERIFY_FRAC})"
  INDUCED_RESULTS_DIR="$GATED_MLP_DIR" \
  SIM_VERIFY_FRAC="$SIM_VERIFY_FRAC" \
  INDUCED_REPORT="$REPORT_DIR/pusht_planning_gated_mlp_cem_mpc_fair_n50.json" \
  COMPARISON_REPORT="$REPORT_DIR/pusht_planning_fair_comparison_gated_mlp_n50.json" \
  bash "$ROOT/scripts/visualpatchworld/run_fair_planning_comparison.sh"
fi

log "Done."
