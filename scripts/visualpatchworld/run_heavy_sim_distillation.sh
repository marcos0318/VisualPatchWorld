#!/usr/bin/env bash
# Heavy simulator distillation: all exported oracle trajectories + large sim
# contact supervision pool, then compositional contact refit and planning eval.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-${CONDA_PREFIX:-}/envs/lewm/bin/python3}"
PATCHWORLD_ROOT="${PATCHWORLD_ROOT:-${ROOT}}"
export STABLEWM_HOME="${STABLEWM_HOME:-${HOME}/.stable-wm}"
export PYTHONPATH="$PATCHWORLD_ROOT:$ROOT/src:${PYTHONPATH:-}"

DATA_DIR="${DATA_DIR:-$ROOT/artifacts/visualpatchworld/data}"
TOOL_DATA_DIR="${TOOL_DATA_DIR:-$ROOT/artifacts/visualpatchworld/lewm_tool_data}"
REPORT_DIR="${REPORT_DIR:-$ROOT/artifacts/visualpatchworld/reports}"

MAX_EXPORT_EPISODES="${MAX_EXPORT_EPISODES:-512}"
MAX_TRAIN="${MAX_TRAIN:-0}"
MAX_EVAL="${MAX_EVAL:-16}"
NUM_SUPERVISION="${NUM_SUPERVISION:-5000}"
MAX_EXTRA_FRACTION="${MAX_EXTRA_FRACTION:-1.0}"
SIM_VERIFY_FRAC="${SIM_VERIFY_FRAC:-0.6}"
RUN_PLANNING="${RUN_PLANNING:-1}"

OFFLINE_DIR="${OFFLINE_DIR:-$ROOT/artifacts/visualpatchworld/lewm_heavy_distill_offline_results}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/artifacts/visualpatchworld/lewm_heavy_distill_results}"
SUPERVISION_JSONL="${SUPERVISION_JSONL:-$TOOL_DATA_DIR/heavy_sim_contact_supervision.jsonl}"
INDUCED_REPORT="${INDUCED_REPORT:-$REPORT_DIR/pusht_planning_heavy_distill_n50.json}"

log() {
  echo "[heavy-distill][$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

log "Export oracle trajectories (max_episodes=${MAX_EXPORT_EPISODES})"
"$PYTHON" "$ROOT/scripts/visualpatchworld/export_pusht.py" \
  --source oracle \
  --output_dir "$DATA_DIR" \
  --max_episodes "$MAX_EXPORT_EPISODES" \
  --step_stride 5

log "Compositional offline on all train trajectories (max_train=${MAX_TRAIN})"
"$PYTHON" "$ROOT/scripts/visualpatchworld/induce_world_model.py" \
  --source oracle \
  --data_dir "$DATA_DIR" \
  --output_dir "$OFFLINE_DIR" \
  --max_train "$MAX_TRAIN" \
  --max_eval "$MAX_EVAL" \
  --induction_strategy pusht_compositional_gated_mlp_oracle_offline \
  --max_refine_rounds 0

log "Collect heavy sim contact supervision (num_samples=${NUM_SUPERVISION})"
"$PYTHON" "$ROOT/scripts/visualpatchworld/collect_sim_contact_supervision.py" \
  --num_samples "$NUM_SUPERVISION" \
  --output "$SUPERVISION_JSONL"

log "Compositional contact refit (max_extra_fraction=${MAX_EXTRA_FRACTION})"
"$PYTHON" "$ROOT/scripts/visualpatchworld/induce_world_model.py" \
  --source oracle \
  --data_dir "$DATA_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --max_train "$MAX_TRAIN" \
  --max_eval "$MAX_EVAL" \
  --induction_strategy pusht_compositional_gated_mlp_oracle_contact_refit \
  --base_results_dir "$OFFLINE_DIR" \
  --counterexample_store "$SUPERVISION_JSONL" \
  --max_extra_fraction "$MAX_EXTRA_FRACTION" \
  --counterexample_modules "contact,contact_angle" \
  --exclude_counterexample_modules "contact_gate" \
  --max_refine_rounds 0

log "Physics audit"
"$PYTHON" "$ROOT/scripts/visualpatchworld/evaluate_physics_rules.py" \
  --trajectories_glob "$DATA_DIR/pusht_oracle_traj_train.jsonl" \
  --source oracle \
  --results_dir "$OUTPUT_DIR" \
  --output "$REPORT_DIR/pusht_physics_rules_heavy_distill.json"

if [[ "$RUN_PLANNING" == "1" ]]; then
  log "Fair planning (sim_verify_frac=${SIM_VERIFY_FRAC})"
  INDUCED_RESULTS_DIR="$OUTPUT_DIR" \
  INDUCED_SOURCE=oracle \
  SIM_VERIFY_FRAC="$SIM_VERIFY_FRAC" \
  INDUCED_REPORT="$INDUCED_REPORT" \
  COMPARISON_REPORT="$REPORT_DIR/pusht_planning_fair_comparison_heavy_distill_n50.json" \
  bash "$ROOT/scripts/visualpatchworld/run_fair_planning_comparison.sh"
fi

log "Done. Results: $OUTPUT_DIR"
