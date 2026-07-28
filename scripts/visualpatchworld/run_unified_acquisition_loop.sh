#!/usr/bin/env bash
# Multi-round unified acquisition loop toward higher planning success.
#
# Each round:
#   1) unified acquisition (sim CE + planning failures + supervision)
#   2) compositional/modular contact refit (+ optional agent refit)
#   3) physics audit + fair planning
#   4) accept if planning improves; carry best model to next round
#
# Stops when success_rate >= TARGET_SUCCESS_RATE or MAX_ROUNDS reached.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-${CONDA_PREFIX:-}/envs/lewm/bin/python3}"
PATCHWORLD_ROOT="${PATCHWORLD_ROOT:-${ROOT}}"
export STABLEWM_HOME="${STABLEWM_HOME:-${HOME}/.stable-wm}"
export PYTHONPATH="$PATCHWORLD_ROOT:$ROOT/src:${PYTHONPATH:-}"

INDUCED_SOURCE="${INDUCED_SOURCE:-oracle}"
COMPOSITIONAL="${COMPOSITIONAL:-1}"
REPORT_DIR="${REPORT_DIR:-$ROOT/artifacts/visualpatchworld/reports}"
TOOL_DATA_DIR="${TOOL_DATA_DIR:-$ROOT/artifacts/visualpatchworld/lewm_tool_data}"

MAX_ROUNDS="${MAX_ROUNDS:-5}"
TARGET_SUCCESS_RATE="${TARGET_SUCCESS_RATE:-80}"
MAX_TRAIN="${MAX_TRAIN:-64}"
MAX_EVAL="${MAX_EVAL:-8}"
MAX_EXTRA_FRACTION="${MAX_EXTRA_FRACTION:-0.6}"

TOP_K="${TOP_K:-300}"
NUM_EXPLORE="${NUM_EXPLORE:-128}"
NUM_SUPERVISION="${NUM_SUPERVISION:-1200}"
PLANNING_NUM_EVAL="${PLANNING_NUM_EVAL:-50}"
PLANNING_NUM_SAMPLES="${PLANNING_NUM_SAMPLES:-128}"
PLANNING_CEM_ITERS="${PLANNING_CEM_ITERS:-5}"
REGRET_WEIGHT="${REGRET_WEIGHT:-2.0}"
SIM_VERIFY_FRAC="${SIM_VERIFY_FRAC:-0.8}"
MIN_SUCCESS_GAIN="${MIN_SUCCESS_GAIN:-0.0}"
MAX_GOAL_DISTANCE_REGRESSION="${MAX_GOAL_DISTANCE_REGRESSION:-5.0}"

RUN_EXPORT="${RUN_EXPORT:-1}"
MAX_EXPORT_EPISODES="${MAX_EXPORT_EPISODES:-256}"
RUN_OFFLINE="${RUN_OFFLINE:-1}"
RUN_AGENT_REFIT="${RUN_AGENT_REFIT:-1}"

OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/artifacts/visualpatchworld/lewm_unified_acquisition_loop_results}"
LOOP_LOG="${LOOP_LOG:-$REPORT_DIR/unified_acquisition_loop.jsonl}"

if [[ "$INDUCED_SOURCE" == "oracle" ]]; then
  DATA_DIR="${DATA_DIR:-$ROOT/artifacts/visualpatchworld/data}"
  TRAJ_JSONL="${TRAJ_JSONL:-$DATA_DIR/pusht_oracle_traj_train.jsonl}"
  TRAIN_GLOB="${TRAIN_GLOB:-$DATA_DIR/pusht_oracle_traj_train.jsonl}"
  if [[ "$COMPOSITIONAL" == "1" ]]; then
    OFFLINE_RESULTS_DIR="${OFFLINE_RESULTS_DIR:-$OUTPUT_ROOT/offline_compositional}"
    INDUCTION_OFFLINE="${INDUCTION_OFFLINE:-pusht_compositional_gated_mlp_oracle_offline}"
    INDUCTION_CONTACT_REFIT="${INDUCTION_CONTACT_REFIT:-pusht_compositional_gated_mlp_oracle_contact_refit}"
  else
    OFFLINE_RESULTS_DIR="${OFFLINE_RESULTS_DIR:-$OUTPUT_ROOT/offline_modular}"
    INDUCTION_OFFLINE="${INDUCTION_OFFLINE:-pusht_modular_gated_mlp_oracle_offline}"
    INDUCTION_CONTACT_REFIT="${INDUCTION_CONTACT_REFIT:-pusht_modular_gated_mlp_oracle_contact_refit}"
  fi
  INDUCTION_AGENT_REFIT="${INDUCTION_AGENT_REFIT:-pusht_modular_gated_mlp_agent_refit}"
  GT_REPORT="${GT_REPORT:-$REPORT_DIR/pusht_planning_simulator_cem_mpc_ground_truth_n50.json}"
else
  DATA_DIR="${DATA_DIR:-$TOOL_DATA_DIR}"
  TRAJ_JSONL="${TRAJ_JSONL:-$DATA_DIR/pusht_tool_traj_train.jsonl}"
  TRAIN_GLOB="${TRAIN_GLOB:-$DATA_DIR/pusht_tool_traj_train.jsonl}"
  OFFLINE_RESULTS_DIR="${OFFLINE_RESULTS_DIR:-$OUTPUT_ROOT/offline_modular}"
  INDUCTION_OFFLINE="${INDUCTION_OFFLINE:-pusht_modular_gated_mlp_offline}"
  INDUCTION_CONTACT_REFIT="${INDUCTION_CONTACT_REFIT:-pusht_modular_gated_mlp_contact_refit}"
  INDUCTION_AGENT_REFIT="${INDUCTION_AGENT_REFIT:-pusht_modular_gated_mlp_agent_refit}"
  GT_REPORT="${GT_REPORT:-$REPORT_DIR/pusht_planning_gated_mlp_weighted_simverify_cem_mpc_fair_n50.json}"
fi

CE_MODULES="${CE_MODULES:-contact,contact_angle}"
EXCLUDE_CE_MODULES="${EXCLUDE_CE_MODULES:-contact_gate}"
# Acquisition pool also drops contact_gate rows (not patchable via contact refit).
ACQUISITION_EXCLUDE_MODULES="${ACQUISITION_EXCLUDE_MODULES:-contact_gate}"
AGENT_CE_MODULES="${AGENT_CE_MODULES:-agent,angle}"

log() {
  echo "[unified-loop][$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

append_loop_record() {
  echo "$1" >> "$LOOP_LOG"
}

resolve_base_dir() {
  local candidate="$1"
  for sub in tool oracle ""; do
    if [[ -n "$sub" ]]; then
      if [[ -d "$candidate/$sub/generated_models" ]] && compgen -G "$candidate/$sub/generated_models/*.py" > /dev/null; then
        echo "$candidate"
        return 0
      fi
    else
      if [[ -d "$candidate/generated_models" ]] && compgen -G "$candidate/generated_models/*.py" > /dev/null; then
        echo "$candidate"
        return 0
      fi
    fi
  done
  return 1
}

read_success_rate() {
  local report_path="$1"
  "$PYTHON" - <<'PY' "$report_path"
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
if not p.is_file():
    print("0")
    raise SystemExit(0)
report = json.loads(p.read_text(encoding="utf-8"))
induced = report.get("induced") or report
print(induced.get("success_rate", 0))
PY
}

read_goal_distance() {
  local report_path="$1"
  "$PYTHON" - <<'PY' "$report_path"
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
if not p.is_file():
    print("inf")
    raise SystemExit(0)
report = json.loads(p.read_text(encoding="utf-8"))
induced = report.get("induced") or report
print(induced.get("mean_goal_distance", "inf"))
PY
}

recommended_patch_from_diagnostics() {
  local diag_path="$1"
  "$PYTHON" - <<'PY' "$diag_path"
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
if not p.is_file():
    print("refit_contact_head")
    raise SystemExit(0)
diag = json.loads(p.read_text(encoding="utf-8"))
print(diag.get("recommended_patch", "refit_contact_head"))
PY
}

mkdir -p "$OUTPUT_ROOT" "$(dirname "$LOOP_LOG")"

if [[ "$RUN_EXPORT" == "1" && "$INDUCED_SOURCE" == "oracle" ]]; then
  log "Phase 0a: export oracle trajectories (max_episodes=${MAX_EXPORT_EPISODES})"
  "$PYTHON" "$ROOT/scripts/visualpatchworld/export_pusht.py" \
    --source oracle \
    --output_dir "$DATA_DIR" \
    --max_episodes "$MAX_EXPORT_EPISODES" \
    --step_stride 5
fi

if [[ "$RUN_OFFLINE" == "1" ]]; then
  log "Phase 0b: offline base (${INDUCTION_OFFLINE}, max_train=${MAX_TRAIN})"
  "$PYTHON" "$ROOT/scripts/visualpatchworld/induce_world_model.py" \
    --source "$INDUCED_SOURCE" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OFFLINE_RESULTS_DIR" \
    --max_train "$MAX_TRAIN" \
    --max_eval "$MAX_EVAL" \
    --induction_strategy "$INDUCTION_OFFLINE" \
    --max_refine_rounds 0
fi

CURRENT_RESULTS_DIR="$(resolve_base_dir "$OFFLINE_RESULTS_DIR" || true)"
if [[ -z "${CURRENT_RESULTS_DIR:-}" ]]; then
  CURRENT_RESULTS_DIR="$ROOT/artifacts/visualpatchworld/lewm_compositional_oracle_contact_refit_results"
  log "Offline missing; seeding from $CURRENT_RESULTS_DIR"
fi
CURRENT_RESULTS_DIR="$(resolve_base_dir "$CURRENT_RESULTS_DIR" || true)"

BEST_RESULTS_DIR="$CURRENT_RESULTS_DIR"
BEST_REPORT=""
BEST_SUCCESS="0"

log "Starting loop from $CURRENT_RESULTS_DIR (target success=${TARGET_SUCCESS_RATE}%)"

for ROUND in $(seq 1 "$MAX_ROUNDS"); do
  ROUND_DIR="$OUTPUT_ROOT/round_${ROUND}"
  ACQUIRED_JSONL="$ROUND_DIR/unified_acquired_transitions.jsonl"
  DIAG_JSON="$ROUND_DIR/acquisition_diagnostics.json"
  PHYSICS_REPORT="$ROUND_DIR/physics_rules.json"
  INDUCED_REPORT="$ROUND_DIR/planning_n50.json"
  COMPARISON_REPORT="$ROUND_DIR/planning_comparison_n50.json"
  ACCEPTANCE_REPORT="$ROUND_DIR/acceptance.json"
  mkdir -p "$ROUND_DIR"

  log "Round ${ROUND}/${MAX_ROUNDS}: unified acquisition (top_k=${TOP_K})"
  "$PYTHON" "$ROOT/scripts/visualpatchworld/acquire_transitions.py" \
    --source "$INDUCED_SOURCE" \
    --results_dir "$CURRENT_RESULTS_DIR" \
    --trajectories_jsonl "$TRAJ_JSONL" \
    --top_k "$TOP_K" \
    --num_explore "$NUM_EXPLORE" \
    --num_supervision "$NUM_SUPERVISION" \
    --planning_num_eval "$PLANNING_NUM_EVAL" \
    --planning_num_samples "$PLANNING_NUM_SAMPLES" \
    --planning_cem_iters "$PLANNING_CEM_ITERS" \
    --regret_weight "$REGRET_WEIGHT" \
    --output "$ACQUIRED_JSONL" \
    --diagnostics_output "$DIAG_JSON"

  RECOMMENDED="$(recommended_patch_from_diagnostics "$DIAG_JSON")"
  log "Round ${ROUND}: recommended_patch=${RECOMMENDED}"

  log "Round ${ROUND}: contact refit -> $ROUND_DIR/contact"
  "$PYTHON" "$ROOT/scripts/visualpatchworld/induce_world_model.py" \
    --source "$INDUCED_SOURCE" \
    --data_dir "$DATA_DIR" \
    --output_dir "$ROUND_DIR/contact" \
    --max_train "$MAX_TRAIN" \
    --max_eval "$MAX_EVAL" \
    --induction_strategy "$INDUCTION_CONTACT_REFIT" \
    --base_results_dir "$CURRENT_RESULTS_DIR" \
    --counterexample_store "$ACQUIRED_JSONL" \
    --max_extra_fraction "$MAX_EXTRA_FRACTION" \
    --counterexample_modules "$CE_MODULES" \
    --exclude_counterexample_modules "$EXCLUDE_CE_MODULES" \
    --max_refine_rounds 0

  CANDIDATE_DIR="$(resolve_base_dir "$ROUND_DIR/contact" || true)"
  if [[ "$RUN_AGENT_REFIT" == "1" && "$COMPOSITIONAL" != "1" && "$RECOMMENDED" == *"agent"* ]]; then
    log "Round ${ROUND}: agent refit -> $ROUND_DIR/agent"
    "$PYTHON" "$ROOT/scripts/visualpatchworld/induce_world_model.py" \
      --source "$INDUCED_SOURCE" \
      --data_dir "$DATA_DIR" \
      --output_dir "$ROUND_DIR/agent" \
      --max_train "$MAX_TRAIN" \
      --max_eval "$MAX_EVAL" \
      --induction_strategy "$INDUCTION_AGENT_REFIT" \
      --base_results_dir "$CANDIDATE_DIR" \
      --counterexample_store "$ACQUIRED_JSONL" \
      --max_extra_fraction "$MAX_EXTRA_FRACTION" \
      --counterexample_modules "$AGENT_CE_MODULES" \
      --exclude_counterexample_modules "$EXCLUDE_CE_MODULES" \
      --max_refine_rounds 0
    CANDIDATE_DIR="$(resolve_base_dir "$ROUND_DIR/agent" || true)"
  fi

  log "Round ${ROUND}: physics audit"
  "$PYTHON" "$ROOT/scripts/visualpatchworld/evaluate_physics_rules.py" \
    --trajectories_glob "$TRAIN_GLOB" \
    --source "$INDUCED_SOURCE" \
    --results_dir "$CANDIDATE_DIR" \
    --output "$PHYSICS_REPORT"

  log "Round ${ROUND}: fair planning (sim_verify_frac=${SIM_VERIFY_FRAC})"
  INDUCED_RESULTS_DIR="$CANDIDATE_DIR" \
  INDUCED_SOURCE="$INDUCED_SOURCE" \
  SIM_VERIFY_FRAC="$SIM_VERIFY_FRAC" \
  INDUCED_REPORT="$INDUCED_REPORT" \
  COMPARISON_REPORT="$COMPARISON_REPORT" \
  bash "$ROOT/scripts/visualpatchworld/run_fair_planning_comparison.sh"

  BASELINE_FOR_ACCEPT="${BASELINE_REPORT:-$REPORT_DIR/pusht_planning_compositional_oracle_contact_refit_n50.json}"
  if [[ -n "$BEST_REPORT" && -f "$BEST_REPORT" ]]; then
    BASELINE_FOR_ACCEPT="$BEST_REPORT"
  elif [[ ! -f "$BASELINE_FOR_ACCEPT" ]]; then
    BASELINE_FOR_ACCEPT="$INDUCED_REPORT"
  fi

  ACCEPT_JSON="$("$PYTHON" - <<'PY' "$BASELINE_FOR_ACCEPT" "$INDUCED_REPORT" "$MIN_SUCCESS_GAIN" "$MAX_GOAL_DISTANCE_REGRESSION" "$ACCEPTANCE_REPORT" "$ROUND"
import json
import sys
from datetime import datetime
from pathlib import Path

from visualpatchworld.active_dynamics_induction import evaluate_acceptance

baseline_path, candidate_path, min_gain, max_regress, out_path, round_id = sys.argv[1:7]
baseline_raw = json.loads(Path(baseline_path).read_text(encoding="utf-8")) if Path(baseline_path).is_file() else {}
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
    "round": int(round_id),
    "timestamp": datetime.now().isoformat(),
    "baseline_report": baseline_path,
    "candidate_report": candidate_path,
    "acceptance": result,
}
Path(out_path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2, sort_keys=True))
PY
)"

  ROUND_SUCCESS="$(read_success_rate "$INDUCED_REPORT")"
  ROUND_GOAL="$(read_goal_distance "$INDUCED_REPORT")"
  log "Round ${ROUND}: success=${ROUND_SUCCESS}% goal_dist=${ROUND_GOAL}"

  append_loop_record "$("$PYTHON" - <<'PY' "$ROUND" "$ROUND_SUCCESS" "$ROUND_GOAL" "$CANDIDATE_DIR" "$ACCEPTANCE_REPORT" "$DIAG_JSON"
import json
import sys
from pathlib import Path

round_id, success, goal, candidate_dir, acceptance_path, diag_path = sys.argv[1:7]
acceptance = json.loads(Path(acceptance_path).read_text(encoding="utf-8"))
diag = json.loads(Path(diag_path).read_text(encoding="utf-8")) if Path(diag_path).is_file() else {}
record = {
    "round": int(round_id),
    "success_rate": float(success),
    "mean_goal_distance": None if goal == "inf" else float(goal),
    "candidate_dir": candidate_dir,
    "acceptance": acceptance.get("acceptance", {}),
    "recommended_patch": diag.get("recommended_patch"),
}
print(json.dumps(record, sort_keys=True))
PY
)"

  ACCEPTED="$(python3 - <<'PY' "$ACCEPTANCE_REPORT"
import json, sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print("1" if payload.get("acceptance", {}).get("accepted") else "0")
PY
)"

  if python3 - <<PY
best = float("$BEST_SUCCESS")
cur = float("$ROUND_SUCCESS")
import sys
sys.exit(0 if cur > best + 1e-9 else 1)
PY
  then
    BEST_SUCCESS="$ROUND_SUCCESS"
    BEST_RESULTS_DIR="$CANDIDATE_DIR"
    BEST_REPORT="$INDUCED_REPORT"
    log "Round ${ROUND}: new best success=${BEST_SUCCESS}%"
  fi

  if [[ "$ACCEPTED" == "1" ]]; then
    CURRENT_RESULTS_DIR="$CANDIDATE_DIR"
    log "Round ${ROUND}: accepted; next round uses $CURRENT_RESULTS_DIR"
  else
    CURRENT_RESULTS_DIR="$BEST_RESULTS_DIR"
    log "Round ${ROUND}: not accepted; next round uses best $CURRENT_RESULTS_DIR"
  fi

  if python3 - <<PY
import sys
target = float("$TARGET_SUCCESS_RATE")
cur = float("$BEST_SUCCESS")
sys.exit(0 if cur >= target - 1e-9 else 1)
PY
  then
    log "Target success rate ${TARGET_SUCCESS_RATE} reached (best success ${BEST_SUCCESS}). Stopping."
    break
  fi
done

log "Loop complete. Best success=${BEST_SUCCESS}% model at ${BEST_RESULTS_DIR}"
log "Log: ${LOOP_LOG}"

# Final summary vs ground truth
if [[ -f "$GT_REPORT" ]]; then
  GT_SUCCESS="$(read_success_rate "$GT_REPORT")"
  log "Ground truth reference: ${GT_SUCCESS}%"
fi
