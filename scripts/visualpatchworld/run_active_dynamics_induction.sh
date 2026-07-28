#!/usr/bin/env bash
# Active counterexample-guided dynamics induction loop.
#
# Each round:
#   1) audit physics on the current model
#   2) mine planning failures
#   3) collect simulator counterexamples (trajectory + active exploration)
#   4) collect rule violations
#   5) merge weighted JSONLs and refit gated+MLP weighted model
#   6) optional fair planning eval with acceptance gate
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

MAX_ROUNDS="${MAX_ROUNDS:-2}"
MAX_TRAIN="${MAX_TRAIN:-16}"
MAX_EVAL="${MAX_EVAL:-4}"
SIM_VERIFY_FRAC="${SIM_VERIFY_FRAC:-0.4}"
NUM_EVAL_MINE="${NUM_EVAL_MINE:-24}"
FAILURE_WEIGHT="${FAILURE_WEIGHT:-8.0}"
NUM_EXPLORE="${NUM_EXPLORE:-96}"
TOP_K_CE="${TOP_K_CE:-160}"
ERROR_THRESHOLD="${ERROR_THRESHOLD:-5.0}"
RUN_PLANNING="${RUN_PLANNING:-1}"
MIN_SUCCESS_GAIN="${MIN_SUCCESS_GAIN:-0.0}"
MAX_GOAL_DISTANCE_REGRESSION="${MAX_GOAL_DISTANCE_REGRESSION:-3.0}"

BASE_FAILURE_JSONL="${BASE_FAILURE_JSONL:-$DATA_DIR/planning_failure_transitions.jsonl,$DATA_DIR/planning_failure_transitions_v2.jsonl,$DATA_DIR/rule_violation_transitions.jsonl}"
INPUT_RESULTS_DIR="${INPUT_RESULTS_DIR:-$ROOT/artifacts/visualpatchworld/lewm_tool_scaffold_fit_gated_mlp_weighted_results}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/artifacts/visualpatchworld/lewm_tool_active_dynamics_results}"

BASELINE_REPORT="${BASELINE_REPORT:-$REPORT_DIR/pusht_planning_gated_mlp_weighted_simverify_cem_mpc_fair_n50.json}"
ACTIVE_LOG="${ACTIVE_LOG:-$REPORT_DIR/active_dynamics_induction.jsonl}"

log() {
  echo "[active-dynamics][$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

append_round_record() {
  local record="$1"
  echo "$record" >> "$ACTIVE_LOG"
}

CURRENT_RESULTS_DIR="$INPUT_RESULTS_DIR"
BEST_RESULTS_DIR="$INPUT_RESULTS_DIR"
BEST_REPORT="$BASELINE_REPORT"
BEST_SUCCESS="$(python3 - <<'PY' "$BASELINE_REPORT"
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
if path.is_file():
    report = json.loads(path.read_text(encoding="utf-8"))
    induced = report.get("induced") or report
    print(induced.get("success_rate", 0.0))
else:
    print(0.0)
PY
)"

for ROUND in $(seq 1 "$MAX_ROUNDS"); do
  log "===== Round ${ROUND}/${MAX_ROUNDS} ====="
  ROUND_DIR="${OUTPUT_ROOT}/round_${ROUND}"
  ROUND_DATA="$DATA_DIR/active_round_${ROUND}"
  mkdir -p "$ROUND_DATA" "$ROUND_DIR"

  PLANNING_FAILURE_JSONL="$ROUND_DATA/planning_failure_transitions.jsonl"
  SIM_CE_JSONL="$ROUND_DATA/sim_counterexample_transitions.jsonl"
  RULE_VIOLATION_JSONL="$ROUND_DATA/rule_violation_transitions.jsonl"
  MERGED_FAILURE_JSONL="${BASE_FAILURE_JSONL},${PLANNING_FAILURE_JSONL},${SIM_CE_JSONL},${RULE_VIOLATION_JSONL}"

  PHYSICS_REPORT="$REPORT_DIR/pusht_physics_rules_active_round_${ROUND}.json"
  INDUCED_REPORT="$REPORT_DIR/pusht_planning_active_round_${ROUND}_fair_n50.json"
  COMPARISON_REPORT="$REPORT_DIR/pusht_planning_fair_comparison_active_round_${ROUND}_n50.json"

  log "Round ${ROUND}: physics audit on ${CURRENT_RESULTS_DIR}"
  "$PYTHON" "$ROOT/scripts/visualpatchworld/evaluate_physics_rules.py" \
    --trajectories_glob "$TRAIN_GLOB" \
    --source tool \
    --results_dir "$CURRENT_RESULTS_DIR" \
    --output "$PHYSICS_REPORT"

  log "Round ${ROUND}: mine planning failures"
  "$PYTHON" "$ROOT/scripts/visualpatchworld/mine_planning_failures.py" \
    --source tool \
    --results_dir "$CURRENT_RESULTS_DIR" \
    --output "$PLANNING_FAILURE_JSONL" \
    --num_eval "$NUM_EVAL_MINE" \
    --failure_weight "$FAILURE_WEIGHT"

  log "Round ${ROUND}: collect simulator counterexamples"
  "$PYTHON" "$ROOT/scripts/visualpatchworld/collect_sim_counterexamples.py" \
    --source tool \
    --results_dir "$CURRENT_RESULTS_DIR" \
    --trajectories_jsonl "$TRAJ_JSONL" \
    --output "$SIM_CE_JSONL" \
    --num_explore "$NUM_EXPLORE" \
    --top_k "$TOP_K_CE" \
    --error_threshold "$ERROR_THRESHOLD" \
    --seed "$((42 + ROUND))"

  log "Round ${ROUND}: collect rule violations"
  "$PYTHON" "$ROOT/scripts/visualpatchworld/collect_rule_violation_transitions.py" \
    --trajectories_glob "$TRAIN_GLOB" \
    --source tool \
    --results_dir "$CURRENT_RESULTS_DIR" \
    --output "$RULE_VIOLATION_JSONL"

  log "Round ${ROUND}: refit gated+MLP weighted model"
  "$PYTHON" "$ROOT/scripts/visualpatchworld/induce_world_model.py" \
    --source tool \
    --data_dir "$DATA_DIR" \
    --output_dir "$ROUND_DIR" \
    --max_train "$MAX_TRAIN" \
    --max_eval "$MAX_EVAL" \
    --induction_strategy pusht_scaffold_fit_contact_gated_mlp_weighted \
    --failure_transitions_jsonl "$MERGED_FAILURE_JSONL" \
    --max_refine_rounds 0

  CURRENT_RESULTS_DIR="$ROUND_DIR"

  if [[ "$RUN_PLANNING" == "1" ]]; then
    log "Round ${ROUND}: fair planning eval"
    INDUCED_RESULTS_DIR="$ROUND_DIR" \
    SIM_VERIFY_FRAC="$SIM_VERIFY_FRAC" \
    INDUCED_REPORT="$INDUCED_REPORT" \
    COMPARISON_REPORT="$COMPARISON_REPORT" \
    bash "$ROOT/scripts/visualpatchworld/run_fair_planning_comparison.sh"

  ACCEPT_JSON="$("$PYTHON" - <<'PY' "$BEST_REPORT" "$INDUCED_REPORT" "$MIN_SUCCESS_GAIN" "$MAX_GOAL_DISTANCE_REGRESSION"
import json
import sys
from pathlib import Path
from visualpatchworld.active_dynamics_induction import evaluate_acceptance

baseline_path, candidate_path, min_gain, max_regress = sys.argv[1:5]
baseline_report = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
candidate_report = json.loads(Path(candidate_path).read_text(encoding="utf-8"))
baseline = baseline_report.get("induced") or baseline_report
candidate = candidate_report.get("induced") or candidate_report
result = evaluate_acceptance(
    baseline_report=baseline,
    candidate_report=candidate,
    min_success_gain=float(min_gain),
    max_goal_distance_regression=float(max_regress),
)
print(json.dumps(result, sort_keys=True))
PY
)"
    log "Round ${ROUND}: acceptance gate -> ${ACCEPT_JSON}"
    append_round_record "$(python3 - <<'PY' "$ROUND" "$ACCEPT_JSON" "$COMPARISON_REPORT" "$PHYSICS_REPORT"
import json, sys
record = {
    "round": int(sys.argv[1]),
    "acceptance": json.loads(sys.argv[2]),
    "comparison_report": sys.argv[3],
    "physics_report": sys.argv[4],
}
print(json.dumps(record, sort_keys=True))
PY
)"

    ACCEPTED="$(python3 - <<'PY' "$ACCEPT_JSON"
import json, sys
print("1" if json.loads(sys.argv[1]).get("accepted") else "0")
PY
)"
    CANDIDATE_SUCCESS="$(python3 - <<'PY' "$INDUCED_REPORT"
import json, sys
from pathlib import Path
report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
induced = report.get("induced") or report
print(induced.get("success_rate", 0.0))
PY
)"
    if [[ "$ACCEPTED" == "1" ]] || python3 - <<'PY' "$CANDIDATE_SUCCESS" "$BEST_SUCCESS"
import sys
exit(0 if float(sys.argv[1]) >= float(sys.argv[2]) else 1)
PY
    then
      BEST_RESULTS_DIR="$ROUND_DIR"
      BEST_REPORT="$INDUCED_REPORT"
      BEST_SUCCESS="$CANDIDATE_SUCCESS"
      log "Round ${ROUND}: accepted as new best (success=${BEST_SUCCESS})"
    else
      log "Round ${ROUND}: rejected; keeping previous best (success=${BEST_SUCCESS})"
      CURRENT_RESULTS_DIR="$BEST_RESULTS_DIR"
    fi
  fi
done

log "Active induction complete. Best model dir: ${BEST_RESULTS_DIR}"
log "Best planning report: ${BEST_REPORT}"
log "Round log: ${ACTIVE_LOG}"
