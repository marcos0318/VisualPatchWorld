#!/usr/bin/env bash
# Fair planning comparison: induced scaffold-fit MPC vs ground-truth simulator MPC.
#
# Uses the same PushT eval protocol (n=50, goal_offset=25, budget=50, frameskip=5,
# seed=42) and the same CEM budget (300 samples x 10 iterations). Ground-truth MPC
# is re-used from a prior run when available; induced MPC is always re-run.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-${CONDA_PREFIX:-}/envs/lewm/bin/python3}"
PATCHWORLD_ROOT="${PATCHWORLD_ROOT:-${ROOT}}"
export STABLEWM_HOME="${STABLEWM_HOME:-${HOME}/.stable-wm}"
export PYTHONPATH="$PATCHWORLD_ROOT:$ROOT/src:${PYTHONPATH:-}"

REPORT_DIR="${REPORT_DIR:-$ROOT/artifacts/visualpatchworld/reports}"
INDUCED_RESULTS_DIR="${INDUCED_RESULTS_DIR:-$ROOT/artifacts/visualpatchworld/lewm_tool_scaffold_fit_results}"
INDUCED_SOURCE="${INDUCED_SOURCE:-tool}"

NUM_EVAL="${NUM_EVAL:-50}"
NUM_SAMPLES="${NUM_SAMPLES:-300}"
CEM_ITERS="${CEM_ITERS:-10}"
SEED="${SEED:-42}"
SKIP_INDUCED="${SKIP_INDUCED:-0}"
SIM_VERIFY_FRAC="${SIM_VERIFY_FRAC:-0.0}"

GT_REPORT="${GT_REPORT:-$REPORT_DIR/pusht_planning_simulator_cem_mpc_ground_truth_n50.json}"
INDUCED_REPORT="${INDUCED_REPORT:-$REPORT_DIR/pusht_planning_tool_scaffold_fit_cem_mpc_fair_n50.json}"
COMPARISON_REPORT="${COMPARISON_REPORT:-$REPORT_DIR/pusht_planning_fair_comparison_n50.json}"

log() {
  echo "[fair-planning][$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

run_induced() {
  log "Running induced MPC (num_eval=${NUM_EVAL}, samples=${NUM_SAMPLES}, iters=${CEM_ITERS}, sim_verify=${SIM_VERIFY_FRAC})"
  induced_args=(
    "$PYTHON" "$ROOT/scripts/visualpatchworld/evaluate_planning.py"
    --source "$INDUCED_SOURCE"
    --results_dir "$INDUCED_RESULTS_DIR"
    --planner cem_mpc
    --observation_mode oracle
    --num_eval "$NUM_EVAL"
    --goal_offset_steps 25
    --eval_budget 50
    --frameskip 5
    --horizon 1
    --model_rollout_horizon 1
    --receding_horizon 1
    --num_samples "$NUM_SAMPLES"
    --cem_iters "$CEM_ITERS"
    --elite_frac 0.1
    --seed "$SEED"
    --output "$INDUCED_REPORT"
  )
  if [[ "$SIM_VERIFY_FRAC" != "0" && "$SIM_VERIFY_FRAC" != "0.0" ]]; then
    induced_args+=(--sim_verify_frac "$SIM_VERIFY_FRAC")
  fi
  "${induced_args[@]}"
}

summarize() {
  log "Writing comparison report to ${COMPARISON_REPORT}"
  "$PYTHON" - <<'PY' "$GT_REPORT" "$INDUCED_REPORT" "$COMPARISON_REPORT" "$NUM_EVAL" "$NUM_SAMPLES" "$CEM_ITERS" "$SEED"
import json
import sys
from datetime import datetime
from pathlib import Path

gt_path, induced_path, out_path, num_eval, num_samples, cem_iters, seed = sys.argv[1:8]
num_eval = int(num_eval)
num_samples = int(num_samples)
cem_iters = int(cem_iters)
seed = int(seed)

def load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))

def pick(report: dict, label: str) -> dict:
    return {
        "label": label,
        "report_path": report.get("_path"),
        "model_path": report.get("model_path"),
        "model_type": report.get("model_type", "induced_symbolic"),
        "dynamics": report.get("model_type", "induced_symbolic"),
        "observation_mode": report.get("observation_mode", "oracle"),
        "planner": report.get("planner"),
        "num_eval": report.get("num_eval"),
        "goal_offset_steps": report.get("goal_offset_steps"),
        "eval_budget": report.get("eval_budget"),
        "frameskip": report.get("frameskip"),
        "num_samples": report.get("num_samples"),
        "cem_iters": report.get("cem_iters"),
        "horizon": report.get("horizon"),
        "receding_horizon": report.get("receding_horizon"),
        "model_rollout_horizon": report.get("model_rollout_horizon"),
        "sim_verify_frac": report.get("sim_verify_frac"),
        "action_block": report.get("action_block"),
        "seed": report.get("seed"),
        "success_rate": report.get("success_rate"),
        "mean_goal_distance": report.get("mean_goal_distance"),
        "mean_planner_model_distance": report.get("mean_planner_model_distance"),
        "failed_model_rollouts": report.get("failed_model_rollouts"),
    }

gt = load(gt_path)
induced = load(induced_path)
gt["_path"] = gt_path
induced["_path"] = induced_path

comparison = {
    "timestamp": datetime.now().isoformat(),
    "protocol": {
        "dataset": gt.get("dataset") or induced.get("dataset"),
        "num_eval": num_eval,
        "goal_offset_steps": 25,
        "eval_budget": 50,
        "frameskip": 5,
        "seed": seed,
        "cem_budget": {"num_samples": num_samples, "cem_iters": cem_iters, "elite_frac": 0.1},
        "notes": [
            "Same eval starts and success criterion for both runs.",
            "Ground truth uses per-step simulator MPC (horizon=5, action_block=5, receding_horizon=5).",
            "Induced uses macro MPC with oracle re-observation (model_rollout_horizon=1, receding_horizon=1).",
            "Induced dynamics come from tool scaffold-fit; observations are oracle simulator graphs at replan time.",
        ],
    },
    "ground_truth": pick(gt, "ground_truth_simulator_mpc"),
    "induced": pick(induced, "tool_scaffold_fit_mpc"),
    "delta": {
        "success_rate_points": float(induced.get("success_rate", 0.0)) - float(gt.get("success_rate", 0.0)),
        "mean_goal_distance": float(induced.get("mean_goal_distance", 0.0)) - float(gt.get("mean_goal_distance", 0.0)),
    },
}
Path(out_path).write_text(json.dumps(comparison, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps(comparison, indent=2, sort_keys=True))
PY
}

if [[ ! -f "$GT_REPORT" ]]; then
  log "Ground-truth report missing; running simulator MPC first"
  "$PYTHON" "$ROOT/scripts/visualpatchworld/evaluate_simulator_planning.py" \
    --planner cem_mpc \
    --num_eval "$NUM_EVAL" \
    --goal_offset_steps 25 \
    --eval_budget 50 \
    --frameskip 5 \
    --horizon 5 \
    --action_block 5 \
    --receding_horizon 5 \
    --num_samples "$NUM_SAMPLES" \
    --cem_iters "$CEM_ITERS" \
    --elite_frac 0.1 \
    --seed "$SEED" \
    --output "$GT_REPORT"
fi

if [[ "$SKIP_INDUCED" != "1" ]]; then
  run_induced
else
  log "Skipping induced run (SKIP_INDUCED=1)"
fi

summarize
log "Done."
