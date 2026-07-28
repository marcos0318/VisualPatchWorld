#!/usr/bin/env bash
# Step 1: Reacher joint-dynamics induction + oracle/tool_live planning eval.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PY:-python3}"
export STABLEWM_HOME="${STABLEWM_HOME:-${HOME}/.stable-wm}"
export MUJOCO_GL=egl
export PYTHONPATH="${ROOT}/src:${ROOT}:${PYTHONPATH:-}"

DATA="${ROOT}/artifacts/visualpatchworld/data"
RESULTS="${ROOT}/artifacts/visualpatchworld/lewm_reacher_joint_results"
REPORT="${ROOT}/artifacts/visualpatchworld/reports"
LOG="${ROOT}/logs/paper_experiments"
NUM_EVAL="${NUM_EVAL:-50}"
NUM_SAMPLES="${NUM_SAMPLES:-300}"
CEM_ITERS="${CEM_ITERS:-10}"
FORCE="${FORCE:-0}"

mkdir -p "$RESULTS" "$REPORT" "$LOG"
log() { echo "[step1-reacher][$(date +%H:%M:%S)] $*"; }

if [[ ! -f "${DATA}/reacher_oracle_traj_train.jsonl" ]]; then
  log "export reacher trajectories"
  "$PY" "$ROOT/scripts/visualpatchworld/export_lewm_env.py" \
    --env reacher \
    --output_dir "$DATA" \
    --max_episodes 64 \
    --source oracle
fi

if [[ ! -d "${RESULTS}/generated_models" ]] || [[ "$FORCE" == "1" ]]; then
  log "induce program_induction (L1 probe → joint+FK)"
  "$PY" -m visualpatchworld.patchworld_benchmark \
    --env reacher \
    --train_glob "${DATA}/reacher_oracle_traj_train.jsonl" \
    --eval_glob "${DATA}/reacher_oracle_traj_val.jsonl" \
    --output_dir "$RESULTS" \
    --induction_strategy program_induction \
    --max_train 0 \
    --max_eval 0 \
    --max_refine_rounds 0 \
    2>&1 | tee "$LOG/reacher_joint_induce.log"
fi

run_eval() {
  local mode="$1"
  local out="$REPORT/reacher_joint_${mode}_n${NUM_EVAL}.json"
  if [[ -f "$out" && "$FORCE" != "1" ]]; then
    log "skip $mode (exists)"
    return 0
  fi
  log "eval $mode -> $out"
  "$PY" "$ROOT/scripts/visualpatchworld/evaluate_cross_env_planning.py" \
    --env reacher \
    --results_dir "$RESULTS" \
    --source oracle \
    --observation_mode "$mode" \
    --num_eval "$NUM_EVAL" \
    --num_samples "$NUM_SAMPLES" \
    --cem_iters "$CEM_ITERS" \
    --output "$out" \
    2>&1 | tee "$LOG/reacher_joint_${mode}.log"
}

run_eval oracle
run_eval tool_live

log "summary"
"$PY" - <<PY
import json
from pathlib import Path
for p in sorted(Path("$REPORT").glob("reacher_joint_*_n${NUM_EVAL}.json")):
    d = json.loads(p.read_text())
    print(p.name, "success=", d.get("success_rate"), "tool_fail=", d.get("tool_extraction_failures"))
PY
