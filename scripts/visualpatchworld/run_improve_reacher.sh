#!/usr/bin/env bash
# Reacher step: warm-start IK + joint model + sim_cem ceiling + tool_live.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PY:-python3}"
export STABLEWM_HOME="${STABLEWM_HOME:-${HOME}/.stable-wm}"
export MUJOCO_GL=egl
export PYTHONPATH="${ROOT}/src:${ROOT}:${PYTHONPATH:-}"

DATA="${ROOT}/artifacts/visualpatchworld/data"
REPORT="${ROOT}/artifacts/visualpatchworld/reports"
LOG="${ROOT}/logs/paper_experiments"
JOINT="${ROOT}/artifacts/visualpatchworld/lewm_reacher_joint_results"
NUM_EVAL="${NUM_EVAL:-50}"
NUM_SAMPLES="${NUM_SAMPLES:-300}"
CEM_ITERS="${CEM_ITERS:-10}"
FORCE="${FORCE:-0}"

mkdir -p "$REPORT" "$LOG"
log() { echo "[reacher-improve][$(date +%H:%M:%S)] $*"; }

if [[ ! -d "${JOINT}/generated_models" ]]; then
  log "induce reacher_joint_fit"
  "$PY" -m visualpatchworld.patchworld_benchmark \
    --env reacher \
    --train_glob "${DATA}/reacher_oracle_traj_train.jsonl" \
    --eval_glob "${DATA}/reacher_oracle_traj_val.jsonl" \
    --output_dir "$JOINT" \
    --induction_strategy program_induction \
    --max_train 0 --max_eval 0 --max_refine_rounds 0
fi

run_eval() {
  local tag="$1"; shift
  local out="$REPORT/reacher_${tag}_n${NUM_EVAL}.json"
  if [[ -f "$out" && "$FORCE" != "1" ]]; then log "skip $tag"; return 0; fi
  log "eval $tag"
  "$PY" "$ROOT/scripts/visualpatchworld/evaluate_cross_env_planning.py" \
    --env reacher --results_dir "$JOINT" --source oracle \
    --num_eval "$NUM_EVAL" --num_samples "$NUM_SAMPLES" --cem_iters "$CEM_ITERS" \
    "$@" --output "$out" 2>&1 | tee "$LOG/reacher_${tag}.log"
}

run_eval "joint_oracle" --observation_mode oracle
run_eval "joint_tool_live" --observation_mode tool_live
run_eval "sim_cem_heavy" --planner sim_cem --num_samples "$NUM_SAMPLES" --cem_iters "$CEM_ITERS"

log "summary"
"$PY" - <<PY
import json
from pathlib import Path
for p in sorted(Path("$REPORT").glob("reacher_*_n${NUM_EVAL}.json")):
    d = json.loads(p.read_text())
    print(p.name, "success=", d.get("success_rate"), "tool_fail=", d.get("tool_extraction_failures"))
PY
