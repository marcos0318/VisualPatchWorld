#!/usr/bin/env bash
# Fill gaps in Table 1 / appendix: PushT tool hybrid, Cube oracle hybrid,
# PushT tool_live pure re-run (canonical 36%), archive oracle hybrid 92%.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PY:-python3}"
export STABLEWM_HOME="${STABLEWM_HOME:-${HOME}/.stable-wm}"
export MUJOCO_GL=egl
export PYTHONPATH="${ROOT}/src:${ROOT}:${PYTHONPATH:-}"

REPORT="${ROOT}/artifacts/visualpatchworld/reports"
LOG="${ROOT}/logs/paper_experiments"
MODEL="${ROOT}/artifacts/visualpatchworld/induced/program_induced_best.py"
CUBE_RESULTS="${ROOT}/artifacts/visualpatchworld/lewm_cube_general_results"
NUM_EVAL="${NUM_EVAL:-50}"
NUM_SAMPLES="${NUM_SAMPLES:-300}"
CEM_ITERS_PUSHT_PURE="${CEM_ITERS_PUSHT_PURE:-30}"
CEM_ITERS_HYBRID="${CEM_ITERS_HYBRID:-10}"
CEM_ITERS_CUBE="${CEM_ITERS_CUBE:-10}"

mkdir -p "$REPORT" "$LOG"
log() { echo "[missing-paper][$(date +%H:%M:%S)] $*"; }

# Archive existing program-oracle hybrid 92% (already computed under induced/)
ORACLE_HYBRID_OUT="$REPORT/pusht_oracle_program_hybrid_sv03_n${NUM_EVAL}.json"
if [[ ! -f "$ORACLE_HYBRID_OUT" ]]; then
  if [[ -f "${ROOT}/artifacts/visualpatchworld/induced/plan_sv03.json" ]]; then
    log "copy oracle hybrid 92% -> $ORACLE_HYBRID_OUT"
    cp "${ROOT}/artifacts/visualpatchworld/induced/plan_sv03.json" "$ORACLE_HYBRID_OUT"
  else
    log "run oracle hybrid (program + oracle + sim_verify=0.3)"
    "$PY" "$ROOT/scripts/visualpatchworld/evaluate_planning.py" \
      --source oracle \
      --model_path "$MODEL" \
      --results_dir "${ROOT}/artifacts/visualpatchworld/induced" \
      --planner cem_mpc \
      --observation_mode oracle \
      --expose_block_velocity \
      --sim_verify_frac 0.3 \
      --num_eval "$NUM_EVAL" --seed 42 \
      --goal_offset_steps 25 --eval_budget 50 \
      --frameskip 1 --horizon 5 --model_rollout_horizon 5 \
      --receding_horizon 1 --model_action_repeat 1 \
      --num_samples "$NUM_SAMPLES" --cem_iters "$CEM_ITERS_HYBRID" \
      --output "$ORACLE_HYBRID_OUT" \
      2>&1 | tee "$LOG/pusht_oracle_hybrid_sv03.log"
  fi
else
  log "skip oracle hybrid (exists)"
fi

# Tier 1: PushT tool_live pure (canonical pixel-fair row)
PUSHT_PURE_OUT="$REPORT/pusht_tool_live_program_n${NUM_EVAL}.json"
if [[ "${FORCE_PUSHT_PURE:-1}" == "1" ]] || [[ ! -f "$PUSHT_PURE_OUT" ]]; then
  log "run PushT tool_live pure (CEM ${NUM_SAMPLES}x${CEM_ITERS_PUSHT_PURE})"
  "$PY" "$ROOT/scripts/visualpatchworld/evaluate_planning.py" \
    --source oracle \
    --model_path "$MODEL" \
    --results_dir "${ROOT}/artifacts/visualpatchworld/induced" \
    --planner cem_mpc \
    --observation_mode tool_live \
    --expose_block_velocity \
    --num_eval "$NUM_EVAL" --seed 42 \
    --goal_offset_steps 25 --eval_budget 50 \
    --frameskip 1 --horizon 5 --model_rollout_horizon 5 \
    --receding_horizon 1 --model_action_repeat 1 \
    --num_samples "$NUM_SAMPLES" --cem_iters "$CEM_ITERS_PUSHT_PURE" \
    --output "$PUSHT_PURE_OUT" \
    2>&1 | tee "$LOG/pusht_tool_live_pure_rerun.log"
else
  log "skip PushT tool_live pure (exists, set FORCE_PUSHT_PURE=1 to rerun)"
fi

# Tier 2: PushT tool hybrid (fills Table 1 `--` cell)
PUSHT_TOOL_HYBRID_OUT="$REPORT/pusht_tool_live_program_hybrid_sv03_n${NUM_EVAL}.json"
if [[ ! -f "$PUSHT_TOOL_HYBRID_OUT" ]] || [[ "${FORCE:-0}" == "1" ]]; then
  log "run PushT tool hybrid (tool_live + sim_verify=0.3)"
  "$PY" "$ROOT/scripts/visualpatchworld/evaluate_planning.py" \
    --source oracle \
    --model_path "$MODEL" \
    --results_dir "${ROOT}/artifacts/visualpatchworld/induced" \
    --planner cem_mpc \
    --observation_mode tool_live \
    --expose_block_velocity \
    --sim_verify_frac 0.3 \
    --num_eval "$NUM_EVAL" --seed 42 \
    --goal_offset_steps 25 --eval_budget 50 \
    --frameskip 1 --horizon 5 --model_rollout_horizon 5 \
    --receding_horizon 1 --model_action_repeat 1 \
    --num_samples "$NUM_SAMPLES" --cem_iters "$CEM_ITERS_HYBRID" \
    --output "$PUSHT_TOOL_HYBRID_OUT" \
    2>&1 | tee "$LOG/pusht_tool_hybrid_sv03.log"
else
  log "skip PushT tool hybrid (exists)"
fi

# Tier 3: Cube oracle hybrid (failed write previously)
CUBE_ORACLE_HYBRID_OUT="$REPORT/cross_env_cube_hybrid_oracle_sv03_n${NUM_EVAL}.json"
if [[ ! -f "$CUBE_ORACLE_HYBRID_OUT" ]] || [[ "${FORCE:-0}" == "1" ]]; then
  log "run Cube oracle hybrid (oracle + sim_verify=0.3)"
  "$PY" "$ROOT/scripts/visualpatchworld/evaluate_cross_env_planning.py" \
    --env cube \
    --results_dir "$CUBE_RESULTS" \
    --source oracle \
    --observation_mode oracle \
    --sim_verify_frac 0.3 \
    --num_eval "$NUM_EVAL" \
    --num_samples "$NUM_SAMPLES" \
    --cem_iters "$CEM_ITERS_CUBE" \
    --output "$CUBE_ORACLE_HYBRID_OUT" \
    2>&1 | tee "$LOG/cube_hybrid_oracle_sv03.log"
else
  log "skip Cube oracle hybrid (exists)"
fi

log "summary"
"$PY" - <<PY
import json
from pathlib import Path
names = [
    "pusht_oracle_program_hybrid_sv03_n${NUM_EVAL}.json",
    "pusht_tool_live_program_n${NUM_EVAL}.json",
    "pusht_tool_live_program_hybrid_sv03_n${NUM_EVAL}.json",
    "cross_env_cube_hybrid_oracle_sv03_n${NUM_EVAL}.json",
]
for name in names:
    p = Path("$REPORT") / name
    if p.exists():
        d = json.loads(p.read_text())
        print(name, "success=", d.get("success_rate"), "sv=", d.get("sim_verify_frac"), "obs=", d.get("observation_mode"))
    else:
        print(name, "MISSING")
PY
