#!/usr/bin/env bash
# Re-induce all four LeWM envs with Level-1 probing + Level-2 fit, then
# re-run fair Oracle+Induced ranking (GT caches reused).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
conda activate lewm
export PYTHONPATH="${ROOT}/src:${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
PY="${PY:-python}"
DATA="${ROOT}/artifacts/visualpatchworld/data"
LOG_DIR="${ROOT}/artifacts/visualpatchworld/reports/program_induction_rerun"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"

induce_env() {
  local env="$1"
  local out="${ROOT}/artifacts/visualpatchworld/lewm_${env}_program_results"
  local report="${out}/structure_report_${STAMP}.json"
  mkdir -p "$out"
  echo "=== [${env}] Level-1 identify + Level-2 induce ==="
  local extra=()
  if [[ "${env}" == "pusht" ]]; then
    # Perceived geometry + multi-restart selection are load-bearing for PushT.
    extra+=(
      --geometry_json "${ROOT}/artifacts/visualpatchworld/perceived_geometry/pusht_geometry.json"
      --rollout_k 6 --rollout_fit_steps 400
    )
  fi
  "$PY" -m visualpatchworld.patchworld_benchmark \
    --env "${env}" \
    --train_glob "${DATA}/${env}_oracle_traj_train.jsonl" \
    --eval_glob "${DATA}/${env}_oracle_traj_val.jsonl" \
    --output_dir "${out}" \
    --induction_strategy program_induction \
    --max_train 0 --max_eval 0 --max_refine_rounds 0 \
    "${extra[@]}" \
    2>&1 | tee "${LOG_DIR}/induce_${env}_${STAMP}.log"
  # Structure report only (no second L2 fit) — avoids a long duplicate PushT fit.
  "$PY" scripts/visualpatchworld/identify_structure.py \
    --env "${env}" \
    --report_out "${report}" \
    ${extra[@]+"${extra[@]}"} \
    2>&1 | tee -a "${LOG_DIR}/induce_${env}_${STAMP}.log" || true
  echo "=== [${env}] wrote models under ${out}/generated_models ==="
}

# Fast envs first in parallel; PushT L2 fit is heavier.
induce_env tworoom &
PID_TWO=$!
induce_env cube &
PID_CUBE=$!
induce_env reacher &
PID_REACH=$!
wait "$PID_TWO" "$PID_CUBE" "$PID_REACH"
induce_env pusht

echo "=== Fair Oracle+Induced ranking (skip GT, reuse caches) ==="
"$PY" scripts/visualpatchworld/run_fair_wm_ranking_gt90.py \
  --envs tworoom,cube,reacher,pusht \
  --num_eval 50 --seed 42 --skip_gt_eval \
  --output_dir "${ROOT}/artifacts/visualpatchworld/reports/fair_wm_ranking_gt90" \
  2>&1 | tee "${LOG_DIR}/fair_gt90_${STAMP}.log"

echo "=== DONE ==="
echo "Summary: ${ROOT}/artifacts/visualpatchworld/reports/fair_wm_ranking_gt90/summary_n50_seed42.json"
echo "Logs: ${LOG_DIR}"
