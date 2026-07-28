#!/usr/bin/env bash
# Export, induce, and evaluate VisualPatchWorld on all LeWM benchmark envs.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PY:-python3}"
export STABLEWM_HOME="${STABLEWM_HOME:-${HOME}/.stable-wm}"
export MUJOCO_GL=egl
export PYTHONPATH="${ROOT}/src:${ROOT}:${PYTHONPATH:-}"

MAX_EPISODES="${MAX_EPISODES:-64}"
NUM_EVAL="${NUM_EVAL:-50}"
NUM_SAMPLES="${NUM_SAMPLES:-300}"
CEM_ITERS="${CEM_ITERS:-10}"
DATA_DIR="${DATA_DIR:-${ROOT}/artifacts/visualpatchworld/data}"
REPORT_DIR="${REPORT_DIR:-${ROOT}/artifacts/visualpatchworld/reports}"

mkdir -p "${DATA_DIR}" "${REPORT_DIR}"

run_env() {
  local env="$1"
  local results_dir="${ROOT}/artifacts/visualpatchworld/lewm_${env}_general_results"
  echo "=== ${env}: export ==="
  "${PY}" "${ROOT}/scripts/visualpatchworld/export_lewm_env.py" \
    --env "${env}" \
    --output_dir "${DATA_DIR}" \
    --max_episodes "${MAX_EPISODES}" \
    --source oracle

  echo "=== ${env}: induce program_induction (L1 probe + L2 fit) ==="
  "${PY}" -m visualpatchworld.patchworld_benchmark \
    --env "${env}" \
    --train_glob "${DATA_DIR}/${env}_oracle_traj_train.jsonl" \
    --eval_glob "${DATA_DIR}/${env}_oracle_traj_val.jsonl" \
    --output_dir "${results_dir}" \
    --induction_strategy program_induction \
    --max_train 0 \
    --max_eval 0 \
    --max_refine_rounds 0

  echo "=== ${env}: planning eval ==="
  "${PY}" "${ROOT}/scripts/visualpatchworld/evaluate_cross_env_planning.py" \
    --env "${env}" \
    --results_dir "${results_dir}" \
    --source oracle \
    --num_eval "${NUM_EVAL}" \
    --num_samples "${NUM_SAMPLES}" \
    --cem_iters "${CEM_ITERS}" \
    --output "${REPORT_DIR}/cross_env_${env}_n${NUM_EVAL}.json"
}

for env in tworoom cube reacher; do
  run_env "${env}"
done

echo "=== Summary ==="
"${PY}" - <<'PY'
import json
from pathlib import Path
root = Path("${ROOT}/artifacts/visualpatchworld/reports")
for path in sorted(root.glob("cross_env_*_n*.json")):
    data = json.loads(path.read_text())
    env = data.get("env", path.stem)
    sr = data.get("success_rate")
    gd = data.get("mean_goal_distance")
    note = data.get("planning_note", "")
    print(f"{env}: success={sr} goal_dist={gd} {note}")
PY
