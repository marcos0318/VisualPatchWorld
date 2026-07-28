#!/usr/bin/env bash
# Run remaining paper experiments in parallel (one job per env/stage).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PY:-python3}"
export STABLEWM_HOME="${STABLEWM_HOME:-${HOME}/.stable-wm}"
export MUJOCO_GL=egl
export PYTHONPATH="${ROOT}/src:${ROOT}:${PYTHONPATH:-}"

REPORT_DIR="${REPORT_DIR:-${ROOT}/artifacts/visualpatchworld/reports}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs/paper_experiments}"
NUM_EVAL="${NUM_EVAL:-50}"
NUM_SAMPLES="${NUM_SAMPLES:-300}"
CEM_ITERS="${CEM_ITERS:-10}"
SIM_CEM_SAMPLES="${SIM_CEM_SAMPLES:-${NUM_SAMPLES:-300}}"
SIM_CEM_ITERS="${SIM_CEM_ITERS:-${CEM_ITERS:-10}}"
MAX_PARALLEL="${MAX_PARALLEL:-8}"

mkdir -p "${REPORT_DIR}" "${LOG_DIR}"

log() { echo "[paper-parallel][$(date '+%H:%M:%S')] $*"; }

ensure_mujoco() {
  local ver
  ver="$("${PY}" -c 'import mujoco; print(mujoco.__version__)' 2>/dev/null || echo missing)"
  if [[ "${ver}" != 3.8.* ]]; then
    log "Pinning mujoco 3.8.1 (was ${ver})"
    "${PY}" -m pip install -q 'mujoco==3.8.1'
  fi
}

run_if_missing() {
  local name="$1"
  local out="$2"
  shift 2
  if [[ -f "${out}" && "${FORCE_RERUN:-0}" != "1" ]]; then
    log "skip ${name} (exists)"
    return 0
  fi
  log "launch ${name}"
  (
    set -euo pipefail
    "$@" > "${LOG_DIR}/${name}.log" 2>&1
    log "done ${name}"
  ) &
  echo $! >> "${LOG_DIR}/pids.txt"
}

ensure_mujoco
: > "${LOG_DIR}/pids.txt"

# --- Cross-env sim ceilings (parallel) ---
for env in tworoom cube reacher; do
  run_if_missing "cross_env_${env}_sim_cem" \
    "${REPORT_DIR}/cross_env_${env}_sim_cem_n${NUM_EVAL}.json" \
    "${PY}" "${ROOT}/scripts/visualpatchworld/evaluate_cross_env_planning.py" \
      --env "${env}" \
      --results_dir "${ROOT}/artifacts/visualpatchworld/lewm_${env}_general_results" \
      --planner sim_cem \
      --num_eval "${NUM_EVAL}" \
      --num_samples "${SIM_CEM_SAMPLES}" \
      --cem_iters "${SIM_CEM_ITERS}" \
      --output "${REPORT_DIR}/cross_env_${env}_sim_cem_n${NUM_EVAL}.json"
done

# --- Cross-env hybrid (parallel) ---
for env in tworoom cube reacher; do
  run_if_missing "cross_env_${env}_hybrid_sv03" \
    "${REPORT_DIR}/cross_env_${env}_hybrid_sv03_n${NUM_EVAL}.json" \
    "${PY}" "${ROOT}/scripts/visualpatchworld/evaluate_cross_env_planning.py" \
      --env "${env}" \
      --results_dir "${ROOT}/artifacts/visualpatchworld/lewm_${env}_general_results" \
      --source oracle \
      --num_eval "${NUM_EVAL}" \
      --num_samples "${NUM_SAMPLES}" \
      --cem_iters "${CEM_ITERS}" \
      --sim_verify_frac 0.3 \
      --output "${REPORT_DIR}/cross_env_${env}_hybrid_sv03_n${NUM_EVAL}.json"
done

# --- PushT tool-live E2E (parallel) ---
if [[ "${SKIP_PUSHT_TOOL:-0}" != "1" ]]; then
  TOOL_DATA="${ROOT}/artifacts/visualpatchworld/lewm_tool_data"
  TOOL_RESULTS="${ROOT}/artifacts/visualpatchworld/lewm_tool_scaffold_fit_results"
  if [[ -f "${TOOL_RESULTS}/tool/generated_models/pusht_patchworld_20260618_004720.py" ]]; then
    run_if_missing "pusht_tool_live_scaffold_macro" \
      "${REPORT_DIR}/pusht_tool_live_scaffold_macro_n${NUM_EVAL}.json" \
      "${PY}" "${ROOT}/scripts/visualpatchworld/evaluate_planning.py" \
        --source tool \
        --results_dir "${TOOL_RESULTS}" \
        --data_dir "${TOOL_DATA}" \
        --planner cem_mpc \
        --observation_mode tool_live \
        --num_eval "${NUM_EVAL}" \
        --frameskip 5 --horizon 1 --model_rollout_horizon 1 --receding_horizon 1 \
        --num_samples "${NUM_SAMPLES}" --cem_iters "${CEM_ITERS}" \
        --output "${REPORT_DIR}/pusht_tool_live_scaffold_macro_n${NUM_EVAL}.json"
  fi
  if [[ -f "${ROOT}/artifacts/visualpatchworld/induced/program_induced_best.py" ]]; then
    run_if_missing "pusht_tool_live_program" \
      "${REPORT_DIR}/pusht_tool_live_program_n${NUM_EVAL}.json" \
      "${PY}" "${ROOT}/scripts/visualpatchworld/evaluate_planning.py" \
        --source oracle \
        --model_path "${ROOT}/artifacts/visualpatchworld/induced/program_induced_best.py" \
        --results_dir "${ROOT}/artifacts/visualpatchworld/induced" \
        --planner cem_mpc \
        --observation_mode tool_live \
        --num_eval "${NUM_EVAL}" \
        --frameskip 1 --horizon 5 --model_rollout_horizon 5 --receding_horizon 1 \
        --num_samples "${NUM_SAMPLES}" --cem_iters "${CEM_ITERS}" \
        --model_action_repeat 1 \
        --output "${REPORT_DIR}/pusht_tool_live_program_n${NUM_EVAL}.json"
  fi
fi

# --- Perception (instant, parallel) ---
for env in tworoom cube reacher; do
  out="${REPORT_DIR}/${env}_oracle_perception.json"
  if [[ ! -f "${out}" || "${FORCE_RERUN:-0}" == "1" ]]; then
    run_if_missing "${env}_oracle_perception" "${out}" \
      "${PY}" -c "
import json
from pathlib import Path
Path('${out}').write_text(json.dumps({
    'env': '${env}',
    'oracle_rmse': 0.0,
    'tool_rmse': None,
    'vlm_rmse': None,
    'object_precision': 1.0,
    'object_recall': 1.0,
    'note': 'Privileged HDF5 state export; pixel tool/VLM extractors pending.',
}, indent=2))
print('wrote ${out}')
"
  fi
done

# PushT tool perception
if [[ -f "${ROOT}/artifacts/visualpatchworld/lewm_tool_data/pusht_tool_traj_val.jsonl" ]]; then
  run_if_missing "pusht_tool_vs_oracle_val" \
    "${REPORT_DIR}/pusht_tool_vs_oracle_val.json" \
    "${PY}" "${ROOT}/scripts/visualpatchworld/evaluate_perception.py" \
      --candidate_source tool \
      --oracle_jsonl "${ROOT}/artifacts/visualpatchworld/lewm_fair_oracle/pusht_lewm_oracle_traj_val.jsonl" \
      --candidate_jsonl "${ROOT}/artifacts/visualpatchworld/lewm_tool_data/pusht_tool_traj_val.jsonl" \
      --output "${REPORT_DIR}/pusht_tool_vs_oracle_val.json"
fi

log "Waiting for $(wc -l < "${LOG_DIR}/pids.txt") parallel jobs..."
fail=0
while read -r pid; do
  [[ -n "${pid}" ]] || continue
  if ! wait "${pid}"; then
    fail=1
    log "job pid ${pid} failed (see ${LOG_DIR}/)"
  fi
done < "${LOG_DIR}/pids.txt"

log "=== Summary ==="
"${PY}" - <<'PY'
import json
from pathlib import Path
root = Path("${ROOT}/artifacts/visualpatchworld/reports")
keys = sorted(root.glob("cross_env_*_n50.json")) + sorted(root.glob("pusht_tool_live*.json")) + sorted(root.glob("*_oracle_perception.json"))
for path in keys:
    try:
        d = json.loads(path.read_text())
    except Exception:
        continue
    print(f"{path.name}: success={d.get('success_rate')} planner={d.get('planner','induced')} sv={d.get('sim_verify_frac')}")
PY

[[ "${fail}" -eq 0 ]] || exit 1
log "All parallel jobs finished."
