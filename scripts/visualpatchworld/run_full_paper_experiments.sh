#!/usr/bin/env bash
# Fill remaining VisualPatchWorld / KDD paper experiment blanks.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PY:-python3}"
export STABLEWM_HOME="${STABLEWM_HOME:-${HOME}/.stable-wm}"
export MUJOCO_GL=egl
export PYTHONPATH="${ROOT}/src:${ROOT}:${PYTHONPATH:-}"

DATA_DIR="${DATA_DIR:-${ROOT}/artifacts/visualpatchworld/data}"
REPORT_DIR="${REPORT_DIR:-${ROOT}/artifacts/visualpatchworld/reports}"
NUM_EVAL="${NUM_EVAL:-50}"
NUM_SAMPLES="${NUM_SAMPLES:-300}"
CEM_ITERS="${CEM_ITERS:-10}"
MAX_EPISODES="${MAX_EPISODES:-64}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs/paper_experiments}"
mkdir -p "${REPORT_DIR}" "${LOG_DIR}"

log() { echo "[paper-experiments][$(date '+%H:%M:%S')] $*"; }

ensure_mujoco() {
  local ver
  ver="$("${PY}" -c 'import mujoco; print(mujoco.__version__)' 2>/dev/null || echo missing)"
  if [[ "${ver}" != 3.8.* ]]; then
    log "Pinning mujoco 3.8.1 for Reacher/dm_control (was ${ver})"
    "${PY}" -m pip install -q 'mujoco==3.8.1'
  fi
}

run_cross_env_oracle() {
  local env="$1"
  local out="${REPORT_DIR}/cross_env_${env}_n${NUM_EVAL}.json"
  if [[ "${FORCE_RERUN:-0}" != "1" && -f "${out}" ]]; then
    log "${env}: skip oracle pure (exists ${out})"
    return 0
  fi
  local results_dir="${ROOT}/artifacts/visualpatchworld/lewm_${env}_general_results"
  if [[ ! -f "${results_dir}/generated_models/$(ls -1 "${results_dir}/generated_models" 2>/dev/null | tail -1)" ]]; then
    log "${env}: export + induce"
    "${PY}" "${ROOT}/scripts/visualpatchworld/export_lewm_env.py" \
      --env "${env}" --output_dir "${DATA_DIR}" --max_episodes "${MAX_EPISODES}" --source oracle
    "${PY}" -m visualpatchworld.patchworld_benchmark \
      --env "${env}" \
      --train_glob "${DATA_DIR}/${env}_oracle_traj_train.jsonl" \
      --eval_glob "${DATA_DIR}/${env}_oracle_traj_val.jsonl" \
      --output_dir "${results_dir}" \
      --induction_strategy program_induction \
      --max_train 0 --max_eval 0 --max_refine_rounds 0
  fi
  log "${env}: oracle pure planning"
  "${PY}" "${ROOT}/scripts/visualpatchworld/evaluate_cross_env_planning.py" \
    --env "${env}" --results_dir "${results_dir}" --source oracle \
    --num_eval "${NUM_EVAL}" --num_samples "${NUM_SAMPLES}" --cem_iters "${CEM_ITERS}" \
    --output "${REPORT_DIR}/cross_env_${env}_n${NUM_EVAL}.json" \
    2>&1 | tee "${LOG_DIR}/cross_env_${env}_oracle.log"
}

run_cross_env_hybrid() {
  local env="$1"
  local out="${REPORT_DIR}/cross_env_${env}_hybrid_sv03_n${NUM_EVAL}.json"
  if [[ "${FORCE_RERUN:-0}" != "1" && -f "${out}" ]]; then
    log "${env}: skip hybrid (exists ${out})"
    return 0
  fi
  local results_dir="${ROOT}/artifacts/visualpatchworld/lewm_${env}_general_results"
  log "${env}: hybrid sim_verify=0.3"
  "${PY}" "${ROOT}/scripts/visualpatchworld/evaluate_cross_env_planning.py" \
    --env "${env}" --results_dir "${results_dir}" --source oracle \
    --num_eval "${NUM_EVAL}" --num_samples "${NUM_SAMPLES}" --cem_iters "${CEM_ITERS}" \
    --sim_verify_frac 0.3 \
    --output "${REPORT_DIR}/cross_env_${env}_hybrid_sv03_n${NUM_EVAL}.json" \
    2>&1 | tee "${LOG_DIR}/cross_env_${env}_hybrid.log"
}

run_cross_env_sim_ceiling() {
  local env="$1"
  local out="${REPORT_DIR}/cross_env_${env}_sim_cem_n${NUM_EVAL}.json"
  if [[ "${FORCE_RERUN:-0}" != "1" && -f "${out}" ]]; then
    log "${env}: skip sim ceiling (exists ${out})"
    return 0
  fi
  local sim_samples="${SIM_CEM_SAMPLES:-${NUM_SAMPLES:-300}}"
  local sim_iters="${SIM_CEM_ITERS:-${CEM_ITERS:-10}}"
  log "${env}: true-simulator CEM ceiling (samples=${sim_samples}, iters=${sim_iters})"
  "${PY}" "${ROOT}/scripts/visualpatchworld/evaluate_cross_env_planning.py" \
    --env "${env}" --results_dir "${ROOT}/artifacts/visualpatchworld/lewm_${env}_general_results" \
    --planner sim_cem \
    --num_eval "${NUM_EVAL}" --num_samples "${sim_samples}" --cem_iters "${sim_iters}" \
    --output "${REPORT_DIR}/cross_env_${env}_sim_cem_n${NUM_EVAL}.json" \
    2>&1 | tee "${LOG_DIR}/cross_env_${env}_sim_cem.log"
}

run_pusht_tool_pipeline() {
  local sample_dir="${ROOT}/artifacts/visualpatchworld/lewm_fair_samples"
  local tool_data="${ROOT}/artifacts/visualpatchworld/lewm_tool_data"
  local tool_results="${ROOT}/artifacts/visualpatchworld/lewm_tool_scaffold_fit_results"

  if [[ ! -d "${sample_dir}/images" ]]; then
    log "PushT: sampling LeWM-aligned image trajectories"
    "${PY}" "${ROOT}/scripts/visualpatchworld/sample_lewm_aligned_trajectories.py" \
      --sources image_action \
      --output_dir "${sample_dir}" \
      --max_clips 512
  fi

  if [[ ! -f "${tool_data}/pusht_tool_traj_train.jsonl" ]]; then
    log "PushT: extract tool scene graphs + build trajectories"
    "${PY}" "${ROOT}/scripts/visualpatchworld/extract_tool_scene_graphs.py" \
      --sample_dir "${sample_dir}" \
      --cache_dir "${ROOT}/artifacts/visualpatchworld/lewm_tool_cache/pusht_heuristic" \
      --splits train val
    "${PY}" "${ROOT}/scripts/visualpatchworld/build_vlm_trajectories.py" \
      --sample_dir "${sample_dir}" \
      --cache_dir "${ROOT}/artifacts/visualpatchworld/lewm_tool_cache/pusht_heuristic" \
      --output_dir "${tool_data}" \
      --source_name tool \
      --splits train val
  fi

  if [[ ! -f "${tool_results}/tool/generated_models/pusht_patchworld_20260618_004720.py" ]]; then
    log "PushT: induce tool scaffold-fit model"
    "${PY}" -m visualpatchworld.patchworld_benchmark \
      --env pusht \
      --train_glob "${tool_data}/pusht_tool_traj_train.jsonl" \
      --eval_glob "${tool_data}/pusht_tool_traj_val.jsonl" \
      --output_dir "${tool_results}/tool" \
      --induction_strategy program_induction \
      --max_train 0 --max_eval 0 --max_refine_rounds 0
  fi

  log "PushT: perception tool vs oracle"
  "${PY}" "${ROOT}/scripts/visualpatchworld/evaluate_perception.py" \
    --candidate_source tool \
    --oracle_jsonl "${ROOT}/artifacts/visualpatchworld/lewm_fair_oracle/pusht_lewm_oracle_traj_val.jsonl" \
    --candidate_jsonl "${tool_data}/pusht_tool_traj_val.jsonl" \
    --output "${REPORT_DIR}/pusht_tool_vs_oracle_val.json" \
    2>&1 | tee "${LOG_DIR}/pusht_perception_tool.log" || true

  log "PushT: tool-live E2E planning (scaffold-fit, macro)"
  "${PY}" "${ROOT}/scripts/visualpatchworld/evaluate_planning.py" \
    --source tool \
    --results_dir "${tool_results}" \
    --data_dir "${tool_data}" \
    --planner cem_mpc \
    --observation_mode tool_live \
    --num_eval "${NUM_EVAL}" \
    --frameskip 5 --horizon 1 --model_rollout_horizon 1 --receding_horizon 1 \
    --num_samples "${NUM_SAMPLES}" --cem_iters "${CEM_ITERS}" \
    --output "${REPORT_DIR}/pusht_tool_live_scaffold_macro_n${NUM_EVAL}.json" \
    2>&1 | tee "${LOG_DIR}/pusht_tool_live_scaffold_macro.log"

  log "PushT: tool-live E2E planning (program-induced, frame)"
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
    --output "${REPORT_DIR}/pusht_tool_live_program_n${NUM_EVAL}.json" \
    2>&1 | tee "${LOG_DIR}/pusht_tool_live_program.log"
}

run_cross_env_perception() {
  for env in tworoom cube reacher; do
    log "${env}: oracle perception (state-derived export, RMSE=0 by construction)"
    "${PY}" - <<PY
import json
from pathlib import Path
out = Path("${REPORT_DIR}") / "${env}_oracle_perception.json"
out.write_text(json.dumps({
    "env": "${env}",
    "oracle_rmse": 0.0,
    "tool_rmse": None,
    "vlm_rmse": None,
    "object_precision": 1.0,
    "object_recall": 1.0,
    "note": "Graphs exported from privileged HDF5 state; pixel tool/VLM extractors not yet implemented for this env.",
}, indent=2))
print(out.read_text())
PY
  done
}

summarize() {
  log "=== Experiment summary ==="
  "${PY}" - <<'PY'
import json
from pathlib import Path
root = Path("${ROOT}/artifacts/visualpatchworld/reports")
patterns = [
    "cross_env_*_n50.json",
    "cross_env_*_hybrid*.json",
    "cross_env_*_sim_cem*.json",
    "pusht_tool_live*.json",
    "pusht_tool_vs_oracle*.json",
]
seen = set()
for pat in patterns:
    for path in sorted(root.glob(pat)):
        if path in seen:
            continue
        seen.add(path)
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        env = data.get("env", path.stem)
        sr = data.get("success_rate")
        gd = data.get("mean_goal_distance")
        planner = data.get("planner", "induced")
        sv = data.get("sim_verify_frac")
        extra = f" sv={sv}" if sv else ""
        print(f"{path.name}: env={env} planner={planner} success={sr} goal={gd}{extra}")
PY
}

ensure_mujoco

if [[ "${SKIP_CROSS_ENV:-0}" != "1" ]]; then
  for env in tworoom cube reacher; do
    run_cross_env_oracle "${env}"
    run_cross_env_sim_ceiling "${env}"
    run_cross_env_hybrid "${env}"
  done
  run_cross_env_perception
fi

if [[ "${SKIP_PUSHT_TOOL:-0}" != "1" ]]; then
  run_pusht_tool_pipeline
fi

summarize
log "Done. Reports in ${REPORT_DIR}"
