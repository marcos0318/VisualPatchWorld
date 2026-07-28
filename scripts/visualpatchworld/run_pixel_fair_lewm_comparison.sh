#!/usr/bin/env bash
# Pixel-fair VisualPatchWorld vs LeWM comparison (PushT first).
#
# Shared eval protocol (matches LeWM config/eval/pusht.yaml):
#   n=50, seed=42, goal_offset_steps=25, eval_budget=50
#
# LeWM (reference, from paper / local eval):
#   RGB pixels -> latent JEPA -> CEM
#   horizon=5, receding_horizon=5, action_block(frameskip)=5
#   CEM: 300 samples x 30 iterations (solver/cem.yaml)
#
# VPW pixel-fair (this script):
#   RGB render -> CV tool extractor -> scene graph -> induced Python dynamics -> CEM-MPC
#   tool_live re-observation at every replan (no simulator state in observations)
#
# VPW uses its best-known induced-model planner protocol (frame resolution for
# program-induced models). This is documented in the comparison JSON because
# induced code rollouts drift under long horizons, unlike latent LeWM.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON:-${CONDA_PREFIX:-}/envs/lewm/bin/python3}"
export STABLEWM_HOME="${STABLEWM_HOME:-${HOME}/.stable-wm}"
export MUJOCO_GL=egl
export PYTHONPATH="${ROOT}/src:${ROOT}:${PYTHONPATH:-}"

REPORT_DIR="${REPORT_DIR:-${ROOT}/artifacts/visualpatchworld/reports}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs/paper_experiments}"
NUM_EVAL="${NUM_EVAL:-50}"
SEED="${SEED:-42}"
NUM_SAMPLES="${NUM_SAMPLES:-300}"
CEM_ITERS="${CEM_ITERS:-30}"
PROGRAM_MODEL="${PROGRAM_MODEL:-${ROOT}/artifacts/visualpatchworld/induced/program_induced_best.py}"
SCAFFOLD_RESULTS="${SCAFFOLD_RESULTS:-${ROOT}/artifacts/visualpatchworld/lewm_tool_scaffold_fit_results}"
COMPARISON_OUT="${COMPARISON_OUT:-${REPORT_DIR}/pixel_fair_lewm_comparison_n${NUM_EVAL}.json}"

mkdir -p "${REPORT_DIR}" "${LOG_DIR}"

log() {
  echo "[pixel-fair][$(date '+%H:%M:%S')] $*"
}

run_vpw() {
  local label="$1"
  shift
  local out="$1"
  shift
  if [[ -f "${out}" && "${FORCE:-0}" != "1" ]]; then
    log "Skip existing ${label} -> ${out}"
    return 0
  fi
  log "Run ${label}"
  "$@" --output "${out}" 2>&1 | tee "${LOG_DIR}/${label}.log"
}

# --- VPW tool-live: scaffold-fit (weakest induced baseline, macro protocol) ---
run_vpw "pusht_tool_live_scaffold" \
  "${REPORT_DIR}/pusht_tool_live_scaffold_macro_n${NUM_EVAL}.json" \
  "${PY}" "${ROOT}/scripts/visualpatchworld/evaluate_planning.py" \
    --source tool \
    --results_dir "${SCAFFOLD_RESULTS}" \
    --planner cem_mpc \
    --observation_mode tool_live \
    --num_eval "${NUM_EVAL}" \
    --seed "${SEED}" \
    --goal_offset_steps 25 \
    --eval_budget 50 \
    --frameskip 5 \
    --horizon 1 \
    --model_rollout_horizon 1 \
    --receding_horizon 1 \
    --num_samples "${NUM_SAMPLES}" \
    --cem_iters "${CEM_ITERS}"

# --- VPW tool-live: program-induced (best pure induced model, frame protocol) ---
if [[ -f "${PROGRAM_MODEL}" ]]; then
  run_vpw "pusht_tool_live_program" \
    "${REPORT_DIR}/pusht_tool_live_program_n${NUM_EVAL}.json" \
    "${PY}" "${ROOT}/scripts/visualpatchworld/evaluate_planning.py" \
      --source oracle \
      --model_path "${PROGRAM_MODEL}" \
      --results_dir "${ROOT}/artifacts/visualpatchworld/induced" \
      --planner cem_mpc \
      --observation_mode tool_live \
      --expose_block_velocity \
      --num_eval "${NUM_EVAL}" \
      --seed "${SEED}" \
      --goal_offset_steps 25 \
      --eval_budget 50 \
      --frameskip 1 \
      --horizon 5 \
      --model_rollout_horizon 5 \
      --receding_horizon 1 \
      --model_action_repeat 1 \
      --num_samples "${NUM_SAMPLES}" \
      --cem_iters "${CEM_ITERS}"
else
  log "Missing program model ${PROGRAM_MODEL}; skipping program-induced tool-live"
fi

# --- Oracle-graph upper bounds (same models, privileged perception) ---
run_vpw "pusht_oracle_scaffold" \
  "${REPORT_DIR}/pusht_oracle_scaffold_macro_n${NUM_EVAL}.json" \
  "${PY}" "${ROOT}/scripts/visualpatchworld/evaluate_planning.py" \
    --source tool \
    --results_dir "${SCAFFOLD_RESULTS}" \
    --planner cem_mpc \
    --observation_mode oracle \
    --num_eval "${NUM_EVAL}" \
    --seed "${SEED}" \
    --goal_offset_steps 25 \
    --eval_budget 50 \
    --frameskip 5 \
    --horizon 1 \
    --model_rollout_horizon 1 \
    --receding_horizon 1 \
    --num_samples "${NUM_SAMPLES}" \
    --cem_iters "${CEM_ITERS}"

if [[ -f "${PROGRAM_MODEL}" ]]; then
  run_vpw "pusht_oracle_program" \
    "${REPORT_DIR}/pusht_oracle_program_frame_n${NUM_EVAL}.json" \
    "${PY}" "${ROOT}/scripts/visualpatchworld/evaluate_planning.py" \
      --source oracle \
      --model_path "${PROGRAM_MODEL}" \
      --results_dir "${ROOT}/artifacts/visualpatchworld/induced" \
      --planner cem_mpc \
      --observation_mode oracle \
      --expose_block_velocity \
      --num_eval "${NUM_EVAL}" \
      --seed "${SEED}" \
      --goal_offset_steps 25 \
      --eval_budget 50 \
      --frameskip 1 \
      --horizon 5 \
      --model_rollout_horizon 5 \
      --receding_horizon 1 \
      --model_action_repeat 1 \
      --num_samples "${NUM_SAMPLES}" \
      --cem_iters "${CEM_ITERS}"
fi

log "Writing comparison table ${COMPARISON_OUT}"
"${PY}" - <<'PY' "${COMPARISON_OUT}" "${REPORT_DIR}" "${NUM_EVAL}"
import json
import sys
from datetime import datetime
from pathlib import Path

out_path = Path(sys.argv[1])
report_dir = Path(sys.argv[2])
num_eval = int(sys.argv[3])

def load(name: str):
    path = report_dir / name
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    data["_path"] = str(path)
    return data

def row(label: str, data: dict | None) -> dict:
    if data is None:
        return {"label": label, "missing": True}
    return {
        "label": label,
        "success_rate": data.get("success_rate"),
        "mean_goal_distance": data.get("mean_goal_distance"),
        "observation_mode": data.get("observation_mode"),
        "planner": data.get("planner"),
        "frameskip": data.get("frameskip"),
        "horizon": data.get("horizon"),
        "receding_horizon": data.get("receding_horizon"),
        "num_samples": data.get("num_samples"),
        "cem_iters": data.get("cem_iters"),
        "tool_extraction_failures": data.get("tool_extraction_failures"),
        "report": data.get("_path"),
    }

comparison = {
    "timestamp": datetime.now().isoformat(),
    "shared_protocol": {
        "num_eval": num_eval,
        "seed": 42,
        "goal_offset_steps": 25,
        "eval_budget": 50,
        "dataset": "pusht_expert_train.h5",
        "success_criterion": "LeWM PushT goal overlap",
    },
    "lewm_reference": {
        "method": "RGB pixels -> latent JEPA -> CEM",
        "pusht_success_rate": 94,
        "note": "Local eval 2026-06-22; paper reports 96. CEM 300x30, h=5, r=5, action_block=5.",
    },
    "vpw_pixel_fair": [
        row("tool_live_scaffold_macro", load(f"pusht_tool_live_scaffold_macro_n{num_eval}.json")),
        row("tool_live_program_frame", load(f"pusht_tool_live_program_n{num_eval}.json")),
    ],
    "vpw_oracle_upper_bound": [
        row("oracle_scaffold_macro", load(f"pusht_oracle_scaffold_macro_n{num_eval}.json")),
        row("oracle_program_frame", load(f"pusht_oracle_program_frame_n{num_eval}.json")),
    ],
    "interpretation": [
        "Compare VPW tool_live rows to lewm_reference for the pixel-fair claim.",
        "Compare tool_live vs oracle rows to isolate perception vs dynamics/planner gaps.",
        "Cross-env pixel-fair rows require env-specific tool extractors (tworoom/cube/reacher).",
    ],
}
out_path.write_text(json.dumps(comparison, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps(comparison, indent=2, sort_keys=True))
PY

log "Done."
