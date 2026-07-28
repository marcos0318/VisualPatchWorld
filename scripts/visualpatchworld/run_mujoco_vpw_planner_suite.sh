#!/usr/bin/env bash
# Fair PushT planner suite: MuJoCo GT vs VPW program, CEM-MPC + beam-MPC.
#
# Protocol (paper program eval):
#   n=50, seed=42, budget=50, frameskip=1, horizon=5, receding=1
# Planners:
#   cem_mpc  — continuous CEM (VPW: 600x15; MuJoCo fair macro: action_block=1)
#   beam_mpc — discrete contact-aware beam search (beam_width from --num_samples)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PY:-${CONDA_PREFIX:-}/envs/lewm/bin/python3}"
export STABLEWM_HOME="${STABLEWM_HOME:-${HOME}/.stable-wm}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export PYTHONPATH="${ROOT}:${ROOT}/src:${PYTHONPATH:-}"

REPORT="${ROOT}/artifacts/visualpatchworld/reports/planner_suite"
LOG="${ROOT}/logs/planner_suite"
MODEL="${MODEL:-${ROOT}/artifacts/visualpatchworld/induced/program_induced_best.py}"
NUM_EVAL="${NUM_EVAL:-50}"
SEED="${SEED:-42}"
CEM_SAMPLES="${CEM_SAMPLES:-600}"
CEM_ITERS="${CEM_ITERS:-15}"
BEAM_WIDTH="${BEAM_WIDTH:-16}"
FORCE="${FORCE:-0}"

mkdir -p "$REPORT" "$LOG"
log() { echo "[planner-suite][$(date +%H:%M:%S)] $*"; }

run_if_needed() {
  local out="$1"; shift
  if [[ -f "$out" && "$FORCE" != "1" ]]; then
    log "skip existing $out"
    return 0
  fi
  log "run -> $out"
  "$@" 2>&1 | tee "${LOG}/$(basename "$out" .json).log"
}

# --- VPW program ---
VPW_CEM_OUT="$REPORT/vpw_cem_mpc_n${NUM_EVAL}_seed${SEED}.json"
PRIOR_VPW_CEM="${ROOT}/artifacts/visualpatchworld/reports/pusht_oracle_program_n50_seed42.json"
if [[ ! -f "$VPW_CEM_OUT" && -f "$PRIOR_VPW_CEM" && "$NUM_EVAL" == "50" && "$SEED" == "42" && "$CEM_SAMPLES" == "600" && "$CEM_ITERS" == "15" && "$FORCE" != "1" ]]; then
  cp "$PRIOR_VPW_CEM" "$VPW_CEM_OUT"
  log "reused paper VPW CEM report -> $VPW_CEM_OUT"
else
  run_if_needed "$VPW_CEM_OUT" \
    "$PY" "$ROOT/scripts/visualpatchworld/evaluate_planning.py" \
      --source oracle \
      --model_path "$MODEL" \
      --results_dir "${ROOT}/artifacts/visualpatchworld/induced" \
      --planner cem_mpc \
      --observation_mode oracle \
      --num_eval "$NUM_EVAL" --seed "$SEED" \
      --goal_offset_steps 25 --eval_budget 50 \
      --frameskip 1 --horizon 5 --model_rollout_horizon 5 --receding_horizon 1 \
      --model_action_repeat 1 \
      --num_samples "$CEM_SAMPLES" --cem_iters "$CEM_ITERS" \
      --output "$VPW_CEM_OUT"
fi

run_if_needed "$REPORT/vpw_beam_mpc_n${NUM_EVAL}_seed${SEED}.json" \
  "$PY" "$ROOT/scripts/visualpatchworld/evaluate_planning.py" \
    --source oracle \
    --model_path "$MODEL" \
    --results_dir "${ROOT}/artifacts/visualpatchworld/induced" \
    --planner beam_mpc \
    --observation_mode oracle \
    --num_eval "$NUM_EVAL" --seed "$SEED" \
    --goal_offset_steps 25 --eval_budget 50 \
    --frameskip 1 --horizon 5 --model_rollout_horizon 5 --receding_horizon 1 \
    --model_action_repeat 1 \
    --num_samples "$BEAM_WIDTH" \
    --output "$REPORT/vpw_beam_mpc_n${NUM_EVAL}_seed${SEED}.json"

# --- MuJoCo GT ---
# Macro-fair CEM uses a lighter budget than VPW 600x15: each CEM candidate is a
# true sim rollout, so 600x15×50 replans is multi-hour. 300x10 remains strong.
MUJOCO_CEM_SAMPLES="${MUJOCO_CEM_SAMPLES:-300}"
MUJOCO_CEM_ITERS="${MUJOCO_CEM_ITERS:-10}"
run_if_needed "$REPORT/mujoco_cem_mpc_macro_n${NUM_EVAL}_seed${SEED}.json" \
  "$PY" "$ROOT/scripts/visualpatchworld/evaluate_simulator_planning.py" \
    --planner cem_mpc \
    --num_eval "$NUM_EVAL" --seed "$SEED" \
    --goal_offset_steps 25 --eval_budget 50 \
    --frameskip 1 --horizon 5 --action_block 1 --receding_horizon 1 \
    --num_samples "$MUJOCO_CEM_SAMPLES" --cem_iters "$MUJOCO_CEM_ITERS" \
    --output "$REPORT/mujoco_cem_mpc_macro_n${NUM_EVAL}_seed${SEED}.json"

run_if_needed "$REPORT/mujoco_beam_mpc_n${NUM_EVAL}_seed${SEED}.json" \
  "$PY" "$ROOT/scripts/visualpatchworld/evaluate_simulator_planning.py" \
    --planner beam_mpc \
    --num_eval "$NUM_EVAL" --seed "$SEED" \
    --goal_offset_steps 25 --eval_budget 50 \
    --frameskip 1 --horizon 5 --receding_horizon 1 \
    --num_samples "$BEAM_WIDTH" \
    --output "$REPORT/mujoco_beam_mpc_n${NUM_EVAL}_seed${SEED}.json"

# Optional high ceiling: classic sim CEM with action_block=5 (reuse if present)
CEILING="$REPORT/mujoco_cem_mpc_block5_n${NUM_EVAL}_seed${SEED}.json"
PRIOR="${ROOT}/artifacts/visualpatchworld/reports/pusht_planning_simulator_cem_mpc_600x15_ground_truth_n50.json"
if [[ ! -f "$CEILING" && -f "$PRIOR" && "$NUM_EVAL" == "50" && "$SEED" == "42" ]]; then
  cp "$PRIOR" "$CEILING"
  log "copied prior block5 ceiling -> $CEILING"
elif [[ ! -f "$CEILING" || "$FORCE" == "1" ]]; then
  run_if_needed "$CEILING" \
    "$PY" "$ROOT/scripts/visualpatchworld/evaluate_simulator_planning.py" \
      --planner cem_mpc \
      --num_eval "$NUM_EVAL" --seed "$SEED" \
      --goal_offset_steps 25 --eval_budget 50 \
      --frameskip 1 --horizon 5 --action_block 5 --receding_horizon 5 \
      --num_samples "$CEM_SAMPLES" --cem_iters "$CEM_ITERS" \
      --output "$CEILING"
fi

"$PY" - <<PY
import json
from pathlib import Path
from datetime import datetime

report = Path("$REPORT")
paths = {
    "vpw_cem_mpc": report / "vpw_cem_mpc_n${NUM_EVAL}_seed${SEED}.json",
    "vpw_beam_mpc": report / "vpw_beam_mpc_n${NUM_EVAL}_seed${SEED}.json",
    "mujoco_cem_mpc_macro": report / "mujoco_cem_mpc_macro_n${NUM_EVAL}_seed${SEED}.json",
    "mujoco_beam_mpc": report / "mujoco_beam_mpc_n${NUM_EVAL}_seed${SEED}.json",
    "mujoco_cem_mpc_block5_ceiling": report / "mujoco_cem_mpc_block5_n${NUM_EVAL}_seed${SEED}.json",
}
arms = {}
for name, path in paths.items():
    if not path.exists():
        continue
    d = json.loads(path.read_text())
    arms[name] = {
        "success_rate": d.get("success_rate"),
        "mean_goal_distance": d.get("mean_goal_distance"),
        "planner": d.get("planner"),
        "beam_width": d.get("beam_width"),
        "num_samples": d.get("num_samples"),
        "cem_iters": d.get("cem_iters"),
        "frameskip": d.get("frameskip"),
        "horizon": d.get("horizon"),
        "receding_horizon": d.get("receding_horizon"),
        "action_block": d.get("action_block"),
        "model_path": d.get("model_path"),
        "model_type": d.get("model_type"),
        "report": str(path),
    }
    print(f"{name:32s} success={d.get('success_rate')} dist={d.get('mean_goal_distance')}")

summary = {
    "timestamp": datetime.now().isoformat(timespec="seconds"),
    "protocol": {
        "num_eval": int("$NUM_EVAL"),
        "seed": int("$SEED"),
        "eval_budget": 50,
        "frameskip": 1,
        "horizon": 5,
        "receding_horizon": 1,
        "cem": {"num_samples": int("$CEM_SAMPLES"), "cem_iters": int("$CEM_ITERS")},
        "beam_width": int("$BEAM_WIDTH"),
        "vpw_model": "$MODEL",
        "notes": [
            "Fair macro protocol: plan H=5 macros, execute 1, replan (frameskip=1).",
            "beam_mpc uses directional+speed alphabet plus contact-aware macros.",
            "mujoco_cem_mpc_block5_ceiling is the stronger per-step CEM (action_block=5).",
        ],
    },
    "arms": arms,
}
out = report / "summary_n${NUM_EVAL}_seed${SEED}.json"
out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
print("wrote", out)
PY

log "done"
