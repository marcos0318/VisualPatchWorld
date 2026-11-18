#!/usr/bin/env bash
# Induce one code baseline on all envs with a chosen LLM (no planning).
# Usage: run_code_baselines_model.sh METHOD MODEL OUT_ROOT [ENVS...]
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
METHOD="$1"; MODEL="$2"; OUT_ROOT="$3"; shift 3
ENVS=("${@:-tworoom cube reacher pusht}")
[ "$#" -eq 0 ] && ENVS=(tworoom cube reacher pusht)
PY="${PYTHON:-/home/comp/baijiaxin/miniconda3/envs/lewm/bin/python}"
DATA="${DATA_DIR:-/home/comp/baijiaxin/le-wm-code/artifacts/visualpatchworld/data}"
export PYTHONPATH="${ROOT}:${ROOT}/src:${PYTHONPATH:-}"
export PATCHWORLD_LLM_REQUEST_TIMEOUT_S="${PATCHWORLD_LLM_REQUEST_TIMEOUT_S:-600}"
export PATCHWORLD_LLM_MAX_RETRIES="${PATCHWORLD_LLM_MAX_RETRIES:-12}"
export PATCHWORLD_LLM_CACHE="${PATCHWORLD_LLM_CACHE:-1}"
mkdir -p "$OUT_ROOT/logs"
for env in "${ENVS[@]}"; do
  if ls "$OUT_ROOT/$METHOD/$env/generated_models/"*.py >/dev/null 2>&1; then
    echo "[skip] $METHOD/$env"; continue
  fi
  echo "[$(date -Iseconds)] START $METHOD/$env"
  "$PY" scripts/visualpatchworld/run_code_baselines.py --method "$METHOD" --env "$env" \
    --data_dir "$DATA" --output_dir "$OUT_ROOT/$METHOD/$env" --model "$MODEL" \
    --max_train 24 --max_eval 12 --max_refine_rounds 3 --poeworld_batches 3 \
    --poeworld_experts_per_batch 4 --gif_mcts_iterations 50 --pomdp_coder_candidates 3 \
    --cwm_retries 50 --seed 42 --skip_planning > "$OUT_ROOT/logs/${METHOD}_${env}.log" 2>&1
  echo "[$(date -Iseconds)] DONE $METHOD/$env rc=$?"
done
