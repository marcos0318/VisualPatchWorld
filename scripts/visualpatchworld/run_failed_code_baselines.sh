#!/usr/bin/env bash
# Rerun programmatic baselines still missing planning results (6 jobs).
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-${CONDA_PREFIX:-}/envs/lewm/bin/python3}"
export PYTHONPATH="${ROOT}:${ROOT}/src:${PYTHONPATH:-}"
export PATCHWORLD_LLM_REQUEST_TIMEOUT_S="${PATCHWORLD_LLM_REQUEST_TIMEOUT_S:-600}"
export PATCHWORLD_LLM_MAX_RETRIES="${PATCHWORLD_LLM_MAX_RETRIES:-12}"
export PATCHWORLD_LLM_RETRY_BASE_SLEEP_S="${PATCHWORLD_LLM_RETRY_BASE_SLEEP_S:-3}"
export PATCHWORLD_LLM_RETRY_MAX_SLEEP_S="${PATCHWORLD_LLM_RETRY_MAX_SLEEP_S:-120}"
export PATCHWORLD_LLM_MAX_RETRY_WAIT_S="${PATCHWORLD_LLM_MAX_RETRY_WAIT_S:-1800}"
export PATCHWORLD_LLM_CACHE="${PATCHWORLD_LLM_CACHE:-1}"
export PATCHWORLD_LLM_CACHE_PATH="${PATCHWORLD_LLM_CACHE_PATH:-$ROOT/artifacts/visualpatchworld/llm_cache.sqlite3}"

LOG_DIR="artifacts/visualpatchworld/code_baselines/logs"
STATUS_FILE="${LOG_DIR}/run_status_rerun2.tsv"
mkdir -p "$LOG_DIR"

: > "$STATUS_FILE"
echo -e "method\tenv\tstatus\tstarted\tfinished" >> "$STATUS_FILE"

run_one() {
  local method="$1"
  local env="$2"
  local log="${LOG_DIR}/${method}_${env}_rerun2.log"
  local started finished status

  started="$(date -Iseconds)"
  echo "[$(date -Iseconds)] START ${method}/${env} -> ${log}"

  if "$PY" scripts/visualpatchworld/run_code_baselines.py \
    --method "$method" \
    --env "$env" \
    --max_train 24 \
    --max_eval 12 \
    --max_refine_rounds 3 \
    --poeworld_batches 3 \
    --poeworld_experts_per_batch 4 \
    --gif_mcts_iterations 50 \
    --num_eval 50 \
    --seed 42 \
    > "$log" 2>&1; then
    status="ok"
  else
    status="fail"
  fi

  finished="$(date -Iseconds)"
  echo -e "${method}\t${env}\t${status}\t${started}\t${finished}" >> "$STATUS_FILE"
  echo "[$(date -Iseconds)] DONE ${method}/${env} status=${status}"
}

# worldcoder/reacher already succeeded in rerun1 (0% measured)
run_one worldcoder pusht
run_one poeworld pusht
run_one gif_mcts tworoom
run_one gif_mcts cube
run_one gif_mcts reacher
run_one gif_mcts pusht

echo "[$(date -Iseconds)] Rerun batch finished. Status: ${STATUS_FILE}"
"$PY" scripts/visualpatchworld/collect_code_baseline_results.py --write "${LOG_DIR}/summary_rerun2.json"
