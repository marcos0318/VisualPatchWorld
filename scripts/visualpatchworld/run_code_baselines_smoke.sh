#!/usr/bin/env bash
# Minimal smoke test: all programmatic baselines on tworoom (induction only, tiny budget).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-${CONDA_PREFIX:-}/envs/lewm/bin/python3}"
export PYTHONPATH="${ROOT}:${ROOT}/src:${PYTHONPATH:-}"

LOG_DIR="artifacts/visualpatchworld/code_baselines/logs/smoke_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

METHODS=(tracepatch worldcoder poeworld gif_mcts pomdp_coder cwm_game)
ENV=tworoom
STATUS="$LOG_DIR/smoke_status.tsv"
echo -e "method\tstatus\tlog" > "$STATUS"

run_smoke() {
  local method="$1"
  local log="$LOG_DIR/${method}_${ENV}.log"
  echo "=== smoke ${method}/${ENV} ==="
  if "$PY" scripts/visualpatchworld/run_code_baselines.py \
    --method "$method" \
    --env "$ENV" \
    --max_train 6 \
    --max_eval 4 \
    --max_refine_rounds 1 \
    --worldcoder_programs 1 \
    --poeworld_batches 1 \
    --poeworld_experts_per_batch 2 \
    --gif_mcts_iterations 6 \
    --pomdp_coder_candidates 1 \
    --cwm_retries 6 \
    --skip_planning \
    > "$log" 2>&1; then
    echo -e "${method}\tok\t${log}" >> "$STATUS"
    echo "OK ${method}"
  else
    echo -e "${method}\tfail\t${log}" >> "$STATUS"
    echo "FAIL ${method} (see $log)"
    return 1
  fi
}

failed=0
for method in "${METHODS[@]}"; do
  run_smoke "$method" || failed=$((failed + 1))
done

echo "--- smoke summary ---"
cat "$STATUS"
exit "$failed"
