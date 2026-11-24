#!/usr/bin/env bash
# One refinement configuration for every environment: LLM-only bank, verified solver helpers,
# identified pusher and state limits (each skipped where the traces contain none).
# Usage: run_unified.sh <env> <seed> <solver_bank_json> <tag>
set -euo pipefail
ENV=$1
SEED=$2
SOLVERS=$3
TAG=$4
cd "$(dirname "$0")/../.."
PY=/home/comp/baijiaxin/miniconda3/envs/lewm/bin/python
LF=artifacts/visualpatchworld/llm_family
if [ "$ENV" = pusht ]; then START=$LF/refine_vsolverstart_pusht_qwen35_s$SEED; else START=$LF/v1_${ENV}_qwen35_s$SEED; fi
OUT=$LF/${TAG}_${ENV}_qwen35_s$SEED
if [ ! -f "$OUT/refine.json" ]; then
  $PY -u scripts/visualpatchworld/refine_llm_family.py --run_dir "$START" --output_dir "$OUT" \
    --rounds 4 --num_revisions 3 --bank_file $LF/bank/qwen35_general.json --solver_helpers "$SOLVERS" \
    --pusher_dynamics --pusher_helper --state_limits --seed "$SEED" --structured_fit \
    --balance_modes --mode_feedback --select balanced --select_h 25 --cover_modes \
    --val_episodes 20 --fit_timeout 2400 > "$OUT.refine.log" 2>&1
fi
echo "done: $OUT"
