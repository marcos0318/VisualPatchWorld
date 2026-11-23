#!/usr/bin/env bash
# PushT with an LLM-written, test-verified solver bank instead of the hand-written contact kit.
# Usage: run_llmsolver_pusht.sh <seed> <bank_json> <tag>
set -euo pipefail
SEED=$1
BANK=$2
TAG=$3
cd "$(dirname "$0")/../.."
PY=/home/comp/baijiaxin/miniconda3/envs/lewm/bin/python
LF=artifacts/visualpatchworld/llm_family
D1=/home/comp/baijiaxin/le-wm-code/artifacts/visualpatchworld/data_stride1
START=$LF/${TAG}start_pusht_qwen35_s$SEED
REF=$LF/${TAG}_pusht_qwen35_s$SEED

if [ ! -f "$START/models/offline_multi_step.py" ]; then
  $PY -u scripts/visualpatchworld/induce_llm_family.py --env pusht \
    --train_file $D1/pusht_oracle_traj_train.jsonl --val_file $D1/pusht_oracle_traj_val.jsonl \
    --max_train_episodes 40 --max_val_episodes 10 --output_dir "$START" \
    --model Qwen/Qwen3.5-397B-A17B --seed "$SEED" --no_active --bank_file "$BANK" \
    --analyze --structured_fit > "$START.induce.log" 2>&1
fi
if [ ! -f "$REF/models/offline_multi_step.py" ]; then
  $PY -u scripts/visualpatchworld/refine_llm_family.py --run_dir "$START" --output_dir "$REF" \
    --rounds 4 --num_revisions 3 --bank_file "$BANK" --seed "$SEED" --structured_fit \
    --balance_modes --mode_feedback --select balanced --select_h 25 --cover_modes \
    --val_episodes 20 --fit_timeout 2400 > "$REF.refine.log" 2>&1
fi
echo "done: $START $REF"
