#!/usr/bin/env bash
# PushT with the contact shape identified from traces in the scene graph, plus rigid-body
# skills in the bank (optionally the full library program): induce, then refine.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-/home/comp/baijiaxin/miniconda3/envs/lewm/bin/python}
FAM=artifacts/visualpatchworld/llm_family
DATA=$HOME/le-wm-code/artifacts/visualpatchworld/data_stride1_sysid
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONPATH="$PWD/src:$PWD"
chain() {  # name model seed bank
  local name=$1 model=$2 seed=$3 bank=$4
  [[ -f $FAM/$name/family.json ]] || $PY -W ignore scripts/visualpatchworld/induce_llm_family.py --env pusht --model "$model" --seed "$seed" \
      --no_active --output_dir "$FAM/$name" --train_file "$DATA/pusht_oracle_traj_train.jsonl" --val_file "$DATA/pusht_oracle_traj_val.jsonl" \
      --max_train_episodes 40 --max_val_episodes 10 --bank_file "$bank" > "$FAM/$name.induce.log" 2>&1
  echo "[sysid-llm] induced $name rc=$?"
  grep -q '"elapsed_s"' "$FAM/refine_$name/refine.json" 2>/dev/null || $PY -W ignore scripts/visualpatchworld/refine_llm_family.py \
      --run_dir "$FAM/$name" --output_dir "$FAM/refine_$name" --bank_file "$bank" > "$FAM/refine_$name.refine.log" 2>&1
  echo "[sysid-llm] refined $name rc=$?"
}
QWEN=Qwen/Qwen3.5-397B-A17B
CODER=Qwen/Qwen3-Coder-480B-A35B-Instruct-Turbo
B=$FAM/bank
chain sysidskill_pusht_qwen35_s0 $QWEN 0 $B/qwen35_general_skills.json &
chain sysidskill_pusht_qwen35_s1 $QWEN 1 $B/qwen35_general_skills.json &
chain sysidskill_pusht_coder_s0 $CODER 0 $B/coder_general_skills.json &
chain sysidlib_pusht_qwen35_s0 $QWEN 0 $B/qwen35_general_skills_lib.json &
chain sysidlib_pusht_coder_s0 $CODER 0 $B/coder_general_skills_lib.json &
wait
echo "[sysid-llm] done"
