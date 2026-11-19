#!/usr/bin/env bash
# PushT with static shape metadata in the scene graph: does the proposer write a contact model?
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-/home/comp/baijiaxin/miniconda3/envs/lewm/bin/python}
FAM=artifacts/visualpatchworld/llm_family
GEOM=$HOME/le-wm-code/artifacts/visualpatchworld/data_stride1_geom
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONPATH="$PWD/src:$PWD"

induce() {  # name model seed [extra args...]
  local name=$1 model=$2 seed=$3; shift 3
  if [[ -f $FAM/$name/family.json ]]; then echo "[geom] have $name"; return; fi
  $PY -W ignore scripts/visualpatchworld/induce_llm_family.py --env pusht --model "$model" --seed "$seed" \
      --no_active --output_dir "$FAM/$name" --train_file "$GEOM/pusht_oracle_traj_train.jsonl" \
      --val_file "$GEOM/pusht_oracle_traj_val.jsonl" --max_train_episodes 40 --max_val_episodes 10 "$@" \
      > "$FAM/$name.induce.log" 2>&1
  echo "[geom] induced $name rc=$?"
}

QWEN=Qwen/Qwen3.5-397B-A17B
CODER=Qwen/Qwen3-Coder-480B-A35B-Instruct-Turbo
GPTOSS=openai/gpt-oss-120b
BANK=$FAM/bank
if [[ ${KIT:-0} == 1 ]]; then
  induce geomkit_pusht_qwen35_s0 $QWEN 0 --bank_file $BANK/qwen35_general_contactkit.json &
  induce geomkit_pusht_qwen35_s1 $QWEN 1 --bank_file $BANK/qwen35_general_contactkit.json &
  induce geomkit_pusht_qwen35_s2 $QWEN 2 --bank_file $BANK/qwen35_general_contactkit.json &
  induce geomkit_pusht_coder_s0 $CODER 0 --bank_file $BANK/coder_general_contactkit.json &
  induce geomkit_pusht_gptoss_s0 $GPTOSS 0 --bank_file $BANK/gptoss_general_contactkit.json &
  GEOM=$HOME/le-wm-code/artifacts/visualpatchworld/data_stride1 induce kit_pusht_qwen35_s0 $QWEN 0 --bank_file $BANK/qwen35_general_contactkit.json &
  wait
  echo "[geom] kit done"
  exit 0
fi
induce geom_pusht_qwen35_s0 $QWEN 0 &
induce geom_pusht_qwen35_s1 $QWEN 1 &
induce geom_pusht_qwen35_s2 $QWEN 2 &
induce geom_pusht_coder_s0 $CODER 0 &
induce geom_pusht_gptoss_s0 $GPTOSS 0 &
induce geombank_pusht_qwen35_s0 $QWEN 0 --bank_file $FAM/bank/qwen35_general.json &
induce geombank_pusht_coder_s0 $CODER 0 --bank_file $FAM/bank/coder_general.json &
wait
echo "[geom] done"
