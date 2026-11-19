#!/usr/bin/env bash
# LLM-side revision experiments:
#   (1) Qwen3.5 proposal seeds 1 and 2 in Two-room, Cube and PushT (Reacher already has them);
#   (2) mechanism banks built without the "articulated chains" topic, used for Reacher.
# Each induction is followed by five-seed planning of the offline selections.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-/home/comp/baijiaxin/miniconda3/envs/lewm/bin/python}
FAM=artifacts/visualpatchworld/llm_family
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONPATH="$PWD/src:$PWD"

induce() {  # name env model seed [extra args...]
  local name=$1 env=$2 model=$3 seed=$4; shift 4
  if [[ -f $FAM/$name/family.json ]]; then echo "[revision] have $name"; return; fi
  $PY -W ignore scripts/visualpatchworld/induce_llm_family.py --env "$env" --model "$model" --seed "$seed" \
      --no_active --output_dir "$FAM/$name" "$@" > "$FAM/$name.induce.log" 2>&1
  echo "[revision] induced $name rc=$?"
}

bank_wo_chains() {  # tag model
  local out=$FAM/bank/$1_general_nochain.json
  [[ -f $out ]] || $PY scripts/visualpatchworld/build_mechanism_bank.py --model "$2" --out "$out" --drop_topics 2 \
      > "$FAM/bank/$1_build_nochain.log" 2>&1
  echo "[revision] bank $out"
}

QWEN=Qwen/Qwen3.5-397B-A17B
CODER=Qwen/Qwen3-Coder-480B-A35B-Instruct-Turbo
GPTOSS=openai/gpt-oss-120b

(
  bank_wo_chains coder $CODER
  induce bank_nochain_reacher_coder_s0 reacher $CODER 0 --bank_file $FAM/bank/coder_general_nochain.json
) &
(
  bank_wo_chains gptoss $GPTOSS
  induce bank_nochain_reacher_gptoss_s0 reacher $GPTOSS 0 --bank_file $FAM/bank/gptoss_general_nochain.json
) &
for env in tworoom cube pusht; do
  for s in 1 2; do
    induce v1_${env}_qwen35_s$s $env $QWEN $s &
  done
done
wait

RUNS="bank_nochain_reacher_coder_s0 bank_nochain_reacher_gptoss_s0"
for env in tworoom cube pusht; do for s in 1 2; do RUNS="$RUNS v1_${env}_qwen35_s$s"; done; done
RUNS="$RUNS v1_reacher_qwen35_s1 v1_reacher_qwen35_s2"
$PY scripts/visualpatchworld/queue_llm_family_evals.py --runs $RUNS --seeds 42,43,44,45,46 --max_parallel 24 \
    --selections offline_one_step,offline_multi_step
echo "[revision] done"
