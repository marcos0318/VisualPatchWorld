#!/usr/bin/env bash
# Full method on PushT: LLM family + identified pusher/body program (--sysid), plain traces, no bank.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-/home/comp/baijiaxin/miniconda3/envs/lewm/bin/python}
FAM=artifacts/visualpatchworld/llm_family
DATA=$HOME/le-wm-code/artifacts/visualpatchworld/data_stride1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONPATH="$PWD/src:$PWD"
induce() {  # name model seed
  [[ -f $FAM/$1/family.json ]] && { echo "[sysid-fam] have $1"; return; }
  $PY -W ignore scripts/visualpatchworld/induce_llm_family.py --env pusht --model "$2" --seed "$3" --no_active --sysid \
      --output_dir "$FAM/$1" --train_file "$DATA/pusht_oracle_traj_train.jsonl" --val_file "$DATA/pusht_oracle_traj_val.jsonl" \
      --max_train_episodes 40 --max_val_episodes 10 > "$FAM/$1.induce.log" 2>&1
  echo "[sysid-fam] $1 rc=$?"
}
induce sysidfam_pusht_qwen35_s0 Qwen/Qwen3.5-397B-A17B 0 &
induce sysidfam_pusht_qwen35_s1 Qwen/Qwen3.5-397B-A17B 1 &
induce sysidfam_pusht_qwen35_s2 Qwen/Qwen3.5-397B-A17B 2 &
induce sysidfam_pusht_coder_s0 Qwen/Qwen3-Coder-480B-A35B-Instruct-Turbo 0 &
induce sysidfam_pusht_gptoss_s0 openai/gpt-oss-120b 0 &
wait
