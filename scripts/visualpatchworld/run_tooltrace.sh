#!/usr/bin/env bash
# Induce Qwen3.5 families from tool (RGB) traces instead of oracle traces, then plan
# with oracle and tool replanning over five seeds.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-/home/comp/baijiaxin/miniconda3/envs/lewm/bin/python}
FAM=artifacts/visualpatchworld/llm_family
TOOL=artifacts/visualpatchworld/data_tool
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONPATH="$PWD/src:$PWD"
ENVS="tworoom reacher cube"

for env in $ENVS; do
  ( [[ -f $TOOL/${env}_tool_traj_train.jsonl ]] || $PY -W ignore scripts/visualpatchworld/export_tool_traces.py --env $env --splits train,val
    name=tooltrace_${env}_qwen35_s0
    [[ -f $FAM/$name/family.json ]] || $PY -W ignore scripts/visualpatchworld/induce_llm_family.py --env $env \
        --model Qwen/Qwen3.5-397B-A17B --seed 0 --no_active --output_dir $FAM/$name \
        --train_file $TOOL/${env}_tool_traj_train.jsonl --val_file $TOOL/${env}_tool_traj_val.jsonl \
        > $FAM/$name.induce.log 2>&1
    echo "[tooltrace] induced $name rc=$?" ) &
done
wait
$PY scripts/visualpatchworld/queue_llm_family_evals.py --runs $(for e in $ENVS; do echo tooltrace_${e}_qwen35_s0; done) \
    --seeds 42,43,44,45,46 --max_parallel 15 --selections offline_multi_step
$PY scripts/visualpatchworld/run_vpw_modes.py --sets qwen35_tooltrace --modes tool_induced --envs tworoom,reacher,cube \
    --max_parallel 15 --workers 4
echo "[tooltrace] done"
