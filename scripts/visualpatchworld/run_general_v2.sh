#!/usr/bin/env bash
# General pipeline v2 on every environment: derived metadata (--analyze), structured fitting,
# general bank + rigid-body skills, then residual-driven refinement. Data and episode budgets
# are copied from the v1 Qwen3.5 run of each environment.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-/home/comp/baijiaxin/miniconda3/envs/lewm/bin/python}
FAM=artifacts/visualpatchworld/llm_family
BANK=$FAM/bank/qwen35_general_skills.json
QWEN=Qwen/Qwen3.5-397B-A17B
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONPATH="$PWD/src:$PWD"
v1_args() {  # env -> induction data args of the v1 run
  $PY - "$1" <<'EOF'
import json, sys
a = json.load(open(f"artifacts/visualpatchworld/llm_family/v1_{sys.argv[1]}_qwen35_s0/family.json"))["args"]
out = ["--data_dir", a["data_dir"]]
for k in ("train_file", "val_file", "max_train_episodes", "max_val_episodes"):
    if a.get(k) is not None:
        out += [f"--{k}", str(a[k])]
print(" ".join(out))
EOF
}
run() {  # env seed
  local env=$1 seed=$2 name=v2_$1_qwen35_s$2
  if [[ ! -f $FAM/$name/family.json ]]; then
    $PY -W ignore scripts/visualpatchworld/induce_llm_family.py --env "$env" --model $QWEN --seed "$seed" --no_active \
        --analyze --structured_fit --bank_file $BANK --output_dir "$FAM/$name" $(v1_args "$env") > "$FAM/$name.induce.log" 2>&1
    echo "[v2] induced $name rc=$?"
  fi
  grep -q '"elapsed_s"' "$FAM/refine_$name/refine.json" 2>/dev/null || $PY -W ignore scripts/visualpatchworld/refine_llm_family.py \
      --run_dir "$FAM/$name" --output_dir "$FAM/refine_$name" --bank_file $BANK --structured_fit > "$FAM/refine_$name.refine.log" 2>&1
  echo "[v2] refined $name rc=$?"
}
for job in "$@"; do run $job & done
wait
echo "[v2] done"
