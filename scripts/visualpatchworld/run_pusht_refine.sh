#!/usr/bin/env bash
# Residual-driven refinement of PushT families (geometry / contact kit / neither).
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-/home/comp/baijiaxin/miniconda3/envs/lewm/bin/python}
FAM=artifacts/visualpatchworld/llm_family
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
refine() {  # src_run out_name [extra args...]
  local src=$1 name=$2; shift 2
  if [[ -f $FAM/$name/refine.json ]] && grep -q '"elapsed_s"' "$FAM/$name/refine.json" && [[ -z ${FORCE:-} ]]; then echo "[refine] have $name"; return; fi
  $PY -W ignore scripts/visualpatchworld/refine_llm_family.py --run_dir "$FAM/$src" --output_dir "$FAM/$name" "$@" \
      > "$FAM/$name.refine.log" 2>&1
  echo "[refine] $name rc=$?"
}
for job in "$@"; do eval "refine $job" & done
wait
