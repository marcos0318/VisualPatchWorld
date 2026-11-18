#!/usr/bin/env bash
# Run every missing (method, env, seed) planning evaluation for one baseline root.
# Usage: fill_baseline_evals.sh BASELINE_ROOT OUTPUT_DIR [MAX_PARALLEL]
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
B="$1"; O="$2"; P="${3:-6}"
PY="${PYTHON:-/home/comp/baijiaxin/miniconda3/envs/lewm/bin/python}"
export PYTHONUTF8=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$O"
for m in worldcoder gif_mcts poeworld pomdp_coder cwm_game tracepatch; do
  for e in tworoom reacher cube pusht; do
    ls "$B/$m/$e/generated_models/"*.py >/dev/null 2>&1 || continue
    for s in 42 43 44 45 46; do
      [ -f "$O/${m}_${e}_n50_seed${s}.json" ] && continue
      while [ "$(jobs -rp | wc -l)" -ge "$P" ]; do sleep 5; done
      echo "[$(date -Iseconds)] START $m/$e seed=$s"
      ( nohup "$PY" scripts/visualpatchworld/run_fair_baseline_replan_gt90.py --methods "$m" --envs "$e" --seed "$s" \
          --baseline_root "$B" --output_dir "$O" > "$O/${m}_${e}_seed${s}.log" 2>&1
        echo "[$(date -Iseconds)] EXIT $m/$e seed=$s rc=$?" ) &
    done
  done
done
while [ "$(jobs -rp | wc -l)" -gt 0 ]; do sleep 10; done
echo "[$(date -Iseconds)] ALL DONE"
