#!/usr/bin/env bash
# Full Cube rerun after z-retention + contact-scaffold + REx tolerance fixes.
# Steps: export JSONL -> induce VPW -> re-induce code baselines -> fair library-shoot.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PY:-python3}"
cd "$ROOT"

# Set DEEPINFRA_API_KEY or PATCHWORLD_LLM_API_KEY in the environment.
export STABLEWM_HOME="${STABLEWM_HOME:-${HOME}/.stable-wm}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONPATH="${ROOT}:${ROOT}/src:${PYTHONPATH:-}"
export PATCHWORLD_LLM_API_KEY="${PATCHWORLD_LLM_API_KEY:-${DEEPINFRA_API_KEY:-}}"
export PATCHWORLD_LLM_REQUEST_TIMEOUT_S="${PATCHWORLD_LLM_REQUEST_TIMEOUT_S:-600}"
export PATCHWORLD_LLM_MAX_RETRIES="${PATCHWORLD_LLM_MAX_RETRIES:-12}"
export PATCHWORLD_LLM_CACHE="${PATCHWORLD_LLM_CACHE:-1}"
export PATCHWORLD_LLM_CACHE_PATH="${PATCHWORLD_LLM_CACHE_PATH:-$ROOT/artifacts/visualpatchworld/llm_cache.sqlite3}"
if [[ -z "${PATCHWORLD_LLM_API_KEY}" ]]; then
  echo "ERROR: PATCHWORLD_LLM_API_KEY / DEEPINFRA_API_KEY not set" >&2
  exit 1
fi
echo "LLM key loaded (len=${#PATCHWORLD_LLM_API_KEY})"

DATA_DIR="${DATA_DIR:-$ROOT/artifacts/visualpatchworld/data}"
VPW_DIR="${VPW_DIR:-$ROOT/artifacts/visualpatchworld/lewm_cube_general_results}"
LOG_DIR="${LOG_DIR:-$ROOT/artifacts/visualpatchworld/reports/cube_rerun_$(date +%Y%m%d_%H%M%S)}"
FAIR_OUT="${FAIR_OUT:-$ROOT/artifacts/visualpatchworld/reports/fair_wm_ranking_gt90}"
BASELINE_METHODS=(tracepatch worldcoder poeworld gif_mcts pomdp_coder cwm_game)
MAX_EPISODES="${MAX_EPISODES:-64}"
NUM_EVAL="${NUM_EVAL:-50}"
SEED="${SEED:-42}"

mkdir -p "$LOG_DIR" "$DATA_DIR" "$VPW_DIR"
STATUS="$LOG_DIR/status.tsv"
echo -e "step\tstatus\tstarted\tfinished\tnote" > "$STATUS"

log() { echo "[$(date -Iseconds)] $*"; }
run_step() {
  local name="$1"; shift
  local started finished status note=""
  started="$(date -Iseconds)"
  log "START ${name}"
  if "$@" >"$LOG_DIR/${name}.log" 2>&1; then
    status="ok"
  else
    status="fail"
    note="see ${LOG_DIR}/${name}.log"
  fi
  finished="$(date -Iseconds)"
  echo -e "${name}\t${status}\t${started}\t${finished}\t${note}" >> "$STATUS"
  log "DONE ${name} status=${status}"
  [[ "$status" == "ok" ]]
}

########################################
# 1) Re-export Cube oracle graphs (with z)
########################################
run_step 01_export_cube \
  "$PY" scripts/visualpatchworld/export_lewm_env.py \
    --env cube \
    --output_dir "$DATA_DIR" \
    --max_episodes "$MAX_EPISODES" \
    --source oracle

# Sanity: exported text must contain z=
"$PY" - <<'PY'
from pathlib import Path
p = Path("artifacts/visualpatchworld/data/cube_oracle_traj_train.jsonl")
text = p.read_text(encoding="utf-8")
assert "z=" in text, "Cube export still missing z="
print("export_has_z=True", "train_lines=", sum(1 for _ in p.open()))
PY

########################################
# 2) Induce VPW Cube contact scaffold
########################################
run_step 02_induce_vpw_cube \
  "$PY" -m visualpatchworld.patchworld_benchmark \
    --env cube \
    --train_glob "${DATA_DIR}/cube_oracle_traj_train.jsonl" \
    --eval_glob "${DATA_DIR}/cube_oracle_traj_val.jsonl" \
    --output_dir "$VPW_DIR" \
    --induction_strategy general_scaffold_fit \
    --max_train 0 \
    --max_eval 0 \
    --max_refine_rounds 0

"$PY" - <<'PY'
from pathlib import Path
d = Path("artifacts/visualpatchworld/lewm_cube_general_results/generated_models")
cands = sorted(d.glob("cube_*.py"), key=lambda p: p.stat().st_mtime)
assert cands, "no cube model generated"
latest = cands[-1]
text = latest.read_text(encoding="utf-8")
assert "CubeContactScaffoldWorldModel" in text or "CONTACT_THRESHOLD" in text or "GeneralScaffoldFitWorldModel" in text
print("latest_vpw_cube_model=", latest)
print("has_contact_scaffold=", "CubeContactScaffoldWorldModel" in text)
PY

########################################
# 3) Re-induce code baselines on Cube (induction only; fair planner next)
########################################
for method in "${BASELINE_METHODS[@]}"; do
  run_step "03_induce_${method}_cube" \
    "$PY" scripts/visualpatchworld/run_code_baselines.py \
      --method "$method" \
      --env cube \
      --data_dir "$DATA_DIR" \
      --max_train 24 \
      --max_eval 12 \
      --max_refine_rounds 3 \
      --worldcoder_programs 2 \
      --poeworld_batches 3 \
      --poeworld_experts_per_batch 4 \
      --gif_mcts_iterations 50 \
      --pomdp_coder_candidates 3 \
      --cwm_retries 50 \
      --seed "$SEED" \
      --skip_planning \
    || log "WARN induction failed for ${method}/cube; continuing"
done

########################################
# 4) Fair GT90 ranking for Cube (MuJoCo GT + VPW Oracle+Induced)
########################################
run_step 04_fair_wm_ranking_cube \
  "$PY" scripts/visualpatchworld/run_fair_wm_ranking_gt90.py \
    --envs cube \
    --num_eval "$NUM_EVAL" \
    --seed "$SEED" \
    --skip_gt_eval \
    --output_dir "$FAIR_OUT"

########################################
# 5) Fair replan: all code baselines + VPW tool/hybrid on Cube
########################################
run_step 05_fair_replan_cube_all \
  "$PY" scripts/visualpatchworld/run_fair_baseline_replan_gt90.py \
    --envs cube \
    --methods all \
    --num_eval "$NUM_EVAL" \
    --seed "$SEED" \
    --output_dir "${FAIR_OUT}/all_methods"

########################################
# 6) Collect + print summary
########################################
"$PY" - <<'PY'
import json
from pathlib import Path
root = Path("artifacts/visualpatchworld/reports/fair_wm_ranking_gt90")
cube_sum = root / "cube" / "summary.json"
all_sum = sorted((root / "all_methods").glob("summary_n*_seed*.json"))
print("=== Cube VPW ranking ===")
if cube_sum.is_file():
    d = json.loads(cube_sum.read_text())
    print("model", d.get("model", {}).get("success_rate"), d.get("model", {}).get("model_path"))
    print("mujoco", d.get("mujoco_gt", {}).get("success_rate"))
print("=== Cube all-methods cells ===")
cells = {}
for p in (root / "all_methods").glob("*_cube_n50_seed42.json"):
    d = json.loads(p.read_text())
    cells[p.name] = d.get("success_rate")
for k in sorted(cells):
    print(f"{k}: {cells[k]}")
if all_sum:
    print("latest_summary", all_sum[-1])
PY

log "Cube rerun finished. Status: $STATUS Logs: $LOG_DIR"
cat "$STATUS"
