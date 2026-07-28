#!/usr/bin/env bash
# Step 2: Cube tool vs oracle perception audit.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PY:-python3}"
export STABLEWM_HOME="${STABLEWM_HOME:-${HOME}/.stable-wm}"
export MUJOCO_GL=egl
export PYTHONPATH="${ROOT}/src:${ROOT}:${PYTHONPATH:-}"

REPORT="${ROOT}/artifacts/visualpatchworld/reports"
NUM_SAMPLES="${NUM_SAMPLES:-200}"
FORCE="${FORCE:-0}"

mkdir -p "$REPORT"
log() { echo "[step2-cube][$(date +%H:%M:%S)] $*"; }

out="$REPORT/cube_tool_perception_n${NUM_SAMPLES}.json"
if [[ -f "$out" && "$FORCE" != "1" ]]; then
  log "skip (exists $out)"
else
  log "run cube perception audit"
  "$PY" "$ROOT/scripts/visualpatchworld/evaluate_env_tool_perception.py" \
    --env cube \
    --num_samples "$NUM_SAMPLES" \
    --fit_affine \
    --output "$out" \
    2>&1 | tee "${ROOT}/logs/paper_experiments/cube_tool_perception.log"
fi

log "also audit reacher + tworoom for paper table"
for env in reacher tworoom; do
  eout="$REPORT/${env}_tool_perception_n${NUM_SAMPLES}.json"
  if [[ -f "$eout" && "$FORCE" != "1" ]]; then
    continue
  fi
  "$PY" "$ROOT/scripts/visualpatchworld/evaluate_env_tool_perception.py" \
    --env "$env" \
    --num_samples "$NUM_SAMPLES" \
    --output "$eout"
done

log "summary"
"$PY" - <<PY
import json
from pathlib import Path
for p in sorted(Path("$REPORT").glob("*_tool_perception_n*.json")):
    d = json.loads(p.read_text())
    m = d.get("metrics", {})
    print(p.name, "rmse_xy=", m.get("position_rmse_xy"), "per_obj=", m.get("per_object_rmse_xy"))
PY
