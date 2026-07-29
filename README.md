# VisualPatchWorld

Structure-aware **executable** world-model induction for planner-usable code.

VisualPatchWorld (VPW) learns programmatic dynamics from LeWM expert trajectories
on four environments (**Two-room**, **Reacher**, **PushT**, **Cube**), then evaluates
them with fair frozen CEM / library-shoot planners. This repository is the public
code and results release accompanying the VisualPatchWorld paper.

**Authors / lab:** Jiaxin Bai, [HKBU KnowComp](https://github.com/HKBU-KnowComp)

## Repository layout

```text
patchworld/                 # vendored PatchWorld induction toolkit (MIT)
src/visualpatchworld/       # VPW perception, induction adapters, planning eval
scripts/visualpatchworld/   # paper + pipeline entrypoints
scripts/aggregate_*_ci.py   # multi-seed CI aggregation
artifacts/visualpatchworld/
  reports/                  # paper JSON reports (fair GT>=90 matrix, CIs, ablations)
  models/                   # Oracle+Induced models (convenience copies)
  lewm_*_program_results/   # induced models used by eval script defaults
  code_baselines/           # programmatic baseline models for the fair matrix
```

PatchWorld is vendored from [`marcos0318/PatchWorld`](https://github.com/marcos0318/PatchWorld)
under `patchworld/` (see `patchworld/LICENSE`). AgentGym RQ3 servers from the
original PatchWorld toolkit are **not** required for VPW paper experiments.

## Install

Python **3.10+** recommended.

```bash
git clone https://github.com/HKBU-KnowComp/VisualPatchWorld.git
cd VisualPatchWorld
python -m venv .venv
source .venv/bin/activate
pip install -e .
# Optional: LLM backends used by induction
pip install -e ".[llm]"
```

### LeWM environments and datasets

Planning evaluation needs LeWM / `stable-worldmodel` datasets and MuJoCo:

```bash
pip install "stable-worldmodel[env]"
# Reacher / dm_control compatibility
pip install 'mujoco==3.8.1'
export STABLEWM_HOME="${STABLEWM_HOME:-$HOME/.stable-wm}"
export MUJOCO_GL=egl
```

Download LeWM datasets and checkpoints from the
[LeWM Hugging Face collection](https://huggingface.co/collections/quentinll/lewm)
into `$STABLEWM_HOME` (layout expected by `stable-worldmodel`). Upstream LeWM code:
[`lucas-maes/le-wm`](https://github.com/lucas-maes/le-wm).

### LLM API keys (induction / code baselines only)

Set one of these in the environment (do **not** commit keys):

```bash
export DEEPINFRA_API_KEY=...          # or OPENAI_API_KEY for OpenAI-compatible APIs
export PATCHWORLD_LLM_API_KEY="${DEEPINFRA_API_KEY}"
```

Released JSON reports and induced `.py` models do **not** require an API key to inspect.

## Paper results (fair GT ≥ 90%, seed 42, n=50)

Main claims use **fair frozen planners** where MuJoCo GT ≥ 90% under the same knobs
as the world-model rows. Seed-42 cells are point estimates on a shared 50-start eval
set so VPW and programmatic baselines share identical starts.

**Matched induced-only comparison** (paper Table `tab:lewm_baselines`):

| Method | Two-room | Reacher | PushT | Cube | Mean |
|---|---:|---:|---:|---:|---:|
| PatchWorld | 98 | 8 | 0 | 66 | 43.0 |
| POMDP-Coder (strongest code) | 98 | 18 | 0 | 66 | 45.5 |
| VPW Tool+Induced | 96 | 60 | 22 | 66 | 61.0 |
| **VPW Oracle+Induced** | **96** | **72** | **22** | **86** | **69.0** |

Oracle+Induced five-seed suite mean (seeds 42–46): **67.5 ± 1.5**.

**Simulator-assisted rows** (not the matched code claim):

| Method | Two-room | Reacher | PushT | Cube | Mean |
|---|---:|---:|---:|---:|---:|
| VPW Tool+Hybrid | 100 | 70 | 88 | 78 | 84.0 |
| VPW Oracle+Hybrid | 100 | 100 | 96 | 84 | 95.0 |
| MuJoCo+CEM | 100 | 100 | 96 | 94 | 97.5 |

Frozen protocols (GT caches under `artifacts/visualpatchworld/reports/gt90_sweep/`):

| Env | Protocol | GT % | Oracle+Induced % |
|---|---|---:|---:|
| Two-room | CEM `fs5 h5 r5 300×10` | 100 | 96 |
| Reacher | CEM `fs1 h1 r1` finger_match `300×10` | 100 | 72 |
| PushT | CEM-MPC `ab5 h5 r5 600×15` | 96 | 22 |
| Cube | library shoot `fs5 h1 r1 400` budget 100 | 94 | 86 |

### Where reports live

- `artifacts/visualpatchworld/reports/fair_wm_ranking_gt90/` — per-env summaries + `all_methods/` cells
- `artifacts/visualpatchworld/reports/gt90_sweep/` — MuJoCo GT planner caches
- `artifacts/visualpatchworld/reports/hybrid_cost_table_seed42.json` — query / wall-clock trade-offs
- `artifacts/visualpatchworld/reports/noprobe_ablation/` — Level-1 probe / wrong-scaffold controls
- `artifacts/visualpatchworld/models/` — convenience copies of Oracle+Induced models

## Reproduce the fair matrix

```bash
cd VisualPatchWorld
export PYTHONPATH="src:.:${PYTHONPATH:-}"
export STABLEWM_HOME="${STABLEWM_HOME:-$HOME/.stable-wm}"
export MUJOCO_GL=egl

python scripts/visualpatchworld/run_fair_wm_ranking_gt90.py \
  --envs tworoom,pusht,cube,reacher --num_eval 50 --seed 42 --skip_gt_eval
python scripts/visualpatchworld/run_fair_baseline_replan_gt90.py \
  --methods all --envs tworoom,pusht,cube,reacher --num_eval 50 --seed 42
python scripts/visualpatchworld/run_fair_multiseed_selected.py \
  --batches A,B,C,D --seeds 42,43,44,45,46 --max_parallel 2
python scripts/visualpatchworld/build_hybrid_cost_table.py
python scripts/visualpatchworld/eval_noprobe_ablation.py
```

Aggregate multi-seed Wilson / binomial intervals:

```bash
python scripts/aggregate_fair_multiseed_ci.py --format text
# or: --format latex
```

## Pipeline overview

VPW has three stages:

1. **Perception** — oracle JSONL from simulator state, or live tool/VLM scene graphs from pixels at replan time.
2. **Induction** — Level-1 sketch probes, then Level-2 parameter fit (joint+FK on Reacher, PD+contact on PushT, linear navigation on Two-room, grip-gated contact on Cube).
3. **Planning** — per-environment frozen CEM / library-shoot planners with optional top-30% MuJoCo hybrid scoring.

Minimal smoke (requires datasets):

```bash
python scripts/visualpatchworld/sample_image_action_trajectories.py \
  --max_episodes 6 --max_steps_per_episode 12
python scripts/visualpatchworld/induce_world_model.py \
  --source oracle --max_train 4 --max_eval 2 --max_refine_rounds 2
```

## Citation

If you use this code or results, please cite the VisualPatchWorld paper and this
repository (see [`CITATION.cff`](CITATION.cff)).

## License

MIT License — see [`LICENSE`](LICENSE). Vendored PatchWorld retains its MIT license
under [`patchworld/LICENSE`](patchworld/LICENSE).
