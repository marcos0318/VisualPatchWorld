"""Evaluate the physics-structured pushing-mechanics hypothesis class on PushT.

This reproduces the scientific-discovery diagnosis documented in
``docs/scientific_discovery_dynamics.md``: it compares one-step block-prediction
error of three hypothesis classes on oracle scene-graph trajectories, stratified
by contact regime (the planner re-observes each macro-step, so one-step accuracy
on contact pushes is the planning-critical metric):

  * ridge          - memoryless linear map on 18 hand-crafted features
                     (the existing ``pusht_scaffold_fit`` hypothesis class)
  * pushing        - quasi-static pushing law on a compact physics basis
                     (this module, memoryless variant)
  * pushing+mom    - same law seeded with finite-difference block velocity
                     (the second-order variant, what the planner can supply
                     from re-observation)

It also verifies that the rendered PatchWorld model loads and predicts.

Run (oracle data is committed under artifacts/visualpatchworld/data):

    PYTHONPATH=src:/path/to/PatchWorld \
      python scripts/visualpatchworld/evaluate_pushing_mechanics.py \
      --data_dir artifacts/visualpatchworld/data
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

from visualpatchworld.pusht_local_induction import (
    _angle_delta,
    _feature_vector,
    _parse_action,
    _parse_observation_text,
    _state_from_graph,
)
from visualpatchworld.pushing_mechanics_dynamics import (
    _phys_feats,
    fit_pushing_mechanics_params,
    generate_pushing_mechanics_model_code,
)

State = Tuple[float, float, float, float, float]


@dataclass
class _Transition:
    observation: str
    action: str
    next_observation: str


@dataclass
class _Trajectory:
    transitions: List[_Transition]


def load_trajectories(path: Path) -> List[_Trajectory]:
    trajs: List[_Trajectory] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        transitions = [
            _Transition(t["observation"], t["action"], t["next_observation"])
            for t in rec.get("transitions", [])
        ]
        if transitions:
            trajs.append(_Trajectory(transitions))
    return trajs


def _state(text: str) -> State:
    return _state_from_graph(_parse_observation_text(text))


def fit_ridge(trajs: Sequence[_Trajectory], ridge: float = 1e-3) -> np.ndarray:
    X, Y = [], []
    for traj in trajs:
        for t in traj.transitions:
            s = _state(t.observation)
            ns = _state(t.next_observation)
            a = _parse_action(t.action)
            X.append(_feature_vector(s, a))
            Y.append([ns[0] - s[0], ns[1] - s[1], ns[2] - s[2], ns[3] - s[3], _angle_delta(ns[4], s[4])])
    X = np.asarray(X)
    Y = np.asarray(Y)
    return np.linalg.solve(X.T @ X + ridge * np.eye(X.shape[1]), X.T @ Y).T


def step_ridge(W: np.ndarray, s: State, a: Tuple[float, float]) -> State:
    d = W @ np.asarray(_feature_vector(s, a))
    return (s[0] + d[0], s[1] + d[1], s[2] + d[2], s[3] + d[3], (s[4] + d[4]) % (2 * math.pi))


def step_pushing(params, s: State, a: Tuple[float, float], vel=(0.0, 0.0, 0.0)) -> Tuple[State, Tuple[float, float, float]]:
    sx, sy, bx, by, ba = s
    dist = math.hypot(bx - sx, by - sy)
    contact = dist <= params["contact_threshold"]
    gain = params["agent_contact"] if contact else params["agent_free"]
    dax, day = gain[0] * a[0], gain[1] * a[1]
    vbx = params["beta"] * vel[0]
    vby = params["beta"] * vel[1]
    vw = params["beta_w"] * vel[2]
    if contact:
        feats = _phys_feats(sx, sy, bx, by, dax, day)
        coef = params["contact_coef"]
        vbx += sum(c * f for c, f in zip(coef[0], feats))
        vby += sum(c * f for c, f in zip(coef[1], feats))
        vw += sum(c * f for c, f in zip(coef[2], feats))

    def clip(v):
        return max(0.0, min(512.0, v))

    return (clip(sx + dax), clip(sy + day), clip(bx + vbx), clip(by + vby), (ba + vw) % (2 * math.pi)), (vbx, vby, vw)


def onestep_table(val: Sequence[_Trajectory], W: np.ndarray, params) -> None:
    THR = params["contact_threshold"]
    regimes = ("contact_moving", "all_contact", "far")
    acc = {r: {"ridge": [], "push": [], "pushmom": []} for r in regimes}
    angacc = {r: {"ridge": [], "push": [], "pushmom": []} for r in regimes}
    for traj in val:
        seq = traj.transitions
        for i, t in enumerate(seq):
            s = _state(t.observation)
            ns = _state(t.next_observation)
            a = _parse_action(t.action)
            dist = math.hypot(s[2] - s[0], s[3] - s[1])
            gt_move = math.hypot(ns[2] - s[2], ns[3] - s[3])
            contact = dist <= THR
            tags = []
            if contact and gt_move > 2.0:
                tags.append("contact_moving")
            if contact:
                tags.append("all_contact")
            if not contact:
                tags.append("far")
            v_seed = (0.0, 0.0, 0.0)
            if i > 0:
                ps = _state(seq[i - 1].observation)
                v_seed = (s[2] - ps[2], s[3] - ps[3], _angle_delta(s[4], ps[4]))
            pr = step_ridge(W, s, a)
            pp, _ = step_pushing(params, s, a)
            pm, _ = step_pushing(params, s, a, v_seed)
            for tag in tags:
                acc[tag]["ridge"].append(math.hypot(pr[2] - ns[2], pr[3] - ns[3]))
                acc[tag]["push"].append(math.hypot(pp[2] - ns[2], pp[3] - ns[3]))
                acc[tag]["pushmom"].append(math.hypot(pm[2] - ns[2], pm[3] - ns[3]))
                angacc[tag]["ridge"].append(abs(_angle_delta(pr[4], ns[4])))
                angacc[tag]["push"].append(abs(_angle_delta(pp[4], ns[4])))
                angacc[tag]["pushmom"].append(abs(_angle_delta(pm[4], ns[4])))

    print(f"\ncontact_threshold = {THR:.1f}")
    print(f"{'regime':16s} {'model':12s} {'block_pos_err':>13s} {'angle_err':>10s} {'n':>6s}")
    for r in regimes:
        for key, label in (("ridge", "ridge"), ("push", "pushing"), ("pushmom", "pushing+mom")):
            bp = acc[r][key]
            an = angacc[r][key]
            if not bp:
                continue
            print(f"{r:16s} {label:12s} {np.mean(bp):13.2f} {np.mean(an):10.3f} {len(bp):6d}")
        print()


def verify_rendered_model(train: Sequence[_Trajectory], patchworld_root: str) -> None:
    import sys

    if patchworld_root and patchworld_root not in sys.path:
        sys.path.insert(0, patchworld_root)
    try:
        from visualpatchworld.patchworld_adapter import (
            apply_pusht_patchworld_adapter,
            sanitize_pusht_model_code,
        )

        apply_pusht_patchworld_adapter()
        from patchworld.worldmodel_validator import load_model_from_code
    except Exception as exc:  # pragma: no cover
        print(f"[verify] skipped (patchworld unavailable): {exc}")
        return

    code = generate_pushing_mechanics_model_code(train)
    model, error = load_model_from_code(sanitize_pusht_model_code(code))
    if error or model is None:
        print(f"[verify] FAILED to load rendered model: {error}")
        return
    t = train[0].transitions[0]
    belief = model.init_belief_from_observation(t.observation)
    nxt = model.predict_belief(belief, t.action)
    out = model.readout_observation(nxt, t.action)
    print("[verify] rendered PatchWorld model loaded and produced a one-step prediction OK")
    print("[verify] sample predicted readout (truncated):")
    print("\n".join(out.splitlines()[:7]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="artifacts/visualpatchworld/data")
    ap.add_argument("--patchworld_root", default="")
    args = ap.parse_args()

    data = Path(args.data_dir)
    train = load_trajectories(data / "pusht_oracle_traj_train.jsonl")
    val = load_trajectories(data / "pusht_oracle_traj_val.jsonl")
    n_train = sum(len(t.transitions) for t in train)
    n_val = sum(len(t.transitions) for t in val)
    print(f"train transitions={n_train}  val transitions={n_val}")

    params = fit_pushing_mechanics_params(train)
    print(
        "fitted params: "
        f"thr={params['contact_threshold']:.1f} "
        f"agent_free={[round(v, 1) for v in params['agent_free']]} "
        f"agent_contact={[round(v, 1) for v in params['agent_contact']]} "
        f"beta={params['beta']:.3f} beta_w={params['beta_w']:.3f}"
    )
    W = fit_ridge(train)
    onestep_table(val, W, params)
    verify_rendered_model(train, args.patchworld_root)


if __name__ == "__main__":
    main()
