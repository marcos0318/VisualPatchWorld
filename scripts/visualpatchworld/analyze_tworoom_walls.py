#!/usr/bin/env python
"""Rate at which Two-room rollouts pass through the interior wall instead of the door.

The wall is vertical at x=WALL_X; the door half-extent is unobserved but at most
MAX_DOOR_HALF, so a crossing farther than that from the door center is a violation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from visualpatchworld.llm_family.fit import _safe_step, _windows  # noqa: E402
from visualpatchworld.llm_family.propose import compile_candidate  # noqa: E402
from visualpatchworld.llm_family.traces import load_episodes  # noqa: E402

WALL_X = 112.0
MAX_DOOR_HALF = 21.0


def _violates(a, b, door_y: float) -> bool:
    (x0, y0), (x1, y1) = a, b
    if (x0 - WALL_X) * (x1 - WALL_X) >= 0:
        return False
    t = (WALL_X - x0) / (x1 - x0)
    return abs(y0 + t * (y1 - y0) - door_y) > MAX_DOOR_HALF


def _rate(rollouts):
    steps = viol = wins = bad_wins = 0
    for door_y, pts in rollouts:
        wins += 1
        hit = False
        for a, b in zip(pts, pts[1:]):
            steps += 1
            if _violates(a, b, door_y):
                viol += 1
                hit = True
        bad_wins += hit
    return {"step_violation_rate": viol / max(steps, 1), "window_violation_rate": bad_wins / max(wins, 1), "windows": wins}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--val", default="/home/comp/baijiaxin/le-wm-code/artifacts/visualpatchworld/data/tworoom_oracle_traj_val.jsonl")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--objective", default="multi_step")
    p.add_argument("--out", default=str(ROOT / "artifacts/visualpatchworld/llm_family/tworoom_walls.json"))
    args = p.parse_args()

    episodes = load_episodes(Path(args.val))
    windows = _windows(episodes, args.k)
    xy = lambda s: (s["agent.x"], s["agent.y"])
    door = lambda w: float(w[0].meta["door_coords"][1])
    result = {"data": _rate([(door(w), [xy(w[0].state)] + [xy(t.next_state) for t in w]) for w in windows])}

    # Counterfactual pushes: start beside the wall away from the door and push
    # toward it at the largest observed action; the simulator always blocks these.
    a_max = max(abs(t.action[0]) for ep in episodes for t in ep.transitions)
    pushes = []
    for ep in episodes:
        t0 = ep.transitions[0]
        door_y = float(t0.meta["door_coords"][1])
        for side in (-1.0, 1.0):
            for y in (door_y - 60.0, door_y - 35.0, door_y + 35.0, door_y + 60.0):
                if not 10.0 <= y <= 214.0:
                    continue
                start = dict(t0.state, **{"agent.x": WALL_X + side * 8.0, "agent.y": y})
                pushes.append((door_y, start, [-side * a_max, 0.0], t0.meta))

    for run in args.runs:
        fam = json.loads((ROOT / "artifacts/visualpatchworld/llm_family" / run / "family.json").read_text())
        winner = fam["selections"][f"offline_{args.objective}"]["winner"]
        for c in fam["candidates"]:
            cand = compile_candidate(c["code"])
            params = fam["fits"][args.objective][c["name"]]["params"]
            rollouts = []
            for w in windows:
                s, pts = dict(w[0].state), [xy(w[0].state)]
                for t in w:
                    s = _safe_step(cand, s, t.action, params, t.meta)
                    if s is None:
                        break
                    pts.append(xy(s))
                rollouts.append((door(w), pts))
            cf = []
            for door_y, start, action, meta in pushes:
                s, pts = dict(start), [xy(start)]
                for _ in range(args.k):
                    s = _safe_step(cand, s, action, params, meta)
                    if s is None:
                        break
                    pts.append(xy(s))
                cf.append((door_y, pts))
            result[f"{run}/{c['name']}"] = {
                **_rate(rollouts),
                "push_violation_rate": _rate(cf)["window_violation_rate"],
                "pushes": len(cf),
                "selected": c["name"] == winner,
            }
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    for k, v in result.items():
        push = f" push={v['push_violation_rate']:.3f}" if "push_violation_rate" in v else ""
        print(f"{k:60s} step={v['step_violation_rate']:.4f} window={v['window_violation_rate']:.4f}{push}{' *' if v.get('selected') else ''}")


if __name__ == "__main__":
    main()
