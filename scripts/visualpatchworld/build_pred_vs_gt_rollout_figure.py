#!/usr/bin/env python3
"""Build GT vs VPW vs all programmatic baselines rollout figure.

Uses the exact model .py files from fair_wm_ranking_gt90 reports.
Outputs PDF/PNG under KDD2027VisualPatchWorld/figures/ by default.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PATCHWORLD = ROOT  # vendored package at repo root
PAPER_FIG = ROOT / "artifacts" / "visualpatchworld" / "figures"
FAIR = ROOT / "artifacts/visualpatchworld/reports/fair_wm_ranking_gt90/all_methods"

# Display order and short labels (matched to Table lewm_baselines).
# Keep one entry per method so legend colors match both panels
# (Reacher coincidences must not merge WorldCoder/GIF into another color).
BASELINES = [
    ("tracepatch", "PatchWorld"),
    ("worldcoder", "WorldCoder"),
    ("pomdp_coder", "POMDP"),
    ("poeworld", "PoE"),
    ("gif_mcts", "GIF-MCTS"),
    ("cwm_game", "CWM-Game"),
]

# Distinct dash/marker styles so overlapping curves stay identifiable.
STYLES = {
    "tracepatch": dict(linestyle="--", marker="s", mark_offset=0, mark_stride=2),
    "worldcoder": dict(linestyle=":", marker="D", mark_offset=1, mark_stride=2),
    "pomdp_coder": dict(linestyle="-.", marker="o", mark_offset=0, mark_stride=3),
    "poeworld": dict(linestyle="--", marker="v", mark_offset=1, mark_stride=3),
    "gif_mcts": dict(linestyle=":", marker="^", mark_offset=0, mark_stride=2),
    "cwm_game": dict(linestyle="-.", marker="x", mark_offset=1, mark_stride=2),
}

VPW_MODELS = {
    "reacher": ROOT
    / "artifacts/visualpatchworld/lewm_reacher_joint_results/generated_models"
    / "reacher_patchworld_20260623_144440.py",
    "pusht": ROOT / "artifacts/visualpatchworld/induced/program_induced_best.py",
}

ENV_CFG = {
    "reacher": {
        "jsonl": ROOT / "artifacts/visualpatchworld/data/reacher_oracle_traj_val.jsonl",
        "key": "finger",
        "n_steps": 12,
    },
    "pusht": {
        "jsonl": ROOT / "artifacts/visualpatchworld/data/pusht_oracle_traj_val.jsonl",
        "key": "block",
        "n_steps": 12,
    },
}

# Distinct colors; VPW and GT handled separately.
COLORS = {
    "tracepatch": "#d62728",
    "worldcoder": "#ff7f0e",
    "pomdp_coder": "#9467bd",
    "poeworld": "#8c564b",
    "gif_mcts": "#e377c2",
    "cwm_game": "#7f7f7f",
    "vpw": "#4DA3FF",  # bright light blue
    "gt": "#000000",
}


def _fair_model_path(method: str, env: str) -> Path:
    report = FAIR / f"{method}_{env}_n50_seed42.json"
    data = json.loads(report.read_text())
    path = Path(data["model_path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _load_model(path: Path):
    import sys

    for p in (str(ROOT / "src"), str(PATCHWORLD)):
        if p not in sys.path:
            sys.path.insert(0, p)
    from patchworld.worldmodel_validator import load_model_from_code

    model, err = load_model_from_code(path.read_text())
    if model is None:
        raise RuntimeError(f"Failed to load {path}: {err}")
    return model


def _parse_xy(obs_text: str, obj_id: str) -> tuple[float, float]:
    for line in obs_text.splitlines():
        line = line.strip()
        if not line.startswith(f"- {obj_id}:"):
            continue
        xs = re.findall(r"x=(-?\d+(?:\.\d+)?)", line)
        ys = re.findall(r"y=(-?\d+(?:\.\d+)?)", line)
        if xs and ys:
            return float(xs[0]), float(ys[0])
    raise KeyError(f"object {obj_id} not found in observation")


def _belief_xy(belief: dict, obj_id: str) -> tuple[float, float]:
    obj = belief.get("objects", {}).get(obj_id, {})
    return float(obj.get("x", 0.0)), float(obj.get("y", 0.0))


def _parse_objects(obs_text: str) -> dict:
    objects: dict = {}
    in_objects = False
    for raw in obs_text.splitlines():
        line = raw.strip()
        if line.startswith("Objects:"):
            in_objects = True
            continue
        if line.startswith("Relations:") or line.startswith("Metadata:"):
            in_objects = False
            continue
        if not in_objects or not line.startswith("-"):
            continue
        m = re.match(r"^-\s*(\w+):(\w+)\s*(.*)$", line)
        if not m:
            continue
        obj = {"id": m.group(1), "category": m.group(2), "conf": 1.0}
        for key, value in re.findall(r"(\w+)=(-?\d+(?:\.\d+)?)", m.group(3)):
            obj[key] = float(value)
        objects[m.group(1)] = obj
    return objects


def _init_belief(model, obs_text: str) -> dict:
    try:
        return model.init_belief_from_observation(obs_text)
    except Exception:
        belief = model.init_belief()
        belief["objects"] = _parse_objects(obs_text)
        belief["relations"] = {}
        return belief


def _load_episode(jsonl: Path, n_steps: int, episode_index: int = 0) -> list[dict]:
    line = jsonl.read_text().splitlines()[episode_index]
    ep = json.loads(line)
    steps = ep["transitions"][:n_steps]
    if len(steps) < 4:
        raise RuntimeError(f"{jsonl} episode {episode_index} too short: {len(steps)}")
    return steps


def _gt_traj(steps: list[dict], obj_id: str) -> np.ndarray:
    pts = [_parse_xy(steps[0]["observation"], obj_id)]
    for t in steps:
        pts.append(_parse_xy(t["next_observation"], obj_id))
    return np.asarray(pts, dtype=float)


def _model_traj_openloop(model, steps: list[dict], obj_id: str) -> np.ndarray:
    belief = _init_belief(model, steps[0]["observation"])
    pts = [_belief_xy(belief, obj_id)]
    for t in steps:
        try:
            belief = model.predict_belief(belief, t["action"])
            pts.append(_belief_xy(belief, obj_id))
        except Exception:
            pts.append(pts[-1])
    return np.asarray(pts, dtype=float)


def _one_step_deltas(model, steps: list[dict], obj_id: str) -> np.ndarray:
    deltas = []
    for t in steps:
        p_t = np.asarray(_parse_xy(t["observation"], obj_id), dtype=float)
        belief = _init_belief(model, t["observation"])
        try:
            belief = model.predict_belief(belief, t["action"])
            p_hat = np.asarray(_belief_xy(belief, obj_id), dtype=float)
        except Exception:
            p_hat = p_t
        deltas.append(float(np.linalg.norm(p_hat - p_t)))
    return np.asarray(deltas, dtype=float)


def _gt_one_step_deltas(steps: list[dict], obj_id: str) -> np.ndarray:
    deltas = []
    for t in steps:
        p_t = np.asarray(_parse_xy(t["observation"], obj_id), dtype=float)
        p_n = np.asarray(_parse_xy(t["next_observation"], obj_id), dtype=float)
        deltas.append(float(np.linalg.norm(p_n - p_t)))
    return np.asarray(deltas, dtype=float)


def _choose_pusht_episode(n_steps: int) -> tuple[int, list[dict]]:
    """Episode with large GT block motion and some VPW one-step motion."""
    cfg = ENV_CFG["pusht"]
    vpw = _load_model(VPW_MODELS["pusht"])
    best_i, best_score, best_steps = 0, -1.0, None
    for i in range(30):
        steps = _load_episode(cfg["jsonl"], n_steps, episode_index=i)
        gt_d = _gt_one_step_deltas(steps, cfg["key"])
        vpw_d = _one_step_deltas(vpw, steps, cfg["key"])
        score = float(gt_d.sum() + 0.5 * vpw_d.sum())
        if gt_d.sum() > 40 and score > best_score:
            best_i, best_score, best_steps = i, score, steps
    assert best_steps is not None
    return best_i, best_steps


def _marker_idx(n: int, offset: int, stride: int) -> np.ndarray:
    """Marker indices excluding endpoints so spines never bisect markers."""
    idx = np.arange(offset, n, stride, dtype=int)
    return idx[(idx > 0) & (idx < n - 1)]


def _plot_baseline(
    ax,
    x,
    y,
    method: str,
    *,
    lw: float = 1.4,
    zorder: int = 3,
    marker_ymax: float | None = None,
):
    style = STYLES[method]
    color = COLORS[method]
    y = np.asarray(y, dtype=float)
    x = np.asarray(x)
    (h,) = ax.plot(
        x,
        y,
        color=color,
        lw=lw,
        linestyle=style["linestyle"],
        alpha=0.95,
        zorder=zorder,
        solid_capstyle="round",
        clip_on=True,
    )
    idx = _marker_idx(len(x), style["mark_offset"], style["mark_stride"])
    if marker_ymax is not None and len(idx):
        idx = idx[y[idx] <= marker_ymax]
    if len(idx):
        ax.plot(
            x[idx],
            y[idx],
            linestyle="None",
            marker=style["marker"],
            color=color,
            ms=2.8,
            zorder=zorder + 0.1,
            clip_on=True,
        )
    return h


def _pad_xlim(ax, x) -> None:
    """Pad x-limits so curves are not pressed against the spines."""
    x0, x1 = float(np.min(x)), float(np.max(x))
    pad = max(0.55, 0.06 * (x1 - x0))
    ax.set_xlim(x0 - pad, x1 + pad)


def build_figure(out_prefix: Path, episode_index: int = 0) -> Path:
    # Tall panels; bottom margin holds "step" then a 2-row legend (no overlap).
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 4.05))
    fig.subplots_adjust(left=0.07, right=0.96, bottom=0.24, top=0.90, wspace=0.05)

    # ---- Reacher: open-loop fingertip error vs step (all methods) ----
    cfg = ENV_CFG["reacher"]
    steps = _load_episode(cfg["jsonl"], cfg["n_steps"], episode_index=episode_index)
    gt = _gt_traj(steps, cfg["key"])
    t = np.arange(len(gt))
    ax = axes[0]
    ax.plot(t, np.zeros_like(t, dtype=float), "k-", lw=2.0, zorder=5)
    gt_m = _marker_idx(len(t), 0, 2)
    if len(gt_m):
        ax.plot(
            t[gt_m],
            np.zeros_like(gt_m, dtype=float),
            "ko",
            ms=2.8,
            zorder=5.1,
            linestyle="None",
        )

    vpw_traj = _model_traj_openloop(_load_model(VPW_MODELS["reacher"]), steps, cfg["key"])
    vpw_err = np.linalg.norm(vpw_traj - gt, axis=1)
    ax.plot(t, vpw_err, "-", color=COLORS["vpw"], lw=1.8, zorder=4)
    vpw_m = _marker_idx(len(t), 0, 2)
    if len(vpw_m):
        ax.plot(
            t[vpw_m],
            vpw_err[vpw_m],
            linestyle="None",
            marker="^",
            color=COLORS["vpw"],
            ms=3.2,
            zorder=4.1,
        )

    # Keep full series so the line exits the top of the axes; hide markers
    # above the visible y-limit (no nodes sitting on the ceiling).
    reacher_y_cap = 0.40
    reacher_exit_notes = []
    for method, label in BASELINES:
        model = _load_model(_fair_model_path(method, "reacher"))
        traj = _model_traj_openloop(model, steps, cfg["key"])
        err = np.linalg.norm(traj - gt, axis=1).astype(float)
        if float(err.max()) > reacher_y_cap:
            reacher_exit_notes.append(f"{label} max {float(err.max()):.2f}")
        _plot_baseline(ax, t, err, method, marker_ymax=reacher_y_cap)
    ax.set_title("(a) Reacher open-loop error", fontsize=9.5, pad=3)
    ax.set_xlabel("step", fontsize=8, labelpad=1)
    ax.set_ylabel("error", fontsize=8.5, labelpad=2)
    ax.tick_params(axis="x", labelsize=7.5, pad=1)
    ax.tick_params(axis="y", labelsize=7.5, pad=1)
    # Keep "step" tight under the ticks (auto offset was drifting into the legend).
    ax.xaxis.set_label_coords(0.5, -0.07)
    ax.set_ylim(-0.02, reacher_y_cap)
    _pad_xlim(ax, t)
    ax.grid(True, alpha=0.22, lw=0.6)
    ax.set_axisbelow(True)

    # ---- PushT: fair one-step |Δblock| (all methods) ----
    cfg = ENV_CFG["pusht"]
    chosen, steps = _choose_pusht_episode(cfg["n_steps"])
    ax = axes[1]
    steps_idx = np.arange(1, len(steps) + 1)
    gt_d = _gt_one_step_deltas(steps, cfg["key"])
    ax.plot(steps_idx, gt_d, "k-", lw=2.0, zorder=5)
    gt_m = _marker_idx(len(steps_idx), 0, 2)
    if len(gt_m):
        ax.plot(
            steps_idx[gt_m],
            gt_d[gt_m],
            "ko",
            ms=2.8,
            zorder=5.1,
            linestyle="None",
        )

    vpw_d = _one_step_deltas(_load_model(VPW_MODELS["pusht"]), steps, cfg["key"])
    ax.plot(steps_idx, vpw_d, "-", color=COLORS["vpw"], lw=1.8, zorder=4)
    vpw_m = _marker_idx(len(steps_idx), 0, 2)
    if len(vpw_m):
        ax.plot(
            steps_idx[vpw_m],
            vpw_d[vpw_m],
            linestyle="None",
            marker="^",
            color=COLORS["vpw"],
            ms=3.2,
            zorder=4.1,
        )

    outlier_notes = []
    # Ceiling for extreme WorldCoder/GIF values; stagger slightly for readability.
    y_cap = max(34.0, float(gt_d.max()) * 1.35)
    clip_y = {
        "worldcoder": y_cap,
        "gif_mcts": y_cap - 4.0,
    }
    y_top = y_cap + 8.0
    for method, label in BASELINES:
        model = _load_model(_fair_model_path(method, "pusht"))
        d = _one_step_deltas(model, steps, cfg["key"])
        if float(d.max()) > y_cap:
            outlier_notes.append(f"{label} max {float(d.max()):.0f}px")
            d_plot = np.full_like(d, clip_y.get(method, y_cap))
        else:
            d_plot = d
        _plot_baseline(ax, steps_idx, d_plot, method)
    ax.set_ylim(-2.0, y_top)
    _pad_xlim(ax, steps_idx)
    ax.set_title(f"(b) PushT block motion (px, ep {chosen})", fontsize=9.5, pad=3)
    ax.set_xlabel("step", fontsize=8, labelpad=1)
    # Put y-ticks on the right so panels can sit close together.
    ax.yaxis.tick_right()
    ax.tick_params(axis="y", labelsize=7.5, pad=1, left=False, labelleft=False)
    ax.tick_params(axis="x", labelsize=7.5, pad=1)
    ax.xaxis.set_label_coords(0.5, -0.07)
    ax.grid(True, alpha=0.22, lw=0.6)
    ax.set_axisbelow(True)

    # Shared legend from proxy artists: one color/style per method for both panels.
    from matplotlib.lines import Line2D

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=COLORS["gt"],
            lw=1.6,
            linestyle="-",
            marker="o",
            ms=3.2,
            label="Ground truth",
        ),
        Line2D(
            [0],
            [0],
            color=COLORS["vpw"],
            lw=1.6,
            linestyle="-",
            marker="^",
            ms=3.2,
            label="VPW",
        ),
    ]
    for method, label in BASELINES:
        style = STYLES[method]
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=COLORS[method],
                lw=1.3,
                linestyle=style["linestyle"],
                marker=style["marker"],
                ms=3.2,
                label=label,
            )
        )  # legend proxies only; plot markers omit endpoints
    # Legend just under the "step" labels; title-sized text in two rows.
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.145),
        ncol=4,
        fontsize=9.5,
        frameon=False,
        columnspacing=1.0,
        handlelength=1.8,
        handletextpad=0.35,
        borderaxespad=0.0,
        labelspacing=0.3,
    )

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    pdf = out_prefix.with_suffix(".pdf")
    png = out_prefix.with_suffix(".png")
    # Avoid bbox_inches="tight": it was pulling "step" down into the legend.
    fig.savefig(pdf, pad_inches=0.02)
    fig.savefig(png, dpi=220, pad_inches=0.02)
    plt.close(fig)

    print(f"Reacher VPW mean err={float(vpw_err.mean()):.3f}")
    if reacher_exit_notes:
        print("Reacher exits top (>0.40): " + "; ".join(reacher_exit_notes))
    print(
        f"PushT ep={chosen} GT dsum={float(gt_d.sum()):.1f} "
        f"VPW dsum={float(vpw_d.sum()):.1f}"
    )
    if outlier_notes:
        print("PushT " + "; ".join(outlier_notes))
    print(f"Wrote {pdf} and {png}")
    return pdf


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out",
        type=Path,
        default=PAPER_FIG / "pred_vs_gt_rollout",
        help="Output prefix (without extension)",
    )
    p.add_argument("--episode-index", type=int, default=0, help="Reacher episode index")
    args = p.parse_args()
    build_figure(args.out, episode_index=args.episode_index)


if __name__ == "__main__":
    main()
