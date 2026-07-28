#!/usr/bin/env python3
"""Build a publication-quality appendix figure for induced world-model examples."""

from __future__ import annotations

import json
import re
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import matplotlib.pyplot as plt
from PIL import Image

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from appendix_figure_layout import (  # noqa: E402
    ACM_TEXTWIDTH_IN,
    chars_for_panel_inches,
    compact_scene_graph,
    draw_footer,
    draw_image_panel,
    draw_text_panel,
    draw_title_bar,
    outer_grid,
    save_figure,
    setup_matplotlib,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "artifacts" / "visualpatchworld" / "world_model_examples" / "manifest.json"
DEFAULT_OUT = REPO_ROOT / "artifacts" / "visualpatchworld" / "figures" / "world_model_examples.pdf"

ENV_ORDER = ("pusht", "tworoom", "cube", "reacher")
ENV_LABELS = {
    "pusht": "PushT",
    "tworoom": "Two-room",
    "cube": "Cube",
    "reacher": "Reacher",
}
CELL_WIDTH_IN = ACM_TEXTWIDTH_IN * 0.92 / 2.0
GRAPH_ATTR_WRAP = chars_for_panel_inches(CELL_WIDTH_IN * 0.70, fontsize=5.6, mono=True)
RULES_WRAP = chars_for_panel_inches(CELL_WIDTH_IN, fontsize=5.6, mono=False)


def _compact_rules(rules: Sequence[str], limit: int = 3) -> str:
    bullets = []
    for rule in rules[:limit]:
        rule = rule.replace("≈", "~").replace("–", "-")
        bullets.append(f"• {rule}")
    return "\n".join(bullets)


def _compact_action(action: Mapping[str, str]) -> str:
    act = action.get("action", "")
    before = action.get("before", "")
    after = action.get("after", "")

    def agent_pos(text: str) -> str:
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped.startswith("-"):
                continue
            if not any(
                stripped.startswith(prefix)
                for prefix in ("- agent:", "- effector:", "- finger:", "- arm:")
            ):
                continue
            match = re.search(r"\bx=([-\d.]+)\s+y=([-\d.]+)", stripped)
            if match:
                label = stripped.split(":")[0].lstrip("- ").strip()
                return f"{label}=({match.group(1)}, {match.group(2)})"
        return "?"

    action_short = textwrap.shorten(act, width=62, placeholder="...")
    return f"Action:\n  {action_short}\nBefore → After: {agent_pos(before)} → {agent_pos(after)}"


def _draw_panel(fig: plt.Figure, spec: plt.GridSpec, example: Mapping[str, Any]) -> None:
    gs = spec.subgridspec(
        3,
        2,
        height_ratios=[0.07, 0.42, 0.51],
        width_ratios=[0.30, 0.70],
        wspace=0.10,
        hspace=0.05,
    )
    ax_title = fig.add_subplot(gs[0, :])
    ax_img = fig.add_subplot(gs[1, 0])
    ax_graph = fig.add_subplot(gs[1, 1])
    ax_rules = fig.add_subplot(gs[2, :])

    title = ENV_LABELS.get(example["env"], example.get("title", example["env"]))
    draw_title_bar(ax_title, title)

    image_path = REPO_ROOT / "docs" / example["image"]
    draw_image_panel(ax_img, Image.open(image_path), square=True)

    draw_text_panel(
        ax_graph,
        "Scene graph (oracle)",
        compact_scene_graph(example["scene_graph_text"], attr_wrap=GRAPH_ATTR_WRAP),
        body_size=5.6,
        max_lines=12,
        preserve_lines=True,
    )

    rules = _compact_rules(example.get("rules", []))
    action = _compact_action(example.get("actions", [{}])[0])
    model = example.get("model_class", "WorldModel")
    footer = f"Induced rules\n{rules}\n\nExample transition\n{action}\n\nModel: {model}"
    draw_text_panel(
        ax_rules,
        "Induced dynamics",
        footer,
        mono=False,
        body_size=5.6,
        wrap_width=RULES_WRAP,
        max_lines=12,
        facecolor="#ffffff",
    )


def build_figure(manifest_path: Path = MANIFEST, output_path: Path = DEFAULT_OUT) -> Path:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_env = {ex["env"]: ex for ex in manifest["examples"]}
    examples = [by_env[env] for env in ENV_ORDER if env in by_env]

    setup_matplotlib()
    fig = plt.figure(figsize=(ACM_TEXTWIDTH_IN, 7.35), dpi=300)
    grid = outer_grid(fig, height=7.35)

    for idx, example in enumerate(examples):
        row, col = divmod(idx, 2)
        _draw_panel(fig, grid[row, col], example)

    save_figure(fig, output_path)
    return output_path


def main() -> None:
    out = build_figure()
    print(f"Wrote {out}")
    print(f"Wrote {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
