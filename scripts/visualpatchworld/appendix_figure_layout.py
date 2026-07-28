"""Shared layout helpers for VisualPatchWorld appendix figures."""

from __future__ import annotations

import re
import textwrap
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

ACM_TEXTWIDTH_PT = 506.295
ACM_TEXTWIDTH_IN = ACM_TEXTWIDTH_PT / 72.0


def setup_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def wrap_block(text: str, *, width: int = 34, max_lines: Optional[int] = None) -> str:
    lines: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        wrapped = textwrap.wrap(line, width=width) or [""]
        lines.extend(wrapped)
        if max_lines is not None and len(lines) >= max_lines:
            break
    if max_lines is not None:
        lines = lines[:max_lines]
        if len(lines) == max_lines and lines:
            lines[-1] = textwrap.shorten(lines[-1], width=width, placeholder="…")
    return "\n".join(lines)


def limit_lines(text: str, *, max_lines: Optional[int]) -> str:
    if max_lines is None:
        return text.strip()
    lines = text.strip().splitlines()
    if len(lines) <= max_lines:
        return "\n".join(lines)
    lines = lines[:max_lines]
    if lines:
        lines[-1] = textwrap.shorten(lines[-1], width=max(8, len(lines[-1])), placeholder="...")
    return "\n".join(lines)


def compact_scene_graph(
    text: str,
    *,
    max_objects: int = 4,
    max_relations: int = 4,
    attr_wrap: int = 46,
) -> str:
    """Format a scene graph for figures without breaking object ids."""
    goal = ""
    objects: list[str] = []
    relations: list[str] = []
    section: Optional[str] = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("Metadata:"):
            continue
        if line.startswith(("Environment:", "Source:", "Step:")):
            continue
        if line.startswith("Goal:"):
            goal = line
            continue
        if line == "Objects:":
            section = "objects"
            continue
        if line == "Relations:":
            section = "relations"
            continue
        if line.startswith("- ") and section == "objects":
            objects.append(line)
        elif line.startswith("- ") and section == "relations":
            relations.append(line)

    lines: list[str] = []
    if goal:
        if len(goal) <= attr_wrap:
            lines.append(goal)
        else:
            lines.extend(textwrap.wrap(goal, width=attr_wrap))
    lines.append("Objects:")
    shown_objects = objects[:max_objects]
    for obj_line in shown_objects:
        lines.extend(_split_object_line(obj_line, wrap_width=attr_wrap))
    if len(objects) > max_objects:
        lines.append(f"  (+{len(objects) - max_objects} more objects)")
    lines.append("Relations:")
    shown_relations = relations[:max_relations]
    for rel_line in shown_relations:
        if len(rel_line) <= attr_wrap:
            lines.append(rel_line)
        else:
            lines.extend(textwrap.wrap(rel_line, width=attr_wrap))
    if len(relations) > max_relations:
        lines.append(f"  (+{len(relations) - max_relations} more relations)")
    return "\n".join(lines)


def _split_object_line(line: str, *, wrap_width: int) -> list[str]:
    """Put object id on one line and wrap attribute chunks below it."""
    match = re.match(r"(-\s+[^:]+:[^\s]+)\s+(.*)", line)
    if not match:
        if len(line) <= wrap_width:
            return [line]
        return textwrap.wrap(line, width=wrap_width)
    head, tail = match.groups()
    chunks = textwrap.wrap(tail, width=max(18, wrap_width - 2)) or [tail]
    return [head, *[f"  {chunk}" for chunk in chunks]]


def chars_for_panel_inches(
    width_in: float,
    *,
    fontsize: float = 7.0,
    mono: bool = True,
    margin_frac: float = 0.18,
) -> int:
    """Estimate safe monospace/sans wrap width for a panel of given width."""
    usable_pt = max(24.0, width_in * 72.0 * (1.0 - margin_frac))
    char_pt = fontsize * (0.62 if mono else 0.52)
    return max(18, int(usable_pt / char_pt))


def draw_title_bar(ax, title: str, *, fontsize: float = 10.0) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(
        0.0,
        0.5,
        title,
        transform=ax.transAxes,
        fontsize=fontsize,
        fontweight="bold",
        va="center",
        ha="left",
        clip_on=False,
    )


def draw_text_panel(
    ax,
    title: str,
    body: str,
    *,
    title_size: float = 8.5,
    body_size: float = 7.0,
    facecolor: str = "#f8fafc",
    edgecolor: str = "#cbd5e1",
    mono: bool = True,
    wrap_width: Optional[int] = None,
    max_lines: Optional[int] = None,
    preserve_lines: bool = False,
) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(
        FancyBboxPatch(
            (0.03, 0.03),
            0.94,
            0.94,
            transform=ax.transAxes,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            linewidth=0.8,
            edgecolor=edgecolor,
            facecolor=facecolor,
            zorder=0,
            clip_on=False,
        )
    )
    if preserve_lines:
        wrapped_body = limit_lines(body, max_lines=max_lines)
    else:
        if wrap_width is None:
            wrap_width = 36
        wrapped_body = wrap_block(body, width=wrap_width, max_lines=max_lines)
    ax.text(
        0.07,
        0.90,
        title,
        transform=ax.transAxes,
        fontsize=title_size,
        fontweight="bold",
        va="top",
        ha="left",
        clip_on=True,
        zorder=1,
    )
    ax.text(
        0.07,
        0.80,
        wrapped_body,
        transform=ax.transAxes,
        fontsize=body_size,
        va="top",
        ha="left",
        family="monospace" if mono else "sans-serif",
        linespacing=1.12,
        clip_on=True,
        zorder=1,
    )


def draw_footer(ax, text: str, *, fontsize: float = 7.0, wrap_width: Optional[int] = None) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    if wrap_width is None:
        wrap_width = 52
    ax.text(
        0.0,
        0.9,
        wrap_block(text, width=wrap_width, max_lines=5),
        transform=ax.transAxes,
        fontsize=fontsize,
        va="top",
        ha="left",
        linespacing=1.15,
        clip_on=True,
    )


def draw_image_panel(ax, frame, *, pad_frac: float = 0.05, square: bool = True) -> None:
    """Draw a frame with preserved aspect ratio inside an optional square axes box."""
    image = np.asarray(frame)
    height, width = image.shape[:2]
    pad = max(1, int(pad_frac * max(width, height)))
    ax.imshow(image, aspect="equal", interpolation="nearest")
    ax.set_xlim(-pad, width + pad)
    ax.set_ylim(height + pad, -pad)
    ax.set_axis_off()
    if square:
        ax.set_box_aspect(1)


def save_figure(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = dict(facecolor="white", edgecolor="none")
    fig.savefig(output_path, bbox_inches=None, pad_inches=0, **kwargs)
    fig.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches=None, pad_inches=0, **kwargs)
    plt.close(fig)


def outer_grid(fig, *, rows: int = 2, cols: int = 2, height: float = 11.0) -> plt.GridSpec:
    return fig.add_gridspec(
        rows,
        cols,
        figure=fig,
        left=0.035,
        right=0.965,
        top=0.985,
        bottom=0.015,
        wspace=0.16,
        hspace=0.0,
    )
