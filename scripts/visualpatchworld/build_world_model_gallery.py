#!/usr/bin/env python3
"""Build static gallery assets for induced world-model examples."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import h5py
import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
import sys

sys.path.insert(0, str(SRC_ROOT))

from visualpatchworld.hdf5 import resolve_dataset_path  # noqa: E402
from visualpatchworld.lewm_env_specs import get_env_spec  # noqa: E402
from visualpatchworld.lewm_env_specs import make_gym_env  # noqa: E402
from visualpatchworld.lewm_graph import graph_from_row, graph_text_from_row, render_env_graph_text, reset_env_from_rows, row_dict_from_h5  # noqa: E402

DOCS_DIR = REPO_ROOT / "docs"
ASSET_DIR = DOCS_DIR / "world_model_examples"
HTML_PATH = DOCS_DIR / "world_model_examples.html"

FRAME_INDICES = {
    "pusht": 120,
    "tworoom": 500,
    "cube": 800,
    "reacher": 300,
}

JSONL_PATHS = {
    "pusht": REPO_ROOT / "artifacts/visualpatchworld/data_stride1/pusht_oracle_traj_train.jsonl",
    "tworoom": REPO_ROOT / "artifacts/visualpatchworld/data/tworoom_oracle_traj_train.jsonl",
    "cube": REPO_ROOT / "artifacts/visualpatchworld/data/cube_oracle_traj_train.jsonl",
    "reacher": REPO_ROOT / "artifacts/visualpatchworld/data/reacher_oracle_traj_train.jsonl",
}

MODEL_PATHS = {
    "pusht": REPO_ROOT / "artifacts/visualpatchworld/induced/program_induced_best.py",
    "tworoom": REPO_ROOT
    / "artifacts/visualpatchworld/lewm_tworoom_general_results/generated_models/tworoom_patchworld_20260622_165539.py",
    "cube": REPO_ROOT
    / "artifacts/visualpatchworld/lewm_cube_general_results/generated_models/cube_patchworld_20260622_165657.py",
    "reacher": REPO_ROOT
    / "artifacts/visualpatchworld/lewm_reacher_joint_results/generated_models/reacher_patchworld_20260623_144440.py",
}

STRUCTURE_REPORT = REPO_ROOT / "artifacts/visualpatchworld/induced/refit.log"


def _encode_png(pixels: Any, path: Path) -> None:
    arr = np.asarray(pixels)
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
        arr = arr.transpose(1, 2, 0)
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0:
            arr = arr * 255.0
        arr = arr.clip(0, 255).astype("uint8")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def _load_h5_pixels(h5: h5py.File, idx: int) -> Optional[np.ndarray]:
    if "pixels" not in h5:
        return None
    try:
        return np.asarray(h5["pixels"][idx])
    except OSError:
        return None


def _render_with_gym(env: str, start_row: Mapping[str, Any], goal_row: Mapping[str, Any]) -> Optional[np.ndarray]:
    spec = get_env_spec(env)
    env_obj = make_gym_env(spec, render_mode="rgb_array")
    try:
        reset_env_from_rows(env_obj, spec, start_row, goal_row)
        frame = env_obj.render()
        if frame is None:
            return None
        return np.asarray(frame)
    except Exception:
        return None
    finally:
        env_obj.close()


def _draw_schematic(env: str, graph_json: Mapping[str, Any], size: int = 224) -> np.ndarray:
    img = Image.new("RGB", (size, size), "#f8fafc")
    draw = ImageDraw.Draw(img)
    objects = {obj["id"]: obj for obj in graph_json.get("objects", [])}

    def pos(obj_id: str) -> Tuple[float, float]:
        obj = objects[obj_id]
        p = obj.get("position", obj)
        x = float(p.get("x", 0.0))
        y = float(p.get("y", 0.0))
        if env == "cube":
            px = int((x + 0.5) * size)
            py = int((1.0 - (y + 0.5)) * size)
        elif env == "reacher":
            px = int((x + 3.0) / 6.0 * size)
            py = int((1.0 - (y + 1.5) / 3.0) * size)
        else:
            px = int(x / 512.0 * size)
            py = int(y / 512.0 * size)
        return px, py

    colors = {
        "agent": "#2563eb",
        "navigator": "#2563eb",
        "effector": "#2563eb",
        "arm": "#64748b",
        "block": "#ea580c",
        "target": "#16a34a",
        "goal": "#16a34a",
        "finger": "#9333ea",
    }

    if env == "reacher" and "arm" in objects and "finger" in objects:
        origin = (size // 2, size - 24)
        ax, ay = pos("arm")
        fx, fy = pos("finger")
        draw.line([origin, (ax, ay), (fx, fy)], fill="#94a3b8", width=4)

    for obj_id, obj in objects.items():
        x, y = pos(obj_id)
        r = 10 if obj_id != "block" else 12
        color = colors.get(obj_id, "#334155")
        draw.ellipse([x - r, y - r, x + r, y + r], fill=color, outline="#ffffff", width=2)
        draw.text((x - 18, y + r + 2), obj_id, fill="#334155")

    draw.text((8, 8), f"{env} (schematic from oracle state)", fill="#64748b")
    return np.asarray(img)


def _capture_frame(
    env: str,
    h5: h5py.File,
    idx: int,
    start_row: Mapping[str, Any],
    goal_row: Mapping[str, Any],
    graph_json: Mapping[str, Any],
) -> Tuple[np.ndarray, str]:
    pixels = _load_h5_pixels(h5, idx)
    if pixels is not None:
        return pixels, f"HDF5 pixels[{idx}]"

    rendered = _render_with_gym(env, start_row, goal_row)
    if rendered is not None:
        return rendered, "simulator rgb_array render"

    return _draw_schematic(env, graph_json), "schematic from oracle coordinates (headless fallback)"


def _graph_from_json(data: Mapping[str, Any]) -> str:
    from visualpatchworld.schema import Relation, SceneGraph, SceneObject

    objects = []
    for obj in data.get("objects", []):
        pos = obj.get("position", {})
        attrs = dict(obj.get("attributes", {}))
        objects.append(
            SceneObject(
                object_id=obj["id"],
                category=obj.get("category", obj["id"]),
                x=float(pos.get("x", 0.0)),
                y=float(pos.get("y", 0.0)),
                angle=float(obj["angle"]) if "angle" in obj else None,
                vx=float(obj["velocity"]["x"]) if "velocity" in obj else None,
                vy=float(obj["velocity"]["y"]) if "velocity" in obj else None,
                attributes=attrs,
            )
        )
    relations = [
        Relation(r["subject"], r["predicate"], r["object"], r.get("value"))
        for r in data.get("relations", [])
    ]
    graph = SceneGraph(
        env=data.get("env", ""),
        source=data.get("source", "oracle"),
        step=int(data.get("step", 0)),
        objects=objects,
        relations=relations,
        goal=data.get("goal", ""),
        metadata=data.get("metadata", {}),
    )
    return render_env_graph_text(graph)


def _svg_for_graph(graph_json: Mapping[str, Any], *, width: int = 320, height: int = 220) -> str:
    objects = graph_json.get("objects", [])
    if not objects:
        return f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}"></svg>'

    xs, ys = [], []
    for obj in objects:
        pos = obj.get("position", obj)
        xs.append(float(pos.get("x", 0.0)))
        ys.append(float(pos.get("y", 0.0)))

    env = graph_json.get("env", "")
    if env == "cube":
        xs = [x * width for x in xs]
        ys = [(1.0 - (y + 0.5)) * height for y in ys]
    elif env == "reacher":
        xs = [(x + 3.0) / 6.0 * width for x in xs]
        ys = [(1.0 - (y + 1.5) / 3.0) * height for y in ys]
    else:
        xs = [x / 512.0 * width for x in xs]
        ys = [y / 512.0 * height for y in ys]

    colors = {
        "agent": "#2563eb",
        "navigator": "#2563eb",
        "effector": "#2563eb",
        "arm": "#64748b",
        "block": "#ea580c",
        "target": "#16a34a",
        "goal": "#16a34a",
        "finger": "#9333ea",
    }
    nodes = []
    for obj, x, y in zip(objects, xs, ys):
        color = colors.get(obj["id"], "#334155")
        nodes.append((obj["id"], x, y, color))

    lines = [
        f'<svg class="graph-svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="#f8fafc" rx="8"/>',
    ]
    for rel in graph_json.get("relations", []):
        subj = rel.get("subject")
        obj_id = rel.get("object")
        pred = rel.get("predicate")
        if pred not in {"distance_to", "near"}:
            continue
        src = next((n for n in nodes if n[0] == subj), None)
        dst = next((n for n in nodes if n[0] == obj_id), None)
        if not src or not dst:
            continue
        stroke = "#cbd5e1" if pred == "distance_to" else "#94a3b8"
        dash = "" if pred == "distance_to" else ' stroke-dasharray="4 3"'
        lines.append(
            f'<line x1="{src[1]:.1f}" y1="{src[2]:.1f}" x2="{dst[1]:.1f}" y2="{dst[2]:.1f}" '
            f'stroke="{stroke}" stroke-width="1.5"{dash}/>'
        )

    for obj_id, x, y, color in nodes:
        lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="10" fill="{color}" stroke="#fff" stroke-width="2"/>')
        lines.append(
            f'<text x="{x:.1f}" y="{y + 22:.1f}" text-anchor="middle" font-size="11" fill="#334155">{obj_id}</text>'
        )
    lines.append("</svg>")
    return "\n".join(lines)


def _load_structure_rules() -> Dict[str, Any]:
    if not STRUCTURE_REPORT.exists():
        return {}
    text = STRUCTURE_REPORT.read_text()
    match = re.search(r"\{\n\s+\"sketch\"", text)
    if not match:
        return {}
    start = text.rfind("{", 0, match.start() + 1)
    depth = 0
    end = start
    for idx in range(start, len(text)):
        if text[idx] == "{":
            depth += 1
        elif text[idx] == "}":
            depth -= 1
            if depth == 0:
                end = idx + 1
                break
    return json.loads(text[start:end])


def _rules_for_env(env: str, structure: Mapping[str, Any]) -> List[str]:
    if env == "pusht":
        sketch = structure.get("sketch", ["pd", "quasi_static", "contact_driven"])
        return [
            f"Structure sketch (active probing): {', '.join(sketch)}.",
            "Agent follows proportional-derivative control toward scaled action target (KP≈92, KV≈26, SCALE≈98).",
            "Block is quasi-static: pushed only through swept agent–T-block contact with normal/tangential springs.",
            "Contact geometry uses T-polygon vertices detected from pixels; agent radius ≈ 14.5 px.",
            "Valid actions: move_left/right/up/down/hold with [dx, dy, speed] discretization.",
        ]
    if env == "tworoom":
        return [
            "Linear scaffold-fit on agent and target coordinates with pairwise distance features.",
            "Wall and door constraints inferred from metadata door_coords; motion clipped at room boundaries.",
            "Target object is static; agent position updates from fitted linear transition matrix.",
            "Valid actions: move_left/right/up/down/hold with [dx, dy, speed].",
        ]
    if env == "cube":
        return [
            "Linear scaffold-fit on effector, block, and goal coordinates (6-D state).",
            "5-D act[] control vector mapped through ridge-regression coefficients.",
            "Goal pose is static; block and effector positions updated jointly from fitted deltas.",
            "Valid actions: act[a0..a4=..., speed=...].",
        ]
    if env == "reacher":
        return [
            "Joint-angle dynamics: qpos_{t+1} = qpos_t + B·action from ridge fit on oracle trajectories.",
            "Fingertip position from analytic forward kinematics on (q1, q2).",
            "Target is static; success when finger–target distance < 0.05.",
            "Valid actions: move_left/right/up/down/hold with [dx, dy, speed].",
        ]
    return []


def _action_examples(env: str, transitions: Sequence[Mapping[str, Any]], limit: int = 3) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for transition in transitions:
        meta = transition.get("metadata", {})
        before = meta.get("observation_graph")
        after = meta.get("next_observation_graph")
        action = transition.get("action", "")
        if not before or not after or not action:
            continue
        out.append(
            {
                "action": action,
                "before": _graph_from_json(before),
                "after": _graph_from_json(after),
            }
        )
        if len(out) >= limit:
            break
    return out


def _model_snippet(path: Path, max_lines: int = 28) -> str:
    lines = path.read_text().splitlines()
    return "\n".join(lines[:max_lines])


def build_manifest() -> Dict[str, Any]:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    structure = _load_structure_rules()
    examples: List[Dict[str, Any]] = []

    for env in ("pusht", "tworoom", "cube", "reacher"):
        spec = get_env_spec(env)
        dataset = resolve_dataset_path(spec.dataset)
        idx = FRAME_INDICES[env]
        image_name = f"{env}_frame.png"
        image_path = ASSET_DIR / image_name

        with h5py.File(dataset, "r") as h5:
            row = row_dict_from_h5(h5, idx, spec)
            goal_row = row_dict_from_h5(h5, min(idx + 25 * spec.frameskip, len(h5["action"]) - 1), spec)
            graph = graph_from_row(env, row, source="oracle", step=0, metadata={"raw_index": idx})
            scene_text = render_env_graph_text(graph)
            graph_json = graph.to_json()
            frame, frame_source = _capture_frame(env, h5, idx, row, goal_row, graph_json)
            _encode_png(frame, image_path)

        with JSONL_PATHS[env].open() as fh:
            episode = json.loads(fh.readline())
        transitions = episode.get("transitions", [])

        model_path = MODEL_PATHS[env]
        examples.append(
            {
                "env": env,
                "title": {
                    "pusht": "PushT (contact manipulation)",
                    "tworoom": "Two-room navigation",
                    "cube": "Cube manipulation (OGBench)",
                    "reacher": "Reacher (dm_control)",
                }[env],
                "image": f"world_model_examples/{image_name}",
                "frame_source": frame_source,
                "scene_graph_text": scene_text,
                "scene_graph_svg": _svg_for_graph(graph_json),
                "rules": _rules_for_env(env, structure),
                "actions": _action_examples(env, transitions),
                "model_path": str(model_path.relative_to(REPO_ROOT)),
                "model_snippet": _model_snippet(model_path),
                "model_class": re.search(r"class\s+(\w+)", model_path.read_text()).group(1),
            }
        )

    manifest = {"generated_by": "scripts/visualpatchworld/build_world_model_gallery.py", "examples": examples}
    manifest_path = ASSET_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _html_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def render_html(manifest: Mapping[str, Any]) -> str:
    cards = []
    for ex in manifest["examples"]:
        action_blocks = []
        for idx, action in enumerate(ex["actions"], start=1):
            action_blocks.append(
                f"""
                <details class="action-example">
                  <summary>Example {idx}: <code>{_html_escape(action['action'])}</code></summary>
                  <div class="action-grid">
                    <div><h4>Before</h4><pre>{_html_escape(action['before'])}</pre></div>
                    <div><h4>After</h4><pre>{_html_escape(action['after'])}</pre></div>
                  </div>
                </details>
                """
            )
        rules = "".join(f"<li>{_html_escape(rule)}</li>" for rule in ex["rules"])
        cards.append(
            f"""
            <section class="env-card" id="{ex['env']}">
              <header>
                <h2>{_html_escape(ex['title'])}</h2>
                <p class="model-ref">Induced model: <code>{_html_escape(ex['model_class'])}</code>
                (<span class="path">{_html_escape(ex['model_path'])}</span>)</p>
              </header>
              <div class="card-grid">
                <figure>
                  <img src="{_html_escape(ex['image'])}" alt="{_html_escape(ex['title'])} frame" loading="lazy"/>
                  <figcaption>{_html_escape(ex.get('frame_source', 'frame'))}</figcaption>
                </figure>
                <div class="graph-panel">
                  <h3>Scene graph (oracle)</h3>
                  {ex['scene_graph_svg']}
                  <pre class="scene-text">{_html_escape(ex['scene_graph_text'])}</pre>
                </div>
              </div>
              <div class="rules-panel">
                <h3>Discovered / induced rules</h3>
                <ul>{rules}</ul>
              </div>
              <div class="model-panel">
                <h3>Executable model (snippet)</h3>
                <pre class="code-snippet">{_html_escape(ex['model_snippet'])}</pre>
              </div>
              <div class="actions-panel">
                <h3>Transition examples (JSONL)</h3>
                {''.join(action_blocks)}
              </div>
            </section>
            """
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>VisualPatchWorld — Generated World Model Examples</title>
  <style>
    :root {{
      --bg: #f1f5f9;
      --card: #ffffff;
      --text: #0f172a;
      --muted: #64748b;
      --accent: #2563eb;
      --border: #e2e8f0;
      --code-bg: #0b1220;
      --code-fg: #e2e8f0;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
    }}
    .page {{
      max-width: 1100px;
      margin: 0 auto;
      padding: 2rem 1.25rem 4rem;
    }}
    header.page-header {{
      margin-bottom: 2rem;
    }}
    header.page-header h1 {{
      margin: 0 0 0.5rem;
      font-size: 1.85rem;
    }}
    header.page-header p {{
      margin: 0.35rem 0;
      color: var(--muted);
      max-width: 70ch;
    }}
    nav.toc {{
      display: flex;
      flex-wrap: wrap;
      gap: 0.5rem;
      margin: 1rem 0 2rem;
    }}
    nav.toc a {{
      padding: 0.35rem 0.75rem;
      border-radius: 999px;
      background: var(--card);
      border: 1px solid var(--border);
      color: var(--accent);
      text-decoration: none;
      font-size: 0.92rem;
    }}
    .env-card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 1.25rem 1.25rem 1.5rem;
      margin-bottom: 1.5rem;
      box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
    }}
    .env-card h2 {{ margin: 0 0 0.35rem; font-size: 1.35rem; }}
    .model-ref {{ margin: 0; color: var(--muted); font-size: 0.92rem; }}
    .model-ref .path {{ font-family: ui-monospace, monospace; font-size: 0.82rem; }}
    .card-grid {{
      display: grid;
      grid-template-columns: minmax(220px, 1fr) minmax(280px, 1.2fr);
      gap: 1rem;
      margin-top: 1rem;
    }}
    figure {{ margin: 0; }}
    figure img {{
      width: 100%;
      border-radius: 10px;
      border: 1px solid var(--border);
      background: #fff;
    }}
    figcaption {{ margin-top: 0.45rem; font-size: 0.82rem; color: var(--muted); }}
    .graph-panel h3, .rules-panel h3, .model-panel h3, .actions-panel h3 {{
      margin: 1rem 0 0.5rem;
      font-size: 1rem;
    }}
    .graph-svg {{
      display: block;
      width: 100%;
      max-width: 360px;
      border: 1px solid var(--border);
      border-radius: 8px;
      margin-bottom: 0.5rem;
    }}
    pre {{
      margin: 0;
      padding: 0.75rem;
      border-radius: 8px;
      overflow-x: auto;
      font-size: 0.78rem;
      line-height: 1.45;
    }}
    pre.scene-text {{
      background: #f8fafc;
      border: 1px solid var(--border);
      color: #1e293b;
    }}
    pre.code-snippet {{
      background: var(--code-bg);
      color: var(--code-fg);
    }}
    .rules-panel ul {{
      margin: 0;
      padding-left: 1.2rem;
    }}
    .rules-panel li {{ margin: 0.35rem 0; }}
    .action-example {{
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 0.5rem 0.75rem;
      margin-bottom: 0.5rem;
      background: #fafafa;
    }}
    .action-example summary {{
      cursor: pointer;
      font-size: 0.9rem;
    }}
    .action-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 0.75rem;
      margin-top: 0.75rem;
    }}
    .action-grid h4 {{
      margin: 0 0 0.35rem;
      font-size: 0.82rem;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    code {{
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 0.88em;
    }}
    @media (max-width: 820px) {{
      .card-grid, .action-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <header class="page-header">
      <h1>Generated executable world models</h1>
      <p>
        Concrete examples from the VisualPatchWorld pipeline: an RGB frame, the corresponding
        oracle scene graph lifted from LeWM trajectories, the induced transition rules, and
        PatchWorld JSONL transitions (before → action → after).
      </p>
      <p>
        Regenerate with
        <code>python scripts/visualpatchworld/build_world_model_gallery.py</code>.
      </p>
      <nav class="toc">
        {''.join(f'<a href="#{ex["env"]}">{_html_escape(ex["title"])}</a>' for ex in manifest["examples"])}
      </nav>
    </header>
    {''.join(cards)}
  </div>
</body>
</html>
"""


def main() -> None:
    manifest = build_manifest()
    HTML_PATH.write_text(render_html(manifest), encoding="utf-8")
    print(f"Wrote {ASSET_DIR / 'manifest.json'}")
    print(f"Wrote {HTML_PATH}")


if __name__ == "__main__":
    main()
