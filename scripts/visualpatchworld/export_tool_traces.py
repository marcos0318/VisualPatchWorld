#!/usr/bin/env python
"""Re-export oracle transition traces with tool (RGB) scene graphs.

Every observation and next observation is replaced by the tool extraction of
the stored 224x224 dataset frame at the same raw index; actions, episode
splits and frame strides are unchanged. Goal fields (the task specification)
are copied as at replanning time. For Reacher the inverse-kinematics elbow
branch is chosen from the previous frame's tool estimate, never from simulator
joint angles.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

try:
    import hdf5plugin  # noqa: F401
except ImportError:  # pragma: no cover
    pass

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from visualpatchworld.hdf5 import require_h5py  # noqa: E402
from visualpatchworld.lewm_env_specs import get_env_spec  # noqa: E402
from visualpatchworld.lewm_graph import row_dict_from_h5  # noqa: E402
from visualpatchworld.tool_observation import graph_text_from_tool_payload  # noqa: E402
from visualpatchworld.visual_tools import PushTHeuristicSceneGraphExtractor, extractor_for_env  # noqa: E402
from visualpatchworld.pusht import graph_from_vlm_payload  # noqa: E402
from visualpatchworld.schema import render_graph_text  # noqa: E402

DATA = ROOT.parent / "le-wm-code/artifacts/visualpatchworld/data"


def _tworoom_agent_only(arr: np.ndarray) -> Dict[str, Any]:
    """Dataset frames do not draw the target; the agent uses the live extractor's mask and mapping."""
    from visualpatchworld.visual_tools import _centroid_from_mask, _img_to_world_xy

    mask = (arr[:, :, 0] > 180) & (arr[:, :, 1] < 120) & (arr[:, :, 2] < 120)
    x, y = _img_to_world_xy(*_centroid_from_mask(mask), width=arr.shape[1], height=arr.shape[0])
    return {"objects": [{"id": "agent", "category": "navigator", "position": {"x": round(x, 3), "y": round(y, 3)}}]}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env", required=True)
    p.add_argument("--splits", default="train,val")
    p.add_argument("--out_dir", default=str(ROOT / "artifacts/visualpatchworld/data_tool"))
    args = p.parse_args()

    from PIL import Image

    spec = get_env_spec(args.env)
    extractor = PushTHeuristicSceneGraphExtractor() if spec.name == "pusht" else extractor_for_env(spec.name)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    h5py = require_h5py()
    tmp = Path(tempfile.mkdtemp(prefix="tooltrace_", dir="/dev/shm" if Path("/dev/shm").is_dir() else None))
    summary: Dict[str, Any] = {"env": spec.name, "splits": {}}

    for split in args.splits.split(","):
        src = DATA / f"{spec.name}_oracle_traj_{split}.jsonl"
        dst = out_dir / f"{spec.name}_tool_traj_{split}.jsonl"
        errs, n, failures = [], 0, 0
        with h5py.File(json.loads(src.open().readline())["metadata"]["dataset"], "r") as h5, \
                src.open() as fin, dst.open("w") as fout:
            for line in fin:
                record = json.loads(line)
                prev_q: Optional[np.ndarray] = None
                cache: Dict[int, str] = {}
                last_payload: Optional[Dict[str, Any]] = None

                def tool_text(raw: int, step: int, meta: Dict[str, Any]) -> str:
                    nonlocal prev_q, failures, last_payload
                    if raw in cache:
                        return cache[raw]
                    row = row_dict_from_h5(h5, raw, spec)
                    frame = np.asarray(h5["pixels"][raw], dtype=np.uint8)
                    path = tmp / f"{raw}.png"
                    Image.fromarray(frame).save(path)
                    hint_row = {"qpos": prev_q} if prev_q is not None else {}
                    try:
                        if spec.name == "tworoom":
                            payload = _tworoom_agent_only(frame)
                        else:
                            payload = extractor.extract(path, metadata={"goal_row": row, "template_row": hint_row})
                    except Exception:
                        failures += 1
                        # As in live replanning, an occluded PushT pusher reuses the previous extraction.
                        if spec.name != "pusht" or last_payload is None:
                            raise
                        payload = last_payload
                    finally:
                        path.unlink(missing_ok=True)
                    last_payload = payload
                    for obj in payload.get("objects", []):
                        if obj.get("id") == "arm" and "qpos" in obj:
                            prev_q = np.asarray(obj["qpos"], dtype=np.float64)
                    if spec.name == "pusht":
                        text = render_graph_text(graph_from_vlm_payload(payload, step=step, metadata=meta,
                                                                        source="tool"))
                    else:
                        text = graph_text_from_tool_payload(spec.name, payload, template_row=row, goal_row=row,
                                                            step=step, metadata=meta)
                    cache[raw] = text
                    return text

                for i, t in enumerate(record["transitions"]):
                    m = t["metadata"]
                    meta = {k: m[k] for k in ("episode_index", "raw_index", "next_raw_index") if k in m}
                    door = (m.get("observation_graph") or {}).get("metadata", {}).get("door_coords")
                    if door is not None:
                        meta["door_coords"] = door
                    t["observation"] = tool_text(int(m["raw_index"]), i, meta)
                    t["next_observation"] = tool_text(int(m["next_raw_index"]), i + 1, meta)
                    for k in ("observation_graph", "next_observation_graph"):
                        m.pop(k, None)
                    n += 1
                record["metadata"]["source"] = "tool"
                fout.write(json.dumps(record) + "\n")
        summary["splits"][split] = {"path": str(dst), "transitions": n, "extraction_failures": failures}
        print(f"[tool-traces] {spec.name}/{split}: {n} transitions -> {dst}", flush=True)
    (out_dir / f"{spec.name}_tool_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
