"""Path helpers for VisualPatchWorld artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional


def resolve_sample_jsonl(sample_dir: Path, split: str) -> Path:
    candidates = [
        sample_dir / f"pusht_lewm_image_action_traj_{split}.jsonl",
        sample_dir / f"pusht_image_action_traj_{split}.jsonl",
    ]
    for candidate in candidates:
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    raise FileNotFoundError(
        f"No image/action JSONL found for split={split!r} under {sample_dir}"
    )


def resolve_oracle_jsonl(data_dir: Path, split: str) -> Path:
    candidates = [
        data_dir / f"pusht_lewm_oracle_traj_{split}.jsonl",
        data_dir / f"pusht_oracle_traj_{split}.jsonl",
    ]
    for candidate in candidates:
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    raise FileNotFoundError(
        f"No oracle JSONL found for split={split!r} under {data_dir}"
    )


def resolve_vlm_jsonl(data_dir: Path, split: str) -> Path:
    path = data_dir / f"pusht_vlm_traj_{split}.jsonl"
    if path.is_file() and path.stat().st_size > 0:
        return path
    raise FileNotFoundError(f"No VLM trajectory JSONL found: {path}")


def vlm_cache_key(metadata: Mapping[str, object]) -> str:
    if metadata.get("lewm_aligned") and "clip_index" in metadata:
        clip_index = int(metadata["clip_index"])
        step = int(metadata.get("step", metadata.get("obs_step", 0)))
        return f"pusht_clip{clip_index:07d}_obs{step:02d}"
    episode_index = int(metadata["episode_index"])
    step = int(metadata["step"])
    return f"pusht_ep{episode_index:06d}_step{step:06d}"


def vlm_cache_key_next(metadata: Mapping[str, object]) -> str:
    next_meta = dict(metadata)
    if "next_step" in metadata:
        next_meta["step"] = metadata["next_step"]
    return vlm_cache_key(next_meta)
