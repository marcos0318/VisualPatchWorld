"""LeWM-aligned trajectory sampling for VisualPatchWorld.

Uses the same clip indexing and train/val split as LeWM training:
  span = num_steps * frameskip
  clip starts at every raw index 0..episode_length-span
  split = random 90/10 on clip indices with seed 3072
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from torch.utils.data import random_split

from .hdf5 import EpisodeSlice, episode_slices, require_h5py, resolve_dataset_path
from .pusht import aggregate_action, discretize_action, make_patchworld_record, make_transition, record_to_jsonl
from .schema import graph_from_pusht_state, render_graph_text


@dataclass(frozen=True)
class LeWMSamplingConfig:
    frameskip: int = 5
    num_steps: int = 4
    train_fraction: float = 0.9
    seed: int = 3072

    @property
    def span(self) -> int:
        return self.num_steps * self.frameskip


@dataclass(frozen=True)
class LeWMClip:
    clip_index: int
    episode_index: int
    local_start: int
    global_start: int
    observation_indices: Tuple[int, ...]
    transition_starts: Tuple[int, ...]


def compute_lewm_clips(slices: Sequence[EpisodeSlice], config: LeWMSamplingConfig) -> List[LeWMClip]:
    """Enumerate LeWM clip starts using stable_worldmodel's Dataset indexing."""

    clips: List[LeWMClip] = []
    clip_index = 0
    span = config.span
    for episode in slices:
        if episode.length < span:
            continue
        for local_start in range(episode.length - span + 1):
            obs_indices = tuple(episode.start + local_start + step * config.frameskip for step in range(config.num_steps))
            transition_starts = obs_indices[:-1]
            clips.append(
                LeWMClip(
                    clip_index=clip_index,
                    episode_index=episode.episode_index,
                    local_start=local_start,
                    global_start=episode.start + local_start,
                    observation_indices=obs_indices,
                    transition_starts=transition_starts,
                )
            )
            clip_index += 1
    return clips


def split_lewm_clips(clips: Sequence[LeWMClip], config: LeWMSamplingConfig) -> Dict[str, List[LeWMClip]]:
    """Split clip indices the same way LeWM splits training clips."""

    if not clips:
        return {"train": [], "val": []}
    train_len = int(len(clips) * config.train_fraction)
    if len(clips) >= 2:
        train_len = max(1, min(train_len, len(clips) - 1))
    val_len = len(clips) - train_len
    indices = list(range(len(clips)))
    train_subset, val_subset = random_split(
        indices,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(config.seed),
    )
    by_split = {
        "train": [clips[i] for i in train_subset.indices],
        "val": [clips[i] for i in val_subset.indices],
    }
    return by_split


def _read_scalar(dataset: Any, idx: int, default: float = 0.0) -> float:
    if dataset is None:
        return default
    try:
        return float(dataset[idx])
    except Exception:
        return default


def _action_block(actions: Any, start: int, frameskip: int) -> List[List[float]]:
    block = actions[start : start + frameskip]
    return [[float(v) for v in row] for row in block]


def flat_action_values(action_block: Sequence[Sequence[float]]) -> List[float]:
    return [float(v) for row in action_block for v in row]


def symbolic_action_from_block(action_block: Sequence[Sequence[float]]) -> str:
    return discretize_action(aggregate_action(action_block))


def _encode_png(pixels: Any, path: Path) -> None:
    try:
        import numpy as np
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise RuntimeError("Sampling image trajectories requires numpy and Pillow") from exc

    arr = np.asarray(pixels)
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
        arr = arr.transpose(1, 2, 0)
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0:
            arr = arr * 255.0
        arr = arr.clip(0, 255).astype("uint8")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def _image_path(image_dir: Path, clip: LeWMClip, obs_step: int) -> Path:
    return image_dir / f"pusht_ep{clip.episode_index:06d}_clip{clip.clip_index:07d}_obs{obs_step:02d}.png"


def build_image_action_record(
    clip: LeWMClip,
    *,
    split: str,
    dataset_path: str,
    pixels: Any,
    actions: Any,
    rewards: Any,
    dones: Any,
    image_dir: Path,
    config: LeWMSamplingConfig,
    overwrite_images: bool = False,
) -> Dict[str, Any]:
    transitions: List[Dict[str, Any]] = []
    image_paths: List[Path] = []
    for obs_step, raw_index in enumerate(clip.observation_indices):
        image_path = _image_path(image_dir, clip, obs_step)
        image_paths.append(image_path)
        if overwrite_images or not image_path.exists():
            _encode_png(pixels[raw_index], image_path)

    for step, raw_index in enumerate(clip.transition_starts):
        next_raw_index = raw_index + config.frameskip
        action_block = _action_block(actions, raw_index, config.frameskip)
        transitions.append(
            {
                "observation_image": str(image_paths[step]),
                "action": symbolic_action_from_block(action_block),
                "action_values": flat_action_values(action_block),
                "action_block": action_block,
                "next_observation_image": str(image_paths[step + 1]),
                "reward": _read_scalar(rewards, next_raw_index),
                "done": bool(_read_scalar(dones, next_raw_index, 0.0)),
                "metadata": {
                    "clip_index": clip.clip_index,
                    "episode_index": clip.episode_index,
                    "local_start": clip.local_start,
                    "global_start": clip.global_start,
                    "step": step,
                    "raw_index": int(raw_index),
                    "next_step": step + 1,
                    "next_raw_index": int(next_raw_index),
                    "frameskip": config.frameskip,
                    "num_steps": config.num_steps,
                    "span": config.span,
                    "lewm_aligned": True,
                },
            }
        )

    return {
        "metadata": {
            "env": "pusht",
            "item_id": f"pusht-lewm-image-action-clip-{clip.clip_index}",
            "task_idx": int(clip.episode_index),
            "rollout_index": int(clip.clip_index),
            "split": split,
            "source": "lewm_image_action",
            "dataset": dataset_path,
            "clip_index": clip.clip_index,
            "local_start": clip.local_start,
            "global_start": clip.global_start,
            "frameskip": config.frameskip,
            "num_steps": config.num_steps,
            "span": config.span,
            "lewm_aligned": True,
        },
        "transitions": transitions,
    }


def build_oracle_record(
    clip: LeWMClip,
    *,
    split: str,
    dataset_path: str,
    states: Any,
    actions: Any,
    rewards: Any,
    dones: Any,
    config: LeWMSamplingConfig,
) -> Dict[str, Any]:
    transitions = []
    for step, raw_index in enumerate(clip.transition_starts):
        next_raw_index = raw_index + config.frameskip
        graph = graph_from_pusht_state(
            states[raw_index],
            source="oracle",
            step=step,
            metadata={
                "clip_index": clip.clip_index,
                "episode_index": clip.episode_index,
                "raw_index": int(raw_index),
                "lewm_aligned": True,
            },
        )
        next_graph = graph_from_pusht_state(
            states[next_raw_index],
            source="oracle",
            step=step + 1,
            metadata={
                "clip_index": clip.clip_index,
                "episode_index": clip.episode_index,
                "raw_index": int(next_raw_index),
                "lewm_aligned": True,
            },
        )
        action_block = _action_block(actions, raw_index, config.frameskip)
        transitions.append(
            make_transition(
                graph,
                aggregate_action(action_block),
                next_graph,
                reward=_read_scalar(rewards, next_raw_index),
                done=bool(_read_scalar(dones, next_raw_index, 0.0)),
                metadata={
                    "clip_index": clip.clip_index,
                    "episode_index": clip.episode_index,
                    "local_start": clip.local_start,
                    "global_start": clip.global_start,
                    "step": step,
                    "raw_index": int(raw_index),
                    "next_raw_index": int(next_raw_index),
                    "source": "oracle",
                    "action_block": action_block,
                    "action_values": flat_action_values(action_block),
                    "frameskip": config.frameskip,
                    "num_steps": config.num_steps,
                    "span": config.span,
                    "lewm_aligned": True,
                },
            )
        )

    record = make_patchworld_record(
        transitions,
        split=split,
        episode_index=clip.episode_index,
        source="oracle",
        dataset=dataset_path,
    )
    record["metadata"].update(
        {
            "item_id": f"pusht-lewm-oracle-clip-{clip.clip_index}",
            "rollout_index": int(clip.clip_index),
            "clip_index": clip.clip_index,
            "local_start": clip.local_start,
            "global_start": clip.global_start,
            "frameskip": config.frameskip,
            "num_steps": config.num_steps,
            "span": config.span,
            "lewm_aligned": True,
        }
    )
    return record


def write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
            count += 1
    return count


def write_oracle_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record_to_jsonl(record) + "\n")
            count += 1
    return count


def sample_lewm_aligned_dataset(
    *,
    dataset: str,
    stablewm_home: Optional[str],
    output_dir: Path,
    sources: Sequence[str],
    config: LeWMSamplingConfig,
    max_clips: int = 0,
    overwrite_images: bool = False,
) -> Dict[str, Any]:
    dataset_path = resolve_dataset_path(dataset, stablewm_home)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "images"
    h5py = require_h5py()

    with h5py.File(dataset_path, "r") as h5:
        if "action" not in h5:
            raise SystemExit("LeWM-aligned sampling requires an `action` dataset.")
        states = h5.get("state")
        pixels = h5.get("pixels")
        actions = h5["action"]
        rewards = h5.get("reward") or h5.get("rewards")
        dones = h5.get("done") or h5.get("dones") or h5.get("terminated") or h5.get("terminals")
        fallback_length = len(actions)
        slices = episode_slices(h5, fallback_length=fallback_length)
        clips = compute_lewm_clips(slices, config)
        if max_clips > 0:
            clips = clips[:max_clips]
        split_clips = split_lewm_clips(clips, config)

        want_image = any(source in ("image_action", "both") for source in sources)
        want_oracle = any(source in ("oracle", "both") for source in sources)
        if want_image and pixels is None:
            raise SystemExit("Image/action export requires `pixels` in the HDF5 file.")
        if want_oracle and states is None:
            raise SystemExit("Oracle export requires `state` in the HDF5 file.")

        image_records: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": []}
        oracle_records: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": []}

        for split, split_items in split_clips.items():
            for clip in split_items:
                if want_image:
                    image_records[split].append(
                        build_image_action_record(
                            clip,
                            split=split,
                            dataset_path=str(dataset_path),
                            pixels=pixels,
                            actions=actions,
                            rewards=rewards,
                            dones=dones,
                            image_dir=image_dir,
                            config=config,
                            overwrite_images=overwrite_images,
                        )
                    )
                if want_oracle:
                    oracle_records[split].append(
                        build_oracle_record(
                            clip,
                            split=split,
                            dataset_path=str(dataset_path),
                            states=states,
                            actions=actions,
                            rewards=rewards,
                            dones=dones,
                            config=config,
                        )
                    )

    summary: Dict[str, Any] = {
        "dataset": str(dataset_path),
        "source": "lewm_aligned",
        "frameskip": config.frameskip,
        "num_steps": config.num_steps,
        "span": config.span,
        "train_fraction": config.train_fraction,
        "seed": config.seed,
        "total_clips": len(clips),
        "image_dir": str(image_dir),
        "splits": {},
    }

    for split in ("train", "val"):
        split_summary: Dict[str, Any] = {}
        if want_image:
            image_path = output_dir / f"pusht_lewm_image_action_traj_{split}.jsonl"
            split_summary["image_action"] = {
                "path": str(image_path),
                "clips": write_jsonl(image_path, image_records[split]),
                "transitions": sum(len(record["transitions"]) for record in image_records[split]),
            }
        if want_oracle:
            oracle_path = output_dir / f"pusht_lewm_oracle_traj_{split}.jsonl"
            split_summary["oracle"] = {
                "path": str(oracle_path),
                "clips": write_oracle_jsonl(oracle_path, oracle_records[split]),
                "transitions": sum(len(record["transitions"]) for record in oracle_records[split]),
            }
        summary["splits"][split] = split_summary

    summary_path = output_dir / "pusht_lewm_sampling_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def _object_fields(obj: Mapping[str, Any]) -> Dict[str, float]:
    """Extract numeric PushT fields from oracle JSON or PatchWorld belief entries."""

    if "position" in obj and isinstance(obj["position"], Mapping):
        pos = obj["position"]
        vel = obj.get("velocity", {})
        fields = {
            "x": float(pos.get("x", 0.0)),
            "y": float(pos.get("y", 0.0)),
            "vx": float(vel.get("x", 0.0)) if isinstance(vel, Mapping) else 0.0,
            "vy": float(vel.get("y", 0.0)) if isinstance(vel, Mapping) else 0.0,
        }
        if "angle" in obj:
            fields["angle"] = float(obj["angle"])
        return fields

    value = obj.get("value", obj)
    if isinstance(value, Mapping):
        props = value.get("properties", value)
        if isinstance(props, Mapping):
            fields = {
                "x": float(props.get("x", 0.0)),
                "y": float(props.get("y", 0.0)),
                "vx": float(props.get("vx", 0.0)),
                "vy": float(props.get("vy", 0.0)),
            }
            if "angle" in props:
                fields["angle"] = float(props["angle"])
            return fields

    return {
        "x": float(obj.get("x", 0.0)),
        "y": float(obj.get("y", 0.0)),
        "vx": float(obj.get("vx", 0.0)),
        "vy": float(obj.get("vy", 0.0)),
        **({"angle": float(obj["angle"])} if "angle" in obj else {}),
    }


def objects_by_id(payload: Mapping[str, Any]) -> Dict[str, Dict[str, float]]:
    """Normalize oracle scene graphs and PatchWorld beliefs to id -> numeric fields."""

    objects = payload.get("objects", {})
    by_id: Dict[str, Dict[str, float]] = {}
    if isinstance(objects, Mapping):
        for obj_id, obj in objects.items():
            if isinstance(obj, Mapping):
                by_id[str(obj_id)] = _object_fields(obj)
        return by_id
    if isinstance(objects, list):
        for obj in objects:
            if isinstance(obj, Mapping) and obj.get("id") is not None:
                by_id[str(obj["id"])] = _object_fields(obj)
    return by_id


def state_vector_from_payload(payload: Mapping[str, Any]) -> List[float]:
    by_id = objects_by_id(payload)
    agent = by_id.get("agent", {})
    block = by_id.get("block", {})
    return [
        float(agent.get("x", 0.0)),
        float(agent.get("y", 0.0)),
        float(block.get("x", 0.0)),
        float(block.get("y", 0.0)),
        float(block.get("angle", 0.0)),
        float(agent.get("vx", 0.0)),
        float(agent.get("vy", 0.0)),
    ]


def belief_from_state_vector(state: Sequence[float], *, step: int = 0) -> Dict[str, Any]:
    graph = graph_from_pusht_state(state, source="simulator", step=step)
    return graph.to_json()


def state_field_errors(
    predicted: Mapping[str, Any],
    target_state: Sequence[float] | Mapping[str, Any],
    *,
    tolerance: float = 5.0,
    velocity_tolerance: float = 25.0,
    angle_tolerance: float = 0.05,
) -> Dict[str, float]:
    """Compare a scene graph or PatchWorld belief against a PushT state vector."""

    if isinstance(target_state, Mapping):
        target = state_vector_from_payload(target_state)
    else:
        target = [float(v) for v in target_state]

    by_id = objects_by_id(predicted)
    agent = by_id.get("agent", {})
    block = by_id.get("block", {})

    agent_pos_err = math.hypot(float(agent.get("x", 0.0)) - target[0], float(agent.get("y", 0.0)) - target[1])
    block_pos_err = math.hypot(float(block.get("x", 0.0)) - target[2], float(block.get("y", 0.0)) - target[3])
    angle_err = abs(float(block.get("angle", 0.0)) - target[4])
    angle_err = min(angle_err, abs(2 * math.pi - angle_err))
    vx_err = abs(float(agent.get("vx", 0.0)) - (target[5] if len(target) > 5 else 0.0))
    vy_err = abs(float(agent.get("vy", 0.0)) - (target[6] if len(target) > 6 else 0.0))
    state_match = (
        agent_pos_err <= tolerance
        and block_pos_err <= tolerance
        and angle_err <= angle_tolerance
        and vx_err <= velocity_tolerance
        and vy_err <= velocity_tolerance
    )
    return {
        "agent_position_error": agent_pos_err,
        "block_position_error": block_pos_err,
        "block_angle_error": angle_err,
        "agent_vx_error": vx_err,
        "agent_vy_error": vy_err,
        "state_match": float(state_match),
    }


def graph_text_from_state(
    state: Sequence[float],
    *,
    step: int = 0,
    metadata: Optional[Mapping[str, Any]] = None,
    block_velocity: Optional[Sequence[float]] = None,
) -> str:
    graph = graph_from_pusht_state(
        state,
        source="simulator",
        step=step,
        metadata=metadata,
        block_velocity=block_velocity,
    )
    return render_graph_text(graph)
