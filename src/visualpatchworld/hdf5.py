"""Small HDF5 helpers for LeWM trajectory exports."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, Optional, Sequence


@dataclass(frozen=True)
class EpisodeSlice:
    episode_index: int
    start: int
    stop: int

    @property
    def length(self) -> int:
        return max(0, self.stop - self.start)


def resolve_dataset_path(dataset: str, stablewm_home: Optional[str] = None) -> Path:
    """Resolve a LeWM dataset name or path to an HDF5 file."""

    path = Path(dataset).expanduser()
    if path.exists():
        return path
    home = Path(stablewm_home or Path.home() / ".stable-wm").expanduser()
    candidates = [
        home / "datasets" / dataset,
        home / "datasets" / f"{dataset}.h5",
        home / dataset,
        home / f"{dataset}.h5",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not resolve dataset {dataset!r}. Checked explicit path and {home}/datasets."
    )


def require_h5py():
    try:
        import h5py  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError("h5py is required to read LeWM HDF5 datasets") from exc
    return h5py


def _as_int_list(values) -> List[int]:
    return [int(v) for v in list(values)]


def episode_slices(h5: Mapping[str, object], fallback_length: int) -> List[EpisodeSlice]:
    """Infer episode boundaries from common stable-worldmodel HDF5 columns."""

    keys = set(h5.keys())
    length_key = next((key for key in ("ep_len", "episode_lengths", "episode_length") if key in keys), None)
    offset_key = next((key for key in ("ep_offset", "episode_offsets", "episode_offset") if key in keys), None)

    if length_key:
        lengths = _as_int_list(h5[length_key][()])  # type: ignore[index]
        if offset_key:
            offsets = _as_int_list(h5[offset_key][()])  # type: ignore[index]
        else:
            offsets = []
            cursor = 0
            for length in lengths:
                offsets.append(cursor)
                cursor += length
        return [
            EpisodeSlice(i, start, start + length)
            for i, (start, length) in enumerate(zip(offsets, lengths))
            if length > 1
        ]

    episode_key = next((key for key in ("episode_idx", "ep_idx") if key in keys), None)
    if episode_key:
        episode_ids = list(h5[episode_key][()])  # type: ignore[index]
        slices: List[EpisodeSlice] = []
        if not episode_ids:
            return slices
        start = 0
        current = episode_ids[0]
        episode_number = 0
        for idx, value in enumerate(episode_ids[1:], start=1):
            if value != current:
                if idx - start > 1:
                    slices.append(EpisodeSlice(episode_number, start, idx))
                start = idx
                current = value
                episode_number += 1
        if len(episode_ids) - start > 1:
            slices.append(EpisodeSlice(episode_number, start, len(episode_ids)))
        return slices

    return [EpisodeSlice(0, 0, int(fallback_length))]


def split_episode_indices(
    episode_count: int,
    *,
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
) -> Dict[str, Sequence[int]]:
    train_count = int(episode_count * train_fraction)
    val_count = int(episode_count * val_fraction)
    if episode_count >= 3:
        val_count = max(1, val_count)
        test_count = max(1, episode_count - train_count - val_count)
        train_count = max(1, episode_count - val_count - test_count)
    elif episode_count > 0:
        train_count = max(1, train_count)
        val_count = 0
    val_start = train_count
    test_start = min(episode_count, train_count + val_count)
    indices = list(range(episode_count))
    return {
        "train": indices[:val_start],
        "val": indices[val_start:test_start],
        "test": indices[test_start:],
    }


def iter_split_slices(
    slices: Sequence[EpisodeSlice],
    *,
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
) -> Iterator[tuple[str, EpisodeSlice]]:
    split_map = split_episode_indices(
        len(slices),
        train_fraction=train_fraction,
        val_fraction=val_fraction,
    )
    by_index = {episode.episode_index: episode for episode in slices}
    for split, indices in split_map.items():
        for idx in indices:
            episode = by_index.get(idx)
            if episode:
                yield split, episode
