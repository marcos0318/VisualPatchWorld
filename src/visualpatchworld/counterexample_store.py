"""Provenance-aware counterexample store with module filtering and weight capping."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set

from .pusht_local_induction import WeightedTransition


@dataclass(frozen=True)
class StoredCounterexample:
    observation: str
    action: str
    next_observation: str
    weight: float
    module: str = "unknown"
    source: str = "unknown"
    metadata: Mapping[str, Any] | None = None


def _parse_jsonl_record(record: Mapping[str, Any]) -> List[StoredCounterexample]:
    rows: List[StoredCounterexample] = []
    record_meta = record.get("metadata", {})
    default_source = str(record_meta.get("source", "jsonl"))
    for transition in record.get("transitions", []):
        meta = transition.get("metadata", {})
        if not isinstance(meta, dict):
            meta = {}
        weight = float(meta.get("weight", transition.get("weight", 1.0)))
        module = str(meta.get("module", meta.get("failure_module", "unknown")))
        source = str(meta.get("source", default_source))
        rows.append(
            StoredCounterexample(
                observation=str(transition.get("observation", "")),
                action=str(transition.get("action", "")),
                next_observation=str(transition.get("next_observation", "")),
                weight=weight,
                module=module,
                source=source,
                metadata=meta,
            )
        )
    return rows


def load_counterexamples_jsonl(path: Path | str) -> List[StoredCounterexample]:
    rows: List[StoredCounterexample] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            rows.extend(_parse_jsonl_record(record))
    return rows


def load_counterexamples_jsonl_paths(paths: Sequence[Path | str]) -> List[StoredCounterexample]:
    rows: List[StoredCounterexample] = []
    for path in paths:
        path = Path(path)
        if path.is_file():
            rows.extend(load_counterexamples_jsonl(path))
    return rows


@dataclass
class CounterexampleStore:
    rows: List[StoredCounterexample]

    @classmethod
    def from_jsonl_paths(cls, paths: Sequence[Path | str]) -> CounterexampleStore:
        return cls(rows=load_counterexamples_jsonl_paths(paths))

    @classmethod
    def from_rows(cls, rows: Sequence[StoredCounterexample]) -> CounterexampleStore:
        return cls(rows=list(rows))

    def filter(
        self,
        *,
        modules: Optional[Set[str]] = None,
        sources: Optional[Set[str]] = None,
        exclude_modules: Optional[Set[str]] = None,
    ) -> CounterexampleStore:
        filtered: List[StoredCounterexample] = []
        for row in self.rows:
            if modules is not None and row.module not in modules:
                continue
            if sources is not None and row.source not in sources:
                continue
            if exclude_modules is not None and row.module in exclude_modules:
                continue
            filtered.append(row)
        return CounterexampleStore(rows=filtered)

    def deduplicate(self) -> CounterexampleStore:
        best: Dict[tuple[str, str], StoredCounterexample] = {}
        for row in self.rows:
            key = (row.observation.strip(), row.action.strip())
            existing = best.get(key)
            if existing is None or row.weight > existing.weight:
                best[key] = row
        return CounterexampleStore(rows=list(best.values()))

    def cap_extra_weight(
        self,
        base_mass: float,
        *,
        max_extra_fraction: float = 0.3,
    ) -> CounterexampleStore:
        """Scale down extra weights so their total mass <= fraction of base mass."""

        if base_mass <= 0.0 or max_extra_fraction <= 0.0 or not self.rows:
            return CounterexampleStore(rows=[])
        extra_mass = sum(row.weight for row in self.rows)
        cap = float(base_mass * max_extra_fraction)
        if extra_mass <= cap + 1e-9:
            return CounterexampleStore(rows=list(self.rows))
        scale = cap / extra_mass
        capped = [replace(row, weight=float(row.weight * scale)) for row in self.rows]
        return CounterexampleStore(rows=capped)

    def to_weighted_transitions(self) -> List[WeightedTransition]:
        return [
            WeightedTransition(
                observation=row.observation,
                action=row.action,
                next_observation=row.next_observation,
                weight=row.weight,
            )
            for row in self.rows
        ]

    def summary(self) -> Dict[str, Any]:
        modules: Dict[str, int] = {}
        sources: Dict[str, int] = {}
        total_weight = 0.0
        for row in self.rows:
            modules[row.module] = modules.get(row.module, 0) + 1
            sources[row.source] = sources.get(row.source, 0) + 1
            total_weight += row.weight
        return {
            "count": len(self.rows),
            "total_weight": float(total_weight),
            "modules": dict(sorted(modules.items())),
            "sources": dict(sorted(sources.items())),
        }


def parse_module_filter(text: Optional[str]) -> Optional[Set[str]]:
    if not text:
        return None
    values = {part.strip() for part in str(text).split(",") if part.strip()}
    return values or None


def build_capped_extra_transitions(
    *,
    base_mass: float,
    counterexample_paths: Sequence[Path | str],
    modules: Optional[Iterable[str]] = None,
    exclude_modules: Optional[Iterable[str]] = None,
    max_extra_fraction: float = 0.3,
) -> tuple[List[WeightedTransition], Dict[str, Any]]:
    store = CounterexampleStore.from_jsonl_paths(counterexample_paths).deduplicate()
    module_set = set(modules) if modules is not None else None
    exclude_set = set(exclude_modules) if exclude_modules is not None else None
    filtered = store.filter(modules=module_set, exclude_modules=exclude_set)
    capped = filtered.cap_extra_weight(base_mass, max_extra_fraction=max_extra_fraction)
    info = {
        "input": store.summary(),
        "filtered": filtered.summary(),
        "capped": capped.summary(),
        "base_mass": float(base_mass),
        "max_extra_fraction": float(max_extra_fraction),
    }
    return capped.to_weighted_transitions(), info
