"""Helpers to adopt Schema process into main VPW induction artifacts.

Call from induce scripts to persist:
  - world_model.accepted.py
  - notes.md (hypothesis / residual / promotion log)
  - curated Timeline fit report
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from patchworld.worldmodel_data import Trajectory

from .backtest import residual_backtest
from .timeline import Timeline


def write_vpw_schema_artifacts(
    *,
    output_dir: str | Path,
    env_name: str,
    model_code: str,
    train_trajectories: Sequence[Trajectory],
    notes: Optional[Sequence[str]] = None,
    residual_epsilon: float = 0.5,
    pass_threshold: float = 0.8,
) -> Dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    accepted = out / "world_model.accepted.py"
    accepted.write_text(model_code, encoding="utf-8")
    (out / "world_model.py").write_text(model_code, encoding="utf-8")

    timeline = Timeline(env=env_name)
    timeline.extend_from_trajectories(train_trajectories, source="seed")
    for note in notes or []:
        timeline.add_note(str(note))
    report = residual_backtest(
        model_code,
        timeline.curated_for_fit(),
        epsilon=residual_epsilon,
        pass_threshold=pass_threshold,
    )
    timeline.add_note(
        f"accepted_fit pass={report.pass_rate:.3f} median={report.median_residual:.4f} "
        f"mean={report.mean_residual:.4f} max={report.max_residual:.4f}"
    )
    notes_path = out / "notes.md"
    notes_path.write_text("\n".join(timeline.notes) + "\n", encoding="utf-8")
    timeline.save(out / "timeline_seed.json")
    payload = {
        "env": env_name,
        "accepted_model": str(accepted),
        "notes": str(notes_path),
        "backtest": report.to_dict(),
    }
    (out / "schema_adoption_report.json").write_text(
        __import__("json").dumps(payload, indent=2),
        encoding="utf-8",
    )
    return payload
