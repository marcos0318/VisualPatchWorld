#!/usr/bin/env python
"""Summarize one-step PushT world-model dynamics metrics from a benchmark report."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


ROOT = Path(__file__).resolve().parents[2]


def _log(message: str) -> None:
    print(f"[visualpatchworld][{datetime.now().isoformat(timespec='seconds')}][dynamics] {message}", flush=True)


def _latest_report(results_dir: Path, source: str) -> Path:
    candidates = sorted((results_dir / source).glob("benchmark_pusht_*.json"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise SystemExit(f"No benchmark report found under {results_dir / source}")
    return candidates[-1]


def _metric_block(block: Mapping[str, Any]) -> Dict[str, Any]:
    experience = block.get("experience_fit", {})
    observation = block.get("observation_fit", {})
    return {
        "experience_fit": {
            key: experience.get(key)
            for key in (
                "parse_success_rate",
                "predict_success_rate",
                "latent_belief_tracking_rate",
                "overall",
                "n_trajectories",
                "n_transitions",
            )
        },
        "observation_fit": {
            key: observation.get(key)
            for key in (
                "mean_token_f1",
                "mean_token_f1_post_correction",
                "exact_match_rate",
                "state_match_rate",
                "mean_state_field_match_rate",
                "predicted_parse_success_rate",
                "objects_match_rate",
                "n_transitions",
            )
        },
    }


def _summarize_model(report: Mapping[str, Any], key: str) -> Optional[Dict[str, Any]]:
    model = report.get(key)
    if not isinstance(model, Mapping):
        return None
    return {
        "train": _metric_block(model.get("train", {})),
        "eval": _metric_block(model.get("eval", {})),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["oracle", "vlm", "tool"], default="oracle")
    parser.add_argument("--results_dir", default=str(ROOT / "artifacts" / "visualpatchworld" / "results"))
    parser.add_argument("--benchmark_report", default=None)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_path = Path(args.benchmark_report) if args.benchmark_report else _latest_report(Path(args.results_dir), args.source)
    _log(f"Reading benchmark report: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))

    summary: Dict[str, Any] = {
        "benchmark_report": str(report_path),
        "env": report.get("env"),
        "source": args.source,
        "model": report.get("model"),
        "induction": {
            "strategy": report.get("induction", {}).get("strategy"),
            "max_refine_rounds": report.get("induction", {}).get("max_refine_rounds"),
            "initial_error_count": report.get("induction", {}).get("stats", {}).get("initial_error_count"),
            "final_error_count": report.get("induction", {}).get("stats", {}).get("final_error_count"),
            "converged": report.get("induction", {}).get("stats", {}).get("converged"),
        },
        "simple_state": _summarize_model(report, "simple_state"),
        "simple_state_residual": _summarize_model(report, "simple_state_residual"),
    }
    output = (
        Path(args.output)
        if args.output
        else ROOT / "artifacts" / "visualpatchworld" / "reports" / f"pusht_dynamics_{args.source}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _log(f"Wrote dynamics summary to {output}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
