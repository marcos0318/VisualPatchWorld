#!/usr/bin/env python
"""Summarize autoregressive rollout metrics from a PushT benchmark report."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


ROOT = Path(__file__).resolve().parents[2]


def _log(message: str) -> None:
    print(f"[visualpatchworld][{datetime.now().isoformat(timespec='seconds')}][rollout] {message}", flush=True)


def _latest_report(results_dir: Path, source: str) -> Path:
    candidates = sorted((results_dir / source).glob("benchmark_pusht_*.json"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise SystemExit(f"No benchmark report found under {results_dir / source}")
    return candidates[-1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["oracle", "vlm", "tool"], default="oracle")
    parser.add_argument("--results_dir", default=str(ROOT / "artifacts" / "visualpatchworld" / "results"))
    parser.add_argument("--benchmark_report", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--require_rollout",
        action="store_true",
        help="Fail if the benchmark report was not produced with --run_rollout.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_path = Path(args.benchmark_report) if args.benchmark_report else _latest_report(Path(args.results_dir), args.source)
    _log(f"Reading benchmark report: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rollout = report.get("rq2_rollout")
    if rollout is None and args.require_rollout:
        raise SystemExit(f"Benchmark report has no rq2_rollout section: {report_path}")

    summary: Dict[str, Any] = {
        "benchmark_report": str(report_path),
        "env": report.get("env"),
        "source": args.source,
        "model": report.get("model"),
        "available": rollout is not None,
        "rq2_rollout": rollout or {},
    }
    output = (
        Path(args.output)
        if args.output
        else ROOT / "artifacts" / "visualpatchworld" / "reports" / f"pusht_rollout_{args.source}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _log(f"Wrote rollout summary to {output}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
