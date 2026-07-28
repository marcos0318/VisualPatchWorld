"""Schema-style coding harness for LeWM continuous control.

Forces: write executable world model → residual backtest on Timeline →
plan only when certified → commit_actions with void-on-mismatch.

Exports are lazy to avoid circular imports with ``structure_induction``.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "BacktestReport",
    "SchemaHarnessConfig",
    "Timeline",
    "residual_backtest",
    "revise_theory",
    "run_schema_harness_eval",
    "run_structure_probes",
    "seed_holdout_ok",
    "should_promote",
    "structure_sketch_from_probes",
    "write_vpw_schema_artifacts",
]


def __getattr__(name: str) -> Any:
    if name in ("SchemaHarnessConfig", "run_schema_harness_eval"):
        from .agent_loop import SchemaHarnessConfig, run_schema_harness_eval

        return {
            "SchemaHarnessConfig": SchemaHarnessConfig,
            "run_schema_harness_eval": run_schema_harness_eval,
        }[name]
    if name in ("BacktestReport", "residual_backtest", "seed_holdout_ok", "should_promote"):
        from .backtest import BacktestReport, residual_backtest, seed_holdout_ok, should_promote

        return {
            "BacktestReport": BacktestReport,
            "residual_backtest": residual_backtest,
            "seed_holdout_ok": seed_holdout_ok,
            "should_promote": should_promote,
        }[name]
    if name in ("run_structure_probes", "structure_sketch_from_probes"):
        from .discovery import run_structure_probes, structure_sketch_from_probes

        return {
            "run_structure_probes": run_structure_probes,
            "structure_sketch_from_probes": structure_sketch_from_probes,
        }[name]
    if name == "revise_theory":
        from .revision import revise_theory

        return revise_theory
    if name == "Timeline":
        from .timeline import Timeline

        return Timeline
    if name == "write_vpw_schema_artifacts":
        from .vpw_adoption import write_vpw_schema_artifacts

        return write_vpw_schema_artifacts
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
