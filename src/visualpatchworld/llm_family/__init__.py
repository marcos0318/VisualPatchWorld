"""LLM-proposed hypothesis families for code world-model induction.

Instead of a hand-written per-environment sketch family, a code LLM proposes a
set of qualitatively different parametric transition programs from the state
schema and a handful of example transitions. Every candidate is fit on the same
offline traces, and a candidate is selected either offline (held-out rollout
error) or actively (simulator probes placed where fitted candidates disagree).
"""
