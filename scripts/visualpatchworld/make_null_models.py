#!/usr/bin/env python
"""Write a null (no-motion) world model per environment.

The null program returns the input state unchanged, so every candidate action
sequence receives the same model score. Under induced scoring it measures the
planner's no-information floor; under hybrid scoring with a random shortlist it
is the "simulator on a random 30% of candidates" control.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts/visualpatchworld"
OUT = ART / "null_models"
ENVS = ("tworoom", "reacher", "cube", "pusht")

NULL_SOURCE = (
    "# HYPOTHESIS: null\n"
    "# MECHANISM: nothing moves; the state is returned unchanged.\n"
    "PARAMS = {}\n\n"
    "def step(state, action, p, meta):\n"
    "    return dict(state)"
)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for env in ENVS:
        src = (ART / f"llm_family/v1_{env}_qwen35_s0/models/offline_multi_step.py").read_text(encoding="utf-8")
        src = re.sub(r"^CANDIDATE_SOURCE = .*$", lambda _: "CANDIDATE_SOURCE = " + repr(NULL_SOURCE), src, count=1, flags=re.M)
        src = re.sub(r"^FITTED_PARAMS = .*$", "FITTED_PARAMS = {}", src, count=1, flags=re.M)
        src = re.sub(r'^"""LLM-proposed.*?"""', '"""Null world model: the state never changes."""', src, count=1, flags=re.S | re.M)
        path = OUT / f"{env}_null.py"
        path.write_text(src, encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
