"""Black-box reset/step access expressed in scene-graph state keys.

The adapter only translates between ``{object.field: value}`` and each
environment's reset/step API; it encodes no dynamics.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from visualpatchworld.lewm_env_specs import get_env_spec, make_gym_env, sim_planning_render_mode
from visualpatchworld.lewm_graph import reset_env_from_rows, sim_row_from_obs, sim_step_with_info


class BlackBoxEnv:
    def __init__(self, env_name: str):
        self.spec = get_env_spec(env_name)
        self.env = make_gym_env(self.spec, render_mode=sim_planning_render_mode(self.spec))
        self.resets = 0
        self.steps = 0
        self._row: Dict[str, Any] = {}

    def _to_rows(self, state: Mapping[str, float]):
        if self.spec.name == "tworoom":
            obs = np.asarray(
                [state["agent.x"], state["agent.y"], state["target.x"], state["target.y"]],
                dtype=np.float64,
            )
            return {"observation": obs}, {"observation": obs}
        raise NotImplementedError(f"BlackBoxEnv adapter for {self.spec.name}")

    def _from_row(self, row: Mapping[str, Any]) -> Dict[str, float]:
        if self.spec.name == "tworoom":
            o = np.asarray(row["observation"], dtype=np.float64)
            return {"agent.x": float(o[0]), "agent.y": float(o[1]), "target.x": float(o[2]), "target.y": float(o[3])}
        raise NotImplementedError(self.spec.name)

    def reset(self, state: Mapping[str, float]) -> Dict[str, float]:
        start, goal = self._to_rows(state)
        obs = reset_env_from_rows(self.env, self.spec, start, goal)
        self.resets += 1
        self._row = sim_row_from_obs(self.spec.name, obs, template_row=start)
        return self._from_row(self._row)

    def step(self, action: Sequence[float]) -> Dict[str, float]:
        a = np.asarray(action, dtype=np.float64)
        info: Mapping[str, Any] = {}
        obs = None
        for _ in range(max(1, self.spec.frameskip)):
            obs, info = sim_step_with_info(self.env, self.spec, a)
            self.steps += 1
        self._row = sim_row_from_obs(self.spec.name, obs, template_row=self._row, info=info)
        return self._from_row(self._row)

    def rollout(self, state: Mapping[str, float], actions: Sequence[Sequence[float]]) -> List[Dict[str, float]]:
        out = [self.reset(state)]
        for a in actions:
            out.append(self.step(a))
        return out

    def close(self) -> None:
        try:
            self.env.close()
        except Exception:  # noqa: BLE001
            pass
