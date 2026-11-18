"""Black-box reset/step access expressed in scene-graph state keys.

Probes start from the exact dataset row a held-out transition was exported
from (``meta["raw_index"]``), so hidden simulator state such as joint
velocities is restored faithfully. Observed states are read back through the
same scene-graph builder that produced the traces. The adapter encodes no
dynamics.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from visualpatchworld.hdf5 import require_h5py, resolve_dataset_path
from visualpatchworld.lewm_env_specs import get_env_spec, make_gym_env, sim_planning_render_mode
from visualpatchworld.lewm_graph import (
    goal_row_from_start,
    graph_text_from_row,
    reset_env_from_rows,
    row_dict_from_h5,
    sim_row_from_obs,
    sim_step_with_info,
)

from .traces import parse_graph_text


class BlackBoxEnv:
    def __init__(self, env_name: str, *, frames_per_step: Optional[int] = None):
        self.spec = get_env_spec(env_name)
        self.env = make_gym_env(self.spec, render_mode=sim_planning_render_mode(self.spec))
        self.frames_per_step = frames_per_step or self.spec.frameskip
        self.h5 = require_h5py().File(resolve_dataset_path(self.spec.dataset), "r")
        self.ep_offset = np.asarray(self.h5["ep_offset"][:], dtype=np.int64)
        self.resets = 0
        self.steps = 0
        self._row: Dict[str, Any] = {}

    def _state(self, row: Mapping[str, Any]) -> Dict[str, float]:
        return parse_graph_text(graph_text_from_row(self.spec.name, row))["state"]

    def reset_to(self, meta: Mapping[str, Any]) -> Dict[str, float]:
        raw = int(meta["raw_index"])
        episode = int(meta.get("episode_index", np.searchsorted(self.ep_offset, raw, side="right") - 1))
        start = row_dict_from_h5(self.h5, raw, self.spec)
        start_step = raw - int(self.ep_offset[episode])
        try:
            goal = goal_row_from_start(self.h5, self.spec, episode, start_step, self.spec.goal_offset_steps)
        except Exception:  # noqa: BLE001
            goal = start
        reset_env_from_rows(self.env, self.spec, start, goal)
        self.resets += 1
        self._row = dict(start)
        return self._state(self._row)

    def step(self, action: Sequence[float]) -> Dict[str, float]:
        a = np.asarray(action, dtype=np.float64)
        info: Mapping[str, Any] = {}
        obs = None
        for _ in range(max(1, self.frames_per_step)):
            obs, info = sim_step_with_info(self.env, self.spec, a)
            self.steps += 1
        self._row = sim_row_from_obs(self.spec.name, obs, template_row=self._row, info=info)
        return self._state(self._row)

    def rollout(self, meta: Mapping[str, Any], actions: Sequence[Sequence[float]]) -> List[Dict[str, float]]:
        out = [self.reset_to(meta)]
        for a in actions:
            out.append(self.step(a))
        return out

    def close(self) -> None:
        try:
            self.env.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            self.h5.close()
        except Exception:  # noqa: BLE001
            pass
