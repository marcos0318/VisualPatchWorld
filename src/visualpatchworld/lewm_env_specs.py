"""LeWM four-environment registry for VisualPatchWorld cross-env experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class LeWMEnvSpec:
    name: str
    dataset: str
    gym_id: str
    gym_kwargs: Mapping[str, Any] = field(default_factory=dict)
    action_dim: int = 2
    frameskip: int = 5
    goal_offset_steps: int = 25
    eval_budget: int = 50
    planning_horizon: int = 5
    planning_receding: int = 5
    can_plan_in_sim: bool = True
    planning_note: str = ""


def get_env_spec(name: str) -> LeWMEnvSpec:
    key = name.strip().lower().replace("-", "").replace("_", "")
    specs = {spec.name: spec for spec in ALL_ENV_SPECS}
    aliases = {
        "pusht": "pusht",
        "tworoom": "tworoom",
        "two": "tworoom",
        "cube": "cube",
        "ogb": "cube",
        "reacher": "reacher",
        "dmc": "reacher",
    }
    resolved = aliases.get(key, key)
    if resolved not in specs:
        raise KeyError(f"Unknown env {name!r}. Choose from {list(specs)}")
    return specs[resolved]


ALL_ENV_SPECS: Tuple[LeWMEnvSpec, ...] = (
    LeWMEnvSpec(
        name="pusht",
        dataset="pusht_expert_train.h5",
        gym_id="swm/PushT-v1",
        action_dim=2,
        frameskip=1,
        planning_horizon=5,
        planning_receding=1,
    ),
    LeWMEnvSpec(
        name="tworoom",
        dataset="tworoom.h5",
        gym_id="swm/TwoRoom-v1",
        gym_kwargs={"render_target": True},
        action_dim=2,
        frameskip=5,
        planning_horizon=5,
        planning_receding=5,
    ),
    LeWMEnvSpec(
        name="cube",
        dataset="ogbench/cube_single_expert.h5",
        gym_id="swm/OGBCube-v0",
        gym_kwargs={
            "env_type": "single",
            "ob_type": "states",
            "multiview": False,
            "width": 224,
            "height": 224,
            "visualize_info": False,
            "terminate_at_goal": True,
        },
        action_dim=5,
        frameskip=5,
        planning_horizon=5,
        planning_receding=5,
    ),
    LeWMEnvSpec(
        name="reacher",
        dataset="reacher.h5",
        gym_id="swm/ReacherDMControl-v0",
        gym_kwargs={"task": "qpos_match"},
        action_dim=2,
        frameskip=5,
        planning_horizon=5,
        planning_receding=5,
        can_plan_in_sim=True,
        planning_note="Requires mujoco<=3.8.x (dm_control flex_bandwidth API).",
    ),
)


def sim_planning_render_mode(spec: LeWMEnvSpec) -> Optional[str]:
    """Return render_mode for true-simulator planning rollouts.

    State-only envs (e.g. Cube with ``ob_type=states``) do not need ``rgb_array``;
    skipping render avoids MuJoCo/pygame buffer accumulation and OOM on long CEM runs.
    """
    if spec.gym_kwargs.get("ob_type") == "states":
        return None
    return "rgb_array"


def make_gym_env(spec: LeWMEnvSpec, *, render_mode: Optional[str] = "rgb_array"):
    import gymnasium as gym
    import stable_worldmodel.envs  # noqa: F401

    kwargs = dict(spec.gym_kwargs)
    if render_mode is None:
        return gym.make(spec.gym_id, **kwargs)
    return gym.make(spec.gym_id, render_mode=render_mode, **kwargs)
