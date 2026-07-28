"""GIF-MCTS synthesis (Dainese et al., arXiv:2405.15383)."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from patchworld.worldmodel_data import Trajectory
from patchworld.worldmodel_inducer import Inducer

from .common import generate_loadable_code, is_loadable_worldmodel, sanitize_generated_code
from .metrics import TransitionEval, evaluate_transition_accuracy, reward_from_eval
from .prompts import GIF_FIX_PROMPT, GIF_GENERATE_PROMPT, GIF_IMPROVE_PROMPT, PATCHWORLD_API_REQUIREMENTS, WORLDCODER_RULES


ACTIONS = ("generate", "improve", "fix")


@dataclass
class GIFNode:
    node_id: int
    code: Optional[str]
    parent: Optional["GIFNode"] = None
    action: Optional[str] = None
    visits: int = 0
    value_sum: float = 0.0
    eval_result: Optional[TransitionEval] = None
    children: Dict[Tuple[str, int], "GIFNode"] = field(default_factory=dict)
    pending_actions: List[str] = field(default_factory=list)
    fix_attempts: int = 0
    buggy_temp_value: Optional[float] = None

    @property
    def value(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits

    @property
    def load_error(self) -> bool:
        return bool(self.eval_result and self.eval_result.load_error)

    def uct_score(self, child: "GIFNode", parent_visits: int, action_count: int, c: float = 0.1) -> float:
        if child.visits == 0:
            return child.selection_prior()
        exploit = child.value
        explore = c * math.sqrt(math.log(max(parent_visits, 1) + 1) / (action_count + 1e-6))
        return exploit + explore

    def selection_prior(self) -> float:
        if self.eval_result is None:
            return 0.55
        if self.load_error:
            if self.buggy_temp_value is not None:
                return self.buggy_temp_value
            return 0.99
        return reward_from_eval(self.eval_result)


class GIFMCTSSearch:
    def __init__(
        self,
        *,
        inducer: Inducer,
        env_name: str,
        class_name: str,
        is_spatial: bool,
        rules: str,
        api_requirements: str,
        transition_text: str,
        validation_trajectories: Sequence[Trajectory],
        rng: random.Random,
        max_fix_attempts: int = 3,
    ):
        self.inducer = inducer
        self.env_name = env_name
        self.class_name = class_name
        self.is_spatial = is_spatial
        self.rules = rules
        self.api_requirements = api_requirements
        self.transition_text = transition_text
        self.validation_trajectories = validation_trajectories
        self.rng = rng
        self.max_fix_attempts = max_fix_attempts
        self._next_id = 0
        self.root = self._new_node(code=None, action=None, parent=None)
        self.root.pending_actions = ["generate"]
        self.action_global_value: Dict[str, Tuple[float, int]] = {a: (0.55, 1) for a in ACTIONS}
        self.action_local_counts: Dict[str, int] = {a: 0 for a in ACTIONS}
        self.records: List[Dict[str, Any]] = []

    def _new_node(self, *, code: Optional[str], action: Optional[str], parent: Optional[GIFNode]) -> GIFNode:
        self._next_id += 1
        return GIFNode(node_id=self._next_id, code=code, action=action, parent=parent)

    def run(self, iterations: int) -> str:
        best_code = ""
        best_reward = -1.0
        best_loadable_code = ""
        best_loadable_reward = -1.0
        for step in range(1, iterations + 1):
            path = self._select(self.root)
            leaf = path[-1]
            action = self._choose_action(leaf)
            try:
                child_code = sanitize_generated_code(self._expand(leaf, action))
            except Exception as exc:
                self.records.append(
                    {
                        "step": step,
                        "action": action,
                        "reward": 0.0,
                        "accuracy": 0.0,
                        "load_error": True,
                        "llm_error": str(exc),
                    }
                )
                continue
            eval_result = evaluate_transition_accuracy(child_code, self.validation_trajectories)
            reward = reward_from_eval(eval_result)
            child = self._attach_child(leaf, action, child_code, eval_result)
            self._backprop(path + [child], reward, skip_buggy_temp=child.load_error)
            self._update_action_stats(action, reward)
            if reward > best_reward and child_code:
                best_reward = reward
                best_code = child_code
            if not eval_result.load_error and reward >= best_loadable_reward and child_code:
                best_loadable_reward = reward
                best_loadable_code = child_code
            self.records.append(
                {
                    "step": step,
                    "action": action,
                    "reward": reward,
                    "accuracy": eval_result.accuracy,
                    "load_error": eval_result.load_error,
                }
            )

        if best_loadable_code:
            return best_loadable_code

        generate_prompt = GIF_GENERATE_PROMPT.format(
            env_name=self.env_name,
            rules=self.rules,
            api_requirements=self.api_requirements,
            transition_text=self.transition_text,
            class_name=self.class_name,
        )
        return generate_loadable_code(
            lambda: sanitize_generated_code(self.inducer._extract_code(self.inducer.llm.generate(generate_prompt))),
            max_attempts=4,
        )

    def _select(self, node: GIFNode) -> List[GIFNode]:
        path = [node]
        current = node
        while current.children and not current.pending_actions:
            parent_visits = max(current.visits, 1)
            scored = []
            for (action, _), child in current.children.items():
                action_count = sum(1 for key in current.children if key[0] == action)
                scored.append((child.uct_score(child, parent_visits, action_count), child))
            if not scored:
                break
            current = max(scored, key=lambda item: item[0])[1]
            path.append(current)
        return path

    def _choose_action(self, node: GIFNode) -> str:
        if node.pending_actions:
            return node.pending_actions.pop(self.rng.randrange(len(node.pending_actions)))
        if node.load_error and node.fix_attempts < self.max_fix_attempts:
            return "fix"
        if node.code is None:
            return "generate"
        choices = ["generate", "improve"]
        if node.load_error or (node.eval_result and node.eval_result.errors):
            choices.append("fix")
        return self.rng.choice(choices)

    def _expand(self, node: GIFNode, action: str) -> str:
        if action == "generate" or node.code is None:
            prompt = GIF_GENERATE_PROMPT.format(
                env_name=self.env_name,
                rules=self.rules,
                api_requirements=self.api_requirements,
                transition_text=self.transition_text,
                class_name=self.class_name,
            )
            return self.inducer._extract_code(self.inducer.llm.generate(prompt))
        if action == "improve":
            improve_text = self.inducer._format_transitions(
                self.inducer._select_induction_transitions(list(self.validation_trajectories))
            )
            prompt = GIF_IMPROVE_PROMPT.format(
                env_name=self.env_name,
                code=node.code or "",
                transition_text=improve_text,
            )
            return self.inducer._extract_code(self.inducer.llm.generate(prompt))
        errors = node.eval_result.errors if node.eval_result else []
        prompt = GIF_FIX_PROMPT.format(
            env_name=self.env_name,
            api_requirements=self.api_requirements,
            code=node.code or "",
            errors=self.inducer._format_errors(errors[:10]),
            class_name=self.class_name,
        )
        return self.inducer._extract_code(self.inducer.llm.generate(prompt))

    def _attach_child(
        self,
        parent: GIFNode,
        action: str,
        code: str,
        eval_result: TransitionEval,
    ) -> GIFNode:
        key = (action, self.action_local_counts[action])
        self.action_local_counts[action] += 1
        child = self._new_node(code=code, action=action, parent=parent)
        child.eval_result = eval_result
        parent.children[key] = child
        if action == "fix" and eval_result.load_error:
            parent.fix_attempts += 1
            parent.buggy_temp_value = max(0.0, 0.99 - 0.33 * parent.fix_attempts)
        elif action in ("generate", "improve") and not eval_result.load_error:
            if action not in parent.pending_actions:
                parent.pending_actions.append(action)
        return child

    def _backprop(self, path: Sequence[GIFNode], reward: float, *, skip_buggy_temp: bool) -> None:
        for node in reversed(path):
            if skip_buggy_temp and node.load_error and node.buggy_temp_value is not None:
                continue
            node.visits += 1
            node.value_sum += reward

    def _update_action_stats(self, action: str, reward: float) -> None:
        avg, count = self.action_global_value[action]
        count += 1
        avg = ((avg * (count - 1)) + reward) / count
        self.action_global_value[action] = (avg, count)


def gif_mcts_induce(
    inducer: Inducer,
    env_name: str,
    train_trajectories: Sequence[Trajectory],
    validation_trajectories: Sequence[Trajectory],
    *,
    iterations: int = 50,
    seed: int = 42,
) -> str:
    """GIF-MCTS port using transition-accuracy rewards and UCT action selection."""
    env = (env_name or "").strip().lower()
    class_name = f"{env.capitalize()}WorldModel"
    from patchworld.worldmodel_inducer import SPATIAL_ENVS

    is_spatial = env in SPATIAL_ENVS
    rng = random.Random(seed)
    transitions = inducer._select_induction_transitions(list(train_trajectories))
    transition_text = inducer._format_transitions(transitions)
    rules = WORLDCODER_RULES
    api_requirements = PATCHWORLD_API_REQUIREMENTS.format(class_name=class_name)

    search = GIFMCTSSearch(
        inducer=inducer,
        env_name=env_name,
        class_name=class_name,
        is_spatial=is_spatial,
        rules=rules,
        api_requirements=api_requirements,
        transition_text=transition_text,
        validation_trajectories=validation_trajectories,
        rng=rng,
    )
    best_code = search.run(iterations)
    inducer.last_induce_stats = {
        "method": "gif_mcts",
        "iterations": iterations,
        "records": search.records,
    }
    ok, err = is_loadable_worldmodel(best_code)
    if not ok:
        raise RuntimeError(f"GIF-MCTS failed to produce a loadable model: {err}")
    return best_code
