from patchworld.worldmodel_base import BaseWorldModel
import re
import math


class CubeWorldModel(BaseWorldModel):
    def __init__(self):
        super().__init__()
        self.GOAL_THRESHOLD = 0.05  # Threshold for considering block at goal

    def parse_observation(self, obs_text: str) -> dict:
        """Parse the textual scene graph into a structured belief state."""
        lines = obs_text.strip().split('\n')
        belief = {
            "environment": "",
            "source": "",
            "step": 0,
            "goal": "",
            "objects": {},
            "relations": {},
            "metadata": {}
        }

        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith("Environment:"):
                belief["environment"] = line.split(":", 1)[1].strip()
            elif line.startswith("Source:"):
                belief["source"] = line.split(":", 1)[1].strip()
            elif line.startswith("Step:"):
                belief["step"] = int(line.split(":", 1)[1].strip())
            elif line.startswith("Goal:"):
                belief["goal"] = line.split(":", 1)[1].strip()
            elif line.startswith("Objects:"):
                i += 1
                while i < len(lines) and not lines[i].startswith("Relations:"):
                    obj_line = lines[i].strip()
                    if obj_line.startswith('-'):
                        # Parse object line like "- block:cube x=0.425 y=0.246 z=0.020 conf=1.000"
                        parts = obj_line[2:].split()
                        obj_id_type = parts[0]  # e.g., "block:cube"
                        obj_id, obj_type = obj_id_type.split(':')
                        obj_data = {"type": obj_type}
                        for part in parts[1:]:
                            if '=' in part:
                                k, v = part.split('=')
                                if k in ['x', 'y', 'z']:
                                    obj_data[k] = float(v)
                                elif k == 'conf':
                                    obj_data[k] = float(v)
                        belief["objects"][obj_id] = obj_data
                    i += 1
                continue  # Skip increment at end since we already incremented
            elif line.startswith("Relations:"):
                i += 1
                while i < len(lines) and not lines[i].startswith("Metadata:"):
                    rel_line = lines[i].strip()
                    if rel_line.startswith('-'):
                        # Parse relation line like "- block distance_to goal = 0.534"
                        # Or "- block near goal = 0.000"
                        parts = rel_line[2:].split()
                        if "distance_to" in parts:
                            idx = parts.index("distance_to")
                            obj1 = parts[0]
                            obj2 = parts[idx + 1]
                            value = float(parts[idx + 3])
                            belief["relations"][(obj1, "distance_to", obj2)] = value
                        elif "near" in parts:
                            idx = parts.index("near")
                            obj1 = parts[0]
                            obj2 = parts[idx + 1]
                            value = float(parts[idx + 3])
                            belief["relations"][(obj1, "near", obj2)] = value
                    i += 1
                continue
            elif line.startswith("Metadata:"):
                metadata_str = line.split(":", 1)[1].strip()
                # Simple parsing assuming JSON format
                import json
                belief["metadata"] = json.loads(metadata_str)
            i += 1
        return belief

    def init_belief(self):
        """Initialize an empty belief state."""
        return {
            "environment": "cube",
            "source": "oracle",
            "step": 0,
            "goal": "move block to goal pose",
            "objects": {},
            "relations": {},
            "metadata": {}
        }

    def init_belief_from_observation(self, obs_text: str):
        """Initialize belief state from the first observation."""
        return self.parse_observation(obs_text)

    def correct_belief(self, belief_prior, obs_text: str):
        """Update belief state with new observation."""
        # In fully observed setting, replace the belief entirely
        return self.parse_observation(obs_text)

    def predict_belief(self, belief, action: str):
        """Predict next belief state given action."""
        # Copy current belief
        new_belief = {k: v.copy() if isinstance(v, dict) else v for k, v in belief.items()}
        new_belief["objects"] = {k: v.copy() for k, v in belief["objects"].items()}
        new_belief["step"] = belief["step"] + 1

        # Parse action parameters
        match = re.search(r'act$$a0=([^,]+),a1=([^,]+),a2=([^,]+),a3=([^,]+),a4=([^,]+),speed=([^$$]+)$$', action)
        if not match:
            # If action doesn't match expected format, return unchanged belief
            return new_belief

        try:
            a0, a1, a2, a3, a4, speed = map(float, match.groups())
        except ValueError:
            return new_belief

        # Apply physics-inspired transition logic
        if "effector" in new_belief["objects"] and "block" in new_belief["objects"]:
            eff = new_belief["objects"]["effector"]
            block = new_belief["objects"]["block"]

            # Effector moves towards target based on action
            dx = a0 * speed * 0.1
            dy = a1 * speed * 0.1
            dz = a2 * speed * 0.1

            # Update effector position
            eff["x"] += dx
            eff["y"] += dy
            eff["z"] += dz

            # Clamp effector to reasonable bounds if needed (optional)
            # For now, we assume no clamping based on examples

            # If effector is close to block, move block with it
            dist = math.sqrt((eff["x"] - block["x"])**2 + (eff["y"] - block["y"])**2 + (eff["z"] - block["z"])**2)
            if dist < 0.05:  # Threshold for gripping
                # Move block with effector
                block["x"] += dx * 0.8  # Slightly less than effector to simulate grip force
                block["y"] += dy * 0.8
                block["z"] += dz * 0.8

        # Update relations based on new positions
        self._update_relations(new_belief)

        return new_belief

    def _update_relations(self, belief):
        """Update distance and near relations based on object positions."""
        belief["relations"] = {}
        objs = belief["objects"]
        if "block" in objs and "goal" in objs:
            b = objs["block"]
            g = objs["goal"]
            dist = math.sqrt((b["x"] - g["x"])**2 + (b["y"] - g["y"])**2 + (b["z"] - g["z"])**2)
            belief["relations"][("block", "distance_to", "goal")] = dist
            belief["relations"][("block", "near", "goal")] = 1.0 if dist < self.GOAL_THRESHOLD else 0.0
        if "effector" in objs and "block" in objs:
            e = objs["effector"]
            b = objs["block"]
            dist = math.sqrt((e["x"] - b["x"])**2 + (e["y"] - b["y"])**2 + (e["z"] - b["z"])**2)
            belief["relations"][("effector", "distance_to", "block")] = dist

    def readout_observation(self, belief, action: str = "") -> str:
        """Generate textual scene graph from belief state."""
        lines = []
        lines.append(f"Environment: {belief['environment']}")
        lines.append(f"Source: {belief['source']}")
        lines.append(f"Step: {belief['step']}")
        lines.append(f"Goal: {belief['goal']}")

        lines.append("Objects:")
        for obj_id, obj_data in belief["objects"].items():
            obj_type = obj_data["type"]
            x = obj_data["x"]
            y = obj_data["y"]
            z = obj_data["z"]
            conf = obj_data.get("conf", 1.0)
            lines.append(f"- {obj_id}:{obj_type} x={x:.3f} y={y:.3f} z={z:.3f} conf={conf:.3f}")

        lines.append("Relations:")
        for (obj1, rel, obj2), value in belief["relations"].items():
            lines.append(f"- {obj1} {rel} {obj2} = {value:.3f}")

        # Metadata placeholder
        lines.append('Metadata: {"episode_index":0,"next_raw_index":0,"raw_index":0}')

        return "\n".join(lines)

    def extract_valid_action_forms(self) -> dict[str, list[str]]:
        """Define valid action templates."""
        return {
            "act": ["act[a0=<float>,a1=<float>,a2=<float>,a3=<float>,a4=<float>,speed=<float>]"]
        }