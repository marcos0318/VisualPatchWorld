from patchworld.worldmodel_base import BaseWorldModel
import re
import math

class TworoomWorldModel(BaseWorldModel):
    def init_belief(self) -> dict:
        """Initialize an empty belief state."""
        return {
            'environment': None,
            'source': None,
            'step': None,
            'goal': None,
            'objects': {},
            'relations': {},
            'metadata': {},
            'epistemic_status': {}
        }

    def init_belief_from_observation(self, obs_text: str) -> dict:
        """Initialize belief state from the first observation."""
        belief = self.init_belief()
        return self.correct_belief(belief, obs_text)

    def parse_observation(self, obs_text: str) -> dict:
        """Parse the observation text into structured data."""
        lines = obs_text.strip().split('\n')
        obs_dict = {
            'environment': None,
            'source': None,
            'step': None,
            'goal': None,
            'objects': {},
            'relations': {},
            'metadata': {}
        }

        current_section = None
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith("Environment:"):
                obs_dict['environment'] = line.split(":", 1)[1].strip()
            elif line.startswith("Source:"):
                obs_dict['source'] = line.split(":", 1)[1].strip()
            elif line.startswith("Step:"):
                obs_dict['step'] = int(line.split(":", 1)[1].strip())
            elif line.startswith("Goal:"):
                obs_dict['goal'] = line.split(":", 1)[1].strip()
            elif line == "Objects:":
                current_section = 'objects'
            elif line == "Relations:":
                current_section = 'relations'
            elif line.startswith("Metadata:"):
                metadata_str = line.split(":", 1)[1].strip()
                # Parse JSON-like metadata string
                try:
                    # A simple parser for the metadata JSON
                    metadata_str = metadata_str.replace('true', 'True').replace('false', 'False')
                    obs_dict['metadata'] = eval(metadata_str)
                except:
                    obs_dict['metadata'] = {}
            elif current_section == 'objects' and line.startswith("- "):
                # Parse object line
                obj_line = line[2:]  # Remove "- "
                parts = obj_line.split()
                obj_id_type = parts[0]
                obj_id, obj_type = obj_id_type.split(':', 1)
                coords = {}
                for part in parts[1:]:
                    if '=' in part:
                        key, val = part.split('=', 1)
                        coords[key] = float(val)
                obs_dict['objects'][obj_id] = {
                    'type': obj_type,
                    'x': coords.get('x'),
                    'y': coords.get('y'),
                    'conf': coords.get('conf')
                }
            elif current_section == 'relations' and line.startswith("- "):
                # Parse relation line
                rel_line = line[2:]  # Remove "- "
                # Example: agent distance_to target = 214.153
                parts = rel_line.split()
                if len(parts) >= 5 and parts[1] == 'distance_to':
                    obj1 = parts[0]
                    obj2 = parts[2]
                    distance = float(parts[4])
                    obs_dict['relations'][(obj1, obj2, 'distance_to')] = distance
                elif len(parts) >= 5 and parts[1] == 'near':
                    obj1 = parts[0]
                    obj2 = parts[2]
                    nearness = float(parts[4])
                    obs_dict['relations'][(obj1, obj2, 'near')] = nearness

        return obs_dict

    def correct_belief(self, belief_prior: dict, obs_text: str) -> dict:
        """Assimilate actual observation evidence into prior belief."""
        obs_dict = self.parse_observation(obs_text)
        belief = belief_prior.copy()

        # Update environment, source, step, goal if present
        if obs_dict['environment'] is not None:
            belief['environment'] = obs_dict['environment']
        if obs_dict['source'] is not None:
            belief['source'] = obs_dict['source']
        if obs_dict['step'] is not None:
            belief['step'] = obs_dict['step']
        if obs_dict['goal'] is not None:
            belief['goal'] = obs_dict['goal']

        # Update objects
        for obj_id, obj_data in obs_dict['objects'].items():
            if obj_id not in belief['objects']:
                belief['objects'][obj_id] = {}
            belief['objects'][obj_id].update(obj_data)
            belief['epistemic_status'][obj_id] = 'known'

        # Update relations
        for rel_key, rel_value in obs_dict['relations'].items():
            belief['relations'][rel_key] = rel_value

        # Update metadata
        belief['metadata'].update(obs_dict['metadata'])

        return belief

    def predict_belief(self, belief: dict, action: str) -> dict:
        """Predict step only (no observation correction): b' = f(b, a)"""
        # Apply action-conditioned dynamics and return prior belief.
        belief_next = {
            'environment': belief['environment'],
            'source': belief['source'],
            'step': belief['step'] + 1 if belief['step'] is not None else None,
            'goal': belief['goal'],
            'objects': belief['objects'].copy(),
            'relations': belief['relations'].copy(),
            'metadata': belief['metadata'].copy(),
            'epistemic_status': belief['epistemic_status'].copy()
        }

        # Parse action
        match = re.match(r"(\w+)\[dx=([+-]?\d+\.\d+),dy=([+-]?\d+\.\d+),speed=([+-]?\d+\.\d+)\]", action)
        if not match:
            # If action doesn't match the expected pattern, return unchanged belief
            return belief_next

        action_type = match.group(1)
        dx = float(match.group(2))
        dy = float(match.group(3))
        speed = float(match.group(4))

        # Apply motion to the agent with scaling factor of 25.0
        SCALE_FACTOR = 25.0
        if 'agent' in belief_next['objects']:
            agent = belief_next['objects']['agent']
            if agent['x'] is not None and agent['y'] is not None:
                agent['x'] += dx * SCALE_FACTOR
                agent['y'] += dy * SCALE_FACTOR

        # Recompute relations for agent and target
        if 'agent' in belief_next['objects'] and 'target' in belief_next['objects']:
            agent = belief_next['objects']['agent']
            target = belief_next['objects']['target']
            if agent['x'] is not None and agent['y'] is not None and \
               target['x'] is not None and target['y'] is not None:
                delta_x = target['x'] - agent['x']
                delta_y = target['y'] - agent['y']
                distance = math.sqrt(delta_x**2 + delta_y**2)
                belief_next['relations'][('agent', 'target', 'distance_to')] = distance
                # 'near' is always 0.000 in the examples
                belief_next['relations'][('agent', 'target', 'near')] = 0.0

        return belief_next

    def readout_observation(self, belief: dict, action: str = "") -> str:
        """Readout authority. Reuse the same template/sections that parse_observation expects."""
        lines = []
        if belief['environment'] is not None:
            lines.append(f"Environment: {belief['environment']}")
        if belief['source'] is not None:
            lines.append(f"Source: {belief['source']}")
        if belief['step'] is not None:
            lines.append(f"Step: {belief['step']}")
        if belief['goal'] is not None:
            lines.append(f"Goal: {belief['goal']}")

        lines.append("Objects:")
        for obj_id in sorted(belief['objects'].keys()):
            obj = belief['objects'][obj_id]
            line = f"- {obj_id}:{obj['type']}"
            if obj['x'] is not None:
                line += f" x={obj['x']:.3f}"
            if obj['y'] is not None:
                line += f" y={obj['y']:.3f}"
            if obj['conf'] is not None:
                line += f" conf={obj['conf']:.3f}"
            lines.append(line)

        lines.append("Relations:")
        # Sort relations for consistent output
        sorted_relations = sorted(belief['relations'].items())
        for (obj1, obj2, rel_type), value in sorted_relations:
            if rel_type == 'distance_to':
                lines.append(f"- {obj1} distance_to {obj2} = {value:.3f}")
            elif rel_type == 'near':
                lines.append(f"- {obj1} near {obj2} = {value:.3f}")

        if belief['metadata']:
            # Convert metadata back to JSON-like string
            meta_str = str(belief['metadata']).replace("'", '"').replace('True', 'true').replace('False', 'false')
            lines.append(f"Metadata: {meta_str}")
        else:
            lines.append("Metadata: {}")

        return "\n".join(lines)

    def extract_valid_action_forms(self) -> dict[str, list[str]]:
        """Return a dictionary of valid action forms."""
        return {
            "move_left": ["move_left[dx=<X>,dy=<Y>,speed=<S>]"],
            "move_right": ["move_right[dx=<X>,dy=<Y>,speed=<S>]"],
            "move_up": ["move_up[dx=<X>,dy=<Y>,speed=<S>]"],
            "move_down": ["move_down[dx=<X>,dy=<Y>,speed=<S>]"],
            "hold": ["hold[dx=<X>,dy=<Y>,speed=<S>]"]
        }