
from .base_interactors import BaseInteractor

class StateInteractor(BaseInteractor):
    """Interactor for managing world state transitions and updates"""

    def __init__(self, perception_interactor, movement_interactor, gripper_interactor):
        super().__init__()
        self.perception = perception_interactor
        self.movement = movement_interactor
        self.gripper = gripper_interactor

        # Dynamic target state - populated based on behavior chain
        self.behavior_targets = {}  # {behavior_name: current_target_name}
        self.target_args = {}       # Store original args for reference
        self.instance_targets = {} # {behavior_name: target_name} for instance-specific targets
        self.global_state = {}      # Global state info if needed

    def initialize_from_behavior_chain(self, behavior_chain, args):
        """Initialize targets dynamically based on the behavior chain and args"""
        self.target_args = args.copy()

        # Analyze behavior chain to determine target requirements
        for level in behavior_chain:
            behavior_name = level['name']
            interactor_type = level.get('interactor_type')

            # Set initial target based on behavior type and args
            if interactor_type == 'perception':
                # Perception behaviors (like 'find') use the primary target
                self.behavior_targets[behavior_name] = args.get('target_object')

            elif interactor_type in ['movement', 'gripper']:
                # Movement/gripper behaviors initially use the primary target
                self.behavior_targets[behavior_name] = args.get('target_object')

        return True, self.behavior_targets.copy()

    def update_behavior_target(self, behavior_name, new_target_name, requesting_behavior=None):
        """Unified method to update target for a specific behavior"""
        old_target = self.behavior_targets.get(behavior_name)
        self.behavior_targets[behavior_name] = new_target_name

        state_data = {
            'behavior_name': behavior_name,
            'old_target': old_target,
            'new_target': new_target_name
        }

        success = True
        self._update_and_publish_state(state_data, success, requesting_behavior)

        return success, new_target_name

    def update_multiple_behavior_targets(self, target_updates):
        """Update targets for multiple behaviors at once"""
        for behavior_name, new_target in target_updates.items():
            if behavior_name in self.behavior_targets:
                old_target = self.behavior_targets[behavior_name]
                self.behavior_targets[behavior_name] = new_target

        return True, self.behavior_targets.copy()

    def get_behavior_target_name(self, behavior_name):
        """Get the current target name for a behavior"""
        target_name = self.behavior_targets.get(behavior_name)
        return target_name

    def get_behavior_target_location(self, behavior_name):
        """Get the current target location for a behavior"""
        target_name = self.behavior_targets.get(behavior_name)

        for behavior, state in self.shared_states.items():
            if 'target_name' in state and state['target_name'] == target_name:
                if 'target_location' in state and state['target_location'] is not None:
                    return state['target_location']

        if target_name and target_name in self.perception.objects:
            location = self.perception.objects[target_name]['location']
            return location

        return None

    def get_behavior_target_info(self, behavior_name):
        """Get full target info (location, angle) for a behavior"""
        target_name = self.behavior_targets.get(behavior_name)

        for behavior, state in self.shared_states.items():
            if 'target_name' in state and state['target_name'] == target_name:
                location = state.get('target_location')
                angle = state.get('target_angle')
                return target_name, location, angle

        if target_name and target_name in self.perception.objects:
            info = self.perception.objects[target_name]
            location = info.get('location')
            angle = info.get('angle')
            return target_name, location, angle

        return None, None, None

    def get_instance_target(self, instance_id):
        """Get the instance-specific target for a behavior"""
        target_name = self.instance_targets.get(instance_id)
        return target_name

    def set_instance_target(self, instance_id, target_info):
        """Set the instance-specific target for a behavior"""
        self.instance_targets[instance_id] = target_info
        return True, target_info
