import torch
from Luca_MSc.elementary_behavior import ElementaryBehavior


class FineReachBehavior(ElementaryBehavior):
    """Behavior to move the gripper the final small distance to optimal grabbing position."""

    def __init__(self, field_params=None):
        super().__init__(field_params)
        self.cos_input = 0.0

    def execute(self, gripper_interactor, target_location, fine_threshold=0.05, external_input=0.0):
        """Execute fine reaching behavior."""
        # Process behavior to determine activity
        state = self.forward(external_input, self.cos_input)

        active = float(state.get('intention_activity', 0.0)) > 0.0

        if active and target_location is not None:
            distance = gripper_interactor.calculate_distance(target_location)

            # Check if we've reached the fine threshold - now truly at the object
            if distance <= fine_threshold:
                self.cos_input = 5.0  # Strong input to CoS

            # If not there yet, move towards target location
            if distance > fine_threshold:
                gripper_interactor.max_speed = 0.05  # Slow down for fine movements
                motor_cmd = gripper_interactor.gripper_move_towards(target_location)

                # Update state
                state['distance'] = distance
                state['motor_commands'] = motor_cmd

            else:
                state['distance'] = distance
                state['motor_commands'] = None

        else:
            state['distance'] = None
            state['motor_commands'] = None

        return state

    def reset(self):
        """Reset fields to initial state."""
        super().reset()
        self.cos_input = 0.0
