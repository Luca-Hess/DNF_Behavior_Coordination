import torch
from Luca_MSc.elementary_behavior import ElementaryBehavior


class FineReachBehavior(ElementaryBehavior):
    """Behavior to move the gripper the final small distance to optimal grabbing position."""

    def __init__(self, field_params=None):
        super().__init__(field_params)

    def execute(self, gripper_interactor, target_location, fine_threshold=0.05, external_input=0.0):
        """Execute fine reaching behavior."""

        # Compute inputs
        cos_input = 0.0

        # Calculate distance to target
        if target_location is not None:
            distance = gripper_interactor.calculate_distance(target_location)

            # Check if we've reached the fine threshold - now truly at the object
            if distance <= fine_threshold:
                cos_input = 6.0  # Strong input to CoS

        # Evolve fields with inputs
        state = self.forward(external_input, cos_input)

        # Only move if intention activity is above threshold
        if state['active'] and target_location is not None:
            # Move with reduced speed for precision
            gripper_interactor.max_speed = 0.05  # Slow down for fine movements
            gripper_interactor.gripper_move_towards(target_location)

        # Consider completed when CoS is active
        state['distance'] = distance

        # Return state
        return state

    def reset(self):
        """Reset fields to initial state."""
        super().reset()