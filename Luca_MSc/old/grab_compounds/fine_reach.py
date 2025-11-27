from Luca_MSc.old.elementary_behavior import ElementaryBehavior


class FineReachBehavior(ElementaryBehavior):
    """Behavior to move the gripper the final small distance to optimal grabbing position."""

    def __init__(self, field_params=None):
        super().__init__(field_params)
        self.cos_input = 0.0

    def execute(self, gripper_interactor, target_location, fine_threshold=0.05, external_input=0.0):
        """Execute fine reaching behavior."""
        # If no target, stay quiescent
        if target_location is None:
            state = self.forward(0.0, 0.0)
            state['distance'] = None
            state['motor_commands'] = None
            return state

        # Sync pose to ensure accurate distance calculation
        gripper_interactor.get_position()

        gripper_arrived = bool(gripper_interactor.gripper_is_at(target_location, thresh=fine_threshold))

        # Prepare inputs for nodes
        cos_input = 5.0 if gripper_arrived else 0.0

        # Process behavior control
        state = self.forward(external_input, cos_input)

        # Generate motor commands if active
        motor_cmd = None
        if float(state.get('intention_activity', 0.0)) > 0.0 and not gripper_arrived:
            gripper_interactor.max_speed = 0.05  # Slow down for fine movement
            motor_cmd = gripper_interactor.gripper_move_towards(target_location)

        # Diagnostics/echo
        state['distance'] = gripper_interactor.calculate_distance(target_location)
        state['motor_commands'] = motor_cmd

        return state

    def reset(self):
        """Reset fields to initial state."""
        super().reset()
        self.cos_input = 0.0
