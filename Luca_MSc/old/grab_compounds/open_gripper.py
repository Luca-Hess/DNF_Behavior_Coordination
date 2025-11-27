from Luca_MSc.old.elementary_behavior import ElementaryBehavior


class OpenGripperBehavior(ElementaryBehavior):
    """Behavior to open the gripper."""

    def __init__(self, field_params=None):
        super().__init__(field_params)
        self._last_active = False
        self._check_open = False

    def execute(self, gripper_interactor, external_input=0.0):
        """Check if the gripper is holding an object."""
        # Process behavior to determine activity
        if self._last_active:
            self._check_open = True

        if self._check_open:
            is_open = gripper_interactor.is_open
        else:
            is_open = False

        # Prepare inputs for nodes
        cos_input = 5.0 if is_open else 0.0

        # Process behavior control
        state = self.forward(external_input, cos_input)

        # Generate motor commands if active
        motor_cmd = None
        if float(state.get('intention_activity', 0.0)) > 0.0 and not is_open:
            motor_cmd = gripper_interactor.open_gripper()

        # Diagnostics/echo
        state['gripper_state'] = 'open' if is_open else 'closed'
        state['motor_commands'] = motor_cmd

        self._last_active = float(state.get('intention_activity', 0.0)) > 0.0

        return state


    def reset(self):
        super().reset()
        self._last_active = False
        self._check_open = False
