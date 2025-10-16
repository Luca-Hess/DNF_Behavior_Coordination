import torch
from Luca_MSc.elementary_behavior import ElementaryBehavior


class OpenGripperBehavior(ElementaryBehavior):
    """Behavior to open the gripper."""

    def __init__(self, field_params=None):
        super().__init__(field_params)
        self._last_active = False
        self._checked_open = False

    def execute(self, gripper_interactor, external_input=0.0):
        """Check if the gripper is holding an object."""

        # Try to check the range using the interactor only if the behavior is active
        if self._last_active:
            self._checked_open = True

        if self._checked_open:
            is_open = gripper_interactor.is_open
        else:
            is_open = False

        # Use range status for CoS
        cos_input = 5.0 if is_open else 0.0

        # Process behavior control
        state = self.forward(external_input, cos_input)

        # Generate motor commands if active
        motor_cmd = None
        if state['active'] and not is_open:
            motor_cmd = gripper_interactor.open_gripper()


        # Store status
        state['gripper_state'] = 'open' if is_open else 'closed'
        state['motor_commands'] = (
            motor_cmd.tolist() if hasattr(motor_cmd, "tolist") else motor_cmd
        )

        self._last_active = state['active']

        return state

    def reset(self):
        super().reset()
        self._last_active = False
        self._checked_open = False