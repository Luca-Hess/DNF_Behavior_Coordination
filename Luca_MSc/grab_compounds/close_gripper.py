import torch
from Luca_MSc.elementary_behavior import ElementaryBehavior


class CloseGripperBehavior(ElementaryBehavior):
    """Behavior to close the gripper to grab an object."""

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
            is_closed = not(gripper_interactor.is_open)
        else:
            is_closed = False

        # Use range status for CoS
        cos_input = 5.0 if is_closed else 0.0

        # Process behavior control
        state = self.forward(external_input, cos_input)

        # Generate motor commands if active
        motor_cmd = None
        if float(state.get('intention_activity', 0.0)) > 0.0 and not is_closed:
            motor_cmd = gripper_interactor.close_gripper()

        # Store status
        state['gripper_state'] = 'closed' if is_closed else 'open'
        state['motor_commands'] = (
            motor_cmd.tolist() if hasattr(motor_cmd, "tolist") else motor_cmd
        )

        self._last_active = float(state.get('intention_activity', 0.0)) > 0.0

        return state

    def reset(self):
        super().reset()
        self._last_active = False
        self._checked_open = False

