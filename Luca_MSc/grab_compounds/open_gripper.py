import torch
from Luca_MSc.elementary_behavior import ElementaryBehavior


class OpenGripperBehavior(ElementaryBehavior):
    """Behavior to open the gripper."""

    def __init__(self, field_params=None):
        super().__init__(field_params)
        self.cos_input = 0.0

    def execute(self, gripper_interactor, external_input=0.0):
        """Check if the gripper is holding an object."""
        # Process behavior to determine activity
        state = self.forward(external_input, self.cos_input)

        active = float(state.get('intention_activity', 0.0)) > 0.0

        # Try to check the range using the interactor only if the behavior is active
        if active:
            # check if gripper is open
            is_open = gripper_interactor.is_open

            # Use range status for CoS update
            self.cos_input = 5.0 if is_open else 0.0

            # Generate motor commands if active
            motor_cmd = None
            if float(state.get('intention_activity', 0.0)) > 0.0 and not is_open:
                motor_cmd = gripper_interactor.open_gripper()

            # Update status
            state['gripper_state'] = 'open' if is_open else 'closed'
            state['motor_commands'] = (
                motor_cmd.tolist() if hasattr(motor_cmd, "tolist") else motor_cmd
            )

        else:
            # Store status
            state['gripper_state'] = 'closed'
            state['motor_commands'] = None

        return state



    def reset(self):
        super().reset()
        self.cos_input = 0.0
