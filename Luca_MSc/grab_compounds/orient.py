import torch
from Luca_MSc.elementary_behavior import ElementaryBehavior


class OrientBehavior(ElementaryBehavior):
    """Behavior to orient the gripper correctly for grabbing."""

    def __init__(self, field_params=None):
        super().__init__(field_params)


    def execute(self, gripper_interactor, target_orientation, threshold, external_input=0.0):
        """Execute orientation behavior toward a target location."""

        if target_orientation is None:
            state = self.forward(0.0, 0.0)
            return state

        # Arrival check based on data from interactor, with specified "arrival" threshold
        oriented = gripper_interactor.is_oriented(target_orientation, thresh=threshold)

        # Prepare inputs for nodes
        cos_input = 5.0 if oriented else 0.0

        # Process behavior control
        state = self.forward(external_input, cos_input)

        # Generate motor commands if active
        motor_cmd = None
        if state['active'] and not oriented:
            motor_cmd = gripper_interactor.gripper_rotate_towards(target_orientation)

        # Diagnostics/echo
        state['oriented'] = oriented
        state['motor_commands'] = (
            motor_cmd.tolist() if hasattr(motor_cmd, "tolist") else motor_cmd
        )

        return state


    def reset(self):
        """Reset fields to initial state."""
        super().reset()


