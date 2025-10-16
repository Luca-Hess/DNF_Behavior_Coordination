import torch
from Luca_MSc.elementary_behavior import ElementaryBehavior


class OrientBehavior(ElementaryBehavior):
    """Behavior to orient the gripper correctly for grabbing."""

    def __init__(self, field_params=None):
        super().__init__(field_params)
        self.cos_input = 0.0

    def execute(self, interactors, target_name, threshold, external_input=0.0):
        """Execute orientation behavior toward a target location."""
        # Process behavior to determine activity
        state = self.forward(external_input, self.cos_input)

        active = float(state.get('intention_activity', 0.0)) > 0.0

        # All actions only carried out if behavior is sufficiently excited
        if active:
            target_angle = interactors.perception.find_object_angle(target_name)

            if target_angle is not None:
                # Arrival check based on data from interactor, with specified threshold
                oriented = interactors.gripper.is_oriented(target_angle, thresh=threshold)

                # Update input based on orientation status
                self.cos_input = 5.0 if oriented else 0.0

                # Generate motor commands if active and not yet oriented
                motor_cmd = None
                if not oriented:
                    motor_cmd = interactors.gripper.gripper_rotate_towards(target_angle)

                # Updating state
                state['oriented'] = oriented
                state['target_angle'] = target_angle
                state['motor_commands'] = (
                    motor_cmd.tolist() if hasattr(motor_cmd, "tolist") else motor_cmd
                )
            else:
                state['oriented'] = False
                state['target_angle'] = None
                state['motor_commands'] = None
        else:
            state['oriented'] = False
            state['target_angle'] = None
            state['motor_commands'] = None

        return state


    def reset(self):
        """Reset fields to initial state."""
        super().reset()
        self._started_orienting = False


