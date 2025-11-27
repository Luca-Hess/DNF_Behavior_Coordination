from Luca_MSc.old.elementary_behavior import ElementaryBehavior


class OrientBehavior(ElementaryBehavior):
    """Behavior to orient the gripper correctly for grabbing."""

    def __init__(self, field_params=None):
        super().__init__(field_params)

    def execute(self, interactors, target_name, threshold, external_input=0.0):
        """Execute orientation behavior toward a target location."""

        # Check if target object exists - reset if not
        target_exists = target_name in interactors.perception.objects
        if not target_exists:
            state = self.forward(0.0, 0.0)
            state['oriented'] = False
            state['target_angle'] = None
            state['motor_commands'] = None
            return state

        target_angle = interactors.perception.find_object_angle(target_name)

        if target_angle is not None:
            # Arrival check based on data from interactor, with specified "arrival" threshold
            oriented = interactors.gripper.is_oriented(target_angle, thresh=threshold)
        else:
            oriented = False

        # Prepare inputs for nodes
        cos_input = 5.0 if oriented else 0.0

        # Process behavior control
        state = self.forward(external_input, cos_input)

        # Generate motor commands if active
        motor_cmd = None
        if float(state.get('intention_activity', 0.0)) > 0.0 and not oriented:
            motor_cmd = interactors.gripper.gripper_rotate_towards(target_angle)


        # Updating state
        state['oriented'] = oriented
        state['target_angle'] = target_angle
        state['motor_commands'] = (
            motor_cmd.tolist() if hasattr(motor_cmd, "tolist") else motor_cmd
        )

        return state

    def reset(self):
        """Reset fields to initial state."""
        super().reset()
        self._started_orienting = False


