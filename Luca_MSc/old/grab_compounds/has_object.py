from Luca_MSc.old.elementary_behavior import ElementaryBehavior


class HasObjectBehavior(ElementaryBehavior):
    """Behavior to close the gripper to grab an object."""

    def __init__(self, field_params=None):
        super().__init__(field_params)
        self._last_active = False
        self._checked_object = False
        self.cos_input = 0.0

    def execute(self, interactors, target_location=None, external_input=0.0):
        """Check if the gripper is holding an object."""
        if self._last_active:
            self._checked_object = True

        if self._checked_object:
            has_object = interactors.gripper.has_object(interactors.gripper.gripper_position, target_location)
        else:
            has_object = False

        # Use object grabbed status for CoS
        cos_input = 5.0 if has_object else 0.0

        # Process behavior control
        state = self.forward(external_input, cos_input)

        # Store object holding status
        state['has_object'] = has_object

        self._last_active = float(state.get('intention_activity', 0.0)) > 0.0

        return state

    def reset(self):
        super().reset()
        self.cos_input = 0.0
        self._last_active = False
        self._checked_object = False
