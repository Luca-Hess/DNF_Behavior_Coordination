from Luca_MSc.elementary_behavior import ElementaryBehavior


class HasObjectBehavior(ElementaryBehavior):
    """Behavior to close the gripper to grab an object."""

    def __init__(self, field_params=None):
        super().__init__(field_params)
        self._last_active = False
        self._checked_object = False
        self.cos_input = 0.0

    def execute(self, gripper_interactor, external_input=0.0):
        """Check if the gripper is holding an object."""
        # Process behavior to determine activity
        state = self.forward(external_input, self.cos_input)

        active = float(state.get('intention_activity', 0.0)) > 0.0

        if active:
            # Check if gripper has an object
            has_object = gripper_interactor.has_object
            self.cos_input = 5.0 if has_object else 0.0
        else:
            has_object = False

        # Store object holding status
        state['has_object'] = has_object

        return state

    def reset(self):
        super().reset()
        self.cos_input = 0.0
