from Luca_MSc.elementary_behavior import ElementaryBehavior


class HasObjectBehavior(ElementaryBehavior):
    """Behavior to close the gripper to grab an object."""

    def __init__(self, field_params=None):
        super().__init__(field_params)
        self._last_active = False
        self._checked_object = False

    def execute(self, gripper_interactor, external_input=0.0):
        """Check if the gripper is holding an object."""

        # Try to check the range using the interactor only if the behavior is active
        if self._last_active:
            self._checked_object = True

        if self._checked_object:
            has_object = gripper_interactor.has_object
        else:
            has_object = False

        # Use range status for CoS
        cos_input = 5.0 if has_object else 0.0

        # Process behavior control
        state = self.forward(external_input, cos_input)

        # Store range status
        state['has_object'] = has_object

        self._last_active = state['active']

        return state

    def reset(self):
        super().reset()
        self._last_active = False
        self._checked_object = False