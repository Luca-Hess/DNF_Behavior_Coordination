import torch
from elementary_behavior import ElementaryBehavior

class CheckEffectorRange(ElementaryBehavior):
    """
    Elementary behavior to check if the end-effector is in range of a target.
    Uses interactor to do so.
    """

    def __init__(self, field_params=None):
        super().__init__(field_params)
        self._last_active = False
        self._checked_range = False

    def execute(self, interactor, target_location, effector_reach = 0.5, external_input=0.0):
        """
        Execute one step of the effector distance checking behavior.

        Args:
            interactor: Object providing 3D distance checking
            target_location: 3D position of the target
            external_input: Additional input to intention node (e.g., from preconditions)

        Returns:
            dict: Behavior state
        """
        # Try to check the range using the interactor only if the behavior has been activated before
        if self._last_active:
            self._checked_range = True

        if self._checked_range:
            in_range = interactor.in_range(target_location, reach=effector_reach)
        else:
            in_range = False

        # Use range status for CoS
        cos_input = 5.0 if in_range else 0.0

        # Process behavior control
        state = self.forward(external_input, cos_input)

        # Store range status
        state['in_range'] = in_range

        self._last_active = float(state.get('intention_activity', 0.0)) > 0.0

        return state

    def reset(self):
        super().reset()
        self._last_active = False
        self._checked_range = False
