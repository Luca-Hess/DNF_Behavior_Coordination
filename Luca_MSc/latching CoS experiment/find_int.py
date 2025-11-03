import os
import sys
# Add DNF_torch package root
sys.path.append(os.path.expanduser('~/nc_ws/DNF_torch'))

# Add Luca_MSc subfolder for local scripts
sys.path.append(os.path.join(os.path.expanduser('~/nc_ws/DNF_torch'), 'Luca_MSc/latching CoS experiment'))

from elementary_behavior_latch import ElementaryBehavior_LatchingCoS


class FindBehavior_IntentionCoupling(ElementaryBehavior_LatchingCoS):
    """
    Elementary behavior to find a target object in 3D space.
    Uses external object recognition capabilities to locate objects.
    """

    def __init__(self, field_params=None):
        super().__init__(field_params)
        self.target_location = None
        self._last_active = False
        self.cos_input = 0.0

    def execute(self, interactor, target_name, external_input=0.0):
        """
        Execute one step of the find behavior.

        Args:
            interactor: Object providing sensory/recognition capabilities
            target_name: Name of the target object to find
            external_input: External input to drive the intention node

        Returns:
            dict: Behavior state and found target information
        """
        if self._last_active:
            target_found, target_location = interactor.find_object(target_name)

            # Store target location if found
            if target_found and target_location is not None:
                self.cos_input = 5.0
                self.target_location = target_location

        else:
            target_found, target_location = False, None

        # Process behavior control
        state = self.forward(external_input, self.cos_input)

        self._last_active = float(state.get('intention_activity', 0.0)) > 0.0

        # Add target information to state
        state['target_found'] = target_found
        state['target_location'] = self.target_location

        return state

    def reset(self):
        """Reset the behavior state and nodes."""
        super().reset()
        self.target_location = None
        self._last_active = False