import torch
import numpy as np
import matplotlib.pyplot as plt
from elementary_behavior_simplified import ElementaryBehaviorSimple


class FindBehavior(ElementaryBehaviorSimple):
    """
    Elementary behavior to find a target object in 3D space.
    Uses external object recognition capabilities to locate objects.
    """

    def __init__(self, field_params=None):
        super().__init__(field_params)
        self.target_location = None
        self._last_active = False
        self._started_searching = False

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
        # Try to locate the target using the interactor only if the behavior is active
        if self._last_active:
            self._started_searching = True

        if self._started_searching:
            target_found, target_location = interactor.find_object(target_name)
        else:
            target_found, target_location = False, None

        # Prepare inputs for nodes
        intention_input = external_input
        # Keep CoS active while target is found
        cos_input = 5.0 if target_found else 0.0

        # Process behavior control
        state = self.forward(intention_input, cos_input)

        # Store target location if found
        if target_found and target_location is not None:
            self.target_location = target_location

        # Add target information to state
        state['target_found'] = target_found
        state['target_location'] = self.target_location

        # Remember last active state
        self._last_active = bool(state.get('active', False))
        #self._last_active = float(state.get('intention_activity', 0.0)) > 0.0

        return state

    def reset(self):
        """Reset the behavior state and nodes."""
        super().reset()
        self.target_location = None
        self._last_active = False
        self._sensing_latch = False