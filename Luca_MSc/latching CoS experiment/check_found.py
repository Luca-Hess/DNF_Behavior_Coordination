from check_behavior import CheckBehavior
import torch

class CheckFoundBehavior(CheckBehavior):
    """
    Low-cost sanity check behavior to verify if a found target is still present.
    """

    def __init__(self, field_params=None):
        super().__init__(field_params)
        self._confidence_low = False

    def execute(self, interactor, target_name, external_input=0.0, passed_find_behavior=None):
        """
        Execute one step of the find behavior.

        Args:
            interactor: Object providing sensory/recognition capabilities
            target_name: Name of the target object to find
            external_input: External input to drive the intention node
            find_behavior: The FindBehavior instance to reset if sanity check fails

        Returns:
            dict: Behavior state and found target information
        """
        # Process behavior control
        state = self.forward(external_input, None)
        self._confidence_low = bool(state.get('confidence_low', False))
        sanity_check_failed = None

        if self._confidence_low:
            # Only perform one query per confidence threshold crossing
            if not hasattr(self, '_sanity_check_done') or not self._sanity_check_done:
                # Add a static counter to simulate target loss on 3rd query
                if not hasattr(self, '_query_count'):
                    self._query_count = 0
                self._query_count += 1
                # On 3rd query, always simulate target not found
                if self._query_count == 2:
                    target_found, target_location = False, None
                else:
                    target_found, target_location = interactor.find_object(target_name)
                self._sanity_check_done = True
                # If object is lost, reset find_behavior
                if not target_found and passed_find_behavior is not None:
                    #manually reset find behavior intention and cos nodes
                    passed_find_behavior.intention.activation.fill_(passed_find_behavior.intention.resting_level)
                    passed_find_behavior.intention.g_u.copy_(torch.sigmoid(passed_find_behavior.intention.beta * passed_find_behavior.intention.activation))

                    passed_find_behavior.CoS.activation.fill_(passed_find_behavior.CoS.resting_level)
                    passed_find_behavior.CoS.g_u.copy_(torch.sigmoid(passed_find_behavior.CoS.beta * passed_find_behavior.CoS.activation))


                    sanity_check_failed = True
                else:
                    # If check passes, reset this check behavior for next cycle
                    self.reset()
                    sanity_check_failed = False
            else:
                # Already performed query for this threshold crossing
                sanity_check_failed = None
        else:
            # Reset flag when confidence is not low
            self._sanity_check_done = False
            sanity_check_failed = None

        state['sanity_check_failed'] = sanity_check_failed
        return state

    def reset(self):
        """Reset the behavior state and nodes."""
        super().reset()
        self._confidence_low = False
