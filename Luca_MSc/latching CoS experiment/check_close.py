from check_behavior import CheckBehavior
import torch

class CheckCloseBehavior(CheckBehavior):
    """
    Low-cost sanity check behavior to verify if a found target is still present.
    """

    def __init__(self, field_params=None):
        super().__init__(field_params)
        self._confidence_low = False

    def execute(self, interactor, target_location, external_input=0.0, passed_move_behavior=None):
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
                still_arrived = bool(interactor.is_at(target_location, thresh=0.1))
                self._sanity_check_done = True
                # If object is lost, reset find_behavior
                if not still_arrived and passed_move_behavior is not None:
                    # change the cos_input the CoS node receives from find_int to 0.0
                    passed_move_behavior.cos_input = 0.0
                    passed_move_behavior.arrived = False 
                    passed_move_behavior.latched_cos = False

                    sanity_check_failed = True
                    self.reset()
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
