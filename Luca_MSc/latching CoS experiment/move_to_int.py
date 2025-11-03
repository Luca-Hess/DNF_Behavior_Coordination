import os
import sys
# Add DNF_torch package root
sys.path.append(os.path.expanduser('~/nc_ws/DNF_torch'))

# Add Luca_MSc subfolder for local scripts
sys.path.append(os.path.join(os.path.expanduser('~/nc_ws/DNF_torch'), 'Luca_MSc/latching CoS experiment'))

from elementary_behavior_latch import ElementaryBehavior_LatchingCoS

class MoveToBehavior_IntentionCoupling(ElementaryBehavior_LatchingCoS):
    """
    Elementary behavior to move towards a target position.
    Uses interactor for position tracking and navigation.
    """

    def __init__(self, field_params=None):
        super().__init__(field_params)
        self._last_active = False
        self.cos_input = 0.0

    def execute(self, interactor, target_location, external_input=0.0):
        """
        Execute one step of the move-to behavior.

        Args:
            interactor: Object providing position and navigation capabilities
            target_location: 3D position of the target
            external_input: Additional input to intention node (e.g., from preconditions)

        Returns:
            dict: Behavior state and movement commands
        """

        if self._last_active:
            # If no target, stay quiescent
            if target_location is None:
                state = self.forward(0.0, 0.0)
                state['arrived'] = False
                state['motor_commands'] = None
                return state

            # Arrival check based on data from interactor, with specified "arrival" threshold
            arrived = bool(interactor.is_at(target_location, thresh=0.1))

            # Prepare inputs for nodes
            self.cos_input = 5.0 if arrived else 0.0

            # Generate motor commands if active
            motor_cmd = None
            if not arrived:
                motor_cmd = interactor.move_towards(target_location)

        else:
            arrived = False
            motor_cmd = None

        # Process behavior control
        state = self.forward(external_input, self.cos_input)

        self._last_active = float(state.get('intention_activity', 0.0)) > 0.0

        # Diagnostics/echo
        state['arrived'] = arrived
        state['motor_commands'] = (
            motor_cmd.tolist() if hasattr(motor_cmd, "tolist") else motor_cmd
        )

        return state


    def reset(self):
        super().reset()
