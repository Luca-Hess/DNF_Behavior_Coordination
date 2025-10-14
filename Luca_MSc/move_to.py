import torch
from elementary_behavior_simplified import ElementaryBehaviorSimple

class MoveToBehavior(ElementaryBehavior):
    """
    Elementary behavior to move towards a target position.
    Uses interactor for position tracking and navigation.
    """

    def __init__(self, field_params=None):
        super().__init__(field_params)

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
        # If no target, stay quiescent
        if target_location is None:
            state = self.forward(0.0, 0.0)
            state['arrived'] = False
            state['motor_commands'] = None
            return state

        # Arrival check based on data from interactor, with specified "arrival" threshold
        arrived = bool(interactor.is_at(target_location, thresh=0.1))

        # Prepare inputs for nodes
        intention_input = external_input
        cos_input = 5.0 if arrived else 0.0

        # Process behavior control
        state = self.forward(intention_input, cos_input)

        # Generate motor commands if active
        motor_cmd = None
        if float(state.get('intention_activity', 0.0)) > 0.0 and not arrived:
            motor_cmd = interactor.move_towards(target_location)

        # Diagnostics/echo
        state['arrived'] = arrived
        state['motor_commands'] = (
            motor_cmd.tolist() if hasattr(motor_cmd, "tolist") else motor_cmd
        )

        return state

    def reset(self):
        super().reset()
