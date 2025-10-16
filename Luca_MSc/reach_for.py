import torch
from elementary_behavior import ElementaryBehavior

class ReachForBehavior(ElementaryBehavior):
    """
    Elementary behavior to move towards a target position.
    Uses interactor for position tracking and navigation.
    """

    def __init__(self, field_params=None):
        super().__init__(field_params)

    def execute(self, interactor, target_location, threshold, external_input=0.0):
        """
        Execute one step of the move-to behavior.

        Args:
            interactor: Object providing position and navigation capabilities
            target_location: 3D position of the target
            threshold: Distance threshold to consider "arrived"
            external_input: Additional input to intention node (e.g., from preconditions)

        Returns:
            dict: Behavior state and movement commands
        """
        # If no target, stay quiescent
        if target_location is None:
            state = self.forward(0.0, 0.0)
            state['gripper_arrived'] = False
            state['motor_commands'] = None
            return state

        # Gripper arrival at object check based on data from interactor, with specified "arrival" threshold
        gripper_arrived = bool(interactor.gripper_is_at(target_location, thresh=threshold))

        # Prepare inputs for nodes
        cos_input = 5.0 if gripper_arrived else 0.0

        # Process behavior control
        state = self.forward(external_input, cos_input)

        # Generate motor commands if active
        motor_cmd = None
        if state['active'] and not gripper_arrived:
            motor_cmd = interactor.gripper_move_towards(target_location)

        # Logs
        state['gripper_arrived'] = gripper_arrived
        state['motor_commands'] = (
            motor_cmd.tolist() if hasattr(motor_cmd, "tolist") else motor_cmd
        )

        return state

    def reset(self):
        super().reset()
