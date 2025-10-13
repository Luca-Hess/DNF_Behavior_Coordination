import torch
from elementary_behavior_simplified import ElementaryBehaviorSimple

class MoveToBehavior(ElementaryBehaviorSimple):
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
        if target_location is None:
            # No target, can't move
            state = self.forward(0.0, 0.0)
            return {**state, 'motor_commands': None, 'distance': float('inf')}

        # Get current position and calculate distance via the interactor
        current_position = interactor.get_position()
        distance_to_target = interactor.calculate_distance(target_location)

        # Calculate scalar inputs for control nodes:
        # Intention: Activate when far from target
        intention_gain = 50.0
        intention_offset = -0.2  # small offset to avoid noise triggering when close
        intention_input = intention_gain * (distance_to_target + intention_offset)
        intention_input = max(0.0, min(6.0, float(intention_input)))  # Clamp to [0,6]

        # Add external input (e.g., from precondition nodes)
        intention_input += external_input

        # CoS: Activate when close enough to target
        cos_gain = 100.0
        cos_threshold = 0.1  # Distance for satisfaction
        cos_input = cos_gain * max(0.0, cos_threshold - distance_to_target)

        # Process behavior control
        state = self.forward(intention_input, cos_input)

        # Generate motor commands if active
        motor_commands = None
        if state['active']:
            # Simple proportional controller
            direction = target_location[:2] - current_position[:2]  # Planar direction
            direction = torch.clamp(direction, -10.0, 10.0)
            motor_gain = 0.5
            motor_commands = motor_gain * direction

            # Execute movement via interactor
            if motor_commands is not None:
                interactor.move_robot(motor_commands)

        # Add additional info to state
        state['distance'] = distance_to_target
        state['motor_commands'] = motor_commands.tolist() if motor_commands is not None else None

        return state