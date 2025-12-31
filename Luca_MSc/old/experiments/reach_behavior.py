import torch
from elementary_behavior import ElementaryBehavior


class ReachBehavior(ElementaryBehavior):
    """
    Example behavior: Reach to a target location using node dynamics.
    """

    def __init__(self):
        # Create with 0D fields (nodes)
        super().__init__()

    def execute(self, target_position, current_position, obstacles=None):
        """
        Execute one step of the reach behavior using node inputs.

        Args:
            target_position: The target position to reach
            current_position: Current position of the robot
            obstacles: Optional information about obstacles

        Returns:
            dict: Contains behavior state and motor commands
        """
        # Calculate scalar inputs for control nodes:
        distance_to_target = torch.norm(target_position - current_position)

        # CoI: Activate when target is different from current position
        coi_gain = 20.0
        coi_offset = -0.2 # small offset to avoid noise triggering when close to target
        coi_input = coi_gain * (distance_to_target + coi_offset)
        coi_input = max(0.0, float(coi_input)) # No negative input

        # CoS: Activate when close enough to target
        cos_gain = 30.0
        cos_threshold = 0.1 # Distance for satisfaction
        cos_input = cos_gain * max(0.0, cos_threshold - distance_to_target)

        # CoF: Activate when obstacle is too close
        cof_input = 0.0
        if obstacles is not None:
            obstacle_distances = obstacles if isinstance(obstacles, list) else [obstacles]
            if len(obstacle_distances) > 0:
                min_obstacle_distance = torch.min(torch.tensor(obstacle_distances))
                cof_gain = 25.0
                cof_threshold = 0.15  # Distance threshold for failure
                cof_input = cof_gain * max(0.0, cof_threshold - min_obstacle_distance)

        # Process behavior control
        state = self.forward(coi_input, cos_input, cof_input)

        # Generate motor commands if active
        motor_commands = None
        if state['active']:
            # Simple proportional controller
            direction = target_position - current_position
            motor_gain = 0.5
            motor_commands = motor_gain * direction

        return {**state, 'motor_commands': motor_commands}


# Example simulation
if __name__ == "__main__":
    behavior = ReachBehavior()

    # Simulate target and current positions
    target = torch.tensor([1.0, 0.5])
    current = torch.tensor([0.0, 0.0])

    # Run behavior for several steps
    for step in range(1000):
        # Update current position based on previous commands
        if step > 0 and result['motor_commands'] is not None:
            current += result['motor_commands']

        # Execute behavior
        result = behavior.execute(target, current)

        # Print status
        print(f"Step {step}: Active={bool(result['active'])}, "
              f"Completed={bool(result['completed'])}, "
              f"Failed={bool(result['failed'])}, "
              f"Position={current.tolist()}, "
              f"CoI={result['coi_g_u']:.2f},{float(result['coi_activity']):.2f}, "
              f"CoS={result['cos_g_u']:.2f},{float(result['cos_activity']):.2f}, "
              f"CoF={result['cof_g_u']:.2f},{float(result['cof_activity']):.2f}")

        # Stop if behavior completed or failed
        if result['completed'] or result['failed']:
            break