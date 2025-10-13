import torch
from DNF_torch.field import Field
from find import FindBehavior

class PerceptionInteractor:
    """Handles object detection and tracking."""

    def __init__(self):
        self.objects = {}
        self.search_attempts = 0

    def register_object(self, name, location):
        """Register a known object with its location."""
        self.objects[name] = location

    def find_object(self, name):
        """Attempt to find an object by name."""
        self.search_attempts += 1

        # Simulate finding the object after a few attempts
        if name in self.objects and self.search_attempts >= 3:
            return True, self.objects[name]
        return False, None

    def reset(self):
        """Reset perception state."""
        self.search_attempts = 0


class MovementInteractor:
    """Handles robot position, movement, and spatial calculations."""

    def __init__(self):
        self.robot_position = torch.tensor([0.0, 0.0, 0.0])

    def get_position(self):
        """Get the current robot position."""
        return self.robot_position.clone()

    def move_robot(self, motor_commands):
        """
        Move the robot according to motor commands.
        Returns the updated position.
        """
        if motor_commands is not None:
            # Apply planar movement
            if len(motor_commands) >= 2:
                self.robot_position[:2] += motor_commands[:2]

        return self.robot_position.clone()

    def calculate_distance(self, target_location):
        """Calculate distance from robot to target."""
        if target_location is None:
            return float('inf')

        # Calculate planar distance
        planar_distance = torch.norm(target_location[:2] - self.robot_position[:2])
        return float(planar_distance)

    def reset(self):
        """Reset movement state."""
        self.robot_position = torch.tensor([0.0, 0.0, 0.0])


class RobotInteractors:
    """Facade that provides access to all interactors."""

    def __init__(self):
        self.perception = PerceptionInteractor()
        self.movement = MovementInteractor()

    def reset(self):
        """Reset all interactors."""
        self.perception.reset()
        self.movement.reset()