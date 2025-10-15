import torch
import random

random.seed(1)

class PerceptionInteractor:
    """Handles object detection and tracking."""

    def __init__(self, tracking_loss_prob=0.2, max_tracking_loss_duration=5, get_robot_position=None):
        self.objects = {}
        self.search_attempts = 0

        # Robot position (for range checks)
        if get_robot_position is None:
            raise ValueError("PerceptionInteractor requires a get_robot_position callable")
        self._get_robot_position = get_robot_position

        # Simulating tracking loss
        self.tracking_loss_probability = tracking_loss_prob
        self.max_tracking_loss_duration = max_tracking_loss_duration
        self.in_tracking_loss = False
        self.tracking_loss_remaining = 0

    def register_object(self, name, location):
        """Register a known object with its location."""
        self.objects[name] = location

    def find_object(self, name):
        """Attempt to find an object by name."""
        self.search_attempts += 1

        # Check if we should start a tracking loss period
        if not self.in_tracking_loss and random.random() < self.tracking_loss_probability:
            self.in_tracking_loss = True
            self.tracking_loss_remaining = random.randint(1, self.max_tracking_loss_duration)

        # If we're in a tracking loss period
        if self.in_tracking_loss:
            self.tracking_loss_remaining -= 1
            if self.tracking_loss_remaining <= 0:
                self.in_tracking_loss = False
            return False, None

        # Normal detection logic
        if name in self.objects and self.search_attempts >= 3:
            return True, self.objects[name]
        return False, None

    def in_range(self, target_location, reach: float = 0.5) -> bool:
        """Check if target is within a certain range in 3D space,
        based on the end-effector reach."""
        if target_location is None:
            return False

        robot_pos = self._get_robot_position()

        distance = torch.norm(target_location - robot_pos)  # 3D distance
        if distance > reach:
            return False
        else:
            return True

    def reset(self):
        """Reset perception state."""
        self.search_attempts = 0


class MovementInteractor:
    """Handles robot position, movement, and spatial calculations."""

    def __init__(
            self,
            max_speed = 0.1,        # max speed per step
            gain = 1.0,
            stop_threshold = 0.1    # distance to consider "arrived"
        ):
        self.robot_position = torch.tensor([0.0, 0.0, 0.0])
        self.max_speed = max_speed
        self.gain = gain
        self.stop_threshold = stop_threshold

    def get_position(self):
        """Get the current robot position."""
        return self.robot_position.clone()

    def calculate_distance(self, target_location):
        """Calculate distance from robot to target."""
        if target_location is None:
            return float('inf')

        # Calculate planar distance
        planar_distance = torch.norm(target_location[:2] - self.robot_position[:2])
        return float(planar_distance)

    def is_at(self, target_location, thresh: float = None) -> bool:
        """Arrival test within threshold."""
        if target_location is None:
            return False
        threshold = self.stop_threshold if thresh is None else float(thresh)
        return self.calculate_distance(target_location) <= threshold

    def _direction_and_distance(self, target_location):
        """Internal: unit direction (2D) and distance."""
        delta = target_location[:2] - self.robot_position[:2]
        dist = torch.norm(delta)
        if dist.item() == 0.0:
            return torch.tensor([0.0, 0.0]), 0.0
        return (delta / dist), float(dist)

    def move_towards(self, target_location):
        """
        Plan and execute a single bounded step toward target.
        Returns the applied motor command tensor [dx, dy, dz] or None if arrived/invalid.
        """
        if target_location is None:
            return None
        if self.is_at(target_location):
            return None

        direction, distance = self._direction_and_distance(target_location)
        # Proportional step with clamp to max_speed and no overshoot
        step_mag = min(self.max_speed, self.gain * distance, distance)
        step_vec_2d = direction * step_mag

        # Update pose (planar)
        self.robot_position[:2] += step_vec_2d

        motor_cmd = torch.tensor([step_vec_2d[0].item(), step_vec_2d[1].item(), 0.0])
        return motor_cmd

    def move_robot(self, motor_commands):
        """
        Backward‑compat: directly apply provided motor delta (planar).
        Returns updated position.
        """
        if motor_commands is not None:
            if len(motor_commands) >= 2:
                if not torch.is_tensor(motor_commands):
                    motor_commands = torch.tensor(motor_commands, dtype=torch.float32)
                self.robot_position[:2] += motor_commands[:2]
        return self.robot_position.clone()

    def reset(self):
        """Reset movement state."""
        self.robot_position = torch.tensor([0.0, 0.0, 0.0])


class RobotInteractors:
    """Facade that provides access to all interactors."""

    def __init__(self):
        self.movement = MovementInteractor()
        self.perception = PerceptionInteractor(get_robot_position=self.movement.get_position)

    def reset(self):
        """Reset all interactors."""
        self.perception.reset()
        self.movement.reset()