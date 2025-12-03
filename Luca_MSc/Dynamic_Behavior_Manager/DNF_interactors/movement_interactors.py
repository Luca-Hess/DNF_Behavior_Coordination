
from .base_interactors import BaseInteractor

import torch


class MovementInteractor(BaseInteractor):
    """Handles robot position, movement, and spatial calculations."""

    def __init__(
            self,
            max_speed = 0.02,        # max speed per step
            gain = 1.0,
            stop_threshold = 0.1,   # distance to consider "arrived"
            **kwargs
    ):
        super().__init__(**kwargs)

        self.robot_position = torch.tensor([0.0, 0.0, 0.0])
        self.max_speed = max_speed
        self.gain = gain
        self.stop_threshold = stop_threshold

        # Failure simulation parameters
        self.move_attempts = 0
        self.stuck_threshold = kwargs.get('stuck_threshold', 0.01) # Minimum movement to consider "not stuck"
        self.previous_position = self.robot_position.clone()

    def move_to(self, target_location, requesting_behavior=None):
        """Move method for robot movement"""
        failure_reason = None

        # For continuous calls, actually move the robot
        if requesting_behavior:
            self.move_towards(target_location)
            self.move_attempts += 1

        # Check for movement failures
        if self._is_stuck():
            failure_reason = "Robot appears to be stuck (no movement progress)"


        # Check arrival status
        arrived = self.is_at(target_location)
        position = self.get_position()

        # Reset movement attempts if arrived
        if arrived:
            self.move_attempts = 0

        # Determine CoS & CoF condition
        cos_condition = arrived
        cof_condition = (failure_reason is not None)

        if failure_reason is not None:
            print(f"MovementInteractor.move_to failure: {failure_reason}")

        state_data = {
            'arrived': arrived,
            'distance': self.calculate_distance(target_location),
            'position': position,
            'move_attempts': self.move_attempts,
            'failure_reason': failure_reason
        }

        # Use base class helper for state management and publishing
        self._update_and_publish_state(state_data, arrived, cof_condition, requesting_behavior)

        return cos_condition, cof_condition, position

    def _is_stuck(self):
        if self.move_attempts < 10:
            return False

        movement = torch.norm(self.robot_position - self.previous_position)
        return movement < self.stuck_threshold

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


    def move_towards(self, target_location):
        """
        Plan and execute a single bounded step toward target.
        Returns the applied motor command tensor [dx, dy, dz] or None if arrived/invalid.
        """
        self.previous_position = self.robot_position.clone()

        if target_location is None or self.is_at(target_location):
            return None

        delta = target_location[:2] - self.robot_position[:2]
        dist = torch.norm(delta)

        if dist.item() == 0.0:
            return None

        direction = delta / dist

        # Proportional step with clamp to max_speed and no overshoot
        step_mag = min(self.max_speed, self.gain * float(dist), float(dist))
        step_vec_2d = direction * step_mag

        # Update pose (planar)
        self.robot_position[:2] += step_vec_2d

        # Also update the gripper and (if they exist, grabbed objects) position
        if hasattr(self, '_robot_interactors'):
            if hasattr(self._robot_interactors, 'gripper'):
                self._robot_interactors.gripper.get_position()


        motor_cmd = torch.tensor([step_vec_2d[0].item(), step_vec_2d[1].item(), 0.0])
        return motor_cmd


    def reset(self):
        """Reset movement state."""
        self.robot_position = torch.tensor([0.0, 0.0, 0.0])

        self.move_attempts = 0
