import torch
import random

random.seed(3)

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

    def register_object(self, name, location, angle):
        """Register a known object with its location and orientation (for grabbing)."""
        if name not in self.objects:
            self.objects[name] = {}

        self.objects[name]['location'] = location
        self.objects[name]['angle'] = angle


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
            return True, self.objects[name]['location']
        return False, None

    def find_object_angle(self, name):
        """Get the orientation of a known object by name."""
        if name in self.objects and 'angle' in self.objects[name]:
            return self.objects[name]['angle']
        return None

    def in_range(self, target_location, reach: float = 2.0) -> bool:
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

class GripperInteractor:
    """Handles gripper position and movement."""

    def __init__(
            self,
            max_speed = 0.1,            # max speed per step
            gain = 1.0,
            stop_threshold = 0.01,      # distance to consider "arrived"
            orient_threshold = 0.01,    # radians to consider "oriented"
            max_reach = 2.0,            # max reach from robot base
            get_robot_position = None,  # anchor arm to robot base
            is_open = False,            # gripper state
            has_object_state = False          # whether gripper is holding an object
        ):
        self.max_speed = max_speed
        self.gain = gain
        self.stop_threshold = stop_threshold
        self.orient_threshold = orient_threshold
        self.max_reach = max_reach
        self.gripper_orientation = torch.tensor([0.0, 0.0, 0.0])
        self.normalize_orientation()

        self.is_open = is_open
        self._has_object = has_object_state
        self.wait_counter = 0  # for simulating object grasp delay

        # Robot position reference
        if get_robot_position is None:
            raise ValueError("GripperInteractor requires a get_robot_position callable")
        self._get_robot_position = get_robot_position

        # Initial gripper position (at robot base)
        robot_position = self._get_robot_position().clone()
        self.gripper_offset = torch.tensor([0.0, 0.0, 0.5]) # gripper is slightly above base
        self.gripper_position = robot_position + self.gripper_offset


    def normalize_orientation(self):
        """Normalize orientation angles to the range [-π, π]."""
        self.gripper_orientation = (self.gripper_orientation + torch.pi) % (2 * torch.pi) - torch.pi

    def get_position(self):
        """Get the current gripper position."""
        robot_position = self._get_robot_position()
        self.gripper_position = robot_position + self.gripper_offset

        return self.gripper_position.clone()

    def get_orientation(self):
        """ Get the current gripper orientation."""
        return self.gripper_orientation.clone()

    def is_oriented(self, target_orientation, thresh: float = 0.01) -> bool:
        """Orientation arrival test within threshold (radians)."""
        if target_orientation is None:
            return False
        threshold = self.orient_threshold if thresh is None else float(thresh)

        # Handle angle wrapping for each component (assuming Euler angles)
        orientation_diff = torch.abs(target_orientation - self.gripper_orientation)
        # For each angle, take the shorter path around the circle
        orientation_diff = torch.min(orientation_diff, 2 * torch.pi - orientation_diff)

        return bool(torch.all(orientation_diff <= threshold))

    def is_open(self):
        """Check if gripper is open."""
        return self.is_open

    def open_gripper(self):
        """Open the gripper."""
        self.is_open = True

        return self.is_open

    def close_gripper(self):
        """Close the gripper."""
        self.is_open = False

        return self.is_open

    def has_object(self, gripper_position=None, object_position=None):
        """Check if gripper is holding an object (closed)."""
        # Return True after some time if gripper is closed and object is close to gripper
        if gripper_position is None:
            gripper_position = self.get_position()

        # Check proximity if object position is provided
        proximity_check = False
        if object_position is not None:
            # Check if object is within 0.05 units of gripper
            distance = torch.norm(gripper_position - object_position)
            proximity_check = distance <= 0.1

        if not self.is_open and proximity_check:
            self.wait_counter += 1
            # Only set to True if both conditions are met
            if self.wait_counter >= 10:
                self._has_object = True
        else:
            # Reset counter if gripper opens
            self.wait_counter = 0
            self._has_object = False

        return self._has_object


    def gripper_rotate_towards(self, target_orientation):
        """
        Rotate gripper towards target orientation.
        Returns the applied rotation command tensor [droll, dpitch, dyaw] or None if arrived/invalid.
        """
        if target_orientation is None:
            return None
        if self.is_oriented(target_orientation):
            return None

        delta = target_orientation - self.gripper_orientation

        # Normalize angles to [-π, π] to ensure shortest path
        delta = (delta + torch.pi) % (2 * torch.pi) - torch.pi

        # Proportional step with clamping
        rotation_step = torch.clamp(self.gain * delta, -self.max_speed, self.max_speed)

        # Update orientation
        self.gripper_orientation += rotation_step

        # Normalize orientation to [-π, π] range after update
        self.gripper_orientation = (self.gripper_orientation + torch.pi) % (2 * torch.pi) - torch.pi

        motor_cmd = rotation_step

        return motor_cmd

    def calculate_distance(self, target_location):
        """Calculate distance from gripper to target."""
        if target_location is None:
            return float('inf')

        self.get_position()  # updates self.gripper_position from robot + offset

        distance = torch.norm(target_location - self.gripper_position)  # 3D distance
        return float(distance)

    def gripper_is_at(self, target_location, thresh: float = 0.01) -> bool:
        """Arrival test within threshold."""
        if target_location is None:
            return False
        threshold = self.stop_threshold if thresh is None else float(thresh)

        return self.calculate_distance(target_location) <= threshold

    def _direction_and_distance(self, target_location):
        """Internal: unit direction (3D) and distance."""
        delta = target_location - self.gripper_position
        dist = torch.norm(delta)
        if dist.item() == 0.0:
            return torch.tensor([0.0, 0.0, 0.0]), 0.0
        return (delta / dist), float(dist)

    def gripper_move_towards(self, target_location):
        """
        Plan and execute a single bounded step toward target.
        Returns the applied motor command tensor [dx, dy, dz] or None if arrived/invalid.
        """
        if target_location is None:
            return None

        direction, distance = self._direction_and_distance(target_location)
        # Proportional step with clamp to max_speed and no overshoot
        step_mag = min(self.max_speed, self.gain * distance, distance)
        step_vec = direction * step_mag

        # Calculate new potential position
        robot_position = self._get_robot_position()
        new_offset = self.gripper_offset + step_vec

        # Check if within reach constraints
        if torch.norm(new_offset) <= self.max_reach:
            self.gripper_offset = new_offset
        else:
            # If beyond reach, project back to max reach sphere
            normalized_offset = new_offset / torch.norm(new_offset)
            self.gripper_offset = normalized_offset * self.max_reach

        # Update pose (3D)
        self.gripper_position = robot_position + self.gripper_offset

        motor_cmd = step_vec
        return motor_cmd

    def reset(self):
        """Reset gripper state."""
        robot_position = self._get_robot_position().clone()
        self.gripper_offset = torch.tensor([0.0, 0.0, 0.5]) # gripper is slightly above base
        self.gripper_position = robot_position + self.gripper_offset

class RobotInteractors:
    """Facade that provides access to all interactors."""

    def __init__(self):
        self.movement = MovementInteractor()
        self.gripper = GripperInteractor(get_robot_position=self.movement.get_position)
        self.perception = PerceptionInteractor(get_robot_position=self.movement.get_position)

    def reset(self):
        """Reset all interactors."""
        self.perception.reset()
        self.movement.reset()
        self.gripper.reset()