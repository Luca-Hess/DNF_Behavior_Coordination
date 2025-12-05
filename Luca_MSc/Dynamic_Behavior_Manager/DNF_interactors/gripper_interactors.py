
from .base_interactors import BaseInteractor

import torch

class GripperInteractor(BaseInteractor):
    """Handles gripper position and movement."""

    def __init__(
            self,
            max_speed=0.01,  # max speed per step
            gain=1.0,
            stop_threshold=0.01,  # distance to consider "arrived"
            orient_threshold=0.01,  # radians to consider "oriented"
            max_reach=3,  # max reach from robot base
            get_robot_position=None,  # anchor arm to robot base
            is_open=False,  # gripper state
            has_object_state=False,  # whether gripper is holding an object
            **kwargs
    ):
        super().__init__(**kwargs)

        self.max_speed = max_speed
        self.gain = gain
        self.stop_threshold = stop_threshold
        self.orient_threshold = orient_threshold
        self.max_reach = max_reach
        self.gripper_orientation = torch.tensor([0.0, 0.0, 0.0])
        self.normalize_orientation()

        self.gripper_is_open = is_open
        self.initial_approach = True  # for grab sequence
        self._has_object = has_object_state
        self.wait_counter = 0  # for simulating object grasp delay

        # Pub/Sub state
        self.grabbed_objects = {}  # Track grabbed objects per behavior

        # Robot position reference
        if get_robot_position is None:
            raise ValueError("GripperInteractor requires a get_robot_position callable")
        self._get_robot_position = get_robot_position

        # Initial gripper position (at robot base)
        robot_position = self._get_robot_position().clone()
        self.gripper_offset = torch.tensor([0.0, 0.0, 0.5])  # gripper is slightly above base
        self.gripper_position = robot_position + self.gripper_offset

    def gripper_can_reach(self, target_location):
        """Check if gripper can reach the target location."""
        if target_location is None:
            return False

        robot_position = self._get_robot_position()
        offset = target_location - robot_position
        distance = torch.norm(offset)

        return distance <= self.max_reach

    def reach_check(self, requesting_behavior=None):
        """Continuous publisher for active behaviors"""
        target_name, target_location, _ = self._robot_interactors.state.get_behavior_target_info('reach_check')


        position = self.get_position()
        reachable = self.gripper_can_reach(target_location)

        # Determine failure condition
        failure_reason = None
        if not reachable and target_location is not None:
            if target_location[2] > self.max_reach:
                failure_reason = f"Target location {target_location} is higher than the gripper can reach (max {self.max_reach})."

        distance = float(torch.norm(target_location - self.gripper_position)) if target_location is not None else float(
            'inf')

        if not requesting_behavior:
            if self.is_target_grabbed(target_name):
                reachable = True
                distance = 0.0
                failure_reason = None

        cos_condition = reachable
        cof_condition = (failure_reason is not None)

        state_data = {
            'reachable': reachable,
            'distance': distance,
            'failure_reason': failure_reason
        }

        self._update_and_publish_state(state_data, reachable, cof_condition, requesting_behavior)

        return cos_condition, cof_condition, position

    def reach_for(self, requesting_behavior=None):
        """Continuous publisher for active behaviors"""
        target_name, target_location, _ = self._robot_interactors.state.get_behavior_target_info('reach_for')

        motor_cmd = None
        failure_reason = None

        if self.is_target_grabbed(target_name):
            target_location = self.gripper_position.clone()

        # For continous calls, actually move the gripper
        if requesting_behavior and target_location is not None:
            if not self.gripper_can_reach(target_location) and target_location[2] > self.max_reach:
                failure_reason = f"Target location {target_location} is out of gripper reach (max {self.max_reach})."
            else:
                motor_cmd = self.gripper_move_towards(target_location)

        at_target = self.gripper_is_at(target_location)
        position = self.get_position()

        if self.is_target_grabbed(target_name):
            at_target = True
            failure_reason = None

        # Determine CoS & CoF condition
        cos_condition = at_target
        cof_condition = (failure_reason is not None)

        state_data = {
            'at_target': at_target,
            'motor_command': motor_cmd,
            'failure_reason': failure_reason
        }

        self._update_and_publish_state(state_data, at_target, cof_condition, requesting_behavior)

        return cos_condition, cof_condition, position, motor_cmd

    ## Gripper interactors for grabbing - consists of orient, open, fine_reach, close, has_object check (as final CoS driver)
    def grab(self, requesting_behavior=None):
        """Continuous publisher for active behaviors - full grab sequence"""
        target_name, target_location, target_orientation = self._robot_interactors.state.get_behavior_target_info('grab')


        motor_cmd_orient = None
        motor_cmd_reach = None
        grabbed = False
        failure_reason = None

        # For continuous calls, execute grab sequence stepwise
        if requesting_behavior:
            if not self.gripper_can_reach(target_location):
                failure_reason = f"Target location {target_location} is out of gripper reach (max {self.max_reach})."
            else:
                # First orient gripper to target if needed
                if not self.is_oriented(target_orientation):
                    motor_cmd_orient = self.gripper_rotate_towards(target_orientation)

                # Open gripper if needed, then reach
                elif not self.gripper_is_open and self.initial_approach:
                    self.open_gripper()
                    self.initial_approach = False

                # Move gripper to target location - fine reaching!
                elif not self.gripper_is_at(target_location):
                    motor_cmd_reach = self.gripper_move_towards(target_location)

                # Once at target, close gripper
                elif self.gripper_is_open:
                    self.close_gripper()

                if self.is_oriented(target_orientation) and self.gripper_is_at(
                        target_location) and not self.gripper_is_open:
                    # Finally, check if object is held
                    grabbed = self.has_object(gripper_position=self.get_position(), object_position=target_location)

                if grabbed:
                    grab_offset = target_location - self.gripper_position
                    self.grabbed_objects[target_name] = grab_offset

        else:
            # For service calls, check current state
            # Update gripper and object positions first
            self.get_position()

            # Simulating a sensor inside the gripper that assesses if the object is held using ground truth
            gt_object_location = None
            if target_name in self._robot_interactors.perception.objects:
                gt_object_location = self._robot_interactors.perception.objects[target_name]['location']

            at_target = self.gripper_is_at(gt_object_location) if gt_object_location is not None else False
            oriented = self.is_oriented(target_orientation)

            if at_target and oriented and not self.gripper_is_open:
                grabbed = self.has_object(gripper_position=self.get_position(), object_position=gt_object_location)

            # Reset flag to allow re-approach if sanity check fails
            if not grabbed:
                self.initial_approach = True
                self.grabbed_objects.clear()

        # Determine CoS & CoF condition
        cos_condition = grabbed
        cof_condition = (failure_reason is not None)

        state_data = {
            'grabbed': grabbed,
            'at_target': self.gripper_is_at(target_location),
            'oriented': self.is_oriented(target_orientation),
            'gripper_open': self.is_open,
            'failure_reason': failure_reason
        }

        # Update shared state with final grab result
        self._update_and_publish_state(state_data, grabbed, cof_condition, requesting_behavior)

        return cos_condition, cof_condition, self.get_position(), motor_cmd_orient, motor_cmd_reach

    def normalize_orientation(self):
        """Normalize orientation angles to the range [-π, π]."""
        self.gripper_orientation = (self.gripper_orientation + torch.pi) % (2 * torch.pi) - torch.pi

    def get_position(self):
        """Get the current gripper position."""
        robot_position = self._get_robot_position()
        self.gripper_position = robot_position + self.gripper_offset

        self._update_grabbed_object_positions()

        return self.gripper_position.clone()

    def _update_grabbed_object_positions(self):
        """Update positions of grabbed objects to follow gripper"""
        for object_name, grab_offset in self.grabbed_objects.items():
            # Update in perception.objects (the main object registry)
            if hasattr(self, '_robot_interactors') and object_name in self._robot_interactors.perception.objects:
                new_position = self.gripper_position
                self._robot_interactors.perception.objects[object_name]['location'] = new_position
                self.grabbed_objects[object_name] = new_position

            # Update in perception.target_states (if it exists there too)
            if hasattr(self, '_robot_interactors') and object_name in self._robot_interactors.perception.target_states:
                new_position = self.gripper_position
                self._robot_interactors.perception.target_states[object_name]['location'] = new_position

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
        return self.gripper_is_open

    def open_gripper(self):
        """Open the gripper."""
        self.gripper_is_open = True

        # Release any grabbed objects
        self.grabbed_objects.clear()

        return self.gripper_is_open

    def close_gripper(self):
        """Close the gripper."""
        self.gripper_is_open = False
        return self.gripper_is_open

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

        if not self.gripper_is_open and proximity_check:
            self._has_object = True
        else:
            self._has_object = False

        return self._has_object

    def is_target_grabbed(self, target_name):
        """Check if a specific target is currently grabbed."""
        return target_name in self.grabbed_objects

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

    def gripper_is_at(self, target_location, thresh: float = 0.01) -> bool:
        """Arrival test within threshold."""
        if target_location is None:
            return False
        threshold = self.stop_threshold if thresh is None else float(thresh)

        return torch.norm(target_location - self.gripper_position) <= threshold

    def gripper_move_towards(self, target_location):
        """
        Plan and execute a single bounded step toward target.
        Returns the applied motor command tensor [dx, dy, dz] or None if arrived/invalid.
        """
        if target_location is None:
            return None

        delta = target_location - self.gripper_position
        dist = torch.norm(delta)

        if dist.item() == 0.0:
            return None

        # Proportional step with clamp to max_speed and no overshoot
        direction = delta / dist
        step_mag = min(self.max_speed, self.gain * float(dist), float(dist))
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
        self.gripper_offset = torch.tensor([0.0, 0.0, 0.5])  # gripper is slightly above base
        self.gripper_position = robot_position + self.gripper_offset
        self.grabbed_objects.clear()
        self.initial_approach = True
        self._has_object = False

        self.wait_counter = 0  # for simulating object grasp delay
