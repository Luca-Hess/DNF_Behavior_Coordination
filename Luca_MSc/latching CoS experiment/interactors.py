import torch
import random
import time

random.seed(2) #seed 2 has no collapses for find, 3 has two

class PerceptionInteractor:
    def __init__(self, tracking_loss_prob=0.3, max_tracking_loss_duration=10, get_robot_position=None):
        self.objects = {}
        self.search_attempts = 0
        self._get_robot_position = get_robot_position

        self.tracking_loss_probability = tracking_loss_prob
        self.max_tracking_loss_duration = max_tracking_loss_duration
        self.in_tracking_loss = False
        
        # Pub/Sub state
        self.subscribers = {}  # {behavior_name: callback}
        self.target_states = {}  # Shared state across all behaviors
        
    def subscribe_cos_updates(self, behavior_name, callback):
        """Register a behavior to receive CoS updates"""
        self.subscribers[behavior_name] = callback
        
    def publish_cos_state(self, behavior_name, cos_value):
        """Publish CoS state to subscribed behavior"""
        if behavior_name in self.subscribers:
            self.subscribers[behavior_name](cos_value)
            
    def find_object_service(self, name):
        """Service call for sanity checks - one-time query"""
        target_found, location, angle = self._find_object_internal(name)
        self._update_and_publish_state(name, target_found, location, angle)
        
        return target_found, location, angle
        
    def find_object_continuous(self, name, requesting_behavior):
        """Continuous publisher for active behaviors"""
        target_found, location, angle = self._find_object_internal(name)
        self._update_and_publish_state(name, target_found, location, angle, requesting_behavior)
        
        return target_found, location, angle
    
    def _update_and_publish_state(self, name, target_found, location, angle, requesting_behavior=None):
        """Common logic for updating shared state and publishing CoS"""
        
        # Update shared target state for other interactors
        self.target_states[name] = {
            'found': target_found,
            'location': location,
            'angle': angle,
            'last_updated': time.time()
        }
        
        # Publish CoS update - same logic for both continuous and service
        cos_value = 5.0 if target_found else 0.0
        
        if requesting_behavior:
            # Continuous call - publish to specific requesting behavior
            self.publish_cos_state(requesting_behavior, cos_value)
        else:
            # Service call
            # For sanity checks, the check behavior will handle CoS publishing separately
            pass
        
    def _find_object_internal(self, name):
        """Internal object finding logic"""
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
            return False, None, None
        
        # Normal detection logic
        if name in self.objects and self.search_attempts >= 3:
            return True, self.objects[name]['location'], self.objects[name]['angle']
        return False, None, None

    def register_object(self, name, location, angle=torch.tensor([0.0, 0.0, 0.0])):
        """Register or update an object's location and angle."""
        self.objects[name] = {
            'location': location,
            'angle': angle,
            'lost_counter': 0
        }


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

        # Pub/Sub state
        self.subscribers = {}  # {behavior_name: callback}
        self.movement_states = {}  # Shared state across behaviors
        
    def subscribe_cos_updates(self, behavior_name, callback):
        """Register a behavior to receive CoS updates"""
        self.subscribers[behavior_name] = callback
        
    def publish_cos_state(self, behavior_name, cos_value):
        """Publish CoS state to subscribed behavior"""
        if behavior_name in self.subscribers:
            self.subscribers[behavior_name](cos_value)
            
    def move_to_service(self, target_location):
        """Service call for sanity checks - one-time query"""
        arrived, position = self._check_arrival_internal(target_location)
        self._update_and_publish_state(target_location, arrived, position)
        return arrived, position
        
    def move_to_continuous(self, target_location, requesting_behavior):
        """Continuous publisher for active behaviors"""
        # Execute movement step
        self.move_towards(target_location)
        arrived, position = self._check_arrival_internal(target_location)
        self._update_and_publish_state(target_location, arrived, position, requesting_behavior)
        return arrived, position
    
    def _update_and_publish_state(self, target_location, arrived, position, requesting_behavior=None):
        """Common logic for updating shared state and publishing CoS"""

        # Determine CoS value based on arrival
        cos_value = 5.0 if arrived else 0.0

        # Update shared state
        if requesting_behavior:
            self.movement_states[requesting_behavior] = {
                'target_location': target_location,
                'arrived': arrived,
                'distance': self.calculate_distance(target_location),
                'last_updated': time.time()
            }
        
            # Continuous call - publish to specific requesting behavior
            self.publish_cos_state(requesting_behavior, cos_value)

        else:
            # Service call
            # For sanity checks, the check behavior will handle CoS publishing separately
            pass
                
    def _check_arrival_internal(self, target_location):
        """Internal arrival check logic"""
        arrived = self.is_at(target_location)
        return arrived, self.get_position()

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

        # Pub/Sub state
        self.subscribers = {}  # {behavior_name: callback}
        self.gripper_states = {}  # Shared state across behaviors

        # Robot position reference
        if get_robot_position is None:
            raise ValueError("GripperInteractor requires a get_robot_position callable")
        self._get_robot_position = get_robot_position

        # Initial gripper position (at robot base)
        robot_position = self._get_robot_position().clone()
        self.gripper_offset = torch.tensor([0.0, 0.0, 0.5]) # gripper is slightly above base
        self.gripper_position = robot_position + self.gripper_offset

    def subscribe_cos_updates(self, behavior_name, callback):
        """Register a behavior to receive CoS updates"""
        self.subscribers[behavior_name] = callback
        
    def publish_cos_state(self, behavior_name, cos_value):
        """Publish CoS state to subscribed behavior"""
        if behavior_name in self.subscribers:
            self.subscribers[behavior_name](cos_value)

    def gripper_can_reach(self, target_location):
        """Check if gripper can reach the target location."""
        if target_location is None:
            return False

        robot_position = self._get_robot_position()
        offset = target_location - robot_position
        distance = torch.norm(offset)

        return distance <= self.max_reach
            
    def reach_check_continuous(self, target_location, requesting_behavior):
        """Continuous publisher for active behaviors"""
        reachable = self.gripper_can_reach(target_location)
        self._update_and_publish_state(target_location, reachable, requesting_behavior)
        return reachable, self.get_position()
    
    def reach_check_service(self, target_location):
        """Service call for sanity checks - one-time query"""
        reachable = self.gripper_can_reach(target_location)
        self._update_and_publish_state(target_location, reachable)
        return reachable, self.get_position()
    
    def _update_and_publish_state(self, target_location, reachable, requesting_behavior=None):
        """Common logic for updating shared state and publishing CoS"""

        # Determine CoS value based on reachability
        cos_value = 5.0 if reachable else 0.0

        # Update shared state
        if requesting_behavior:
            self.gripper_states[requesting_behavior] = {
                'target_location': target_location,
                'reachable': reachable,
                'distance': self.calculate_distance(target_location),
                'last_updated': time.time()
            }
            # Continuous call - publish to specific requesting behavior
            self.publish_cos_state(requesting_behavior, cos_value)

        else:
            # Service call
            # For sanity checks, the check behavior will handle CoS publishing separately
            pass

    def reach_for_continuous(self, target_location, requesting_behavior):   
        """Continuous publisher for active behaviors"""
        motor_cmd = self.gripper_move_towards(target_location)
        at_target = self.gripper_is_at(target_location)
        self._update_and_publish_state(target_location, at_target, requesting_behavior)
        return at_target, self.get_position(), motor_cmd
    
    def reach_for_service(self, target_location):
        """Service call for sanity checks - one-time query"""
        motor_cmd = self.gripper_move_towards(target_location)
        at_target = self.gripper_is_at(target_location)
        self._update_and_publish_state(target_location, at_target)
        return at_target, self.get_position(), motor_cmd
    
    ## Gripper interactors for grabbing - consists of orient, open, fine_reach, close, has_object check (as final CoS driver)
    def grab_continuous(self, target_location, target_orientation, requesting_behavior):
        """Continuous publisher for active behaviors - full grab sequence"""
        motor_cmd_orient = None
        motor_cmd_reach = None
        grabbed = False

        # First orient gripper to target if needed
        if not self.is_oriented(target_orientation):
            motor_cmd_orient = self.gripper_rotate_towards(target_orientation)

            # Update shared state - not grabbed yet, still orienting
            self._update_and_publish_grab_state(target_location, target_orientation, grabbed, requesting_behavior)
            return grabbed, self.get_position(), motor_cmd_orient, motor_cmd_reach

        # Open gripper if needed, then reach
        if not self.is_open and not self.gripper_is_at(target_location):
            self.open_gripper()
            # Update shared state - not grabbed yet, but oriented
            self._update_and_publish_grab_state(target_location, target_orientation, grabbed, requesting_behavior)
            return grabbed, self.get_position(), motor_cmd_orient, motor_cmd_reach

        # Move gripper to target location - fine reaching!
        if not self.gripper_is_at(target_location):
            motor_cmd_reach = self.gripper_move_towards(target_location)
            # Update shared state - not grabbed yet, still reaching
            self._update_and_publish_grab_state(target_location, target_orientation, grabbed, requesting_behavior)
            return grabbed, self.get_position(), motor_cmd_orient, motor_cmd_reach
        
        # Once at target, close gripper
        if self.is_open:
            self.close_gripper()
            # Update shared state - not grabbed yet, just closed
            self._update_and_publish_grab_state(target_location, target_orientation, grabbed, requesting_behavior)
            return grabbed, self.get_position(), motor_cmd_orient, motor_cmd_reach
        
        # Finally, check if object is held
        grabbed = self.has_object(gripper_position=self.get_position(), object_position=target_location)

        # Update shared state with final grab result
        self._update_and_publish_grab_state(target_location, target_orientation, grabbed, requesting_behavior)
    
        return grabbed, self.get_position(), motor_cmd_orient, motor_cmd_reach

    def grab_service(self, target_location, target_orientation):
        """Service call for sanity checks - one-time query"""
        # For service calls, just check current state without executing actions
        at_target = self.gripper_is_at(target_location)
        oriented = self.is_oriented(target_orientation)
        grabbed = False
        
        if at_target and oriented and not self.is_open:
            grabbed = self.has_object(gripper_position=self.get_position(), object_position=target_location)
        
        self._update_and_publish_grab_state(target_location, target_orientation, grabbed)
        return grabbed, self.get_position(), None, None

    def _update_and_publish_grab_state(self, target_location, target_orientation, grabbed, requesting_behavior=None):
        """Common logic for updating shared state and publishing CoS for grab operations"""
        
        # Determine CoS value based on successful grab
        cos_value = 5.0 if grabbed else 0.0
        
        # Update shared state
        if requesting_behavior:
            self.gripper_states[requesting_behavior] = {
                'target_location': target_location,
                'target_orientation': target_orientation,
                'grabbed': grabbed,
                'at_target': self.gripper_is_at(target_location),
                'oriented': self.is_oriented(target_orientation),
                'gripper_open': self.is_open,
                'has_object': self._has_object,
                'distance': self.calculate_distance(target_location),
                'last_updated': time.time()
            }
            # Continuous call - publish to specific requesting behavior
            self.publish_cos_state(requesting_behavior, cos_value)
        else:
            # Service call - check behavior will handle CoS publishing separately
            pass

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


class StateInteractor:
    """Interactor for managing world state transitions and updates"""
    
    def __init__(self, perception_interactor, movement_interactor, gripper_interactor):
        self.perception = perception_interactor
        self.movement = movement_interactor
        self.gripper = gripper_interactor
        self.subscribers = {}
    
    def subscribe_cos_updates(self, behavior_name, callback):
        """Subscribe to CoS state updates"""
        self.subscribers[behavior_name] = callback
    
    def publish_cos_state(self, behavior_name, cos_value):
        """Publish CoS state to subscribed behavior"""
        if behavior_name in self.subscribers:
            self.subscribers[behavior_name](cos_value)
    
    def update_target_location(self, target_name, new_location, requesting_behavior=None):
        """Update target location in shared state"""
        # Check if object "transport_target" is properly registered
        if target_name in self.perception.target_states:
            old_location = self.perception.target_states[target_name].get('location')
            self.perception.target_states[target_name]['location'] = new_location
            
            print(f"[STATE] Updated {target_name} location: {old_location} → {new_location}")
            
            return True, new_location
        return False, None
    
    def update_object_property(self, object_name, property_name, new_value, requesting_behavior=None):
        """Update object property in perception system"""
        if object_name in self.perception.objects:
            old_value = self.perception.objects[object_name].get(property_name)
            self.perception.objects[object_name][property_name] = new_value
            
            print(f"[STATE] Updated {object_name}.{property_name}: {old_value} → {new_value}")
            
            return True, new_value
        return False, None
    
    def announce_message(self, message, requesting_behavior=None):
        """Announce a message (could interface with speech system, logging, etc.)"""
        print(f"[ANNOUNCEMENT] {message}")

        return True, message


class RobotInteractors:
    """Facade that provides access to all interactors."""

    def __init__(self):
        self.movement = MovementInteractor()
        self.gripper = GripperInteractor(get_robot_position=self.movement.get_position)
        self.perception = PerceptionInteractor(get_robot_position=self.movement.get_position)
        self.state = StateInteractor(self.perception, self.movement, self.gripper)

    def reset(self):
        """Reset all interactors."""
        self.perception.reset()
        self.movement.reset()
        self.gripper.reset()