import torch
import random
import time

random.seed(2) #seed 2 has no collapses for find, 3 has two


class BaseInteractor:
    """Base class for all interactors with common pub/sub and state management"""
    
    def __init__(self, **kwargs):
        # Common pub/sub infrastructure
        self.cos_subscribers = {}  # {behavior_name: callback}
        self.cof_subscribers = {}  # {behavior_name: callback}
        self.shared_states = {}  # Shared state across behaviors
        
        # Store initialization parameters
        self.params = kwargs
        
    def subscribe_cos_updates(self, behavior_name, callback):
        """Register a behavior to receive CoS updates"""
        self.cos_subscribers[behavior_name] = callback
        
    def publish_cos_state(self, behavior_name, cos_value):
        """Publish CoS state to subscribed behavior"""
        if behavior_name in self.cos_subscribers:
            self.cos_subscribers[behavior_name](cos_value)

    def subscribe_cof_updates(self, behavior_name, callback):
        """Register a behavior to receive CoF updates"""
        self.cof_subscribers[behavior_name] = callback

    def publish_cof_state(self, behavior_name, cof_value):
        """Publish CoF state to subscribed behavior"""
        if behavior_name in self.cof_subscribers:
            self.cof_subscribers[behavior_name](cof_value)
    
    def _update_and_publish_state(self, state_data, cos_condition, cof_condition=False, requesting_behavior=None):
        """Generic state update and CoS publishing"""
        cos_value = 5.0 if cos_condition else 0.0
        cof_value = 5.0 if cof_condition else 0.0
        
        if requesting_behavior:
            # Add timestamp and store in shared state
            state_data['last_updated'] = time.time()
            state_data['cos_value'] = cos_value
            state_data['cof_value'] = cof_value
            self.shared_states[requesting_behavior] = state_data
            
            # Publish CoS for continuous calls
            self.publish_cos_state(requesting_behavior, cos_value)
            self.publish_cof_state(requesting_behavior, cof_value)
        # For service calls, just return the data without publishing
        
        return cos_condition, cof_condition, state_data
    
    def reset(self):
        """Reset interactor state - to be overridden"""
        self.shared_states.clear()
        self.cos_subscribers.clear()
        self.cof_subscribers.clear()

class PerceptionInteractor(BaseInteractor):
    def __init__(self, get_robot_position=None, **kwargs):
        super().__init__(**kwargs)

        self.objects = {}
        self.target_states = {}
        self._get_robot_position = get_robot_position

        # Simulation parameters for object finding behavior
        self.search_attempts = 0
        self.tracking_loss_probability = 0.1  # Probability to lose tracking on each
        self.max_tracking_loss_duration = 5   # Max steps to lose tracking
        self.in_tracking_loss = False
        self.tracking_loss_remaining = 0
        self.max_search_attempts = kwargs.get('max_search_attempts', 50) # Max attempts before reporting failure

    def find_object(self, name, requesting_behavior=None):
        """Unified find object method - behaves differently based on requesting_behavior"""
        target_found, location, angle, failure_reason = self._find_object_internal(name)

        # Determine CoF condition
        cof_condition = (failure_reason is not None)

        state_data = {
            'target_name': name,
            'target_found': target_found,
            'target_location': location,
            'target_angle': angle,
            'failure_reason': failure_reason
        }
        
        # Use base class helper for state management and publishing
        self._update_and_publish_state(state_data, target_found, cof_condition, requesting_behavior)
        
        return target_found, location, angle
        
    def _find_object_internal(self, name):
        """Internal object finding logic"""
        self.search_attempts += 1
        failure_reason = None

        # Check for search timeout failure
        if self.search_attempts > self.max_search_attempts:
            failure_reason = f"Search timeout: {name} not found after {self.max_search_attempts} attempts"
            return False, None, None, failure_reason

        # Check if we should start a tracking loss period
        if not self.in_tracking_loss and random.random() < self.tracking_loss_probability:
            self.in_tracking_loss = True
            self.tracking_loss_remaining = random.randint(1, self.max_tracking_loss_duration)

        # If we're in a tracking loss period
        if self.in_tracking_loss:
            self.tracking_loss_remaining -= 1
            if self.tracking_loss_remaining <= 0:
                self.in_tracking_loss = False
            return False, None, None, None

        # Normal detection logic
        if name in self.objects and self.search_attempts >= 3:
            return True, self.objects[name]['location'], self.objects[name], None
        return False, None, None, None
    
    def register_object(self, name, location, angle=torch.tensor([0.0, 0.0, 0.0])):
        """Register or update an object's location and angle."""
        self.objects[name] = {
            'location': location,
            'angle': angle,
            'lost_counter': 0
        }

    def reset(self):
        """Reset perception state."""
        super().reset()
        self.target_states = {}
    

class MovementInteractor(BaseInteractor):
    """Handles robot position, movement, and spatial calculations."""

    def __init__(
            self,
            max_speed = 0.1,        # max speed per step
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
        
        # Determine CoF condition
        cof_condition = (failure_reason is not None)

        if failure_reason is not None:
            print(f"MovementInteractor.move_to failure: {failure_reason}")

        state_data = {
            'target_location': target_location,
            'arrived': arrived,
            'distance': self.calculate_distance(target_location),
            'position': position,
            'move_attempts': self.move_attempts,
            'failure_reason': failure_reason
        }
        
        # Use base class helper for state management and publishing
        self._update_and_publish_state(state_data, arrived, cof_condition, requesting_behavior)
        
        return arrived, position
                
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

        motor_cmd = torch.tensor([step_vec_2d[0].item(), step_vec_2d[1].item(), 0.0])
        return motor_cmd


    def reset(self):
        """Reset movement state."""
        self.robot_position = torch.tensor([0.0, 0.0, 0.0])

class GripperInteractor(BaseInteractor):
    """Handles gripper position and movement."""

    def __init__(
            self,
            max_speed = 0.1,            # max speed per step
            gain = 1.0,
            stop_threshold = 0.01,      # distance to consider "arrived"
            orient_threshold = 0.01,    # radians to consider "oriented"
            max_reach = 3,            # max reach from robot base
            get_robot_position = None,  # anchor arm to robot base
            is_open = False,            # gripper state
            has_object_state = False,   # whether gripper is holding an object
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
        self.gripper_offset = torch.tensor([0.0, 0.0, 0.5]) # gripper is slightly above base
        self.gripper_position = robot_position + self.gripper_offset


    def gripper_can_reach(self, target_location):
        """Check if gripper can reach the target location."""
        if target_location is None:
            return False

        robot_position = self._get_robot_position()
        offset = target_location - robot_position
        distance = torch.norm(offset)

        return distance <= self.max_reach
            
    def reach_check(self, target_location, requesting_behavior=None):
        """Continuous publisher for active behaviors"""
        reachable = self.gripper_can_reach(target_location)
        position = self.get_position()

        # Determine failure condition
        failure_reason = None
        if not reachable:
            failure_reason = f"Target location {target_location} is out of gripper reach (max {self.max_reach})."

        cof_condition = (failure_reason is not None)

        distance = float(torch.norm(target_location - self.gripper_position)) if target_location is not None else float('inf')

        state_data = {
            'target_location': target_location,
            'reachable': reachable,
            'distance': distance,
            'failure_reason': failure_reason
        }

        self._update_and_publish_state(state_data, reachable, cof_condition, requesting_behavior)

        return reachable, position


    def reach_for(self, target_location, requesting_behavior=None):   
        """Continuous publisher for active behaviors"""
        motor_cmd = None
        failure_reason = None

        # For continous calls, actually move the gripper
        if requesting_behavior:
            if not self.gripper_can_reach(target_location):
                failure_reason = f"Target location {target_location} is out of gripper reach (max {self.max_reach})."
            else:
                motor_cmd = self.gripper_move_towards(target_location)

        at_target = self.gripper_is_at(target_location)
        position = self.get_position()

        cof_condition = (failure_reason is not None)

        state_data = {
            'target_location': target_location,
            'at_target': at_target,
            'motor_command': motor_cmd,
            'failure_reason': failure_reason
        }

        self._update_and_publish_state(state_data, at_target, cof_condition, requesting_behavior)
        
        return at_target, position, motor_cmd
    

    
    ## Gripper interactors for grabbing - consists of orient, open, fine_reach, close, has_object check (as final CoS driver)
    def grab(self, target_name, target_location, target_orientation, requesting_behavior=None):
        """Continuous publisher for active behaviors - full grab sequence"""
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
        
                # Finally, check if object is held
                grabbed = self.has_object(gripper_position=self.get_position(), object_position=target_location)

                if grabbed:
                    grab_offset = target_location - self.gripper_position
                    self.grabbed_objects[target_name] = grab_offset

        else:
            # For service calls, check current state
            at_target = self.gripper_is_at(target_location)
            oriented = self.is_oriented(target_orientation)

            if at_target and oriented and not self.gripper_is_open:
                grabbed = self.has_object(gripper_position=self.get_position(), object_position=target_location)

        cof_condition = (failure_reason is not None)

        state_data = {
            'target_name': target_name,
            'target_location': target_location,
            'target_orientation': target_orientation,
            'grabbed': grabbed,
            'at_target': self.gripper_is_at(target_location),
            'oriented': self.is_oriented(target_orientation),
            'gripper_open': self.is_open,
            'failure_reason': failure_reason
        }

        # Update shared state with final grab result
        self._update_and_publish_state(state_data, grabbed, cof_condition, requesting_behavior)
    
        return grabbed, self.get_position(), motor_cmd_orient, motor_cmd_reach


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
            # self.wait_counter += 1
            # # Only set to True if both conditions are met
            # if self.wait_counter >= 10:
                self._has_object = True
        else:
            # Reset counter if gripper opens
            # self.wait_counter = 0
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
        self.gripper_offset = torch.tensor([0.0, 0.0, 0.5]) # gripper is slightly above base
        self.gripper_position = robot_position + self.gripper_offset
        self.grabbed_objects.clear()
        self.initial_approach = True
        self._has_object = False


class StateInteractor(BaseInteractor):
    """Interactor for managing world state transitions and updates"""
    
    def __init__(self, perception_interactor, movement_interactor, gripper_interactor):
        super().__init__()
        self.perception = perception_interactor
        self.movement = movement_interactor
        self.gripper = gripper_interactor

        # Dynamic target state - populated based on behavior chain
        self.behavior_targets = {}  # {behavior_name: current_target_name}
        self.target_args = {}       # Store original args for reference
        self.instance_targets = {} # {behavior_name: target_name} for instance-specific targets
        self.global_state = {}      # Global state info if needed
    
    def initialize_from_behavior_chain(self, behavior_chain, args):
        """Initialize targets dynamically based on the behavior chain and args"""
        self.target_args = args.copy()
        
        # Analyze behavior chain to determine target requirements
        for level in behavior_chain:
            behavior_name = level['name']
            interactor_type = level.get('interactor_type')
            
            # Set initial target based on behavior type and args
            if interactor_type == 'perception':
                # Perception behaviors (like 'find') use the primary target
                self.behavior_targets[behavior_name] = args.get('target_object')
                
            elif interactor_type in ['movement', 'gripper']:
                # Movement/gripper behaviors initially use the primary target
                self.behavior_targets[behavior_name] = args.get('target_object')
        
        return True, self.behavior_targets.copy()
    
    def update_behavior_target(self, behavior_name, new_target_name, requesting_behavior=None):
        """Unified method to update target for a specific behavior"""
        old_target = self.behavior_targets.get(behavior_name)
        self.behavior_targets[behavior_name] = new_target_name
                
        state_data = {
            'behavior_name': behavior_name,
            'old_target': old_target,
            'new_target': new_target_name
        }
        
        success = True
        self._update_and_publish_state(state_data, success, requesting_behavior)
        
        return success, new_target_name
    
    def update_multiple_behavior_targets(self, target_updates):
        """Update targets for multiple behaviors at once"""
        for behavior_name, new_target in target_updates.items():
            if behavior_name in self.behavior_targets:
                old_target = self.behavior_targets[behavior_name]
                self.behavior_targets[behavior_name] = new_target
        
        return True, self.behavior_targets.copy()
    
    def get_behavior_target_name(self, behavior_name):
        """Get the current target name for a behavior"""
        target_name = self.behavior_targets.get(behavior_name)
        return target_name
    
    def get_behavior_target_location(self, behavior_name):
        """Get the current target location for a behavior"""
        target_name = self.behavior_targets.get(behavior_name)
        if target_name and target_name in self.perception.target_states:
            location = self.perception.target_states[target_name]['location']
            return location
        
        elif target_name and target_name in self.perception.objects:
            location = self.perception.objects[target_name]['location']
            return location

        return None
    
    def get_behavior_target_info(self, behavior_name):
        """Get full target info (location, angle) for a behavior"""
        target_name = self.behavior_targets.get(behavior_name)
        if target_name and target_name in self.perception.target_states:
            info = self.perception.target_states[target_name]
            location = info.get('location')
            angle = info.get('angle')
            return target_name, location, angle
        
        elif target_name and target_name in self.perception.objects:
            info = self.perception.objects[target_name]
            location = info.get('location')
            angle = info.get('angle')
            return target_name, location, angle

        return None, None, None

    def get_instance_target(self, instance_id):
        """Get the instance-specific target for a behavior"""
        target_name = self.instance_targets.get(instance_id)
        return target_name
    
    def set_instance_target(self, instance_id, target_info):
        """Set the instance-specific target for a behavior"""
        self.instance_targets[instance_id] = target_info
        return True, target_info


class RobotInteractors:
    """Facade that provides access to all interactors."""

    def __init__(self):
        self.movement = MovementInteractor()
        self.gripper = GripperInteractor(get_robot_position=self.movement.get_position)
        self.perception = PerceptionInteractor(get_robot_position=self.movement.get_position)
        self.state = StateInteractor(self.perception, self.movement, self.gripper)

        # Allow interactors to reference back to this facade if needed
        self.movement._robot_interactors = self
        self.gripper._robot_interactors = self
        self.perception._robot_interactors = self

    def reset(self):
        """Reset all interactors."""
        self.perception.reset()
        self.movement.reset()
        self.gripper.reset()