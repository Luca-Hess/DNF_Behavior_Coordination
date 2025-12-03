from Luca_MSc.Dynamic_Behavior_Manager.DNF_interactors.robot_interactors import BaseInteractor
from interactor_config import INTERACTOR_CONFIG

import torch

def create_dynamic_method(method_name, method_config):
    """Create continuous and service methods dynamically"""
    
    def continuous_method(self, *args, requesting_behavior):  # Make requesting_behavior positional
        # Call the internal method
        internal_method = getattr(self, method_config['internal_method'])
        result = internal_method(*args)
        
        # Build state data
        state_data = method_config['state_builder'](self, *args, result)
        
        # Determine CoS condition
        cos_condition = method_config['cos_condition'](result)
        
        # Update and publish state
        self._update_and_publish_state(state_data, cos_condition, requesting_behavior)
        
        return result
    
    def service_method(self, *args):
        # Call the internal method (same logic, no publishing)
        internal_method = getattr(self, method_config['internal_method'])
        result = internal_method(*args)
        
        # Build state data
        state_data = method_config['state_builder'](self, *args, result)
        
        # Determine CoS condition  
        cos_condition = method_config['cos_condition'](result)
        
        # Update state without publishing
        self._update_and_publish_state(state_data, cos_condition, None)
        
        return result
    
    # Set method names
    continuous_method.__name__ = f"{method_name}_continuous"
    service_method.__name__ = f"{method_name}_service"
    
    return continuous_method, service_method

def add_dynamic_methods(interactor_class, interactor_type):
    """Add dynamic methods to interactor class"""
    config = INTERACTOR_CONFIG[interactor_type]
    
    for method_name, method_config in config.get('methods', {}).items():
        continuous_method, service_method = create_dynamic_method(method_name, method_config)
        
        # Add methods to class
        setattr(interactor_class, f"{method_name}_continuous", continuous_method)
        setattr(interactor_class, f"{method_name}_service", service_method)

# Apply dynamic methods to all interactor classes
class DynamicPerceptionInteractor(BaseInteractor):
    def __init__(self, get_robot_position=None, **kwargs):
        config = INTERACTOR_CONFIG['perception']['init_params']
        config.update(kwargs)
        super().__init__(**config)
        
        # Perception-specific initialization
        self.objects = {}
        self.search_attempts = 0
        self._get_robot_position = get_robot_position
        self.tracking_loss_probability = config['tracking_loss_prob']
        self.max_tracking_loss_duration = config['max_tracking_loss_duration']
        self.in_tracking_loss = False
        self.target_states = {}
    
        
    def _find_object_internal(self, name):
        """Internal object finding logic"""
        if name in self.objects:
            return True, self.objects[name]['location'], self.objects[name]['angle']
        return False, None, None
    
    def register_object(self, name, location, angle=torch.tensor([0.0, 0.0, 0.0])):
        """Register or update an object's location and angle."""
        self.objects[name] = {
            'location': location,
            'angle': angle,
            'lost_counter': 0
        }

# Add dynamic methods
add_dynamic_methods(DynamicPerceptionInteractor, 'perception')

class DynamicMovementInteractor(BaseInteractor):
    def __init__(self, **kwargs):
        config = INTERACTOR_CONFIG['movement']['init_params']
        config.update(kwargs)
        super().__init__(**config)
        
        # Movement-specific initialization
        self.robot_position = torch.tensor([0.0, 0.0, 0.0])
        self.max_speed = config['max_speed']
        self.gain = config['gain'] 
        self.stop_threshold = config['stop_threshold']

    def _move_to_internal(self, target_location):
        """Internal movement logic"""
        self.move_towards(target_location)
        arrived = self.is_at(target_location)
        position = self.get_position()
        return arrived, position
    
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

    # def move_robot(self, motor_commands):
    #     """
    #     Backward‑compat: directly apply provided motor delta (planar).
    #     Returns updated position.
    #     """
    #     if motor_commands is not None:
    #         if len(motor_commands) >= 2:
    #             if not torch.is_tensor(motor_commands):
    #                 motor_commands = torch.tensor(motor_commands, dtype=torch.float32)
    #             self.robot_position[:2] += motor_commands[:2]
    #     return self.robot_position.clone()



# Add dynamic methods
add_dynamic_methods(DynamicMovementInteractor, 'movement')


class DynamicGripperInteractor(BaseInteractor):
    def __init__(self, get_robot_position=None, **kwargs):
        # Load default config and merge with kwargs
        from interactor_config import INTERACTOR_CONFIG
        config = INTERACTOR_CONFIG['gripper']['init_params']
        config.update(kwargs)
        super().__init__(**config)
        
        # Gripper-specific initialization
        self.max_speed = config['max_speed']
        self.gain = config['gain']
        self.stop_threshold = config['stop_threshold']
        self.orient_threshold = config['orient_threshold']
        self.max_reach = config['max_reach']
        self.gripper_is_open = config['is_open']
        self._has_object = config['has_object_state']
        
        self.gripper_orientation = torch.tensor([0.0, 0.0, 0.0])
        self.normalize_orientation()
        self.initial_approach = True
        self.wait_counter = 0
        self.grabbed_objects = {}
        
        # Robot position reference
        if get_robot_position is None:
            raise ValueError("GripperInteractor requires a get_robot_position callable")
        self._get_robot_position = get_robot_position
        
        # Initial gripper position
        robot_position = self._get_robot_position().clone()
        self.gripper_offset = torch.tensor([0.0, 0.0, 0.5])
        self.gripper_position = robot_position + self.gripper_offset

    # ============ INTERNAL IMPLEMENTATION METHODS ============
    
    def _reach_check_internal(self, target_location):
        """Internal reachability check logic"""
        reachable = self.gripper_can_reach(target_location)
        position = self.get_position()
        return reachable, position
    
    def _reach_for_internal(self, target_location):
        """Internal reach logic"""
        motor_cmd = self.gripper_move_towards(target_location)
        at_target = self.gripper_is_at(target_location)
        position = self.get_position()
        return at_target, position, motor_cmd
    
    def _grab_internal(self, target_name, target_location, target_orientation):
        """Internal grab sequence logic"""
        motor_cmd_orient = None
        motor_cmd_reach = None
        grabbed = False

        # First orient gripper to target if needed
        if not self.is_oriented(target_orientation):
            motor_cmd_orient = self.gripper_rotate_towards(target_orientation)
            return grabbed, self.get_position(), motor_cmd_orient, motor_cmd_reach

        # Open gripper if needed, then reach
        if not self.gripper_is_open and self.initial_approach:
            self.open_gripper()
            self.initial_approach = False
            return grabbed, self.get_position(), motor_cmd_orient, motor_cmd_reach

        # Move gripper to target location
        if not self.gripper_is_at(target_location):
            motor_cmd_reach = self.gripper_move_towards(target_location)
            return grabbed, self.get_position(), motor_cmd_orient, motor_cmd_reach

        # Once at target, close gripper
        if self.gripper_is_open:
            self.close_gripper()
            return grabbed, self.get_position(), motor_cmd_orient, motor_cmd_reach
        
        # Finally, check if object is held
        grabbed = self.has_object(gripper_position=self.get_position(), object_position=target_location)

        if grabbed:
            # Store relative offset when object is grabbed
            grab_offset = target_location - self.gripper_position
            self.grabbed_objects[target_name] = grab_offset

        return grabbed, self.get_position(), motor_cmd_orient, motor_cmd_reach

    # ============ GRIPPER-SPECIFIC UTILITY METHODS ============
    
    def gripper_can_reach(self, target_location):
        """Check if gripper can reach the target location."""
        if target_location is None:
            return False
        robot_position = self._get_robot_position()
        offset = target_location - robot_position
        distance = torch.norm(offset)
        return distance <= self.max_reach

    def normalize_orientation(self):
        """Normalize orientation angles to the range [-π, π]."""
        self.gripper_orientation = (self.gripper_orientation + torch.pi) % (2 * torch.pi) - torch.pi

    def get_position(self):
        """Get the current gripper position with automatic object following."""
        robot_position = self._get_robot_position()
        self.gripper_position = robot_position + self.gripper_offset
        self._update_grabbed_object_positions()
        return self.gripper_position.clone()

    def _update_grabbed_object_positions(self):
        """Update positions of grabbed objects to follow gripper"""
        for object_name, grab_offset in self.grabbed_objects.items():
            new_position = self.gripper_position + grab_offset
            
            # Update in perception.objects
            if hasattr(self, '_robot_interactors') and object_name in self._robot_interactors.perception.objects:
                self._robot_interactors.perception.objects[object_name]['location'] = new_position
            
            # Update in perception.target_states
            if hasattr(self, '_robot_interactors') and object_name in self._robot_interactors.perception.target_states:
                self._robot_interactors.perception.target_states[object_name]['location'] = new_position

    def get_orientation(self):
        """Get the current gripper orientation."""
        return self.gripper_orientation.clone()

    def is_oriented(self, target_orientation, thresh: float = None) -> bool:
        """Check if gripper is oriented correctly."""
        if target_orientation is None:
            return True
        threshold = self.orient_threshold if thresh is None else float(thresh)
        
        orientation_diff = torch.abs(target_orientation - self.gripper_orientation)
        orientation_diff = torch.min(orientation_diff, 2 * torch.pi - orientation_diff)
        return bool(torch.all(orientation_diff <= threshold))

    def is_open(self):
        """Check if gripper is open."""
        return self.gripper_is_open

    def open_gripper(self):
        """Open the gripper and release objects."""
        self.gripper_is_open = True
        self.grabbed_objects.clear()
        return self.gripper_is_open

    def close_gripper(self):
        """Close the gripper."""
        self.gripper_is_open = False
        return self.gripper_is_open

    def has_object(self, gripper_position=None, object_position=None):
        """Check if gripper is holding an object."""
        if gripper_position is None:
            gripper_position = self.get_position()

        proximity_check = False
        if object_position is not None:
            distance = torch.norm(gripper_position - object_position)
            proximity_check = distance <= 0.1

        if not self.gripper_is_open and proximity_check:
            self._has_object = True
        else:
            self._has_object = False

        return self._has_object

    def gripper_rotate_towards(self, target_orientation):
        """Rotate gripper towards target orientation."""
        if target_orientation is None or self.is_oriented(target_orientation):
            return None

        delta = target_orientation - self.gripper_orientation
        delta = (delta + torch.pi) % (2 * torch.pi) - torch.pi
        rotation_step = torch.clamp(self.gain * delta, -self.max_speed, self.max_speed)
        
        self.gripper_orientation += rotation_step
        self.gripper_orientation = (self.gripper_orientation + torch.pi) % (2 * torch.pi) - torch.pi
        
        return rotation_step

    def gripper_is_at(self, target_location, thresh: float = None) -> bool:
        """Check if gripper is at target location."""
        if target_location is None:
            return False
        threshold = self.stop_threshold if thresh is None else float(thresh)
        return torch.norm(target_location - self.gripper_position) <= threshold

    def gripper_move_towards(self, target_location):
        """Move gripper towards target location."""
        if target_location is None:
            return None

        delta = target_location - self.gripper_position
        dist = torch.norm(delta)
        
        if dist.item() == 0.0:
            return None
            
        direction = delta / dist
        step_mag = min(self.max_speed, self.gain * float(dist), float(dist))
        step_vec = direction * step_mag

        # Update gripper offset with reach constraints
        robot_position = self._get_robot_position()
        new_offset = self.gripper_offset + step_vec

        if torch.norm(new_offset) <= self.max_reach:
            self.gripper_offset = new_offset
        else:
            normalized_offset = new_offset / torch.norm(new_offset)
            self.gripper_offset = normalized_offset * self.max_reach

        self.gripper_position = robot_position + self.gripper_offset
        return step_vec

    def reset(self):
        """Reset gripper state."""
        super().reset()
        robot_position = self._get_robot_position().clone()
        self.gripper_offset = torch.tensor([0.0, 0.0, 0.5])
        self.gripper_position = robot_position + self.gripper_offset
        self.grabbed_objects.clear()
        self.initial_approach = True
        self._has_object = False

add_dynamic_methods(DynamicGripperInteractor, 'gripper')

class StateInteractor(BaseInteractor):
    def __init__(self, perception_interactor, movement_interactor, gripper_interactor, **kwargs):
        config = INTERACTOR_CONFIG['state']['init_params']
        config.update(kwargs)
        super().__init__(**config)
        
        self.perception = perception_interactor
        self.movement = movement_interactor
        self.gripper = gripper_interactor

        # Dynamic target state - populated based on behavior chain
        self.behavior_targets = {}  # {behavior_name: current_target_name}
        self.target_args = {}       # Store original args for reference
    
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
    
    def update_behavior_target(self, behavior_name, new_target_name):
        """Update target for a specific behavior"""
        old_target = self.behavior_targets.get(behavior_name)
        self.behavior_targets[behavior_name] = new_target_name
        
        return True, new_target_name
    
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

add_dynamic_methods(StateInteractor, 'state')

class DynamicRobotInteractors:
    """Dynamic facade using config-based interactors"""

    def __init__(self):
        self.movement = DynamicMovementInteractor()
        self.gripper = DynamicGripperInteractor(get_robot_position=self.movement.get_position)
        self.perception = DynamicPerceptionInteractor(get_robot_position=self.movement.get_position)
        self.state = StateInteractor(self.perception, self.movement, self.gripper)

        # Allow interactors to reference back to this facade
        self.movement._robot_interactors = self
        self.gripper._robot_interactors = self
        self.perception._robot_interactors = self

    def reset(self):
        """Reset all interactors."""
        self.perception.reset()
        self.movement.reset()
        self.gripper.reset()
        self.state.behavior_targets.clear()
        self.state.target_args.clear()