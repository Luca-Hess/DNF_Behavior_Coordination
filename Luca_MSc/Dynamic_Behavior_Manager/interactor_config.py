import torch

INTERACTOR_CONFIG = {
    'perception': {
        'class': 'PerceptionInteractor',
        'methods': {
            'find_object': {
                'internal_method': '_find_object_internal',
                'cos_condition': lambda result: result[0],  # target_found
                'state_builder': lambda self, name, result: {
                    'target_name': name,
                    'found': result[0],
                    'location': result[1],
                    'angle': result[2]
                },
                'shared_state_key': lambda name: name,  # Store by target name
                'args': ['name']
            }
        }, # paremeters only used for this simulation
        'init_params': {
            'tracking_loss_prob': 0.3,
            'max_tracking_loss_duration': 10
        }
    },
    
    'movement': {
        'class': 'MovementInteractor',
        'methods': {
            'move_to': {
                'internal_method': '_move_to_internal',
                'cos_condition': lambda result: result[0],  # arrived
                'state_builder': lambda self, target_location, result: {
                    'target_location': target_location,
                    'arrived': result[0],
                    'distance': self.calculate_distance(target_location)
                },
                'args': ['target_location']
            }
        },
        'init_params': {
            'max_speed': 0.1,
            'gain': 1.0,
            'stop_threshold': 0.1
        }
    },
    
    'gripper': {
        'class': 'GripperInteractor',
        'methods': {
            'reach_check': {
                'internal_method': '_reach_check_internal',
                'cos_condition': lambda result: result[0],  # reachable
                'state_builder': lambda self, target_location, result: {
                    'target_location': target_location,
                    'reachable': result[0],
                    'distance': float(torch.norm(target_location - self.gripper_position)) if target_location is not None else float('inf')
                },
                'args': ['target_location']
            },
            'reach_for': {
                'internal_method': '_reach_for_internal', 
                'cos_condition': lambda result: result[0],  # at_target
                'state_builder': lambda self, target_location, result: {
                    'target_location': target_location,
                    'at_target': result[0],
                    'motor_cmd': result[2]
                },
                'args': ['target_location']
            },
            'grab': {
                'internal_method': '_grab_internal',
                'cos_condition': lambda result: result[0],  # grabbed
                'state_builder': lambda self, target_name, target_location, target_orientation, result: {
                    'target_name': target_name,
                    'target_location': target_location,
                    'target_orientation': target_orientation,
                    'grabbed': result[0],
                    'at_target': self.gripper_is_at(target_location),
                    'oriented': self.is_oriented(target_orientation),
                    'gripper_open': self.gripper_is_open
                },
                'args': ['target_name', 'target_location', 'target_orientation']
            }
        },
        'init_params': {
            'max_speed': 0.1,
            'gain': 1.0,
            'stop_threshold': 0.01,
            'orient_threshold': 0.01,
            'max_reach': 2.5,
            'is_open': False,
            'has_object_state': False
        }
    },
    
    'state': {
        'class': 'StateInteractor',
        'methods': {
            # State methods don't follow the same pattern
        },
        'init_params': {}
    }
}