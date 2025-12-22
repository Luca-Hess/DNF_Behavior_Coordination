
##################################
# Define Behavior Configurations #
##################################
#
# Elementary Behaviors
#
# Interactor Types:
# Define which interactor type is used for each behavior (e.g., perception, movement, gripper)
#
# Continuous Method:
# Define which continuous interactor a behavior uses to get real-time updates or affect the world
#
# Service Method:
# Define which service interactor a behavior uses for discrete, one-time sanity checks
#
# Service Args Func:
# Related to the service method, this function generates the required arguments for the service call.
#
#
# Extending Elementary Behaviors
# Supersets of elementary behaviors that extend their functionality by adding post-success actions.


ELEMENTARY_BEHAVIOR_CONFIG = {
    'find': {
            'interactor_type': 'perception',
            'method': 'find_object',
            'required_args': ['target_object'],
    },
   'move':{
            'interactor_type': 'movement',
            'method': 'move_to',
            'required_args': ['target_object'],
   },
    'check_reach':{
            'interactor_type': 'gripper',
            'method': 'reach_check',
            'required_args': ['target_object'],
    },
    'reach_for':{
            'interactor_type': 'gripper',
            'method': 'reach_for',
            'required_args': ['target_object'],
    },
    'grab':{
            'interactor_type': 'gripper',
            'method': 'grab',
            'required_args': ['target_object'],
    }
}

# Action-only behaviors (only service calls, no continuous updates)
ACTION_BEHAVIOR_CONFIG = {
    'update_movement_target': {
            'interactor_type': 'state',
            'service_method': 'update_behavior_target',
            'service_args_func': lambda interactors, args, action: (
                'move_to',
                action['new_movement_target'](interactors, args),
            )
    }
}


# Composite/Extended behaviors
EXTENDED_BEHAVIOR_CONFIG = {
    'grab_transport': {
        'extends': 'grab',  # Inherit from base grab behavior
        'required_args': ['drop_off_target'],
        'on_success': [
            {
                'action': 'update_movement_target',
                'new_movement_target': lambda interactors, args: args.get('drop_off_target')
            }
        ]
    }
}

# Parallel behaviors - execute multiple behaviors simultaneously
PARALLEL_BEHAVIOR_CONFIG = {
    'move_and_reach': {
        'parallel_behaviors': ['move', 'reach_for'],
        'completion_strategy': 'all',  # 'all' or 'any'
    }
}
