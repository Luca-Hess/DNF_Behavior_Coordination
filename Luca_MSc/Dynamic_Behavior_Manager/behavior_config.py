
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
            'service_args_func': lambda interactors, args, behavior_name: (
                interactors.state.get_behavior_target_name(behavior_name),
            )
    },
   'move':{
            'interactor_type': 'movement',
            'method': 'move_to',
            'service_args_func': lambda interactors, args, behavior_name: (
                interactors.state.get_behavior_target_location(behavior_name),
            )
    },
    'check_reach':{
            'interactor_type': 'gripper',
            'method': 'reach_check',
            'service_args_func': lambda interactors, args, behavior_name:
            interactors.state.get_behavior_target_info(behavior_name)
    },
    'reach_for':{
            'interactor_type': 'gripper',
            'method': 'reach_for',
            'service_args_func': lambda interactors, args, behavior_name:
            interactors.state.get_behavior_target_info(behavior_name)
    },
    'grab':{
            'interactor_type': 'gripper',
            'method': 'grab',
            'service_args_func': lambda interactors, args, behavior_name: 
            interactors.state.get_behavior_target_info(behavior_name)
    }
}

# Action-only behaviors (only service calls, no continuous updates)
ACTION_BEHAVIOR_CONFIG = {
    'update_movement_target': {
            'interactor_type': 'state',
            'service_method': 'update_behavior_target',
            'service_args_func': lambda interactors, args, action: (
                'move',
                action['new_movement_target'](interactors, args),
            )
    },
    'announce': {
            'interactor_type': 'state',
            'service_method': 'announce_message',
            'service_args_func': lambda interactors, args, action: (
                action['message'](interactors, args),
           )
    }
}


# Composite/Extended behaviors
EXTENDED_BEHAVIOR_CONFIG = {
    'grab_transport': {
        'extends': 'grab',  # Inherit from base grab behavior
        'on_success': [
            {
                'action': 'update_movement_target',
                'new_movement_target': lambda interactors, args: args.get('drop_off_target')
            }
        ]
    }
}