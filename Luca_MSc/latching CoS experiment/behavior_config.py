
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


ElEMENTARY_BEHAVIOR_CONFIG = {
    'find': {
            'interactor_type': 'perception',
            'continuous_method': 'find_object_continuous',
            'service_method': 'find_object_service',
            'service_args_func': lambda interactors, target_name: (target_name,)
    },
   'move':{
            'interactor_type': 'movement',
            'continuous_method': 'move_to_continuous',
            'service_method': 'move_to_service',
            'service_args_func': lambda interactors, target_name: (
                interactors.perception.target_states.get(target_name, {}).get('location'),
                )
    },
    'check_reach':{
            'interactor_type': 'gripper',
            'continuous_method': 'reach_check_continuous',
            'service_method': 'reach_check_service',
            'service_args_func': lambda interactors, target_name: (
                interactors.perception.target_states.get(target_name, {}).get('location'),
                )
    },
    'reach_for':{
            'interactor_type': 'gripper',
            'continuous_method': 'reach_for_continuous',
            'service_method': 'reach_for_service',
            'service_args_func': lambda interactors, target_name: (
                interactors.perception.target_states.get(target_name, {}).get('location'),
                )
    },
    'grab':{
            'interactor_type': 'gripper',
            'continuous_method': 'grab_continuous',
            'service_method': 'grab_service',
            'service_args_func': lambda interactors, target_name: (
                interactors.perception.target_states.get(target_name, {}).get('location'),
                interactors.perception.target_states.get(target_name, {}).get('angle')
            ),
    }
}

# Action-only behaviors (only service calls, no continuous updates)
ACTION_BEHAVIOR_CONFIG = {
    'update_target': {
            'interactor_type': 'state',
            'service_method': 'update_target_location',
            'service_args_func': lambda interactors, target_name, action: (
                action['target_name'](interactors, target_name),
                action['new_location'](interactors, target_name)
            )
    },
    'announce': {
            'interactor_type': 'state',
            'service_method': 'announce_message',
            'service_args_func': lambda interactors, target_name, action: (
                action['message'](interactors, target_name),
           )
    }
}


# Composite/Extended behaviors
EXTENDED_BEHAVIOR_CONFIG = {
    'grab_transport': {
        'extends': 'grab',  # Inherit from base grab behavior
        'on_success': [
            {
                'action': 'update_target',
                'target_name': lambda interactors, target_name: target_name,
                'new_location': lambda interactors, target_name: interactors.perception.target_states.get('transport_target', {}).get('location'),
            }
        ]
    },
    'find_and_announce': {
        'extends': 'find',
        'on_success': [
            {
                'action': 'print_message',
                'message': lambda interactors, target_name: f"Successfully found {target_name}!"
            }
        ]
    }
}