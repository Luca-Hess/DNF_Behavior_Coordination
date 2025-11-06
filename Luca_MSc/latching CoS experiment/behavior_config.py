

# Define behavior configurations

BEHAVIOR_CONFIG = {
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