
from Luca_MSc.Dynamic_Behavior_Manager.DNF_interactors.parallel_interactors import ParallelInteractor

import behavior_config

class RuntimeManagement:
    def __init__(self, behavior_manager):
        self.behavior_manager = behavior_manager
        self.initializer = behavior_manager.initializer
        self.weights = behavior_manager.weights
        self.runtime_weights = behavior_manager.runtime_weights
        self.behavior_args = behavior_manager.behavior_args
        self.behavior_chain = behavior_manager.behavior_chain

    def setup_subscriptions(self, interactors):
        """Setup pub/sub connections using behavior chain data"""
        # Initialize StateInteractor based on behavior chain
        interactors.state.initialize_from_behavior_chain(self.behavior_chain, self.behavior_args)

        # Main behavior subscribes to interactor CoS updates
        for level in self.behavior_chain:

            # Creating the ParallelInteractor for parallel behaviors
            if level.get('interactor_type') == 'parallel':
                # Build interactor
                wrapped_interactors = []
                for single_component_config in level['component_configs']:
                    actual_interactor = getattr(interactors, single_component_config['interactor_type'])
                    wrapped_interactors.append({
                        'interactor': actual_interactor,
                        'method': single_component_config['method'],
                    })

                # Creating the wrapper around the component interactors
                parallel_interactor = ParallelInteractor(
                    wrapped_interactors,
                    level['parallel_behaviors']
                )

                # Store wrapper like normal interactor
                level['interactor_instance'] = parallel_interactor

                # Sanity check handled for all component behaviors via the wrapper
                level['check'].set_interactor(parallel_interactor)

                # Additionally set the targets of the component behaviors and subscribe them to CoS & CoF update
                parallel_behaviors = []
                for component_name in level['parallel_behaviors']:
                    parallel_behaviors.append({'name': component_name})
                for i, config in enumerate(level['component_configs']):
                    interactor_type = config['interactor_type']
                    actual_interactor = getattr(interactors, interactor_type)
                    method = config['method']
                    name = parallel_behaviors[i]['name']
                    parallel_behaviors[i].update({'interactor_type': interactor_type,
                                                  'actual_interactor': actual_interactor,
                                                  'method': method})

                    actual_interactor.subscribe_cos_updates(
                        name, getattr(self.behavior_manager, f'{name}_behavior').set_cos_input
                    )

                    actual_interactor.subscribe_cof_updates(
                        name, getattr(self.behavior_manager, f'{name}_behavior').set_cof_input
                    )

                interactors.state.initialize_from_behavior_chain(parallel_behaviors, self.behavior_args)

            # Normal single, non-parallel interactors
            else:
                interactor = getattr(interactors, level['interactor_type'])
                level['interactor_instance'] = interactor

                # Subscribe to CoS and CoF updates
                interactor.subscribe_cos_updates(
                    level['name'], level['behavior'].set_cos_input
                )

                interactor.subscribe_cof_updates(
                    level['name'], level['behavior'].set_cof_input
                )

                # Set up check behavior to publish back to the same interactor
                level['check'].set_interactor(interactor)

    def clear_subscriptions(self, interactors):
        """Clear all subscriptions from interactors and release check behaviors"""
        # Clear interactor subscriptions from base interactor class
        interactors.reset()

        # Release check behaviors
        for level in self.behavior_chain:
            level['check'].set_interactor(None)

    def get_active_behavior(self):
        """Determine which behavior should be actively interacting with world"""
        for level in self.behavior_chain:
            if level['behavior'].execute()['intention_active']:
                return level['name']
        return None

    def process_success_actions(self, actions, interactors, behavior_args):
        """
        Process declarative success actions for a behavior.
        => Some behaviors trigger additional one-time actions upon successful completion.
        """
        for action in actions:
            action_type = action.get('action')

            # Look up action in action-only behaviors
            if action_type in behavior_config.ACTION_BEHAVIOR_CONFIG:
                config = behavior_config.ACTION_BEHAVIOR_CONFIG[action_type]
                interactor = getattr(interactors, config['interactor_type'])
                service_method = getattr(interactor, config['service_method'])
                service_args = config['service_args_func'](interactors, behavior_args, action)

                try:
                    result = service_method(*service_args)
                    if not result[0]:  # Check if action succeeded
                        print(f"[WARNING] Success action {action_type} failed")
                except Exception as e:
                    print(f"[ERROR] Success action {action_type} failed: {e}")
            else:
                print(f"[ERROR] Unknown action type: {action_type}")

    def process_system_level_nodes(self):
        """Process system-level CoS and CoF nodes"""

        # Update system-level nodes (they receive inputs from behavior nodes automatically via connections)
        self.behavior_manager.system_intention.cache_prev()
        self.behavior_manager.system_cos.cache_prev()
        self.behavior_manager.system_cos_reporter.cache_prev()
        self.behavior_manager.system_cof.cache_prev()

        # Execute system-level dynamics
        system_intention_activation, system_intention_activity = self.behavior_manager.system_intention()
        system_cos_activation, system_cos_activity = self.behavior_manager.system_cos()
        system_cos_reporter_activation, system_cos_reporter_activity = self.behavior_manager.system_cos_reporter()
        system_cof_activation, system_cof_activity = self.behavior_manager.system_cof()

        # Determine system state
        system_success = float(system_cos_activity) > 0.7
        system_failure = float(system_cof_activity) > 0.7

        system_state = {
            'intention_activation': float(system_intention_activation.detach()),
            'intention_activity': float(system_intention_activity.detach()),
            'cos_activation': float(system_cos_activation.detach()),
            'cos_activity': float(system_cos_activity.detach()),
            'cos_reporter_activation': float(system_cos_reporter_activation.detach()),
            'cos_reporter_activity': float(system_cos_reporter_activity.detach()),
            'cof_activation': float(system_cof_activation.detach()),
            'cof_activity': float(system_cof_activity.detach()),
            'system_success': system_success,
            'system_failure': system_failure,
            'system_status': self.determine_system_status(system_success, system_failure)
        }

        self.behavior_manager.debug_print(
            f"System state: {system_state['system_status']} (Intention: {system_state['intention_activity']:.3f}, CoS: {system_state['cos_activity']:.3f}, CoF: {system_state['cof_activity']:.3f})")

        return system_state

    def determine_system_status(self, system_success, system_failure):
        """Determine overall system status based on CoS and CoF"""
        if system_failure:
            return "FAILED"
        elif system_success:
            return "SUCCESS"
        else:
            return "IN_PROGRESS"