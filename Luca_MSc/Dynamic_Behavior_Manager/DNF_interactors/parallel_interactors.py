
from Luca_MSc.Dynamic_Behavior_Manager.dnf_weights import dnf_weights

class ParallelInteractor():
    """
    Wraps multiple interactors to execute their methods in parallel.
    Presents a unified interface that looks like a single interactor.
    """
    def __init__(self, interactors_config, parallel_behaviors):
        """
        Args:
            interactors_config: List of dicts with 'interactor', 'method', etc
        """
        self.interactors_config = interactors_config
        self.parallel_behaviors = parallel_behaviors
        self.cos_callbacks = []
        self.cof_callbacks = []
        self.component_checks = []


    def execute_parallel(self, requesting_behavior=None):
        """Execute all wrapped interactor methods in parallel"""
        results = []
        for i, config in enumerate(self.interactors_config):
            interactor = config['interactor']
            method = getattr(interactor, config['method'])
            base_name = self.parallel_behaviors[i]

            result = method(requesting_behavior=base_name if requesting_behavior else None)
            results.append([result, base_name])

        return results

    def process_sanity_results(self, results, behavior_manager):
        """
        Process sanity check results for all component behaviors.
        Updates each component behavior's CoS input based on check results.
        """
        weights = dnf_weights
        cos = weights.connection_weights['cos_cof_default']['cos_active']
        cof = weights.connection_weights['cos_cof_default']['cof_active']

        for result, component_name in results:
            base_name = behavior_manager.initializer.get_base_behavior_name(component_name)
            component_behavior = getattr(behavior_manager, f"{base_name}_behavior")
            # Update component behavior CoS based on sanity check result
            if result[0]:  # CoS check passed
                component_behavior.set_cos_input(cos)
            else:  # Failure
                component_behavior.set_cos_input(0)

            if result[1]:  # CoF reached
                component_behavior.set_cof_input(cof)


    def subscribe_cos_updates(self, behavior_name, callback):
        """Subscribe to CoS updates from ALL wrapped interactors"""
        for config in self.interactors_config:
            interactor = config['interactor']
            # Each interactor calls the SAME callback (aggregated via DNF)
            interactor.subscribe_cos_updates(behavior_name, callback)

    def subscribe_cof_updates(self, behavior_name, callback):
        """Subscribe to CoF updates from ALL wrapped interactors"""
        for config in self.interactors_config:
            interactor = config['interactor']
            interactor.subscribe_cof_updates(behavior_name, callback)

    def set_check_behaviors(self, check_behaviors):
        """Set component sanity check functions for each interactor"""
        self.component_checks = check_behaviors

    def get_check_behaviors(self):
        """Get the list of component sanity check functions"""
        return self.component_checks
