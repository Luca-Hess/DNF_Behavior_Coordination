
from Luca_MSc.Dynamic_Behavior_Manager.dnf_weights import dnf_weights

class ParallelInteractor():
    """
    Wraps multiple interactors to execute their methods in parallel.
    Presents a unified interface that looks like a single interactor.
    """
    def __init__(self, component_configs):
        """
        Args:
            interactors_config: List of dicts with 'interactor', 'method', etc
        """
        self.components = component_configs
        self.cos_callbacks = []
        self.cof_callbacks = []
        self.component_checks = []


    def execute_parallel(self, continuous_behavior=None):
        """Execute all wrapped interactor methods in parallel"""
        results = []
        for comp in self.components:
            method = getattr(comp['actual_interactor'], comp['method'])

            result = method(continuous_behavior=comp['name'] if continuous_behavior else None)
            results.append((result, comp))

        return results

    def process_sanity_results(self, results, behavior_manager):
        """
        Process sanity check results for all component behaviors.
        Updates each component behavior's CoS input based on check results.
        """
        weights = dnf_weights
        cos = weights.connection_weights['cos_cof_default']['cos_active']
        cof = weights.connection_weights['cos_cof_default']['cof_active']

        for result, comp in results:
            behavior = comp['behavior_instance']
            # Update component behavior CoS based on sanity check result
            behavior.set_cos_input(cos if result[0] else 0.0) # CoS passed

            if result[1]:  # CoF reached
                behavior.set_cof_input(cof)

    def setup_component_subscriptions(self):
        """Setup CoS and CoF subscriptions for all component behaviors"""
        for comp in self.components:
            interactor = comp['actual_interactor']
            behavior = comp['behavior_instance']

            # Subscribe to CoS updates
            interactor.subscribe_cos_updates(comp['name'], behavior.set_cos_input)
            # Subscribe to CoF updates
            interactor.subscribe_cof_updates(comp['name'], behavior.set_cof_input)

