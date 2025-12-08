import time

from Luca_MSc.Dynamic_Behavior_Manager.dnf_weights import dnf_weights


class BaseInteractor:
    """Base class for all interactors with common pub/sub and state management"""

    def __init__(self, **kwargs):
        # Common pub/sub infrastructure
        self.cos_subscribers = {}  # {behavior_name: callback}
        self.cof_subscribers = {}  # {behavior_name: callback}
        self.shared_states = {}  # Shared state across behaviors

        # Store initialization parameters
        self.params = kwargs

        self.cos_value = dnf_weights.get_connection_weight('cos_cof_default', 'cos_active')
        self.cof_value = dnf_weights.get_connection_weight('cos_cof_default', 'cof_active')

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

    def _update_and_publish_state(self, state_data, cos_condition, cof_condition=False, continuous_behavior=None):
        """Generic state update and CoS publishing"""
        cos_value = self.cos_value if cos_condition else 0.0
        cof_value = self.cof_value if cof_condition else 0.0

        if continuous_behavior:
            # Add timestamp and store in shared state
            state_data['last_updated'] = time.time()
            state_data['cos_value'] = cos_value
            state_data['cof_value'] = cof_value
            self.shared_states[continuous_behavior] = state_data

            # Publish CoS for continuous calls
            self.publish_cos_state(continuous_behavior, cos_value)
            self.publish_cof_state(continuous_behavior, cof_value)
        # For service calls, just return the data without publishing

        return cos_condition, cof_condition, state_data

    def reset(self):
        """Reset interactor state - to be overridden"""
        self.shared_states.clear()
        self.cos_subscribers.clear()
        self.cof_subscribers.clear()

        self.params.clear()
