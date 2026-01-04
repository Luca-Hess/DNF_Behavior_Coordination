from Luca_MSc.Dynamic_Behavior_Manager.elementary_behavior import ElementaryBehavior

class ElementaryBehaviorInterface (ElementaryBehavior):
    """This interface extends the ElementaryBehavior to allow external setting of CoS input and processing DNF dynamics each step."""
    def __init__(self, behavior_name=None, dynamics_params=None):
        super().__init__(dynamics_params)
        self.cos_input = 0.0  # Subscribed value from external publisher
        self.cof_input = 0.0  # Subscribed value from external publisher
        self.behavior_name = behavior_name  # Name of the elementary behavior

    def set_cos_input(self, value):
        """Called by external publisher when CoS state changes"""
        self.cos_input = value

    def set_cof_input(self, value):
        """Called by external publisher when CoF state changes"""
        self.cof_input = value
        
    def execute(self, external_input=0.0):
        """Execute only DNF dynamics - no world interaction"""

        state = self.forward(intention_input=external_input, cos_input=self.cos_input, cof_input=self.cof_input)

        return {
            'intention_active': float(state.get('intention_activity', 0.0)) > 0.5,
            'cos_active': float(state.get('cos_activity', 0.0)) > 0.7,
            'cof_active': float(state.get('cof_activity', 0.0)) > 0.7,
            'intention_activation': float(state.get('intention_activation', 0.0)),
            'intention_activity': float(state.get('intention_activity', 0.0)),
            'cos_activation': float(state.get('cos_activation', 0.0)),
            'cos_activity': float(state.get('cos_activity', 0.0)),
            'cof_activation': float(state.get('cof_activation', 0.0)),
            'cof_activity': float(state.get('cof_activity', 0.0)),
        }
    