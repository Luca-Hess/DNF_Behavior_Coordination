from check_behavior import CheckBehavior

class SanityCheckBehavior(CheckBehavior):
    """
    Low-cost sanity check behavior to verify if a behavior target-state is still achieved.
    """
    def __init__(self, behavior_name=None, field_params=None):
        super().__init__(field_params)
        self.cos_input = 5.0
        self.behavior_name = behavior_name  # Name of the associated elementary behavior
        self.interactor = None              # To be set externally if needed

    def set_interactor(self, interactor):
        """Set the interactor object for publishing CoS updates"""
        self.interactor = interactor

    def execute(self, external_input=0.0):
        """Execute only DNF dynamics - no world interaction"""
        state = self.forward(external_input, None)

        sanity_check_triggered = float(state.get('confidence_activity', 0.0)) > 0.7

        # Once the signal for a sanity check is sent, reset the node and start next checking cycle
        # => If the check fails, the sanity check will only re-engage once the intention is re-activated
        if sanity_check_triggered:
            self.reset()

        return {
            'intention_active': float(state.get('intention_activity', 0.0)) > 0.5,
            'confidence_active': float(state.get('confidence_activity', 0.0)) > 0.7,
            'intention_activation': float(state.get('intention_activation', 0.0)),
            'intention_activity': float(state.get('intention_activity', 0.0)),
            'confidence_activation': float(state.get('confidence_activation', 0.0)),
            'confidence_activity': float(state.get('confidence_activity', 0.0)),
            'sanity_check_triggered': sanity_check_triggered
        }
    
    def process_sanity_result(self, result, check_failed_func, behavior_name):
        """Process the result of the sanity check and update CoS input to the associated elementary behavior accordingly"""
        if check_failed_func(result):
            # Check failed - set CoS input to 0
            cos_value = 0.0 

        else:
            # Check passed - set/confirm CoS input at 5.0
            cos_value = 5.0

        # Update CoS input of the associated elementary behavior
        if self.interactor and behavior_name:
            self.interactor.publish_cos_state(behavior_name, cos_value)

        # Update internal CoS state
        self.cos_input = cos_value