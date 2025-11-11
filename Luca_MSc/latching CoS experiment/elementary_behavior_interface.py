import os
import sys
# Add DNF_torch package root
sys.path.append(os.path.expanduser('~/nc_ws/DNF_torch'))

# Add Luca_MSc subfolder for local scripts
sys.path.append(os.path.join(os.path.expanduser('~/nc_ws/DNF_torch'), 'Luca_MSc/latching CoS experiment'))

from elementary_behavior import ElementaryBehavior

class ElementaryBehaviorInterface (ElementaryBehavior):
    """This interface extends the ElementaryBehavior to allow external setting of CoS input and processing DNF dynamics each step."""
    def __init__(self, field_params=None):
        super().__init__(field_params)
        self.cos_input = 0.0  # Subscribed value from external publisher

    def set_cos_input(self, value):
        """Called by external publisher when CoS state changes"""
        self.cos_input = value
        
    def execute(self, external_input=0.0):
        """Execute only DNF dynamics - no world interaction"""

        state = self.forward(external_input, self.cos_input)

        return {
            'intention_active': float(state.get('intention_activity', 0.0)) > 0.5,
            'cos_active': float(state.get('cos_activity', 0.0)) > 0.7,
            'intention_activation': float(state.get('intention_activation', 0.0)),
            'intention_activity': float(state.get('intention_activity', 0.0)),
            'cos_activation': float(state.get('cos_activation', 0.0)),
            'cos_activity': float(state.get('cos_activity', 0.0))
        }
    