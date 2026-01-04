import torch
import torch.nn as nn

from DNF_torch.field import Field

class ElementaryBehavior(nn.Module):
    """
    Elementary behavior structure using Dynamic Neural Fields.
    Includes CoI and CoS nodes for behavior control.
    """

    def __init__(self, dynamics_params=None):
        super().__init__()

        # Default parameters for fields if none provided
        default_params = {
            'shape': (),
            'time_step': 5.0,
            'time_scale': 100.0,
            'resting_level': -3.0,
            'noise_strength': 0.00,
            'beta': 100.0,  # Steeper sigmoid for clearer decisions
            'self_connection_w0': 1.0
        }

        default_node_params = {
            'intention': {**default_params},
            'cos': {**default_params},
            'cos_inverter': {**default_params, 'resting_level': 1.0}, # Inverter starts active
            'cof': {**default_params}
        }

        default_weights = {
            'cos_to_intention': -7.0,
            'cos_to_inverter': -4.0,
            'cof_to_intention': -7.0,
            'cof_to_cos': -7.0
        }

        # Use provided parameters if available, otherwise use defaults
        params = default_node_params.copy()
        weights = default_weights.copy()
        if dynamics_params is not None:
            for node_type, node_type_params in dynamics_params.get_field_params('behavior_nodes').items():
                params[node_type].update(node_type_params)

            for connection_type, connection_type_weight in dynamics_params.get_connection_weight('behavior_internal').items():
                weights[connection_type] = connection_type_weight

        # Create the intention node that drives behavior execution
        self.intention = Field(**{**params['intention']})

        # Create the control nodes
        # Condition of Satisfaction - active when behavior is fulfilled
        self.CoS = Field(**{**params['cos']})
        # Inverter for CoS - active when CoS is inactive
        self.CoS_inverter = Field(**{**params['cos_inverter']})
        # Condition of Failure - active when behavior cannot be fulfilled
        self.CoF = Field(**{**params['cof']})


        # Initialize connection weights
        # Inhibitory connection
        self.CoS.connection_to(self.intention, weights['cos_to_intention'])
        self.CoS.connection_to(self.CoS_inverter, weights['cos_to_inverter'])
        self.CoF.connection_to(self.intention, weights['cof_to_intention'])
        self.CoF.connection_to(self.CoS, weights['cof_to_cos'])

        # Create a buffer for previous g_u (activity) values (for synchronous updates)
        self.intention.register_buffer("g_u_prev", torch.zeros_like(self.intention.g_u))
        self.CoS.register_buffer("g_u_prev", torch.zeros_like(self.CoS.g_u))
        self.CoS_inverter.register_buffer("g_u_prev", torch.zeros_like(self.CoS.g_u))
        self.CoF.register_buffer("g_u_prev", torch.zeros_like(self.CoF.g_u))

        # Initialize behavior state
        self.is_active = False
        self.is_completed = False
        self.is_failed = False

    def forward(self, intention_input=None, cos_input=None, cof_input=None):
        """
        Process one step of the behavior control.

        Args:
            intention_input: Input to drive the intention node
            cos_input: Input tensor for Condition of Satisfaction
        Returns:
            dict: Current state of the behavior
        """
        # Cache previous g_u values for synchronous updates
        self.intention.cache_prev()
        self.CoS.cache_prev()
        self.CoF.cache_prev()
        self.CoS_inverter.cache_prev()

        # Process control fields
        intention_activation, intention_activity = self.intention(intention_input)
        cos_activation, cos_activity = self.CoS(cos_input)
        cof_activation, cof_activity = self.CoF(cof_input)
        _, _ = self.CoS_inverter()  # Update inverter state (no external input)

        # Update behavior state based on field activities
        cos_active = float(cos_activity) > 0.7
        intention_active = float(intention_activity) > 0.5 and not self.is_completed
        cof_active = float(cof_activity) > 0.7

        # Update behavior state
        if cof_active:
            self.is_active = False
            self.is_completed = False
            self.is_failed = True
        elif cos_active:
            self.is_completed = True
            self.is_active = False
        elif intention_active:
            self.is_active = True
        else:
            self.is_active = False

        # Return current behavior state
        return {
            'active': bool(self.is_active),
            'completed': bool(self.is_completed),
            'failed': bool(self.is_failed),
            'intention_activation': float(intention_activation.detach()),
            'intention_activity': float(intention_activity.detach()),
            'cos_activation': float(cos_activation.detach()),
            'cos_activity': float(cos_activity.detach()),
            'cof_activation': float(cof_activation.detach()),
            'cof_activity': float(cof_activity.detach())
        }

    def execute(self, *args, **kwargs):
        raise NotImplementedError("This method should be implemented in subclasses.")

    def reset(self):
        """Reset the behavior state and all fields."""
        self.intention.reset()
        self.CoS.reset()
        self.CoF.reset()
        self.CoS_inverter.reset()

        self.intention.clear_connections()
        self.CoS.clear_connections()
        self.CoF.clear_connections()
        self.CoS_inverter.clear_connections()

        self.is_active = False
        self.is_completed = False
        self.is_failed = False