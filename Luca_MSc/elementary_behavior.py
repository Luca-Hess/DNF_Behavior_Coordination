import torch
import torch.nn as nn
from DNF_torch.field import Field


class ElementaryBehavior(nn.Module):
    """
    Elementary behavior structure using Dynamic Neural Fields.
    Includes CoI, CoS, and CoF nodes for behavior control.
    """

    def __init__(self, field_params=None):
        super().__init__()

        # Default parameters for fields if not provided
        default_params = {
            'shape': (),
            'time_step': 5.0,
            'time_scale': 100.0,
            'resting_level': -3.0,
            'noise_strength': 0.01,
            'beta': 100.0,  # Steeper sigmoid for clearer decisions
            'self_connection_w0': 1.0
        }

        # Use provided parameters if available, otherwise use defaults
        params = default_params.copy()
        if field_params is not None:
            params.update(field_params)

        # Create the intention node that drives behavior execution
        self.intention = Field(**params)

        # Create the control nodes
        self.CoS = Field(**params)  # Condition of Satisfaction

        # Initialize connection weights
        #intention_to_cos_weight = 6.0   # Excitatory
        cos_to_intention_weight = -6.0  # Inhibitory

        # Set up connections
        # Excitatory
        #self.intention.connection_to(self.CoS, intention_to_cos_weight)

        # Inhibitory
        self.CoS.connection_to(self.intention, cos_to_intention_weight)


        # Create a buffer for previous g_u values (for synchronous updates)
        self.intention.register_buffer("g_u_prev", torch.zeros_like(self.intention.g_u))
        self.CoS.register_buffer("g_u_prev", torch.zeros_like(self.CoS.g_u))

        # Initialize behavior state
        self.is_active = False
        self.is_completed = False

    def forward(self, intention_input = None, cos_input=None):
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

        # Process control fields
        intention_activation, intention_activity = self.intention(intention_input)
        cos_activation, cos_activity = self.CoS(cos_input)

        # Update behavior state based on field activities
        cos_active = float(cos_activity) > 0.7
        intention_active = float(intention_activity) > 0.5 and not self.is_completed

        # Update behavior state
        if cos_active:
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
            'intention_activation': float(intention_activation.detach()),
            'intention_activity': float(intention_activity.detach()),
            'cos_activation': float(cos_activation.detach()),
            'cos_activity': float(cos_activity.detach())
        }

    def execute(self, *args, **kwargs):
        raise NotImplementedError("This method should be implemented in subclasses.")

    def reset(self):
        """Reset the behavior state and all fields."""
        self.intention.reset()
        self.CoS.reset()

        self.is_active = False
        self.is_completed = False