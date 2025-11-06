import torch
import torch.nn as nn

import sys
import os
# Add DNF_torch package root
sys.path.append(os.path.expanduser('~/nc_ws/DNF_torch'))

# Add Luca_MSc subfolder for local scripts
sys.path.append(os.path.join(os.path.expanduser('~/nc_ws/DNF_torch'), 'Luca_MSc/latching CoS experiment'))

from DNF_torch.field import Field


class CheckBehavior(nn.Module):
    """
    Check behavior structure using Dynamic Neural Fields.
    Includes intention and confidence node to time & tune sanity checks.
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

        # Create the confidence node that triggers sanity checks - currently tuned to trigger with 1Hz 
        self.confidence = Field(**{**params, 'beta': 2.0, 'resting_level': -4.0, 'time_scale': 300.0})

        # Initialize connection weights
        intention_to_confidence_weight = 4  # Excitatory

        # Excitatory (reversed logic => once confidence activity > 0, confidence is considered to be too low)
        self.intention.connection_to(self.confidence, intention_to_confidence_weight)

        # Create a buffer for previous g_u values (for synchronous updates)
        self.intention.register_buffer("g_u_prev", torch.zeros_like(self.intention.g_u))
        self.confidence.register_buffer("g_u_prev", torch.zeros_like(self.confidence.g_u))

        # Initialize behavior state
        self.confidence_low = False

    def forward(self, intention_input = None, confidence_input=None):
        """
        Process one step of the behavior control.

        Args:
            intention_input: Input to drive the intention node
            confidence_input: Input tensor for Confidence measure
        Returns:
            dict: Current state of the behavior
        """
        # Cache previous g_u values for synchronous updates
        self.intention.cache_prev()
        self.confidence.cache_prev()

        # Process control fields
        intention_activation, intention_activity = self.intention(intention_input)
        confidence_activation, confidence_activity = self.confidence(confidence_input)


        # Return current behavior state
        return {
            'intention_activation': float(intention_activation.detach()),
            'intention_activity': float(intention_activity.detach()),
            'confidence_activation': float(confidence_activation.detach()),
            'confidence_activity': float(confidence_activity.detach())
        }

    def execute(self, *args, **kwargs):
        raise NotImplementedError("This method should be implemented in subclasses.")

    def reset(self):
        """Reset the behavior state and all fields."""
        self.intention.reset()
        self.confidence.reset()

        self.confidence_low = False



# simple demo with plot
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Create behavior
    behavior = CheckBehavior()

    # Simulation parameters
    time_steps = 200
    intention_input = 5.0  # Constant input to intention

    # Store activities for plotting
    intention_activities = []
    intention_activations = []
    confidence_activities = []
    confidence_activations = []
    confidence_state = []

    # Run simulation
    for t in range(time_steps):

        state = behavior.forward(intention_input=intention_input, confidence_input=0.0)

        intention_activities.append(state['intention_activity'])
        intention_activations.append(state['intention_activation'])
        confidence_activities.append(state['confidence_activity'])
        confidence_activations.append(state['confidence_activation'])
        confidence_state.append(state['confidence_low'])

    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(intention_activations, label='Intention Activation')
    plt.plot(intention_activities, label='Intention Activity')
    plt.plot(confidence_activations, label='Confidence Activation')
    plt.plot(confidence_activities, label='Confidence Activity')
    plt.plot([1.0 if cs else 0.0 for cs in confidence_state], label='Confidence Low State', linestyle='--')
    plt.axhline(0.7, color='r', linestyle='--', label='Confidence Threshold')
    plt.xlabel('Time Step')
    plt.ylabel('Activity')
    plt.title('Check Behavior Activities Over Time')
    plt.legend()
    plt.grid()
    plt.show()