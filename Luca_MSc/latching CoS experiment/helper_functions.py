import torch
import numpy as np
import matplotlib.pyplot as plt

from DNF_torch.field import Field


def nodes_list(node_names = list, type_str="precond", params=dict()):
    """
    Create precondition nodes with given parameters.
    """
    nodes = dict()
    for name in node_names:
        nodes[f'{name}_{type_str}'] = Field(
            shape=params.get('shape', ()),
            time_step=params.get('time_step', 5.0),
            time_scale=params.get('time_scale', 100.0),
            resting_level=params.get('resting_level', -3.0),
            beta=params.get('beta', 20.0),
            self_connection_w0=params.get('self_connection_w0', 2),
            noise_strength=params.get('noise_strength', 0.0),
            global_inhibition=params.get('global_inhibition', 0.0),
            scale=params.get('scale', 1.0)
        )

        # Register buffer for prev state
        nodes[f'{name}_{type_str}'].register_buffer(
            "g_u_prev",
            torch.zeros_like(nodes[f'{name}_{type_str}'].g_u)
        )

    return nodes


### Find Grab Behavior Simulation ###

# Moving object once after robot gets close to it
def move_object(find_state, target_name, interactors, offset=(5.0, -2.0, 0.0)):
    """
    Move an object to a new location by applying an offset to its current location.
    """
    current_location = find_state['target_location']
    current_angle = interactors.perception.objects[target_name]['angle']

    # Ensure offset matches device/dtype of current_location
    offset_tensor = torch.tensor(offset, dtype=current_location.dtype, device=current_location.device)
    new_location = current_location.clone().detach() + offset_tensor

    interactors.perception.register_object(target_name, new_location, angle=current_angle)
    print(f"Object {target_name} moved to new location: {new_location.tolist()}")
    return new_location


# Initialize and update logs of node activities
def initalize_log(behavior_chain):
    """
    Initialize the log dictionary adaptively based on the behavior chain.
    
    Args:
        behavior_chain: List of behavior dictionaries from FindMoveBehavior_Experimental
        
    Returns:
        dict: Empty log dictionary with all required keys
    """
    log = {'steps': []}
    
    # Generate keys for each behavior in the chain
    for level in behavior_chain:
        behavior_name = level['name']
        
        # Elementary behavior logs (intention - activation and activity)
        log[f'{behavior_name}_intention_activation'] = []
        log[f'{behavior_name}_intention_activity'] = []
        log[f'{behavior_name}_intention_active'] = []
        
        # CoS logs (activation and activity)
        log[f'{behavior_name}_cos_activation'] = []
        log[f'{behavior_name}_cos_activity'] = []
        log[f'{behavior_name}_cos_active'] = []
        
        # Precondition logs (activation and activity)
        log[f'{behavior_name}_precond_activation'] = []
        log[f'{behavior_name}_precond_activity'] = []
        log[f'{behavior_name}_precond_active'] = []
        
        # Check behavior logs (confidence and intention nodes)
        log[f'{behavior_name}_check_confidence_activation'] = []
        log[f'{behavior_name}_check_confidence_activity'] = []
        log[f'{behavior_name}_check_intention_activation'] = []
        log[f'{behavior_name}_check_intention_activity'] = []
        log[f'{behavior_name}_check_confidence_low'] = []
    
    return log


def update_log(log, state, step, behavior_chain):
    """
    Logging function that uses the behavior chain structure.
    
    Args:
        log: Dictionary containing log arrays
        state: State dictionary from execute_step()
        step: Current simulation step
        behavior_chain: List of behavior dictionaries
    """
    # Log elementary behavior states
    for level in behavior_chain:
        behavior_name = level['name']
        
        if behavior_name in state and state[behavior_name] is not None:
            behavior_state = state[behavior_name]
            
            # Log intention activation and activity
            log[f'{behavior_name}_intention_activation'].append(behavior_state.get('intention_activation', 0.0))
            log[f'{behavior_name}_intention_activity'].append(behavior_state['intention_activity'])
            log[f'{behavior_name}_intention_active'].append(1.0 if behavior_state['intention_active'] else 0.0)
            
            # Log CoS activation and activity  
            log[f'{behavior_name}_cos_activation'].append(behavior_state.get('cos_activation', 0.0))
            log[f'{behavior_name}_cos_activity'].append(behavior_state['cos_activity'])
            log[f'{behavior_name}_cos_active'].append(1.0 if behavior_state['cos_active'] else 0.0)
        else:
            # Fill with zeros if behavior state is None
            log[f'{behavior_name}_intention_activation'].append(0.0)
            log[f'{behavior_name}_intention_activity'].append(0.0)
            log[f'{behavior_name}_intention_active'].append(0.0)
            log[f'{behavior_name}_cos_activation'].append(0.0)
            log[f'{behavior_name}_cos_activity'].append(0.0)
            log[f'{behavior_name}_cos_active'].append(0.0)
    
    # Log precondition states
    if 'preconditions' in state:
        for level in behavior_chain:
            behavior_name = level['name']
            
            if behavior_name in state['preconditions']:
                precond_state = state['preconditions'][behavior_name]
                log[f'{behavior_name}_precond_activation'].append(precond_state['activation'])
                log[f'{behavior_name}_precond_activity'].append(precond_state['activity'])
                log[f'{behavior_name}_precond_active'].append(1.0 if precond_state['active'] else 0.0)
            else:
                log[f'{behavior_name}_precond_activation'].append(0.0)
                log[f'{behavior_name}_precond_activity'].append(0.0)
                log[f'{behavior_name}_precond_active'].append(0.0)
    
    # Log check behavior states
    if 'checks' in state:
        for level in behavior_chain:
            behavior_name = level['name']
            
            if behavior_name in state['checks']:
                check_state = state['checks'][behavior_name]
                log[f'{behavior_name}_check_confidence_activation'].append(check_state.get('confidence_activation', 0.0))
                log[f'{behavior_name}_check_confidence_activity'].append(check_state.get('confidence_activity', 0.0))
                log[f'{behavior_name}_check_intention_activation'].append(check_state.get('intention_activation', 0.0))
                log[f'{behavior_name}_check_intention_activity'].append(check_state.get('intention_activity', 0.0))
                log[f'{behavior_name}_check_confidence_low'].append(1.0 if check_state.get('confidence_low', False) else 0.0)
            else:
                log[f'{behavior_name}_check_confidence_activation'].append(0.0)
                log[f'{behavior_name}_check_confidence_activity'].append(0.0)
                log[f'{behavior_name}_check_intention_activation'].append(0.0)
                log[f'{behavior_name}_check_intention_activity'].append(0.0)
                log[f'{behavior_name}_check_confidence_low'].append(0.0)
    
    # Log step number
    log['steps'].append(step)


def plot_logs(log, steps, behavior_chain):
    """
    Updated plotting function that adapts to the behavior chain structure.
    
    Args:
        log: Dictionary containing logged data
        steps: Number of simulation steps
        behavior_chain: List of behavior dictionaries
    """
    num_behaviors = len(behavior_chain)
    fig, axes = plt.subplots(3, num_behaviors, figsize=(6 * num_behaviors, 12))
    
    # Handle single behavior case
    if num_behaviors == 1:
        axes = axes.reshape(-1, 1)
    
    fig.suptitle('Behavior Chain Dynamics', fontsize=16)
    
    time_steps = list(range(steps))
    
    # Plot each behavior
    for col, level in enumerate(behavior_chain):
        behavior_name = level['name']
          
        # Row 1: Intention + CoS (Activation & Activity combined)
        axes[0, col].plot(time_steps, log[f'{behavior_name}_intention_activation'][:steps], 
                         'b--', label='Intention Activation', linewidth=2, alpha=0.7)
        axes[0, col].plot(time_steps, log[f'{behavior_name}_intention_activity'][:steps], 
                         'b-', label='Intention Activity', linewidth=2)
        axes[0, col].plot(time_steps, log[f'{behavior_name}_cos_activation'][:steps], 
                         'r--', label='CoS Activation', linewidth=2, alpha=0.7)
        axes[0, col].plot(time_steps, log[f'{behavior_name}_cos_activity'][:steps], 
                         'r-', label='CoS Activity', linewidth=2)
        axes[0, col].set_title(f'{behavior_name.title()} - Intention & CoS', fontsize=14)
        axes[0, col].set_ylabel('Value', fontsize=12)
        axes[0, col].legend()
        axes[0, col].grid(True, alpha=0.3)
        axes[0, col].set_ylim(-6, 6)
        
        # Row 2: Precondition Node (Activation & Activity)
        axes[1, col].plot(time_steps, log[f'{behavior_name}_precond_activation'][:steps], 
                         'g--', label='Precond Activation', linewidth=2, alpha=0.7)
        axes[1, col].plot(time_steps, log[f'{behavior_name}_precond_activity'][:steps], 
                         'g-', label='Precond Activity', linewidth=2)
        axes[1, col].set_title(f'{behavior_name.title()} - Precondition', fontsize=14)
        axes[1, col].set_ylabel('Value', fontsize=12)
        axes[1, col].legend()
        axes[1, col].grid(True, alpha=0.3)
        axes[1, col].set_ylim(-6, 6)
        
        # Row 3: Check Behavior (Confidence & Intention nodes)
        axes[2, col].plot(time_steps, log[f'{behavior_name}_check_confidence_activation'][:steps], 
                         'orange', linestyle='--', label='Confidence Activation', linewidth=2, alpha=0.7)
        axes[2, col].plot(time_steps, log[f'{behavior_name}_check_confidence_activity'][:steps], 
                         'orange', label='Confidence Activity', linewidth=2)
        axes[2, col].plot(time_steps, log[f'{behavior_name}_check_intention_activation'][:steps], 
                         'purple', linestyle='--', label='Check Intention Activation', linewidth=2, alpha=0.7)
        axes[2, col].plot(time_steps, log[f'{behavior_name}_check_intention_activity'][:steps], 
                         'purple', label='Check Intention Activity', linewidth=2)
        axes[2, col].set_title(f'{behavior_name.title()} - Check Behavior', fontsize=14)
        axes[2, col].set_xlabel('Time Steps', fontsize=12)
        axes[2, col].set_ylabel('Value', fontsize=12)
        axes[2, col].legend()
        axes[2, col].grid(True, alpha=0.3)
        axes[2, col].set_ylim(-6, 6)

    plt.tight_layout()
    plt.show()