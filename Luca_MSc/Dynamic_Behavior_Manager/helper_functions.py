import torch
import numpy as np
import matplotlib.pyplot as plt

from DNF_torch.field import Field




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

        # CoF logs (activation and activity)
        log[f'{behavior_name}_cof_activation'] = []
        log[f'{behavior_name}_cof_activity'] = []
        log[f'{behavior_name}_cof_active'] = []
        
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

    # Log system-level nodes
    log['system_cos_activation'] = []
    log['system_cos_activity'] = []
    log['system_cof_activation'] = []
    log['system_cof_activity'] = []
    log['system_success'] = []
    log['system_failure'] = []
    
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

            # Log CoF activation and activity  
            log[f'{behavior_name}_cof_activation'].append(behavior_state.get('cof_activation', 0.0))
            log[f'{behavior_name}_cof_activity'].append(behavior_state['cof_activity'])
            log[f'{behavior_name}_cof_active'].append(1.0 if behavior_state['cof_active'] else 0.0)
        else:
            # Fill with zeros if behavior state is None
            log[f'{behavior_name}_intention_activation'].append(0.0)
            log[f'{behavior_name}_intention_activity'].append(0.0)
            log[f'{behavior_name}_intention_active'].append(0.0)
            log[f'{behavior_name}_cos_activation'].append(0.0)
            log[f'{behavior_name}_cos_activity'].append(0.0)
            log[f'{behavior_name}_cos_active'].append(0.0)
            log[f'{behavior_name}_cof_activation'].append(0.0)
            log[f'{behavior_name}_cof_activity'].append(0.0)
            log[f'{behavior_name}_cof_active'].append(0.0)
    
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

    # Log system-level nodes
    if 'system' in state:
        system_state = state['system']
        log['system_cos_activation'].append(system_state.get('cos_activation', 0.0))
        log['system_cos_activity'].append(system_state.get('cos_activity', 0.0))
        log['system_cof_activation'].append(system_state.get('cof_activation', 0.0))
        log['system_cof_activity'].append(system_state.get('cof_activity', 0.0))
        log['system_success'].append(system_state.get('system_success', False))
        log['system_failure'].append(system_state.get('system_failure', False))

def plot_logs(log, steps, behavior_chain):
    """
    Updated plotting function that adapts to the behavior chain structure.
    
    Args:
        log: Dictionary containing logged data
        steps: Number of simulation steps
        behavior_chain: List of behavior dictionaries
    """
    num_behaviors = len(behavior_chain)
    fig, axes = plt.subplots(3, num_behaviors+1, figsize=(6 * num_behaviors, 12))
    
    # Handle single behavior case
    if num_behaviors == 1:
        axes = axes.reshape(-1, 1)
    
    fig.suptitle('Behavior Chain Dynamics', fontsize=16)
    
    time_steps = list(range(steps))
    
    # Plot each behavior
    for col, level in enumerate(behavior_chain):
        behavior_name = level['name']
          
        # Row 1: Intention, CoS & CoF (Activation & Activity combined)
        axes[0, col].plot(time_steps, log[f'{behavior_name}_intention_activation'][:steps], 
                         'b--', label='Intention Activation', linewidth=2, alpha=0.7)
        axes[0, col].plot(time_steps, log[f'{behavior_name}_intention_activity'][:steps], 
                         'b-', label='Intention Activity', linewidth=2)
        axes[0, col].plot(time_steps, log[f'{behavior_name}_cos_activation'][:steps], 
                         'r--', label='CoS Activation', linewidth=2, alpha=0.7)
        axes[0, col].plot(time_steps, log[f'{behavior_name}_cos_activity'][:steps], 
                         'r-', label='CoS Activity', linewidth=2)
        axes[0, col].plot(time_steps, log[f'{behavior_name}_cof_activation'][:steps], 
                         'm--', label='CoF Activation', linewidth=2, alpha=0.7)
        axes[0, col].plot(time_steps, log[f'{behavior_name}_cof_activity'][:steps], 
                         'm-', label='CoF Activity', linewidth=2)
        axes[0, col].set_title(f'{behavior_name.title()} - Intention & CoS', fontsize=14)
        axes[0, col].set_ylabel('Value', fontsize=12)
        axes[0, col].legend()
        axes[0, col].grid(True, alpha=0.3)
        axes[0, col].set_ylim(-8, 4)
        
        # Row 2: Precondition Node (Activation & Activity)
        axes[1, col].plot(time_steps, log[f'{behavior_name}_precond_activation'][:steps], 
                         'g--', label='Precond Activation', linewidth=2, alpha=0.7)
        axes[1, col].plot(time_steps, log[f'{behavior_name}_precond_activity'][:steps], 
                         'g-', label='Precond Activity', linewidth=2)
        axes[1, col].set_title(f'{behavior_name.title()} - Precondition', fontsize=14)
        axes[1, col].set_ylabel('Value', fontsize=12)
        axes[1, col].legend()
        axes[1, col].grid(True, alpha=0.3)
        axes[1, col].set_ylim(-7, 7)
        
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

    # Row 3: System Level Nodes (Three rows)
    system_col = num_behaviors
    
    # System CoS
    axes[0, system_col].plot(log['steps'], log['system_cos_activation'], 'g-', linewidth=2, label='Activation')
    axes[0, system_col].plot(log['steps'], log['system_cos_activity'], 'g--', linewidth=2, label='Activity')
    axes[0, system_col].set_title('System CoS')
    axes[0, system_col].set_xlabel('Time Steps', fontsize=12)
    axes[0, system_col].set_ylabel('Value', fontsize=12)
    axes[0, system_col].legend()
    axes[0, system_col].grid(True)
    
    # System CoF
    axes[1, system_col].plot(log['steps'], log['system_cof_activation'], 'r-', linewidth=2, label='Activation')
    axes[1, system_col].plot(log['steps'], log['system_cof_activity'], 'r--', linewidth=2, label='Activity')
    axes[1, system_col].set_title('System CoF')
    axes[1, system_col].set_xlabel('Time Steps', fontsize=12)
    axes[1, system_col].set_ylabel('Value', fontsize=12)
    axes[1, system_col].legend()
    axes[1, system_col].grid(True)
    
    # System status overview
    axes[2, system_col].plot(log['steps'], [1.0 if s else 0.0 for s in log['system_success']], 'g-', linewidth=3, label='Success')
    axes[2, system_col].plot(log['steps'], [1.0 if f else 0.0 for f in log['system_failure']], 'r-', linewidth=3, label='Failure')
    axes[2, system_col].set_title('System Status')
    axes[2, system_col].set_ylim(-0.1, 1.1)
    axes[2, system_col].legend()
    axes[2, system_col].grid(True)

    plt.tight_layout()
    plt.show()


import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def animate_fixed_chain(log, behavior_chain):
    G = nx.DiGraph()
    fig, ax = plt.subplots(figsize=(12, 10))

    # Predefine positions: behaviors in a horizontal chain
    pos = {}
    spacing = 15
    for i, level in enumerate(behavior_chain):
        x = i * spacing
        pos[level['name']] = (x-3, 2)  # main behavior node
        pos[f"{level['name']}_cos"] = (x+3, 2)
        pos[f"{level['name']}_precond"] = (x+2, 2.5)
        pos[f"{level['name']}_check"] = (x+2, 1.5)

    # Add interactor nodes below (y = 0)
    interactor_types = ['perception', 'movement', 'gripper', 'state']
    # spacing_interactors = 10
    for i, name in enumerate(interactor_types):
        pos[name] = (i * spacing, 0)

    # Example: what each interactor provides to others
    provided_info = {
        ('perception', 'movement'): ['target_location'],
        ('perception', 'gripper'): ['target_location', 'target_orientation'],
        ('state', 'movement'): ['current_targets'],
        ('state', 'gripper'): ['current_targets'],
    }
    # Map behaviors to what they query from interactors
    consumed_info = {
        'perception': ['raw_sensor_data'],
        'movement': ['target_location', 'current_targets'],
        'gripper': ['target_location', 'target_orientation', 'current_targets'],
        'state': ['status', 'external_commands'],
    }

    def update(frame):
        ax.clear()
        step = log['steps'][frame]
        G.clear()

        edges_to_curve_exc = []
        edges_to_curve_inh = []

        # Add nodes and edges for this step
        for level in behavior_chain:
            bname = level['name']
            active = log[f'{bname}_intention_active'][frame] > 0.5
            G.add_node(bname, active=active)

            # Sub-nodes
            G.add_node(f"{bname}_precond", active=log[f'{bname}_precond_active'][frame] > 0.5)
            G.add_node(f"{bname}_cos", active=log[f'{bname}_cos_active'][frame] > 0.5)
            G.add_node(f"{bname}_check", active=log[f'{bname}_check_confidence_activity'][frame] > 0.5)

            # Connections
            G.add_edge(bname, f"{bname}_cos", weight=5.0)
            edges_to_curve_exc.append((bname, f"{bname}_cos"))

            G.add_edge(f"{bname}_cos", bname, weight=6)
            edges_to_curve_inh.append((f"{bname}_cos", bname))

            G.add_edge(f"{bname}_cos", f"{bname}_precond", weight=6.0)
            G.add_edge(f"{bname}_cos", f"{bname}_check", weight=5.0)

            # Edge from intention to interactors
            interactor_type = level.get('interactor_type', None)
            if interactor_type:
                method = level.get('continuous_method', '')
                G.add_edge(bname, interactor_type, label=method, weight=2.0)
                cos_val = log.get(f'{bname}_cos_activity', [0.0])[frame]
                G.add_edge(interactor_type, f"{bname}_cos", label=f'CoS: {cos_val:.2f}', weight=2.0)

        # Add interactor nodes and dynamic info
        for name in interactor_types:
            G.add_node(name, layer='interactor')
            consumed = ', '.join(consumed_info.get(name, []))
            G.nodes[name]['info'] = f"Consumes: {consumed}"

        # Collect interactor edges
        interactor_edges = [(u, v) for u, v in G.edges if u in interactor_types and v in interactor_types]

        # Precondition → next intention
        for i, level in enumerate(behavior_chain[:-1]):
            next_level = behavior_chain[i+1]
            G.add_edge(f"{level['name']}_precond", next_level['name'], weight=6.0)

        # Draw behavior nodes with activity coloring
        behavior_nodes = [n for n in G.nodes if not n in interactor_types]
        colors = ["green" if G.nodes[n].get("active") else "gray" for n in behavior_nodes]
        nx.draw_networkx_nodes(G, pos, nodelist=behavior_nodes, node_color=colors, ax=ax)
        nx.draw_networkx_labels(G, pos, labels={n: n for n in behavior_nodes}, ax=ax)

        # Draw interactor nodes separately
        nx.draw_networkx_nodes(G, pos, nodelist=interactor_types, node_color='lightblue', node_size=1500, ax=ax)
        nx.draw_networkx_labels(G, pos, labels={n: n for n in interactor_types}, font_size=12, ax=ax)

        # Draw normal edges
        normal_edges = [(u, v) for u, v in G.edges
                        if (u, v) not in edges_to_curve_exc + edges_to_curve_inh + interactor_edges]
        nx.draw_networkx_edges(G, pos, edgelist=normal_edges,
                            width=[G[u][v].get("weight", 1)/2 for u, v in normal_edges],
                            edge_color="green", ax=ax)

        # Draw curved edges
        nx.draw_networkx_edges(G, pos, edgelist=edges_to_curve_exc,
                               width=3, connectionstyle="arc3,rad=0.3",
                               edge_color="green", ax=ax)
        nx.draw_networkx_edges(G, pos, edgelist=edges_to_curve_inh,
                               width=3, connectionstyle="arc3,rad=0.3",
                               edge_color="red", ax=ax)
        # Draw edge labels
        edge_labels = {(u, v): d['label'].replace('_continuous', '')
               for u, v, d in G.edges(data=True) if d.get('label')}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                                    font_color='darkred', font_size=10, ax=ax)

        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='darkred', font_size=10, ax=ax)

        # Annotate interactor info below nodes
        for interactor in interactor_types:
            x, y = pos[interactor]
            ax.text(x, y-0.5, G.nodes[interactor].get('info', ''),
                    ha='center', va='top', fontsize=10, color='gray')

        ax.set_title(f"Step {step}")

    ani = animation.FuncAnimation(fig, update, frames=len(log['steps']), interval=100)
    plt.show()
