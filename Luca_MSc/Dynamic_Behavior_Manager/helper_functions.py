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
    log['system_intention_activation'] = []
    log['system_intention_activity'] = []
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
        log['system_intention_activation'].append(system_state.get('intention_activation', 0.0))
        log['system_intention_activity'].append(system_state.get('intention_activity', 0.0))
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
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Ellipse
import matplotlib.animation as animation


def animate_fixed_chain(log, behavior_chain):
    """
    Animate the behavior chain with a hierarchical layered architecture.
    Organized into System Layer, Behavior Layer, and Robot Layer.
    """
    # Set backend before creating figure
    matplotlib.use('TkAgg')

    G = nx.DiGraph()
    fig, ax = plt.subplots(figsize=(20, 12))

    pos = {}

    # Calculate horizontal spacing based on number of behaviors
    num_behaviors = len(behavior_chain)
    h_spacing = 20 / max(num_behaviors, 1)

    # Layer Y positions
    y_system = 9
    y_behavior_top = 6.5
    y_behavior_mid = 5
    y_behavior_bot = 3.5
    y_selector = 2
    y_robot = 0.5

    # === SYSTEM LAYER ===
    pos['external_input'] = (0, y_system)
    pos['system_cos'] = (h_spacing * num_behaviors / 2 - 1, y_system)
    pos['system_cof'] = (h_spacing * num_behaviors / 2 + 1, y_system)
    pos['system_intention'] = (h_spacing * num_behaviors / 2 - 2.5, y_system)

    # === BEHAVIOR LAYER ===
    for i, level in enumerate(behavior_chain):
        x_base = i * h_spacing + 3
        bname = level['name']

        pos[f"{bname}_intention"] = (x_base - 1.2, y_behavior_top)
        pos[f"{bname}_cos"] = (x_base + 1.2, y_behavior_top)
        pos[f"{bname}_cof"] = (x_base, y_behavior_top + 0.35)
        pos[f"{bname}_precond"] = (x_base + 1.5, y_behavior_mid + 0.5)
        pos[f"{bname}_elementary"] = (x_base - 0.5, y_behavior_bot)
        pos[f"{bname}_check"] = (x_base + 1.5, y_behavior_bot)

    pos['behavior_selector'] = (h_spacing * num_behaviors / 2, y_selector)

    # === ROBOT LAYER ===
    interactor_types = list(
        dict.fromkeys(
            level.get('interactor_type')
            for level in behavior_chain
            if level.get('interactor_type')
        ))
    for i, itype in enumerate(interactor_types):
        x_pos = 2 + i * (h_spacing * num_behaviors / max(len(interactor_types), 1))
        pos[f"interactor_{itype}"] = (x_pos, y_robot)

    def get_node_color(activity_value, threshold=0.5):
        """Convert activity value to color gradient"""
        if activity_value > threshold:
            intensity = min(1.0, activity_value / 2.0)
            return (1.0 - intensity * 0.5, 1.0, 1.0 - intensity * 0.5)
        else:
            return 'white'

    def update(frame):
        ax.clear()
        step = log['steps'][frame]
        G.clear()

        # === DRAW LAYER BOXES ===
        system_box = FancyBboxPatch((0, y_system - 0.8), h_spacing * num_behaviors + 2, 2,
                                    boxstyle="round,pad=0.1", edgecolor='blue',
                                    facecolor='lightblue', alpha=0.1, linewidth=2, zorder=0)
        ax.add_patch(system_box)
        ax.text(0.5, y_system + 1, 'System Layer', fontsize=14, color='blue', weight='bold', zorder=0)

        behavior_box = FancyBboxPatch((0, y_behavior_bot - 1), h_spacing * num_behaviors + 2, 5,
                                      boxstyle="round,pad=0.1", edgecolor='green',
                                      facecolor='lightgreen', alpha=0.1, linewidth=2, zorder=0)
        ax.add_patch(behavior_box)
        ax.text(0.5, y_behavior_top + 0.8, 'Behavior Layer', fontsize=14, color='green', weight='bold', zorder=0)

        robot_box = FancyBboxPatch((0, y_robot - 0.8), h_spacing * num_behaviors + 2, 2,
                                   boxstyle="round,pad=0.1", edgecolor='black',
                                   facecolor='lightgray', alpha=0.1, linewidth=2, zorder=0)
        ax.add_patch(robot_box)
        ax.text(0.5, y_robot + 1, 'Robot Layer', fontsize=14, color='black', weight='bold', zorder=0)

        # === ADD NODES WITH ACTIVITY-BASED COLORS ===
        system_intention_activity = log['system_intention_activity'][frame]
        system_cos_activity = log['system_cos_activity'][frame]
        system_cof_activity = log['system_cof_activity'][frame]

        G.add_node('system_cos', activity=system_cos_activity, label='CoS')
        G.add_node('system_cof', activity=system_cof_activity, label='CoF')
        G.add_node('system_intention', activity=system_intention_activity, label='I')
        G.add_node('external_input', activity=0.0, label='External Input')

        interactor_activations = {f"interactor_{itype}": 0.0 for itype in interactor_types}

        for level in behavior_chain:
            bname = level['name']

            intention_activity = log[f'{bname}_intention_activity'][frame]
            cos_activity = log[f'{bname}_cos_activity'][frame]
            cof_activity = log[f'{bname}_cof_activity'][frame]
            precond_activity = log[f'{bname}_precond_activity'][frame]
            check_activity = log[f'{bname}_check_confidence_activity'][frame]

            # Chain activation: if intention is active or sanity check is being performed, activate interactor
            interactor_type = level.get('interactor_type')
            if interactor_type and intention_activity > 0.5 or check_activity > 0.7:
                interactor_activations[f"interactor_{interactor_type}"] = 1.0

            G.add_node(f"{bname}_intention", activity=intention_activity, label='I')
            G.add_node(f"{bname}_cos", activity=cos_activity, label='CoS')
            G.add_node(f"{bname}_cof", activity=cof_activity, label='CoF')
            G.add_node(f"{bname}_precond", activity=precond_activity, label='Precond.')
            G.add_node(f"{bname}_elementary", activity=intention_activity,
                       label=f'{bname}')

            G.add_node(f"{bname}_check", activity=check_activity, label='Check')

        G.add_node('behavior_selector', activity=0.0, label='Behavior\nSelector')

        for itype in interactor_types:
            G.add_node(f"interactor_{itype}", activity=interactor_activations[f"interactor_{itype}"],
                       label=f'Robot\nInteractor\n({itype})')

        # === ADD EDGES ===
        # System layer edges
        G.add_edge('external_input', 'system_intention', color='green')
        G.add_edge('system_cos', 'system_cof', color='red')
        G.add_edge('system_cof', 'system_cos', color='red')

        # System to Behavior layer edges
        for level in behavior_chain:
            bname = level['name']
            G.add_edge(f"{bname}_cos", 'system_cos', color='green')
            G.add_edge(f"{bname}_cof", 'system_cof', color='green')
            G.add_edge('system_intention', f"{bname}_intention", color='green')
            G.add_edge("system_intention", f"{bname}_precond", color='green')

        for i, level in enumerate(behavior_chain):
            bname = level['name']

            G.add_edge(f"{bname}_intention", f"{bname}_elementary", color='black')
            G.add_edge(f"{bname}_cof", f"{bname}_intention", color='red')
            G.add_edge(f"{bname}_cof", f"{bname}_cos", color='red')
            G.add_edge(f"{bname}_cos", f"{bname}_precond", color='red')
            G.add_edge(f"{bname}_cos", f"{bname}_cof", color='red')
            G.add_edge(f"{bname}_cos", f"{bname}_intention", color='red')
            G.add_edge(f"{bname}_cos", f"{bname}_check", color='green')
            G.add_edge(f"{bname}_elementary", 'behavior_selector', color='black')

            if i < len(behavior_chain) - 1:
                next_bname = behavior_chain[i + 1]['name']
                G.add_edge(f"{bname}_precond", f"{next_bname}_intention", color='red')

            interactor_type = level.get('interactor_type')
            if interactor_type:
                G.add_edge('behavior_selector', f"interactor_{interactor_type}", color='blue')
                G.add_edge(f"{bname}_check", f"interactor_{interactor_type}", color='blue')
                G.add_edge(f"interactor_{interactor_type}", f"{bname}_cos", color='blue')
                G.add_edge(f"interactor_{interactor_type}", f"{bname}_cof", color='blue')

        # === DRAW NODES WITH ACTIVITY COLORS ===
        for node in G.nodes():
            x, y = pos[node]
            radius = 0.3
            activity = G.nodes[node].get('activity', 0.0)
            label = G.nodes[node].get('label', node)

            if 'elementary' in node:
                color = get_node_color(activity, threshold=0.5)
                shape = 'box'
            elif 'interactor' in node:
                color = get_node_color(activity, threshold=0.5)
                if color == 'white':
                    color = 'lightgray'
                shape = 'box'
            elif node == 'behavior_selector':
                color = 'lightyellow'
                shape = 'box'
            elif node == 'external_input':
                radius = 0.9
                shape = 'ellipse'
            elif "_check" in node:
                color = get_node_color(activity, threshold=0.7)
                radius = 0.45
                shape = 'ellipse'
            elif "_precond" in node:
                radius = 0.6
                shape = 'ellipse'
                color = get_node_color(activity, threshold=0.5)
            else:
                color = get_node_color(activity, threshold=0.1)
                shape = 'circle'

            if shape == 'box':
                bbox = FancyBboxPatch((x - 0.8, y - 0.4), 1.6, 0.8,
                                      boxstyle="round,pad=0.2",
                                      edgecolor='black', facecolor=color, linewidth=2, zorder=5)
                ax.add_patch(bbox)

            elif shape == 'ellipse':
                ellipse = Ellipse((x, y), width=radius * 2, height=radius * 1.2,
                                  edgecolor='black', facecolor=color, linewidth=2, zorder=5)
                ax.add_patch(ellipse)

            else:
                circle = plt.Circle((x, y), radius, color=color, ec='black', linewidth=2, zorder=5)
                ax.add_patch(circle)

            ax.text(x, y, label, ha='center', va='center', fontsize=10, weight='bold', zorder=6)

        # === DRAW EDGES ===
        for u, v, data in G.edges(data=True):
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            color = data.get('color', 'black')
            style = data.get('style', 'solid')

            # Calculate direction vector
            dx = x2 - x1
            dy = y2 - y1
            dist = np.sqrt(dx**2 + dy**2)

            if dist == 0:
                continue  # Avoid division by zero

            dx_norm = dx / dist
            dy_norm = dy / dist

            # Determine node radii
            # Circles have radius 0.3, boxes have radius based on corner distance
            if 'elementary' in u or 'interactor' in u or u == 'behavior_selector':
                radius_u = 1.1 # Approx half box width
            else:
                radius_u = 0.35

            if 'interactor' in v or v == 'behavior_selector':
                radius_v = 1.1
            else:
                radius_v = 0.35

            # Reduce radius for boxes with very short distances between them
            if dist < (radius_u + radius_v):
                radius_u = dist / 2.5
                radius_v = dist / 2.5

            # Adjust start and end points
            start_x = x1 + dx_norm * radius_u
            start_y = y1 + dy_norm * radius_u
            end_x = x2 - dx_norm * radius_v
            end_y = y2 - dy_norm * radius_v

            ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                        arrowprops=dict(arrowstyle='->', color=color, lw=1.5, linestyle=style),
                        zorder=1)

        ax.set_xlim(-1, h_spacing * num_behaviors + 3)
        ax.set_ylim(-2, y_system + 2)
        ax.axis('off')
        ax.set_title(f"Behavior Chain Architecture - Step {step} / {len(log['steps'])}",
                     fontsize=16, weight='bold')

    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=len(log['steps']), interval=200, repeat=True, blit=False)

    # Show with explicit draw
    plt.draw()
    plt.pause(0.001)
    plt.show(block=True)  # Block to keep window open

    return ani
