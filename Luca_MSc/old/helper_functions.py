import torch
import numpy as np
import matplotlib.pyplot as plt

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

# Define behavior structure
BEHAVIOR_STRUCTURE = {
    # Top-level behaviors
    "find": {"path": "find", "type": "behavior"},
    "move": {"path": "move", "type": "behavior"},
    "in_reach": {"path": "in_reach", "type": "behavior"},
    "reach_for": {"path": "reach_for", "type": "behavior"},
    "transport": {"path": "transport", "type": "behavior"},

    # Grab behavior and sub-behaviors
    "grab": {"path": "grab.grab", "type": "behavior"},
    "grab_orient": {"path": "grab.orient", "type": "behavior"},
    "grab_open_gripper": {"path": "grab.open_gripper", "type": "behavior"},
    "grab_fine_reach": {"path": "grab.fine_reach", "type": "behavior"},
    "grab_close_gripper": {"path": "grab.close_gripper", "type": "behavior"},
    "grab_has_object": {"path": "grab.has_object", "type": "behavior"},
    "toggle_node": {"path": "grab.toggle_node", "type": "node"},

    # Preconditions
    "found_precond": {"path": "preconditions.found", "type": "precondition"},
    "close_precond": {"path": "preconditions.close", "type": "precondition"},
    "in_reach_precond": {"path": "preconditions.in_reach", "type": "precondition"},
    "reached_precond": {"path": "preconditions.reached", "type": "precondition"},
    "has_grabbed_precond": {"path": "preconditions.has_grabbed", "type": "precondition"},
    "transported_precond": {"path": "preconditions.transported", "type": "precondition"}
}

# Group definitions for plotting
PLOT_GROUPS = [
    ("Find", ["find", "found_precond"]),
    ("Move", ["move", "close_precond"]),
    ("Reach Check", ["in_reach", "in_reach_precond"]),
    ("ReachFor", ["reach_for", "reached_precond"]),
    ("Grab (Main)", ["grab", "has_grabbed_precond"]),
    ("Grab - Orient", ["grab_orient"]),
    ("Grab - Open Gripper", ["grab_open_gripper"]),
    ("Grab - Fine Reach", ["grab_fine_reach"]),
    ("Grab - Toggle Node", ["toggle_node"]),
    ("Grab - Close Gripper", ["grab_close_gripper"]),
    ("Grab - Has Object", ["grab_has_object"]),
    ("Transport", ["transport", "transported_precond"])
]


def get_nested_value(data_dict, path_str):
    """Helper to safely get nested values from dictionary using dot notation."""
    if data_dict is None:
        return None

    parts = path_str.split('.')
    current = data_dict
    for part in parts:
        if not current[part]:
            return None
        elif part not in current:
            return None
        current = current[part]
    return current


def initalize_log():
    """Initialize the log dictionary with all keys and empty lists."""
    log = {}

    # Generate keys for each behavior
    for behavior_id, behavior_info in BEHAVIOR_STRUCTURE.items():
        if behavior_info["type"] == "node" or behavior_info["type"] == "precondition":
            # Special case for nodes that only have activation/activity
            log[f"{behavior_id}_activation"] = []
            log[f"{behavior_id}_activity"] = []
        else:
            # Regular behavior with intention/CoS
            log[f"{behavior_id}_intention_activation"] = []
            log[f"{behavior_id}_intention_activity"] = []
            log[f"{behavior_id}_cos_activation"] = []
            log[f"{behavior_id}_cos_activity"] = []

    return log


def update_log(log, state):
    """Append values from the current state into the log dict."""

    # Process each behavior defined in our structure
    for behavior_id, behavior_info in BEHAVIOR_STRUCTURE.items():
        # Get the nested value using the defined path
        behavior_state = get_nested_value(state, behavior_info["path"])

        if behavior_info["type"] == "node" or behavior_info["type"] == "precondition":
            # Handle special case for nodes (only activation/activity)
            if behavior_state and "activation" in behavior_state and "activity" in behavior_state:
                log[f"{behavior_id}_activation"].append(behavior_state["activation"])
                log[f"{behavior_id}_activity"].append(behavior_state["activity"])
            else:
                log[f"{behavior_id}_activation"].append(-3.0)
                log[f"{behavior_id}_activity"].append(0.0)
        else:
            # Handle regular behavior with intention/CoS
            if behavior_state:
                log[f"{behavior_id}_intention_activation"].append(behavior_state["intention_activation"])
                log[f"{behavior_id}_intention_activity"].append(behavior_state["intention_activity"])
                log[f"{behavior_id}_cos_activation"].append(behavior_state["cos_activation"])
                log[f"{behavior_id}_cos_activity"].append(behavior_state["cos_activity"])
            else:
                log[f"{behavior_id}_intention_activation"].append(-3.0)
                log[f"{behavior_id}_intention_activity"].append(0.0)
                log[f"{behavior_id}_cos_activation"].append(-3.0)
                log[f"{behavior_id}_cos_activity"].append(0.0)


def plot_logs(log, steps):
    """Plot all behavior signals organized by groups."""
    ts = np.arange(steps)

    # Create plot groups based on our defined structure
    plot_data = []
    for title, behavior_ids in PLOT_GROUPS:
        signals = []
        for behavior_id in behavior_ids:
            behavior_info = BEHAVIOR_STRUCTURE.get(behavior_id)

            if behavior_info["type"] == "node" or behavior_info["type"] == "precondition":
                # Special case for node with only activation/activity
                signals.append(
                    (f"{behavior_id}_activation", f"{behavior_id.replace('_', ' ').title()} (activation)", "-"))
                signals.append((f"{behavior_id}_activity", f"{behavior_id.replace('_', ' ').title()} (activity)", "--"))
            else:
                # Regular behavior with intention/CoS
                signals.append((f"{behavior_id}_intention_activation",
                                f"Intention {behavior_id.replace('_', ' ').title()} (activation)", "-"))
                signals.append(
                    (f"{behavior_id}_cos_activation", f"CoS {behavior_id.replace('_', ' ').title()} (activation)", "-"))
                signals.append((f"{behavior_id}_intention_activity",
                                f"Intention {behavior_id.replace('_', ' ').title()} (activity)", "--"))
                signals.append(
                    (f"{behavior_id}_cos_activity", f"CoS {behavior_id.replace('_', ' ').title()} (activity)", "--"))

        plot_data.append((title, signals))

    # Create the plots
    fig, axes = plt.subplots(len(plot_data), 1, figsize=(12, 22), sharex=True)

    for ax, (title, signals) in zip(axes, plot_data):
        for key, label, style in signals:
            ax.plot(ts, log[key], style, label=label)
        ax.set_ylabel("Value")
        ax.set_title(title)
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Step")
    plt.tight_layout()
    plt.show()
