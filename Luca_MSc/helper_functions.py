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
def initalize_log():
    """Initialize the log dictionary with all keys and empty lists."""
    keys = [
        "find_intention_activation", "find_intention_activity",
        "find_cos_activation", "find_cos_activity",
        "found_precond_activation", "found_precond_activity",
        "move_intention_activation", "move_intention_activity",
        "move_cos_activation", "move_cos_activity",
        "close_precond_activation", "close_precond_activity",
        "orient_intention_activation", "orient_intention_activity",
        "orient_cos_activation", "orient_cos_activity",
        "oriented_precond_activation", "oriented_precond_activity",
        "reach_intention_activation", "reach_intention_activity",
        "reach_cos_activation", "reach_cos_activity",
        "in_reach_precond_activation", "in_reach_precond_activity",
        "reach_for_intention_activation", "reach_for_intention_activity",
        "reach_for_cos_activation", "reach_for_cos_activity",
        "reached_precond_activation", "reached_precond_activity",

        # Grab top-level behavior
        "grab_intention_activation", "grab_intention_activity",
        "grab_cos_activation", "grab_cos_activity",

        # Grab sub-behaviors
        "grab_orient_intention_activation", "grab_orient_intention_activity",
        "grab_orient_cos_activation", "grab_orient_cos_activity",
        "grab_open_gripper_intention_activation", "grab_open_gripper_intention_activity",
        "grab_open_gripper_cos_activation", "grab_open_gripper_cos_activity",
        "grab_fine_reach_intention_activation", "grab_fine_reach_intention_activity",
        "grab_fine_reach_cos_activation", "grab_fine_reach_cos_activity",
        "grab_close_gripper_intention_activation", "grab_close_gripper_intention_activity",
        "grab_close_gripper_cos_activation", "grab_close_gripper_cos_activity",
        "grab_has_object_intention_activation", "grab_has_object_intention_activity",
        "grab_has_object_cos_activation", "grab_has_object_cos_activity",
        "has_grabbed_precond_activation", "has_grabbed_precond_activity"
    ]
    return {k: [] for k in keys}


def update_log(log, state):
    """Append values from the current state into the log dict."""

    # Find block
    if state['find'] is not None:
        log["find_intention_activation"].append(state['find']['intention_activation'])
        log["find_intention_activity"].append(state['find']['intention_activity'])
        log["find_cos_activation"].append(state['find']['cos_activation'])
        log["find_cos_activity"].append(state['find']['cos_activity'])
        log["found_precond_activation"].append(state['preconditions']['found']['activation'])
        log["found_precond_activity"].append(state['preconditions']['found']['activity'])
    else:
        for k in ["find_intention_activation", "find_intention_activity",
                  "find_cos_activation", "find_cos_activity",
                  "found_precond_activation", "found_precond_activity"]:
            log[k].append(-3.0 if "activation" in k else 0.0)

    # Move block
    if state['move'] is not None:
        log["move_intention_activation"].append(state['move']['intention_activation'])
        log["move_intention_activity"].append(state['move']['intention_activity'])
        log["move_cos_activation"].append(state['move']['cos_activation'])
        log["move_cos_activity"].append(state['move']['cos_activity'])
        log["close_precond_activation"].append(state['preconditions']['close']['activation'])
        log["close_precond_activity"].append(state['preconditions']['close']['activity'])
    else:
        for k in ["move_intention_activation", "move_intention_activity",
                  "move_cos_activation", "move_cos_activity",
                  "close_precond_activation", "close_precond_activity"]:
            log[k].append(-3.0 if "activation" in k else 0.0)

    # In reach block
    if state['in_reach'] is not None:
        log["reach_intention_activation"].append(state['in_reach']['intention_activation'])
        log["reach_intention_activity"].append(state['in_reach']['intention_activity'])
        log["reach_cos_activation"].append(state['in_reach']['cos_activation'])
        log["reach_cos_activity"].append(state['in_reach']['cos_activity'])
        log["in_reach_precond_activation"].append(state['preconditions']['in_reach']['activation'])
        log["in_reach_precond_activity"].append(state['preconditions']['in_reach']['activity'])
    else:
        for k in ["reach_intention_activation", "reach_intention_activity",
                  "reach_cos_activation", "reach_cos_activity",
                  "in_reach_precond_activation", "in_reach_precond_activity"]:
            log[k].append(-3.0 if "activation" in k else 0.0)

    # Reach-for block
    if state['reach_for'] is not None:
        log["reach_for_intention_activation"].append(state['reach_for']['intention_activation'])
        log["reach_for_intention_activity"].append(state['reach_for']['intention_activity'])
        log["reach_for_cos_activation"].append(state['reach_for']['cos_activation'])
        log["reach_for_cos_activity"].append(state['reach_for']['cos_activity'])
        log["reached_precond_activation"].append(state['preconditions']['reached']['activation'])
        log["reached_precond_activity"].append(state['preconditions']['reached']['activity'])
    else:
        for k in ["reach_for_intention_activation", "reach_for_intention_activity",
                  "reach_for_cos_activation", "reach_for_cos_activity",
                  "reached_precond_activation", "reached_precond_activity"]:
            log[k].append(-3.0 if "activation" in k else 0.0)

    # Grab block with its sub-behaviors
    if state['grab'] is not None:
        # Top-level grab behavior
        if 'grab' in state['grab']:
            log["grab_intention_activation"].append(state['grab']['grab']['intention_activation'])
            log["grab_intention_activity"].append(state['grab']['grab']['intention_activity'])
            log["grab_cos_activation"].append(state['grab']['grab']['cos_activation'])
            log["grab_cos_activity"].append(state['grab']['grab']['cos_activity'])
        else:
            for k in ["grab_intention_activation", "grab_intention_activity",
                      "grab_cos_activation", "grab_cos_activity"]:
                log[k].append(-3.0 if "activation" in k else 0.0)

        # Orient sub-behavior
        if 'orient' in state['grab']:
            log["grab_orient_intention_activation"].append(state['grab']['orient']['intention_activation'])
            log["grab_orient_intention_activity"].append(state['grab']['orient']['intention_activity'])
            log["grab_orient_cos_activation"].append(state['grab']['orient']['cos_activation'])
            log["grab_orient_cos_activity"].append(state['grab']['orient']['cos_activity'])
        else:
            for k in ["grab_orient_intention_activation", "grab_orient_intention_activity",
                      "grab_orient_cos_activation", "grab_orient_cos_activity"]:
                log[k].append(-3.0 if "activation" in k else 0.0)

        # Open gripper sub-behavior
        if 'open_gripper' in state['grab']:
            log["grab_open_gripper_intention_activation"].append(state['grab']['open_gripper']['intention_activation'])
            log["grab_open_gripper_intention_activity"].append(state['grab']['open_gripper']['intention_activity'])
            log["grab_open_gripper_cos_activation"].append(state['grab']['open_gripper']['cos_activation'])
            log["grab_open_gripper_cos_activity"].append(state['grab']['open_gripper']['cos_activity'])
        else:
            for k in ["grab_open_gripper_intention_activation", "grab_open_gripper_intention_activity",
                      "grab_open_gripper_cos_activation", "grab_open_gripper_cos_activity"]:
                log[k].append(-3.0 if "activation" in k else 0.0)

        # Fine reach sub-behavior
        if 'fine_reach' in state['grab']:
            log["grab_fine_reach_intention_activation"].append(state['grab']['fine_reach']['intention_activation'])
            log["grab_fine_reach_intention_activity"].append(state['grab']['fine_reach']['intention_activity'])
            log["grab_fine_reach_cos_activation"].append(state['grab']['fine_reach']['cos_activation'])
            log["grab_fine_reach_cos_activity"].append(state['grab']['fine_reach']['cos_activity'])
        else:
            for k in ["grab_fine_reach_intention_activation", "grab_fine_reach_intention_activity",
                      "grab_fine_reach_cos_activation", "grab_fine_reach_cos_activity"]:
                log[k].append(-3.0 if "activation" in k else 0.0)

        # Close gripper sub-behavior
        if 'close_gripper' in state['grab']:
            log["grab_close_gripper_intention_activation"].append(state['grab']['close_gripper']['intention_activation'])
            log["grab_close_gripper_intention_activity"].append(state['grab']['close_gripper']['intention_activity'])
            log["grab_close_gripper_cos_activation"].append(state['grab']['close_gripper']['cos_activation'])
            log["grab_close_gripper_cos_activity"].append(state['grab']['close_gripper']['cos_activity'])
        else:
            for k in ["grab_close_gripper_intention_activation", "grab_close_gripper_intention_activity",
                      "grab_close_gripper_cos_activation", "grab_close_gripper_cos_activity"]:
                log[k].append(-3.0 if "activation" in k else 0.0)

        # Has object sub-behavior
        if 'has_object' in state['grab']:
            log["grab_has_object_intention_activation"].append(state['grab']['has_object']['intention_activation'])
            log["grab_has_object_intention_activity"].append(state['grab']['has_object']['intention_activity'])
            log["grab_has_object_cos_activation"].append(state['grab']['has_object']['cos_activation'])
            log["grab_has_object_cos_activity"].append(state['grab']['has_object']['cos_activity'])
        else:
            for k in ["grab_has_object_intention_activation", "grab_has_object_intention_activity",
                      "grab_has_object_cos_activation", "grab_has_object_cos_activity"]:
                log[k].append(-3.0 if "activation" in k else 0.0)

        # Has grabbed precondition
        log["has_grabbed_precond_activation"].append(state['preconditions']['has_grabbed']['activation'])
        log["has_grabbed_precond_activity"].append(state['preconditions']['has_grabbed']['activity'])
    else:
        for k in ["grab_intention_activation", "grab_intention_activity",
                  "grab_cos_activation", "grab_cos_activity",
                  "grab_orient_intention_activation", "grab_orient_intention_activity",
                  "grab_orient_cos_activation", "grab_orient_cos_activity",
                  "grab_open_gripper_intention_activation", "grab_open_gripper_intention_activity",
                  "grab_open_gripper_cos_activation", "grab_open_gripper_cos_activity",
                  "grab_fine_reach_intention_activation", "grab_fine_reach_intention_activity",
                  "grab_fine_reach_cos_activation", "grab_fine_reach_cos_activity",
                  "grab_close_gripper_intention_activation", "grab_close_gripper_intention_activity",
                  "grab_close_gripper_cos_activation", "grab_close_gripper_cos_activity",
                  "grab_has_object_intention_activation", "grab_has_object_intention_activity",
                  "grab_has_object_cos_activation", "grab_has_object_cos_activity",
                  "has_grabbed_precond_activation", "has_grabbed_precond_activity"]:
            log[k].append(-3.0 if "activation" in k else 0.0)

# Plotting the nodes

def plot_logs(log, steps):
    ts = np.arange(steps)

    # Define groups of signals to plot together
    groups = [
        (
            "Find",
            [
                ("find_intention_activation", "Intention Find (activation)", "-"),
                ("find_cos_activation", "CoS Find (activation)", "-"),
                ("find_intention_activity", "Intention Find (activity)", "--"),
                ("find_cos_activity", "CoS Find (activity)", "--"),
                ("found_precond_activation", "Found Precond (activation)", "-"),
                ("found_precond_activity", "Found Precond (activity)", "--"),
            ],
        ),
        (
            "Move",
            [
                ("move_intention_activation", "Intention Move (activation)", "-"),
                ("move_cos_activation", "CoS Move (activation)", "-"),
                ("move_intention_activity", "Intention Move (activity)", "--"),
                ("move_cos_activity", "CoS Move (activity)", "--"),
                ("close_precond_activation", "Close Precond (activation)", "-"),
                ("close_precond_activity", "Close Precond (activity)", "--"),
            ],
        ),
        (
            "Reach Check",
            [
                ("reach_intention_activation", "Intention InReach (activation)", "-"),
                ("reach_cos_activation", "CoS InReach (activation)", "-"),
                ("reach_intention_activity", "Intention InReach (activity)", "--"),
                ("reach_cos_activity", "CoS InReach (activity)", "--"),
                ("in_reach_precond_activation", "In Reach Precond (activation)", "-"),
                ("in_reach_precond_activity", "In Reach Precond (activity)", "--"),
            ],
        ),
        (
            "ReachFor",
            [
                ("reach_for_intention_activation", "Intention ReachFor (activation)", "-"),
                ("reach_for_cos_activation", "CoS ReachFor (activation)", "-"),
                ("reach_for_intention_activity", "Intention ReachFor (activity)", "--"),
                ("reach_for_cos_activity", "CoS ReachFor (activity)", "--"),
                ("reached_precond_activation", "Reached Precond (activation)", "-"),
                ("reached_precond_activity", "Reached Precond (activity)", "--"),
            ],
        ),
        (
            "Grab (Main)",
            [
                ("grab_intention_activation", "Intention Grab (activation)", "-"),
                ("grab_cos_activation", "CoS Grab (activation)", "-"),
                ("grab_intention_activity", "Intention Grab (activity)", "--"),
                ("grab_cos_activity", "CoS Grab (activity)", "--"),
                ("has_grabbed_precond_activation", "Has Grabbed Precond (activation)", "-"),
                ("has_grabbed_precond_activity", "Has Grabbed Precond (activity)", "--"),
            ],
        ),
        (
            "Grab - Orient",
            [
                ("grab_orient_intention_activation", "Intention Orient (activation)", "-"),
                ("grab_orient_cos_activation", "CoS Orient (activation)", "-"),
                ("grab_orient_intention_activity", "Intention Orient (activity)", "--"),
                ("grab_orient_cos_activity", "CoS Orient (activity)", "--"),
            ],
        ),
        (
            "Grab - Open Gripper",
            [
                ("grab_open_gripper_intention_activation", "Intention Open (activation)", "-"),
                ("grab_open_gripper_cos_activation", "CoS Open (activation)", "-"),
                ("grab_open_gripper_intention_activity", "Intention Open (activity)", "--"),
                ("grab_open_gripper_cos_activity", "CoS Open (activity)", "--"),
            ],
        ),
        (
            "Grab - Fine Reach",
            [
                ("grab_fine_reach_intention_activation", "Intention FineReach (activation)", "-"),
                ("grab_fine_reach_cos_activation", "CoS FineReach (activation)", "-"),
                ("grab_fine_reach_intention_activity", "Intention FineReach (activity)", "--"),
                ("grab_fine_reach_cos_activity", "CoS FineReach (activity)", "--"),
            ],
        ),
        (
            "Grab - Close Gripper",
            [
                ("grab_close_gripper_intention_activation", "Intention Close (activation)", "-"),
                ("grab_close_gripper_cos_activation", "CoS Close (activation)", "-"),
                ("grab_close_gripper_intention_activity", "Intention Close (activity)", "--"),
                ("grab_close_gripper_cos_activity", "CoS Close (activity)", "--"),
            ],
        ),
        (
            "Grab - Has Object",
            [
                ("grab_has_object_intention_activation", "Intention HasObject (activation)", "-"),
                ("grab_has_object_cos_activation", "CoS HasObject (activation)", "-"),
                ("grab_has_object_intention_activity", "Intention HasObject (activity)", "--"),
                ("grab_has_object_cos_activity", "CoS HasObject (activity)", "--"),
            ],
        )
    ]

    fig, axes = plt.subplots(len(groups), 1, figsize=(12, 22), sharex=True)

    for ax, (title, signals) in zip(axes, groups):
        for key, label, style in signals:
            ax.plot(ts, log[key], style, label=label)
        ax.set_ylabel("Value")
        ax.set_title(title)
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Step")
    plt.tight_layout()
    plt.show()
