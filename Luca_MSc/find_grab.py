import torch
import numpy as np
import matplotlib.pyplot as plt
from DNF_torch.field import Field
from interactors import RobotInteractors

from find import FindBehavior
from move_to import MoveToBehavior


class FindGrabBehavior:
    """
    Composite behavior that chains find and grab behaviors.
    """

    def __init__(self):
        # Create component behaviors
        self.find_behavior = FindBehavior()
        self.move_to_behavior = MoveToBehavior()

        # Create precondition nodes

        # Object found precondition
        self.found_precond = Field(
            shape=(),
            time_step=5.0,
            time_scale=100.0,
            resting_level=-3.0,
            beta=20.0,
            self_connection_w0=3.5
        )

        # Close enough to target
        self.close_precond = Field(
            shape=(),
            time_step=5.0,
            time_scale=100.0,
            resting_level=-3.0,
            beta=20.0,
            self_connection_w0=3.5
        )

        # Register buffer for prev state
        self.found_precond.register_buffer(
            "g_u_prev",
            torch.zeros_like(self.found_precond.g_u)
        )
        self.close_precond.register_buffer(
            "g_u_prev",
            torch.zeros_like(self.close_precond.g_u)
        )

        # Connections
        self.find_behavior.CoS.connection_to(self.found_precond, 15.0)

        #self.found_precond.connection_to(self.move_to_behavior.intention, 6.0)
        self.move_to_behavior.CoS.connection_to(self.close_precond, 10.0)


    def execute_step(self, interactors, target_name, external_input=5.0):
        """Execute the find portion of the behavior chain."""

        # Cache prev state for the precondition
        self.found_precond.cache_prev()
        self.close_precond.cache_prev()

        # Execute find behavior
        find_state = self.find_behavior.execute(interactors.perception, target_name, external_input)

        # Process the found precondition node (no external input)
        found_activation, found_activity = self.found_precond()
        close_activation, close_activity = self.close_precond()

        # Execute move-to behavior with precondition input, only if found is active
        move_state = None
        if find_state['target_location'] is not None:
            move_state = self.move_to_behavior.execute(
                interactors.movement,
                find_state['target_location']
            )
        # Get current robot position from movement interactor
        robot_position = interactors.movement.get_position()

        state = {
            'find': find_state,
            'move': move_state,
            'preconditions': {
                'found': {
                    'activation': float(found_activation.detach()),
                    'activity': float(found_activity.detach()),
                    'active': float(found_activity) > 0.7
                },
                'close': {
                    'activation': float(close_activation.detach()),
                    'activity': float(close_activity.detach()),
                    'active': float(close_activity) > 0.7
                }
            },
            'robot':{
                'position': robot_position.tolist()
            },
        }

        return state

    def reset(self):
        """Reset all behaviors and preconditions."""
        self.find_behavior.reset()
        self.found_precond.reset()
        self.move_to_behavior.reset()
        self.close_precond.reset()


# Example usage
if __name__ == "__main__":
    # Log all activations and activities for plotting
    log = {"intention_activation": [],
           "intention_activity": [],
           "cos_activation": [],
           "cos_activity": [],
           "found_precond_activation": [],
           "found_precond_activity": [],
           "move_intention_activation": [],
           "move_intention_activity": [],
           "move_cos_activation": [],
           "move_cos_activity": [],
           "close_precond_activation": [],
           "close_precond_activity": []}

    # External input activates find_grab behavior sequence
    external_input = 5.0


    # Create interactors with a test object
    interactors = RobotInteractors()
    interactors.perception.register_object("cup", torch.tensor([1.2, 0.5, 0.8]))

    # Create the find-grab behavior
    find_grab = FindGrabBehavior()

    # Run the find behavior until completion
    print("Starting find behavior for 'cup'...")
    i = 0
    done = 0
    for step in range(100):
        # Execute find behavior
        state = find_grab.execute_step(interactors, "cup", external_input)

        # Log activities for plotting
        log["intention_activation"].append(state['find']['intention_activation'])
        log["intention_activity"].append(state['find']['intention_activity'])
        log["cos_activation"].append(state['find']['cos_activation'])
        log["cos_activity"].append(state['find']['cos_activity'])
        log["found_precond_activation"].append(state['preconditions']['found']['activation'])
        log["found_precond_activity"].append(state['preconditions']['found']['activity'])
        if state['move'] is not None:
            log["move_intention_activation"].append(state['move']['intention_activation'])
            log["move_intention_activity"].append(state['move']['intention_activity'])
            log["move_cos_activation"].append(state['move']['cos_activation'])
            log["move_cos_activity"].append(state['move']['cos_activity'])
            log["close_precond_activation"].append(state['preconditions']['close']['activation'])
            log["close_precond_activity"].append(state['preconditions']['close']['activity'])
        else:
            log["move_intention_activation"].append(-3.0)
            log["move_intention_activity"].append(0.0)
            log["move_cos_activation"].append(-3.0)
            log["move_cos_activity"].append(0.0)
            log["close_precond_activation"].append(-3.0)
            log["close_precond_activity"].append(0.0)

        # Print status
        print(f"Step {step}: Active={state['find']['active']}, Completed={state['find']['completed']}")
        print(f"Active Movement={state['move']['active'] if state['move'] else False}, Completed Movement={state['move']['completed'] if state['move'] else False}")
        # print(f"  Found={state['target_found']}, "
        #       f"Location={state['target_location']}")
        # print(f"  Intention={state['intention_activity']:.2f}, "
        #       f"CoS={state['cos_activity']:.2f}, "
        #       f"Found Precond={state['found_precond_activity']:.2f}")
        if state['move'] is not None:
            print(f" Move Intention Activation = {state['move']['intention_activation']} "
                  f" Move CoS Activation = {state['move']['cos_activation']}"
                  f" Close Precond Activation = {state['preconditions']['close']['activation']}")

        i += 1
        # Stop if completed
        if state['move'] is not None and state['move']['completed']:
            done += 1
            print(f"Reached target at step {step}!")

        if done >= 10:
            break

    # Plotting the activities of all nodes over time
    ts = np.arange(i)
    plt.figure(figsize=(8,4))
    plt.plot(ts, log["intention_activation"], label="Intention Find (activation)")
    plt.plot(ts, log["cos_activation"], label="CoS Find (activation)")
    plt.plot(ts, log["intention_activity"], '--', label="Intention Find (activity)")
    plt.plot(ts, log["cos_activity"], '--', label="CoS Find (activity)")
    plt.plot(ts, log["found_precond_activation"], label="Found Precond (activation)")
    plt.plot(ts, log["found_precond_activity"], '--', label="Found Precond (activity)")
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8,4))
    plt.plot(ts, log["move_intention_activation"], label="Intention Move (activation)")
    plt.plot(ts, log["move_cos_activation"], label="CoS Move (activation)")
    plt.plot(ts, log["move_intention_activity"], '--', label="Intention Move (activity)")
    plt.plot(ts, log["move_cos_activity"], '--', label="CoS Move (activity)")
    plt.plot(ts, log["close_precond_activation"], label="Close Precond (activation)")
    plt.plot(ts, log["close_precond_activity"], '--', label="Close Precond (activity)")
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.show()