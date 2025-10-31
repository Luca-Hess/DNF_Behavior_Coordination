import sys
import os
# Add DNF_torch package root
sys.path.append(os.path.expanduser('~/nc_ws/DNF_torch'))

# Add Luca_MSc subfolder for local scripts
sys.path.append(os.path.join(os.path.expanduser('~/nc_ws/DNF_torch'), 'Luca_MSc/latching CoS experiment'))

import torch
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from DNF_torch.field import Field
from SimulationVisualizer import RobotSimulationVisualizer
from interactors import RobotInteractors

from find_int import FindBehavior_IntentionCoupling
from move_to import MoveToBehavior
from check_effector_range import CheckEffectorRange
from reach_for import ReachForBehavior
from grab_behavior_compound import GrabBehavior

from helper_functions import move_object, initalize_log, update_log, plot_logs

class FindGrabBehavior():
    """
    Composite behavior that chains find and grab behaviors.
    """

    def __init__(self):
        # Create component behaviors
        self.find_behavior = FindBehavior_IntentionCoupling()
        self.move_to_behavior = MoveToBehavior()

        # Flag for object movement - simulating object changing location!
        self.object_moved = False
        self.object_dropped = False
        self.picked_up = False

        ## Create precondition nodes
        # Object found precondition
        self.found_precond = Field(
            shape=(),
            time_step=5.0,
            time_scale=100.0,
            resting_level=-3.0,
            beta=20.0,
            self_connection_w0=2
        )

        # Robot base close enough to target
        self.close_precond = Field(
            shape=(),
            time_step=5.0,
            time_scale=100.0,
            resting_level=-3.0,
            beta=20.0,
            self_connection_w0=2
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
        # Find -> Found Precondition
        self.find_behavior.CoS.connection_to(self.found_precond, 6.0)

        # Found Precondition -> MoveTo
        self.found_precond.connection_to(self.move_to_behavior.intention, 6.0)
        self.move_to_behavior.CoS.connection_to(self.close_precond, 6.0)


    def execute_step(self, interactors, target_name, drop_off, external_input=5.0):
        """Execute the find portion of the behavior chain."""

        # Cache prev state for the precondition
        self.found_precond.cache_prev()
        self.close_precond.cache_prev()

        # Execute find behavior
        find_state = self.find_behavior.execute(interactors.perception,
                                                target_name,
                                                external_input)

        # Process the precondition nodes (no external input)
        found_activation, found_activity = self.found_precond()
        close_activation, close_activity = self.close_precond()

        # Execute move-to behavior with precondition input, only if found is active
        move_state = None
        if find_state['target_location'] is not None:
            move_state = self.move_to_behavior.execute(
                interactors.movement,
                find_state['target_location'],
                external_input = 0.0
            )

        # Get current robot position from movement interactor
        robot_position = interactors.movement.get_position()
        gripper_position = interactors.gripper.get_position()


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
            'gripper':{
                'position': gripper_position.tolist()
            }
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
    log = initalize_log()

    # Create simulation visualizer
    visualize = False
    if visualize:
        matplotlib.use('TkAgg')  # Use TkAgg backend which supports animation better
        visualizer = RobotSimulationVisualizer()
        simulation_states = []


    # External input activates find_grab behavior sequence
    external_input = 5.0

    # Create interactors with a test object
    interactors = RobotInteractors()
    interactors.perception.register_object(name="cup",
                                           location=torch.tensor([5.2, 10.5, 1.8]),
                                           angle=torch.tensor([0.0, -1.0, 0.0]))

    # Define drop-off location
    interactors.perception.register_object(name="drop_off",
                                           location=torch.tensor([5.0, 0.0, 1.0]),
                                           angle=torch.tensor([0.0, 0.0, 0.0]))

    # Create the find-grab behavior
    find_grab = FindGrabBehavior()

    # Run the find behavior until completion
    print("Starting find behavior for 'cup'...")
    i = 0
    done = 0
    for step in range(1200):
        # Execute find behavior
        state = find_grab.execute_step(interactors, "cup", "drop_off", external_input)

        # Update the visualizer
        if visualize:
            simulation_states.append(state)

        # Store logs
        update_log(log, state)

        i += 1


    # Plotting the activities of all nodes over time
    if not visualize:
        plot_logs(log, steps=i)  # or steps=500 if you always run 500 steps

    # Create animation
    if visualize:
        ani = FuncAnimation(
            visualizer.fig,  # Use visualizer's figure
            lambda i: visualizer.update(simulation_states[min(i, len(simulation_states) - 1)], interactors),
            frames=len(simulation_states),
            interval=10,
            blit=True,
            repeat=True
        )

        # Use this instead of plt.ion() and plt.show(block=True)
        plt.rcParams['animation.html'] = 'html5'  # For better compatibility
        manager = plt.get_current_fig_manager()
        if hasattr(manager, 'window'):
            manager.window.state('normal')  # Ensure window is not minimized
        plt.show()