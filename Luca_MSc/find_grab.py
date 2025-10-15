import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from DNF_torch.field import Field
from SimulationVisualizer import RobotSimulationVisualizer
from interactors import RobotInteractors

from find import FindBehavior
from move_to import MoveToBehavior
from check_effector_range import CheckEffectorRange
from reach_for import ReachForBehavior

from helper_functions import move_object, initalize_log, update_log, plot_logs


class FindGrabBehavior:
    """
    Composite behavior that chains find and grab behaviors.
    """

    def __init__(self):
        # Create component behaviors
        self.find_behavior = FindBehavior()
        self.move_to_behavior = MoveToBehavior()
        self.check_effector_range_behavior = CheckEffectorRange()
        self.reach_for_behavior = ReachForBehavior()

        # Flag for object movement - simulating object changing location!
        self.object_moved = False

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

        # Robot end-effector has enough reach to grab the target object
        self.in_reach_precond = Field(
            shape=(),
            time_step=5.0,
            time_scale=100.0,
            resting_level=-3.0,
            beta=20.0,
            self_connection_w0=2
        )

        # Robot end-effector has reached the target object
        self.reached_target_precond = Field(
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
        self.in_reach_precond.register_buffer(
            "g_u_prev",
            torch.zeros_like(self.in_reach_precond.g_u)
        )
        self.reached_target_precond.register_buffer(
            "g_u_prev",
            torch.zeros_like(self.reached_target_precond.g_u)
        )

        # Connections
        # Find -> Found Precondition
        self.find_behavior.CoS.connection_to(self.found_precond, 6.0)

        # Found Precondition -> MoveTo
        self.found_precond.connection_to(self.move_to_behavior.intention, 6.0)
        self.move_to_behavior.CoS.connection_to(self.close_precond, 6.0)

        # Found Precondition -> CheckEffectorRange
        self.found_precond.connection_to(self.check_effector_range_behavior.intention, 6.0)
        self.check_effector_range_behavior.CoS.connection_to(self.in_reach_precond, 6.0)

        # Close & In Reach Preconditions -> ReachFor (two preconditions needed)
        self.close_precond.connection_to(self.reach_for_behavior.intention, 3.0)
        self.in_reach_precond.connection_to(self.reach_for_behavior.intention, 3.0)
        self.reach_for_behavior.CoS.connection_to(self.reached_target_precond, 6.0)


    def execute_step(self, interactors, target_name, external_input=5.0):
        """Execute the find portion of the behavior chain."""

        # Cache prev state for the precondition
        self.found_precond.cache_prev()
        self.close_precond.cache_prev()
        self.in_reach_precond.cache_prev()
        self.reached_target_precond.cache_prev()

        # Execute find behavior
        find_state = self.find_behavior.execute(interactors.perception, target_name, external_input)

        # Process the found precondition node (no external input)
        found_activation, found_activity = self.found_precond()
        close_activation, close_activity = self.close_precond()
        in_reach_activation, in_reach_activity = self.in_reach_precond()
        reached_activation, reached_activity = self.reached_target_precond()

        # Execute move-to behavior with precondition input, only if found is active
        move_state = None
        if find_state['target_location'] is not None:
            move_state = self.move_to_behavior.execute(
                interactors.movement,
                find_state['target_location'],
                external_input = 0.0
            )

        # Execute check-effector-range behavior
        in_reach_state = self.check_effector_range_behavior.execute(
            interactors.perception,
            find_state['target_location'],
            effector_reach = 2.0,
            external_input = 0.0
        )

        # Execute reach-for behavior
        reach_for_state = None
        if find_state['target_location'] is not None:
            reach_for_state = self.reach_for_behavior.execute(
                interactors.gripper,
                find_state['target_location'],
                threshold=0.1,
                external_input = 0.0
            )


        # Check if robot is close to object and object hasn't been moved yet
        close_is_active = float(close_activity) > 0.7
        if close_is_active and not self.object_moved and find_state['target_location'] is not None:
            move_object(find_state, target_name, interactors)
            self.object_moved = True

        # Get current robot position from movement interactor
        robot_position = interactors.movement.get_position()
        gripper_position = interactors.gripper.get_position()

        state = {
            'find': find_state,
            'move': move_state,
            'in_reach': in_reach_state,
            'reach_for': reach_for_state,
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
                },
                'in_reach': {
                    'activation': float(in_reach_activation.detach()),
                    'activity': float(in_reach_activity.detach()),
                    'active': float(in_reach_activity) > 0.7
                },
                'reached': {
                    'activation': float(reached_activation.detach()),
                    'activity': float(reached_activity.detach()),
                    'active': float(reached_activity) > 0.7
                },
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
        self.check_effector_range_behavior.reset()
        self.in_reach_precond.reset()
        self.reach_for_behavior.reset()


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
    interactors.perception.register_object("cup", torch.tensor([5.2, 10.5, 1.8]))

    # Create the find-grab behavior
    find_grab = FindGrabBehavior()

    # Run the find behavior until completion
    print("Starting find behavior for 'cup'...")
    i = 0
    done = 0
    for step in range(500):
        # Execute find behavior
        state = find_grab.execute_step(interactors, "cup", external_input)

        # Update the visualizer
        if visualize:
            simulation_states.append(state)

        # Store logs
        update_log(log, state)

        # Print status
        print(f"Step {step}: Active={state['find']['active']}, Completed={state['find']['completed']}")
        print(f"  Found={state['find']['target_found']}, "
              f"Location={state['find']['target_location']}")
        print(f"  Intention={state['find']['intention_activity']:.2f}, "
              f"CoS={state['find']['cos_activity']:.2f}, "
              f"Found Precond={state['preconditions']['found']['activity']:.2f}")

        # print(f"Active Movement={state['move']['active'] if state['move'] else False}, Completed Movement={state['move']['completed'] if state['move'] else False}")
        # if state['move'] is not None:
        #     print(f" Move Intention Activation = {state['move']['intention_activation']} "
        #           f" Move CoS Activation = {state['move']['cos_activation']}"
        #           f" Close Precond Activation = {state['preconditions']['close']['activation']}")

        i += 1

    # Plotting the activities of all nodes over time
    if not visualize:
        plot_logs(log, steps=i)  # or steps=500 if you always run 500 steps

    # Create animation
    if visualize:
        ani = FuncAnimation(
            visualizer.fig,  # Use visualizer's figure
            lambda i: visualizer.update(simulation_states[min(i, len(simulation_states) - 1)]),
            frames=len(simulation_states),
            interval=100,
            blit=False,
            repeat=True
        )

        # Use this instead of plt.ion() and plt.show(block=True)
        plt.rcParams['animation.html'] = 'html5'  # For better compatibility
        manager = plt.get_current_fig_manager()
        if hasattr(manager, 'window'):
            manager.window.state('normal')  # Ensure window is not minimized
        plt.show()  # This will open in a separate window, not in PyCharm's viewer