import torch
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
from grab_behavior_compound import GrabBehavior

from helper_functions import move_object, initalize_log, update_log, plot_logs


class FindGrabBehavior():
    """
    Composite behavior that chains find and grab behaviors.
    """

    def __init__(self):
        # Create component behaviors
        self.find_behavior = FindBehavior()
        self.move_to_behavior = MoveToBehavior()
        self.check_effector_range_behavior = CheckEffectorRange()
        self.reach_for_behavior = ReachForBehavior()
        self.grab_behavior = GrabBehavior()
        self.transport_to_behavior = MoveToBehavior()

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

        # Robot has grabbed the target object
        self.has_grabbed_precond = Field(
            shape=(),
            time_step=5.0,
            time_scale=100.0,
            resting_level=-3.0,
            beta=20.0,
            self_connection_w0=2
        )

        self.transported_precond = Field(
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
        self.has_grabbed_precond.register_buffer(
            "g_u_prev",
            torch.zeros_like(self.has_grabbed_precond.g_u)
        )
        self.transported_precond.register_buffer(
            "g_u_prev",
            torch.zeros_like(self.transported_precond.g_u)
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

        # Reached Target Precondition -> Grab Behavior
        self.reached_target_precond.connection_to(self.grab_behavior.intention, 6.0)
        self.grab_behavior.CoS.connection_to(self.has_grabbed_precond, 6.0)

        # Has Grabbed Precondition -> TransportTo Behavior
        self.has_grabbed_precond.connection_to(self.transport_to_behavior.intention, 6.0)
        self.transport_to_behavior.CoS.connection_to(self.transported_precond, 6.0)

    def execute_step(self, interactors, target_name, drop_off, external_input=5.0):
        """Execute the find portion of the behavior chain."""

        # Cache prev state for the precondition
        self.found_precond.cache_prev()
        self.close_precond.cache_prev()
        self.in_reach_precond.cache_prev()
        self.reached_target_precond.cache_prev()
        self.has_grabbed_precond.cache_prev()
        self.transported_precond.cache_prev()

        # Execute find behavior
        find_state = self.find_behavior.execute(interactors.perception,
                                                target_name,
                                                external_input)

        # Process the precondition nodes (no external input)
        found_activation, found_activity = self.found_precond()
        close_activation, close_activity = self.close_precond()
        in_reach_activation, in_reach_activity = self.in_reach_precond()
        reached_activation, reached_activity = self.reached_target_precond()
        has_grabbed_activation, has_grabbed_activity = self.has_grabbed_precond()
        transported_activation, transported_activity = self.transported_precond()

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
                threshold=0.25,
                external_input = 0.0
            )

        # Execute grab behavior
        grab_state = None
        if find_state['target_location'] is not None:
            grab_state = self.grab_behavior.execute(
                interactors,
                target_name,
                find_state['target_location'],  # Example target orientation for gripper
                external_input = 0.0
            )

        # Transport to a new location (e.g., drop-off point)
        transport_state = None
        if drop_off is not None:
            drop_off_location = interactors.perception.objects["drop_off"]['location']
            transport_state = self.transport_to_behavior.execute(
                interactors.movement,
                drop_off_location,
                external_input = 0.0
            )

        # Get current robot position from movement interactor
        robot_position = interactors.movement.get_position()
        gripper_position = interactors.gripper.get_position()

        ##### SIMULATION RELEVANT CODE #####
        # Check if robot is close to object and object hasn't been moved yet
        close_is_active = float(close_activity) > 0.7
        if close_is_active and not self.object_moved and find_state['target_location'] is not None:
            move_object(find_state, target_name, interactors)
            self.object_moved = True

        # Second move to simulate object being taken out of robot's gripper after pick up
        has_grabbed_is_active = float(has_grabbed_activity) > 0.7
        if has_grabbed_is_active and not self.object_dropped and find_state['target_location'] is not None:
            move_object(find_state, target_name, interactors)
            self.object_dropped = True

        # Locking the object to the gripper if grabbed
        gripper_position = interactors.gripper.get_position()
        actual_has_object = interactors.gripper.has_object(gripper_position,
                                                              interactors.perception.objects[target_name]['location'])
        if self.object_moved and self.object_dropped and actual_has_object:
            self.picked_up = True

        if has_grabbed_is_active and self.picked_up:
            interactors.perception.objects[target_name]['location'] = gripper_position.clone()

        ####################################

        state = {
            'find': find_state,
            'move': move_state,
            'in_reach': in_reach_state,
            'reach_for': reach_for_state,
            'grab': grab_state,
            'transport': transport_state,
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
                'has_grabbed': {
                    'activation': float(has_grabbed_activation.detach()),
                    'activity': float(has_grabbed_activity.detach()),
                    'active': float(has_grabbed_activity) > 0.7
                },
                'transported': {
                    'activation': float(transported_activation.detach()),
                    'activity': float(transported_activity.detach()),
                    'active': float(transported_activity) > 0.7
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
        self.check_effector_range_behavior.reset()
        self.in_reach_precond.reset()
        self.reach_for_behavior.reset()
        self.reached_target_precond.reset()
        self.grab_behavior.reset()
        self.has_grabbed_precond.reset()
        self.transport_to_behavior.reset()
        self.transported_precond.reset()


# Example usage
if __name__ == "__main__":
    # Log all activations and activities for plotting
    log = initalize_log()

    # Create simulation visualizer
    visualize = True
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