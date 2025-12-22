# import sys
# import os
# # Add DNF_torch package root
# sys.path.append(os.path.expanduser('~/nc_ws/DNF_torch'))

# Python package imports
import torch
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


from SimulationVisualizer import RobotSimulationVisualizer
from Luca_MSc.Dynamic_Behavior_Manager.DNF_interactors.robot_interactors import RobotInteractors


from helper_functions import initalize_log, update_log, plot_logs, animate_fixed_chain

from dnf_weights import dnf_weights

from runtime_weights import RuntimeWeightManager
from initializer import Initializer
from connection_builder import ConnectionBuilder
from runtime_management import RuntimeManagement
from state_collector import StateCollector

import time

"""
Important Notice: 
This code is conceptual work and currently lacks the higher level infrastructure 
that would use this behavior manager in a complete system.
The interactor system for device interactions works, but are also simple placeholders.
"""


class BehaviorManager():
    def __init__(self, behaviors=list, args=dict(), debug=False, weights=None):
        """
        Initialize the Behavior Manager with specified behaviors and arguments.
        Parameters:
        - behaviors: List of behavior names to execute in sequence.
        - args: Dictionary of arguments required by behaviors.
                Behaviors will validate args they require are present.
        - debug: Enable debug printing.
        - weights: Custom DNF weights for connections and dynamics (optional).
        """

        self.behavior_args = args
        self.debug = debug

        # Use provided weights or default values
        self.weights = weights if weights is not None else dnf_weights

        # Runtime weight manager to allow dynamic adjustments of connections and dynamics during execution
        self.runtime_weights = RuntimeWeightManager()

        # Initialize individual behaviors, preconditions, and checks as well as system level nodes
        self.initializer = Initializer(self)
        self.initializer.initialize_nodes_and_behaviors(behaviors)

        # Build behavior chain data structure
        self.behavior_chain = self.initializer.build_behavior_chain(behaviors)

        # Validate if all required arguments are provided for behaviors
        is_valid, error = self.initializer.validate_behavior_args(self.behavior_chain, self.behavior_args)
        if not is_valid:
            raise ValueError(f"Behavior args validation error: {error}")

        # Keep track of which success actions have been executed
        self.success_actions_executed = set()

        # Setup DNF node connections according to behavior chain (internal and external connections)
        self.connection_builder = ConnectionBuilder(self)
        self.connection_builder.setup_connections()

        # Handle run-time setup and processing
        self.runtime_manager = RuntimeManagement(self)



    def debug_print(self, message):
        """Print debug message if debugging is enabled"""
        if self.debug:
            print(f"[DEBUG] {message}")

    def reset(self):
        """Reset all behaviors and preconditions using behavior chain data"""
        for level in self.behavior_chain:
            level['behavior'].reset()
            level['precondition'].reset()
            level['check'].reset()

        # Reset system-level nodes
        self.system_intention.reset()
        self.system_intention.clear_connections()
        self.system_cos.reset()
        self.system_cos.clear_connections()
        self.system_cof.reset()
        self.system_cof.clear_connections()

        self.debug = False
        self.success_actions_executed.clear()

    def execute_active_behavior_interactors(self, active_behavior):
        level = self.initializer._behavior_lookup[active_behavior]

        # Get method dynamically
        method = getattr(level['interactor_instance'], level['method'])

        # Execute method associated with active behavior continuously (while active)
        method(continuous_behavior=level['name'])

        self.debug_print(f"Executed continuous interaction for {active_behavior}")

    def advance_behavior_dynamics(self, collector):
        """Advance dynamics of all behaviors, handling also parallel components"""
        for level in self.behavior_chain:
            behavior_state = level['behavior'].execute(external_input=0.0)

            if collector.enabled:
                collector.add(level['name'], behavior_state)

            else:
                collector.add(level['name'], behavior_state.get('cos_active', False))

            # Also advance component behaviors of parallel behaviors
            if level.get('is_parallel', False):
                for component_config in level['component_configs']:
                    component_behavior = component_config['behavior_instance']
                    component_behavior.execute(external_input=0.0)

    def advance_precondition_dynamics(self, collector):
        """Advance dynamics of all preconditions"""
        if collector.enabled:
            precond_states = {}
            for level in self.behavior_chain:
                level['precondition'].cache_prev()
                activation, activity = level['precondition']()
                precond_states[level['name']] = {
                    'activation': float(activation.detach()),
                    'activity': float(activity.detach()),
                    'active': float(activity) > 0.7
                    }
            collector.add('preconditions', precond_states)

        else:
            for level in self.behavior_chain:
                level['precondition'].cache_prev()
                level['precondition']()

    def check_and_process_success_actions(self, collector, interactors):
        """Check for behavior successes and process any defined success actions"""
        for level in self.behavior_chain:
            behavior_name = level['name']

            behavior_state = collector.get(behavior_name)
            if collector.enabled:
                cos_active = behavior_state.get('cos_active', False)
            else:
                cos_active = behavior_state

            if (level.get('on_success') and                             # There are "on success" actions defined
                cos_active and                                          # Behavior succeeded
                behavior_name not in self.success_actions_executed):    # Behavior "on success" action not yet executed

                self.runtime_manager.process_success_actions(level['on_success'], interactors, self.behavior_args)
                self.success_actions_executed.add(behavior_name)
                self.debug_print(f"Processed success actions for {behavior_name}")

    def process_check_behaviors(self, collector):
        """
        Process all check behaviors for sanity checks.
        This includes advancing their dynamics and executing sanity checks if any are triggered.
        """
        if collector.enabled:
            check_states = {}

        for level in self.behavior_chain:
            # Execute check behavior autonomously
            check_state = level['check'].execute(external_input=0.0)
            sanity_triggered = check_state.get('sanity_check_triggered', False)

            if collector.enabled:
                level['check'].confidence.cache_prev()
                conf_activation, conf_activity = level['check'].confidence()

                level['check'].intention.cache_prev()
                int_activation, int_activity = level['check'].intention()

                check_states[level['name']] = {
                    'confidence_activation': float(conf_activation.detach()),
                    'confidence_activity': float(conf_activity.detach()),
                    'intention_activation': float(int_activation.detach()),
                    'intention_activity': float(int_activity.detach()),
                    'sanity_check_triggered': sanity_triggered
                }
                self.debug_print(f"Sanity check state for {level['name']}: {check_states[level['name']]}")

            if sanity_triggered:
                method = getattr(level['interactor_instance'], level['method'])
                result = method(continuous_behavior=None)

                if level.get('is_parallel', False):
                    # Parallel interactor processes all component results
                    level['interactor_instance'].process_sanity_results(result, self)
                else:
                    # Check behavior processes result and updates its own CoS input to the associated elementary behavior
                    level['check'].process_sanity_result(result, level['check_failed_func'], level['name'])
                self.debug_print(f"Santiy check state for {level['name']} with results {result}")

        if collector.enabled:
            collector.add('checks', check_states)


    def execute_step(self, interactors, external_input=6.0, track_states=False):
        """
        Execute a single step of the behavior manager.
        External input is the initial driver and currently fixed.
        It could come from a higher-level planner in a complete system.
        """
        collector = StateCollector(enabled=track_states)

        # Determine which behavior is currently active
        active_behavior = self.runtime_manager.get_active_behavior()
        self.debug_print(f"Active behavior: {active_behavior}")

        # Execute continuous world interaction for active behavior
        if active_behavior:
            self.execute_active_behavior_interactors(active_behavior)

        # External input to system intention node
        self.system_intention(external_input)

        # Advance all behaviors (just processing DNF dynamics)
        self.advance_behavior_dynamics(collector)


        # Process special actions upon success of behaviors
        # => Only execute once per behavior success, resets if behavior fails sanity check
        self.check_and_process_success_actions(collector, interactors)

        # Advance precondition dynamics
        self.advance_precondition_dynamics(collector)
        self.debug_print(f'Precondition states: {collector.get("preconditions", {})}')

        # Process check behaviors for sanity checks
        self.process_check_behaviors(collector)

        # Process system-level CoS and CoF states
        system_states = self.runtime_manager.process_system_level_nodes()
        collector.add('system', system_states)

        return collector.get_result(minimal_data={'system': system_states})


def run_behavior_manager(behaviors,
                         behavior_args,
                         interactors,
                         external_input,
                         max_steps=2000,
                         debug=False,
                         visualize_sim=False,
                         visualize_architecture=False,
                         visualize_logs=False,
                         timing=False,
                         verbose=False,
                         perturbation_simulation=None):

    """
    Run the behavior manager with specified behaviors and arguments.
    Parameters:
    - behaviors: List of behavior names to execute in sequence.
    - behavior_args: Dictionary of arguments required by behaviors.
    - interactors: Interactor instances for device/world interaction.
    - external_input: External input to system intention node.
    - max_steps: Maximum number of execution steps.
    - debug: Enable debug printing.
    - visualize_sim: Enable 3D simulation visualization.
    - visualize_architecture: Enable behavior architecture visualization.
    - visualize_logs: Enable behavior logs visualization.
    - timing: Measure execution time.
    - verbose: Enable verbose output.
    - perturbation_simulation: Optional function to simulate perturbations during execution.
                               Was used for benchmarking purposes.
    Returns:
    - state: Final state of the behavior manager after execution.
    - result: Summary of execution results including success status, steps taken, and time elapsed.
    """

    # Create behavior manager
    behavior_seq = BehaviorManager(
        behaviors=behaviors,
        args=behavior_args,
        debug=debug,
        weights=dnf_weights
    )

    # These are examples of how to modify runtime weights before execution

    ## Receive information about current configuration
    # print('All Fields in Behavior Manager:', behavior_seq.runtime_weights.list_all_fields())
    # behavior_seq.runtime_weights.get_field_params('move_CoS')

    ## Disable specific connections or modify weights
    # behavior_seq.runtime_weights.disable_connection('check_reach_precond_to_reach_for_intention')
    # behavior_seq.runtime_weights.set_connection_weight('move_precond_to_check_reach_intention', -0.1)
    # behavior_seq.runtime_weights.set_field_param('move_CoS', 'self_connection_w0', 50.0)

    ## Get Reinforcement Learning-ready parameters
    # params, metadata = behavior_seq.runtime_weights.get_trainable_params()
    #
    # print("Trainable Parameters Metadata:")
    # i = 0
    # for meta in metadata:
    #     if 'instance_id' not in meta:
    #         print('Connection: ', meta['source_id'], '->', meta['target_id'])
    #     if 'instance_id' in meta:
    #         print('Node: ', meta['instance_id'])
    #     print('Value: ', params[0])
    #     print('---')
    #     i += 1

    ## Save optimized configuration
    #behavior_seq.runtime_weights.save('optimized_weights.json')

    # Log all activations and activities for plotting
    if visualize_logs or visualize_architecture:
        log = initalize_log(behavior_seq.behavior_chain)

    # Simulation visualizer
    if visualize_sim:
        matplotlib.use('TkAgg')  # Use TkAgg backend which supports animation better
        visualizer = RobotSimulationVisualizer(behavior_chain=behavior_seq.behavior_chain)
        simulation_states = []

    track_states = True
    if not visualize_sim and not visualize_logs and not visualize_architecture:
        track_states = False

    # Setup subscriptions for runtime management
    behavior_seq.runtime_manager.setup_subscriptions(interactors)

    if verbose:
        print("[INFO] Starting behavior execution.")

    if timing:
        start_time = time.time()

    for step in range(max_steps):

        # Only used for benchmarking perturbations versus other approaches
        if perturbation_simulation is not None:
            perturbation_simulation(step, interactors)

        state = behavior_seq.execute_step(interactors, external_input, track_states)


        # Update logs for 3D sim
        if visualize_sim:
            state.update({'target_position': interactors.perception.objects[behavior_args['target_object']]['location'].clone()})
            state.update({'robot_pos': interactors.movement.get_position()})
            state.update({'gripper_pos': interactors.gripper.get_position()})
            simulation_states.append(state)

        # Store logs
        if visualize_logs or visualize_architecture:
            update_log(log, state, step, behavior_seq.behavior_chain)

        if state.get('system', {}).get('system_success', False):
            # Verify that robot is at target location to check false positives
            robot_pos = interactors.movement.get_position()
            target_loc = interactors.perception.objects[behavior_args['drop_off_target']]['location']
            if torch.norm(robot_pos[:2] - target_loc[:2]) > interactors.movement.stop_threshold:
                state['system']['system_success'] = False
                print("[WARN] System reported success but robot is not at target location.")
                break
            if verbose:
                print(f"[INFO] Behavior sequence completed successfully in {step} steps.")
            break
        if state.get('system', {}).get('system_failure', False):
            if verbose:
                print(f"[INFO] Behavior sequence failed after {step} steps.")
            break

    if timing:
        end_time = time.time()

    if visualize_sim:
        ani = FuncAnimation(
            visualizer.fig,  # Use visualizer's figure
            lambda i: visualizer.update(simulation_states[min(i, len(simulation_states) - 1)], interactors),
            frames=len(simulation_states),
            interval=1,
            blit=True,
            repeat=True
        )

        plt.rcParams['animation.html'] = 'html5'  # For better compatibility
        manager = plt.get_current_fig_manager()
        if hasattr(manager, 'window'):
            manager.window.state('normal')  # Ensure window is not minimized
        plt.show()

    if visualize_logs:
        plot_logs(log, step, behavior_seq.behavior_chain)

    if visualize_architecture:
        animate_fixed_chain(log, behavior_seq.behavior_chain)

    state['steps'] = step + 1
    state['time'] = end_time - start_time if timing else None

    result = {
        'success': state['system']['system_success'],
        'steps': state['steps'],
        'time': state['time'],
        'final_status': state['system']['system_status']
    }

    # Clean up subscriptions and reset behavior manager after execution
    behavior_seq.runtime_manager.clear_subscriptions(interactors)
    behavior_seq.reset()

    return state, result



# Example usage
if __name__ == "__main__":
    behaviors = ['find', 'move', 'check_reach', 'reach_for', 'grab_transport']
    behaviors = ['find', 'move_and_reach', 'grab_transport']
    behavior_args = {
        'target_object': 'cup',
        'drop_off_target': 'transport_target'
    }

    # External input activates find_grab behavior sequence
    external_input = 6.0

    # Create interactors with a test object
    interactors = RobotInteractors()
    interactors.perception.register_object(name="cup",
                                           location=torch.tensor([5.2, 10.5, 1.8]),
                                           angle=torch.tensor([0.0, -1.0, 0.0]))

    # Define drop-off location for transport
    # Important: As a matter of definition, this is considered to be "given" and thus registered from the start
    # in a scenario that involves transporting an object to a specific location provided by the user (similar to the way the target object is given).
    # An alternative scenario could be to have a behavior subsequent to "grab" that actively queries the user for a drop-off location.
    interactors.perception.register_object(name="transport_target",
                                           location=torch.tensor([5.0, 0.0, 1.0]),
                                           angle=torch.tensor([0.0, 0.0, 0.0]))

    interactors.perception.register_object(name="bottle",
                                           location=torch.tensor([8.0, 12.0, 1.5]),
                                           angle=torch.tensor([0.0, -1.0, 0.0]))


    state, results = run_behavior_manager(
        behaviors=behaviors,                # Behavior sequence to execute
        behavior_args=behavior_args,        # Arguments for behaviors
        interactors=interactors,            # Interactors for device/world interaction
        external_input=external_input,      # External input to system intention node - simulates higher-level planner
        max_steps=2000,                     # Maximum steps to execute
        debug=False,                        # Enable debug printing
        visualize_sim=False,                # Visualize robot simulation - 3D animation
        visualize_logs=True,                # Visualize behavior logs - activation/activity plots
        visualize_architecture=False,       # Visualize behavior architecture - fixed chain animation
        timing=True,                        # Measure execution time
        verbose=False                       # Enable verbose output
    )



    print(f"Behavior Manager Results: {results}")


