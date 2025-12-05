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

import time

"""
Important Notice: 
This code is conceptual work and currently lacks the higher level infrastructure that would use this behavior manager in a complete system.
The interactor system for device interactions works, but are also simple placeholders.
"""


class BehaviorManager():
    def __init__(self, behaviors=list, args=dict(), debug=False, weights=None):
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
        level = next(l for l in self.behavior_chain if l['name'] == active_behavior)
        interactor = level['interactor_instance']

        # Get method dynamically
        method = getattr(interactor, level['method'])

        method(requesting_behavior=level['name'])
        self.debug_print(f"Executed continuous interaction for {active_behavior}")

    def advance_behavior_dynamics(self, states, collect_states=True):
        """Advance dynamics of all behaviors, handling also parallel components"""
        if collect_states and states is None:
            states = {}

        # minimal states when not collecting all state information
        minimal_states = {}

        for level in self.behavior_chain:
            # always executed
            behavior_state = level['behavior'].execute(external_input=0.0)

            if collect_states:
                states[level['name']] = behavior_state

            else:
                minimal_states[level['name']] = behavior_state.get('cos_active', False)

            # Also advance component behaviors of parallel behaviors
            if level.get('is_parallel', False):
                for component_name in level['parallel_behaviors']:
                    base_name = self.initializer.get_base_behavior_name(component_name)
                    component_behavior = getattr(self, f"{base_name}_behavior")
                    component_behavior.execute(external_input=0.0)

        return states if collect_states else minimal_states

    def advance_precondition_dynamics(self, states, collect_states=True):
        """Advance dynamics of all preconditions"""
        if collect_states and states is None:
            states = {}
        if collect_states:
            states['preconditions'] = {}
        for level in self.behavior_chain:
            level['precondition'].cache_prev()
            activation, activity = level['precondition']()
            if collect_states:
                states['preconditions'][level['name']] = {
                    'activation': float(activation.detach()),
                    'activity': float(activity.detach()),
                    'active': float(activity) > 0.7
                }

        return states

    def check_and_process_success_actions(self, states, interactors, collect_states=True):
        """Check for behavior successes and process any defined success actions"""
        for level in self.behavior_chain:
            behavior_name = level['name']

            if collect_states:
                cos_active = states[behavior_name].get('cos_active', False)
            else:
                cos_active = states.get(behavior_name, False)

            if (level.get('on_success') and                             # There are "on success" actions defined
                cos_active and                                          # Behavior succeeded
                behavior_name not in self.success_actions_executed):    # Behavior "on success" action not yet executed

                self.runtime_manager.process_success_actions(level['on_success'], interactors, self.behavior_args)
                self.success_actions_executed.add(behavior_name)
                self.debug_print(f"Processed success actions for {behavior_name}")

    def process_check_behaviors(self, states, interactors, collect_states=True):
        """
        Process all check behaviors for sanity checks.
        This includes advancing their dynamics and executing sanity checks if any are triggered.
        """
        if collect_states and states is None:
            states = {}
        if collect_states:
            states['checks'] = {}

        for level in self.behavior_chain:
            # Execute check behavior autonomously
            check_state = level['check'].execute(external_input=0.0)

            if collect_states:
                level['check'].confidence.cache_prev()
                conf_activation, conf_activity = level['check'].confidence()
                confidence_activation = float(conf_activation.detach())
                confidence_activity = float(conf_activity.detach())

                level['check'].intention.cache_prev()
                int_activation, int_activity = level['check'].intention()
                intention_activation = float(int_activation.detach())
                intention_activity = float(int_activity.detach())

                states['checks'][level['name']] = {
                    'confidence_activation': confidence_activation,
                    'confidence_activity': confidence_activity,
                    'intention_activation': intention_activation,
                    'intention_activity': intention_activity,
                    'sanity_check_triggered': check_state.get('sanity_check_triggered', False)
                }
                self.debug_print(f"Sanity check state for {level['name']}: {states['checks'][level['name']]}")

            if check_state.get('sanity_check_triggered', False):
                interactor = level['interactor_instance']
                method = getattr(interactor, level['method'])
                result = method(requesting_behavior=None)
                if level.get('is_parallel', False):
                    # Parallel interactor processes all component results
                    interactor.process_sanity_results(result, self)
                else:
                    # Check behavior processes result and updates its own CoS input to the associated elementary behavior
                    level['check'].process_sanity_result(result, level['check_failed_func'], level['name'])
                self.debug_print(f"Santiy check state for {level['name']} with results {result}")




    def execute_step(self, interactors, external_input=6.0, track_states=False):
        """
        Execute a single step of the behavior manager.
        External input is the initial driver and currently fixed.
        It could come from a higher-level planner in a complete system.
        """

        # Determine which behavior is currently active
        active_behavior = self.runtime_manager.get_active_behavior()
        self.debug_print(f"Active behavior: {active_behavior}")

        # Execute continuous world interaction for active behavior
        if active_behavior:
            self.execute_active_behavior_interactors(active_behavior)

        # External input to system intention node
        self.system_intention(external_input)

        # Advance all behaviors (just processing DNF dynamics)
        if track_states:
            states = {}
            states = self.advance_behavior_dynamics(states, collect_states=track_states)
            self.debug_print(f"Behavior states: {states}")
        else:
            minimal_states = self.advance_behavior_dynamics(states=None, collect_states=track_states)
            states = None


        # Process special actions upon success of behaviors
        # => Only execute once per behavior success, resets if behavior fails sanity check
        if track_states:
            self.check_and_process_success_actions(states, interactors, collect_states=track_states)
        else:
            self.check_and_process_success_actions(minimal_states, interactors, collect_states=track_states)

        # Advance precondition and check behavior dynamics
        if track_states:
            states = self.advance_precondition_dynamics(states, collect_states=track_states)
            self.debug_print(f"Precondition states: {states['preconditions']}")

            # Process check behaviors - sanity checks triggered by low confidence
            self.process_check_behaviors(states, interactors)
        else:
            self.advance_precondition_dynamics(states=None, collect_states=track_states)
            self.process_check_behaviors(states=None, interactors=interactors, collect_states=track_states)

        # Process system-level CoS and CoF states
        system_states = self.runtime_manager.process_system_level_nodes()
        if track_states:
            states['system'] = system_states
            return states
        else:
            return {'system': system_states}


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

    # Create behavior manager
    behavior_seq = BehaviorManager(
        behaviors=behaviors,
        args=behavior_args,
        debug=debug,
        weights=dnf_weights
    )

    #print('All Fields in Behavior Manager:', behavior_seq.runtime_weights.list_all_fields())
    # behavior_seq.runtime_weights.disable_connection('check_reach_precond_to_reach_for_intention')
    # behavior_seq.runtime_weights.set_connection_weight('move_precond_to_check_reach_intention', -0.1)
    # behavior_seq.runtime_weights.set_field_param('move_CoS', 'self_connection_w0', 50.0)
    # behavior_seq.runtime_weights.get_field_params('move_CoS')

    ## Get RL-ready parameters
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

    #
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

    state, results = run_behavior_manager(behaviors=behaviors,
                         behavior_args=behavior_args,
                         interactors=interactors,
                         external_input=external_input,
                         max_steps=2000,
                         debug=False,
                         visualize_sim=False,
                         visualize_logs=True,
                         visualize_architecture=False,
                         timing=True,
                         verbose=False)

    print(f"Behavior Manager Results: {results}")


