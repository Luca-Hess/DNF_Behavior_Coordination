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

from elementary_behavior_interface import ElementaryBehaviorInterface

from sanity_check import SanityCheckBehavior

from helper_functions import initalize_log, update_log, plot_logs, animate_fixed_chain

import behavior_config

import rclpy

class BehaviorManager():
    def __init__(self, behaviors=list, args=dict(), debug=False):
        self.behavior_args = args
        self.debug = debug

        self.initialize_nodes_and_behaviors(behaviors)

        # Keep track of which success actions have been executed
        self.success_actions_executed = set()

        ## Behavior chain with all node information
        # Initialize shared structures - all behaviors share these
        self.behavior_chain = [
            {
                'name': name,                                                                     # Behavior name string
                'behavior': getattr(self, f"{self._get_base_behavior_name(name)}_behavior"),      # Elementary behavior
                'check': getattr(self, f"check_{self._get_base_behavior_name(name)}"),            # Sanity check behavior
                'precondition': getattr(self, f"{self._get_base_behavior_name(name)}_precond"),   # Precondition node
                'has_next_precondition': i < len(behaviors) - 1,                                  # Last behavior has no next precondition
                'check_failed_func': lambda result: not result[0]                                 # State of sanity check
            }
            for i, name in enumerate(behaviors)
        ]

        # Add additional behavior chain info specific to each behavior - defined in behavior_config
        for level in self.behavior_chain:
            level.update(self._resolve_behavior_config(level['name']))

        # Setup DNF node connections according to behavior chain
        self.setup_connections()


    def nodes_list(self, node_params=dict, type_str="precond"):
        """
        Create multiple (precondition) nodes with given parameters.
        """
        nodes = {}
        for name, params in node_params.items():
            nodes[f'{name}_{type_str}'] = Field(
                shape=params.get('shape', ()),
                time_step=params.get('time_step', 5.0),
                time_scale=params.get('time_scale', 100.0),
                resting_level=params.get('resting_level', -3.0),
                beta=params.get('beta', 20.0),
                self_connection_w0=params.get('self_connection_w0', 2.0),
                noise_strength=params.get('noise_strength', 0.0),
                global_inhibition=params.get('global_inhibition', 0.0),
                scale=params.get('scale', 1.0)
            )

            # Register buffer for prev state
            nodes[f'{name}_{type_str}'].register_buffer(
                "g_u_prev",
                torch.zeros_like(nodes[f'{name}_{type_str}'].g_u)
            )

        return nodes
    
    def initialize_nodes_and_behaviors(self, behaviors=list):
        """
        Initialize all nodes and behaviors for the behavior chain.
        Also makes sure to include the base behavior for extended behaviors.
        => "grab_transport" requires "grab" behavior to be initialized as well, this handles that.
        """
        # Collect all required behaviors (Elementary Behaviors for extensions)
        required_behaviors = set()
        
        for name in behaviors:
            required_behaviors.add(name)
            base_name = self._get_base_behavior_name(name)
            if base_name != name:  # This is an extended behavior
                required_behaviors.add(base_name)
        
        # Create precondition nodes for all behaviors
        preconditions = self.nodes_list(node_params={name: {} for name in required_behaviors}, type_str="precond")

        # Create system level CoS and CoF nodes
        system_level_nodes = self.initialize_system_level_nodes(behaviors)

        # Finally initialize all required behaviors and nodes
        for name in required_behaviors:
            setattr(self, f"{name}_behavior", ElementaryBehaviorInterface())
            setattr(self, f"check_{name}", SanityCheckBehavior(behavior_name=name))
            setattr(self, f"{name}_precond", preconditions[f"{name}_precond"])

        setattr(self, "system_cos", system_level_nodes['cos_system'])
        setattr(self, "system_cof", system_level_nodes['cof_system'])

    def initialize_system_level_nodes(self, behaviors=list):
        # Setup system-level nodes
        scale_system_CoS_resting_level = -4.0 * len(behaviors)  # Scale resting level based on number of behaviors
        node_parameters = {
            'cos': {
                'resting_level': scale_system_CoS_resting_level,
                'beta': 20.0,
                'self_connection_w0': 15.0 * len(behaviors)
            },
            'cof': {
                'time_scale': 150.0,
                'resting_level': -4.75,
                'beta': 2.0,
                'self_connection_w0': 6.5
            }
        }
        system_nodes = self.nodes_list(node_params=node_parameters, type_str="system")

        return system_nodes


    def _get_base_behavior_name(self, behavior_name):
        """
        Get the base behavior name to use for potential extended behaviors.
        Behaviors can have "supersets" that extend their functionality.
        This function resolves to the base behavior name for initialization.
        Example: "grab_transport" initializes the base "grab" behavior.
        """
        if behavior_name in behavior_config.EXTENDED_BEHAVIOR_CONFIG:
            extended_config = behavior_config.EXTENDED_BEHAVIOR_CONFIG[behavior_name]
            return extended_config.get('extends', behavior_name)
        return behavior_name
        
    def setup_connections(self):
        """Setup all neural field connections using behavior chain"""
        for i, level in enumerate(self.behavior_chain):
            # CoS to precondition (weight 6)
            level['behavior'].CoS.connection_to(level['precondition'], 6.0)
            
            # CoS to check (weight 5)
            level['behavior'].CoS.connection_to(level['check'].intention, 5.0)
            
            # Precondition to next intention (weight 6) - if there's a next level
            if level.get('has_next_precondition', False) and i + 1 < len(self.behavior_chain):
                next_level = self.behavior_chain[i + 1]
                level['precondition'].connection_to(next_level['behavior'].intention, 6.0)

        # System-level connections
        # System CoS: Requires ALL behavior CoS nodes to be active (AND logic)
        cos_weight = 4.0
        for level in self.behavior_chain:
            level['behavior'].CoS.connection_to(self.system_cos, cos_weight)
        
        # System CoF: ANY behavior CoF can trigger system failure (OR logic)
        cof_weight = 4.0
        for level in self.behavior_chain:
            level['behavior'].CoF.connection_to(self.system_cof, cof_weight)

        # Make System CoS and CoF mutually exclusive (high inhibitory weights)
        self.system_cos.connection_to(self.system_cof, -15.0)
        self.system_cof.connection_to(self.system_cos, -15.0)
        
        self._debug_print(f"Setup system-level connections: {len(self.behavior_chain)} behaviors connected to system CoS/CoF")

    
    def _resolve_behavior_config(self, behavior_name):
        """Resolve behavior configuration, including extended behaviors."""
        try:
            if behavior_name in behavior_config.EXTENDED_BEHAVIOR_CONFIG:
                extended_config = behavior_config.EXTENDED_BEHAVIOR_CONFIG[behavior_name]
                base_name = extended_config.get('extends')
                if base_name and base_name in behavior_config.ELEMENTARY_BEHAVIOR_CONFIG:
                    # Merge base config with extended config
                    config = behavior_config.ELEMENTARY_BEHAVIOR_CONFIG[base_name].copy()
                    config.update({k: v for k, v in extended_config.items() if k != 'extends'})
                    return config

            if behavior_name in behavior_config.ELEMENTARY_BEHAVIOR_CONFIG:
                return behavior_config.ELEMENTARY_BEHAVIOR_CONFIG.get(behavior_name, {})

            raise ValueError(f"Unknown behavior: {behavior_name}")

        except Exception as e:
            print(f"[ERROR] Failed to resolve behavior config for {behavior_name}: {e}")
            return {}

        
    def setup_subscriptions(self, interactors):
        """Setup pub/sub connections using behavior chain data"""
        # Initialize StateInteractor based on behavior chain
        interactors.state.initialize_from_behavior_chain(self.behavior_chain, self.behavior_args)
    

        # Main behavior subscribes to interactor CoS updates
        for level in self.behavior_chain:
            interactor = getattr(interactors, level['interactor_type'])

            # Subscribe to CoS and CoF updates
            interactor.subscribe_cos_updates(
                level['name'], level['behavior'].set_cos_input
            )

            interactor.subscribe_cof_updates(
                level['name'], level['behavior'].set_cof_input
            )

            # Set up check behavior to publish back to the same interactor
            level['check'].set_interactor(interactor)

    def clear_subscriptions(self, interactors):
        """Clear all subscriptions from interactors and release check behaviors"""
        # Clear interactor subscriptions from base interactor class
        interactors.reset()
            
        # Release check behaviors
        for level in self.behavior_chain:
            level['check'].set_interactor(None)
        


    def _get_active_behavior(self):
        """Determine which behavior should be actively interacting with world"""
        for level in self.behavior_chain:
            if level['behavior'].execute()['intention_active']:
                return level['name']
        return None
    
    def _process_success_actions(self, actions, interactors, behavior_args):
        """
        Process declarative success actions for a behavior.
        => Some behaviors trigger additional one-time actions upon successful completion.
        """
        for action in actions:
            action_type = action.get('action')
            
            # Look up action in action-only behaviors
            if action_type in behavior_config.ACTION_BEHAVIOR_CONFIG:
                config = behavior_config.ACTION_BEHAVIOR_CONFIG[action_type]
                interactor = getattr(interactors, config['interactor_type'])
                service_method = getattr(interactor, config['service_method'])
                service_args = config['service_args_func'](interactors, behavior_args, action)
                
                try:
                    result = service_method(*service_args)
                    if not result[0]:  # Check if action succeeded
                        print(f"[WARNING] Success action {action_type} failed")
                except Exception as e:
                    print(f"[ERROR] Success action {action_type} failed: {e}")
            else:
                print(f"[ERROR] Unknown action type: {action_type}")

    def process_system_level_nodes(self):
        """Process system-level CoS and CoF nodes"""
        
        # Update system-level nodes (they receive inputs from behavior nodes automatically via connections)
        self.system_cos.cache_prev()
        self.system_cof.cache_prev()
        
        # Execute system-level dynamics
        system_cos_activation, system_cos_activity = self.system_cos()
        system_cof_activation, system_cof_activity = self.system_cof()
        
        # Determine system state
        system_success = float(system_cos_activity) > 0.7
        system_failure = float(system_cof_activity) > 0.7
        
        system_state = {
            'cos_activation': float(system_cos_activation.detach()),
            'cos_activity': float(system_cos_activity.detach()),
            'cof_activation': float(system_cof_activation.detach()),
            'cof_activity': float(system_cof_activity.detach()),
            'system_success': system_success,
            'system_failure': system_failure,
            'system_status': self._determine_system_status(system_success, system_failure)
        }
        
        self._debug_print(f"System state: {system_state['system_status']} (CoS: {system_state['cos_activity']:.3f}, CoF: {system_state['cof_activity']:.3f})")
        
        return system_state

    def _determine_system_status(self, system_success, system_failure):
        """Determine overall system status based on CoS and CoF"""
        if system_failure:
            return "FAILED"
        elif system_success:
            return "SUCCESS"
        else:
            return "IN_PROGRESS"


    def _debug_print(self, message):
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
        self.system_cos.reset()
        self.system_cos.clear_connections()
        self.system_cof.reset()
        self.system_cof.clear_connections()        

        self.debug = False
        self.success_actions_executed.clear()        
    

    def execute_step(self, interactors, external_input=6.0):
        # Determine which behavior is currently active
        active_behavior = self._get_active_behavior()
        self._debug_print(f"Active behavior: {active_behavior}")

        # Execute continuous world interaction for active behavior
        if active_behavior:
            level = next(l for l in self.behavior_chain if l['name'] == active_behavior)
            interactor = getattr(interactors, level['interactor_type'])
            
            # Get method dynamically
            method = getattr(interactor, level['method'])

            # Get service args and call continuous method
            service_args = level['service_args_func'](interactors, self.behavior_args, level['name'])
            if service_args[0] is not None:  # Only call if we have valid args
                method(*service_args, requesting_behavior=level['name'])
                self._debug_print(f"Executed continuous interaction for {active_behavior} with args {service_args}")
                
        # Execute all behaviors (just DNF dynamics)
        states = {}
        for level in self.behavior_chain:
            ext_input = external_input if level['name'] == self.behavior_chain[0]['name'] else 0.0 # Only first behavior gets external input
            states[level['name']] = level['behavior'].execute(ext_input)

        self._debug_print(f"Behavior states: {states}")

        # Process special actions upon success of behaviors
        # => Only execute once per behavior success
        for level in self.behavior_chain:
            behavior_name = level['name']
            if (level.get('on_success') and                             # There are "on success" actions defined
                states[level['name']].get('cos_active', False) and      # Behavior succeeded
                behavior_name not in self.success_actions_executed):    # Behavior "on success" action not yet executed

                self._process_success_actions(level['on_success'], interactors, self.behavior_args)
                self.success_actions_executed.add(behavior_name)
                self._debug_print(f"Processed success actions for {behavior_name}")


        # Process preconditions and add to state
        states['preconditions'] = {}
        for level in self.behavior_chain:
            level['precondition'].cache_prev()
            activation, activity = level['precondition']()
            states['preconditions'][level['name']] = {
                'activation': float(activation.detach()),
                'activity': float(activity.detach()),
                'active': float(activity) > 0.7
            }
        self._debug_print(f"Precondition states: {states['preconditions']}")
            
        # Process check behaviors - sanity checks triggered by low confidence
        states['checks'] = {}
        for level in self.behavior_chain:
            # Execute check behavior autonomously
            check_state = level['check'].execute(external_input=0.0)

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
            self._debug_print(f"Sanity check state for {level['name']}: {states['checks'][level['name']]}")
            
            # If check behavior triggered sanity check
            if check_state.get('sanity_check_triggered', False):
                interactor = getattr(interactors, level['interactor_type'])
                method = getattr(interactor, level['method'])
                service_args = level['service_args_func'](interactors, self.behavior_args, level['name'])
                
                if service_args[0] is not None:
                    # Single service call to verify current state of behavior goal
                    result = method(*service_args, requesting_behavior=None)
                    
                    # Check behavior processes result and updates its own CoS input to the associated elementary behavior
                    level['check'].process_sanity_result(result, level['check_failed_func'], level['name'])
                    
                    self._debug_print(f"Processed sanity check for {level['name']} with result {result}")

        # Process system-level CoS and CoF states
        states['system'] = self.process_system_level_nodes()

        return states
        

# Example usage
if __name__ == "__main__":

    find_move = BehaviorManager(
        behaviors=['find', 'move', 'check_reach', 'reach_for', 'grab_transport'],
        args={ 
            'target_object': 'cup',
            'drop_off_target': 'transport_target'
        }, debug=False)
    
    find_move_2 = BehaviorManager(
        behaviors=['find', 'move', 'check_reach', 'reach_for', 'grab_transport'],
        args={ 
            'target_object': 'bottle',
        }, debug=False)

    # Log all activations and activities for plotting
    log = initalize_log(find_move.behavior_chain)
    log2 = initalize_log(find_move_2.behavior_chain)

    # Create simulation visualizer
    visualize = False
    if visualize:
        matplotlib.use('TkAgg')  # Use TkAgg backend which supports animation better
        visualizer = RobotSimulationVisualizer(behavior_chain=find_move.behavior_chain)
        simulation_states = []


    # External input activates find_grab behavior sequence
    external_input = 10.0

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

    # Create the find-grab behavior
    find_move.setup_subscriptions(interactors)

    # Run the find behavior until completion
    print("Starting find behavior for 'cup'...")
    i = 0

    state = {}
    state2 = {}
    initial = True

    for step in range(1200):
        # Execute find behavior
        if not state.get('system', {}).get('system_success', False):
            state = find_move.execute_step(interactors, external_input)

        if state.get('system', {}).get('system_success', False) and initial:
            initial = False
            find_move.reset()
            find_move.clear_subscriptions(interactors)
            find_move_2.setup_subscriptions(interactors)

        # Update the visualizer
        if visualize:
            simulation_states.append(state)

        # Store logs
        update_log(log, state, step, find_move.behavior_chain)

        if state.get('system', {}).get('system_success', False):
            state2 = find_move_2.execute_step(interactors, external_input)

            update_log(log2, state2, i, find_move_2.behavior_chain)

            i += 1
    
    print('Final Position:', interactors.movement.get_position())
    print('Gripper Position:', interactors.gripper.get_position())
    print('Object Position:', interactors.perception.objects['cup']['location'])

    # Plotting the activities of all nodes over time
    if not visualize:
        plot_logs(log, step, find_move.behavior_chain)
        plot_logs(log2, i, find_move_2.behavior_chain)
        #animate_fixed_chain(log, find_move.behavior_chain)

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