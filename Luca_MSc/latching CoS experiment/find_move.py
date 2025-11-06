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

from SimulationVisualizer import RobotSimulationVisualizer
from interactors import RobotInteractors

from find_int import ElementaryBehavior_IntentionCoupling

from check_found import SanityCheckBehavior

from helper_functions import nodes_list, initalize_log, update_log, plot_logs, move_object
import behavior_config

class FindMoveBehavior_Experimental():
    def __init__(self, behaviors=list):

        self.initialize_nodes_and_behaviors(behaviors)

        ## Behavior chain with all node information
        # Initialize shared structures
        self.behavior_chain = [
            {
                'name': name,                                       # Behavior name string
                'behavior': getattr(self, f"{name}_behavior"),      # Elementary behavior
                'check': getattr(self, f"check_{name}"),            # Sanity check behavior
                'precondition': getattr(self, f"{name}_precond"),   # Precondition node
                'has_next_precondition': i < len(behaviors) - 1,    # Last behavior has no next precondition
                'check_failed_func': lambda result: not result[0]   # State of sanity check
            }
            for i, name in enumerate(behaviors)
        ]

        # Add additional behavior chain info specific to each behavior - imported from behavior_config
        for level in self.behavior_chain:
            level.update(behavior_config.BEHAVIOR_CONFIG[level['name']])
        
        # Setup connections using behavior chain information
        self.setup_connections()

    
    def initialize_nodes_and_behaviors(self, behaviors=list):
        """Initialize all nodes and behaviors for the find-move behavior chain."""
        preconditions = nodes_list(node_names=[f"{name}" for name in behaviors], type_str="precond")

        for name in behaviors:
            setattr(self, f"{name}_behavior", ElementaryBehavior_IntentionCoupling())
            setattr(self, f"check_{name}", SanityCheckBehavior(behavior_name=name))
            setattr(self, f"{name}_precond", preconditions[f"{name}_precond"])
        
    def setup_connections(self):
        """Setup all neural field connections using behavior chain data"""
        for i, level in enumerate(self.behavior_chain):
            # CoS to precondition (weight 6)
            level['behavior'].CoS.connection_to(level['precondition'], 6.0)
            
            # CoS to check (weight 5)
            level['behavior'].CoS.connection_to(level['check'].intention, 5.0)
            
            # Precondition to next intention (weight 6) - if there's a next level
            if level.get('has_next_precondition', False) and i + 1 < len(self.behavior_chain):
                next_level = self.behavior_chain[i + 1]
                level['precondition'].connection_to(next_level['behavior'].intention, 6.0)
        
    def setup_subscriptions(self, interactors):
        """Setup pub/sub connections using behavior chain data"""
        # Main behavior subscribes to interactor CoS updates
        for level in self.behavior_chain:
            interactor = getattr(interactors, level['interactor_type'])
            interactor.subscribe_cos_updates(
                level['name'], level['behavior'].set_cos_input
            )

            # Set up check behavior to publish back to the same interactor
            level['check'].set_interactor(interactor)
        
    def execute_step(self, interactors, target_name, external_input=6.0):
        # Determine which behavior is currently active
        active_behavior = self._get_active_behavior()

        # Execute continuous world interaction for active behavior
        if active_behavior:
            level = next(l for l in self.behavior_chain if l['name'] == active_behavior)
            interactor = getattr(interactors, level['interactor_type'])
            continuous_method = getattr(interactor, level['continuous_method'])
            
            # Get service args and call continuous method
            service_args = level['service_args_func'](interactors, target_name)
            if service_args[0] is not None:  # Only call if we have valid args
                continuous_method(*service_args, level['name'])
                
        # Execute all behaviors (just DNF dynamics)
        states = {}
        for level in self.behavior_chain:
            ext_input = external_input if level['name'] == 'find' else 0.0
            states[level['name']] = level['behavior'].execute(ext_input)

            
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
            
        # Process check behaviors - they decide autonomously about sanity checks
        states['checks'] = {}
        for level in self.behavior_chain:
            # Execute check behavior autonomously
            check_state = level['check'].execute(external_input=0.0)
            
            # Extract node states
            if hasattr(level['check'], 'confidence'):
                level['check'].confidence.cache_prev()
                conf_activation, conf_activity = level['check'].confidence()
                confidence_activation = float(conf_activation.detach())
                confidence_activity = float(conf_activity.detach())
            else:
                confidence_activation = 0.0
                confidence_activity = 0.0
                
            if hasattr(level['check'], 'intention'):
                level['check'].intention.cache_prev()
                int_activation, int_activity = level['check'].intention()
                intention_activation = float(int_activation.detach())
                intention_activity = float(int_activity.detach())
            else:
                intention_activation = 0.0
                intention_activity = 0.0
            
            states['checks'][level['name']] = {
                'confidence_activation': confidence_activation,
                'confidence_activity': confidence_activity,
                'intention_activation': intention_activation,
                'intention_activity': intention_activity,
                'sanity_check_triggered': check_state.get('sanity_check_triggered', False)
            }
            
            # If check behavior autonomously decided to trigger sanity check
            if check_state.get('sanity_check_triggered', False):
                interactor = getattr(interactors, level['interactor_type'])
                service_method = getattr(interactor, level['service_method'])
                service_args = level['service_args_func'](interactors, target_name)
                
                if service_args[0] is not None:
                    # Single service call to verify current state
                    result = service_method(*service_args)
                    
                    # Check behavior processes result and updates its own CoS input
                    level['check'].process_sanity_result(result, level['check_failed_func'])

            
        return states
        
    def _get_active_behavior(self):
        """Determine which behavior should be actively interacting with world"""
        for level in self.behavior_chain:
            if level['behavior'].execute()['intention_active']:
                return level['name']
        return None

    def reset(self):
        """Reset all behaviors and preconditions using behavior chain data"""
        for level in self.behavior_chain:
            level['behavior'].reset()
            level['precondition'].reset()
            level['check'].reset()

# Example usage
if __name__ == "__main__":

    find_move = FindMoveBehavior_Experimental(behaviors=['find', 'move', 'check_reach', 'reach_for', 'grab'])
    # Log all activations and activities for plotting
    log = initalize_log(find_move.behavior_chain)

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

    # Define drop-off location
    interactors.perception.register_object(name="drop_off",
                                           location=torch.tensor([5.0, 0.0, 1.0]),
                                           angle=torch.tensor([0.0, 0.0, 0.0]))

    # Create the find-grab behavior
    find_move.setup_subscriptions(interactors)

    # Run the find behavior until completion
    print("Starting find behavior for 'cup'...")
    i = 0
    done = False

    for step in range(1200):
        # Execute find behavior
        state = find_move.execute_step(interactors, "cup", external_input)

        # Update the visualizer
        if visualize:
            simulation_states.append(state)

        # Store logs
        update_log(log, state, step, find_move.behavior_chain)


        if i == 450:
            # Move the cup to test tracking and recovery
            print(f"[Step {step}] Moving cup to test robustness...")
            new_location = torch.tensor([8.0, 12.0, 1.8])
            interactors.perception.objects["cup"]["location"] = new_location
            # Optionally cause tracking loss
            if hasattr(interactors.perception, 'cause_tracking_loss'):
                interactors.perception.cause_tracking_loss("cup", duration=10)

        i += 1
    
    print('Final Position:', interactors.movement.get_position())

    # Plotting the activities of all nodes over time
    if not visualize:
        plot_logs(log, step, find_move.behavior_chain)  # or steps=500 if you always run 500 steps

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