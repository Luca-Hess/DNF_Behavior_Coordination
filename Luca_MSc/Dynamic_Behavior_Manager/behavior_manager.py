# import sys
# import os
# # Add DNF_torch package root
# sys.path.append(os.path.expanduser('~/nc_ws/DNF_torch'))

# Python package imports
import torch
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# DNF specific imports
from DNF_torch.field import Field

from SimulationVisualizer import RobotSimulationVisualizer
from Luca_MSc.Dynamic_Behavior_Manager.DNF_interactors.robot_interactors import RobotInteractors
from Luca_MSc.Dynamic_Behavior_Manager.DNF_interactors.parallel_interactors import ParallelInteractor

from elementary_behavior_interface import ElementaryBehaviorInterface

from sanity_check_interface import SanityCheckInterface

from helper_functions import initalize_log, update_log, plot_logs, animate_fixed_chain

from dnf_weights import dnf_weights
from runtime_weights import RuntimeWeightManager

import behavior_config

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

        # Create helper classes
        self.runtime_weights = RuntimeWeightManager()

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
                'check_failed_func': lambda result: not result[0]                                 # State of sanity check regarding CoS
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
            # Update default parameters with provided ones where available
            final_params = {}
            final_params.update(params)

            # Creating Field node with specified parameters (or defaults if not provided)
            nodes[f'{name}_{type_str}'] = Field(
                shape=final_params.get('shape', ()),
                time_step=final_params.get('time_step', 5.0),
                time_scale=final_params.get('time_scale', 100.0),
                resting_level=final_params.get('resting_level', -3.0),
                beta=final_params.get('beta', 20.0),
                self_connection_w0=final_params.get('self_connection_w0', 2.0),
                noise_strength=final_params.get('noise_strength', 0.0),
                global_inhibition=final_params.get('global_inhibition', 0.0),
                scale=final_params.get('scale', 1.0)
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
        required_behaviors = self._collect_required_behaviors(behaviors)

        # Collect parallel component behaviors
        parallel_component_behaviors = self._collect_parallel_component_behaviors(behaviors)

        # Create and register precondition nodes for all behaviors
        self._initialize_preconditions(required_behaviors)

        # Create and register system-level nodes
        self._initialize_system_level_nodes()


        # Initialize behaviors and their nodes + their sanity checks
        self._initialize_behaviors_and_checks(required_behaviors)
        self._initialize_behaviors_and_checks(parallel_component_behaviors, include_checks=False)

    def _resolve_behavior_config(self, behavior_name):
        """
        Resolve behavior configuration, including extended behaviors and parallel behaviors.
        """
        try:
            # Creating a wrapper for parallel behaviors
            if behavior_name in behavior_config.PARALLEL_BEHAVIOR_CONFIG:
                parallel_config = behavior_config.PARALLEL_BEHAVIOR_CONFIG[behavior_name]

                # Building list of component interactor configs that make up the parallel behaviors
                # as well as the names of those behaviors
                component_configs = []
                component_behavior_names = []

                for component_name in parallel_config['parallel_behaviors']:
                    single_component_config = behavior_config.ELEMENTARY_BEHAVIOR_CONFIG[component_name]
                    component_configs.append({
                        'interactor_type': single_component_config['interactor_type'],
                        'method': single_component_config['method'],
                        'service_args_func': single_component_config['service_args_func']
                    })
                    component_behavior_names.append(component_name)

                # Storing metadata to create the Parallel Interactor with
                return {
                    'is_parallel': True,
                    'component_configs': component_configs,
                    'parallel_behaviors': component_behavior_names,
                    'completion_strategy': parallel_config['completion_strategy'],

                    'interactor_type': 'parallel',  # Special interactor type flag for "setup_subscriptions"
                    'method': 'execute_parallel'
                }


            # Handling extended behaviors
            if behavior_name in behavior_config.EXTENDED_BEHAVIOR_CONFIG:
                extended_config = behavior_config.EXTENDED_BEHAVIOR_CONFIG[behavior_name]
                base_name = extended_config.get('extends')
                if base_name and base_name in behavior_config.ELEMENTARY_BEHAVIOR_CONFIG:
                    # Merge base config with extended config
                    config = behavior_config.ELEMENTARY_BEHAVIOR_CONFIG[base_name].copy()
                    config.update({k: v for k, v in extended_config.items() if k != 'extends'})
                    return config

            # Handling elementary behaviors
            if behavior_name in behavior_config.ELEMENTARY_BEHAVIOR_CONFIG:
                return behavior_config.ELEMENTARY_BEHAVIOR_CONFIG.get(behavior_name, {})

            raise ValueError(f"Unknown behavior: {behavior_name}")

        except Exception as e:
            print(f"[ERROR] Failed to resolve behavior config for {behavior_name}: {e}")
            return {}

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

    def _collect_required_behaviors(self, behaviors):
        """Collect all required base behaviors"""
        required_behaviors = set()

        for name in behaviors:
            required_behaviors.add(name)
            base_name = self._get_base_behavior_name(name)
            if base_name != name:  # This is an extended behavior
                required_behaviors.add(base_name)

        return required_behaviors

    def _collect_parallel_component_behaviors(self, behaviors):
        """Collect all component behaviors required for parallel behaviors"""
        parallel_component_behaviors = set()
        for name in behaviors:
            if name in behavior_config.PARALLEL_BEHAVIOR_CONFIG:
                parallel_config = behavior_config.PARALLEL_BEHAVIOR_CONFIG[name]
                for component_name in parallel_config['parallel_behaviors']:
                    base_name = self._get_base_behavior_name(component_name)
                    parallel_component_behaviors.add(base_name if base_name != component_name else component_name)

        return parallel_component_behaviors

    def _initialize_preconditions(self, required_behaviors):
        """Create and register precondition nodes for all behaviors"""
        default_params = self.weights.get_field_params('precondition_nodes', 'default')
        preconditions = self.nodes_list(
            node_params={name: default_params for name in required_behaviors},
            type_str="precond")

        # Register precondition nodes with runtime weights manager
        for name in required_behaviors:
            precond_field = preconditions[f"{name}_precond"]
            self.runtime_weights.register_field(
                precond_field,
                category='precondition_nodes',
                node_type='precondition',
                behavior_name=name,
                instance_id=f'{name}_precond'
            )
            setattr(self, f'{name}_precond', precond_field)

    def _initialize_system_level_nodes(self):
        """Create and register system-level nodes"""
        system_level_nodes = self.initialize_system_level_nodes()

        # Register system-level nodes with runtime weights manager
        for node_name, field in system_level_nodes.items():
            node_type = node_name.replace('_system', '')
            self.runtime_weights.register_field(
                field,
                category='system_nodes',
                node_type=node_type,
                instance_id=node_name
            )
            setattr(self, f'system_{node_type}', field)

    def _initialize_behaviors_and_checks(self, required_behaviors, include_checks=True):

        for name in required_behaviors:
            behavior = ElementaryBehaviorInterface(behavior_name=name, dynamics_params=self.weights)
            setattr(self, f"{name}_behavior", behavior)

            if include_checks:
                check = SanityCheckInterface(behavior_name=name, dynamics_params=self.weights)
                setattr(self, f"check_{name}", check)

            self._register_behavior_nodes(name)


    def _register_behavior_nodes(self, behavior_name):
        """Register all nodes for a single behavior with runtime weights manager"""
        behavior = getattr(self, f"{behavior_name}_behavior")
        check = getattr(self, f"check_{behavior_name}") if hasattr(self, f"check_{behavior_name}") else None

        # Register behavior nodes
        for node_type in ['intention', 'CoS', 'CoS_inverter', 'CoF']:
            field = getattr(behavior, node_type)
            self.runtime_weights.register_field(
                field,
                category='behavior_nodes',
                node_type=node_type,
                behavior_name=behavior_name,
                instance_id=f'{behavior_name}_{node_type}'
            )

        if check is not None:
            # Register check nodes
            for node_type in ['intention', 'confidence']:
                field = getattr(check, node_type)
                self.runtime_weights.register_field(
                    field,
                    category='check_nodes',
                    node_type=node_type,
                    behavior_name=behavior_name,
                    instance_id=f'check_{behavior_name}_{node_type}'
                )

    def initialize_system_level_nodes(self):
        # Setup system-level nodes
        node_parameters = {
            'intention': self.weights.get_field_params('system_nodes', 'intention'),
            'cos': self.weights.get_field_params('system_nodes', 'cos'),
            'cos_reporter': self.weights.get_field_params('system_nodes', 'cos_reporter'),
            'cof': self.weights.get_field_params('system_nodes', 'cof')
        }

        system_nodes = self.nodes_list(node_params=node_parameters, type_str="system")

        return system_nodes


    def setup_connections(self):
        """Setup all neural field connections using behavior chain"""
        w = self.weights

        for i, level in enumerate(self.behavior_chain):
            # Connecting the component behaviors to the parallel behavior envelope
            if level.get('is_parallel', False):
                self._setup_parallel_behavior_connections(level, w)

            # Standard connections for all behaviors
            self._setup_standard_behavior_connections(level, w)

            # Preconditions to the next behavior's intention where applicable
            if level.get('has_next_precondition', False) and i + 1 < len(self.behavior_chain):
                self._setup_precondition_connections(level, self.behavior_chain[i + 1], w)

        self._setup_system_level_connections(w)

        self._debug_print(f"Setup system-level connections: {len(self.behavior_chain)} behaviors connected to system CoS/CoF")

    def _register_and_connect(self, source_field, target_field, weight, source_id, target_id, connection_type):
        """Helper function to register connection with runtime weights manager"""
        source_field.connection_to(target_field, weight)
        self.runtime_weights.register_connection(
            source_field,
            target_field,
            weight,
            source_id=source_id,
            target_id=target_id,
            connection_type=connection_type
        )


    def _setup_parallel_behavior_connections(self, level, weights):
        """Setup connections for parallel behavior components to envelope behavior"""
        behavior_name = level['name']

        for component_name in level['parallel_behaviors']:
            base_name = self._get_base_behavior_name(component_name)
            component_behavior = getattr(self, f"{base_name}_behavior")

            # Component CoS to Envelope CoS (logic based on completion strategy)
            # any => OR, all => AND (weights are reduced based on number of component behaviors)
            weight = weights.get_connection_weight('parallel_behavior_to_envelope', 'cos_to_envelope_cos')
            if level['completion_strategy'] == 'all':
                weight /= len(level['parallel_behaviors'])

            self._register_and_connect(
                component_behavior.CoS,
                level['behavior'].CoS,
                weight,
                source_id=f'{base_name}_cos',
                target_id=f'{behavior_name}_cos',
                connection_type='component_cos_to_envelope_cos'
            )

            # Component CoF to Envelope CoF (always OR logic)
            weight = weights.get_connection_weight('parallel_behavior_to_envelope', 'cof_to_envelope_cof')
            self._register_and_connect(
                component_behavior.CoF,
                level['behavior'].CoF,
                weight,
                f'{base_name}_cof',
                f'{behavior_name}_cof',
                'component_cof_to_envelope_cof'
            )

            # Envelope intention to all Component intentions (shared activation)
            weight = weights.get_connection_weight('envelope_to_component_behavior', 'envelope_intention_to_component_intention')
            self._register_and_connect(
                level['behavior'].intention,
                component_behavior.intention,
                weight,
                f'{behavior_name}_intention',
                f'{base_name}_intention',
                'envelope_intention_to_component_intention'
            )

    def _setup_standard_behavior_connections(self, level, weights):
        """Setup standard connections for a single behavior"""
        behavior_name = level['name']

        # CoS to precondition
        weight = weights.get_connection_weight('behavior_to_precond', 'cos_to_precond')

        self._register_and_connect(
            level['behavior'].CoS,
            level['precondition'],
            weight,
            f'{behavior_name}_cos',
            f'{behavior_name}_precond',
            'cos_to_precond'
        )

        # CoS to check
        weight = weights.get_connection_weight('behavior_to_check', 'cos_to_check_intention')
        self._register_and_connect(
            level['behavior'].CoS,
            level['check'].intention,
            weight,
            f'{behavior_name}_cos',
            f'check_{behavior_name}_intention',
            'cos_to_check_intention'
        )


    def _setup_precondition_connections(self, level, next_level, weights):
        """Setup connections from precondition to next behavior's intention"""
        current_name = level['name']
        next_name = next_level['name']

        weight = weights.get_connection_weight('precond_to_next', 'precond_to_intention')
        self._register_and_connect(
            level['precondition'],
            next_level['behavior'].intention,
            weight,
            f'{current_name}_precond',
            f'{next_name}_intention',
            'precond_to_intention'
        )

    def _setup_system_level_connections(self, weights):
        """Setup connections between behavior and system levels as well as within system level"""

        for level in self.behavior_chain:
            behavior_name = level['name']

            # System Intention activates preconditions and behavior intention
            # (which inhibit their respective behavior intentions until CoS is achieved)
            weight = weights.get_connection_weight('system_level', 'intention_to_precond')
            self._register_and_connect(
                self.system_intention,
                level['precondition'],
                weight,
                'intention_system',
                f'{behavior_name}_precond',
                'system_intention_to_precond'
            )

            weight = weights.get_connection_weight('system_level', 'intention_to_behavior')
            self._register_and_connect(
                self.system_intention,
                level['behavior'].intention,
                weight,
                'intention_system',
                f'{behavior_name}_intention',
                'system_intention_to_behavior'
            )

            # System CoS Reporter: Inverted CoS from all behaviors (OR logic)
            # Any activity in CoS inverters will inhibit reporter
            weight = weights.get_connection_weight('system_level', 'cos_inverter_to_reporter')
            self._register_and_connect(
                level['behavior'].CoS_inverter,
                self.system_cos_reporter,
                weight,
                f'{behavior_name}_cos_inverter',
                'cos_reporter_system',
                'cos_inverter_to_reporter'
            )

            # System CoF: ANY behavior CoF can trigger system failure (OR logic)
            weight = weights.get_connection_weight('system_level', 'behavior_cof_to_system_cof')
            self._register_and_connect(
                level['behavior'].CoF,
                self.system_cof,
                weight,
                f'{behavior_name}_cof',
                'cof_system',
                'behavior_cof_to_system_cof'
            )

        # System CoS: Requires ALL behavior CoS nodes to be active (AND logic)
        # Achieved via inverted CoS connections to reporter, then reporter to system CoS
        weight = weights.get_connection_weight('system_level', 'reporter_to_system_cos')
        self._register_and_connect(
            self.system_cos_reporter,
            self.system_cos,
            weight,
            'cos_reporter_system',
            'cos_system',
            'reporter_to_system_cos'
        )

        # Make System CoS and CoF mutually exclusive (high inhibitory weights)
        weight = weights.get_connection_weight('mutual_inhibition', 'system_cos_to_cof')
        self.system_cos.connection_to(self.system_cof, weight)
        self.system_cof.connection_to(self.system_cos, weight)
        self.runtime_weights.register_connection(
            self.system_cos,
            self.system_cof,
            weight,
            'cos_system',
            'cof_system',
            'system_cos_to_cof'
        )
        self.runtime_weights.register_connection(
            self.system_cof,
            self.system_cos,
            weight,
            'cof_system',
            'cos_system',
            'system_cof_to_cos'
        )


    def setup_subscriptions(self, interactors):
        """Setup pub/sub connections using behavior chain data"""
        # Initialize StateInteractor based on behavior chain
        interactors.state.initialize_from_behavior_chain(self.behavior_chain, self.behavior_args)

        # Main behavior subscribes to interactor CoS updates
        for level in self.behavior_chain:

            # Creating the ParallelInteractor for parallel behaviors
            if level.get('interactor_type') == 'parallel':
                # Build interactor
                wrapped_interactors = []
                for single_component_config in level['component_configs']:
                    actual_interactor = getattr(interactors, single_component_config['interactor_type'])
                    wrapped_interactors.append({
                        'interactor': actual_interactor,
                        'method': single_component_config['method'],
                        'service_args_func': single_component_config['service_args_func']
                    })

                # Creating the wrapper around the component interactors
                parallel_interactor = ParallelInteractor(
                    wrapped_interactors,
                    level['parallel_behaviors']
                )

                # Store wrapper like normal interactor
                level['interactor_instance'] = parallel_interactor

                # Sanity check handled for all component behaviors via the wrapper
                level['check'].set_interactor(parallel_interactor)

                # Additionally set the targets of the component behaviors and subscribe them to CoS & CoF update
                parallel_behaviors = []
                for component_name in level['parallel_behaviors']:
                    parallel_behaviors.append({'name': component_name})
                for i, config in enumerate(level['component_configs']):
                    interactor_type = config['interactor_type']
                    actual_interactor = getattr(interactors, interactor_type)
                    name = parallel_behaviors[i]['name']
                    parallel_behaviors[i].update({'interactor_type': interactor_type,
                                                  'actual_interactor': actual_interactor})

                    actual_interactor.subscribe_cos_updates(
                        name , getattr(self, f'{name}_behavior').set_cos_input
                    )

                    actual_interactor.subscribe_cof_updates(
                        name , getattr(self, f'{name}_behavior').set_cof_input
                    )

                interactors.state.initialize_from_behavior_chain(parallel_behaviors, self.behavior_args)

            # Normal single, non-parallel interactors
            else:
                interactor = getattr(interactors, level['interactor_type'])
                level['interactor_instance'] = interactor

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
        self.system_intention.cache_prev()
        self.system_cos.cache_prev()
        self.system_cos_reporter.cache_prev()
        self.system_cof.cache_prev()

        # Execute system-level dynamics
        system_intention_activation, system_intention_activity = self.system_intention()
        system_cos_activation, system_cos_activity = self.system_cos()
        system_cos_reporter_activation, system_cos_reporter_activity = self.system_cos_reporter()
        system_cof_activation, system_cof_activity = self.system_cof()

        # Determine system state
        system_success = float(system_cos_activity) > 0.7
        system_failure = float(system_cof_activity) > 0.7

        system_state = {
            'intention_activation': float(system_intention_activation.detach()),
            'intention_activity': float(system_intention_activity.detach()),
            'cos_activation': float(system_cos_activation.detach()),
            'cos_activity': float(system_cos_activity.detach()),
            'cos_reporter_activation': float(system_cos_reporter_activation.detach()),
            'cos_reporter_activity': float(system_cos_reporter_activity.detach()),
            'cof_activation': float(system_cof_activation.detach()),
            'cof_activity': float(system_cof_activity.detach()),
            'system_success': system_success,
            'system_failure': system_failure,
            'system_status': self._determine_system_status(system_success, system_failure)
        }

        self._debug_print(f"System state: {system_state['system_status']} (Intention: {system_state['intention_activity']:.3f}, CoS: {system_state['cos_activity']:.3f}, CoF: {system_state['cof_activity']:.3f})")

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
        self.system_intention.reset()
        self.system_intention.clear_connections()
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
            interactor = level['interactor_instance']

            # Get method dynamically
            method = getattr(interactor, level['method'])


            if level.get('is_parallel', False):
                method(interactors, self.behavior_args, requesting_behavior=level['name'])

            else:
                # Get service args and call continuous method
                service_args = level['service_args_func'](interactors, self.behavior_args, level['name'])

                # Only call if we have valid args
                if service_args[0] is not None:
                    method(*service_args, requesting_behavior=level['name'])
                    self._debug_print(f"Executed continuous interaction for {active_behavior} with args {service_args}")

        # External input to system intention node
        self.system_intention(external_input)

        # Execute all behaviors (just DNF dynamics)
        states = {}
        for level in self.behavior_chain:
            states[level['name']] = level['behavior'].execute(external_input=0.0)

            # Also advance component behaviors of parallel behaviors
            if level.get('is_parallel', False):
                for component_name in level['parallel_behaviors']:
                    base_name = self._get_base_behavior_name(component_name)
                    component_behavior = getattr(self, f"{base_name}_behavior")
                    state_par = component_behavior.execute(external_input=0.0)
                    #print(f"[DEBUG] Component behavior {component_name} state: {state_par}")

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
                interactor = level['interactor_instance']
                method = getattr(interactor, level['method'])

                if level.get('is_parallel', False):
                    results = method(interactors, self.behavior_args, requesting_behavior=None)
                    interactor.process_sanity_results(results, self)
                    self._debug_print(f"Processed sanity checks for parallel behavior {level['name']} with results {results}")

                else:
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

    behavior_seq.setup_subscriptions(interactors)

    if verbose:
        print("[INFO] Starting behavior execution.")

    if timing:
        start_time = time.time()

    for step in range(max_steps):

        # Only used for benchmarking perturbations versus other approaches
        if perturbation_simulation is not None:
            perturbation_simulation(step, interactors)

        state = behavior_seq.execute_step(interactors, external_input)


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
    behavior_seq.clear_subscriptions(interactors)
    behavior_seq.reset()

    return state, result



# Example usage
if __name__ == "__main__":
    behaviors = ['find', 'move', 'check_reach', 'reach_for', 'grab_transport']
    #behaviors = ['find', 'move_and_reach', 'grab_transport']
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


