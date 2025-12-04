import torch

from DNF_torch.field import Field

from elementary_behavior_interface import ElementaryBehaviorInterface
from sanity_check_interface import SanityCheckInterface

import behavior_config

class Initializer:
    def __init__(self, behavior_manager):
        self.behavior_manager = behavior_manager
        self.weights = behavior_manager.weights
        self.runtime_weights = behavior_manager.runtime_weights

    def _nodes_list(self, node_params=dict, type_str="precond"):
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

    def resolve_behavior_config(self, behavior_name):
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

    def get_base_behavior_name(self, behavior_name):
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
            base_name = self.get_base_behavior_name(name)
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
                    base_name = self.get_base_behavior_name(component_name)
                    parallel_component_behaviors.add(base_name if base_name != component_name else component_name)

        return parallel_component_behaviors

    def _initialize_preconditions(self, required_behaviors):
        """Create and register precondition nodes for all behaviors"""
        default_params = self.weights.get_field_params('precondition_nodes', 'default')
        preconditions = self._nodes_list(
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
            setattr(self.behavior_manager, f'{name}_precond', precond_field)

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
            setattr(self.behavior_manager, f'system_{node_type}', field)

    def _initialize_behaviors_and_checks(self, required_behaviors, include_checks=True):

        for name in required_behaviors:
            behavior = ElementaryBehaviorInterface(behavior_name=name, dynamics_params=self.weights)
            setattr(self.behavior_manager, f"{name}_behavior", behavior)

            if include_checks:
                check = SanityCheckInterface(behavior_name=name, dynamics_params=self.weights)
                setattr(self.behavior_manager, f"check_{name}", check)

            self._register_behavior_nodes(name)


    def _register_behavior_nodes(self, behavior_name):
        """Register all nodes for a single behavior with runtime weights manager"""
        behavior = getattr(self.behavior_manager, f"{behavior_name}_behavior")
        check = getattr(self.behavior_manager, f"check_{behavior_name}") if hasattr(self.behavior_manager, f"check_{behavior_name}") else None

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

        system_nodes = self._nodes_list(node_params=node_parameters, type_str="system")

        return system_nodes

    def build_behavior_chain(self, behaviors):
        """ Build the behavior chain with all necessary information for execution. """

        # Shared structures of all behaviors
        behavior_chain = [
            {
                'name': name,                                                                     # Behavior name string
                'behavior': getattr(self.behavior_manager, f"{self.get_base_behavior_name(name)}_behavior"),      # Elementary behavior
                'check': getattr(self.behavior_manager, f"check_{self.get_base_behavior_name(name)}"),            # Sanity check behavior
                'precondition': getattr(self.behavior_manager, f"{self.get_base_behavior_name(name)}_precond"),   # Precondition node
                'has_next_precondition': i < len(behaviors) - 1,                                  # Last behavior has no next precondition
                'check_failed_func': lambda result: not result[0]                                 # State of sanity check regarding CoS
            }
            for i, name in enumerate(behaviors)
        ]

        # Add additional behavior chain info specific to each behavior - defined in behavior_config
        for level in behavior_chain:
            level.update(self.resolve_behavior_config(level['name']))

        return behavior_chain