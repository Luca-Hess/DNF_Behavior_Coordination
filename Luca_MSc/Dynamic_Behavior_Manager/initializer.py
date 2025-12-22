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

    def _register_and_set_field(self, field, attr_name, category, node_type, **kwargs):
        """Register field with runtime weights and set as behavior manager attribute"""
        self.runtime_weights.register_field(field, category=category, node_type=node_type, **kwargs)
        setattr(self.behavior_manager, attr_name, field)

    def _initialize_preconditions(self, required_behaviors):
        """Create and register precondition nodes for all behaviors"""
        default_params = self.weights.get_field_params('precondition_nodes', 'default')
        preconditions = self._nodes_list(
            node_params={name: default_params for name in required_behaviors},
            type_str="precond")

        # Register precondition nodes with runtime weights manager
        for name in required_behaviors:
            self._register_and_set_field(
                field=preconditions[f"{name}_precond"],
                attr_name=f'{name}_precond',
                category='precondition_nodes',
                node_type='precondition',
                behavior_name=name,
                instance_id=f'{name}_precond'
            )

    def _initialize_system_level_nodes(self):
        """Create and register system-level nodes"""
        system_level_nodes = self.initialize_system_level_nodes()

        # Register system-level nodes with runtime weights manager
        for node_name, field in system_level_nodes.items():
            node_type = f'system_{node_name.replace('_system', '')}'
            self._register_and_set_field(
                field=field,
                attr_name=node_type,
                category='system_nodes',
                node_type=node_type,
                instance_id=node_name
            )

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
        check = getattr(self.behavior_manager, f"check_{behavior_name}") \
            if hasattr(self.behavior_manager, f"check_{behavior_name}") else None

        # Node registration scheme
        NODE_SCHEMA = [
            ('intention', 'behavior_nodes', behavior),
            ('CoS', 'behavior_nodes', behavior),
            ('CoS_inverter', 'behavior_nodes', behavior),
            ('CoF', 'behavior_nodes', behavior)
        ]

        # Register behavior nodes
        for node_type, category, obj in NODE_SCHEMA:
            field = getattr(obj, node_type)
            self.runtime_weights.register_field(
                field,
                category=category,
                node_type=node_type,
                behavior_name=behavior_name,
                instance_id=f'{behavior_name}_{node_type}'
            )

        if check is not None:
            CHECK_SCHEMA = [
                ('intention', 'check_nodes'),
                ('confidence', 'check_nodes')
            ]

            # Register check nodes
            for node_type, category in CHECK_SCHEMA:
                field = getattr(check, node_type)
                self.runtime_weights.register_field(
                    field,
                    category=category,
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

        behavior_chain = []
        self._behavior_lookup = {}

        for i, name in enumerate(behaviors):
            base_name = self.get_base_behavior_name(name)
            behavior_info = {
                'name': name,
                'base_name': base_name,
                'behavior': getattr(self.behavior_manager, f"{base_name}_behavior"),
                'check': getattr(self.behavior_manager, f"check_{base_name}"),
                'precondition': getattr(self.behavior_manager, f"{base_name}_precond"),
                'has_next_precondition': i < len(behaviors) - 1,
                'check_failed_func': lambda result: not result[0]
            }
            behavior_info.update(self.resolve_behavior_config(name, base_name))
            behavior_chain.append(behavior_info)
            self._behavior_lookup[name] = behavior_info

        return behavior_chain


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


    def resolve_behavior_config(self, behavior_name, base_name):
        """
        Resolve behavior configuration, including extended behaviors and parallel behaviors.
        """
        try:
            # Creating a wrapper for parallel behaviors
            if behavior_name in behavior_config.PARALLEL_BEHAVIOR_CONFIG:
                parallel_config = behavior_config.PARALLEL_BEHAVIOR_CONFIG[behavior_name]

                return self.parallel_behavior_config(parallel_config)

            # Handling extended behaviors
            if behavior_name in behavior_config.EXTENDED_BEHAVIOR_CONFIG:
                extended_config = behavior_config.EXTENDED_BEHAVIOR_CONFIG[behavior_name]
                if base_name in behavior_config.ELEMENTARY_BEHAVIOR_CONFIG:
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

    def parallel_behavior_config(self, parallel_config):
        """
        Building list of component interactor configs that make up the parallel behaviors
        as well as the names of those behaviors
        """

        component_configs = []
        component_behavior_names = []

        for component_name in parallel_config['parallel_behaviors']:
            base_name = self.get_base_behavior_name(component_name)
            single_component_config = behavior_config.ELEMENTARY_BEHAVIOR_CONFIG[component_name]
            component_configs.append({
                'name': component_name,
                'base_name': base_name,
                'behavior_instance': getattr(self.behavior_manager, f"{base_name}_behavior"),
                'interactor_type': single_component_config['interactor_type'],
                'method': single_component_config['method'],
            })
            component_behavior_names.append(component_name)

        # Storing metadata to create the Parallel Interactor with
        return {
            'is_parallel': True,
            'component_configs': component_configs,
            'parallel_behaviors': component_behavior_names,
            'completion_strategy': parallel_config['completion_strategy'],
            'interactor_type': 'parallel',  # Special interactor type flag for "setup_subscriptions"
            'method': 'execute_parallel'  # Special method for parallel execution
        }

    def validate_behavior_args(self, beh_chain, behavior_args):
        """
        Validate that all required arguments are provided for behaviors in the chain.

        Args:
            behavior_chain: List of behavior level dictionaries
            behavior_args: Dictionary of arguments passed to behaviors

        Returns:
            tuple: (is_valid, error_message)
        """
        missing_args = []

        for level in beh_chain:
            behavior_name = level['name']

            # Check elementary behaviors
            if behavior_name in behavior_config.ELEMENTARY_BEHAVIOR_CONFIG:
                required = behavior_config.ELEMENTARY_BEHAVIOR_CONFIG[behavior_name].get('required_args', [])
                for arg in required:
                    if arg not in behavior_args:
                        missing_args.append(f"{behavior_name} requires '{arg}'")

            # Check extended behaviors
            if behavior_name in behavior_config.EXTENDED_BEHAVIOR_CONFIG:
                required = behavior_config.EXTENDED_BEHAVIOR_CONFIG[behavior_name].get('required_args', [])
                for arg in required:
                    if arg not in behavior_args:
                        missing_args.append(f"{behavior_name} requires '{arg}'")

            # Check parallel behaviors
            if behavior_name in behavior_config.PARALLEL_BEHAVIOR_CONFIG:
                parallel_behaviors = behavior_config.PARALLEL_BEHAVIOR_CONFIG[behavior_name]['parallel_behaviors']
                for parallel_behavior in parallel_behaviors:
                    required = behavior_config.ELEMENTARY_BEHAVIOR_CONFIG[parallel_behavior].get('required_args', [])
                    for arg in required:
                        if arg not in behavior_args:
                            missing_args.append(f"{parallel_behavior} (parallel behavior) requires '{arg}'")

        if missing_args:
            error_msg = "Missing required arguments:\n" + "\n".join(f"  - {msg}" for msg in missing_args)
            return False, error_msg

        return True, None