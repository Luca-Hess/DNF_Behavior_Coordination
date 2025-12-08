

class ConnectionBuilder:
    def __init__(self, behavior_manager):
        self.behavior_manager = behavior_manager
        self.initializer = behavior_manager.initializer
        self.weights = behavior_manager.weights
        self.runtime_weights = behavior_manager.runtime_weights
        self.behavior_args = behavior_manager.behavior_args
        self.behavior_chain = behavior_manager.behavior_chain

        # Defining connection schema to automate repetitive connection patterns
        self.CONNECTION_SCHEMA = {
            'standard_behavior': [
                # Source Node,     Target Node,     Weight Category,     Weight Key,     Connection Type
                ('behavior.CoS', 'precondition', 'behavior_to_precond', 'cos_to_precond', 'cos_to_precond'),
                ('behavior.CoS', 'check.intention', 'behavior_to_check', 'cos_to_check_intention', 'cos_to_check_intention'),
            ],
            'system_to_behavior': [
                ('system_intention', 'precondition', 'system_level', 'intention_to_precond', 'system_intention_to_precond'),
                ('system_intention', 'behavior.intention', 'system_level', 'intention_to_behavior', 'system_intention_to_behavior'),
            ],
            'behavior_to_system': [
                ('behavior.CoS_inverter', 'system_cos_reporter', 'system_level', 'cos_inverter_to_reporter', 'cos_inverter_to_reporter'),
                ('behavior.CoF', 'system_cof', 'system_level', 'behavior_cof_to_system_cof', 'behavior_cof_to_system_cof'),
            ]
        }

    def _apply_connection_schema(self, schema_name, level, behavior_name=None):
        """Apply predefined connection schema to a behavior level"""
        for source_path, target_path, weight_category, weight_key, connection_type in (
                self.CONNECTION_SCHEMA)[schema_name]:
            source_field = self._resolve_field_path(source_path, level, behavior_name)
            target_field = self._resolve_field_path(target_path, level, behavior_name)

            weight = self.weights.get_connection_weight(weight_category, weight_key)

            source_id = self._get_field_id(source_path, behavior_name or level['name'])
            target_id = self._get_field_id(target_path, behavior_name or level['name'])

            self._register_and_connect(
                source_field,
                target_field,
                weight,
                source_id,
                target_id,
                connection_type
            )

    def _resolve_field_path(self, path, level, behavior_name=None):
        """Resolve field path string to actual field object"""
        parts = path.split('.')

        # Handling system level nodes specially
        if parts[0].startswith('system_'):
            return getattr(self.behavior_manager, parts[0])

        # Handling behavior level nodes
        field = level
        for part in parts:
            field = field[part] if isinstance(field, dict) else getattr(field, part)
        return field

    def _get_field_id(self, path, behavior_name):
        """Generate unique field ID based on path and behavior name"""
        parts = path.split('.')
        if parts[0].startswith('system_'):
            return parts[0]
        elif parts[0] == 'behavior':
            return f"{behavior_name}_{parts[-1].lower()}"
        elif parts[0] == 'check':
            return f"check_{behavior_name}_{parts[-1].lower()}"
        else:
            return f"{behavior_name}_{parts[0]}"


    def setup_connections(self):
        """Setup all neural field connections using behavior chain"""
        w = self.weights

        for i, level in enumerate(self.behavior_chain):
            # Connecting the component behaviors to the parallel behavior envelope
            if level.get('is_parallel', False):
                self._setup_parallel_behavior_connections(level, w)

            # Standard connections for all behaviors
            self._setup_standard_behavior_connections(level)

            # Preconditions to the next behavior's intention where applicable
            if level.get('has_next_precondition', False) and i + 1 < len(self.behavior_chain):
                self._setup_precondition_connections(level, self.behavior_chain[i + 1], w)

        self._setup_system_level_connections(w)

        self.behavior_manager.debug_print(f"Setup system-level connections: {len(self.behavior_chain)} behaviors connected to system CoS/CoF")

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

        for component_config in level['component_configs']:
            base_name = component_config['name']
            component_behavior = component_config['behavior_instance']

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

    def _setup_standard_behavior_connections(self, level):
        """Setup standard connections for a single behavior"""
        self._apply_connection_schema('standard_behavior', level)


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
            self._apply_connection_schema('system_to_behavior', level)
            self._apply_connection_schema('behavior_to_system', level)

        # System CoS: Requires ALL behavior CoS nodes to be active (AND logic)
        # Achieved via inverted CoS connections to reporter, then reporter to system CoS
        weight = weights.get_connection_weight('system_level', 'reporter_to_system_cos')
        self._register_and_connect(
            self.behavior_manager.system_cos_reporter,
            self.behavior_manager.system_cos,
            weight,
            'cos_reporter_system',
            'cos_system',
            'reporter_to_system_cos'
        )

        # Make System CoS and CoF mutually exclusive (high inhibitory weights)
        weight_cos_cof = weights.get_connection_weight('mutual_inhibition', 'system_cos_to_cof')
        weight_cof_cos = weights.get_connection_weight('mutual_inhibition', 'system_cof_to_cos')

        self._register_and_connect(
            self.behavior_manager.system_cos,
            self.behavior_manager.system_cof,
            weight_cof_cos,
            'cos_system',
            'cof_system',
            'system_cos_to_cof'
        )
        self._register_and_connect(
            self.behavior_manager.system_cof,
            self.behavior_manager.system_cos,
            weight_cof_cos,
            'cof_system',
            'cos_system',
            'system_cof_to_cos'
        )