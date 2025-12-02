"""
DNF Weight Configuration
Explicit storage of all neural field parameters and connection weights.
This serves as the interface for RL-based weight optimization.
"""

import torch


class DNFWeights:
    """Storage and management of DNF architecture weights"""

    def __init__(self):
        # Field parameters organized by category
        self.field_params = {
            'precondition_nodes': {},
            'system_nodes': {},
            'behavior_nodes': {},
            'check_nodes': {}
        }

        # Connection weights organized by connection type
        self.connection_weights = {
            'behavior_internal': {},  # Connections within behaviors
            'behavior_to_precond': {},  # Behavior CoS to preconditions
            'behavior_to_check': {},  # Behavior CoS to sanity checks
            'precond_to_next': {},  # Precondition to next behavior intention
            'system_level': {},  # System-level connections
            'mutual_inhibition': {}  # CoS/CoF mutual inhibition
        }

        self._initialize_defaults()

    def _initialize_defaults(self):
        """Initialize all default weights"""
        self._init_field_parameters()
        self._init_connection_weights()

    def _init_field_parameters(self):
        """Initialize neural field parameters"""

        # Precondition node defaults (applied to all behaviors)
        self.field_params['precondition_nodes']['default'] = {
            'shape': (),
            'time_step': 5.0,
            'time_scale': 100.0,
            'resting_level': -1.0,
            'beta': 20.0,
            'self_connection_w0': 2.0,
            'noise_strength': 0.0,
            'global_inhibition': 0.0,
            'scale': 1.0
        }

        # System-level nodes
        self.field_params['system_nodes']['intention'] = {
            'shape': (),
            'time_step': 5.0,
            'time_scale': 100.0,
            'resting_level': -3.0,
            'beta': 20.0,
            'self_connection_w0': 1.0,
            'noise_strength': 0.0,
            'global_inhibition': 0.0,
            'scale': 1.0
        }

        self.field_params['system_nodes']['cos'] = {
            'shape': (),
            'time_step': 5.0,
            'time_scale': 150.0,
            'resting_level': -4.75,
            'beta': 2.0,
            'self_connection_w0': 6.5,
            'noise_strength': 0.0,
            'global_inhibition': 0.0,
            'scale': 1.0
        }

        self.field_params['system_nodes']['cos_reporter'] = {
            'shape': (),
            'time_step': 5.0,
            'time_scale': 100.0,
            'resting_level': 1.0,
            'beta': 20.0,
            'self_connection_w0': 2.0,
            'noise_strength': 0.0,
            'global_inhibition': 0.0,
            'scale': 1.0
        }

        self.field_params['system_nodes']['cof'] = {
            'shape': (),
            'time_step': 5.0,
            'time_scale': 150.0,
            'resting_level': -4.75,
            'beta': 2.0,
            'self_connection_w0': 6.5,
            'noise_strength': 0.0,
            'global_inhibition': 0.0,
            'scale': 1.0
        }

        # Behavior node defaults (from ElementaryBehavior)
        self.field_params['behavior_nodes']['intention'] = {
            'shape': (),
            'time_step': 5.0,
            'time_scale': 100.0,
            'resting_level': -3.0,
            'beta': 100.0,
            'self_connection_w0': 1.0,
            'noise_strength': 0.0,
            'global_inhibition': 0.0,
            'scale': 1.0
        }

        self.field_params['behavior_nodes']['cos'] = {
            'shape': (),
            'time_step': 5.0,
            'time_scale': 100.0,
            'resting_level': -3.0,
            'beta': 100.0,
            'self_connection_w0': 1.0,
            'noise_strength': 0.0,
            'global_inhibition': 0.0,
            'scale': 1.0
        }

        self.field_params['behavior_nodes']['cos_inverter'] = {
            'shape': (),
            'time_step': 5.0,
            'time_scale': 100.0,
            'resting_level': 1.0,
            'beta': 100.0,
            'self_connection_w0': 1.0,
            'noise_strength': 0.0,
            'global_inhibition': 0.0,
            'scale': 1.0
        }

        self.field_params['behavior_nodes']['cof'] = {
            'shape': (),
            'time_step': 5.0,
            'time_scale': 100.0,
            'resting_level': -3.0,
            'beta': 100.0,
            'self_connection_w0': 1.0,
            'noise_strength': 0.0,
            'global_inhibition': 0.0,
            'scale': 1.0
        }

        # Check behavior nodes (from SanityCheckBehavior)
        self.field_params['check_nodes']['intention'] = {
            'shape': (),
            'time_step': 5.0,
            'time_scale': 100.0,
            'resting_level': -3.0,
            'beta': 100.0,
            'self_connection_w0': 1.0,
            'noise_strength': 0.0,
            'global_inhibition': 0.0,
            'scale': 1.0
        }

        self.field_params['check_nodes']['confidence'] = {
            'shape': (),
            'time_step': 5.0,
            'time_scale': 300.0,
            'resting_level': -4.0,
            'beta': 2.0,
            'self_connection_w0': 1.0,
            'noise_strength': 0.0,
            'global_inhibition': 0.0,
            'scale': 1.0
        }

    def _init_connection_weights(self):
        """Initialize connection weights"""

        # Behavior internal connections (ElementaryBehavior)
        self.connection_weights['behavior_internal'] = {
            'cos_to_intention': -7.0,
            'cos_to_inverter': -4.0,
            'cof_to_intention': -7.0, # Inverter Inhibits higher level reporter node
            'cof_to_cos': -7.0
        }

        # Check behavior internal connections (SanityCheckBehavior)
        self.connection_weights['check_internal'] = {
            'intention_to_confidence': 4.0
        }

        # Behavior to precondition
        self.connection_weights['behavior_to_precond'] = {
            'cos_to_precond': -10.0
        }

        # Behavior to check
        self.connection_weights['behavior_to_check'] = {
            'cos_to_check_intention': 5.0
        }

        # Precondition to next behavior intention
        self.connection_weights['precond_to_next'] = {
            'precond_to_intention': -4.5
        }

        # System-level connections
        self.connection_weights['system_level'] = {
            'intention_to_precond': 5.1,
            'intention_to_behavior': 5.0,
            'cos_inverter_to_reporter': -4.0,
            'reporter_to_system_cos': 3.3,
            'behavior_cof_to_system_cof': 4.0
        }

        # Mutual inhibition
        self.connection_weights['mutual_inhibition'] = {
            'system_cos_to_cof': -15.0,
            'system_cof_to_cos': -15.0
        }

    def get_field_params(self, category, node_type=None):
        """Get field parameters for a specific node type or category"""
        if node_type is None:
            return self.field_params[category].copy()

        return self.field_params[category].get(node_type, {}).copy()

    def get_connection_weight(self, category, connection_name=None):
        """Get a specific connection weight or the whole category"""
        if connection_name is None:
            return self.connection_weights[category].copy()

        return self.connection_weights[category].get(connection_name, 0.0)

    def set_field_param(self, category, node_type, param_name, value):
        """Set a specific field parameter"""
        if category not in self.field_params:
            self.field_params[category] = {}
        if node_type not in self.field_params[category]:
            self.field_params[category][node_type] = {}
        self.field_params[category][node_type][param_name] = value

    def set_connection_weight(self, category, connection_name, value):
        """Set a specific connection weight"""
        if category not in self.connection_weights:
            self.connection_weights[category] = {}
        self.connection_weights[category][connection_name] = value

    def to_dict(self):
        """Export all weights as a dictionary"""
        return {
            'field_params': self.field_params,
            'connection_weights': self.connection_weights
        }

    def from_dict(self, config_dict):
        """Load weights from a dictionary"""
        self.field_params = config_dict.get('field_params', {})
        self.connection_weights = config_dict.get('connection_weights', {})

    def save(self, filepath):
        """Save weights to a JSON file"""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    def load(self, filepath):
        """Load weights from a JSON file"""
        import json
        with open(filepath, 'r') as f:
            config = json.load(f)
            self.from_dict(config)

    def get_trainable_params(self):
        """
        Get all trainable parameters as a flat tensor for RL optimization.
        Returns:
            torch.Tensor: Flattened parameter vector
            list: Parameter metadata for reconstruction
        """
        params = []
        metadata = []

        # Field parameters
        for category, nodes in self.field_params.items():
            for node_type, params_dict in nodes.items():
                for param_name, value in params_dict.items():
                    if isinstance(value, (int, float)):
                        params.append(float(value))
                        metadata.append({
                            'type': 'field_param',
                            'category': category,
                            'node_type': node_type,
                            'param_name': param_name
                        })

        # Connection weights
        for category, weights in self.connection_weights.items():
            for conn_name, value in weights.items():
                params.append(float(value))
                metadata.append({
                    'type': 'connection_weight',
                    'category': category,
                    'conn_name': conn_name
                })

        return torch.tensor(params), metadata

    def set_from_trainable_params(self, param_tensor, metadata):
        """
        Update weights from a flat parameter tensor.

        Args:
            param_tensor: Flattened parameter tensor from RL
            metadata: Parameter metadata for reconstruction
        """
        for i, meta in enumerate(metadata):
            value = float(param_tensor[i])

            if meta['type'] == 'field_param':
                self.set_field_param(
                    meta['category'],
                    meta['node_type'],
                    meta['param_name'],
                    value
                )
            elif meta['type'] == 'connection_weight':
                self.set_connection_weight(
                    meta['category'],
                    meta['conn_name'],
                    value
                )


# Global instance
dnf_weights = DNFWeights()
