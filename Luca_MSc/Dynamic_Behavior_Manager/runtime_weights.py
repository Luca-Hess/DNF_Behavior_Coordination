"""
Runtime Weight Manager
Provides instance-level control over all DNF nodes and connections
after the behavior manager is built.
"""

import torch
from typing import Dict, List, Tuple, Any


class RuntimeWeightManager:
    """
    Manages all weights and parameters for a specific DNF execution instance.
    Generated at runtime after the behavior chain is constructed.
    """

    def __init__(self):
        # Instance-level storage
        self.field_instances = {}  # {unique_id: Field object}
        self.connection_instances = []  # List of connection dicts

        # Metadata for each instance
        self.field_metadata = {}  # {unique_id: metadata dict}
        self.connection_metadata = []  # List of metadata dicts

        # Original values for reset
        self.original_field_params = {}
        self.original_connection_weights = {}

    def register_field(self, field_obj, category: str, node_type: str,
                       behavior_name: str = None, instance_id: str = None):
        """
        Register a field instance for weight management.

        Args:
            field_obj: The Field object
            category: 'behavior_nodes', 'check_nodes', 'precondition_nodes', 'system_nodes'
            node_type: 'intention', 'cos', 'cof', etc.
            behavior_name: Name of the behavior this belongs to (if applicable)
            instance_id: Unique identifier (auto-generated if None)
        """
        if instance_id is None:
            if behavior_name:
                instance_id = f"{behavior_name}_{node_type}"
            else:
                instance_id = f"{category}_{node_type}"


        self.field_instances[instance_id] = field_obj
        self.field_metadata[instance_id] = {
            'category': category,
            'node_type': node_type,
            'behavior_name': behavior_name,
            'instance_id': instance_id
        }

        # Store original parameters
        self.original_field_params[instance_id] = {
            'time_step': float(field_obj.time_step),
            'time_scale': float(field_obj.time_scale),
            'resting_level': float(field_obj.resting_level),
            'beta': float(field_obj.beta),
            'self_connection_w0': float(field_obj.self_w),
            'noise_strength': float(field_obj.noise_strength),
            'global_inhibition': float(field_obj.global_inhibition),
            'scale': float(field_obj.scale)
        }

        return instance_id

    def register_connection(self, source_field, target_field, weight: float,
                            source_id: str, target_id: str,
                            connection_type: str = None):
        """
        Register a connection instance for weight management.

        Args:
            source_field: Source Field object
            target_field: Target Field object
            weight: Connection weight
            source_id: Unique ID of source field
            target_id: Unique ID of target field
            connection_type: Optional semantic type ('cos_to_precond', etc.)
        """
        conn_id = f"{source_id}_to_{target_id}"

        connection = {
            'id': conn_id,
            'source': source_field,
            'target': target_field,
            'source_id': source_id,
            'target_id': target_id,
            'connection_type': connection_type,
            'weight': weight
        }

        self.connection_instances.append(connection)
        self.connection_metadata.append({
            'id': conn_id,
            'source_id': source_id,
            'target_id': target_id,
            'connection_type': connection_type
        })

        # Store original weight
        self.original_connection_weights[conn_id] = weight

        return conn_id

    def set_field_param(self, instance_id: str, param_name: str, value: float):
        """Set a specific field parameter for a specific instance."""
        if instance_id not in self.field_instances:
            raise ValueError(f"Unknown field instance: {instance_id}")

        field = self.field_instances[instance_id]

        if param_name == 'time_step':
            field.time_step = value
        elif param_name == 'time_scale':
            field.time_scale = value
        elif param_name == 'resting_level':
            field.resting_level = value
        elif param_name == 'beta':
            field.beta = value
        elif param_name == 'self_connection_w0':
            field.self_w = value
        elif param_name == 'noise_strength':
            field.noise_strength = value
        elif param_name == 'global_inhibition':
            field.global_inhibition = value
        elif param_name == 'scale':
            field.scale = value
        else:
            raise ValueError(f"Unknown field parameter: {param_name}")

    def set_connection_weight(self, connection_id: str, weight: float):
        """Set weight for a specific connection instance."""
        # Find the connection
        conn = next((c for c in self.connection_instances if c['id'] == connection_id), None)
        if conn is None:
            raise ValueError(f"Unknown connection: {connection_id}")

        # Update the weight
        conn['weight'] = weight

        # Update the actual connection in the Field
        # This requires re-establishing the connection
        source = conn['source']
        target = conn['target']

        # Remove old connection and add new one
        source.clear_connection_to(target)
        source.connection_to(target, weight)

    def disable_connection(self, connection_id: str):
        """Disable a connection by setting its weight to 0."""
        self.set_connection_weight(connection_id, 0.0)

    def enable_connection(self, connection_id: str):
        """Re-enable a connection by restoring its original weight."""
        if connection_id in self.original_connection_weights:
            self.set_connection_weight(connection_id,
                                       self.original_connection_weights[connection_id])

    def get_field_params(self, instance_id: str) -> Dict[str, float]:
        """Get all parameters for a field instance."""
        if instance_id not in self.field_instances:
            raise ValueError(f"Unknown field instance: {instance_id}")

        field = self.field_instances[instance_id]
        return {
            'time_step': float(field.time_step),
            'time_scale': float(field.time_scale),
            'resting_level': float(field.resting_level),
            'beta': float(field.beta),
            'self_connection_w0': float(field.self_w),
            'noise_strength': float(field.noise_strength),
            'global_inhibition': float(field.global_inhibition),
            'scale': float(field.scale)
        }

    def get_connection_weight(self, connection_id: str) -> float:
        """Get weight for a specific connection."""
        conn = next((c for c in self.connection_instances if c['id'] == connection_id), None)
        if conn is None:
            raise ValueError(f"Unknown connection: {connection_id}")
        return conn['weight']

    def list_all_fields(self) -> List[str]:
        """Get list of all registered field instance IDs."""
        return list(self.field_instances.keys())

    def list_all_connections(self) -> List[str]:
        """Get list of all registered connection IDs."""
        return [c['id'] for c in self.connection_instances]

    def list_fields_by_behavior(self, behavior_name: str) -> List[str]:
        """Get all field instances for a specific behavior."""
        return [
            instance_id for instance_id, meta in self.field_metadata.items()
            if meta.get('behavior_name') == behavior_name
        ]

    def list_connections_by_behavior(self, behavior_name: str) -> List[str]:
        """Get all connections involving a specific behavior."""
        behavior_fields = self.list_fields_by_behavior(behavior_name)
        return [
            c['id'] for c in self.connection_instances
            if c['source_id'] in behavior_fields or c['target_id'] in behavior_fields
        ]

    def get_trainable_params(self) -> Tuple[torch.Tensor, List[Dict]]:
        """
        Get all instance-level parameters as a flat tensor for RL.

        Returns:
            params: Flattened parameter tensor
            metadata: Parameter metadata for reconstruction
        """
        params = []
        metadata = []

        # Field parameters
        for instance_id, field in self.field_instances.items():
            field_params = self.get_field_params(instance_id)
            for param_name, value in field_params.items():
                params.append(float(value))
                metadata.append({
                    'type': 'field_param',
                    'instance_id': instance_id,
                    'param_name': param_name,
                    **self.field_metadata[instance_id]
                })

        # Connection weights
        for conn in self.connection_instances:
            params.append(float(conn['weight']))
            metadata.append({
                'type': 'connection_weight',
                'connection_id': conn['id'],
                'source_id': conn['source_id'],
                'target_id': conn['target_id'],
                'connection_type': conn['connection_type']
            })

        return torch.tensor(params), metadata

    def set_from_trainable_params(self, param_tensor: torch.Tensor,
                                  metadata: List[Dict]):
        """
        Update all instance weights from a flat parameter tensor.

        Args:
            param_tensor: Flattened parameter tensor from RL
            metadata: Parameter metadata for reconstruction
        """
        for i, meta in enumerate(metadata):
            value = float(param_tensor[i])

            if meta['type'] == 'field_param':
                self.set_field_param(
                    meta['instance_id'],
                    meta['param_name'],
                    value
                )
            elif meta['type'] == 'connection_weight':
                self.set_connection_weight(
                    meta['connection_id'],
                    value
                )

    def reset_to_original(self):
        """Reset all parameters to their original values."""
        # Reset field parameters
        for instance_id, original_params in self.original_field_params.items():
            for param_name, value in original_params.items():
                self.set_field_param(instance_id, param_name, value)

        # Reset connection weights
        for conn_id, original_weight in self.original_connection_weights.items():
            self.set_connection_weight(conn_id, original_weight)

    def export_config(self) -> Dict[str, Any]:
        """Export current configuration as a dictionary."""
        config = {
            'fields': {},
            'connections': []
        }

        for instance_id in self.field_instances:
            config['fields'][instance_id] = {
                'params': self.get_field_params(instance_id),
                'metadata': self.field_metadata[instance_id]
            }

        for conn in self.connection_instances:
            config['connections'].append({
                'id': conn['id'],
                'source_id': conn['source_id'],
                'target_id': conn['target_id'],
                'connection_type': conn['connection_type'],
                'weight': conn['weight']
            })

        return config

    def save(self, filepath: str):
        """Save current configuration to JSON."""
        import json
        config = self.export_config()
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)

    def __repr__(self):
        return (f"RuntimeWeightManager("
                f"fields={len(self.field_instances)}, "
                f"connections={len(self.connection_instances)})")
