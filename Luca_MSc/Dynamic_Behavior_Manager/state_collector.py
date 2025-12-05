

class StateCollector:
    """Class to collect states during dynamic behavior management if enabled."""
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.states = {} if enabled else None

    def add(self, key, value):
        """Add state data"""
        if self.enabled:
            self.states[key] = value

    def get(self, key, default=None):
        """Get state data by key"""
        if not self.enabled:
            return default
        return self.states.get(key, default)

    def get_result(self, minimal_data=None):
        """Get final result - full states if enabled, minimal otherwise"""
        return self.states if self.enabled else minimal_data
