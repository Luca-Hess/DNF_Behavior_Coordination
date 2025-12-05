
from .base_interactors import BaseInteractor

import random
import torch

random.seed(2)  # seed 2 has no collapses for find, 3 has two


class PerceptionInteractor(BaseInteractor):
    def __init__(self, get_robot_position=None, **kwargs):
        super().__init__(**kwargs)

        self.objects = {}
        self.target_states = {}
        self._get_robot_position = get_robot_position

        # Simulation parameters for object finding behavior
        self.search_attempts = 0
        self.tracking_loss_probability = 0.1  # Probability to lose tracking on each
        self.max_tracking_loss_duration = 5  # Max steps to lose tracking
        self.in_tracking_loss = False
        self.tracking_loss_remaining = 0

    def find_object(self, requesting_behavior=None):
        """Unified find object method - behaves differently based on requesting_behavior"""
        target_name = self._robot_interactors.state.get_behavior_target_name('find_object')

        target_found, location, angle, failure_reason = self._find_object_internal(target_name)

        # Determine CoS & CoF condition
        cos_condition = target_found
        cof_condition = (failure_reason is not None)

        state_data = {
            'target_name': target_name,
            'target_found': target_found,
            'target_location': location,
            'target_angle': angle,
            'failure_reason': failure_reason
        }

        # Use base class helper for state management and publishing
        self._update_and_publish_state(state_data, target_found, cof_condition, requesting_behavior)

        return cos_condition, cof_condition, location, angle

    def _find_object_internal(self, name):
        """Internal object finding logic"""
        self.search_attempts += 1

        # Target does not exist in the environment
        if name not in self.objects:
            failure_reason = f"Inexistent: {name} not found in environment."
            return False, None, None, failure_reason

        # Check if we should start a tracking loss period
        if not self.in_tracking_loss and random.random() < self.tracking_loss_probability:
            self.in_tracking_loss = True
            self.tracking_loss_remaining = random.randint(1, self.max_tracking_loss_duration)

        # If we're in a tracking loss period
        if self.in_tracking_loss:
            self.tracking_loss_remaining -= 1
            if self.tracking_loss_remaining <= 0:
                self.in_tracking_loss = False
            return False, None, None, None

        # Normal detection logic
        if name in self.objects and self.search_attempts >= 3:
            return True, self.objects[name]['location'], self.objects[name]['angle'], None
        return False, None, None, None

    def register_object(self, name, location, angle=torch.tensor([0.0, 0.0, 0.0])):
        """Register or update an object's location and angle."""
        self.objects[name] = {
            'location': location,
            'angle': angle,
            'lost_counter': 0
        }

    def reset(self):
        """Reset perception state."""
        super().reset()
        self.target_states.clear()
        self.objects.clear()
        self.search_attempts = 0
        self.tracking_loss_remaining = 0
