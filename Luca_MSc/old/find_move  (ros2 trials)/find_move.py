import sys
import os
sys.path.append(os.path.expanduser('~/nc_ws/DNF_torch'))

import torch

from DNF_torch.field import Field
from ros_interactors import RobotInteractors

from find import FindBehavior
from move_to import MoveToBehavior


class FindMoveBehavior():
    """
    Composite behavior that chains find and grab behaviors.
    """

    def __init__(self):
        # Create component behaviors
        self.find_behavior = FindBehavior()
        self.move_to_behavior = MoveToBehavior()

        ## Create precondition nodes
        # Object found precondition
        self.found_precond = Field(
            shape=(),
            time_step=5.0,
            time_scale=100.0,
            resting_level=-3.0,
            beta=20.0,
            self_connection_w0=2
        )

        # Robot base close enough to target
        self.close_precond = Field(
            shape=(),
            time_step=5.0,
            time_scale=100.0,
            resting_level=-3.0,
            beta=20.0,
            self_connection_w0=2
        )

        # Robot end-effector has enough reach to grab the target object
        self.in_reach_precond = Field(
            shape=(),
            time_step=5.0,
            time_scale=100.0,
            resting_level=-3.0,
            beta=20.0,
            self_connection_w0=2
        )

        # Robot end-effector has reached the target object
        self.reached_target_precond = Field(
            shape=(),
            time_step=5.0,
            time_scale=100.0,
            resting_level=-3.0,
            beta=20.0,
            self_connection_w0=2
        )

        # Robot has grabbed the target object
        self.has_grabbed_precond = Field(
            shape=(),
            time_step=5.0,
            time_scale=100.0,
            resting_level=-3.0,
            beta=20.0,
            self_connection_w0=2
        )

        self.transported_precond = Field(
            shape=(),
            time_step=5.0,
            time_scale=100.0,
            resting_level=-3.0,
            beta=20.0,
            self_connection_w0=2
        )

        # Register buffer for prev state
        self.found_precond.register_buffer(
            "g_u_prev",
            torch.zeros_like(self.found_precond.g_u)
        )
        self.close_precond.register_buffer(
            "g_u_prev",
            torch.zeros_like(self.close_precond.g_u)
        )


        # Connections
        # Find -> Found Precondition
        self.find_behavior.CoS.connection_to(self.found_precond, 6.0)

        # Found Precondition -> MoveTo
        self.found_precond.connection_to(self.move_to_behavior.intention, 6.0)
        self.move_to_behavior.CoS.connection_to(self.close_precond, 6.0)

    def execute_step(self, interactors, target_name, external_input=5.0):
        """Execute the find portion of the behavior chain."""

        # Cache prev state for the precondition
        self.found_precond.cache_prev()
        self.close_precond.cache_prev()

        # Execute find behavior
        find_state = self.find_behavior.execute(interactors.perception,
                                                target_name,
                                                external_input)

        # Process the precondition nodes (no external input)
        found_activation, found_activity = self.found_precond()
        close_activation, close_activity = self.close_precond()


        # Execute move-to behavior with precondition input, only if found is active
        move_state = None
        if find_state['target_location'] is not None:
            move_state = self.move_to_behavior.execute(
                interactors.movement,
                find_state['target_location'],
                external_input = 0.0
            )


        state = {
            'find': find_state,
            'move': move_state,
            'preconditions': {
                'found': {
                    'activation': float(found_activation.detach()),
                    'activity': float(found_activity.detach()),
                    'active': float(found_activity) > 0.7
                },
                'close': {
                    'activation': float(close_activation.detach()),
                    'activity': float(close_activity.detach()),
                    'active': float(close_activity) > 0.7
                }
            },
        }

        return state

    def reset(self):
        """Reset all behaviors and preconditions."""
        self.find_behavior.reset()
        self.found_precond.reset()
        self.move_to_behavior.reset()
        self.close_precond.reset()

# Example usage
if __name__ == "__main__":

    # External input activates find_grab behavior sequence
    external_input = 5.0

    # Create interactors with a test object
    interactors = RobotInteractors()

    # Create the find-grab behavior
    find_move = FindMoveBehavior()

    # Run the find behavior until completion
    print("Starting find behavior for 'cricket ball'...")
    i = 0
    done = 0
    for step in range(1200):
        # Execute find behavior
        state = find_move.execute_step(interactors, "cricket_ball", external_input)


