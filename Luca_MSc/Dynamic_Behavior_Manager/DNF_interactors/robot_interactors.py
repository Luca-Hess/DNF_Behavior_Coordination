
from .base_interactors  import BaseInteractor
from .movement_interactors import MovementInteractor
from .gripper_interactors import GripperInteractor
from .perception_interactors import PerceptionInteractor
from .state_interactors import StateInteractor


class RobotInteractors:
    """Facade that provides access to all interactors."""

    def __init__(self):
        self.shared_states = {}

        self.base = BaseInteractor()
        self.movement = MovementInteractor()
        self.gripper = GripperInteractor(get_robot_position=self.movement.get_position)
        self.perception = PerceptionInteractor(get_robot_position=self.movement.get_position)
        self.state = StateInteractor(self.perception, self.movement, self.gripper)

        # Point all interactors to the centralized shared state, allows cross-interactor state sharing
        self.base.shared_states = self.shared_states
        self.movement.shared_states = self.shared_states
        self.gripper.shared_states = self.shared_states
        self.perception.shared_states = self.shared_states
        self.state.shared_states = self.shared_states

        # Allow interactors to reference back to this facade if needed
        self.base._robot_interactors = self
        self.movement._robot_interactors = self
        self.gripper._robot_interactors = self
        self.perception._robot_interactors = self



    def reset(self):
        """Reset all interactors."""
        self.base.reset()
        self.perception.reset()
        self.movement.reset()
        self.gripper.reset()
        self.state.reset()