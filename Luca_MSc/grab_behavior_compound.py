import torch

from DNF_torch.field import Field
from Luca_MSc.elementary_behavior import ElementaryBehavior
from interactors import RobotInteractors

from grab_compounds.orient import OrientBehavior
from grab_compounds.open_gripper import OpenGripperBehavior
from grab_compounds.fine_reach import FineReachBehavior
from grab_compounds.close_gripper import CloseGripperBehavior
from grab_compounds.has_object import HasObjectBehavior


class GrabBehavior(ElementaryBehavior):
    """
    Composite behavior that chains orientation, opening, fine reaching, and closing
    to grab an object.
    """

    def __init__(self, field_params=None):
        super().__init__(field_params)

        # Create component behaviors
        self.orient_behavior = OrientBehavior()
        self.open_gripper_behavior = OpenGripperBehavior()
        self.fine_reach_behavior = FineReachBehavior()
        self.close_gripper_behavior = CloseGripperBehavior()
        self.has_object_behavior = HasObjectBehavior()

        # Toggle Node for Gripper Open/Close
        self.toggle_node = Field(
            shape=(),
            time_step=5.0,
            time_scale=100.0,
            resting_level=-3.0,
            beta=20.0,
            self_connection_w0=3.0
        )
        # Register buffer for prev state
        self.toggle_node.register_buffer(
            "g_u_prev",
            torch.zeros_like(self.toggle_node.g_u)
        )

        # Connections
        # Grab Behavior CoI -> Orient
        self.intention.connection_to(self.orient_behavior.intention, 6.0)

        # Orient -> OpenGripper
        self.orient_behavior.CoS.connection_to(self.open_gripper_behavior.intention, 6.0)

        # OpenGripper -> Fine Reach
        self.open_gripper_behavior.CoS.connection_to(self.fine_reach_behavior.intention, 6.0)

        # Toggle Node to inhibit Gripper oscillation
        self.intention.connection_to(self.toggle_node, 2.0)
        self.fine_reach_behavior.CoS.connection_to(self.toggle_node, 3.0)

        self.toggle_node.connection_to(self.close_gripper_behavior.intention, 6.0)
        self.toggle_node.connection_to(self.open_gripper_behavior.intention, -6.0)

        # CloseGripper -> HasObject
        self.close_gripper_behavior.CoS.connection_to(self.has_object_behavior.intention, 6.0)

        # HasObject -> Grab Behavior CoS
        self.has_object_behavior.CoS.connection_to(self.CoS, 6.0)

    def execute(self, interactors, target_name, target_location, external_input=0.0):
        """Execute the grab behavior sequence."""

        # Processing activity of the grab behavior intention node itself
        grab_state = self.forward()

        # Execute orient behavior
        orient_state = self.orient_behavior.execute(
            interactors,
            target_name,
            threshold=0.1,
            external_input=0.0
        )

        # Execute open gripper behavior (with no external input unless from precondition)
        open_gripper_state = self.open_gripper_behavior.execute(
            interactors.gripper,
            external_input=0.0
        )

        # Execute fine reach behavior
        fine_reach_state = self.fine_reach_behavior.execute(
            interactors.gripper,
            target_location,
            fine_threshold=0.1,
            external_input=0.0
        )

        # Cache prev state for the precondition
        self.toggle_node.cache_prev()

        # Process the precondition nodes (no external input)
        toggle_activation, toggle_activity = self.toggle_node()

        # Execute close gripper behavior
        close_gripper_state = self.close_gripper_behavior.execute(
            interactors.gripper,
            external_input=0.0
        )

        # Execute has object check behavior
        has_object_state = self.has_object_behavior.execute(
            interactors,
            target_location,
            external_input=0.0
        )

        # Compile state information
        state = {
            'grab': grab_state,
            'orient': orient_state,
            'open_gripper': open_gripper_state,
            'fine_reach': fine_reach_state,
            'toggle_node': {
                'activation': float(toggle_activation.detach()),
                'activity': float(toggle_activity.detach())
            },
            'close_gripper': close_gripper_state,
            'has_object': has_object_state
        }

        return state

    def reset(self):
        """Reset all behaviors and preconditions."""
        self.orient_behavior.reset()
        self.open_gripper_behavior.reset()
        self.fine_reach_behavior.reset()
        self.close_gripper_behavior.reset()
        self.has_object_behavior.reset()
        super().reset()

if __name__ == "__main__":
    # Example usage
    interactors = RobotInteractors()
    grab_behavior = GrabBehavior()

    target = torch.tensor([0.5, 0.0, 0.2])  # Example target location
    target_angle = torch.tensor([0.0, -1.0, 0.0])  # Example target orientation


    for step in range(300):
        state = grab_behavior.execute(interactors, target, target_angle, external_input=5.0)

        if state['has_object']['completed']:
            print("Grab behavior completed.")
