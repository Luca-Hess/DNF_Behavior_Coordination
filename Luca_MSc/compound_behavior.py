import torch
import numpy as np
import matplotlib.pyplot as plt

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

        # Connections
        # Orient -> OpenGripper
        self.intention.connection_to(self.orient_behavior.intention, 6.0)
        self.orient_behavior.CoS.connection_to(self.open_gripper_behavior.intention, 6.0)

        # OpenGripper -> Fine Reach
        self.open_gripper_behavior.CoS.connection_to(self.fine_reach_behavior.intention, 6.0)

        # Fine Reach -> GripperToggled (proximity to target prevents gripper from being held open)
        self.fine_reach_behavior.CoS.connection_to(self.open_gripper_behavior.intention, -6.0)

        # FineReach -> CloseGripper
        self.fine_reach_behavior.CoS.connection_to(self.close_gripper_behavior.intention, 6.0)

        # CloseGripper -> HasObject
        self.close_gripper_behavior.CoS.connection_to(self.has_object_behavior.intention, 6.0)

        # HasObject -> Grab Behavior CoS
        self.has_object_behavior.CoS.connection_to(self.CoS, 6.0)

    def execute(self, interactors, target_location, target_angle, external_input=0.0):
        """Execute the grab behavior sequence."""

        # Processing activity of the grab behavior intention node itself
        self.forward(intention_input=external_input, cos_input=0.0)

        # Execute orient behavior
        orient_state = self.orient_behavior.execute(
            interactors.gripper,
            target_angle,
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
            fine_threshold=0.05,
            external_input=0.0
        )

        # Execute close gripper behavior
        close_gripper_state = self.close_gripper_behavior.execute(
            interactors.gripper,
            external_input=0.0
        )

        # Execute has object behavior
        has_object_state = self.has_object_behavior.execute(
            interactors.gripper,
            external_input=0.0
        )

        # Compile state information
        state = {
            'orient': orient_state,
            'open_gripper': open_gripper_state,
            'fine_reach': fine_reach_state,
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

    log = {
        "orient_intention_activation": [],
        "orient_intention_activity": [],
        "orient_cos_activation": [],
        "orient_cos_activity": [],
        "open_intention_activation": [],
        "open_intention_activity": [],
        "open_cos_activation": [],
        "open_cos_activity": [],
        "fine_reach_intention_activation": [],
        "fine_reach_intention_activity": [],
        "fine_reach_cos_activation": [],
        "fine_reach_cos_activity": [],
        "close_intention_activation": [],
        "close_intention_activity": [],
        "close_cos_activation": [],
        "close_cos_activity": [],
        "has_object_intention_activation": [],
        "has_object_intention_activity": [],
        "has_object_cos_activation": [],
        "has_object_cos_activity": []
    }

    for step in range(300):
        state = grab_behavior.execute(interactors, target, target_angle, external_input=5.0)

        #print(f"Step {step}: {state}")
        if state['has_object']['completed']:
            print("Grab behavior completed.")
            #break

        log["orient_intention_activation"].append(state['orient']['intention_activation'])
        log["orient_intention_activity"].append(state['orient']['intention_activity'])
        log["orient_cos_activation"].append(state['orient']['cos_activation'])
        log["orient_cos_activity"].append(state['orient']['cos_activity'])

        log["open_intention_activation"].append(state['open_gripper']['intention_activation'])
        log["open_intention_activity"].append(state['open_gripper']['intention_activity'])
        log["open_cos_activation"].append(state['open_gripper']['cos_activation'])
        log["open_cos_activity"].append(state['open_gripper']['cos_activity'])

        log["fine_reach_intention_activation"].append(state['fine_reach']['intention_activation'])
        log["fine_reach_intention_activity"].append(state['fine_reach']['intention_activity'])
        log["fine_reach_cos_activation"].append(state['fine_reach']['cos_activation'])
        log["fine_reach_cos_activity"].append(state['fine_reach']['cos_activity'])

        log["close_intention_activation"].append(state['close_gripper']['intention_activation'])
        log["close_intention_activity"].append(state['close_gripper']['intention_activity'])
        log["close_cos_activation"].append(state['close_gripper']['cos_activation'])
        log["close_cos_activity"].append(state['close_gripper']['cos_activity'])

        log["has_object_intention_activation"].append(state['has_object']['intention_activation'])
        log["has_object_intention_activity"].append(state['has_object']['intention_activity'])
        log["has_object_cos_activation"].append(state['has_object']['cos_activation'])
        log["has_object_cos_activity"].append(state['has_object']['cos_activity'])



    ts = np.arange(300)

    groups = [
        (
            "Orient Behavior",
            [
                ("orient_intention_activation", "Intention Orient (activation)", "-"),
                ("orient_cos_activation", "CoS orient (activation)", "-"),
                ("orient_intention_activity", "Intention orient (activity)", "--"),
                ("orient_cos_activity", "CoS orient (activity)", "--"),
            ],
        ),
        (
            "Open Gripper Behavior",
            [
                ("open_intention_activation", "Intention Open (activation)", "-"),
                ("open_cos_activation", "CoS Open (activation)", "-"),
                ("open_intention_activity", "Intention Open (activity)", "--"),
                ("open_cos_activity", "CoS Open (activity)", "--"),
            ],
        ),
        (
            "Fine Reach Behavior",
            [
                ("fine_reach_intention_activation", "Intention Fine Reach (activation)", "-"),
                ("fine_reach_cos_activation", "CoS Fine Reach (activation)", "-"),
                ("fine_reach_intention_activity", "Intention Fine Reach (activity)", "--"),
                ("fine_reach_cos_activity", "CoS Fine Reach (activity)", "--"),
            ],
        ),
        (
            "Close Gripper Behavior",
            [
                ("close_intention_activation", "Intention Close (activation)", "-"),
                ("close_cos_activation", "CoS Close (activation)", "-"),
                ("close_intention_activity", "Intention Close (activity)", "--"),
                ("close_cos_activity", "CoS Close (activity)", "--"),
            ],
        ),
        (
            "Has Object Behavior",
            [
                ("has_object_intention_activation", "Intention Has Object (activation)", "-"),
                ("has_object_cos_activation", "CoS Has Object (activation)", "-"),
                ("has_object_intention_activity", "Intention Has Object (activity)", "--"),
                ("has_object_cos_activity", "CoS Has Object (activity)", "--"),
            ],
        ),
    ]


    fig, axes = plt.subplots(len(groups), 1, figsize=(10, 12), sharex=True)

    for ax, (title, signals) in zip(axes, groups):
        for key, label, style in signals:
            ax.plot(ts, log[key], style, label=label)
        ax.set_ylabel("Value")
        ax.set_title(title)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Step")
    plt.tight_layout()
    plt.show()