import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import py_trees
import torch
import time

from functools import partial
from typing import Dict, Any
from Luca_MSc.Dynamic_Behavior_Manager.behavior_manager import BehaviorManager
from Luca_MSc.Dynamic_Behavior_Manager.initializer import Initializer
from Luca_MSc.Dynamic_Behavior_Manager.DNF_interactors.robot_interactors import RobotInteractors

class BehaviorTreeNode(py_trees.behaviour.Behaviour):
    """Base class for behavior tree nodes that use interactors"""

    def __init__(self,
                 name: str,
                 interactors: RobotInteractors,
                 behavior_args: Dict[str, Any],
                 shared_state: dict):
        super().__init__(name=name)
        self.interactors = interactors
        self.behavior_args = behavior_args
        self.shared_state = shared_state

    def write_state(self, key: str, value: Any):
        """Write to shared state dictionary."""
        self.shared_state[key] = value

    def read_state(self, key: str) -> Any:
        """Read from shared state dictionary."""
        return self.shared_state.get(key, None)


class FindObjectBT(BehaviorTreeNode):
    """Find target object using perception interactor"""

    def update(self):
        target = self.behavior_args.get('target_object')
        if not target:
            return py_trees.common.Status.FAILURE

        # Simulate search
        result = self.interactors.perception.find_object(continuous_behavior=self.name)

        if result[0]:
            self.write_state(f'{target}_location', result[2].clone())
            self.write_state(f'{target}_orientation', result[3].clone())
            return py_trees.common.Status.SUCCESS
        elif result[1]:
            return py_trees.common.Status.FAILURE
        else:
            return py_trees.common.Status.RUNNING


class MoveToObjectBT(BehaviorTreeNode):
    """Move to target using only shared state sensor data - detects failure via progress stagnation."""

    def initialise(self):
        """Reset progress tracking when node starts/restarts."""
        self.last_distance = None
        self.stagnant_ticks = 0
        self.max_stagnant_ticks = 50  # ~250ms at 5ms/tick - realistic timeout

    def update(self):
        target = self.behavior_args.get('target_object')
        target_pos = self.read_state(f'{target}_location')

        if target_pos is None:
            return py_trees.common.Status.FAILURE

        current_pos = self.interactors.movement.get_position()
        self.interactors.movement.move_to(continuous_behavior=self.name)

        distance = torch.norm(target_pos[:2] - current_pos[:2]).item()

        # Success: close enough to *sensor* target
        if distance < 0.1:
            return py_trees.common.Status.SUCCESS

        # Failure detection: distance stopped decreasing (object moved or never reached)
        if self.last_distance is not None:
            if abs(distance - self.last_distance) < 0.01:  # No progress
                self.stagnant_ticks += 1
                if self.stagnant_ticks >= self.max_stagnant_ticks:
                    # Observable failure: robot stuck or target moved
                    return py_trees.common.Status.FAILURE
            else:
                self.stagnant_ticks = 0  # Reset if making progress

        self.last_distance = distance
        return py_trees.common.Status.RUNNING


class CheckReachBT(BehaviorTreeNode):
    """Check if object is reachable"""

    def update(self):
        target = self.behavior_args.get('target_object')
        if target not in self.interactors.perception.objects:
            return py_trees.common.Status.FAILURE

        target_pos = self.read_state(f'{target}_location')
        result = self.interactors.gripper.reach_check(continuous_behavior=self.name)

        if result[0]:  # Reachable
            return py_trees.common.Status.SUCCESS
        elif result[1]:  # Never reachable (height of object larger than max reach)
            return py_trees.common.Status.FAILURE
        else:
            return py_trees.common.Status.RUNNING


class ReachForObjectBT(BehaviorTreeNode):
    """Reach for target object"""

    def update(self):
        target = self.behavior_args.get('target_object')
        if target not in self.interactors.perception.objects:
            return py_trees.common.Status.FAILURE

        target_pos = self.read_state(f'{target}_location')
        self.interactors.gripper.reach_for(continuous_behavior=self.name)

        # Check if reached
        gripper_pos = self.interactors.gripper.get_position()
        distance = torch.norm(target_pos - gripper_pos).item()

        if distance < 0.01:
            return py_trees.common.Status.SUCCESS
        else:
            return py_trees.common.Status.RUNNING


class GrabObjectBT(BehaviorTreeNode):
    """Grab and transport object"""

    def initialise(self):
        self.phase = 'grab'

    def update(self):
        target = self.behavior_args.get('target_object')
        drop_off = self.behavior_args.get('drop_off_target')

        if target not in self.interactors.perception.objects:
            return py_trees.common.Status.FAILURE

        if self.phase == 'grab':
            # Grab object
            target_pos = self.read_state(f'{target}_location')
            target_orientation = self.read_state(f'{target}_orientation')
            result = self.interactors.gripper.grab(continuous_behavior=self.name)

            if result[0]:
                self.phase = 'transport'
                self.interactors.state.update_behavior_target('move_to',
                                                              'transport_target',
                                                              continuous_behavior=self.name)
                return py_trees.common.Status.RUNNING
            elif result[1]:
                return py_trees.common.Status.FAILURE
            else:
                return py_trees.common.Status.RUNNING

        elif self.phase == 'transport':
            # Transport to drop-off location
            if drop_off not in self.interactors.perception.objects:
                return py_trees.common.Status.FAILURE

            drop_pos = self.interactors.perception.objects[drop_off]['location']
            current_pos = self.interactors.movement.get_position()

            self.interactors.movement.move_to(continuous_behavior=self.name)

            # Check if reached (in plane)
            distance = torch.norm(drop_pos[:2] - current_pos[:2]).item()

            # Check if the object is actually still held - to make sure BT does not report false success at the end
            gt_object_location = self.interactors.perception.objects['cup']['location']
            gripper_position = self.interactors.gripper.get_position()
            gripper_state = self.interactors.gripper.has_object(gripper_position, gt_object_location)

            if not gripper_state:
                return py_trees.common.Status.FAILURE
            if distance < 0.1:
                return py_trees.common.Status.SUCCESS
            else:
                return py_trees.common.Status.RUNNING


        return py_trees.common.Status.FAILURE


class BehaviorTreeComparison:
    """Comparison system for benchmarking DNF-based vs Behavior Tree approach"""

    def __init__(self, interactors: RobotInteractors, behavior_args: Dict[str, Any], max_retries: int = 3):
        self.interactors = interactors
        self.behavior_args = behavior_args
        self.max_retries = max_retries
        self.tree = None

    def build_tree(self, shared_state: dict):
        """Build the behavior tree sequence"""
        # Create behavior nodes
        find = FindObjectBT("Find", self.interactors, self.behavior_args, shared_state)
        move = MoveToObjectBT("Move", self.interactors, self.behavior_args, shared_state)
        check = CheckReachBT("CheckReach", self.interactors, self.behavior_args, shared_state)
        reach = ReachForObjectBT("ReachFor", self.interactors, self.behavior_args, shared_state)
        grab = GrabObjectBT("Grab", self.interactors, self.behavior_args, shared_state)

        # Create sequence (all must succeed in order)
        sequence = py_trees.composites.Sequence(
            name="FindGrabSequence",
            memory=True,  # Remember progress
            children=[find, move, check, reach, grab]
        )

        # Wrap sequence in retry decorator to allow BT recovery
        retry_sequence = py_trees.decorators.Retry(
            name="RetrySequence",
            child=sequence,
            num_failures = self.max_retries
        )

        # Provide interactors with target information
        behaviors = ['find', 'move', 'check_reach', 'reach_for', 'grab']
        behavior_manager = BehaviorManager(behaviors, self.behavior_args)
        initializer = Initializer(behavior_manager)
        behavior_chain = initializer.build_behavior_chain(behaviors)
        self.interactors.state.initialize_from_behavior_chain(behavior_chain, self.behavior_args)

        self.tree = py_trees.trees.BehaviourTree(root=retry_sequence)

    def execute(self, max_steps: int = 4000, external_perturbation=None):
        """Execute the behavior tree

        Args:
            max_steps: Maximum number of execution steps
            external_perturbation: Optional function(step, interactors) -> None for perturbations

        Returns:
            dict with execution metrics
        """
        shared_state = {}
        self.build_tree(shared_state=shared_state)

        start_time = time.time()

        for step in range(max_steps):
            # Apply external perturbation if provided
            if external_perturbation is not None:
                external_perturbation(step, self.interactors)

            # Tick the tree
            self.tree.tick()
            time.sleep(0.005) # Simulate DNF 5ms system time per step

            # Check completion
            if self.tree.root.status == py_trees.common.Status.SUCCESS:
                end_time = time.time()
                return {
                    'success': True,
                    'steps': step + 1,
                    'time': end_time - start_time,
                    'final_status': 'SUCCESS'
                }
            elif self.tree.root.status == py_trees.common.Status.FAILURE:
                end_time = time.time()
                return {
                    'success': False,
                    'steps': step + 1,
                    'time': end_time - start_time,
                    'final_status': 'FAILURE'
                }

        # Timeout
        end_time = time.time()
        return {
            'success': False,
            'steps': max_steps,
            'time': end_time - start_time,
            'final_status': 'TIMEOUT'
        }

    def reset(self):
        """Reset the behavior tree"""
        self.tree.root.stop(py_trees.common.Status.INVALID)
        self.build_tree()
