import random

import py_trees
import torch
import time
from functools import partial
from typing import Dict, Any
from interactors import RobotInteractors
from behavior_manager import run_behavior_manager


class BehaviorTreeNode(py_trees.behaviour.Behaviour):
    """Base class for behavior tree nodes that use interactors"""

    def __init__(self, name: str, interactors: RobotInteractors, behavior_args: Dict[str, Any]):
        super().__init__(name=name)
        self.interactors = interactors
        self.behavior_args = behavior_args


class FindObjectBT(BehaviorTreeNode):
    """Find target object using perception interactor"""

    def update(self):
        target = self.behavior_args.get('target_object')
        if not target:
            return py_trees.common.Status.FAILURE

        # Simulate search (in real scenario, this would trigger camera scan)
        result = self.interactors.perception.find_object(target)

        if result[0]:
            return py_trees.common.Status.SUCCESS
        elif result[1]:
            return py_trees.common.Status.FAILURE
        else:
            return py_trees.common.Status.RUNNING


class MoveToObjectBT(BehaviorTreeNode):
    """Move to target object location"""
    def update(self):
        target = self.behavior_args.get('target_object')
        if target not in self.interactors.perception.objects:
            return py_trees.common.Status.FAILURE

        target_pos = self.interactors.perception.objects[target]['location']
        current_pos = self.interactors.movement.get_position()

        self.interactors.movement.move_to(target_pos, requesting_behavior=self.name)

        # Check if reached (in plane
        distance = torch.norm(target_pos[:2] - current_pos[:2]).item()

        if distance < 0.1:
            return py_trees.common.Status.SUCCESS
        else:
            return py_trees.common.Status.RUNNING


class CheckReachBT(BehaviorTreeNode):
    """Check if object is reachable"""

    def update(self):
        target = self.behavior_args.get('target_object')
        if target not in self.interactors.perception.objects:
            return py_trees.common.Status.FAILURE

        target_pos = self.interactors.perception.objects[target]['location']
        result = self.interactors.gripper.reach_check(target_pos, requesting_behavior=self.name)

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

        target_pos = self.interactors.perception.objects[target]['location']

        self.interactors.gripper.reach_for(target_pos, requesting_behavior=self.name)

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
            target_pos = self.interactors.perception.objects[target]['location']
            target_orientation = self.interactors.perception.objects[target]['angle']
            result = self.interactors.gripper.grab(target, target_pos, target_orientation, requesting_behavior=self.name)

            if result[0]:
                self.phase = 'transport'
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

            self.interactors.movement.move_to(drop_pos, requesting_behavior=self.name)

            # Check if reached (in plane
            distance = torch.norm(drop_pos[:2] - current_pos[:2]).item()

            if distance < 0.1:
                return py_trees.common.Status.SUCCESS
            else:
                return py_trees.common.Status.RUNNING

        return py_trees.common.Status.FAILURE


class BehaviorTreeComparison:
    """Comparison system for benchmarking DNF-based vs Behavior Tree approach"""

    def __init__(self, interactors: RobotInteractors, behavior_args: Dict[str, Any]):
        self.interactors = interactors
        self.behavior_args = behavior_args
        self.tree = None
        self.build_tree()

    def build_tree(self):
        """Build the behavior tree sequence"""
        # Create behavior nodes
        find = FindObjectBT("Find", self.interactors, self.behavior_args)
        move = MoveToObjectBT("Move", self.interactors, self.behavior_args)
        check = CheckReachBT("CheckReach", self.interactors, self.behavior_args)
        reach = ReachForObjectBT("ReachFor", self.interactors, self.behavior_args)
        grab = GrabObjectBT("Grab", self.interactors, self.behavior_args)

        # Create sequence (all must succeed in order)
        sequence = py_trees.composites.Sequence(
            name="FindGrabSequence",
            memory=True,  # Remember progress
            children=[find, move, check, reach, grab]
        )

        self.tree = py_trees.trees.BehaviourTree(root=sequence)

    def execute(self, max_steps: int = 4000, external_perturbation=None):
        """Execute the behavior tree

        Args:
            max_steps: Maximum number of execution steps
            external_perturbation: Optional function(step, interactors) -> None for perturbations

        Returns:
            dict with execution metrics
        """
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


# Benchmark functions
def benchmark_completion_speed(num_runs: int = 10) -> Dict[str, Any]:
    """Compare completion speed between BT and DNF systems"""

    bt_times = []
    bt_steps = []
    bt_success = 0
    dnf_times = []
    dnf_steps = []
    dnf_success = 0

    print(f"\n=== Completion Speed Benchmark ({num_runs} runs) ===")

    behavior_args = {
        'target_object': 'cup',
        'drop_off_target': 'transport_target'
    }

    for run in range(num_runs):

        print(f"\nRun {run + 1}/{num_runs}")

        # Test Behavior Tree
        bt_interactors = RobotInteractors()
        # Reset object locations
        bt_interactors.perception.register_object(
            name="cup",
            location=torch.tensor([5.2, 10.5, 1.8]),
            angle=torch.tensor([0.0, -1.0, 0.0])
        )
        bt_interactors.perception.register_object(
            name="transport_target",
            location=torch.tensor([5.0, 0.0, 1.0]),
            angle=torch.tensor([0.0, 0.0, 0.0])
        )

        bt_system = BehaviorTreeComparison(bt_interactors, behavior_args)

        bt_result = bt_system.execute()
        bt_times.append(bt_result['time'])
        bt_steps.append(bt_result['steps'])
        if bt_result['success']:
            bt_success += 1
        print(f"  BT: {bt_result['steps']} steps, "
              f"{bt_result['time']:.3f}s, "
              f"{'Succeeded' if bt_result['success'] else 'Failed'}")



        # Test DNF System
        dnf_interactors = RobotInteractors()
        # Reset object locations
        dnf_interactors.perception.register_object(
            name="cup",
            location=torch.tensor([5.2, 10.5, 1.8]),
            angle=torch.tensor([0.0, -1.0, 0.0])
        )
        dnf_interactors.perception.register_object(
            name="transport_target",
            location=torch.tensor([5.0, 0.0, 1.0]),
            angle=torch.tensor([0.0, 0.0, 0.0])
        )

        behaviors = ['find', 'move', 'check_reach', 'reach_for', 'grab_transport']

        dnf_states, dnf_result = run_behavior_manager(behaviors=behaviors,
                             behavior_args=behavior_args,
                             interactors=dnf_interactors,
                             external_input=6.0,
                             max_steps=2000,
                             debug=False,
                             visualize_sim=False,
                             visualize_logs=True,
                             visualize_architecture=False,
                             timing=True,
                             verbose=False)

        dnf_times.append(dnf_result['time'])
        dnf_steps.append(dnf_result['steps'])
        if dnf_result['success']:
            dnf_success += 1

        print(f"  DNF: {dnf_result['steps']} steps, "
              f"{dnf_result['time']:.3f}s, "
              f"{'Succeeded' if dnf_result['success'] else 'Failed'}")

    return {
        'bt_avg_time': sum(bt_times) / len(bt_times),
        'bt_avg_steps': sum(bt_steps) / len(bt_steps),
        'bt_times': bt_times,
        'bt_steps': bt_steps,
        'bt_success': bt_success,
        'dnf_avg_time': sum(dnf_times) / len(dnf_times),
        'dnf_avg_steps': sum(dnf_steps) / len(dnf_steps),
        'dnf_times': dnf_times,
        'dnf_steps': dnf_steps,
        'dnf_success': dnf_success
    }


def benchmark_robustness(perturbation_types: list, num_runs: int = 5) -> Dict[str, Any]:
    """Compare robustness to perturbations"""

    random.seed(42)  # For reproducibility

    results = {}

    print(f"\n=== Robustness Benchmark ===")

    for perturbation_name, perturbation_func in perturbation_types:
        print(f"\nTesting perturbation: {perturbation_name}")

        bt_successes = 0
        dnf_successes = 0

        for run in range(num_runs):

            # Determine random trigger step for perturbation and pass it to the function
            trigger_step = random.randint(200, 1900)

            perturbation_with_trigger = partial(perturbation_func, trigger_step=trigger_step)

            print(f"  Run {run + 1}: Perturbation will trigger at step {trigger_step}")

            # Test BT
            # Reset object locations
            bt_interactors = RobotInteractors()  # Create new instance
            bt_interactors.perception.register_object(
                name="cup",
                location=torch.tensor([5.2, 10.5, 1.8]),
                angle=torch.tensor([0.0, -1.0, 0.0])
            )
            bt_interactors.perception.register_object(
                name="transport_target",
                location=torch.tensor([5.0, 0.0, 1.0]),
                angle=torch.tensor([0.0, 0.0, 0.0])
            )

            # Create new BT system with isolated interactors
            behavior_args = {
                'target_object': 'cup',
                'drop_off_target': 'transport_target'
            }
            bt_test_system = BehaviorTreeComparison(bt_interactors, behavior_args)

            bt_result = bt_test_system.execute(external_perturbation=perturbation_with_trigger)

            if bt_result['success']:
                bt_successes += 1

            print(f"  Run {run + 1}: BT {bt_result['final_status']}")

            # Test DNF
            dnf_interactors = RobotInteractors()  # Create separate instance
            dnf_interactors.perception.register_object(
                name="cup",
                location=torch.tensor([5.2, 10.5, 1.8]),
                angle=torch.tensor([0.0, -1.0, 0.0])
            )
            dnf_interactors.perception.register_object(
                name="transport_target",
                location=torch.tensor([5.0, 0.0, 1.0]),
                angle=torch.tensor([0.0, 0.0, 0.0])
            )

            behaviors = ['find', 'move', 'check_reach', 'reach_for', 'grab_transport']
            behavior_args = {
                'target_object': 'cup',
                'drop_off_target': 'transport_target'
            }

            _, dnf_results = run_behavior_manager(
                behaviors=behaviors,
                behavior_args=behavior_args,
                interactors=dnf_interactors,  # Use isolated interactors
                external_input=6.0,
                max_steps=4000,
                debug=False,
                visualize_sim=False,
                visualize_logs=False,
                visualize_architecture=False,
                timing=True,
                perturbation_simulation=perturbation_with_trigger
            )

            if dnf_results['success']:
                dnf_successes += 1

            print(f"  Run {run + 1}: DNF {dnf_results['final_status']}")

        results[perturbation_name] = {
            'bt_success_rate': bt_successes / num_runs,
            'dnf_success_rate': dnf_successes / num_runs
        }

    return results


# Example perturbation functions
def object_displacement_perturbation(step, interactors, trigger_step):
    """Move target object at step 100"""
    if step == trigger_step:
        target = interactors.perception.objects.get('cup')
        old_location = target['location'].clone() if target else None
        if target:
            target['location'] += torch.tensor([2.0, 2.0, 0.0])
            print(f"[PERTURBATION] Moved object from {old_location} to {target['location']}")


def sensor_noise_perturbation(step, interactors, trigger_step):
    """Add random noise to perception"""
    if trigger_step % 50 == 0:
        for obj_name, obj_data in interactors.perception.objects.items():
            noise = torch.randn(3) * 0.5
            obj_data['location'] += noise


if __name__ == "__main__":
    # Run benchmarks
    perturbations = [
        ("Object Displacement", object_displacement_perturbation),
        ("Sensor Noise", sensor_noise_perturbation),
    ]

    # Uncomment when DNF system interface is ready
    speed_results = benchmark_completion_speed(num_runs=10)
    robustness_results = benchmark_robustness(perturbations, num_runs=5)
