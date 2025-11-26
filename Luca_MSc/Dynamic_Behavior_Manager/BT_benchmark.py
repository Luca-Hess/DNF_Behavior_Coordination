import py_trees
import torch
import time
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

        # Check if object is already found
        if target in self.interactors.perception.objects:
            self.feedback_message = f"Object '{target}' found"
            return py_trees.common.Status.SUCCESS

        # Simulate search (in real scenario, this would trigger camera scan)
        result = self.interactors.perception.find_object(target)

        if result[0]:
            return py_trees.common.Status.SUCCESS
        else:
            return py_trees.common.Status.RUNNING


class MoveToObjectBT(BehaviorTreeNode):
    """Move to target object location"""

    def initialise(self):
        self.started = False

    def update(self):
        target = self.behavior_args.get('target_object')
        if target not in self.interactors.perception.objects:
            return py_trees.common.Status.FAILURE

        target_pos = self.interactors.perception.objects[target]['location']
        current_pos = self.interactors.movement.get_position()

        # Start movement if not started
        if not self.started:
            self.interactors.movement.move_to_position(target_pos, requesting_behavior=self.name)
            self.started = True

        # Check if reached
        distance = torch.norm(target_pos - current_pos).item()

        if distance < 0.5:
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
        result = self.interactors.gripper.check_reachability(target_pos, requesting_behavior=self.name)

        if result[0] and result[1]:  # Success and reachable
            return py_trees.common.Status.SUCCESS
        elif result[0] and not result[1]:  # Success but not reachable
            return py_trees.common.Status.FAILURE
        else:
            return py_trees.common.Status.RUNNING


class ReachForObjectBT(BehaviorTreeNode):
    """Reach for target object"""

    def initialise(self):
        self.started = False

    def update(self):
        target = self.behavior_args.get('target_object')
        if target not in self.interactors.perception.objects:
            return py_trees.common.Status.FAILURE

        target_pos = self.interactors.perception.objects[target]['location']

        if not self.started:
            self.interactors.gripper.reach_to_position(target_pos, requesting_behavior=self.name)
            self.started = True

        # Check if reached
        gripper_pos = self.interactors.gripper.get_position()
        distance = torch.norm(target_pos - gripper_pos).item()

        if distance < 0.1:
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
            result = self.interactors.gripper.grab_object(target, requesting_behavior=self.name)

            if result[0]:
                self.phase = 'transport'
            return py_trees.common.Status.RUNNING

        elif self.phase == 'transport':
            # Transport to drop-off location
            if drop_off not in self.interactors.perception.objects:
                return py_trees.common.Status.FAILURE

            drop_pos = self.interactors.perception.objects[drop_off]['location']
            current_pos = self.interactors.gripper.get_position()

            self.interactors.gripper.reach_to_position(drop_pos, requesting_behavior=self.name)

            distance = torch.norm(drop_pos - current_pos).item()
            if distance < 0.1:
                # Release object
                self.interactors.gripper.release_object(requesting_behavior=self.name)
                return py_trees.common.Status.SUCCESS

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

    def execute(self, max_steps: int = 2000, external_perturbation=None):
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
def benchmark_completion_speed(bt_system: BehaviorTreeComparison, dnf_system,
                               num_runs: int = 10) -> Dict[str, Any]:
    """Compare completion speed between BT and DNF systems"""

    bt_times = []
    bt_steps = []
    dnf_times = []
    dnf_steps = []

    print(f"\n=== Completion Speed Benchmark ({num_runs} runs) ===")

    for run in range(num_runs):
        print(f"\nRun {run + 1}/{num_runs}")

        # Test Behavior Tree
        bt_system.reset()
        bt_system.interactors.reset()
        bt_result = bt_system.execute()
        bt_times.append(bt_result['time'])
        bt_steps.append(bt_result['steps'])
        print(f"  BT: {bt_result['steps']} steps, {bt_result['time']:.3f}s")

        # Test DNF System
        dnf_system.reset()
        dnf_system.interactors.reset()

        behaviors = ['find', 'move', 'check_reach', 'reach_for', 'grab_transport']
        behavior_args = {
            'target_object': 'cup',
            'drop_off_target': 'transport_target'
        }

        # External input activates find_grab behavior sequence
        external_input = 6.0

        # Create interactors with a test object
        interactors = RobotInteractors()
        interactors.perception.register_object(name="cup",
                                               location=torch.tensor([5.2, 10.5, 1.8]),
                                               angle=torch.tensor([0.0, -1.0, 0.0]))

        # Define drop-off location for transport
        # Important: As a matter of definition, this is considered to be "given" and thus registered from the start
        # in a scenario that involves transporting an object to a specific location provided by the user (similar to the way the target object is given).
        # An alternative scenario could be to have a behavior subsequent to "grab" that actively queries the user for a drop-off location.
        interactors.perception.register_object(name="transport_target",
                                               location=torch.tensor([5.0, 0.0, 1.0]),
                                               angle=torch.tensor([0.0, 0.0, 0.0]))

        interactors.perception.register_object(name="bottle",
                                               location=torch.tensor([8.0, 12.0, 1.5]),
                                               angle=torch.tensor([0.0, -1.0, 0.0]))

        run_behavior_manager(behaviors=behaviors,
                             behavior_args=behavior_args,
                             interactors=interactors,
                             external_input=external_input,
                             max_steps=2000,
                             debug=False,
                             visualize_sim=False,
                             visualize_logs=True,
                             visualize_architecture=False)

        dnf_times.append(dnf_result['time'])
        dnf_steps.append(dnf_result['steps'])

        print(f"  BT: {dnf_result['steps']} steps, {dnf_result['time']:.3f}s")

    return {
        'bt_avg_time': sum(bt_times) / len(bt_times),
        'bt_avg_steps': sum(bt_steps) / len(bt_steps),
        'bt_times': bt_times,
        'bt_steps': bt_steps,
        # Add DNF metrics
    }


def benchmark_robustness(bt_system: BehaviorTreeComparison, dnf_system,
                         perturbation_types: list, num_runs: int = 5) -> Dict[str, Any]:
    """Compare robustness to perturbations"""

    results = {}

    print(f"\n=== Robustness Benchmark ===")

    for perturbation_name, perturbation_func in perturbation_types:
        print(f"\nTesting perturbation: {perturbation_name}")

        bt_successes = 0

        for run in range(num_runs):
            # Test BT
            bt_system.reset()
            bt_system.interactors.reset()
            bt_result = bt_system.execute(external_perturbation=perturbation_func)

            if bt_result['success']:
                bt_successes += 1

            print(f"  Run {run + 1}: BT {bt_result['final_status']}")

        results[perturbation_name] = {
            'bt_success_rate': bt_successes / num_runs,
            # Add DNF metrics
        }

    return results


# Example perturbation functions
def object_displacement_perturbation(step, interactors):
    """Move target object at step 100"""
    if step == 100:
        target = interactors.perception.objects.get('cup')
        if target:
            target['location'] += torch.tensor([2.0, 2.0, 0.0])
            print(f"[PERTURBATION] Moved object to {target['location']}")


def sensor_noise_perturbation(step, interactors):
    """Add random noise to perception"""
    if step % 50 == 0:
        for obj_name, obj_data in interactors.perception.objects.items():
            noise = torch.randn(3) * 0.5
            obj_data['location'] += noise


if __name__ == "__main__":
    # Setup
    interactors = RobotInteractors()
    interactors.perception.register_object(
        name="cup",
        location=torch.tensor([5.2, 10.5, 1.8]),
        angle=torch.tensor([0.0, -1.0, 0.0])
    )
    interactors.perception.register_object(
        name="transport_target",
        location=torch.tensor([5.0, 0.0, 1.0]),
        angle=torch.tensor([0.0, 0.0, 0.0])
    )

    behavior_args = {
        'target_object': 'cup',
        'drop_off_target': 'transport_target'
    }

    # Create BT system
    bt_system = BehaviorTreeComparison(interactors, behavior_args)

    # Run single test
    print("Running behavior tree system...")
    result = bt_system.execute()
    print(f"\nResult: {result}")

    # Run benchmarks
    # perturbations = [
    #     ("Object Displacement", object_displacement_perturbation),
    #     ("Sensor Noise", sensor_noise_perturbation)
    # ]

    # Uncomment when DNF system interface is ready
    speed_results = benchmark_completion_speed(bt_system, dnf_system, num_runs=10)
    # robustness_results = benchmark_robustness(bt_system, dnf_system, perturbations, num_runs=5)
