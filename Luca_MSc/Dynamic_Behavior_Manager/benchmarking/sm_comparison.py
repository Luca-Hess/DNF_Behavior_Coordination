import smach
import torch
import time

from typing import Dict, Any
from Luca_MSc.Dynamic_Behavior_Manager.behavior_manager import BehaviorManager
from Luca_MSc.Dynamic_Behavior_Manager.initializer import Initializer
from Luca_MSc.Dynamic_Behavior_Manager.DNF_interactors.robot_interactors import RobotInteractors

# Ensuring silent SMACH logs
def silent_log(msg):
    pass

smach.set_loggers(silent_log, silent_log, silent_log, silent_log)


class FindObjectSM(smach.State):
    """Find target object using perception"""

    def __init__(self, interactors: RobotInteractors, behavior_args: Dict[str, Any]):
        smach.State.__init__(self, outcomes=['success', 'failure', 'preempted'],
                             output_keys=['target_location', 'target_orientation'])
        self.interactors = interactors
        self.behavior_args = behavior_args


    def execute(self, userdata):
        target = self.behavior_args.get('target_object')
        if not target:
            return 'failure'

        result = self.interactors.perception.find_object(continuous_behavior='FindObject')

        if result[0]:
            userdata.target_location = result[2].clone()
            userdata.target_orientation = result[3].clone()
            return 'success'
        elif result[1]:
            return 'failure'
        else:
            return 'preempted'


class MoveToObjectSM(smach.State):
    """Move to target object with progress monitoring"""

    def __init__(self, interactors: RobotInteractors, behavior_args: Dict[str, Any]):
        smach.State.__init__(self, outcomes=['success', 'failure', 'preempted'],
                             input_keys=['target_location'])
        self.interactors = interactors
        self.behavior_args = behavior_args
        self.last_distance = None
        self.stagnant_ticks = 0
        self.max_stagnant_ticks = 50

    def execute(self, userdata):
        target_pos = userdata.target_location._obj


        if target_pos is None:
            return 'failure'

        current_pos = self.interactors.movement.get_position()
        result = self.interactors.movement.move_to(continuous_behavior='MoveToObject')

        distance = torch.norm(target_pos[:2] - current_pos[:2]).item()

        if distance < 0.1:
            self.last_distance = None
            self.stagnant_ticks = 0
            return 'success'

        if self.last_distance is not None:
            if abs(distance - self.last_distance) < 0.01:
                self.stagnant_ticks += 1
                if self.stagnant_ticks >= self.max_stagnant_ticks:
                    self.last_distance = None
                    self.stagnant_ticks = 0
                    return 'failure'
            else:
                self.stagnant_ticks = 0

        elif result[1]:
            return 'failure'

        self.last_distance = distance
        return 'preempted'


class CheckReachSM(smach.State):
    """Check if object is reachable"""

    def __init__(self, interactors: RobotInteractors, behavior_args: Dict[str, Any]):
        smach.State.__init__(self, outcomes=['success', 'failure', 'preempted'],
                             input_keys=['target_location'])
        self.interactors = interactors
        self.behavior_args = behavior_args

    def execute(self, userdata):
        target = self.behavior_args.get('target_object')
        target_pos = userdata.target_location._obj

        result = self.interactors.gripper.reach_check(continuous_behavior='CheckReach')

        if result[0]:
            return 'success'
        elif result[1]:
            return 'failure'
        else:
            return 'preempted'


class ReachForObjectSM(smach.State):
    """Reach for target object"""

    def __init__(self, interactors: RobotInteractors, behavior_args: Dict[str, Any]):
        smach.State.__init__(self, outcomes=['success', 'failure', 'preempted'],
                             input_keys=['target_location'])
        self.interactors = interactors
        self.behavior_args = behavior_args

    def execute(self, userdata):
        target = self.behavior_args.get('target_object')
        target_pos = userdata.target_location._obj

        self.interactors.gripper.reach_for(continuous_behavior='ReachFor')

        gripper_pos = self.interactors.gripper.get_position()
        distance = torch.norm(target_pos - gripper_pos).item()

        if distance < 0.01:
            return 'success'
        else:
            return 'preempted'


class GrabTransportSM(smach.State):
    """Grab and transport object to target"""

    def __init__(self, interactors: RobotInteractors, behavior_args: Dict[str, Any]):
        smach.State.__init__(self, outcomes=['success', 'failure', 'preempted'],
                             input_keys=['target_location', 'target_orientation'])
        self.interactors = interactors
        self.behavior_args = behavior_args
        self.phase = 'grab'

    def execute(self, userdata):
        target = self.behavior_args.get('target_object')
        drop_off = self.behavior_args.get('drop_off_target')

        # Unwrap SMACH Const wrappers
        target_pos = userdata.target_location._obj

        target_orientation = userdata.target_orientation._obj

        if self.phase == 'grab':
            result = self.interactors.gripper.grab(continuous_behavior='GrabTransport')

            if result[0]:
                self.phase = 'transport'
                self.interactors.state.update_behavior_target('move_to',
                                                              'transport_target',
                                                              'GrabTransport')
                return 'preempted'
            elif result[1]:
                self.phase = 'grab'
                return 'failure'
            else:
                return 'preempted'

        elif self.phase == 'transport':
            if drop_off not in self.interactors.perception.objects:
                self.phase = 'grab'
                return 'failure'

            drop_pos = self.interactors.perception.objects[drop_off]['location']
            current_pos = self.interactors.movement.get_position()

            self.interactors.movement.move_to(continuous_behavior='GrabTransport')

            distance = torch.norm(drop_pos[:2] - current_pos[:2]).item()

            # Query ground truth to prevent false positive success reporting
            gt_object_location = self.interactors.perception.objects[target]['location']
            gripper_position = self.interactors.gripper.get_position()
            gripper_state = self.interactors.gripper.has_object(gripper_position, gt_object_location)

            if not gripper_state:
                self.phase = 'grab'
                return 'failure'

            if distance < 0.1:
                self.phase = 'grab'
                return 'success'
            else:
                return 'preempted'

        self.phase = 'grab'
        return 'failure'


class StateMachineComparison:
    """SMACH-based state machine for benchmarking against DNF"""

    def __init__(self, interactors: RobotInteractors, behavior_args: Dict[str, Any], max_retries: int = 3):
        self.interactors = interactors
        self.behavior_args = behavior_args
        self.max_retries = max_retries
        self.sm = None
        self.retry_count = 0

    def build_state_machine(self):
        """Build the hierarchical state machine"""
        sm = smach.StateMachine(outcomes=['SUCCESS', 'FAILURE', 'TIMEOUT'])
        sm.userdata.target_location = None
        sm.userdata.target_orientation = None

        # Provide interactors with target information
        behaviors = ['find', 'move', 'check_reach', 'reach_for', 'grab']
        behavior_manager = BehaviorManager(behaviors, self.behavior_args)
        initializer = Initializer(behavior_manager)
        behavior_chain = initializer.build_behavior_chain(behaviors)
        self.interactors.state.initialize_from_behavior_chain(behavior_chain, self.behavior_args)

        with sm:
            # Create retry wrapper for robustness
            sm_retry = smach.StateMachine(outcomes=['success', 'failure', 'timeout'])
            sm_retry.userdata.target_location = None
            sm_retry.userdata.target_orientation = None

            with sm_retry:
                smach.StateMachine.add('FIND',
                                       FindObjectSM(self.interactors, self.behavior_args),
                                       transitions={'success': 'MOVE',
                                                    'failure': 'failure',
                                                    'preempted': 'FIND'})

                smach.StateMachine.add('MOVE',
                                       MoveToObjectSM(self.interactors, self.behavior_args),
                                       transitions={'success': 'CHECK_REACH',
                                                    'failure': 'failure',
                                                    'preempted': 'MOVE'})

                smach.StateMachine.add('CHECK_REACH',
                                       CheckReachSM(self.interactors, self.behavior_args),
                                       transitions={'success': 'REACH',
                                                    'failure': 'failure',
                                                    'preempted': 'CHECK_REACH'})

                smach.StateMachine.add('REACH',
                                       ReachForObjectSM(self.interactors, self.behavior_args),
                                       transitions={'success': 'GRAB',
                                                    'failure': 'failure',
                                                    'preempted': 'REACH'})

                smach.StateMachine.add('GRAB',
                                       GrabTransportSM(self.interactors, self.behavior_args),
                                       transitions={'success': 'success',
                                                    'failure': 'failure',
                                                    'preempted': 'GRAB'})

            # Add retry wrapper
            smach.StateMachine.add('MAIN_SEQUENCE',
                                   sm_retry,
                                   transitions={'success': 'SUCCESS',
                                                'failure': 'FAILURE',
                                                'timeout': 'TIMEOUT'})

        self.sm = sm

    def execute(self, max_steps: int = 4000, external_perturbation=None):
        """Execute the state machine with retry logic

        Args:
            max_steps: Maximum number of execution steps over all attempts
            external_perturbation: Optional function(step, interactors) -> None

        Returns:
            dict with execution metrics
        """
        total_start_time = time.time()
        total_steps = 0

        for attempt in range(self.max_retries):

            self.build_state_machine()

            start_time = time.time()
            step_counter = 0

            # Store original "_update_once" method that is used to step the state machine
            original_methods = {}

            def wrap_state_machine(sm):
                # Execute with step-based control for perturbations

                # Don't wrap multiple times
                if sm in original_methods:
                    return

                # Store original method
                original_methods[sm] = sm._update_once

                # Create wrapped version of "_update_once" for all nested state machines
                def wrapped_update_once():
                    nonlocal step_counter

                    if external_perturbation is not None:
                        external_perturbation(step_counter, self.interactors)

                    step_counter += 1

                    if step_counter >= max_steps:
                        sm._is_running = False
                        return 'timeout'

                    result = original_methods[sm]()

                    # Centralized timing control mimicking DNF 5ms step duration
                    time.sleep(0.005)

                    return result

                # Replace the original method with the wrapped version
                sm._update_once = wrapped_update_once

                # Recursively also wrap any nested state machines
                for state in sm._states.values():
                    if isinstance(state, smach.StateMachine):
                        wrap_state_machine(state)

            # Wrap all state machines in the hierarchy
            wrap_state_machine(self.sm)

            # Execute the state machine
            outcome = self.sm.execute()

            for sm, original_methods in original_methods.items():
                sm._update_once = original_methods

            attempt_steps = step_counter
            total_steps += attempt_steps

            #print(f"Attempt {attempt + 1} finished in {attempt_steps} steps and {time.time() - start_time:.2f} seconds with outcome: {outcome}")

            if outcome == 'SUCCESS':
                return {
                    'success': outcome == 'SUCCESS',
                    'steps': total_steps,
                    'time': time.time() - total_start_time,
                    'final_status': outcome,
                    'attempts': attempt + 1
                }

            if outcome is None and attempt >= max_steps:
                # Continue to next retry unless this is the last attempt
                if attempt < self.max_retries - 1:
                    continue

        print("All attempts exhausted without success.")

        # All attempts exhausted without success
        return {
            'success': False,
            'steps': total_steps,
            'time': time.time() - total_start_time,
            'final_status': outcome,
            'attempts': self.max_retries
        }


if __name__ == '__main__':
    # Example usage
    interactors = RobotInteractors()
    behavior_args = {
        'target_object': 'cup',
        'drop_off_target': 'transport_target'
    }

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

    sm_comparison = StateMachineComparison(interactors, behavior_args)
    result = sm_comparison.execute()

    print("Execution Result:", result)
