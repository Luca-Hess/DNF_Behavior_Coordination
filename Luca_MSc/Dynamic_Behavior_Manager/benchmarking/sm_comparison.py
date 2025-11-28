import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import smach
import torch
import time
from typing import Dict, Any
from interactors import RobotInteractors


# Ensuring silent logs
def silent_log(msg):
    """Completely silence SMACH logs"""
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

        result = self.interactors.perception.find_object(target)

        if result[0]:
            userdata.target_location = result[2].clone()
            userdata.target_orientation = result[3].clone()
            return 'success'
        elif result[1]:
            return 'failure'
        else:
            time.sleep(0.005)
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
        target_pos_temp = userdata.target_location

        # Unwrapping tensor from Const
        target_pos = torch.tensor([pos.item() for pos in target_pos_temp])

        if target_pos is None:
            return 'failure'

        current_pos = self.interactors.movement.get_position()
        self.interactors.movement.move_to(target_pos, requesting_behavior='MoveToObject')


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

        self.last_distance = distance
        time.sleep(0.005)
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
        target_pos_temp = userdata.target_location
        target_pos = torch.tensor([pos.item() for pos in target_pos_temp])

        result = self.interactors.gripper.reach_check(target, target_pos, None, requesting_behavior='CheckReach')

        if result[0]:
            return 'success'
        elif result[1]:
            return 'failure'
        else:
            time.sleep(0.005)
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
        target_pos_temp = userdata.target_location
        target_pos = torch.tensor([pos.item() for pos in target_pos_temp])

        self.interactors.gripper.reach_for(target, target_pos, None, requesting_behavior='ReachFor')

        gripper_pos = self.interactors.gripper.get_position()
        distance = torch.norm(target_pos - gripper_pos).item()

        if distance < 0.01:
            return 'success'
        else:
            time.sleep(0.005)
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
        target_pos_temp = userdata.target_location
        target_pos = torch.tensor([pos.item() for pos in target_pos_temp])

        target_or_temp = userdata.target_orientation
        target_orientation = torch.tensor([orientation.item() for orientation in target_or_temp])

        if self.phase == 'grab':
            result = self.interactors.gripper.grab(target,
                                                   target_pos,
                                                   target_orientation,
                                                   requesting_behavior='GrabTransport')

            if result[0]:
                self.phase = 'transport'
                time.sleep(0.005)
                return 'preempted'
            elif result[1]:
                self.phase = 'grab'
                return 'failure'
            else:
                time.sleep(0.005)
                return 'preempted'

        elif self.phase == 'transport':
            if drop_off not in self.interactors.perception.objects:
                self.phase = 'grab'
                return 'failure'

            drop_pos = self.interactors.perception.objects[drop_off]['location']
            current_pos = self.interactors.movement.get_position()

            self.interactors.movement.move_to(drop_pos, requesting_behavior='GrabTransport')

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
                time.sleep(0.005)
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

    def build_state_machine(self):
        """Build the hierarchical state machine"""
        sm = smach.StateMachine(outcomes=['SUCCESS', 'FAILURE'])
        sm.userdata.target_location = None
        sm.userdata.target_orientation = None

        with sm:
            # Create retry wrapper for robustness
            sm_retry = smach.StateMachine(outcomes=['success', 'failure'])
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
                                                    'failure': 'FIND',
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
                                                    'failure': 'FIND',
                                                    'preempted': 'GRAB'})

            # Add retry wrapper
            smach.StateMachine.add('RETRY_WRAPPER',
                                   sm_retry,
                                   transitions={'success': 'SUCCESS',
                                                'failure': 'FAILURE'})

        self.sm = sm

    def execute(self, max_steps: int = 4000, external_perturbation=None):
        """Execute the state machine

        Args:
            max_steps: Maximum number of execution steps
            external_perturbation: Optional function(step, interactors) -> None

        Returns:
            dict with execution metrics
        """
        self.build_state_machine()

        start_time = time.time()
        step = 0

        # Execute with step-based control for perturbations
        while step < max_steps:
            if external_perturbation is not None:
                external_perturbation(step, self.interactors)

            outcome = self.sm.execute()
            step += 1

            if outcome in ['SUCCESS', 'FAILURE']:
                end_time = time.time()
                return {
                    'success': outcome == 'SUCCESS',
                    'steps': step,
                    'time': end_time - start_time,
                    'final_status': outcome
                }

            time.sleep(0.005)

        end_time = time.time()
        return {
            'success': False,
            'steps': max_steps,
            'time': end_time - start_time,
            'final_status': 'TIMEOUT'
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
