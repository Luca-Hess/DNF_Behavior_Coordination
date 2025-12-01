import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import csv
import random

from functools import partial
from typing import Dict, Any

from bt_comparison import BehaviorTreeComparison
from sm_comparison import StateMachineComparison

from interactors import RobotInteractors
from behavior_manager import run_behavior_manager

# Benchmark functions
def benchmark_completion_speed(num_runs: int = 10) -> Dict[str, Any]:
    """Compare completion speed between BT and DNF systems"""

    bt_times = []
    bt_steps = []
    bt_success = 0
    sm_times = []
    sm_steps = []
    sm_success = 0
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

        bt_system = BehaviorTreeComparison(bt_interactors, behavior_args, max_retries=3)

        bt_result = bt_system.execute()
        bt_times.append(bt_result['time'])
        bt_steps.append(bt_result['steps'])
        if bt_result['success']:
            bt_success += 1
        print(f"  BT: {bt_result['steps']} steps, "
              f"{bt_result['time']:.3f}s, "
              f"{'Succeeded' if bt_result['success'] else 'Failed'}")

        # Test State Machine
        sm_interactors = RobotInteractors()
        sm_interactors.perception.register_object(
            name="cup",
            location=torch.tensor([5.2, 10.5, 1.8]),
            angle=torch.tensor([0.0, -1.0, 0.0])
        )
        sm_interactors.perception.register_object(
            name="transport_target",
            location=torch.tensor([5.0, 0.0, 1.0]),
            angle=torch.tensor([0.0, 0.0, 0.0])
        )

        sm_system = StateMachineComparison(sm_interactors, behavior_args, max_retries=3)
        sm_result = sm_system.execute()
        sm_times.append(sm_result['time'])
        sm_steps.append(sm_result['steps'])
        if sm_result['success']:
            sm_success += 1
        print(f"  SM: {sm_result['steps']} steps, "
              f"{sm_result['time']:.3f}s, "
              f"{'Succeeded' if sm_result['success'] else 'Failed'}")

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
                             visualize_logs=False,
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
        'sm_avg_time': sum(sm_times) / len(sm_times),
        'sm_avg_steps': sum(sm_steps) / len(sm_steps),
        'sm_times': sm_times,
        'sm_steps': sm_steps,
        'sm_success': sm_success,
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

        bt_times = []
        bt_steps = []
        bt_success = 0
        sm_times = []
        sm_steps = []
        sm_success = 0
        dnf_times = []
        dnf_steps = []
        dnf_success = 0

        for run in range(num_runs):

            # Determine random trigger step for perturbation and pass it to the function
            trigger_step = random.randint(200, 1200)

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

            bt_test_system = BehaviorTreeComparison(bt_interactors, behavior_args, max_retries=3)

            bt_result = bt_test_system.execute(external_perturbation=perturbation_with_trigger)

            bt_times.append(bt_result['time'])
            bt_steps.append(bt_result['steps'])
            if bt_result['success']:
                bt_success += 1

            print(f"  Run {run + 1}: BT {bt_result['final_status']}")

            # Test SM
            sm_interactors = RobotInteractors()  # Create separate instance
            sm_interactors.perception.register_object(
                name="cup",
                location=torch.tensor([5.2, 10.5, 1.8]),
                angle=torch.tensor([0.0, -1.0, 0.0])
            )
            sm_interactors.perception.register_object(
                name="transport_target",
                location=torch.tensor([5.0, 0.0, 1.0]),
                angle=torch.tensor([0.0, 0.0, 0.0])
            )

            sm_test_system = StateMachineComparison(sm_interactors, behavior_args, max_retries=3)
            sm_result = sm_test_system.execute(external_perturbation=perturbation_with_trigger)

            sm_times.append(sm_result['time'])
            sm_steps.append(sm_result['steps'])
            if sm_result['success']:
                sm_success += 1

            print(f"  Run {run + 1}: SM {sm_result['final_status']}")

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

            dnf_times.append(dnf_results['time'])
            dnf_steps.append(dnf_results['steps'])
            if dnf_results['success']:
                dnf_success += 1

            print(f"  Run {run + 1}: DNF {dnf_results['final_status']}")

        results[perturbation_name] = {
            'bt_avg_time': sum(bt_times) / len(bt_times),
            'bt_avg_steps': sum(bt_steps) / len(bt_steps),
            'bt_success_rate': bt_success / num_runs,
            'bt_times': bt_times,
            'bt_steps': bt_steps,
            'bt_success': bt_success,
            'sm_avg_time': sum(sm_times) / len(sm_times),
            'sm_avg_steps': sum(sm_steps) / len(sm_steps),
            'sm_success_rate': sm_success / num_runs,
            'sm_times': sm_times,
            'sm_steps': sm_steps,
            'sm_success': sm_success,
            'dnf_avg_time': sum(dnf_times) / len(dnf_times),
            'dnf_avg_steps': sum(dnf_steps) / len(dnf_steps),
            'dnf_success_rate': dnf_success / num_runs,
            'dnf_times': dnf_times,
            'dnf_steps': dnf_steps,
            'dnf_success': dnf_success,
        }

    return results


# Example perturbation functions
def object_displacement_perturbation(step, interactors, trigger_step):
    """Move target object at random step"""

    if step == trigger_step:
        target = interactors.perception.objects['cup']
        old_location = target['location'].clone() if target else None
        if target:
            # Use random triggerstep as source of deterministic random values between -2 and 2
            eps_1 = ((trigger_step ** 2 / 123.456) - int(trigger_step ** 2 / 123.456)) * 4 - 2
            eps_2 = ((trigger_step ** 3 / 789.123) - int(trigger_step ** 3 / 789.123)) * 4 - 2

            # Perturbation during carrying drops the object from gripper
            if interactors.gripper.grabbed_objects is not None:
                interactors.gripper.grabbed_objects.clear()

            target['location'] += torch.tensor([eps_1, eps_2, 0.0])
            print(f"[PERTURBATION] Moved object from {old_location} to {target['location']}")



def sensor_noise_perturbation(step, interactors, trigger_step):
    """Add random noise to perception"""
    if trigger_step == step:
        for obj_name, obj_data in interactors.perception.objects.items():
            print(obj_data)
            noise = torch.randn(3) * 0.5
            obj_data['location'] += noise
            print(obj_data)


import logging

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.getLogger('smach').setLevel(logging.WARNING)  # Reduce SMACH verbosity
    logging.getLogger('my_smach_logger').setLevel(logging.INFO)  # Your custom logger

    # Run benchmarks
    perturbations = [
        ("Object Displacement", object_displacement_perturbation),
        #("Sensor Noise", sensor_noise_perturbation),
    ]

    #speed_results = benchmark_completion_speed(num_runs=10)
    robustness_results = benchmark_robustness(perturbations, num_runs=5)

    # Store results in CSV file
    with open('bt_dnf_benchmark_results.csv', mode='w', newline='') as file:
        writer = csv.writer(file)

        # Write speed results
        if 'speed_results' in locals():
            writer.writerow(['Speed Benchmark - Time and Steps to Completion',
                             'BT Avg Time (s)', 'BT Avg Steps',
                             'SM Avg Time (s)', 'SM Avg Steps',
                             'DNF Avg Time (s)', 'DNF Avg Steps'])
            writer.writerow(['Completion Speed',
                             f"{speed_results['bt_avg_time']:.3f}",
                             speed_results['bt_avg_steps'],
                             f"{speed_results['sm_avg_time']:.3f}",
                             speed_results['sm_avg_steps'],
                             f"{speed_results['dnf_avg_time']:.3f}",
                             speed_results['dnf_avg_steps']])

            writer.writerow('Individual Run Times and Steps')
            writer.writerow(['Run',
                             'BT Time (s)', 'BT Steps', 'BT Successes',
                             'SM Time (s)', 'SM Steps', 'SM Successes',
                             'DNF Time (s)', 'DNF Steps', 'DNF Successes'])
            for i in range(len(speed_results['bt_times'])):
                writer.writerow([i + 1,
                                 f"{speed_results['bt_times'][i]:.3f}",
                                 speed_results['bt_steps'][i],
                                 1 if i < speed_results['bt_success'] else 0,
                                 f"{speed_results['sm_times'][i]:.3f}",
                                 speed_results['sm_steps'][i],
                                 1 if i < speed_results['sm_success'] else 0,
                                 f"{speed_results['dnf_times'][i]:.3f}",
                                 speed_results['dnf_steps'][i],
                                 1 if i < speed_results['dnf_success'] else 0])

        # Write robustness results
        if 'robustness_results' in locals():
            writer.writerow([])
            writer.writerow(['Perturbation',
                             'BT Avg Time (s)', 'BT Avg Steps', 'BT Success Rate',
                             'SM Avg Time (s)', 'SM Avg Steps', 'SM Success Rate',
                             'DNF Avg Time (s)', 'DNF Avg Steps', 'DNF Success Rate'])
            for perturbation_name, result in robustness_results.items():
                writer.writerow([perturbation_name,
                                    f"{result['bt_avg_time']:.2f}",
                                    f"{result['bt_avg_steps']}",
                                    f"{result['bt_success_rate']:.2f}",
                                    f"{result['bt_success']:.2f}",
                                    f"{result['sm_avg_time']:.2f}",
                                    f"{result['sm_avg_steps']}",
                                    f"{result['sm_success_rate']:.2f}",
                                    f"{result['sm_success']:.2f}",
                                    f"{result['dnf_avg_time']:.2f}",
                                    f"{result['dnf_avg_steps']:.2f}",
                                    f"{result['dnf_success_rate']:.2f}",
                                    f"{result['dnf_success']:.2f}"])

            writer.writerow(['Individual Run Times and Steps'])
            writer.writerow(['Perturbation', 'Run', 'Successes',
                             'BT Time (s)', 'BT Steps', 'BT Successes',
                             'SM Time (s)', 'SM Steps', 'SM Successes',
                             'DNF Time (s)', 'DNF Steps', 'DNF Successes'])

            for perturbation_name, result in robustness_results.items():
                num_runs = len(result['bt_times'])
                for i in range(num_runs):
                    writer.writerow([perturbation_name,
                                     i + 1,
                                     1 if i < result['bt_success'] else 0,
                                     f"{result['bt_times'][i]:.2f}",
                                     result['bt_steps'][i],
                                     1 if i < result['bt_success'] else 0,
                                     f"{result['sm_times'][i]:.2f}",
                                     result['sm_steps'][i],
                                     1 if i < result['sm_success'] else 0,
                                     f"{result['dnf_times'][i]:.2f}",
                                     result['dnf_steps'][i],
                                     1 if i < result['dnf_success'] else 0])
