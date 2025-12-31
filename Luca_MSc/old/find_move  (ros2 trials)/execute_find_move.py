import sys, os
sys.path.append(os.path.expanduser('~/nc_ws/DNF_torch/Luca_MSc'))
sys.path.append(os.path.expanduser('~/nc_ws/DNF_torch/Luca_MSc/find_move'))

import time
import multiprocessing
from find_move import FindMoveBehavior
from ros_interactors import RobotInteractors

def run_find_move_behavior(external_input_value=5.0):
    """Run the find-move behavior with ROS interactors"""
    
    try:
        # Create ROS-based interactors
        interactors = RobotInteractors()
        
        # Create find-move behavior
        find_move = FindMoveBehavior()
        
        print("Starting find-move behavior system...")
        print(f"External input: {external_input_value}")
        
        # Behavior execution loop
        rate = 200  # 200Hz
        dt = 1.0 / rate
        step_count = 0
        
        while True:
            start_time = time.time()
            
            # Execute one behavior step
            state = find_move.execute_step(
                interactors,
                target_name="cricket_ball",
                external_input=external_input_value
            )
            
            # Print status every 100 steps (0.5 seconds)
            if step_count % 100 == 0:
                find_active = state['find']['intention_activity'] > 0.5
                found = state['preconditions']['found']['active']
                move_active = state['move'] is not None
                
                print(f"Step {step_count}: Find={find_active}, Found={found}, Move={move_active}")
                
                if state['find']['target_location'] is not None:
                    loc = state['find']['target_location']
                    print(f"  Target location: [{loc[0]:.2f}, {loc[1]:.2f}, {loc[2]:.2f}]")
            
            step_count += 1
            
            # Maintain timing
            elapsed = time.time() - start_time
            if elapsed < dt:
                time.sleep(dt - elapsed)
                
    except KeyboardInterrupt:
        print("Shutting down find-move behavior...")
    finally:
        # Clean shutdown
        if 'interactors' in locals():
            interactors.shutdown()

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run DNF Find-Move Behavior')
    parser.add_argument('--external-input', type=float, default=5.0,
                       help='External input value to activate behavior')
    
    args = parser.parse_args()
    
    run_find_move_behavior(args.external_input)