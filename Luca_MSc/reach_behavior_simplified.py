import torch
import numpy as np
import matplotlib.pyplot as plt
from elementary_behavior_simplified import ElementaryBehaviorSimple



class ReachBehaviorSimple(ElementaryBehaviorSimple):
    """
    Example behavior: Reach to a target location using node dynamics.
    """

    def __init__(self):
        # Create with 0D fields (nodes)
        super().__init__()

    def execute(self, target_position, current_position, obstacles=None):
        """
        Execute one step of the reach behavior using node inputs.

        Args:
            target_position: The target position to reach
            current_position: Current position of the robot
            obstacles: Optional information about obstacles

        Returns:
            dict: Contains behavior state and motor commands
        """
        # Calculate scalar inputs for control nodes:
        distance_to_target = torch.norm(target_position - current_position)

        # Intention: Activate when target is different from current position
        intention_gain = 50.0
        intention_offset = -0.2 # small offset to avoid noise triggering when close to target
        intention_input = intention_gain * (distance_to_target + intention_offset)
        intention_input = max(0.0, float(intention_input)) if intention_input < 6 else 6 # No negative input, max val

        # CoS: Activate when close enough to target
        cos_gain = 100.0
        cos_threshold = 0.1 # Distance for satisfaction
        cos_input = cos_gain * max(0.0, cos_threshold - distance_to_target)

        # Process behavior control
        state = self.forward(intention_input, cos_input)

        # Generate motor commands if active
        motor_commands = None
        if state['active']:
            # Simple proportional controller
            direction = target_position - current_position
            direction = torch.clamp(direction, -10.0, 10.0)
            motor_gain = 0.5
            motor_commands = motor_gain * direction

        return {**state, 'motor_commands': motor_commands}


# Example simulation
if __name__ == "__main__":
    behavior = ReachBehaviorSimple()

    # Simulate target and current positions
    #target = torch.tensor([100.0, 50.0])
    target = torch.tensor([1000.0, 100.0])
    current = torch.tensor([0.0, 0.0])

    log = {
        "intention_activation": [],
        "intention_activity": [],
        "cos_activation": [],
        "cos_activity": []
    }

    # Run behavior for several steps
    i = 0
    done = 0
    for step in range(1000):
        # Update current position based on previous commands
        if step > 0 and result['motor_commands'] is not None:
            current += result['motor_commands']



        # Execute behavior
        result = behavior.execute(target, current)

        # Print status
        print(f"Step {step}: Active={bool(result['active'])}, "
              f"Completed={bool(result['completed'])}, "
              f"Position={current.tolist()}, "
              f"Intention={result['intention_activation']:.2f},{float(result['intention_activity']):.2f}, "
              f"CoS={result['cos_activation']:.2f},{float(result['cos_activity']):.2f}")

        # Store logs
        log["intention_activation"].append(result['intention_activation'])
        log["intention_activity"].append(result['intention_activity'])
        log["cos_activation"].append(result['cos_activation'])
        log["cos_activity"].append(result['cos_activity'])

        i += 1

        # Stop if behavior completed or failed
        if result['completed']:
            done += 1

        if done >= 10:
            break





    # Plotting the activities of both nodes over time
    ts = np.arange(i)
    plt.figure(figsize=(8,4))
    plt.plot(ts, log["intention_activation"], label="Intention (activation)")
    plt.plot(ts, log["cos_activation"], label="CoS (activation)")
    plt.plot(ts, log["intention_activity"], '--', label="Intention (activity)")
    plt.plot(ts, log["cos_activity"], '--', label="CoS (activity)")
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.title("Two 0D DNF Nodes: 1 excites 2, 2 inhibits 1")
    plt.legend()
    plt.tight_layout()
    plt.show()