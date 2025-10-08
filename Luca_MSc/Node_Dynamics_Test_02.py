from DNF_torch.field import Field
from DNF_torch.viz import run_tuner

import torch
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(0)


tune = False

# Run simulation
node_1 = Field(shape=(), resting_level=-5.0, noise_strength=0.01, global_inhibition=0.0,
                interaction=True, beta=5.0, scale=1.0, kernel_scale=1.0,
                kernel_type='stabilized', debug=False, self_connection_w0=1.0,
                time_step=5.0, time_scale=100.0,)

node_2 = Field(shape=(), resting_level=-5.0, noise_strength=0.01, global_inhibition=0.0,
                interaction=True, beta=5.0, scale=1.0, kernel_scale=1.0,
                kernel_type='stabilized', debug=False, self_connection_w0=1.0,
                time_step=5.0, time_scale=100.0,)


# Node 1 is constantly being excited at a fixed rate
constant_excitation = 6

# Defining relationships between nodes
excitation_weight = 6
inhibition_weight = -6

fields = [node_1, node_2]

node1_activities = []
node2_activities = []

node_1.connection_to(node_2, weight=excitation_weight)
node_2.connection_to(node_1, weight=inhibition_weight)

# alternative approach: node_1.connection_from(node_2, weight=inhibition_weight)

# --- Simulation loop ---
log = {
    "u1": [],
    "u2": [],
    "g1": [],
    "g2": []
}

if tune:
    print("Starting parameter tuner.")
    run_tuner(fields)
else:
    print("Running simulation without visualization.")
    for _step in range(1000):
        # storing current activations for synchronous update
        for n in (node_1, node_2):
            n.cache_prev()

        u1, g1 = node_1.forward(input_tensor=constant_excitation)
        u2, g2 = node_2.forward()

        # Store logs
        log["u1"].append(u1.item())
        log["u2"].append(u2.item())
        log["g1"].append(g1.item())
        log["g2"].append(g2.item())

        print(f"Step {_step}: Node1 {float(g1):.3f}, Node2 {float(g2):.3f}")


# Plotting the activities of both nodes over time
ts = np.arange(1000)
plt.figure(figsize=(8,4))
plt.plot(ts, log["u1"], label="u1 (activation)")
plt.plot(ts, log["u2"], label="u2 (activation)")
plt.plot(ts, log["g1"], '--', label="g1 (rate)")
plt.plot(ts, log["g2"], '--', label="g2 (rate)")
plt.xlabel("Step")
plt.ylabel("Value")
plt.title("Two 0D DNF Nodes: 1 excites 2, 2 inhibits 1")
plt.legend()
plt.tight_layout()
plt.show()
