# **DNF Behavior Coordination Project**

Using Dynamic Neural Fields implemented in PyTorch.  
Cloned from: **github.com:neuromorphic-zhaw/DNF_torch.git**

---

## **Repository Structure**

### **1. `Luca_MSc/`**

#### **Dynamic_Behavior_Manager**
**Path:** `Luca_MSc/Dynamic_Behavior_Manager`

- Contains the **Dynamic Behavior Manager** implementation.  
- Main executable: **`behavior_manager.py`**  
- Additional scripts at `Luca_MSc/old` are **deprecated** and included only for completeness and documentation of the development process.

---

### **2. `Stats_Analysis/`**

- Contains the **R workspace** and scripts used for statistical analysis of experimental results.  
- Includes data processing, visualization, and statistical testing relevant to the MSc thesis.

---

## **Project Overview**

This project uses **Dynamic Neural Fields (DNFs)** implemented in PyTorch and integrates them into a behavior coordination framework.  
The repository supports:
- Behavior arbitration  
- Experimental evaluation via benchmarking 
- Visualization of simulations

---

## **Installation & Setup**

### Clone repository
```bash

git clone https://github.com/Luca-Hess/DNF_Behavior_Coordination.git
cd DNF_Behavior_Coordination

```

### Install uv (if not installed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
````

### Create and sync the virtual environment

```bash
uv sync
```

### Activate the environment

```bash
source .venv/bin/activate   # Linux/macOS
```

---

## **Running the Behavior Manager**

```bash
python3 Luca_MSc/Dynamic_Behavior_Manager/behavior_manager.py
```

#### Potential Issues:

On some systems, running the behavior manager with visualizations might fail due to issues with matplotlib interactivity. If this should be the case, change 

```text
plt.show()
```
to

```text
plt.savefig(<'your figure title'>)
```

in the relevant files (likely helper_functions.py).

---

### **Dependencies**

This project targets **Python ≥ 3.12** and uses the following core dependencies (managed via `pyproject.toml` and `uv.lock`):

#### **Python Requirements**

| Package | Version / Constraint |
|--------|-----------------------|
| `matplotlib` | `3.10.1` |
| `numpy` | `1.26.4` |
| `py-trees` | `>=2.4.0` |
| `pygame` | `2.6.1` |
| `pygame-ce` | `2.5.5` |
| `pygame-gui` | `0.6.5` |
| `pyyaml` | `>=6.0.3` |
| `standalone-smach` | `>=0.0.9` |
| `torch` | `2.6.0` |
| `torchaudio` | `2.6.0` |
| `torchmetrics` | `1.7.4` |
| `torchvision` | `0.21.0` |

---


## **Citation**

If you use this code, please cite:

```
Luca Hess, “Dynamic Neural Field-Based Behavior Coordination,”
MSc Thesis, Zurich University of Applied Sciences, 2026.

```

---

### **High-level architecture diagram**

<img width="940" height="588" alt="image" src="https://github.com/user-attachments/assets/9df0b37f-13f5-4888-8d70-ae9d103c8007" />

Demonstration implementation of a simple behavior chain, showcasing the three system layers, the elementary behavior modules and the sanity check nodes.



### **Known Issues**

-	Parameter sensitivity: General system performance is influenced by parameters chosen. CoS switching in particular depends heavily on weight settings and timing.

-	Lack of ROS2 integration: This prevented realistic benchmarking of communication overhead and concurrency, as well as real-life implementation. 

- Logic error which allows extended actions to trigger only once, instead of anytime a corresponding elementary behavior becomes active again. 


### **Possible Future Work**

•	ROS2 integration: This will enable realistic communication patterns, concurrency and sensor handling, meaning interactors can be moved beyond their current dummy state.

•	Integrating higher-level control: The DNF behavior control system does not currently have an outward facing interface for human interaction, nor does it include higher-level planning and control. Using modern AI methods for these applications should be investigated.

•	Behavior specific sanity-check frequencies: Some behaviors naturally require more sanity checks than others. For example, obstacle avoidance behaviors need more than one sensor poll per second. One way to address this issue could be to include sanity check frequency as a parameter in the behavior configuration file. On this topic, it might also be interesting to explore different ways of including obstacle avoidance and other basic reactive behaviors into the behavior chain. Perhaps an Avoid behavior could always implicitly be included in the chain, or as part of the base interactor class. 

•	Expand extended behaviors: Currently, behaviors can only be extended to execute special triggered actions once they have succeeded. However, adding explicit actions on failure is a sensible addition. For example, Find could be extended with Search, meaning that a robot would actively explore its environment to look for an object if it did not find it, rather than reporting failure of the behavior sequence right away.

•	Improved system CoS logic: If possible, extended behaviors should not be in danger of prematurely triggering system CoS reporting.

•	Parameter tuning: It might be possible to improve performance and find a balance between stability and speed. Particularly the DNF time scale parameter could speed up the simulations. This could potentially include reinforcement learning to find the optimal setup. 

•	More complex connection architecture: Should the need arise; it might be necessary to add more complex connection types than a simple linear execution as the system features now.



