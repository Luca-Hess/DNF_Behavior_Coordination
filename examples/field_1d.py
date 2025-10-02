from DNF_torch.field import Field
from DNF_torch.viz import run_tuner


tune = True

# Run simulation
field_1 = Field(shape=50, resting_level=-5.0, noise_strength=0.01, global_inhibition=0.0,
                kernel_size=15, kernel_sigma_exc=2.0, kernel_sigma_inh=7.0,
                kernel_height_exc=4.0, kernel_height_inh=-8.0,
                interaction=True, beta=1.0, scale=1.0, kernel_scale=1.0,
                kernel_type='stabilized', debug=False)

fields = [field_1]

if tune:
    print("Starting parameter tuner.")
    run_tuner(fields)
else:
    print("Running simulation without visualization.")
    for _step in range(100):
        activation, active_vals = field_1.forward()
        print(f"Step {_step}: Activation mean {activation.mean().item():.4f}, Active mean {active_vals.mean().item():.4f}")
