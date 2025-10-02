from DNF_torch.field import Field
from DNF_torch.viz import animate_fields
from DNF_torch.viz import run_tuner

viz = False
tune = False

# Run simulation
field_1 = Field(shape=(), resting_level=-5.0, noise_strength=0.01, global_inhibition=0.0,
                kernel_size=1, kernel_sigma_exc=1.0, kernel_sigma_inh=1.0,
                kernel_height_exc=1.0, kernel_height_inh=-1.0,
                interaction=True, beta=1.0, scale=1.0, kernel_scale=1.0,
                kernel_type='stabilized', debug=False)

fields = [field_1]

if viz:
    print("Starting visualization.")
    animate_fields(fields)
elif tune:
    print("Starting parameter tuner.")
    run_tuner(fields)
else:
    print("Running simulation without visualization.")
    for _step in range(100):
        activation, active_vals = field_1.forward()
        print(f"Step {_step}: Activation mean {activation.mean().item():.4f}, Active mean {active_vals.mean().item():.4f}")
