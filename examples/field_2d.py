from DNF_torch.field import Field
from DNF_torch.viz import animate_fields
from DNF_torch.viz import run_tuner

viz = False
tune = True

# Run simulation
field_1 = Field(resting_level=-5.0, noise_strength=0.01, global_inhibition=0.0,
                kernel_size=15, kernel_sigma_exc=2.0, kernel_sigma_inh=7.0,
                kernel_height_exc=4.0, kernel_height_inh=-8.0,
                interaction=True, beta=1.0, scale=1.0, kernel_scale=1.0,
                kernel_type='stabilized', debug=False)
field_2 = Field(resting_level=-0.0, noise_strength=0.01, global_inhibition=0.0,
                kernel_size=15, kernel_sigma_exc=1.0, kernel_sigma_inh=3.0,
                kernel_height_exc=3.0, kernel_height_inh=-6.0,
                interaction=True, beta=1.0, scale=1.0, kernel_scale=1.0,
                kernel_type='stabilized', debug=False)
fields = [field_1, field_2]

if viz:
    print("Starting visualization.")
    animate_fields(fields)
elif tune:
    print("Starting parameter tuner.")
    run_tuner(fields)
else:
    print("Running simulation without visualization.")
    for _step in range(100):
        activation, active_vals = field_1.update()
        activation, active_vals = field_2.update()
        print(f"Step {_step}: Activation mean {activation.mean().item():.4f}, Active mean {active_vals.mean().item():.4f}")
