import torch
from dnf_module.dnf_module.dnf_torch.interaction_kernels import InteractionKernel, InteractionKernelND
from dnf_module.dnf_module.dnf_torch.vis_utils import plot_kernel
import torch.nn as nn


class Field(nn.Module):
    """
    A class that simulates a 2D dynamic neural field using Euler integration.
    """

    def __init__(self,
                 shape=(50, 50),
                 time_step=5.0,
                 time_scale=100.0,
                 resting_level=-5.0,
                 noise_strength=0.01,
                 global_inhibition=0.0,
                 kernel_size=5,
                 in_channels=1,
                 out_channels=1,
                 conv_groups=1,
                 kernel_sigma_exc=2.0,
                 kernel_sigma_inh=7.0,
                 kernel_height_exc=4.0,
                 kernel_height_inh=-8.0,
                 interaction=True, 
                 beta=1.0,
                 scale=1.0,
                 kernel_scale=1.0,
                 kernel_type='stabilized',
                 debug=False):
        super().__init__()
        """
        Initialize the dynamic neural field.

        Args:
            shape (tuple): The spatial size of the field (H, W).
            time_step (float): Simulation time step in ms.
            time_scale (float): Time constant τ in ms.
            resting_level (float): Initial activation baseline.
            noise_strength (float): Amplitude of Gaussian noise.
            global_inhibition (float): Strength of global inhibition.
            kernel_params (dict): Parameters for interaction kernel.
        """
        H, W = shape
        assert isinstance(kernel_size, int) or (len(kernel_size) == 2)
        if isinstance(kernel_size, int):
            kH = kW = kernel_size
        else:
            kH, kW = kernel_size
        padding = (kH // 2, kW // 2)

        self.HW = (H, W)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = conv_groups

        self.interaction = interaction

        self.time_scale = time_scale
        self.resting_level = resting_level
        self.noise_strength = noise_strength
        self.global_inhibition = global_inhibition
        self.kernel_size = kernel_size
        self.beta = beta
        self.debug = debug
        self.scale = scale
        self.kernel_scale = kernel_scale
        self.kernel_type = kernel_type


        self.kernel = InteractionKernelND(
            dimensions=2,
            conv_groups=conv_groups,
            kernel_size=kernel_size,
            padding=padding,
            sigma_exc=kernel_sigma_exc,
            sigma_inh=kernel_sigma_inh,
            height_exc=kernel_height_exc,
            height_inh=kernel_height_inh,
            kernel_type=self.kernel_type,
            in_channels=in_channels,
            out_channels=out_channels,
        )
        
        # tensors that should live on the same device as the model
        self.register_buffer("time_step", torch.tensor(time_step, dtype=torch.float32))
        self.register_buffer("activation", torch.full((1, in_channels, H, W), resting_level, dtype=torch.float32))
        self.register_buffer("g_u",        torch.full((1, in_channels, H, W), resting_level, dtype=torch.float32))
        self.register_buffer("noise_tensor", torch.empty_like(self.activation))

        # precompute constants on-device
        self.register_buffer("dt_over_tau",  self.time_step / self.time_scale)
        self.register_buffer("sqrt_dt_over_tau", self.noise_strength * torch.sqrt(self.time_step / self.time_scale))


        # plot the interaction kernel
        if self.debug:
            plot_kernel(self.kernel, shape)
           
        # Keep history of activations
        self.activation_history = []
        self.activity_history = []

    def update(self, input_tensor=None):
        """
        Perform one Euler integration step.
        """
        # dt_over_tau = self.time_step / self.time_scale
        # sqrt_dt_over_tau = self.noise_strength * torch.sqrt(self.time_step / self.time_scale)

        if self.interaction:
            interaction_term = (self.kernel_scale * self.kernel(self.g_u))
            interaction_term -= (torch.sum(self.g_u) * self.global_inhibition) # / (self.shape[0] * self.shape[1])
        else:
            interaction_term = 0.0

        input_term = input_tensor if input_tensor is not None else 0.0
        noise = self.sqrt_dt_over_tau * torch.randn_like(self.activation)
        # import pdb; pdb.set_trace()
        rate_change = (-self.activation + interaction_term + self.scale * input_term + self.resting_level)
        self.activation = self.activation + self.dt_over_tau * rate_change + noise
        self.g_u = torch.sigmoid(self.beta * self.activation)
   
        if self.debug:
            self.activation_history.append(self.activation.detach().cpu().squeeze().numpy())
            self.activity_history.append(self.g_u.detach().cpu().squeeze().numpy())

        return self.activation, self.g_u
    

    def forward(self, input_tensor=None):
        """
        Perform one Euler integration step.
        """
        # dt_over_tau = self.time_step / self.time_scale
        # sqrt_dt_over_tau = self.noise_strength * torch.sqrt(self.time_step / self.time_scale)

        if self.interaction:
            interaction_term = (self.kernel_scale * self.kernel(self.g_u))
            interaction_term -= (torch.sum(self.g_u) * self.global_inhibition) # / (self.shape[0] * self.shape[1])
        else:
            interaction_term = 0.0

        input_term = input_tensor if input_tensor is not None else 0.0
        noise = self.sqrt_dt_over_tau * torch.randn_like(self.activation)
        # import pdb; pdb.set_trace()
        rate_change = (-self.activation + interaction_term + self.scale * input_term + self.resting_level)
        self.activation = self.activation + self.dt_over_tau * rate_change + noise
        self.g_u = torch.sigmoid(self.beta * self.activation)
   
        if self.debug:
            self.activation_history.append(self.activation.detach().cpu().squeeze().numpy())
            self.activity_history.append(self.g_u.detach().cpu().squeeze().numpy())

        return self.activation, self.g_u
    
        
    def get_activation(self):
        """
        Return the current field activation tensor.
        """
        return self.g_u

    def get_history(self):
        """
        Return the full history of activations as a list of NumPy arrays.
        """
        return self.activation_history, self.activity_history

    @torch.no_grad()
    def reset(self):
        B, C, H, W = self.activation.shape
        self.activation.fill_(self.resting_level)
        self.g_u.fill_(self.resting_level)
        self.noise_tensor.resize_as_(self.activation)
        self.activation_history.clear()
        self.activity_history.clear()