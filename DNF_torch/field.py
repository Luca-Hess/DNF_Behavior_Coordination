import torch
from .interaction_kernels import InteractionKernelND
#from .vis_utils import plot_kernel
import torch.nn as nn

def dimensions_from_shape(shape):
    if isinstance(shape, int):
        return 1 if shape >= 1 else 0
    elif isinstance(shape, tuple):
        return len(shape) if len(shape) > 0 else 0
    else:
        raise TypeError("shape must be int or tuple")
    

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
                 kernel_size=5, # only simmetric kernels for now
                 interaction=True,
                 in_channels=1,
                 out_channels=1,
                 conv_groups=1,  # if >1, use depthwise conv
                 kernel_sigma_exc=2.0,
                 kernel_sigma_inh=7.0,
                 kernel_height_exc=4.0,
                 kernel_height_inh=-8.0,
                 self_connection_w0=1, # self connection weight for 0D field     
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
        assert isinstance(kernel_size, int), "kernel_size must be int (symmetric)"

        self._dimensions = dimensions_from_shape(shape)
        self.interaction = interaction

        if self._dimensions == 0:
            ksize = ()
            print("Warning: 0D field, no spatial interactions.")
        else:
            ksize = (kernel_size,) * self._dimensions
        
        # ----- spatial shape bookkeeping (and fix undefined H/W earlier)
        if self._dimensions == 0:
            spatial_shape = ()
        elif self._dimensions == 1:
            W = shape if isinstance(shape, int) else shape[0]
            spatial_shape = (W,)
        elif self._dimensions == 2:
            H, W = shape
            spatial_shape = (H, W)
        elif self._dimensions == 3:
            # choose an order and stick to it; here (D,H,W)
            D, H, W = shape
            spatial_shape = (D, H, W)
        else:
            raise ValueError("4D not supported yet.")
        
        self.shape = spatial_shape
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = conv_groups

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

        if self._dimensions > 0:
            self.kernel = InteractionKernelND(
                dimensions=self._dimensions,
                conv_groups=conv_groups,
                kernel_size=ksize,
                sigma_exc=kernel_sigma_exc,
                sigma_inh=kernel_sigma_inh,
                height_exc=kernel_height_exc,
                height_inh=kernel_height_inh,
                kernel_type=self.kernel_type,
                in_channels=in_channels,
                out_channels=out_channels,
            )
        else:
            self.kernel = None
            self.self_w = self_connection_w0
        
        self.register_buffer("time_step", torch.tensor(time_step, dtype=torch.float32))
        act_shape = (1, in_channels, *spatial_shape) if self._dimensions > 0 else (1,)
        self.register_buffer("activation", torch.full(act_shape, self.resting_level, dtype=torch.float32))
        # firing rate g(u) initialized as sigmoid(beta * u)
        g0 = torch.sigmoid(self.beta * self.activation)
        self.register_buffer("g_u", g0.clone())
        self.register_buffer("noise_tensor", torch.empty_like(self.activation))
        # precompute constants
        dt = self.time_step
        tau = torch.tensor(self.time_scale, dtype=torch.float32)
        self.register_buffer("dt_over_tau", dt / tau)
        self.register_buffer("sqrt_dt_over_tau", torch.tensor(self.noise_strength, dtype=torch.float32) * torch.sqrt(dt / tau))

        if self.debug and self.kernel is not None:
            plot_kernel(self.kernel, self.shape)

        self.activation_history = []
        self.activity_history = []


    def _global_inhibition_term(self):
        if self.global_inhibition == 0.0:
            return 0.0
        n = self.g_u[0, :].numel() if self._dimensions > 0 else 1
        return self.global_inhibition * torch.sum(self.g_u) / n

    def forward(self, input_tensor=None):
        # interaction
        if self._dimensions == 0:
            interaction_term = self.self_w * self.g_u
        elif self.interaction and self.kernel is not None:
            # InteractionKernelND should already apply the chosen padding_mode
            interaction_term = self.kernel_scale * self.kernel(self.g_u)
        else:
            interaction_term = 0.0

        # subtract global inhibition (broadcast-safe)
        gi = self._global_inhibition_term()
        if isinstance(interaction_term, torch.Tensor):
            interaction_term = interaction_term - gi
        else:
            interaction_term = -gi

        # input & noise
        input_term = input_tensor if input_tensor is not None else 0.0
        noise = self.sqrt_dt_over_tau * torch.randn_like(self.activation)

        # Euler update
        rate_change = (-self.activation + interaction_term + self.scale * input_term + self.resting_level)
        self.activation = self.activation + self.dt_over_tau * rate_change + noise
        self.g_u = torch.sigmoid(self.beta * self.activation)

        if self.debug:
            self.activation_history.append(self.activation.detach().cpu().squeeze().numpy())
            self.activity_history.append(self.g_u.detach().cpu().squeeze().numpy())

        return self.activation, self.g_u
    
        
    def get_activation(self):
        return self.g_u

    def get_history(self):
        return self.activation_history, self.activity_history

    @torch.no_grad()
    def reset(self):
        self.activation.fill_(self.resting_level)
        self.g_u.copy_(torch.sigmoid(self.beta * self.activation))
        self.noise_tensor.resize_as_(self.activation)
        self.activation_history.clear()
        self.activity_history.clear()

    @property
    def dimensions(self):
        return self._dimensions