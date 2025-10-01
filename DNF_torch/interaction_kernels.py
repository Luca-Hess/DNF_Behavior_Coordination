import torch
import torch.nn as nn
from typing import Tuple
import numpy as np

import torch
import numpy as np


def stabilized_nd(kernel_size: Tuple[int, ...],
                  sigmas: Tuple[float, ...],
                  heights: Tuple[float, float], conv_groups: int = 1) -> torch.Tensor:
    """
    Creates an N-dimensional Mexican hat (difference of Gaussians) kernel.
    
    Args:
        kernel_size (tuple of int): Size of the kernel in each dimension.
        sigmas (tuple): (sigma_excitatory, sigma_inhibitory)
        heights (tuple): (height_excitatory, height_inhibitory)
    
    Returns:
        torch.Tensor: N-dimensional interaction kernel.
    """
    assert len(kernel_size) in (2, 3), "Only 2D or 3D kernels are supported"
    assert len(sigmas) == 2 and len(heights) == 2, "Sigmas and heights must be (ex, in)"
    
    dims = len(kernel_size)
    ax = [torch.arange(-(k // 2), k // 2 + 1, dtype=torch.float32) for k in kernel_size]
    mesh = torch.meshgrid(*ax, indexing='ij')
    r_squared = sum(coord ** 2 for coord in mesh)

    sigma_ex, sigma_in = sigmas
    height_ex, height_in = heights

    pi = torch.tensor(torch.pi)  # or just use `torch.pi` directly if on 1.8+

    norm_ex = 1.0 / (torch.sqrt(2 * pi) * sigma_ex)
    norm_in = 1.0 / (torch.sqrt(2 * pi) * sigma_in)

    excitatory = height_ex * norm_ex * torch.exp(-r_squared / (2 * sigma_ex ** 2))
    inhibitory = height_in * norm_in * torch.exp(-r_squared / (2 * sigma_in ** 2))


    return excitatory + inhibitory


class InteractionKernelND(nn.Module):
    """
    Generalized interaction kernel wrapper for 2D or 3D convolution.
    """
    def __init__(self,
                 dimensions: int = 2,
                 conv_groups: int = 1,
                 kernel_size: Tuple[int, ...] = (5, 5),
                 in_channels: int = 1,
                 out_channels: int = 1,
                 stride: int = 1,
                 padding: int = 2,
                 # sigmas: Tuple[float, float] = (2.0, 4.0),
                 sigma_exc: float = 2.0,
                 sigma_inh: float = 4.0,
                 height_exc: float = 0.4,
                 height_inh: float = -0.11,
                 kernel_type: str = "stabilized",
                 requires_grad: bool = False,
                 # heights: Tuple[float, float] = (0.4, -0.11)
                 ):
        super().__init__()
        """
        Initialize an InteractionKernelND module.
        
        Args:
            dimensions (int): Number of spatial dimensions (2 or 3).
            kernel_size (tuple): Kernel size per dimension.
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            stride (int): Convolution stride.
            padding (int): Convolution padding.
            sigma_ex (float): Excitatory Gaussian sigma.
            sigma_in (float): Inhibitory Gaussian sigma.
            height_exc (float): Excitatory Gaussian height.
            height_in (float): Inhibitory Gaussian height.
            kernel_type (str): Type of kernel ("stabilized" or "von_mises").
        """
        super().__init__()
        assert dimensions in (2, 3), "Only 2D or 3D supported"

        # register sigmas and heights as parameters to be learned
        self.sigma_exc = torch.nn.Parameter(torch.tensor(sigma_exc), requires_grad=requires_grad)
        self.sigma_inh = torch.nn.Parameter(torch.tensor(sigma_inh), requires_grad=requires_grad)
        self.height_exc = torch.nn.Parameter(torch.tensor(height_exc), requires_grad=requires_grad)
        self.height_inh = torch.nn.Parameter(torch.tensor(height_inh), requires_grad=requires_grad)
        self.kernel_size = kernel_size
        self.padding = padding
        self.kernel_type = kernel_type

        # Create kernel
        if self.kernel_type == "stabilized":
            kernel = stabilized_nd(
                kernel_size=kernel_size,
                conv_groups=conv_groups,
                sigmas=(self.sigma_exc, self.sigma_inh),
                heights=(self.height_exc, self.height_inh))
            # Choose Conv layer
            Conv = nn.Conv2d # if kernel_size == 2 else nn.Conv3d
            # Init convolution layer
            self.conv = Conv(
                in_channels, out_channels,
                kernel_size=kernel_size,
                stride=stride, padding=padding, bias=False, groups=conv_groups
            )
            # Set weights and freeze them
            self.conv.weight.data = kernel.view(1, 1, kernel_size[0], kernel_size[1]).repeat(out_channels, in_channels // conv_groups, 1, 1)
            # self.conv.weight.data = kernel.unsqueeze(0).unsqueeze(0)  # [1, 1, ...]
            self.conv.weight.requires_grad = False

     
        else:
            raise ValueError(f"Unsupported kernel_type: {kernel_type}")
        


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # if self.training:
        #     # Rebuild kernel dynamically during training
        #     kernel = stabilized_nd(
        #         kernel_size=self.kernel_size,
        #         sigmas=(self.sigma_exc, self.sigma_inh),
        #         heights=(self.height_exc, self.height_inh)
        #     ).to(x.device)
            
        #     kernel = kernel.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        #     return F.conv2d(x, kernel, padding=self.padding)
        
        # else:
            # Use static conv weight in inference
        return self.conv(x)
    
    def freeze_kernel(self):
        kernel = stabilized_nd(
            kernel_size=self.kernel_size,
            sigmas=(self.sigma_exc, self.sigma_inh),
            heights=(self.height_exc, self.height_inh)
        ).to(self.conv.weight.device)

        with torch.no_grad():
            self.conv.weight.copy_(kernel.unsqueeze(0).unsqueeze(0))
            
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)
