import torch
import torch.nn as nn
import torch.nn.functional as F


def stabilized_2d(kernel_size=5, sigma_ex=2.0, sigma_in=4.0,
                  height_exc=0.4, height_in=-0.11) -> torch.Tensor:
    """
    Creates a 2D Mexican hat kernel (difference of Gaussians).
    
    Args:
        kernel_size (int): Size of the square kernel (must be odd).
        sigma_ex (float): Excitatory Gaussian std deviation.
        sigma_in (float): Inhibitory Gaussian std deviation.
        height_exc (float): Excitatory Gaussian height.
        height_in (float): Inhibitory Gaussian height.
    
    Returns:
        torch.Tensor: 2D interaction kernel.
    """
    ax = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, dtype=torch.float32)
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    r_squared = xx**2 + yy**2

    excitatory = height_exc * torch.exp(-r_squared / (2 * sigma_ex**2))
    inhibitory = height_in * torch.exp(-r_squared / (2 * sigma_in**2))

    return excitatory - inhibitory


class InteractionKernel(nn.Module):
    """
    Wrapper class for creating and applying 2D convolutional kernels for dynamic neural fields.
    """
    def __init__(self, 
                 kernel_type: str = "stabilized",
                 kernel_size: int = 5,
                 in_channels: int = 1,
                 out_channels: int = 1,
                 stride: int = 1,
                 padding: int = 2,
                 requires_grad: bool = False,
                 **kwargs):
        super().__init__() 

        """
        Initialize an InteractionKernel module.
        
        Args:
            kernel_type (str): Type of kernel. Currently supports "stabilized".
            kernel_size (int): Size of the convolution kernel.
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            stride (int): Stride for convolution.
            padding (int): Padding for convolution.
            **kwargs: Parameters passed to the kernel function.
        """
        # Create the kernel tensor
        if kernel_type == "stabilized":
            kernel = stabilized_2d(kernel_size=kernel_size, **kwargs)
        else:
            raise ValueError(f"Unsupported kernel type '{kernel_type}'")

        # Initialize the Conv2D layer
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=False)

        # Set weights and freeze
        self.conv.weight.data = kernel.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, H, W]
        self.conv.weight.requires_grad = requires_grad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the interaction kernel as a convolution to input x.

        Args:
            x (torch.Tensor): Input tensor of shape [B, C, H, W]
        
        Returns:
            torch.Tensor: Convolved output tensor
        """
        return self.conv(x)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


import torch
import torch.nn as nn
from typing import Tuple
import numpy as np

import torch
import numpy as np

@torch.no_grad()
def create_von_mises_torch(kH: int, kW: int,
                           r0: float, theta0: float,
                           filter_type: str = 'NORMAL',
                           w: float = 1.0, w2: float = 1.0,
                           eps: float = 1e-8) -> torch.Tensor:
    """
    Torch version of the provided create_von_mises (no SciPy needed).

    Args:
        kH, kW: kernel height/width
        r0: preferred radius
        theta0: preferred angle (radians)
        filter_type: 'NORMAL' | 'CENTERED' | 'OPPOSITE'
        w, w2: scalars used exactly as in your formula
        eps: numerical epsilon to avoid division by zero

    Returns:
        (kH, kW) tensor
    """
    # Coordinate grid (match your numpy version: Y axis is flipped)
    x = torch.arange(kW, dtype=torch.float32) - (kW / 2.0)
    y = -(torch.arange(kH, dtype=torch.float32) - (kH / 2.0))
    Y, X = torch.meshgrid(y, x, indexing='ij')  # (kH, kW)

    # Shift based on filter_type
    if filter_type == 'CENTERED':
        X = X + r0 * torch.cos(torch.tensor(theta0))
        Y = Y + r0 * torch.sin(torch.tensor(theta0))
    elif filter_type == 'OPPOSITE':
        X = X + 2.0 * r0 * torch.cos(torch.tensor(theta0))
        Y = Y + 2.0 * r0 * torch.sin(torch.tensor(theta0))
    # else: NORMAL → no shift

    # Polar coords
    r = torch.sqrt(X**2 + Y**2)
    theta = torch.atan2(Y, X)

    # von Mises per your formula:
    # exp(r0 * w * cos(theta - theta0)) / I0(w2 * (r - r0))
    num = torch.exp(r0 * w * torch.cos(theta - torch.tensor(theta0)))
    # modified Bessel I0 in torch:
    denom = torch.special.i0(w2 * (r - r0))

    vm = num / (denom + eps)
    # avoid NaNs/Infs if any
    vm = torch.nan_to_num(vm, nan=0.0, posinf=0.0, neginf=0.0)
    return vm


@torch.no_grad()
def von_mises_kernels_2d(num_orient=4,
                         kernel_size=15,
                         orientations=(0, np.pi/4, np.pi/2, 3*np.pi/4),
                         r0: float = 5.0,
                         w: float = 1.0,
                         w2: float = 1.0,
                         filter_type: str = 'NORMAL',
                         normalize: bool = True,
                         zero_mean: bool = True) -> torch.Tensor:
    """
    Build a bank of von Mises kernels using create_von_mises_torch.

    Returns:
        weight tensor shaped (O, 1, kH, kW) for depthwise Conv2d.
    """
    if isinstance(kernel_size, int):
        kH = kW = kernel_size
    else:
        kH, kW = kernel_size

    kernels = []
    for theta0 in orientations[:num_orient]:
        k = create_von_mises_torch(kH, kW, r0=r0, theta0=float(theta0),
                                   filter_type=filter_type, w=w, w2=w2)
        if normalize:
            s = torch.sum(torch.abs(k))
            if s > 1e-8:
                k = k / s
        if zero_mean:
            k = k - k.mean()
        kernels.append(k)

    W = torch.stack(kernels, dim=0).unsqueeze(1)  # (O, 1, kH, kW)
    return W


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

        elif kernel_type == "von_mises":
            num_orient = in_channels
            ksz = kernel_size if isinstance(kernel_size, int) else kernel_size[-1]
            kernels = von_mises_kernels_2d(num_orient=num_orient, kernel_size=ksz, filter_type='NORMAL')

            # Conv3d with groups = num_orient
            self.conv = nn.Conv2d(in_channels=num_orient, 
                                out_channels=num_orient,
                                kernel_size=kernel_size,
                                groups=num_orient,
                                padding=padding,
                                bias=False)
            with torch.no_grad():
                self.conv.weight.copy_(kernels)
            self.conv.weight.requires_grad = requires_grad

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
