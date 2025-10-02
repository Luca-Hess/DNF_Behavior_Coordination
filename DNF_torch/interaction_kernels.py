import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

# ---------- kernel builder ----------

def stabilized_nd(kernel_size: Tuple[int, ...],
                  sigmas: Tuple[float, float],
                  heights: Tuple[float, float]) -> torch.Tensor:
    """
    N-D difference-of-Gaussians (Mexican hat) kernel of shape kernel_size.
    Returns an N-D tensor (no channel dims).
    """
    assert len(kernel_size) in (1, 2, 3), "Only 1D, 2D or 3D kernels are supported"
    assert len(sigmas) == 2 and len(heights) == 2, "sigmas/heights must be (exc, inh)"

    # coordinate grids centered at 0
    axes = [torch.arange(-(k // 2), k // 2 + 1, dtype=torch.float32) for k in kernel_size]
    mesh = torch.meshgrid(*axes, indexing='ij')
    r2 = torch.zeros_like(mesh[0])
    for c in mesh:
        r2 = r2 + c**2

    sigma_ex, sigma_in = sigmas
    h_ex, h_in = heights

    # norm_ex = 1.0 / (torch.sqrt(2 * pi) * sigma_ex)
    # norm_in = 1.0 / (torch.sqrt(2 * pi) * sigma_in)

    # (Simple scaling; exact N-D normalization constants are not required unless you need them)
    excit = h_ex * torch.exp(-r2 / (2 * sigma_ex**2))
    inhib = h_in * torch.exp(-r2 / (2 * sigma_in**2))
    return excit + inhib


# ---------- conv wrapper with internal padding ----------

class InteractionKernelND(nn.Module):
    """
    Generalized interaction kernel with internal padding for 1D/2D/3D.

    3D behavior:
      - If dimensions==3 and conv_groups==1: apply a 2D kernel to each depth slice (planar conv).
      - If dimensions==3 and conv_groups>1: use full 3D convolution.

    Padding is applied via F.pad to support padding_mode: 'zeros'|'reflect'|'replicate'|'circular'
    for all dimensionalities. Convolutions then use padding=0 (already padded).

    Shapes:
      input:
        1D: (B, C, W)
        2D: (B, C, H, W)
        3D: (B, C, D, H, W)
      output matches input spatial shape ("same" padding).
    """
    def __init__(self,
                 dimensions: int = 2,
                 conv_groups: int = 1,
                 kernel_size: Tuple[int, ...] = (5, 5),
                 in_channels: int = 1,
                 out_channels: int = 1,
                 stride: int = 1,
                 padding_mode: str = "zeros",  # 'zeros'|'reflect'|'replicate'|'circular'
                 sigma_exc: float = 2.0,
                 sigma_inh: float = 4.0,
                 height_exc: float = 0.4,
                 height_inh: float = -0.11,
                 kernel_type: str = "stabilized",
                 requires_grad: bool = False):
        super().__init__()

        assert dimensions in (1, 2, 3), "dimensions must be 1, 2 or 3"
        assert all(k % 2 == 1 for k in kernel_size), "odd kernel sizes required for exact 'same' padding"
        assert padding_mode in ("zeros", "reflect", "replicate", "circular")

        self.dimensions = dimensions
        self.conv_groups = conv_groups
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding_mode = padding_mode
        self.kernel_type = kernel_type

        # validate groups
        assert in_channels % conv_groups == 0, "in_channels must be divisible by conv_groups"
        assert out_channels % conv_groups == 0, "out_channels must be divisible by conv_groups"

        # build base conv (padding=0, we will pad manually)
        if dimensions == 1:
            self.conv = nn.Conv1d(in_channels, out_channels,
                                  kernel_size=kernel_size,
                                  stride=stride, padding=0, bias=False,
                                  groups=conv_groups)
        elif dimensions == 2:
            self.conv = nn.Conv2d(in_channels, out_channels,
                                  kernel_size=kernel_size,
                                  stride=stride, padding=0, bias=False,
                                  groups=conv_groups)
        else:  # dimensions == 3
            if conv_groups == 1:
                # planar conv: apply 2D conv per depth slice
                k2d = kernel_size[1:] if len(kernel_size) == 3 else kernel_size
                self.planar_kernel_size = k2d
                self.conv2d_planar = nn.Conv2d(in_channels, out_channels,
                                               kernel_size=k2d,
                                               stride=stride, padding=0, bias=False,
                                               groups=conv_groups)
                self.conv = None
            else:
                # full 3D conv
                self.conv = nn.Conv3d(in_channels, out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride, padding=0, bias=False,
                                      groups=conv_groups)
                self.conv2d_planar = None

        # make kernel tensor (N-D spatial only), then copy into conv weights
        if self.kernel_type == "stabilized":
            base_kernel = stabilized_nd(
                kernel_size=kernel_size,
                sigmas=(sigma_exc, sigma_inh),
                heights=(height_exc, height_inh)
            )
        else:
            raise ValueError(f"Unsupported kernel_type: {kernel_type}")

        # register kernel parameters (optional learnable scalars)
        self.sigma_exc = nn.Parameter(torch.tensor(sigma_exc, dtype=torch.float32), requires_grad=requires_grad)
        self.sigma_inh = nn.Parameter(torch.tensor(sigma_inh, dtype=torch.float32), requires_grad=requires_grad)
        self.height_exc = nn.Parameter(torch.tensor(height_exc, dtype=torch.float32), requires_grad=requires_grad)
        self.height_inh = nn.Parameter(torch.tensor(height_inh, dtype=torch.float32), requires_grad=requires_grad)

        # assign weights with correct shape per conv variant
        with torch.no_grad():
            if dimensions == 1:
                # weight: (out_ch, in_ch/groups, K)
                w = base_kernel.view(1, 1, kernel_size[0]).repeat(
                    out_channels, in_channels // conv_groups, 1
                )
                self.conv.weight.copy_(w)
                self.conv.weight.requires_grad = False
            elif dimensions == 2:
                # weight: (out_ch, in_ch/groups, Kh, Kw)
                w = base_kernel.view(1, 1, kernel_size[0], kernel_size[1]).repeat(
                    out_channels, in_channels // conv_groups, 1, 1
                )
                self.conv.weight.copy_(w)
                self.conv.weight.requires_grad = False
            else:  # 3D
                if conv_groups == 1:
                    # planar 2D weight for conv2d_planar
                    Kh, Kw = self.planar_kernel_size
                    # base_kernel is (Kd, Kh, Kw); we need the 2D slice kernel applied at each depth.
                    # Use the *same* 2D kernel across depth slices by taking the center slice
                    # or the marginal over D. Here we take the center slice:
                    kd = kernel_size[0] // 2
                    kernel_2d = base_kernel[kd, :, :]  # (Kh, Kw)
                    w = kernel_2d.view(1, 1, Kh, Kw).repeat(
                        out_channels, in_channels // conv_groups, 1, 1
                    )
                    self.conv2d_planar.weight.copy_(w)
                    self.conv2d_planar.weight.requires_grad = False
                else:
                    # full 3D weight: (out_ch, in_ch/groups, Kd, Kh, Kw)
                    w = base_kernel.view(1, 1, kernel_size[0], kernel_size[1], kernel_size[2]).repeat(
                        out_channels, in_channels // conv_groups, 1, 1, 1
                    )
                    self.conv.weight.copy_(w)
                    self.conv.weight.requires_grad = False

        # precompute “same” paddings per dimension for F.pad
        self._pad_1d = (kernel_size[0] // 2,) if dimensions == 1 else None
        self._pad_2d = (kernel_size[1] // 2, kernel_size[1] // 2,
                        kernel_size[0] // 2, kernel_size[0] // 2) if dimensions == 2 else None
        if dimensions == 3:
            kd, kh, kw = kernel_size
            # F.pad expects pads as (W_left, W_right, H_left, H_right, D_left, D_right)
            self._pad_3d = (kw // 2, kw // 2, kh // 2, kh // 2, kd // 2, kd // 2)
            # planar 2D padding (H, W) only for each slice:
            self._pad_2d_planar = (kw // 2, kw // 2, kh // 2, kh // 2)

    # ---- helpers ----

    def _pad_1d_fn(self, x):
        if self.padding_mode == "zeros":
            # Conv1d supports padding='zeros' internally, but we unify via F.pad for consistency
            return F.pad(x, (self._pad_1d[0], self._pad_1d[0]), mode='constant', value=0.0)
        return F.pad(x, (self._pad_1d[0], self._pad_1d[0]), mode=self.padding_mode)

    def _pad_2d_fn(self, x, pads):
        if self.padding_mode == "zeros":
            return F.pad(x, pads, mode='constant', value=0.0)
        return F.pad(x, pads, mode=self.padding_mode)

    def _pad_3d_fn(self, x):
        if self.padding_mode == "zeros":
            return F.pad(x, self._pad_3d, mode='constant', value=0.0)
        return F.pad(x, self._pad_3d, mode=self.padding_mode)

    # ---- forward ----

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dimensions == 1:
            # x: (B, C, W)
            x = self._pad_1d_fn(x)
            return self.conv(x)  # padding=0 inside conv

        elif self.dimensions == 2:
            # x: (B, C, H, W)
            x = self._pad_2d_fn(x, self._pad_2d)
            return self.conv(x)

        else:  # 3D
            # x: (B, C, D, H, W)
            if self.conv_groups == 1:
                # planar: pad over H,W per slice, then 2D conv per slice
                B, C, D, H, W = x.shape
                # reshape to (B*D, C, H, W)
                x2 = x.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)
                x2 = self._pad_2d_fn(x2, self._pad_2d_planar)
                y2 = self.conv2d_planar(x2)  # (B*D, Cout, H, W)
                # reshape back to (B, Cout, D, H, W)
                y = y2.reshape(B, D, self.out_channels, H, W).permute(0, 2, 1, 3, 4)
                return y
            else:
                # full 3D: pad D,H,W, then Conv3d
                x = self._pad_3d_fn(x)
                return self.conv(x)

    # optional: rebuild weights after changing scalar params
    @torch.no_grad()
    def freeze_kernel(self):
        base_kernel = stabilized_nd(
            kernel_size=self.kernel_size,
            sigmas=(float(self.sigma_exc), float(self.sigma_inh)),
            heights=(float(self.height_exc), float(self.height_inh))
        ).to(next(self.parameters()).device)

        if self.dimensions == 1:
            w = base_kernel.view(1, 1, self.kernel_size[0]).repeat(
                self.out_channels, self.in_channels // self.conv_groups, 1
            )
            self.conv.weight.copy_(w)
        elif self.dimensions == 2:
            w = base_kernel.view(1, 1, self.kernel_size[0], self.kernel_size[1]).repeat(
                self.out_channels, self.in_channels // self.conv_groups, 1, 1
            )
            self.conv.weight.copy_(w)
        else:
            if self.conv_groups == 1:
                kd = self.kernel_size[0] // 2
                kernel_2d = base_kernel[kd, :, :]
                Kh, Kw = kernel_2d.shape
                w = kernel_2d.view(1, 1, Kh, Kw).repeat(
                    self.out_channels, self.in_channels, 1, 1
                )
                self.conv2d_planar.weight.copy_(w)
            else:
                w = base_kernel.view(1, 1, *self.kernel_size).repeat(
                    self.out_channels, self.in_channels // self.conv_groups, 1, 1, 1
                )
                self.conv.weight.copy_(w)
