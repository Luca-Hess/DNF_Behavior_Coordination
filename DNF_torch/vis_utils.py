import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

def plot_kernel(kernel, dimensions=2):
    """
    Auto-detects whether kernel.conv is Conv2d or Conv3d.
    - Conv2d:
        * If depthwise (groups=in_ch=out_ch>1): plot one subplot per channel (e.g., 4 for von Mises)
        * Else: plot the first [0,0] kernel
    - Conv3d: show orthogonal slices and an optional voxel view
    The `dimensions` arg is kept for backward compatibility but not trusted.
    """
    W = kernel.conv.weight.detach().cpu()

    # --- Detect conv type from the module itself ---
    if isinstance(kernel.conv, nn.Conv3d):
        # Conv3d weight: (out_c, in_c, kD, kH, kW)
        ker3d = W[0, 0].numpy()          # (D, H, W)
        D, H, W_ = ker3d.shape

        # Central orthogonal slices
        zc, yc, xc = D // 2, H // 2, W_ // 2
        slice_z = ker3d[zc, :, :]
        slice_y = ker3d[:, yc, :]
        slice_x = ker3d[:, :, xc]

        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        im0 = axs[0].imshow(slice_z, cmap='hot', origin='lower'); axs[0].set_title(f'Z slice @ {zc}')
        plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)
        im1 = axs[1].imshow(slice_y, cmap='hot', origin='lower'); axs[1].set_title(f'Y slice @ {yc}')
        plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)
        im2 = axs[2].imshow(slice_x, cmap='hot', origin='lower'); axs[2].set_title(f'X slice @ {xc}')
        plt.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)
        fig.suptitle('Interaction Kernel (3D) – orthogonal slices')
        plt.tight_layout(); plt.show()

        # Optional voxel view
        vmin, vmax = ker3d.min(), ker3d.max()
        eps = 1e-12
        vol01 = (ker3d - vmin) / (vmax - vmin + eps)
        thr = vol01.mean() + 0.5 * vol01.std()
        filled = vol01 >= thr
        if filled.any():
            fig = plt.figure(figsize=(7, 6))
            ax = fig.add_subplot(111, projection='3d')
            cmap = plt.get_cmap('hot')
            facecolors = cmap(vol01)
            facecolors[~filled] = [0, 0, 0, 0]
            ax.voxels(filled, facecolors=facecolors, edgecolor='k', linewidth=0.1)
            ax.set_title('Interaction Kernel (3D) – voxels')
            ax.set_xlabel('W'); ax.set_ylabel('H'); ax.set_zlabel('D')
            plt.tight_layout(); plt.show()
        return

    # --- Conv2d path ---
    # Conv2d weight: (out_c, in_c/groups, kH, kW)
    is_depthwise = (
        isinstance(kernel.conv, nn.Conv2d)
        and kernel.conv.groups > 1
        and kernel.conv.groups == kernel.conv.in_channels == kernel.conv.out_channels
    )

    if is_depthwise:
        # von Mises depthwise case: W shape (O, 1, kH, kW)
        O = W.shape[0]
        rows = int(np.floor(np.sqrt(O)))
        cols = int(np.ceil(O / rows))
        fig, axs = plt.subplots(rows, cols, figsize=(3.2*cols, 3.2*rows))
        axs = np.atleast_1d(axs).ravel()
        for i in range(O):
            ker2d = W[i, 0].numpy()
            im = axs[i].imshow(ker2d, cmap='hot', interpolation='nearest')
            axs[i].set_title(f'Orientation {i}')
            axs[i].axis('off')
            plt.colorbar(im, ax=axs[i], fraction=0.046, pad=0.04)
        for j in range(O, len(axs)):
            axs[j].axis('off')
        fig.suptitle('Interaction Kernel (2D – depthwise)')
        plt.tight_layout(); plt.show()
    else:
        # Single kernel (e.g., stabilized Conv2d)
        ker2d = W[0, 0].numpy()
        plt.imshow(ker2d, cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.title('Interaction Kernel (2D)')
        plt.show()




def plot_color_dimension_activations(activation, active_values, title_prefix="Field"):
    """
    Plot activations and active values for all color dimensions in two separate figures.

    Args:
        activation: torch.Tensor of shape (1, 1, H, W, D) where D = number of color dims
        active_values: same shape as activation
        title_prefix: prefix string for subplot titles
    """
    def _plot_single(tensor, fig_title):
        act = tensor.squeeze(0).squeeze(0)   # (H, W, D)
        H, W, D = act.shape
        assert D == 10, f"Expected 10 color dims, got {D}"

        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.flatten()

        for i in range(D):
            ax = axes[i]
            im = ax.imshow(act[:, :, i].cpu().numpy(), cmap='viridis', origin='lower')
            ax.set_title(f"{fig_title} dim {i}")
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        fig.suptitle(fig_title)
        plt.tight_layout()
        plt.show()

    # First figure: activation maps
    _plot_single(activation, f"{title_prefix} Activations")

    # Second figure: active values maps
    _plot_single(active_values, f"{title_prefix} Active Values")




    import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def _to_CHW(x):
    """Return a numpy array shaped (C,H,W) from x which may be torch or numpy."""
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    if x.ndim == 2:      # (H,W) -> (1,H,W)
        x = x[None, ...]
    assert x.ndim == 3, f"Expected (C,H,W) or (H,W); got shape {x.shape}"
    return x

def animate_tensor(seq, interval=60, cmap="cividis", titles=None):
    """
    seq: list/tuple of length T of arrays/tensors shaped (C,H,W) or (H,W)
         OR a torch.Tensor/np.ndarray shaped (T,C,H,W) or (T,H,W)
    interval: milliseconds between frames
    cmap: matplotlib colormap name
    titles: optional list of channel titles (len == C)
    """
    # Normalize input to a list of frames
    if torch.is_tensor(seq) or isinstance(seq, np.ndarray):
        if seq.ndim == 4:   # (T,C,H,W)
            frames = [seq[t] for t in range(seq.shape[0])]
        elif seq.ndim == 3: # (T,H,W) -> treat as single-channel
            frames = [seq[t][None, ...] for t in range(seq.shape[0])]
        else:
            raise ValueError(f"Unsupported array shape {seq.shape}")
    else:
        frames = list(seq)

    T = len(frames)
    first = _to_CHW(frames[0])
    C, H, W = first.shape

    # Precompute fixed vmin/vmax per channel across time (stable colors)
    vmin = np.full(C, np.inf, dtype=np.float32)
    vmax = np.full(C, -np.inf, dtype=np.float32)
    for f in frames:
        a = _to_CHW(f).reshape(C, -1)
        vmin = np.minimum(vmin, a.min(axis=1))
        vmax = np.maximum(vmax, a.max(axis=1))
    # Avoid zero range
    for c in range(C):
        if np.isclose(vmax[c], vmin[c]):
            vmax[c] = vmin[c] + 1e-6

    # Build figure/axes
    if C == 1:
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        ims = [ax.imshow(first[0], vmin=vmin[0], vmax=vmax[0], cmap=cmap, animated=True)]
        ax.set_axis_off()
        if titles:
            ax.set_title(titles[0])
    elif C == 4:
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        axs = axs.ravel()
        ims = []
        for c in range(4):
            im = axs[c].imshow(first[c], vmin=vmin[c], vmax=vmax[c], cmap=cmap, animated=True)
            axs[c].set_axis_off()
            axs[c].set_title(titles[c] if titles and c < len(titles) else f"Channel {c}")
            ims.append(im)
    else:
        # Generic fallback: one row with C subplots
        fig, axs = plt.subplots(1, C, figsize=(4*C, 4))
        if C == 1:
            axs = [axs]
        ims = []
        for c in range(C):
            im = axs[c].imshow(first[c], vmin=vmin[c], vmax=vmax[c], cmap=cmap, animated=True)
            axs[c].set_axis_off()
            axs[c].set_title(titles[c] if titles and c < len(titles) else f"Channel {c}")
            ims.append(im)

    supt = fig.suptitle("t = 0 / {}".format(T-1))

    def update(t):
        a = _to_CHW(frames[t])
        for c in range(C):
            ims[c].set_data(a[c])
        supt.set_text(f"t = {t} / {T-1}")
        # Return artists for blitting compatibility
        return ims + [supt]

    ani = FuncAnimation(fig, update, frames=T, interval=interval, blit=False, repeat=True)
    plt.tight_layout()
    plt.show()
    return ani
