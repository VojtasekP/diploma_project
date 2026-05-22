"""10×3 grid: real image, disk mask, Gaussian heatmap for 10 frames."""
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))

from dataset.real_ccp import CCPCenterDataset

DATA_DIR     = HERE.parent / "data" / "CME_tracking_validation"
N_FRAMES     = 10
FRAME_STRIDE = 12          # evenly sample 10 frames out of 120
DISK_RADIUS  = 3.0
GAUSS_SIGMA  = 2
CLUSTER_THR  = 5.0
OUT          = HERE / "mask_examples.png"


frames = list(range(0, N_FRAMES * FRAME_STRIDE, FRAME_STRIDE))

ds_disk = CCPCenterDataset(DATA_DIR, frames=frames, patch_size=256,
                           disk_radius=DISK_RADIUS,
                           cluster_threshold=CLUSTER_THR)
ds_gauss = CCPCenterDataset(DATA_DIR, frames=frames, patch_size=256,
                            gaussian_sigma=GAUSS_SIGMA,
                            cluster_threshold=CLUSTER_THR)

fig, axes = plt.subplots(N_FRAMES, 3, figsize=(9, 3 * N_FRAMES))
axes[0, 0].set_title("Image",                            fontsize=11)
axes[0, 1].set_title(f"Disk mask (r = {DISK_RADIUS})",    fontsize=11)
axes[0, 2].set_title(f"Gaussian heatmap (σ = {GAUSS_SIGMA})", fontsize=11)

for row, frame in enumerate(frames):
    img,   _      = ds_disk[row]
    _,     disk   = ds_disk[row]
    _,     gauss  = ds_gauss[row]

    axes[row, 0].imshow(img[0],   cmap="gray",  interpolation="nearest")
    axes[row, 1].imshow(disk[0],  cmap="Reds",  interpolation="nearest", vmin=0, vmax=1)
    axes[row, 2].imshow(gauss[0], cmap="magma", interpolation="nearest", vmin=0, vmax=1)

    axes[row, 0].set_ylabel(f"frame {frame}", fontsize=9)
    for ax in axes[row]:
        ax.set_xticks([]); ax.set_yticks([])

fig.tight_layout()
fig.savefig(OUT, dpi=120, bbox_inches="tight")
print(f"Saved → {OUT}")
