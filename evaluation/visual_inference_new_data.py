"""Qualitative inference of trained networks on the unannotated new_data stacks.

For each (stack, ckpt) pair this script:
  1. Runs sigmoid+peak detection on every frame.
  2. Saves overlay PNGs (input + heatmap + detected peaks) for a few sample
     frames per stack so the predictions can be inspected visually.
  3. Writes a per-stack CSV with detection counts per frame and mean
     prediction confidence (proxy for confidence under domain shift).

Usage::

    PYTHONPATH=src .venv/bin/python evaluation/visual_inference_new_data.py \\
        --ckpt checkpoints/equiunet_sf16_seed0_synthetic_*.ckpt \\
        --ckpt checkpoints/unet_sf16_seed0_real_*-best.ckpt \\
        --out  evaluation/new_data_visual/ \\
        --frames 0 50 100

Outputs land in ``out/<ckpt_stem>/<stack_stem>_f{NN}.png`` for overlays and
``out/<ckpt_stem>/<stack_stem>_dets.csv`` for the per-frame counts.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from skimage import feature

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dataset.utils import open_image_file
from detection import normalize_img, _merge_adjacent_peaks
from train import build_model


# Unannotated new_data stacks. DYNAMIN_195 is excluded because the
# quantitative pipeline (eval_new_data.py) covers it.
STACKS = [
    "data/new_data/20231207_RPE-CLC-EGFP_TSIM_cc150_td_R_04_SIR.dv",
    "data/new_data/20231207_RPE-CLC-EGFP_TSIM_cc150_td_R_06_SIR.dv",
    "data/new_data/ADIPOCYTES_GLUT4_PHLUORIN_GREEN_SNAP_CLC_RED_BASAL_SORTED_242_subset_SIR_R.tif",
    "data/new_data/ADIPOCYTES_GLUT4_PHLUORIN_GREEN_SNAP_CLC_RED_BASAL_SORTED_67_SIR.dv",
    "data/new_data/ADIPOCYTES_GLUT4_PHLUORIN_GREEN_SNAP_CLC_RED_STIMULATED_1551_SORTED_142_subset_SIR_R.tif",
    "data/new_data/SHSY5Y_RUSH_LAMP1_GREEN_CLC_SNAP_RED_BIOTIN_TSIM_109_R.tif",
    "data/new_data/SHSY5Y_RUSHLAMP_CLCSNAP_108_subset_RR.tif",
    "data/new_data/SHSY5Y_RUSHLAMP_CLCSNAP_111_subset_RR.tif",
    "data/new_data/U2OS_HRBKO_CLC_SNAP_RED_228_SIR.dv",
    "data/new_data/U2OS_WT_CLC_SNAP_RED_218_SIR.dv",
    "data/new_data/U2OS_WT_CLC_SNAP_RED_225_SIR.dv",
]


def _load_model(ckpt_path: str, device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    mc = ckpt["hyper_parameters"]["model_config"]
    model = build_model(mc)
    sd = {k[len("model."):]: v for k, v in ckpt["state_dict"].items()
          if k.startswith("model.")}
    model.load_state_dict(sd, strict=True)
    return model.to(device).eval()


def _predict_frame(model: torch.nn.Module, img: np.ndarray,
                   device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with torch.no_grad():
        t = (torch.from_numpy(normalize_img(img))
             .to(device, torch.float32).unsqueeze(0).unsqueeze(0))
        pred = torch.sigmoid(model(t)).squeeze().cpu().numpy()
    peaks = feature.peak_local_max(pred, threshold_abs=0.1)
    if len(peaks) == 0:
        return pred, np.empty((0, 2)), np.empty(0)
    pv = pred[tuple(peaks.T)]
    dets, _ = _merge_adjacent_peaks(peaks, pv)
    return pred, dets, pv  # dets: (N, 2) y,x; pv: confidence at the peak


def _save_overlay(out_path: Path, img: np.ndarray, pred: np.ndarray,
                  dets: np.ndarray, title: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("input")
    axes[1].imshow(pred, cmap="hot", vmin=0, vmax=1)
    axes[1].set_title(f"sigmoid output (max={pred.max():.2f})")
    axes[2].imshow(img, cmap="gray")
    if len(dets) > 0:
        axes[2].scatter(dets[:, 1], dets[:, 0], s=8, facecolors="none",
                        edgecolors="lime", linewidths=0.7)
    axes[2].set_title(f"detections ({len(dets)})")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def process(ckpt_path: str, stacks: list[str], out_dir: Path,
            frames: list[int], device: torch.device) -> None:
    print(f"\n══ {Path(ckpt_path).name} ══", flush=True)
    model = _load_model(ckpt_path, device)
    ckpt_dir = out_dir / Path(ckpt_path).stem
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for stack_path in stacks:
        stack_stem = Path(stack_path).stem
        print(f"  {stack_stem}", end=" ", flush=True)
        try:
            imgs = open_image_file(stack_path)
        except Exception as e:
            print(f"-> SKIP ({e})")
            continue
        T = imgs.shape[0]
        print(f"({T} frames)", end=" ", flush=True)

        # Per-frame counts + mean confidence over the entire stack
        counts = np.zeros(T, dtype=int)
        mean_conf = np.zeros(T, dtype=float)
        # Overlays only for the requested sample frames
        wanted = {f for f in frames if 0 <= f < T}
        for f in range(T):
            pred, dets, pv = _predict_frame(model, imgs[f], device)
            counts[f] = len(dets)
            mean_conf[f] = float(pv.mean()) if len(pv) else 0.0
            if f in wanted:
                _save_overlay(
                    ckpt_dir / f"{stack_stem}_f{f:03d}.png",
                    imgs[f], pred, dets,
                    title=f"{Path(ckpt_path).stem}  |  {stack_stem}  |  frame {f}",
                )
        pd.DataFrame({"frame": np.arange(T), "n_detections": counts,
                      "mean_conf": mean_conf}).to_csv(
            ckpt_dir / f"{stack_stem}_dets.csv", index=False)
        print(f"-> mean {counts.mean():.0f} dets/frame")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", action="append", required=True,
                   help="checkpoint path (repeat for multiple)")
    p.add_argument("--out", default="evaluation/new_data_visual",
                   help="output root directory")
    p.add_argument("--frames", nargs="+", type=int, default=[0, 50, 100],
                   help="frame indices to save overlays for")
    p.add_argument("--stacks", nargs="*", default=None,
                   help="restrict to these stacks (default: all 11)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out)
    stacks = args.stacks if args.stacks else STACKS

    for ckpt in args.ckpt:
        process(ckpt, stacks, out_dir, args.frames, device)

    print(f"\nAll overlays + per-frame CSVs in {out_dir}/")


if __name__ == "__main__":
    main()
