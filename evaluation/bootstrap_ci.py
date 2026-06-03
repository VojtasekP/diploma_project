"""Bootstrap confidence intervals for val DetA on the real (033) stack.

The effective sample size for DetA is the number of distinct CCP *tracks*
(~150-180 in the 24-frame val split), not the ~2500 per-frame detections —
detections of the same CCP across consecutive frames are strongly correlated.
This script resamples whole tracks (recall side) and whole frames (false-positive
side) to produce an honest CI, and — given two checkpoints — a *paired*
EquiUNet − UNet difference CI on the identical val frames, where per-CCP
difficulty cancels.

Usage:
  .venv/bin/python evaluation/bootstrap_ci.py \
      --equi-ckpt checkpoints/equiunet_..._real_...-best.ckpt \
      --unet-ckpt checkpoints/unet_..._real_...-best.ckpt \
      --annotators 1,2            # held-out annotators (when trained on ann3)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

SRC = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC))

from detection import generate_ccp_detections          # noqa: E402
from dataset.utils import open_image_file               # noqa: E402
from train import load_lit_from_checkpoint, CROP, MATCHING_THRESHOLD  # noqa: E402

DATA = Path(__file__).parent.parent / "data" / "CME_tracking_validation"
STEM = "RPE1_egfpCLCa_033"


def load_val_detections(ckpt_path: str, val_frames: set[int]) -> pd.DataFrame:
    """Run a checkpoint's detector over the 033 stack, keep ROI + val frames."""
    recon = DATA / f"{STEM}-recon.mrc"
    img = DATA / f"{STEM}.mrc"
    stack = open_image_file(str(recon if recon.exists() else img)).astype(np.float32)
    lit = load_lit_from_checkpoint(ckpt_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    det = generate_ccp_detections(lit, device, stack)
    det = det[
        (det.x >= CROP["x_min"]) & (det.x <= CROP["x_max"]) &
        (det.y >= CROP["y_min"]) & (det.y <= CROP["y_max"]) &
        (det.frame.isin(val_frames))
    ].reset_index(drop=True)
    return det


def match(det: pd.DataFrame, gt: pd.DataFrame, threshold: float):
    """Per-frame Hungarian matching (same rule as train._approx_deta).

    Returns:
      track_tp:  {track_id: #GT points of that track detected (TP)}
      track_n:   {track_id: #GT points of that track total (TP+FN)}
      frame_fp:  {frame: #detections matching no GT (FP)}
    """
    track_tp: dict[int, int] = {}
    track_n: dict[int, int] = {}
    frame_fp: dict[int, int] = {}
    frames = set(gt.frame.unique()) | set(det.frame.unique())
    for f in frames:
        g = gt[gt.frame == f]
        d = det[det.frame == f]
        g_xy, tids = g[["x", "y"]].values, g["track_id"].values
        d_xy = d[["x", "y"]].values
        hit = np.zeros(len(g_xy), dtype=bool)
        if len(g_xy) and len(d_xy):
            D = cdist(d_xy, g_xy)
            ri, ci = linear_sum_assignment(D)
            ok = D[ri, ci] < threshold
            for g_i, m in zip(ci, ok):
                if m:
                    hit[g_i] = True
            frame_fp[f] = int(len(d_xy) - ok.sum())
        else:
            frame_fp[f] = int(len(d_xy))
        for t, h in zip(tids, hit):
            track_n[t] = track_n.get(t, 0) + 1
            track_tp[t] = track_tp.get(t, 0) + int(h)
    return track_tp, track_n, frame_fp


def deta_from_units(track_tp, track_n, frame_fp, tracks, frames) -> float:
    tp = sum(track_tp[t] for t in tracks)
    fn = sum(track_n[t] - track_tp[t] for t in tracks)
    fp = sum(frame_fp[f] for f in frames)
    denom = tp + fp + fn
    return tp / denom if denom else 0.0


def bootstrap(models: dict, gt_by_ann: dict, frames_sorted: list, B: int, seed: int):
    """Paired track+frame bootstrap.

    models:    {name: {ann: (track_tp, track_n, frame_fp)}}
    Returns per-model arrays of bootstrap DetA and (if 2 models) paired diffs.
    """
    rng = np.random.default_rng(seed)
    names = list(models)
    boot = {n: np.empty(B) for n in names}
    diff = np.empty(B) if len(names) == 2 else None
    anns = list(gt_by_ann)

    # Per-annotator track lists (resampling populations).
    tracks_by_ann = {a: list(models[names[0]][a][1].keys()) for a in anns}

    for b in range(B):
        per_model_ann = {n: [] for n in names}
        for a in anns:
            tr = tracks_by_ann[a]
            samp_tracks = rng.choice(tr, size=len(tr), replace=True)
            samp_frames = rng.choice(frames_sorted, size=len(frames_sorted), replace=True)
            for n in names:
                t_tp, t_n, f_fp = models[n][a]
                per_model_ann[n].append(
                    deta_from_units(t_tp, t_n, f_fp, samp_tracks, samp_frames))
        for n in names:
            boot[n][b] = float(np.mean(per_model_ann[n]))
        if diff is not None:
            diff[b] = boot[names[1]][b] - boot[names[0]][b]
    return boot, diff


def ci(arr, lo=2.5, hi=97.5):
    return float(np.percentile(arr, lo)), float(np.percentile(arr, hi))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--equi-ckpt", default=None)
    ap.add_argument("--unet-ckpt", default=None)
    ap.add_argument("--annotators", default="1,2",
                    help="comma-separated held-out annotator ids to score against")
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--threshold", type=float, default=float(MATCHING_THRESHOLD))
    ap.add_argument("-B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ckpts = {}
    if args.unet_ckpt:
        ckpts["unet"] = args.unet_ckpt
    if args.equi_ckpt:
        ckpts["equiunet"] = args.equi_ckpt
    if not ckpts:
        ap.error("provide at least one of --unet-ckpt / --equi-ckpt")

    # Contiguous val split (last fraction), matching train.split_real_ccp.
    n_total = 120
    n_val = max(1, round(n_total * args.val_fraction))
    val_frames = set(range(n_total - n_val, n_total))
    frames_sorted = sorted(val_frames)
    ann_ids = [int(a) for a in args.annotators.split(",")]
    print(f"val frames: {frames_sorted[0]}..{frames_sorted[-1]} ({n_val})  "
          f"annotators: {ann_ids}  threshold: {args.threshold}px  B={args.B}")

    gt_by_ann = {}
    for a in ann_ids:
        df = pd.read_csv(DATA / f"{STEM}-annotations_{a}.csv")
        df = df[
            (df.x >= CROP["x_min"]) & (df.x <= CROP["x_max"]) &
            (df.y >= CROP["y_min"]) & (df.y <= CROP["y_max"]) &
            (df.frame.isin(val_frames))
        ]
        gt_by_ann[a] = df

    models = {}
    for name, path in ckpts.items():
        print(f"\nrunning detector: {name}  ({Path(path).name})")
        det = load_val_detections(path, val_frames)
        per_ann = {}
        for a, gt in gt_by_ann.items():
            per_ann[a] = match(det, gt, args.threshold)
        models[name] = per_ann
        # point estimate
        detas = [deta_from_units(*per_ann[a],
                                 list(per_ann[a][1].keys()), frames_sorted)
                 for a in ann_ids]
        n_tracks = sum(len(per_ann[a][1]) for a in ann_ids)
        print(f"  point DetA (mean over ann): {np.mean(detas):.4f}   "
              f"per-ann: {[round(x, 4) for x in detas]}   "
              f"({n_tracks} tracks total)")

    boot, diff = bootstrap(models, gt_by_ann, frames_sorted, args.B, args.seed)

    print(f"\n=== Bootstrap CIs ({args.B} resamples, tracks + frames) ===")
    for n in models:
        lo, hi = ci(boot[n])
        print(f"  {n:9s} DetA = {boot[n].mean():.4f}  95% CI [{lo:.4f}, {hi:.4f}]  "
              f"(±{(hi - lo) / 2:.4f})")
    if diff is not None:
        names = list(models)
        lo, hi = ci(diff)
        p = float((diff <= 0).mean())  # one-sided: P(equi <= unet)
        print(f"\n  paired {names[1]} − {names[0]} = {diff.mean():+.4f}  "
              f"95% CI [{lo:+.4f}, {hi:+.4f}]")
        print(f"  P(diff <= 0) = {p:.3f}  →  "
              f"{'significant at 95%' if (lo > 0 or hi < 0) else 'NOT significant at 95%'}")


if __name__ == "__main__":
    main()
