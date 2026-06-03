"""Robustness evaluation of trained networks on the new DYNAMIN_195 stack.

Runs detection + tracking + HOTA for any number of self-identifying
checkpoints from ``checkpoints/`` against the trajectories shipped with the
RPE-DYNAMIN dataset. Output is a single results CSV with one row per ckpt.

Usage::

    PYTHONPATH=src .venv/bin/python evaluation/eval_new_data.py \\
        checkpoints/equiunet_sf16_seed0_synthetic_*.ckpt \\
        checkpoints/unet_sf16_seed0_real_*-best.ckpt \\
        --output evaluation/new_data_eval.csv

The stack and GT trajectories are hard-coded to DYNAMIN_195 (the only file
in ``data/new_data/`` for which we have point/trajectory ground truth).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import linking
import metrics
from dataset.utils import open_image_file
from detection import generate_ccp_detections
from parameters import LinkingParameters
from train import build_model


STACK_PATH = "data/new_data/DYNAMIN_MRUBY2_RED_CLC_SNAP_FARRED_TSIM_195_R.tif"
GT_PATH    = "data/new_data/RPE trajectories/DYNAMIN_MRUBY2_RED_CLC_SNAP_FARRED_TSIM_195-trajectories.csv"
MASK_PATH  = "data/new_data/RPE trajectories/DYNAMIN_MRUBY2_RED_CLC_SNAP_FARRED_TSIM_195_MASK.tif"

MATCHING_THRESHOLD = 5  # px, same as train.py

# Optional ROI filter. For 033 val/test where annotators only labelled within
# this crop, the eval must restrict detections to the ROI (otherwise
# whole-frame detections outside the ROI count as FPs against missing GT).
# Pass --roi 256 512 512 768 (x_min x_max y_min y_max) on CLI to enable.

LP = LinkingParameters(
    birth_death_cost=5,
    edge_removal_cost=10,
    feature_cost_multiplier=1,
    maximum_distance=7.5,
    maximum_skipped_frames=1,
    minimum_length=5,
)


def _load_model(ckpt_path: str) -> torch.nn.Module:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    mc = ckpt.get("hyper_parameters", {}).get("model_config")
    if mc is None:
        raise ValueError(
            f"{ckpt_path} has no model_config in hyper_parameters — "
            "cannot reconstruct architecture. Use a self-identifying ckpt "
            "from the post-rename era."
        )
    model = build_model(mc)
    sd = ckpt["state_dict"]
    sd = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
    model.load_state_dict(sd, strict=True)
    return model


def _distance_fn(a: pd.DataFrame, b: pd.DataFrame) -> np.ndarray:
    from scipy import spatial
    euclid = spatial.distance.cdist(a[["x", "y"]].values, b[["x", "y"]].values)
    si_diff = np.square(a["cls"].values[:, None] - b["cls"].values[None, :])
    return euclid + LP.feature_cost_multiplier * si_diff


def _track(det: pd.DataFrame) -> pd.DataFrame | None:
    lg = linking.LinkingGraph(
        det, _distance_fn, LP.maximum_distance, LP.birth_death_cost,
        max_skipped_frames=LP.maximum_skipped_frames,
    )
    if lg.solve() != lg.solver.OPTIMAL:
        return None
    tracklets = lg.get_result()
    ug = linking.UntanglingGraph(tracklets, LP.edge_removal_cost)
    if ug.solve() != ug.solver.OPTIMAL:
        return None
    trajectories = ug.get_result(det)
    return trajectories.groupby("particle").filter(
        lambda t: t.frame.count() >= LP.minimum_length
    )


def _det_only_deta(det: pd.DataFrame, gt: pd.DataFrame, threshold: int) -> float:
    """Hungarian per-frame DetA — fast, no tracking."""
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.distance import cdist as _cdist
    tp = fp = fn = 0
    for f in set(gt.frame.unique()) | set(det.frame.unique()):
        gt_xy  = gt[gt.frame == f][["x", "y"]].values
        det_xy = det[det.frame == f][["x", "y"]].values
        if len(gt_xy) == 0:
            fp += len(det_xy); continue
        if len(det_xy) == 0:
            fn += len(gt_xy); continue
        D = _cdist(det_xy, gt_xy)
        ri, ci = linear_sum_assignment(D)
        n = int((D[ri, ci] < threshold).sum())
        tp += n; fp += len(det_xy) - n; fn += len(gt_xy) - n
    return tp / (tp + fp + fn) if (tp + fp + fn) else 0.0


def evaluate(ckpt_path: str, images: np.ndarray, gt: pd.DataFrame,
             device: torch.device, do_tracking: bool,
             roi: tuple[int, int, int, int] | None = None) -> dict:
    print(f"\n── {Path(ckpt_path).name} ──", flush=True)
    model = _load_model(ckpt_path).to(device).eval()

    det = generate_ccp_detections(model, device, images)
    if roi is not None:
        xmin, xmax, ymin, ymax = roi
        mask = ((det.x >= xmin) & (det.x <= xmax) &
                (det.y >= ymin) & (det.y <= ymax))
        n_before = len(det)
        det = det.loc[mask].reset_index(drop=True)
        print(f"  detections: {n_before} → {len(det)} after ROI {roi}")
    else:
        print(f"  detections: {len(det)} across {det.frame.nunique()} frames")

    out = {
        "ckpt": Path(ckpt_path).name,
        "n_detections": len(det),
        "det_only_DetA": _det_only_deta(det, gt, MATCHING_THRESHOLD),
    }
    print(f"  det-only DetA = {out['det_only_DetA']:.4f}")

    if do_tracking:
        traj = _track(det)
        if traj is None:
            print("  tracking FAILED (solver non-optimal)")
            out.update({"HOTA": np.nan, "DetA": np.nan, "AssA": np.nan,
                        "n_tracks": 0})
        else:
            r = metrics.hota(gt.rename(columns={"track_id": "particle"}), traj,
                             MATCHING_THRESHOLD)
            out.update({
                "HOTA": float(r["HOTA"]), "DetA": float(r["DetA"]),
                "AssA": float(r["AssA"]), "n_tracks": int(traj.particle.nunique()),
            })
            print(f"  HOTA={out['HOTA']:.4f}  DetA={out['DetA']:.4f}  "
                  f"AssA={out['AssA']:.4f}  ({out['n_tracks']} tracks)")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("ckpts", nargs="+", help="checkpoint paths")
    p.add_argument("--output", default="evaluation/new_data_eval.csv")
    p.add_argument("--stack", default=STACK_PATH,
                   help=f"image stack path (default: {STACK_PATH})")
    p.add_argument("--gt", default=GT_PATH,
                   help=f"GT trajectories CSV (default: {GT_PATH})")
    p.add_argument("--no-tracking", action="store_true",
                   help="skip the linking/untangling step (det-only DetA only)")
    p.add_argument("--max-frames", type=int, default=None,
                   help="limit eval to first N frames (for debugging)")
    p.add_argument("--roi", nargs=4, type=int, metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
                   default=None,
                   help="restrict detections to this ROI (e.g. for 033: 256 512 512 768)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading stack: {args.stack}")
    images = open_image_file(args.stack)
    if args.max_frames is not None:
        images = images[:args.max_frames]
    print(f"  shape: {images.shape}, dtype: {images.dtype}")

    print(f"Loading GT: {args.gt}")
    gt = pd.read_csv(args.gt).rename(columns={"track_id": "particle"})
    if args.max_frames is not None:
        gt = gt[gt.frame < args.max_frames]
    print(f"  {len(gt)} detections, {gt.particle.nunique()} tracks, "
          f"frames {gt.frame.min()}-{gt.frame.max()}")

    rows = []
    for ckpt in args.ckpts:
        try:
            rows.append(evaluate(ckpt, images, gt, device,
                                 do_tracking=not args.no_tracking,
                                 roi=tuple(args.roi) if args.roi else None))
        except Exception as e:
            print(f"  ERROR on {ckpt}: {e}")
            rows.append({"ckpt": Path(ckpt).name, "error": str(e)})

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"\nResults → {out_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
