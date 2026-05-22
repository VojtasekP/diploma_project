import matplotlib.animation as animation
from IPython.display import HTML, display, clear_output
import subprocess
import matplotlib as mpl
mpl.rcParams['animation.embed_limit'] = 150
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import matplotlib.collections as mc
import pandas as pd
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation
from torch.utils.data import IterableDataset
from typing import Tuple, Union
from scipy.ndimage import gaussian_filter, maximum_filter
import ipywidgets as widgets
import math
from scipy import spatial, optimize
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as torch_data
import time
import threading
import random

def hota(gt: pd.DataFrame, tr: pd.DataFrame, threshold: float = 5) -> dict[str, float]:
    """Slightly adapted from https://github.com/JonathonLuiten/TrackEval"""

    # Ensure particle ids are sorted from 0 to max(n)
    gt = gt.copy()
    tr = tr.copy()

    gt.track_id = gt.track_id.map({old: new for old, new in zip(gt.track_id.unique(), range(gt.track_id.nunique()))})
    tr.track_id = tr.track_id.map({old: new for old, new in zip(tr.track_id.unique(), range(tr.track_id.nunique()))})

    # Initialization
    num_gt_ids = gt.track_id.nunique()
    num_tr_ids = tr.track_id.nunique()

    frames = sorted(set(gt.frame.unique()) | set(tr.frame.unique()))

    potential_matches_count = np.zeros((num_gt_ids, num_tr_ids))
    gt_id_count = np.zeros((num_gt_ids, 1))
    tracker_id_count = np.zeros((1, num_tr_ids))

    HOTA_TP, HOTA_FN, HOTA_FP = 0, 0, 0
    LocA = 0.0

    # Compute similarities (inverted normalized distance)
    similarities = [1 - np.clip(spatial.distance.cdist(gt[gt.frame == t][['x', 'y']],
                                                       tr[tr.frame == t][['x', 'y']]) / threshold, 0, 1)
                    for t in frames]

    # Accumulate global track information
    for t in frames:
        gt_ids_t = gt[gt.frame == t].track_id.to_numpy()
        tr_ids_t = tr[tr.frame == t].track_id.to_numpy()

        similarity = similarities[t]
        sim_iou_denom = similarity.sum(0)[np.newaxis, :] + similarity.sum(1)[:, np.newaxis] - similarity
        sim_iou = np.zeros_like(similarity)
        sim_iou_mask = sim_iou_denom > 0 + np.finfo('float').eps
        sim_iou[sim_iou_mask] = similarity[sim_iou_mask] / sim_iou_denom[sim_iou_mask]
        potential_matches_count[gt_ids_t[:, None], tr_ids_t[None, :]] += sim_iou

        gt_id_count[gt_ids_t] += 1
        tracker_id_count[0, tr_ids_t] += 1

    global_alignment_score = potential_matches_count / (gt_id_count + tracker_id_count - potential_matches_count)
    matches_count = np.zeros_like(potential_matches_count)

    # Calculate scores for each timestep
    for t in frames:
        gt_ids_t = gt[gt.frame == t].track_id.to_numpy()
        tr_ids_t = tr[tr.frame == t].track_id.to_numpy()

        if len(gt_ids_t) == 0:
            HOTA_FP += len(tr_ids_t)
            continue

        if len(tr_ids_t) == 0:
            HOTA_FN += len(gt_ids_t)
            continue

        similarity = similarities[t]
        score_mat = global_alignment_score[gt_ids_t[:, None], tr_ids_t[None, :]] * similarity

        match_rows, match_cols = optimize.linear_sum_assignment(-score_mat)

        actually_matched_mask = similarity[match_rows, match_cols] > 0
        alpha_match_rows = match_rows[actually_matched_mask]
        alpha_match_cols = match_cols[actually_matched_mask]

        num_matches = len(alpha_match_rows)

        HOTA_TP += num_matches
        HOTA_FN += len(gt_ids_t) - num_matches
        HOTA_FP += len(tr_ids_t) - num_matches

        if num_matches > 0:
            LocA += sum(similarity[alpha_match_rows, alpha_match_cols])
            matches_count[gt_ids_t[alpha_match_rows], tr_ids_t[alpha_match_cols]] += 1

    ass_a = matches_count / np.maximum(1, gt_id_count + tracker_id_count - matches_count)
    AssA = np.sum(matches_count * ass_a) / np.maximum(1, HOTA_TP)
    DetA = HOTA_TP / np.maximum(1, HOTA_TP + HOTA_FN + HOTA_FP)
    HOTA = np.sqrt(DetA * AssA)

    return {'HOTA': HOTA, 'AssA': AssA, 'DetA': DetA, 'LocA': LocA,
            'HOTA TP': HOTA_TP, 'HOTA FN': HOTA_FN, 'HOTA FP': HOTA_FP}


def link_detections(detections_per_frame: list[list[tuple[int, int]]],
                    max_dist: float = 7.0) -> pd.DataFrame:
    """Link detections across frames into tracks using nearest neighbour association.

    Args:
        detections_per_frame: a list where each element is a list of (x,y)
            detections for the corresponding frame index.
        max_dist: maximum allowed distance in pixels between a detection and
            an existing track’s last position for association.  If no
            detection falls within this radius the track is terminated and
            a new track is started for the unmatched detection.

    Returns:
        A pandas DataFrame with columns ['frame','x','y','track_id'] containing
        the linked tracks.
    """
    next_track_id = 0
    active_tracks: dict[int, tuple[int, int, int]] = {}  # track_id -> (x, y, last_frame)
    records: list[dict[str, int]] = []
    for frame_idx, detections in enumerate(detections_per_frame):
        assigned = [False] * len(detections)
        detection_track_id: list[int | None] = [None] * len(detections)
        updated_tracks: dict[int, tuple[int, int, int]] = {}
        # attempt to match existing tracks to current detections
        for track_id, (tx, ty, last_frame) in list(active_tracks.items()):
            best_dist = max_dist
            best_idx: int | None = None
            for i, (x, y) in enumerate(detections):
                if assigned[i]:
                    continue
                dist = math.hypot(x - tx, y - ty)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i
            if best_idx is not None:
                # assign detection to this track
                assigned[best_idx] = True
                detection_track_id[best_idx] = track_id
                updated_tracks[track_id] = (detections[best_idx][0], detections[best_idx][1], frame_idx)
            # tracks with no assignment are dropped (no occlusion handling)
        # start new tracks for unmatched detections
        for i, (x, y) in enumerate(detections):
            if not assigned[i]:
                track_id = next_track_id
                next_track_id += 1
                detection_track_id[i] = track_id
                updated_tracks[track_id] = (x, y, frame_idx)
        # update active tracks
        active_tracks = updated_tracks
        # record detections with assigned track ids
        for i, (x, y) in enumerate(detections):
            tid = detection_track_id[i]
            records.append({'frame': frame_idx, 'x': x, 'y': y, 'track_id': tid})
    return pd.DataFrame(records)


def show_tracking(data, image_stack,
                  y_min=512, y_max=768, x_min=256, x_max=512,
                  tail_length=10, color='yellow', show_roi=True):
    """
    Visualize CCP trajectories within a defined ROI, from either a CSV file or a DataFrame.

    Args:
        data (str | pd.DataFrame): Path to CSV file or a DataFrame containing trajectories.
        image_stack (np.ndarray): 3D numpy array (frames, height, width).
        y_min, y_max, x_min, x_max (int): ROI bounds.
        tail_length (int): Number of frames for trajectory tails.
        color (str): Trajectory color.
        show_roi (bool): Whether to display cyan ROI rectangle on the full frame.
    """

    if isinstance(data, str):
        trajectories_df = pd.read_csv(data)
    elif isinstance(data, pd.DataFrame):
        trajectories_df = data.copy()
    else:
        raise TypeError("`data` must be a CSV file path or a pandas DataFrame.")

    tracks_in_roi = trajectories_df.groupby('track_id').filter(
        lambda t: (y_min < t.y.mean() < y_max) and (x_min < t.x.mean() < x_max)
    )

    html_code_linking = loading_html("Loading cropped region and tracks, please wait...")
    display(HTML(html_code_linking))

    if show_roi:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(image_stack[0], cmap='magma')
        rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                         linewidth=2, edgecolor='cyan', facecolor='none')
        ax.add_patch(rect)
        ax.set_title("Full image (cyan box shows cropped region)")
        plt.show()

    def animate_trajectories_cropped(trajectories_df, image_stack, tail_length=10, color='yellow'):
        cropped_stack = image_stack[:, y_min:y_max, x_min:x_max]

        fig, ax = plt.subplots()
        im = ax.imshow(cropped_stack[0], cmap='magma')
        particles = trajectories_df['track_id'].unique()

        line_collections = {pid: mc.LineCollection([], linewidths=1, colors=color) for pid in particles}
        for lc in line_collections.values():
            ax.add_collection(lc)

        dot = ax.scatter([], [], s=5, c=color)

        def animate(i):
            im.set_array(cropped_stack[i])

            window = trajectories_df[
                (trajectories_df['frame'] >= i - tail_length) &
                (trajectories_df['frame'] <= i)
            ]

            now = window[window['frame'] == i]
            if len(now) > 0:
                coords = np.column_stack((now.x.values - x_min, now.y.values - y_min))
                dot.set_offsets(coords)
            else:
                dot.set_offsets(np.empty((0, 2)))

            for pid in particles:
                traj = window[window['track_id'] == pid].sort_values('frame')
                if len(traj) >= 2:
                    segs = [
                        [(x0 - x_min, y0 - y_min), (x1 - x_min, y1 - y_min)]
                        for (x0, y0, x1, y1) in zip(
                            traj.x.values[:-1], traj.y.values[:-1],
                            traj.x.values[1:], traj.y.values[1:]
                        )
                    ]
                    line_collections[pid].set_segments(segs)
                else:
                    line_collections[pid].set_segments([])

            return [im, dot] + list(line_collections.values())

        ani = FuncAnimation(fig, animate, frames=cropped_stack.shape[0], interval=100, blit=True)
        plt.close(fig)
        return HTML(ani.to_jshtml())

    html = animate_trajectories_cropped(tracks_in_roi, image_stack, tail_length, color)
    display(html)
    display(HTML(replace_loading_js_empty))

    print("Total number of trajectories in ROI:", len(tracks_in_roi['track_id'].unique()))


def visualize_model_on_dataset(model, dataset, device, num_samples=4, threshold=0.5, sigma=1.0):
    """
    Visualize model predictions on a few samples from the dataset.
    Shows: input image, ground truth mask, predicted probability map, and binary detection mask.
    """
    model.eval()
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    with torch.no_grad():
        for i, (img_tensor, mask_tensor) in enumerate(loader):
            if i >= num_samples:
                break

            img_tensor = img_tensor.to(device)
            mask_tensor = mask_tensor.to(device)

            # Run through model
            logits = model(img_tensor)
            prob_map = torch.sigmoid(logits[0, 0]).cpu().numpy()

            # Optional smoothing
            if sigma > 0:
                prob_map = gaussian_filter(prob_map, sigma=sigma)

            # Threshold for binary detection
            pred_mask = (prob_map >= threshold).astype(float)

            img = img_tensor[0, 0].cpu().numpy()
            gt_mask = mask_tensor[0, 0].cpu().numpy()

            fig, axes = plt.subplots(1, 4, figsize=(14, 4))
            axes[0].imshow(img, cmap='magma')
            axes[0].set_title("Input Image")
            axes[1].imshow(gt_mask, cmap='gray', vmin=0, vmax=1)
            axes[1].set_title("Ground Truth Mask")
            axes[2].imshow(prob_map, cmap='viridis')
            axes[2].set_title("Predicted Heatmap")
            axes[3].imshow(pred_mask, cmap='gray', vmin=0, vmax=1)
            axes[3].set_title(f"Thresholded Output (>{threshold})")
            for ax in axes:
                ax.axis('off')
            plt.tight_layout()
            plt.show()


def show_detections(detections_per_frame, image_stack,
                    y_min=512, y_max=768, x_min=256, x_max=512,
                    color='yellow', max_frames=5):
    """
    Visualize detections in the specified ROI.

    Args:
        detections_per_frame (list[list[tuple]]): List of per-frame detection coordinates [(x, y)].
        image_stack (np.ndarray): Image sequence (frames, height, width).
        y_min, y_max, x_min, x_max (int): ROI bounds.
        color (str): Color of detection markers.
        max_frames (int): Number of frames to animate (default: 5).
    """

    rows = [(frame_idx, x, y)
            for frame_idx, dets in enumerate(detections_per_frame)
            for (x, y) in dets]
    detections_df = pd.DataFrame(rows, columns=["frame", "x", "y"])

    detections_df = detections_df[
        (detections_df.y.between(y_min, y_max)) &
        (detections_df.x.between(x_min, x_max))
    ]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image_stack[0], cmap='magma')
    rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                     linewidth=2, edgecolor='cyan', facecolor='none')
    ax.add_patch(rect)
    ax.set_title("Full image (cyan box shows cropped region)")
    plt.show()

    def animate_detections_cropped(detections_df, image_stack, color='yellow'):
        cropped_stack = image_stack[:, y_min:y_max, x_min:x_max]

        fig, ax = plt.subplots()
        im = ax.imshow(cropped_stack[0], cmap='magma')
        dot = ax.scatter([], [], s=10, c=color)

        def animate(i):
            im.set_array(cropped_stack[i])
            now = detections_df[detections_df['frame'] == i]
            if len(now) > 0:
                coords = np.column_stack((now.x.values - x_min, now.y.values - y_min))
                dot.set_offsets(coords)
            else:
                dot.set_offsets(np.empty((0, 2)))
            return [im, dot]

        ani = FuncAnimation(fig, animate,
                            frames=min(max_frames, cropped_stack.shape[0]),
                            interval=1000, blit=True)
        plt.close(fig)
        return HTML(ani.to_jshtml())

    html = animate_detections_cropped(detections_df, np.array(image_stack), color)
    display(html)

    print(f"Total detections in ROI : {len(detections_df)} \nShowing first {max_frames} frames")



def calculate_performance(gt_path, tracks,
                          y_min=512, y_max=768, x_min=256, x_max=512,
                          name="Method"):
    """
    Calculate and print HOTA-based performance metrics for a tracking result.

    Args:
        gt_path (str): Path to ground truth CSV file.
        tracks (str | pd.DataFrame): Tracking results (as DataFrame or CSV path).
        y_min, y_max, x_min, x_max (int): ROI bounds.
        name (str): Name of the method (for display).
    """

    val_gt = pd.read_csv(gt_path)
    val_gt = val_gt.groupby('track_id').filter(
        lambda t: (y_min < t.y.mean() < y_max) and (x_min < t.x.mean() < x_max)
    )

    if isinstance(tracks, str):
        val_tracks = pd.read_csv(tracks)
    elif isinstance(tracks, pd.DataFrame):
        val_tracks = tracks.copy()
    else:
        raise TypeError("`tracks` must be a CSV path or a pandas DataFrame.")

    val_tracks = val_tracks.groupby('track_id').filter(
        lambda t: (y_min < t.y.mean() < y_max) and (x_min < t.x.mean() < x_max)
    )

    # Compute metrics
    results = hota(val_gt, val_tracks)

    print(f"{name}:")
    print(f"  HOTA: {results['HOTA']:.2f} "
          f"(AssA: {results['AssA']:.2f}, DetA: {results['DetA']:.2f})\n")

    return results

    print(f"✅ Done")