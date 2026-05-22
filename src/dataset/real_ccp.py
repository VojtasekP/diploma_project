"""Real CCP dataset built from point-center annotations.

Loads `.mrc` stacks plus their `*-annotations_*.csv` files, fuses centers
across annotators by greedy clustering, and renders a disk (or Gaussian)
mask around each fused center. Returns ``(image, mask)`` tensors compatible
with the existing UNet/EquiUNet training loop.
"""
from pathlib import Path
from typing import Union

import lightning as L
import numpy as np
import pandas as pd
import torch
import torch.utils.data as torch_data

from .utils import open_image_file


CROP_Y_START = 512
CROP_X_START = 256
CROP_SIZE    = 256


def _fuse_annotators(annotator_points: list[np.ndarray], threshold: float) -> np.ndarray:
    clusters: list[list[np.ndarray]] = []
    centroids: np.ndarray = np.zeros((0, 2), np.float32)
    for ann in annotator_points:
        for p in ann:
            if len(clusters):
                d = np.linalg.norm(centroids - p, axis=1)
                j = int(d.argmin())
                if d[j] < threshold:
                    clusters[j].append(p)
                    centroids[j] = np.mean(clusters[j], axis=0)
                    continue
            clusters.append([p])
            centroids = np.vstack([centroids, p[None]])
    if not clusters:
        return np.zeros((0, 2), dtype=np.float32)
    return centroids.astype(np.float32)


def _disk_mask(centers: np.ndarray, shape: tuple[int, int], radius: float) -> np.ndarray:
    if len(centers) == 0:
        return np.zeros(shape, dtype=np.float32)
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    d2 = (xx[..., None] - centers[:, 0]) ** 2 + (yy[..., None] - centers[:, 1]) ** 2
    return (d2.min(axis=-1) <= radius ** 2).astype(np.float32)


def _gaussian_mask(centers: np.ndarray, shape: tuple[int, int], sigma: float) -> np.ndarray:
    if len(centers) == 0:
        return np.zeros(shape, dtype=np.float32)
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    d2 = (xx[..., None] - centers[:, 0]) ** 2 + (yy[..., None] - centers[:, 1]) ** 2
    g = np.exp(-d2 / (2 * sigma ** 2)).sum(axis=-1)
    return np.minimum(g, 1.0).astype(np.float32)


def _load_real_ccp(data_dir: Union[str, Path],
                   *,
                   disk_radius: float,
                   gaussian_sigma: float,
                   cluster_threshold: float,
                   use_recon: bool,
                   dominant_annotator: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """``dominant_annotator`` selects fusion order:
      * 1, 2, 3 → that annotator's CSV seeds every cluster (deterministic bias)
      * 0       → randomise the per-frame fusion order (deterministic by
        frame index, so reproducible across runs) — removes annotator bias
        in the aggregated mask without breaking tracking GT (tracking metrics
        score against raw per-annotator CSVs, not the fused mask)
    """
    data_dir = Path(data_dir)
    img_files = sorted(p for p in data_dir.glob('*.mrc') if '-recon' not in p.stem)
    if not img_files:
        raise FileNotFoundError(f"No .mrc files found in {data_dir}")

    stack_list, mask_list = [], []
    for img_path in img_files:
        stem = img_path.stem
        recon = data_dir / f'{stem}-recon.mrc'
        load_path = recon if use_recon and recon.exists() else img_path
        arr = open_image_file(str(load_path))
        if arr.ndim == 2:
            arr = arr[np.newaxis]

        csv_files = sorted(data_dir.glob(f'{stem}-annotations_*.csv'))
        if not csv_files:
            continue
        if dominant_annotator in (1, 2, 3):
            # Deterministic: chosen annotator's CSV first.
            dom_name = f"{stem}-annotations_{dominant_annotator}.csv"
            csv_files = ([p for p in csv_files if p.name == dom_name]
                         + [p for p in csv_files if p.name != dom_name])
        # For dominant_annotator==0, csv_files keep sorted order; per-frame
        # shuffling happens just before fusion below.

        annotator_frames = []
        for csv_path in csv_files:
            df = pd.read_csv(csv_path)
            annotator_frames.append({
                int(frame): grp[['x', 'y']].values.astype(np.float32)
                for frame, grp in df.groupby('frame')
            })

        stack_list.append(arr)
        for local_frame in range(len(arr)):
            pts = []
            for a in annotator_frames:
                p = a.get(local_frame, np.zeros((0, 2), np.float32)).copy()
                p[:, 0] -= CROP_X_START
                p[:, 1] -= CROP_Y_START
                inside = ((p[:, 0] >= 0) & (p[:, 0] < CROP_SIZE) &
                          (p[:, 1] >= 0) & (p[:, 1] < CROP_SIZE))
                pts.append(p[inside])
            if dominant_annotator == 0:
                # Deterministic per-frame shuffle: same shuffling for the
                # same frame across runs (frame index = seed).
                rng = np.random.default_rng(local_frame)
                order = rng.permutation(len(pts))
                pts = [pts[i] for i in order]
            fused = _fuse_annotators(pts, cluster_threshold)
            if gaussian_sigma > 0:
                m = _gaussian_mask(fused, (CROP_SIZE, CROP_SIZE), gaussian_sigma)
            else:
                m = _disk_mask(fused, (CROP_SIZE, CROP_SIZE), disk_radius)
            mask_list.append(m)

    full_stack = np.concatenate(stack_list, axis=0).astype(np.float32)
    masks = np.stack(mask_list, axis=0)
    return full_stack, masks


class CCPCenterDataset(torch_data.Dataset):
    """Real CCP dataset with disk/Gaussian masks built from center annotations.

    Frames are restricted to ROI ``[512:768, 256:512]`` (256×256), matching the
    crop used by ``AnnotatedPytorchDataset``. Annotator points are fused by
    greedy clustering with radius ``cluster_threshold``.

    Construct via :func:`split_real_ccp` (recommended), or pass already-loaded
    ``(imgs, masks)`` via the ``data`` argument to share storage between
    train / val instances.
    """

    def __init__(self,
                 data_dir: Union[str, Path] | None = None,
                 *,
                 data: tuple[np.ndarray, np.ndarray] | None = None,
                 frames: list[int] | None = None,
                 patch_size: int = 128,
                 epoch_length: int | None = None,
                 augment: bool = False,
                 disk_radius: float = 3.0,
                 gaussian_sigma: float = 0.0,
                 cluster_threshold: float = 5.0,
                 use_recon: bool = True,
                 dominant_annotator: int = 1):
        super().__init__()
        if data is None:
            data = _load_real_ccp(data_dir,
                                  disk_radius=disk_radius,
                                  gaussian_sigma=gaussian_sigma,
                                  cluster_threshold=cluster_threshold,
                                  use_recon=use_recon,
                                  dominant_annotator=dominant_annotator)
        self._frames_imgs, self._frames_masks = data

        n_frames = len(self._frames_imgs)
        self._frame_indices = list(range(n_frames)) if frames is None else list(frames)
        if self._frame_indices and max(self._frame_indices) >= n_frames:
            raise ValueError(f"frame index out of range; have {n_frames} frames")

        self.patch_size  = patch_size
        self.augment     = augment
        self._epoch_len  = epoch_length if epoch_length is not None else len(self._frame_indices)

    def __len__(self) -> int:
        return self._epoch_len

    @property
    def stack(self) -> np.ndarray:
        """Full image stack (T, H, W) — used by detection eval pipeline."""
        return self._frames_imgs

    @property
    def frame_indices(self) -> list[int]:
        """Frames belonging to this split — used to filter eval metrics."""
        return list(self._frame_indices)

    @staticmethod
    def _normalize(img: np.ndarray) -> np.ndarray:
        # Match the synthetic pipeline: zero-mean unit-variance per patch.
        # The previous min-max scaling produced an [0,1] distribution that
        # collides with the synthetic-pretrained model and destroys it during
        # fine-tuning.
        return (img - img.mean()) / (img.std() + 1e-6)

    def __getitem__(self, idx: int):
        rng = np.random.default_rng() if self.augment else np.random.default_rng(idx)
        if self.augment:
            frame = self._frame_indices[rng.integers(0, len(self._frame_indices))]
        else:
            frame = self._frame_indices[idx % len(self._frame_indices)]

        # _frames_imgs is the full (T, H, W) stack so that .stack stays in
        # original coordinates for the detection pipeline; crop training
        # patches from the same ROI used at eval time.
        img_full  = self._frames_imgs[frame]
        img_roi   = img_full[CROP_Y_START:CROP_Y_START + CROP_SIZE,
                             CROP_X_START:CROP_X_START + CROP_SIZE]
        mask = self._frames_masks[frame]

        ps = self.patch_size
        if ps < CROP_SIZE:
            if self.augment:
                y0 = int(rng.integers(0, CROP_SIZE - ps + 1))
                x0 = int(rng.integers(0, CROP_SIZE - ps + 1))
            else:
                y0 = (CROP_SIZE - ps) // 2
                x0 = (CROP_SIZE - ps) // 2
            img  = img_roi[y0:y0 + ps, x0:x0 + ps]
            mask = mask[y0:y0 + ps, x0:x0 + ps]
        else:
            img = img_roi

        if self.augment:
            if rng.random() < 0.5:
                img, mask = img[:, ::-1], mask[:, ::-1]
            if rng.random() < 0.5:
                img, mask = img[::-1, :], mask[::-1, :]
            k = int(rng.integers(0, 4))
            if k:
                img, mask = np.rot90(img, k), np.rot90(mask, k)
            img, mask = np.ascontiguousarray(img), np.ascontiguousarray(mask)

        img = self._normalize(img)
        return (torch.from_numpy(img).unsqueeze(0).float(),
                torch.from_numpy(mask).unsqueeze(0).float())


def split_real_ccp(data_dir: Union[str, Path],
                   *,
                   val_fraction: float = 0.2,
                   seed: int = 0,
                   split_mode: str = "random",
                   train_kwargs: dict | None = None,
                   val_kwargs: dict | None = None,
                   disk_radius: float = 3.0,
                   gaussian_sigma: float = 0.0,
                   cluster_threshold: float = 5.0,
                   use_recon: bool = True,
                   dominant_annotator: int = 1,
                   ) -> tuple[CCPCenterDataset, CCPCenterDataset]:
    """Build two ``CCPCenterDataset``s with disjoint frame indices.

    Data is loaded once and shared between the two views.

    ``split_mode``:
      * ``"random"`` — frames shuffled with ``seed`` then split. Safer i.i.d.
        but the val set is non-contiguous → tracking metrics not meaningful.
      * ``"contiguous"`` — last ``val_fraction`` of frames as val. Allows
        HOTA/AssA on val, but val is correlated with train (temporal drift).
    """
    data = _load_real_ccp(data_dir,
                          disk_radius=disk_radius,
                          gaussian_sigma=gaussian_sigma,
                          cluster_threshold=cluster_threshold,
                          use_recon=use_recon,
                          dominant_annotator=dominant_annotator)
    n = len(data[0])
    n_val = max(1, int(round(n * val_fraction)))

    if split_mode == "random":
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        val_frames   = sorted(perm[:n_val].tolist())
        train_frames = sorted(perm[n_val:].tolist())
    elif split_mode == "contiguous":
        train_frames = list(range(0, n - n_val))
        val_frames   = list(range(n - n_val, n))
    else:
        raise ValueError(f"split_mode must be 'random' or 'contiguous', got {split_mode!r}")

    train = CCPCenterDataset(data=data, frames=train_frames, augment=True,
                             **(train_kwargs or {}))
    val   = CCPCenterDataset(data=data, frames=val_frames, augment=False,
                             **(val_kwargs or {}))
    return train, val


class RealCCPLightningDataModule(L.LightningDataModule):
    """Train/val from real annotations (033, frame-disjoint); test from 012."""

    _DATA_ROOT = Path(__file__).parent.parent.parent / "data"

    def __init__(self,
                 batch_size: int = 16,
                 num_workers: int = 4,
                 val_fraction: float = 0.2,
                 split_seed: int = 0,
                 split_mode: str = "random",
                 train_patch_size: int = 128,
                 train_epoch_length: int = 1600,
                 val_patch_size: int = 256,
                 disk_radius: float = 3.0,
                 gaussian_sigma: float = 0.0,
                 cluster_threshold: float = 5.0,
                 use_recon: bool = True,
                 dominant_annotator: int = 1):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_fraction = val_fraction
        self.split_seed = split_seed
        self.split_mode = split_mode
        self.train_patch_size = train_patch_size
        self.train_epoch_length = train_epoch_length
        self.val_patch_size = val_patch_size
        self.disk_radius = disk_radius
        self.gaussian_sigma = gaussian_sigma
        self.cluster_threshold = cluster_threshold
        self.use_recon = use_recon
        self.dominant_annotator = dominant_annotator

    def setup(self, stage=None):
        from .dataset import AnnotatedPytorchDataset

        if stage in ('fit', None):
            self.train_dataset, self.val_dataset = split_real_ccp(
                self._DATA_ROOT / "CME_tracking_validation",
                val_fraction=self.val_fraction,
                seed=self.split_seed,
                split_mode=self.split_mode,
                disk_radius=self.disk_radius,
                gaussian_sigma=self.gaussian_sigma,
                cluster_threshold=self.cluster_threshold,
                use_recon=self.use_recon,
                dominant_annotator=self.dominant_annotator,
                train_kwargs=dict(patch_size=self.train_patch_size,
                                  epoch_length=self.train_epoch_length),
                val_kwargs=dict(patch_size=self.val_patch_size),
            )
        if stage in ('test', 'predict', None):
            self.test_dataset = AnnotatedPytorchDataset(
                self._DATA_ROOT / "CME_tracking_testing",
                mode='test',
            )

    def train_dataloader(self):
        return torch_data.DataLoader(self.train_dataset, batch_size=self.batch_size,
                                     shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return torch_data.DataLoader(self.val_dataset, batch_size=self.batch_size,
                                     shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return torch_data.DataLoader(self.test_dataset, batch_size=self.batch_size,
                                     shuffle=False, num_workers=self.num_workers)

    def predict_dataloader(self):
        return self.test_dataloader()
