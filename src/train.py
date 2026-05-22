import sys
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import lightning as L
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from detection import generate_ccp_detections
from models.UNet import UNet
from models.eUNet import EquiUNet
from parameters import LinkingParameters






MATCHING_THRESHOLD = 5  # pixels

LP = LinkingParameters(
    birth_death_cost=5,
    edge_removal_cost=10,
    feature_cost_multiplier=1,
    maximum_distance=7.5,
    maximum_skipped_frames=1,
    minimum_length=5,
)

CROP = dict(x_min=256, x_max=512, y_min=512, y_max=768)


def _approx_deta(det_df: pd.DataFrame, csv_paths: list[str], threshold: int,
                 eval_frames: set[int] | None = None) -> dict:
    """Fast detection-only DetA: per-frame Hungarian matching, no tracking.

    If ``eval_frames`` is given, both detections and GT are restricted to that
    frame set (use to evaluate on a held-out split of a single stack).
    """
    if eval_frames is not None:
        det_df = det_df[det_df['frame'].isin(eval_frames)]
    per_ann = []
    for csv_path in csv_paths:
        gt = pd.read_csv(csv_path)
        if eval_frames is not None:
            gt = gt[gt['frame'].isin(eval_frames)]
        all_frames = set(gt['frame'].unique()) | set(det_df['frame'].unique())
        tp = fp = fn = 0
        for frame in all_frames:
            gt_xy  = gt[gt.frame == frame][['x', 'y']].values
            det_xy = det_df[det_df.frame == frame][['x', 'y']].values
            if len(gt_xy) == 0 and len(det_xy) == 0:
                continue
            elif len(gt_xy) == 0:
                fp += len(det_xy)
            elif len(det_xy) == 0:
                fn += len(gt_xy)
            else:
                D = cdist(det_xy, gt_xy)
                ri, ci = linear_sum_assignment(D)
                n_match = int((D[ri, ci] < threshold).sum())
                tp += n_match
                fp += len(det_xy) - n_match
                fn += len(gt_xy)  - n_match
        rec  = tp / (tp + fn)       if (tp + fn) > 0 else 0.0
        deta = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        per_ann.append({"DetA": float(deta), "Recall": float(rec)})
    return {
        "DetA":   float(np.mean([a["DetA"]   for a in per_ann])),
        "Recall": float(np.mean([a["Recall"] for a in per_ann])),
        "per_annotator": per_ann,
    }


def _track_and_eval(det_records: list[dict], csv_paths: list[str],
                    lp: dict, threshold: int, src_dir: str,
                    eval_frames: set[int] | None = None) -> Optional[dict]:
    """Background worker: linking → untangling → HOTA. No GPU needed.

    If ``eval_frames`` is given, GT trajectories are restricted to those frames
    (matches the filter already applied to ``det_records`` by the caller).
    """
    sys.path.insert(0, src_dir)
    import pandas as _pd
    import numpy as _np
    from scipy import spatial as _spatial
    import linking as _linking
    import metrics as _metrics

    det = _pd.DataFrame.from_records(det_records)

    def dist_fn(a, b):
        euclid = _spatial.distance.cdist(a[["x", "y"]].values, b[["x", "y"]].values)
        si_diff = _np.square(a["cls"].values[:, None] - b["cls"].values[None, :])
        return euclid + lp["feature_cost_multiplier"] * si_diff

    lg = _linking.LinkingGraph(det, dist_fn, lp["maximum_distance"], lp["birth_death_cost"],
                               max_skipped_frames=lp["maximum_skipped_frames"])
    if lg.solve() != lg.solver.OPTIMAL:
        return None
    tracklets = lg.get_result()

    ug = _linking.UntanglingGraph(tracklets, lp["edge_removal_cost"])
    if ug.solve() != ug.solver.OPTIMAL:
        return None
    trajectories = ug.get_result(det)

    filtered = trajectories.groupby("particle").filter(
        lambda t: t.frame.count() >= lp["minimum_length"]
    )

    per_ann = []
    for csv_path in csv_paths:
        gt = _pd.read_csv(csv_path).rename(columns={"track_id": "particle"})
        if eval_frames is not None:
            gt = gt[gt.frame.isin(eval_frames)]
        r = _metrics.hota(gt, filtered, threshold)
        per_ann.append({"HOTA": float(r["HOTA"]), "DetA": float(r["DetA"]), "AssA": float(r["AssA"])})

    return {
        "HOTA": float(_np.mean([r["HOTA"] for r in per_ann])),
        "DetA": float(_np.mean([r["DetA"] for r in per_ann])),
        "AssA": float(_np.mean([r["AssA"] for r in per_ann])),
        "per_annotator": per_ann,
    }


class BCEDiceLoss(nn.Module):
    def __init__(self, dice_weight: float = 1.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = self.bce(logits, targets)

        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(2, 3))
        dice = 1 - (2 * intersection + 1) / (probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + 1)
        dice = dice.mean()

        return bce + self.dice_weight * dice


def build_model(mc: dict) -> torch.nn.Module:
    """Build a model from a flat config dict (used both for training and
    for reconstructing a model from a saved checkpoint's hparams)."""
    if mc["model"] == "unet":
        return UNet(in_channels=1, out_channels=1,
                    depth=mc["depth"], start_filters=mc["start_filters"],
                    up_mode=mc["up_mode"])
    if mc["model"] == "equiunet":
        return EquiUNet(in_channels=1, n_classes=1,
                        depth=mc["depth"], start_filters=mc["start_filters"],
                        max_rot_order=mc["max_rot_order"], group_order=mc["group_order"],
                        activation_type=mc["activation_type"], fourier_N=mc["fourier_N"],
                        fourier_function=mc["fourier_function"],
                        kernel_size=mc["kernel_size"], conv_sigma=mc["conv_sigma"])
    raise ValueError(f"Unknown model '{mc['model']}'")


def load_lit_from_checkpoint(path: str) -> "LitUnet":
    """Rebuild a LitUnet from a checkpoint whose hparams include ``model_config``."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    mc = ckpt["hyper_parameters"]["model_config"]
    return LitUnet.load_from_checkpoint(path, model=build_model(mc))


def load_pretrained_into_model(model: torch.nn.Module, ckpt_path: str) -> None:
    """Load ``model.*`` weights from a Lightning checkpoint into ``model``.

    Works with both new (``self.save_hyperparameters()``) and old checkpoints
    where the only thing we can trust is the ``state_dict`` having a
    ``model.`` prefix from being wrapped in ``LitUnet``.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd   = ckpt.get("state_dict", ckpt)
    # Old Lightning ckpts wrap weights under ``model.``; the archive format
    # has already stripped that prefix. Accept either.
    if any(k.startswith("model.") for k in sd):
        sd = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"  warn: {len(missing)} missing, {len(unexpected)} unexpected keys "
              f"when loading {ckpt_path}")


class LitUnet(L.LightningModule):
    def __init__(self, model: torch.nn.Module, lr: float = 1e-4, wd: float = 1e-3,
                 epochs: int = 30, burn_in: int = 30, val_every: int = 1,
                 model_config: dict | None = None):
        super().__init__()
        # Persist everything except the live model graph; ``model_config`` is
        # enough to reconstruct the architecture via ``build_model``.
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.lr = lr
        self.wd = wd
        self.epochs = epochs
        self.burn_in = burn_in
        self.val_every = val_every
        self.loss_fn = nn.BCEWithLogitsLoss()
        self._last_val_deta: float = 0.0
        self._last_val_deta_per_ann: list = []



    def forward(self, x):
        return self.model(x)  # raw logits; detection.py applies sigmoid

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.model(inputs)  # raw logits for loss
        loss = self.loss_fn(outputs, targets)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, *_):
        pass  # HOTA/DetA computed once per epoch in on_validation_epoch_end

    def on_validation_epoch_end(self):
        if self.current_epoch < self.burn_in or self.current_epoch % self.val_every != 0:
            self.log('val_DetA', self._last_val_deta, prog_bar=True)
            return

        torch.cuda.empty_cache()
        ds = self.trainer.datamodule.val_dataset  # type: ignore[union-attr]
        # In real-data mode, val_dataset only owns a subset of frames; the CNN
        # runs over the whole stack but the metric is scored on val frames only.
        eval_frames = set(ds.frame_indices) if hasattr(ds, 'frame_indices') else None
        n_eval = len(eval_frames) if eval_frames is not None else len(ds.stack)
        eval_str = f" (scoring on {n_eval} val frames)" if eval_frames is not None else ""
        print(f"\nEpoch {self.current_epoch} — running detection on {len(ds.stack)} frames{eval_str} ...")
        det_df = generate_ccp_detections(self, self.device, ds.stack)
        torch.cuda.empty_cache()

        det_crop = det_df[
            (det_df.x >= CROP['x_min']) & (det_df.x <= CROP['x_max']) &
            (det_df.y >= CROP['y_min']) & (det_df.y <= CROP['y_max'])
        ].reset_index(drop=True)
        det_crop_eval = (det_crop[det_crop.frame.isin(eval_frames)] if eval_frames is not None else det_crop)

        data_dir = self.trainer.datamodule._DATA_ROOT / "CME_tracking_validation"  # type: ignore[union-attr]
        csv_paths = [str(p) for p in sorted(data_dir.glob('*-annotations_*.csv'))]
        result = _approx_deta(det_crop, csv_paths, MATCHING_THRESHOLD, eval_frames=eval_frames)

        self._last_val_deta = result['DetA']
        self._last_val_deta_per_ann = result['per_annotator']

        ann_str = "  ".join(f"Ann{i+1}: {a['DetA']:.4f}" for i, a in enumerate(result['per_annotator']))
        print(f"  {len(det_df)} total, {len(det_crop_eval)} in crop on {n_eval} val frames — "
              f"approx DetA: {result['DetA']:.4f}  Recall: {result['Recall']:.4f}  [{ann_str}]")

        self.log('val_DetA',   self._last_val_deta, prog_bar=True)
        self.log('val_Recall', result['Recall'],    prog_bar=True)
        for i, a in enumerate(self._last_val_deta_per_ann):
            self.log(f'val_DetA_ann{i+1}',   a['DetA'])
            self.log(f'val_Recall_ann{i+1}', a['Recall'])

    def test_step(self, *_):
        pass  # detections computed in on_test_epoch_end

    def on_test_epoch_end(self):
        dm = self.trainer.datamodule  # type: ignore[union-attr]
        src_dir = str(Path(__file__).parent)

        def _run_hota(stack, data_dir, glob_pattern, prefix,
                      eval_frames: set[int] | None = None):
            eval_str = (f" (scoring on {len(eval_frames)} val frames only)"
                        if eval_frames is not None else "")
            print(f"\n{prefix} — running detection on {len(stack)} frames{eval_str} ...")
            det_df = generate_ccp_detections(self, self.device, stack)
            det_crop = det_df[
                (det_df.x >= CROP['x_min']) & (det_df.x <= CROP['x_max']) &
                (det_df.y >= CROP['y_min']) & (det_df.y <= CROP['y_max'])
            ].reset_index(drop=True)
            if eval_frames is not None:
                det_crop = det_crop[det_crop.frame.isin(eval_frames)].reset_index(drop=True)
            print(f"  {len(det_df)} total, {len(det_crop)} in crop "
                  f"({len(eval_frames) if eval_frames else len(stack)} frames) — running linking ...")
            csv_paths = [str(p) for p in sorted(data_dir.glob(glob_pattern))]
            result = _track_and_eval(det_crop.to_dict('records'), csv_paths,
                                     asdict(LP), MATCHING_THRESHOLD, src_dir,
                                     eval_frames=eval_frames)
            if result is None:
                print(f"  {prefix}: linking did not converge.")
                return
            per_ann = result.get("per_annotator", [])
            ann_str = "  ".join(f"Ann{i+1}: DetA={a['DetA']:.4f}" for i, a in enumerate(per_ann))
            print(f"  {prefix} — HOTA: {result['HOTA']:.4f}  DetA: {result['DetA']:.4f}  AssA: {result['AssA']:.4f}")
            if ann_str:
                print(f"    {ann_str}")
            tag = prefix.lower()
            self.log(f'{tag}_HOTA', result['HOTA'])
            self.log(f'{tag}_DetA', result['DetA'])
            self.log(f'{tag}_AssA', result['AssA'])
            for i, a in enumerate(per_ann):
                self.log(f'{tag}_DetA_ann{i+1}', a['DetA'])
                self.log(f'{tag}_HOTA_ann{i+1}', a['HOTA'])
                self.log(f'{tag}_AssA_ann{i+1}', a['AssA'])

        # Real-data mode: val_dataset owns a frame subset. Run val HOTA only
        # if those frames are contiguous (tracking continuity required).
        val_ds = dm.val_dataset
        if hasattr(val_ds, 'frame_indices'):
            fi = sorted(val_ds.frame_indices)
            if fi and fi == list(range(fi[0], fi[-1] + 1)):
                _run_hota(val_ds.stack,
                          dm._DATA_ROOT / "CME_tracking_validation",
                          '*-annotations_*.csv', 'val', eval_frames=set(fi))
            else:
                print("\nval HOTA skipped (non-contiguous real-data val split)")
        else:
            _run_hota(val_ds.stack,
                      dm._DATA_ROOT / "CME_tracking_validation",
                      '*-annotations_*.csv', 'val')
        _run_hota(dm.test_dataset.stack,
                  dm._DATA_ROOT / "CME_tracking_testing",   '*-annotations.csv',   'test')

    def configure_optimizers(self):  # type: ignore[override]
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        return {'optimizer': optimizer}
    

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    import wandb
    from dataset.dataset import SyntheticCCPLightningDataModule
    from dataset.real_ccp import RealCCPLightningDataModule
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    from lightning.pytorch.loggers import WandbLogger

    train_config = {
        # --- training ---
        "lr":                   1e-3,
        "wd":                   1e-3,
        "epochs":               300,
        "batch_size":           16,
        "dice_weight":          0,
        "dataset_length":       1600,
        "early_stop_patience":  30,
        "burn_in":              30,
        "val_every":            2,
        "seed":                 0,
        "run_test":             True,
        "pretrained_ckpt":      None,   # path to Lightning ckpt; init weights then fine-tune
        # --- data ---
        "data":                 "synthetic",   # "synthetic" or "real"
        # real-mode only:
        "real_val_fraction":    0.2,
        "real_split_seed":      0,
        "real_split_mode":      "random",      # "random" or "contiguous"
        "real_disk_radius":     3.0,
        "real_gaussian_sigma":  0.0,           # > 0 → Gaussian heatmap, else disk
        "real_cluster_threshold": 5.0,
        "real_train_patch_size": 128,
        "real_dominant_annotator": 0,          # 0 = random per frame; 1, 2, 3 = that annotator dominates
        # --- model ---
        "model":                "equiunet",
        "depth":                3,
        "start_filters":        16,
        # unet
        "up_mode":              "nearest",
        # equiunet
        "max_rot_order":        2,
        "group_order":          -1,
        "activation_type":      "fourierbn",
        "fourier_N":            16,
        "fourier_function":     "p_relu",
        "kernel_size":          3,
        "conv_sigma":           0.6,
    }

    wandb.init(project="unet-ccp", config=train_config)
    cfg = wandb.config

    set_seed(cfg.seed)
    torch.set_float32_matmul_precision('high')

    if cfg.data == "synthetic":
        data_module = SyntheticCCPLightningDataModule(
            batch_size=cfg.batch_size,
            num_workers=16,
            length=cfg.dataset_length,
        )
    elif cfg.data == "real":
        data_module = RealCCPLightningDataModule(
            batch_size=cfg.batch_size,
            num_workers=16,
            val_fraction=cfg.real_val_fraction,
            split_seed=cfg.real_split_seed,
            split_mode=cfg.real_split_mode,
            train_patch_size=cfg.real_train_patch_size,
            train_epoch_length=cfg.dataset_length,
            disk_radius=cfg.real_disk_radius,
            gaussian_sigma=cfg.real_gaussian_sigma,
            cluster_threshold=cfg.real_cluster_threshold,
            dominant_annotator=cfg.real_dominant_annotator,
        )
    else:
        raise ValueError(f"Unknown data source '{cfg.data}' (expected 'synthetic' or 'real')")

    model_config = {
        "model":         cfg.model,
        "depth":         cfg.depth,
        "start_filters": cfg.start_filters,
        "up_mode":       cfg.up_mode,
        "max_rot_order": cfg.max_rot_order,
        "group_order":   cfg.group_order,
        "activation_type": cfg.activation_type,
        "fourier_N":     cfg.fourier_N,
        "fourier_function": cfg.fourier_function,
        "kernel_size":   cfg.kernel_size,
        "conv_sigma":    cfg.conv_sigma,
    }
    model = build_model(model_config)

    pre = cfg.get("pretrained_ckpt") if hasattr(cfg, "get") else getattr(cfg, "pretrained_ckpt", None)
    if pre:
        print(f"Loading pretrained weights from {pre}")
        load_pretrained_into_model(model, pre)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    wandb.log({"total_params": total_params})

    lit_model = LitUnet(
        model=model,
        lr=cfg.lr,
        wd=cfg.wd,
        epochs=cfg.epochs,
        burn_in=cfg.burn_in,
        val_every=cfg.val_every,
        model_config=model_config,
    )

    logger = WandbLogger()
    # Self-describing ckpt name: model, size, seed, data source, run name.
    # Lets `ls checkpoints/` answer "what do I have" without opening files.
    ckpt_filename = (f"{cfg.model}_sf{int(cfg.start_filters):02d}_seed{int(cfg.seed)}"
                     f"_{cfg.data}_{wandb.run.name}-best")
    checkpoint = ModelCheckpoint(monitor='val_DetA', mode='max', save_top_k=1,
                                 filename=ckpt_filename,
                                 dirpath='checkpoints/')
    early_stop = EarlyStopping(monitor='val_DetA', patience=cfg.early_stop_patience, mode='max')
    trainer = L.Trainer(
        max_epochs=cfg.epochs,
        accelerator='auto',
        devices=1,
        precision='bf16-mixed',
        callbacks=[checkpoint, early_stop],
        logger=logger,
    )
    trainer.fit(lit_model, datamodule=data_module)
    if cfg.run_test:
        trainer.test(lit_model, datamodule=data_module, ckpt_path='best')


if __name__ == "__main__":
    main()