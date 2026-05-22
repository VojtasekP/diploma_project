# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Diploma thesis comparing a classic UNet against an equivariant UNet (escnn / SO(2)) for **clathrin-coated pit (CCP) detection and tracking** in SIM microscopy. The pipeline is:

```
SIM image stack  →  CNN segmentation  →  peak-finding detection  →  linking (min-cost flow + untangling)  →  HOTA / DetA / AssA
```

Training uses an infinite synthetic CCP generator (`SyntheticCCPDataset`) by default; a real-data mode loads point-annotated frames from `data/CME_tracking_validation/` and renders disk or Gaussian masks. Test set is `data/CME_tracking_testing/` (012).

## Environment

```bash
uv venv && source .venv/bin/activate         # Python 3.10, deps from pyproject.toml
.venv/bin/python -m <script>                 # always invoke through this Python (escnn etc. installed there)
```

## Common commands

```bash
# Run a single training run (uses train_config dict in src/train.py)
PYTHONPATH=src .venv/bin/python src/train.py

# Launch a wandb sweep (creates the sweep, then runs N agents)
./sweep.sh src/sweep_config/<sweep>.yaml --cuda0 1            # 1 agent on GPU 0
./sweep.sh src/sweep_config/<sweep>.yaml --cuda0 1 --cuda1 1  # one per GPU

# Pull all wandb runs into evaluation/runs.csv
.venv/bin/python evaluation/extract_runs.py

# Generate the LaTeX tables + DetA-vs-size plot from runs.csv
.venv/bin/python evaluation/make_table.py
.venv/bin/python evaluation/plot_deta.py

# Visualise disk vs Gaussian masks for the real-data ROI
.venv/bin/python evaluation/plot_mask_examples.py
```

## Architecture — the parts that span multiple files

**Training entry point: [src/train.py](src/train.py)**
- `train_config` dict + `wandb.init(config=...)` is the single source of truth for hyperparameters; sweep YAMLs just override keys in this dict via wandb's CLI.
- `build_model(model_config: dict)` is module-level so it can be used both for training and for reconstructing a model from a saved checkpoint's `hyper_parameters`. **Always** build models through it.
- `LitUnet` calls `self.save_hyperparameters(ignore=["model"])` and persists `model_config` so new checkpoints are self-describing. Reload with `load_lit_from_checkpoint(path)` or `load_pretrained_into_model(model, path)`.
- `on_validation_epoch_end` runs **fast** detection-only DetA every `val_every` epochs (`_approx_deta` — per-frame Hungarian, no tracking). `on_test_epoch_end` runs the full tracking pipeline via `_track_and_eval` in a background process.

**Data modes — pick via `cfg.data` ∈ {`"synthetic"`, `"real"`}**
- Synthetic: [src/dataset/dataset.py](src/dataset/dataset.py) — `SyntheticCCPLightningDataModule` wraps the infinite `SyntheticCCPDataset` for train, and `AnnotatedPytorchDataset(mode='validation')` for val (full 033 stack with 3-annotator point CSVs).
- Real: [src/dataset/real_ccp.py](src/dataset/real_ccp.py) — `RealCCPLightningDataModule` splits the 033 stack into disjoint train/val frame sets (`split_mode: "random" | "contiguous"`) and renders **disk** (`gaussian_sigma=0`) or **Gaussian** heatmap masks around fused annotator points. Multi-annotator fusion is greedy clustering with `cluster_threshold` (default 5 px) — see `_fuse_annotators`.

**ROI convention.** The training loop, detection, and evaluation all use the same ROI for the real data: `[y=512:768, x=256:512]` (a 256×256 region of the 1024×1024 SIM frame), defined as `CROP` in `train.py` and `CROP_Y_START/CROP_X_START/CROP_SIZE` in `real_ccp.py`. `CCPCenterDataset.stack` returns the **full** 1024×1024 stack (for detection); training patches are cropped from the ROI inside `__getitem__`.

**Detection — [src/detection.py](src/detection.py)**
- `generate_ccp_detections(model, device, images)` runs sigmoid + local-max peak finding frame-by-frame and returns a DataFrame with `(frame, x, y, cls)`. Output coords are in original-image space (not ROI-local), so `CROP` filtering happens downstream.

**Linking & tracking — [src/linking.py](src/linking.py)**
- `LinkingGraph` builds a Min-Cost Flow with detection/birth/death/skip/merge/split nodes solved by OR-tools (SCIP). Followed by `UntanglingGraph` to remove conflicting edges. Inputs/outputs are pandas DataFrames.
- `LinkingParameters` (in `train.py` as `LP`) holds the costs and `maximum_skipped_frames`. Tracking on a non-contiguous frame split is undefined — `on_test_epoch_end` skips val-HOTA in real-data mode unless `frame_indices` are contiguous.

**Metrics — [src/metrics.py](src/metrics.py)**
- `hota(gt, tr, threshold)` returns `HOTA / DetA / AssA`; matching uses Hungarian with `MATCHING_THRESHOLD = 5` px in `train.py`. For val-time per-epoch monitoring, the lightweight `_approx_deta` is used instead (no linking).

**Models**
- [src/models/UNet.py](src/models/UNet.py) — plain UNet, params controlled by `depth` + `start_filters` + `up_mode`.
- [src/models/eUNet.py](src/models/eUNet.py) — escnn-based EquiUNet on `gspaces.rot2dOnR2`. Channels are scaled by `sqrt(k²/basis_dim)` so that param count roughly matches a UNet at the same `start_filters`. Key knobs: `max_rot_order`, `group_order` (-1 = SO(2)), `activation_type` (`fourierbn` / `gated`), `kernel_size`, `conv_sigma`.

## Sweep configs (`src/sweep_config/`)

Naming pattern: `sweep_<data>_<model>_<purpose>.yaml` where `data ∈ {synth, real}`, `model ∈ {unet, equi}`, `purpose ∈ {grid, hparams, pretrain, finetune}`.

- `sweep_synth_unet_grid.yaml`, `sweep_synth_equi_grid.yaml` — 5-seed × 14-`start_filters` synthetic grid (the main thesis comparison).
- `sweep_synth_equi_hparams.yaml` — random hparam search on synthetic (lr/wd/kernel/activation).
- `sweep_synth_equi_pretrain.yaml` — single-run EquiUNet on synthetic; produces the checkpoint consumed by `sweep_real_equi_finetune.yaml`.
- `sweep_real_unet_grid.yaml`, `sweep_real_equi_grid.yaml` — 5-seed from-scratch comparison on real data (contiguous split, Gaussian masks).
- `sweep_real_equi_hparams.yaml` — random hparam search for real-data masks (`gaussian_sigma`, `disk_radius`, `cluster_threshold`) + lr/wd, with hyperband early-termination.
- `sweep_real_equi_finetune.yaml` — fine-tune the synthetic-pretrained EquiUNet on real data.

## Conventions / gotchas

- **Always run Python from the project venv** (`.venv/bin/python …`). `escnn` and `ortools` are not in the system Python.
- **Synthetic vs real normalisation differ.** Synthetic data is zero-mean unit-variance; `AnnotatedPytorchDataset` and `CCPCenterDataset` use per-frame min-max. This is intentional but means fine-tuning a synthetic-pretrained checkpoint on real data has a small input distribution shift.
- **`cfg.run_test=True`** at the end of training kicks off the full HOTA pipeline (linking + untangling). It is expensive — keep it off during hparam search.
- **EquiUNet checkpoints are ~4.5× larger** than UNet at the same param count because escnn caches basis tensors as buffers inside `state_dict`.
- **Checkpoint naming** (set in `train.py`): `{model}_sf{NN}_seed{S}_{data}_{run_name}-best.ckpt`. Self-identifying via `ls` alone; full `model_config` is also inside the ckpt via `save_hyperparameters()`. Lightning's `save_top_k=1` overwrites the file within a run as `val_DetA` improves.
