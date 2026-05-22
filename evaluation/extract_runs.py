"""
Extract all runs from wandb project "vojtasek-petr/UNet-CCP endocytosis/"
and save config + summary metrics to evaluation/runs.csv.

Note: config.yaml must be downloaded per-run because wandb's Python API
does not surface configs for runs that were moved between projects.
"""
import pathlib
import tempfile
import yaml
import wandb
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT = "vojtasek-petr/UNet-CCP endocytosis"
OUT = pathlib.Path(__file__).parent / "runs.csv"

CONFIG_KEYS = [
    # training
    "model", "seed", "lr", "wd", "epochs", "batch_size",
    "dice_weight", "dataset_length", "early_stop_patience", "burn_in",
    "val_every", "link",
    # architecture — shared
    "depth", "start_filters",
    # equiunet-specific
    "max_rot_order", "group_order", "activation_type",
    "fourier_N", "fourier_function", "kernel_size", "conv_sigma",
    # unet-specific
    "up_mode",
]

SUMMARY_KEYS = [
    "total_params", "epoch",
    # validation (3 annotators)
    "val_DetA",   "val_DetA_ann1",   "val_DetA_ann2",   "val_DetA_ann3",
    "val_Recall", "val_Recall_ann1", "val_Recall_ann2", "val_Recall_ann3",
    "val_HOTA",   "val_HOTA_ann1",   "val_HOTA_ann2",   "val_HOTA_ann3",
    "val_AssA",   "val_AssA_ann1",   "val_AssA_ann2",   "val_AssA_ann3",
    # test (1 annotator logged so far)
    "test_DetA",  "test_DetA_ann1",
    "test_HOTA",  "test_HOTA_ann1",
    "test_AssA",  "test_AssA_ann1",
]


def _extract_run(run) -> dict:
    row = {
        "run_id":     run.id,
        "run_name":   run.name,
        "state":      run.state,
        "sweep_id":   run.sweep.id if run.sweep else None,
        "created_at": run.created_at,
    }

    # Config: download config.yaml and unpack {'value': ...} structure
    try:
        with tempfile.TemporaryDirectory() as tmp:
            run.file("config.yaml").download(root=tmp, replace=True)
            with open(pathlib.Path(tmp) / "config.yaml") as f:
                raw_cfg = yaml.safe_load(f)
        for k in CONFIG_KEYS:
            entry = raw_cfg.get(k)
            row[k] = entry["value"] if isinstance(entry, dict) and "value" in entry else entry
    except Exception:
        for k in CONFIG_KEYS:
            row[k] = None

    # Summary metrics
    summary = run.summary
    for k in SUMMARY_KEYS:
        row[k] = summary.get(k)

    return row


api = wandb.Api()
runs = list(api.runs(PROJECT))
print(f"Found {len(runs)} runs in '{PROJECT}'")

rows = []
with ThreadPoolExecutor(max_workers=16) as pool:
    futures = {pool.submit(_extract_run, r): r.name for r in runs}
    for i, fut in enumerate(as_completed(futures), 1):
        rows.append(fut.result())
        if i % 20 == 0:
            print(f"  {i}/{len(runs)} done …")

df = pd.DataFrame(rows).sort_values("created_at").reset_index(drop=True)
df.to_csv(OUT, index=False)
print(f"\nSaved {len(df)} rows → {OUT}")
print(df[["run_name", "model", "start_filters", "seed", "val_DetA", "state"]].to_string())
