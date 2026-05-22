"""
Two LaTeX tables side by side in concept:
  1) deta_table.tex  — val / test DetA per approximate parameter count.
  2) arch_table.tex  — exact parameter count and per-layer channel widths
                       for UNet and EquiUNet.
"""
import sys
import math
import pathlib

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))

import escnn.nn as enn
from escnn import gspaces
from models.utils import FourierPointwiseInnerBn

DEPTH         = 3
KERNEL_EQUI   = 5
KERNEL_UNET   = 3
MAX_ROT_ORDER = 1
FOURIER_N     = 16

_r2  = gspaces.rot2dOnR2(N=-1, maximum_frequency=MAX_ROT_ORDER)
_act = FourierPointwiseInnerBn(_r2, channels=1,
                               irreps=_r2.fibergroup.bl_irreps(MAX_ROT_ORDER),
                               function="p_relu", N=FOURIER_N)
_basis_dim = float(enn.R2Conv(_act.in_type, _act.in_type,
                              kernel_size=KERNEL_EQUI, sigma=0.6, bias=False)
                   .basisexpansion.dimension())
EQUI_SCALE = math.sqrt(KERNEL_UNET ** 2 / _basis_dim)
del _act


def unet_channels(sf: int) -> list[int]:
    return [sf * (2 ** i) for i in range(DEPTH + 1)]


def equiunet_channels(sf: int) -> list[int]:
    return [max(1, round(sf * (2 ** i) * EQUI_SCALE)) for i in range(DEPTH + 1)]


def fmt_params(p: float) -> str:
    return f"{p/1e6:.2f}M" if p >= 1e6 else f"{p/1e3:.1f}k"


def fmt_metric(mean: float, std: float) -> str:
    return f"{mean:.4f} $\\pm$ {std:.4f}"


def fmt_channels(chans: list[int]) -> str:
    return r"$" + r"{\to}".join(str(c) for c in chans) + r"$"


df = pd.read_csv(HERE / "runs.csv")

agg = (df.groupby(["model", "start_filters"])
         .agg(params=("total_params", "mean"),
              val_mean=("val_DetA", "mean"), val_std=("val_DetA", "std"),
              test_mean=("test_DetA", "mean"), test_std=("test_DetA", "std"))
         .reset_index())

unet_a = agg[agg.model == "unet"].set_index("start_filters").sort_index()
equi_a = agg[agg.model == "equiunet"].set_index("start_filters").sort_index()
sfs    = sorted(set(unet_a.index) & set(equi_a.index))


# --- Table 1: DetA per approximate param count -------------------------------
deta_rows = []
for sf in sfs:
    approx_p = (unet_a.loc[sf, "params"] + equi_a.loc[sf, "params"]) / 2
    deta_rows.append([
        f"$\\approx$ {fmt_params(approx_p)}",
        fmt_metric(unet_a.loc[sf, "val_mean"],  unet_a.loc[sf, "val_std"]),
        fmt_metric(unet_a.loc[sf, "test_mean"], unet_a.loc[sf, "test_std"]),
        fmt_metric(equi_a.loc[sf, "val_mean"],  equi_a.loc[sf, "val_std"]),
        fmt_metric(equi_a.loc[sf, "test_mean"], equi_a.loc[sf, "test_std"]),
    ])

deta_tex = [
    r"\begin{table}[ht]",
    r"\centering",
    r"\caption{Detection accuracy per approximate parameter budget. "
    r"Each row reports val and test DetA (mean $\pm$ std over 5 seeds) "
    r"for UNet and EquiUNet trained at matched model size. "
    r"The parameter count is averaged across the two models "
    r"(exact counts in Table~\ref{tab:arch_comparison}).}",
    r"\label{tab:deta_results}",
    r"\begin{tabular}{l ll ll}",
    r"\toprule",
    r" & \multicolumn{2}{c}{\textbf{UNet}} "
    r"& \multicolumn{2}{c}{\textbf{EquiUNet}} \\",
    r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
    r"\textbf{Params} & Val DetA & Test DetA & Val DetA & Test DetA \\",
    r"\midrule",
    *[" & ".join(row) + r" \\" for row in deta_rows],
    r"\bottomrule",
    r"\end{tabular}",
    r"\end{table}",
]


# --- Table 2: architecture (channels + exact params) -------------------------
arch_rows = []
for sf in sfs:
    arch_rows.append([
        fmt_channels(unet_channels(sf)),
        fmt_params(unet_a.loc[sf, "params"]),
        fmt_channels(equiunet_channels(sf)),
        fmt_params(equi_a.loc[sf, "params"]),
    ])

arch_tex = [
    r"\begin{table}[ht]",
    r"\centering",
    r"\caption{Per-layer channel widths and total parameter count for UNet "
    r"and EquiUNet (depth=3, four encoder levels). "
    r"Rows are matched by model size; reading across each row compares the "
    r"two architectures at a comparable parameter budget.}",
    r"\label{tab:arch_comparison}",
    r"\begin{tabular}{ll ll}",
    r"\toprule",
    r"\multicolumn{2}{c}{\textbf{UNet}} & "
    r"\multicolumn{2}{c}{\textbf{EquiUNet}} \\",
    r"\cmidrule(lr){1-2}\cmidrule(lr){3-4}",
    r"Channels & Params & Channels & Params \\",
    r"\midrule",
    *[" & ".join(row) + r" \\" for row in arch_rows],
    r"\bottomrule",
    r"\end{tabular}",
    r"\end{table}",
]


# --- write -------------------------------------------------------------------
out_deta = HERE / "deta_table.tex"
out_arch = HERE / "arch_table.tex"
out_deta.write_text("\n".join(deta_tex))
out_arch.write_text("\n".join(arch_tex))
print(out_deta.read_text())
print()
print(out_arch.read_text())
print(f"\nSaved -> {out_deta}")
print(f"Saved -> {out_arch}")
