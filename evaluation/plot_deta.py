"""
Detection accuracy vs model size: per-seed points, mean line, ±1 std band.
Two panels: val_DetA / test_DetA. x-axis is total_params (log scale).
"""
import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

HERE = pathlib.Path(__file__).parent
df = pd.read_csv(HERE / "runs.csv")

MODELS  = ["equiunet", "unet"]
METRICS = ["val_DetA", "test_DetA"]
TITLES  = {"val_DetA": "Val DetA", "test_DetA": "Test DetA"}
COLORS  = {"equiunet": "#4C72B0", "unet": "#DD8452"}
LABELS  = {"equiunet": "EquiUNet", "unet": "UNet"}

agg = (df.groupby(["model", "start_filters"])
         .agg(params=("total_params", "mean"),
              **{m: (m, list) for m in METRICS})
         .reset_index())

all_vals = df[METRICS].to_numpy().ravel()
all_vals = all_vals[~np.isnan(all_vals)]
y_lo = max(0.0, all_vals.min() - 0.01)
y_hi = min(1.0, all_vals.max() + 0.01)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

for ax, metric in zip(axes, METRICS):
    for model in MODELS:
        sub = agg[agg["model"] == model].sort_values("params")
        x   = sub["params"].to_numpy()
        runs = [np.asarray(v, dtype=float) for v in sub[metric]]
        runs = [v[~np.isnan(v)] for v in runs]
        mean = np.array([v.mean() if len(v) else np.nan for v in runs])
        std  = np.array([v.std(ddof=1) if len(v) > 1 else 0.0 for v in runs])

        # individual seed points (jittered slightly in log-x for legibility)
        for xi, vs in zip(x, runs):
            jitter = xi * (1 + 0.01 * np.random.uniform(-1, 1, len(vs)))
            ax.scatter(jitter, vs, s=12, color=COLORS[model],
                       alpha=0.35, edgecolor="none", zorder=2)

        ax.fill_between(x, mean - std, mean + std,
                        color=COLORS[model], alpha=0.15, zorder=1)
        ax.plot(x, mean, color=COLORS[model], linewidth=1.8,
                marker="o", markersize=4, label=LABELS[model], zorder=3)

    ax.set_xscale("log")
    ax.set_title(TITLES[metric], fontsize=12, fontweight="bold")
    ax.set_xlabel("Parameters", fontsize=10)
    ax.set_ylim(y_lo, y_hi)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    tick_params = sorted(agg.groupby("start_filters")["params"].mean().values)
    fmt = lambda v: f"{v/1e3:.0f}k" if v < 1e6 else f"{v/1e6:.2f}M"
    ax.set_xticks(tick_params)
    ax.set_xticklabels([fmt(v) for v in tick_params], rotation=45, ha="right")
    ax.xaxis.set_minor_locator(mticker.NullLocator())

    ax.grid(which="major", axis="y", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.legend(fontsize=9, loc="lower right")

axes[0].set_ylabel("DetA", fontsize=10)
fig.suptitle("Detection accuracy vs model size  (5 seeds per point, mean ±1 std)",
             fontsize=12)
fig.tight_layout()

out = HERE / "deta_by_start_filters.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
plt.show()
