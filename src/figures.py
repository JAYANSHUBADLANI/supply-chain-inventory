"""Charts for the README and the case write-up.

Every figure is generated from the same tables the numbers in the write-up come from, so
a chart and a claim cannot drift apart between runs.
"""

from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INK = "#1b2733"
MUTED = "#7c8b9a"
ACCENT = "#c05621"
COOL = "#2c5f8a"
WARM = "#b8873a"
GRID = "#dfe4e9"

SEGMENT_ORDER = ["AX", "BX", "BZ", "CZ"]
POLICY_LABELS = {
    "segmented": "Segmented (combined variance)",
    "demand_only": "Demand variance only",
    "flat_baseline": "Flat 50% baseline",
}
POLICY_COLOURS = {"segmented": COOL, "demand_only": ACCENT, "flat_baseline": MUTED}


def _style(ax, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_title(title, color=INK, fontsize=11, loc="left", pad=12)
    ax.set_xlabel(xlabel, color=INK, fontsize=9)
    ax.set_ylabel(ylabel, color=INK, fontsize=9)
    ax.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    return path


def leadtime_by_mode(orders: pd.DataFrame, path: Path) -> Path:
    """Realised lead time per shipping mode against what each mode promised."""
    modes = ["Same Day", "First Class", "Second Class", "Standard Class"]
    present = [m for m in modes if m in set(orders["Shipping Mode"])]
    fig, axes = plt.subplots(1, len(present), figsize=(11, 3.0), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, mode in zip(axes, present):
        block = orders.loc[orders["Shipping Mode"] == mode]
        values = block["Days for shipping (real)"]
        scheduled = block["Days for shipment (scheduled)"].iloc[0]
        counts = values.value_counts(normalize=True).sort_index()
        ax.bar(counts.index, counts.values, color=COOL, width=0.72)
        ax.axvline(scheduled, color=ACCENT, linewidth=1.6, linestyle="--")
        _style(ax, f"{mode}\npromised {scheduled:.0f}d, sd {values.std():.2f}", "days to ship")
        ax.set_xlim(-0.7, 6.7)
        ax.set_xticks(range(0, 7))

    axes[0].set_ylabel("share of orders", color=INK, fontsize=9)
    return _save(fig, path)


def variance_decomposition(decomposition: pd.DataFrame, path: Path) -> Path:
    """Share of safety stock variance coming from lead time against demand, by segment."""
    frame = decomposition.reindex([s for s in SEGMENT_ORDER if s in decomposition.index])
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    y = np.arange(len(frame))
    lead = frame["median_leadtime_share"].values
    demand = frame["median_demand_share"].values

    ax.barh(y, lead, color=COOL, label="lead-time variance", height=0.62)
    ax.barh(y, demand, left=lead, color=WARM, label="demand variance", height=0.62)
    for i, (l, units) in enumerate(zip(lead, frame["units"].values)):
        ax.text(0.012, i, f"{l:.0%}", va="center", ha="left", color="white", fontsize=9)
        ax.text(1.02, i, f"n={int(units)}", va="center", ha="left", color=MUTED, fontsize=8)

    ax.set_yticks(y)
    ax.set_yticklabels(frame.index)
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    _style(ax, "Which uncertainty drives the buffer flips across the catalogue",
           "share of demand-over-lead-time variance")
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, -0.24))
    return _save(fig, path)


def service_level_curve(curve: pd.DataFrame, segment: str, path: Path) -> Path:
    """Safety stock and its holding cost against the target, for one representative unit."""
    block = curve[curve["segment"] == segment].sort_values("cycle_service_level")
    unit = block["planning_unit"].iloc[0]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    x = block["cycle_service_level"] * 100

    ax.plot(x, block["safety_stock_units"], color=COOL, linewidth=2.0, marker="o", markersize=4)
    _style(ax, f"Cost of service: {unit} ({segment})",
           "target cycle service level (%)", "safety stock (units)")

    twin = ax.twinx()
    twin.plot(x, block["safety_stock_holding_cost"], color=ACCENT, linewidth=1.6,
              linestyle="--", marker="s", markersize=3)
    twin.set_ylabel("annual holding cost of buffer", color=ACCENT, fontsize=9)
    twin.tick_params(colors=ACCENT, labelsize=8)
    for spine in ("top", "left"):
        twin.spines[spine].set_visible(False)
    twin.spines["right"].set_color(GRID)
    twin.grid(False)

    knee = block[block["cycle_service_level"] == 0.98]
    if len(knee):
        ax.axvline(98, color=MUTED, linewidth=1.0, linestyle=":")
        ax.annotate("marginal cost turns", xy=(98, knee["safety_stock_units"].iloc[0]),
                    xytext=(86, knee["safety_stock_units"].iloc[0] * 1.02),
                    color=MUTED, fontsize=8)
    return _save(fig, path)


def normal_versus_empirical(comparison: pd.DataFrame, path: Path) -> Path:
    """Closed form safety stock against the buffer the observed distribution needs."""
    segments = list(comparison["segment"].unique())
    fig, axes = plt.subplots(1, len(segments), figsize=(10.5, 3.8))
    axes = np.atleast_1d(axes)

    for ax, segment in zip(axes, segments):
        block = comparison[comparison["segment"] == segment].sort_values("cycle_service_level")
        x = block["cycle_service_level"] * 100
        ax.plot(x, block["normal_safety_stock"], color=COOL, linewidth=2.0,
                marker="o", markersize=4, label="normal approximation")
        ax.plot(x, block["empirical_safety_stock"], color=ACCENT, linewidth=2.0,
                marker="s", markersize=4, label="empirical (bootstrap)")
        skew = block["empirical_skewness"].median()
        kurt = block["empirical_excess_kurtosis"].median()
        _style(ax, f"{segment}: {block['planning_unit'].iloc[0][:34]}\n"
                   f"skew {skew:+.2f}, excess kurtosis {kurt:+.2f}",
               "target cycle service level (%)", "safety stock (units)")
        ax.legend(frameon=False, fontsize=8, loc="upper left")

    return _save(fig, path)


def leadtime_sensitivity(sensitivity: pd.DataFrame, path: Path) -> Path:
    """Total safety stock against the assumed lead-time variability."""
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    x = sensitivity["leadtime_std_days"]
    ax.plot(x, sensitivity["total_safety_stock_units"], color=COOL, linewidth=2.0,
            marker="o", markersize=4)

    base = sensitivity[sensitivity["sigma_multiplier"] == 1.0]
    if len(base):
        ax.axvline(float(base["leadtime_std_days"].iloc[0]), color=ACCENT,
                   linewidth=1.4, linestyle="--")
        ax.annotate("proxy value used", xy=(float(base["leadtime_std_days"].iloc[0]),
                                            float(base["total_safety_stock_units"].iloc[0])),
                    xytext=(float(base["leadtime_std_days"].iloc[0]) * 1.05,
                            float(base["total_safety_stock_units"].iloc[0]) * 0.72),
                    color=ACCENT, fontsize=8)
    zero = sensitivity[sensitivity["sigma_multiplier"] == 0.0]
    if len(zero):
        ax.annotate("fixed lead time assumption",
                    xy=(0, float(zero["total_safety_stock_units"].iloc[0])),
                    xytext=(0.15, float(zero["total_safety_stock_units"].iloc[0]) * 0.80),
                    color=MUTED, fontsize=8)

    _style(ax, "Safety stock across the assumed lead-time variability",
           "lead-time standard deviation (days)", "total safety stock (units)")
    return _save(fig, path)


def backtest_service_cost(summary: pd.DataFrame, path: Path) -> Path:
    """Realised service against what each policy cost to run on the held-out window."""
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    order = [p for p in ("demand_only", "flat_baseline", "segmented") if p in summary.index]
    colours = [POLICY_COLOURS[p] for p in order]
    labels = [POLICY_LABELS[p].replace(" (", "\n(") for p in order]

    lost = summary.loc[order, "units_lost"].values
    axes[0].bar(range(len(order)), lost, color=colours, width=0.6)
    for i, v in enumerate(lost):
        axes[0].text(i, v, f"{v:,.0f}", ha="center", va="bottom", color=INK, fontsize=9)
    axes[0].set_xticks(range(len(order)))
    axes[0].set_xticklabels(labels, fontsize=8)
    _style(axes[0], "Units of demand lost on the held-out window", "", "units lost")

    cost = summary.loc[order, "holding_cost"].values
    axes[1].bar(range(len(order)), cost, color=colours, width=0.6)
    for i, v in enumerate(cost):
        axes[1].text(i, v, f"{v:,.0f}", ha="center", va="bottom", color=INK, fontsize=9)
    axes[1].set_xticks(range(len(order)))
    axes[1].set_xticklabels(labels, fontsize=8)
    _style(axes[1], "Holding cost incurred", "", "holding cost")

    return _save(fig, path)


def demand_series(panel: pd.DataFrame, units: List[str], path: Path) -> Path:
    """Weekly demand for the units carrying the policy, showing how flat the series are."""
    fig, ax = plt.subplots(figsize=(9.5, 3.6))
    for unit in units:
        ax.plot(panel.columns, panel.loc[unit].fillna(0), linewidth=1.0, alpha=0.85)
    _style(ax, "Weekly demand, high volume units", "", "units per week")
    ax.text(0.005, 0.94,
            "no trend, no seasonality, lag-1 autocorrelation near zero",
            transform=ax.transAxes, color=MUTED, fontsize=8)
    return _save(fig, path)
