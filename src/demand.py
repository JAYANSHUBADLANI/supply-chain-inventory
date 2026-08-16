"""Empirical weekly demand statistics per planning unit.

This is deliberately not a forecasting model. The policy work downstream needs a mean and
a variance of demand per period, and both are taken from the observed history rather than
projected. What matters here is getting the denominator right: which weeks count as
observations for a given product, and which products have enough of them to support a
variance estimate that anyone should plan against.

A week inside a product's active span with no orders is genuine zero demand and counts.
A week before the product first appeared or after it last appeared is not an observation
at all and is excluded, because averaging in the weeks when a product was not on the
catalogue would understate its mean and overstate its variability.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ActiveSpan:
    """The window over which a planning unit was genuinely on the catalogue."""

    first_week: pd.Timestamp
    last_week: pd.Timestamp
    weeks: int
    active_weeks: int
    zero_weeks: int


def active_span(row: pd.Series) -> Optional[ActiveSpan]:
    """First to last week in which a planning unit recorded any order."""
    active = row.notna().values
    if not active.any():
        return None
    first = int(active.argmax())
    last = int(len(active) - 1 - active[::-1].argmax())
    window = row.iloc[first : last + 1]
    return ActiveSpan(
        first_week=row.index[first],
        last_week=row.index[last],
        weeks=len(window),
        active_weeks=int(window.notna().sum()),
        zero_weeks=int(window.isna().sum()),
    )


def demand_series(row: pd.Series) -> pd.Series:
    """Weekly demand over a unit's active span, with unsold weeks read as zero."""
    span = active_span(row)
    if span is None:
        return pd.Series(dtype=float)
    return row.loc[span.first_week : span.last_week].fillna(0.0)


def demand_statistics(
    panel: pd.DataFrame,
    min_observations: int,
    week_slice: Optional[slice] = None,
) -> pd.DataFrame:
    """Mean, standard deviation and coefficient of variation of weekly demand.

    The sample standard deviation is used rather than the population one, since these are
    estimates from a sample of weeks rather than a known distribution. Units whose
    observation count falls below the threshold are kept in the output but flagged, so a
    thin sample is visible in the table rather than quietly feeding a safety stock.
    """
    frame = panel if week_slice is None else panel.iloc[:, week_slice]
    records = []
    for unit, row in frame.iterrows():
        series = demand_series(row)
        if len(series) == 0:
            continue
        mean = float(series.mean())
        std = float(series.std(ddof=1)) if len(series) > 1 else 0.0
        records.append(
            {
                "planning_unit": unit,
                "observations": int(len(series)),
                "active_weeks": int((series > 0).sum()),
                "zero_weeks": int((series == 0).sum()),
                "first_week": series.index.min(),
                "last_week": series.index.max(),
                "weekly_mean": mean,
                "weekly_std": std,
                "weekly_variance": std ** 2,
                "coefficient_of_variation": float(std / mean) if mean > 0 else np.nan,
                "zero_week_share": float((series == 0).mean()),
                "total_units": float(series.sum()),
            }
        )
    stats = pd.DataFrame(records).set_index("planning_unit")
    stats["sufficient_observations"] = stats["observations"] >= min_observations
    stats["intermittent"] = stats["zero_week_share"] > 0.10
    stats["variance_estimate_trustworthy"] = (
        stats["sufficient_observations"] & ~stats["intermittent"]
    )
    return stats.sort_values("total_units", ascending=False)


def exclusion_ledger(stats: pd.DataFrame) -> Dict:
    """Why each planning unit is or is not carried into the policy stage."""
    total = len(stats)
    thin = int((~stats["sufficient_observations"]).sum())
    intermittent = int(
        (stats["sufficient_observations"] & stats["intermittent"]).sum()
    )
    kept = int(stats["variance_estimate_trustworthy"].sum())
    if thin + intermittent + kept != total:
        raise ValueError(
            f"demand exclusion ledger does not reconcile: {thin + intermittent + kept} "
            f"accounted for against {total} planning units"
        )
    return {
        "planning_units_seen": total,
        "excluded_thin_sample": thin,
        "excluded_intermittent": intermittent,
        "carried_forward": kept,
        "carried_forward_demand_share": float(
            stats.loc[stats["variance_estimate_trustworthy"], "total_units"].sum()
            / stats["total_units"].sum()
        ),
    }


def pooled_weekly_demand(panel: pd.DataFrame) -> pd.Series:
    """Total weekly demand across the supplied planning units."""
    return panel.fillna(0.0).sum(axis=0)
