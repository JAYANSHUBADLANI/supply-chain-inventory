"""ABC by revenue contribution, XYZ by demand variability, and the resulting grid.

The point of segmenting is that a single blanket service level is either too expensive on
the tail or too thin on the items that matter. ABC ranks planning units by what they
contribute, XYZ by how predictable they are, and the cross of the two decides how much
buffer each unit earns.

This module also measures how far the observed concentration departs from the textbook
20/80 shape, because on this extract it departs a long way and that changes how much the
segmentation is really doing. A grid whose top tier holds a handful of units is still the
correct answer, but it is a different situation from one where the A tier holds a fifth of
the catalogue, and the write-up should not imply the second when the data shows the first.
"""

from typing import Dict, List

import numpy as np
import pandas as pd


def abc_classify(revenue: pd.Series, thresholds: List[float]) -> pd.DataFrame:
    """Rank by revenue and cut at the cumulative share thresholds.

    A unit is placed in the tier its cumulative share first reaches, so the unit that
    carries the total past 80 percent belongs to A rather than B. Cutting the other way
    would push a material contributor down a tier on a rounding boundary.
    """
    if not 0 < thresholds[0] < thresholds[1] < 1:
        raise ValueError("abc thresholds must be increasing and inside (0, 1)")
    ordered = revenue.sort_values(ascending=False)
    share = ordered / ordered.sum()
    cumulative = share.cumsum()
    previous = cumulative.shift(1).fillna(0.0)

    tier = pd.Series("C", index=ordered.index, dtype=object)
    tier[previous < thresholds[1]] = "B"
    tier[previous < thresholds[0]] = "A"

    return pd.DataFrame(
        {
            "revenue": ordered,
            "revenue_share": share,
            "cumulative_revenue_share": cumulative,
            "abc_class": tier,
        }
    )


def xyz_classify(coefficient_of_variation: pd.Series, thresholds: List[float]) -> pd.Series:
    """Cut demand variability into X (steady), Y (variable) and Z (erratic)."""
    if not 0 < thresholds[0] < thresholds[1]:
        raise ValueError("xyz thresholds must be increasing and positive")
    bins = [-np.inf, thresholds[0], thresholds[1], np.inf]
    return pd.Series(
        pd.cut(coefficient_of_variation, bins=bins, labels=["X", "Y", "Z"], right=False),
        index=coefficient_of_variation.index,
        name="xyz_class",
    ).astype(object)


def build_grid(
    revenue: pd.Series,
    coefficient_of_variation: pd.Series,
    abc_thresholds: List[float],
    xyz_thresholds: List[float],
    service_levels: Dict[str, float],
) -> pd.DataFrame:
    """Join the two axes into one segment table with a service target per unit."""
    abc = abc_classify(revenue, abc_thresholds)
    xyz = xyz_classify(coefficient_of_variation.reindex(abc.index), xyz_thresholds)
    grid = abc.join(xyz)
    grid["coefficient_of_variation"] = coefficient_of_variation.reindex(grid.index)
    grid["segment"] = grid["abc_class"] + grid["xyz_class"].fillna("?")
    grid["target_service_level"] = grid["abc_class"].map(service_levels)
    if grid["target_service_level"].isna().any():
        missing = sorted(grid.loc[grid["target_service_level"].isna(), "abc_class"].unique())
        raise ValueError(f"no service level configured for ABC class(es): {missing}")
    return grid


def grid_summary(grid: pd.DataFrame) -> pd.DataFrame:
    """Counts and revenue share for each populated cell of the 3x3 grid."""
    summary = (
        grid.groupby(["abc_class", "xyz_class"], observed=False)
        .agg(
            units=("revenue", "size"),
            revenue=("revenue", "sum"),
            revenue_share=("revenue_share", "sum"),
            median_cv=("coefficient_of_variation", "median"),
        )
        .reset_index()
    )
    summary = summary[summary["units"] > 0]
    summary["revenue_share"] = summary["revenue_share"].round(4)
    summary["median_cv"] = summary["median_cv"].round(3)
    return summary.sort_values(["abc_class", "xyz_class"])


def concentration_profile(revenue: pd.Series) -> Dict:
    """How far the observed Pareto departs from the textbook 20/80 assumption."""
    ordered = revenue.sort_values(ascending=False)
    cumulative = (ordered / ordered.sum()).cumsum()
    n = len(ordered)
    top_fifth = int(np.ceil(n * 0.20))
    return {
        "planning_units": n,
        "units_for_50pct_revenue": int((cumulative < 0.50).sum() + 1),
        "units_for_80pct_revenue": int((cumulative < 0.80).sum() + 1),
        "units_for_95pct_revenue": int((cumulative < 0.95).sum() + 1),
        "share_of_units_for_80pct_revenue": round((int((cumulative < 0.80).sum()) + 1) / n, 4),
        "revenue_share_of_top_fifth": round(float(cumulative.iloc[top_fifth - 1]), 4),
        "textbook_expectation": "top 20 percent of units carry about 80 percent of revenue",
    }


def degeneracy_flags(grid: pd.DataFrame) -> Dict:
    """Cells and axes where the segmentation has too little spread to be doing work.

    Reported rather than hidden. If every unit in the A tier lands in the same XYZ bucket
    then the XYZ axis is not differentiating anything at the top of the catalogue, and a
    write-up that presents a full 3x3 policy would be overclaiming.
    """
    populated = grid.groupby(["abc_class", "xyz_class"], observed=True).size()
    a_tier = grid.loc[grid["abc_class"] == "A", "xyz_class"]
    return {
        "populated_cells": int((populated > 0).sum()),
        "possible_cells": 9,
        "a_tier_units": int(len(a_tier)),
        "a_tier_xyz_classes": sorted(a_tier.dropna().unique().tolist()),
        "a_tier_collapses_to_one_xyz_class": bool(a_tier.nunique(dropna=True) <= 1),
        "cells_with_single_unit": int((populated == 1).sum()),
    }
