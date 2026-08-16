"""Data quality audit run before any policy is fitted.

The premise this project started from was that DataCo supplies an observed lead-time
distribution, and that using real lead-time variability instead of a textbook fixed
lead time is what separates it from the usual inventory exercise. That premise does not
survive contact with the data, and the tests in this module are what establish it.

Three findings come out of here and all three change the design downstream:

The lead-time column is generated, not observed. Within a shipping mode it is either a
constant or a discrete uniform draw, it carries no regional or product signal, and the
two highest volume modes are statistically indistinguishable from each other despite
promising different delivery windows.

The extract changes character in October 2017. Order quantity collapses to a constant,
the catalogue turns over almost completely, and volume halves. The two eras are
different data generating processes and are not poolable.

Markets are time sliced rather than concurrent. Each market is active in a minority of
weeks, so any product by market weekly series is zero inflated by construction and its
apparent variability is mostly the market being switched off.

Everything here is a test with a number attached rather than an assertion, so the
conclusions in the write-up can be traced to a statistic.
"""

from dataclasses import dataclass, asdict
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats

REAL_LEADTIME = "Days for shipping (real)"
SCHEDULED_LEADTIME = "Days for shipment (scheduled)"
MODE = "Shipping Mode"


@dataclass
class ModeLeadTimeTest:
    """Whether one shipping mode's realised lead time looks like a random draw."""

    shipping_mode: str
    rows: int
    scheduled_days: float
    observed_mean: float
    observed_std: float
    support: List[int]
    is_degenerate: bool
    uniform_chi2: float
    uniform_p_value: float
    reads_as_uniform: bool
    late_share: float

    def to_dict(self) -> Dict:
        return asdict(self)


def leadtime_by_mode(frame: pd.DataFrame, alpha: float, min_rows: int) -> List[ModeLeadTimeTest]:
    """Test each shipping mode's realised lead time for constancy or uniformity.

    A mode whose realised lead time takes a single value carries zero variance and can
    contribute nothing to a safety stock calculation. A mode whose realised lead time is
    indistinguishable from a discrete uniform draw over its support is a random number
    generator rather than a delivery process. Both are recorded rather than smoothed
    over, because a mean and a standard deviation can be computed from either and would
    look perfectly respectable in a README.
    """
    results = []
    for mode, block in frame.groupby(MODE):
        values = block[REAL_LEADTIME]
        counts = values.value_counts().sort_index()
        support = [int(v) for v in counts.index]
        degenerate = len(support) == 1

        if degenerate or len(block) < min_rows:
            chi2, p_value, reads_uniform = float("nan"), float("nan"), False
        else:
            expected = np.full(len(counts), len(values) / len(counts))
            chi2, p_value = stats.chisquare(counts.values, expected)
            reads_uniform = bool(p_value > alpha)

        results.append(
            ModeLeadTimeTest(
                shipping_mode=str(mode),
                rows=int(len(block)),
                scheduled_days=float(block[SCHEDULED_LEADTIME].iloc[0]),
                observed_mean=float(values.mean()),
                observed_std=float(values.std()),
                support=support,
                is_degenerate=bool(degenerate),
                uniform_chi2=float(chi2),
                uniform_p_value=float(p_value),
                reads_as_uniform=reads_uniform,
                late_share=float((values > block[SCHEDULED_LEADTIME]).mean()),
            )
        )
    return results


def mode_pair_homogeneity(frame: pd.DataFrame, left: str, right: str) -> Dict:
    """Test whether two shipping modes draw realised lead time from the same law.

    Two modes that promise different delivery windows but deliver from an identical
    distribution are not two service levels, they are one process with two labels.
    """
    a = frame.loc[frame[MODE] == left, REAL_LEADTIME]
    b = frame.loc[frame[MODE] == right, REAL_LEADTIME]
    support = sorted(set(a.unique()) | set(b.unique()))
    table = np.vstack(
        [
            [int((a == v).sum()) for v in support],
            [int((b == v).sum()) for v in support],
        ]
    )
    chi2, p_value, dof, _ = stats.chi2_contingency(table)
    return {
        "left_mode": left,
        "right_mode": right,
        "left_scheduled": float(frame.loc[frame[MODE] == left, SCHEDULED_LEADTIME].iloc[0]),
        "right_scheduled": float(frame.loc[frame[MODE] == right, SCHEDULED_LEADTIME].iloc[0]),
        "left_mean": float(a.mean()),
        "right_mean": float(b.mean()),
        "left_std": float(a.std()),
        "right_std": float(b.std()),
        "chi2": float(chi2),
        "dof": int(dof),
        "p_value": float(p_value),
        "same_distribution": bool(p_value > 0.01),
    }


def leadtime_regional_signal(
    frame: pd.DataFrame, within_mode: str, min_eta_squared: float = 0.01
) -> Dict:
    """Test for any regional effect on realised lead time, holding shipping mode fixed.

    If replenishment lead time responded to geography the way a real network does, the
    between region spread would exceed what sampling noise alone produces. This runs a
    one way test across order regions inside a single mode so the mode effect cannot
    manufacture a result.

    The verdict is taken from eta squared rather than the p value. At six figure row
    counts a one way test will reject the null on a difference far too small to plan
    against, so the question worth asking is not whether region has any effect but how
    much of the variance it accounts for.
    """
    block = frame.loc[frame[MODE] == within_mode]
    groups = [g[REAL_LEADTIME].values for _, g in block.groupby("Order Region") if len(g) >= 30]
    f_stat, p_value = stats.f_oneway(*groups)

    grand_mean = np.concatenate(groups).mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_total = sum(((g - grand_mean) ** 2).sum() for g in groups)
    eta_squared = float(ss_between / ss_total)

    means = block.groupby("Order Region")[REAL_LEADTIME].mean()
    pooled_std = float(block[REAL_LEADTIME].std())
    spread = float(means.max() - means.min())
    return {
        "within_mode": within_mode,
        "regions_tested": len(groups),
        "rows": int(len(block)),
        "f_statistic": float(f_stat),
        "p_value": float(p_value),
        "eta_squared": eta_squared,
        "variance_explained_pct": round(eta_squared * 100, 4),
        "regional_mean_min": float(means.min()),
        "regional_mean_max": float(means.max()),
        "regional_mean_spread": spread,
        "spread_in_pooled_std": round(spread / pooled_std, 3),
        "pooled_std": pooled_std,
        "statistically_significant": bool(p_value < 0.01),
        "regional_signal_detected": bool(eta_squared >= min_eta_squared),
    }


def derived_column_check(frame: pd.DataFrame) -> Dict:
    """Check whether Late_delivery_risk is just a restatement of real versus scheduled."""
    implied = (frame[REAL_LEADTIME] > frame[SCHEDULED_LEADTIME]).astype(int)
    agreement = float((implied == frame["Late_delivery_risk"]).mean())
    return {
        "column": "Late_delivery_risk",
        "reproduced_from": "Days for shipping (real) > Days for shipment (scheduled)",
        "agreement": agreement,
        "is_derived": bool(agreement > 0.95),
    }


def structural_break_scan(frame: pd.DataFrame, config) -> pd.DataFrame:
    """Monthly profile of the extract used to locate where it changes character."""
    quantity = config.section("data")["quantity_column"]
    unit = config.section("data")["planning_unit"]
    monthly = frame.assign(month=frame["order_date"].dt.to_period("M")).groupby("month")
    return pd.DataFrame(
        {
            "order_lines": monthly.size(),
            "total_quantity": monthly[quantity].sum(),
            "mean_quantity": monthly[quantity].mean().round(3),
            "share_quantity_one": monthly[quantity].apply(lambda s: (s == 1).mean()).round(3),
            "distinct_products": monthly[unit].nunique(),
            "sales": monthly["Sales"].sum().round(2),
        }
    )


def locate_quantity_collapse(monthly: pd.DataFrame, threshold: float) -> Dict:
    """First month from which order quantity stops varying, and stays that way."""
    flagged = monthly.index[monthly["share_quantity_one"] >= threshold]
    if len(flagged) == 0:
        return {"collapse_detected": False, "first_collapsed_month": None, "months_affected": 0}
    first = flagged.min()
    tail = monthly.loc[monthly.index >= first, "share_quantity_one"]
    return {
        "collapse_detected": True,
        "first_collapsed_month": str(first),
        "months_affected": int(len(tail)),
        "persists_to_end": bool((tail >= threshold).all()),
        "share_quantity_one_in_tail": float(tail.mean()),
        "order_lines_before": int(monthly.loc[monthly.index < first, "order_lines"].mean()),
        "order_lines_after": int(monthly.loc[monthly.index >= first, "order_lines"].mean()),
        "products_before": float(monthly.loc[monthly.index < first, "distinct_products"].mean()),
        "products_after": float(monthly.loc[monthly.index >= first, "distinct_products"].mean()),
    }


def catalogue_turnover(frame: pd.DataFrame, config, boundary: str) -> Dict:
    """How much of the product catalogue is shared across the structural break."""
    unit = config.section("data")["planning_unit"]
    cut = pd.Timestamp(boundary)
    before = set(frame.loc[frame["order_date"] < cut, unit].unique())
    after = set(frame.loc[frame["order_date"] >= cut, unit].unique())
    return {
        "boundary": boundary,
        "products_before": len(before),
        "products_after": len(after),
        "shared": len(before & after),
        "after_only": len(after - before),
        "before_only": len(before - after),
    }


def market_concurrency(frame: pd.DataFrame) -> pd.DataFrame:
    """Weeks in which each market records any activity, against the full week index.

    A market that is live in a minority of weeks cannot be treated as a standing demand
    stream. This is what rules out product by market as a planning grain: the zeros are
    the market being switched off, not customers declining to buy.
    """
    total_weeks = frame["week"].nunique()
    active = frame.groupby("Market")["week"].nunique().rename("weeks_active")
    out = active.to_frame()
    out["total_weeks"] = total_weeks
    out["share_of_weeks_active"] = (out["weeks_active"] / total_weeks).round(3)
    out["first_week"] = frame.groupby("Market")["week"].min()
    out["last_week"] = frame.groupby("Market")["week"].max()
    return out.sort_values("weeks_active", ascending=False)


def demand_autocorrelation(panel: pd.DataFrame, max_lag: int = 4) -> Dict:
    """Serial correlation in aggregate weekly demand.

    Real demand carries persistence. A series whose autocorrelation is indistinguishable
    from zero at every lag has no trend, no seasonality and no momentum, which is a
    signature of independent draws rather than a market.
    """
    series = panel.fillna(0).sum(axis=0)
    return {
        "weeks": int(len(series)),
        "mean": float(series.mean()),
        "std": float(series.std()),
        "coefficient_of_variation": float(series.std() / series.mean()),
        "autocorrelation": {f"lag_{lag}": float(series.autocorr(lag)) for lag in range(1, max_lag + 1)},
        "max_abs_autocorrelation": float(
            max(abs(series.autocorr(lag)) for lag in range(1, max_lag + 1))
        ),
    }
