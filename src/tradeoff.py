"""Service level against cost, so a recommended target comes off a curve rather than a hunch.

Safety stock rises with the normal quantile, which is close to linear in the middle of the
range and then turns sharply as the target approaches certainty. The last stretch of
service is therefore the expensive one, and the point of sweeping the target rather than
picking 95 percent by convention is to see where that turn happens for a given unit and
decide whether the segment is worth spending past it.

The curve reports cycle service level and item fill rate side by side. They are different
quantities and the gap between them is not small, so a target quoted without saying which
one it is has not really been specified.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src import policy


def service_level_curve(
    planning_unit: str,
    demand_mean: float,
    demand_std: float,
    leadtime_mean: float,
    leadtime_std: float,
    unit_price: float,
    annual_demand: float,
    order_quantity: float,
    service_levels: List[float],
    costs,
) -> pd.DataFrame:
    """Safety stock, holding cost and fill rate across a sweep of service targets.

    The cost of the buffer is also expressed against the annual revenue it protects. That
    ratio, rather than the shape of the cost curve, is what differentiates one segment from
    another: see knee_point for why the curve shape alone cannot.
    """
    annual_revenue = annual_demand * unit_price
    rows = []
    for level in service_levels:
        result = policy.safety_stock(
            planning_unit=planning_unit,
            demand_mean=demand_mean,
            demand_std=demand_std,
            leadtime_mean=leadtime_mean,
            leadtime_std=leadtime_std,
            service_level=level,
        )
        cost_block = policy.total_annual_policy_cost(
            annual_demand=annual_demand,
            order_quantity=order_quantity,
            safety_stock_units=result.safety_stock,
            unit_price=unit_price,
            costs=costs,
        )
        rows.append(
            {
                "planning_unit": planning_unit,
                "cycle_service_level": level,
                "z": result.z,
                "safety_stock_units": result.safety_stock,
                "reorder_point": policy.reorder_point(
                    demand_mean, leadtime_mean, result.safety_stock
                ),
                "expected_fill_rate": policy.expected_fill_rate(
                    result.sigma_demand_over_leadtime, result.z, order_quantity
                ),
                "safety_stock_holding_cost": cost_block["safety_stock_holding_cost"],
                "cycle_stock_holding_cost": cost_block["cycle_stock_holding_cost"],
                "ordering_cost": cost_block["ordering_cost"],
                "total_annual_cost": cost_block["total_annual_cost"],
                "annual_revenue": annual_revenue,
                "buffer_cost_per_1000_revenue": (
                    1000.0 * cost_block["safety_stock_holding_cost"] / annual_revenue
                    if annual_revenue > 0
                    else np.nan
                ),
            }
        )
    curve = pd.DataFrame(rows)
    curve["marginal_cost_per_service_point"] = _marginal_cost(curve)
    return curve


def differentiation_basis(
    policy_table: pd.DataFrame, segment_column: str = "segment"
) -> pd.DataFrame:
    """What a service target costs per unit of revenue it protects, by segment.

    This is the number that earns a differentiated policy. Buying a high service level on a
    unit whose buffer costs a fraction of a percent of the revenue it protects is close to
    free. Buying the same level on a unit where that ratio is an order of magnitude worse is
    not, and that gap is what a tiered target is responding to.
    """
    frame = policy_table.copy()
    frame["annual_revenue"] = frame["annual_demand"] * frame["unit_price"]
    frame["buffer_cost_per_1000_revenue"] = (
        1000.0 * frame["safety_stock_holding_cost"] / frame["annual_revenue"]
    )
    return (
        frame.groupby(segment_column)
        .agg(
            units=("safety_stock", "size"),
            target_service_level=("service_level", "median"),
            annual_revenue=("annual_revenue", "sum"),
            safety_stock_holding_cost=("safety_stock_holding_cost", "sum"),
            median_buffer_cost_per_1000_revenue=("buffer_cost_per_1000_revenue", "median"),
            median_days_of_cover=("days_of_cover_at_eoq", "median"),
        )
        .round(4)
    )


def _marginal_cost(curve: pd.DataFrame) -> pd.Series:
    """Extra annual holding cost bought by each additional point of service."""
    delta_cost = curve["safety_stock_holding_cost"].diff()
    delta_service = curve["cycle_service_level"].diff() * 100.0
    return (delta_cost / delta_service).replace([np.inf, -np.inf], np.nan)


def knee_point(curve: pd.DataFrame) -> Dict:
    """Where the curve turns from buying service cheaply to buying it expensively.

    Taken as the last swept level whose marginal cost per service point is still below
    twice the median marginal cost across the sweep. This is a heuristic for reading the
    curve, not an optimisation: the honest recommendation is that a target above this
    point needs a stockout cost to justify it, and the extract does not carry one.

    One property of this knee is worth stating plainly rather than leaving for someone to
    discover. Safety stock is proportional to the normal quantile of the target, and
    holding cost is proportional to safety stock, so the marginal cost curve has the same
    shape for every planning unit and differs only by a scale factor. The knee therefore
    lands at the same service level for every unit, and cannot on its own justify giving
    one segment a different target from another. It says where the curve turns, which is a
    real and useful thing to know, but the case for differentiation has to come from
    differentiation_basis, where the cost of the buffer is set against the revenue it is
    protecting and the segments genuinely separate.
    """
    marginal = curve["marginal_cost_per_service_point"].dropna()
    if marginal.empty:
        return {"knee_service_level": None, "reason": "curve too short to locate a knee"}
    threshold = 2.0 * float(marginal.median())
    cheap = curve.loc[curve["marginal_cost_per_service_point"].fillna(0) <= threshold]
    knee = float(cheap["cycle_service_level"].max())
    row = curve.loc[curve["cycle_service_level"] == knee].iloc[0]
    return {
        "knee_service_level": knee,
        "marginal_cost_threshold": threshold,
        "safety_stock_at_knee": float(row["safety_stock_units"]),
        "holding_cost_at_knee": float(row["safety_stock_holding_cost"]),
        "fill_rate_at_knee": float(row["expected_fill_rate"]),
    }


def normal_versus_empirical(
    planning_unit: str,
    daily_demand: np.ndarray,
    leadtime_values: np.ndarray,
    leadtime_probabilities: np.ndarray,
    demand_mean: float,
    demand_std: float,
    leadtime_mean: float,
    leadtime_std: float,
    service_levels: List[float],
    draws: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Closed form safety stock against the buffer the observed distribution actually needs.

    A gap here is not a bug in either method. It is the normal approximation being asked to
    describe a demand-over-lead-time distribution that is skewed or zero inflated, which is
    what happens on low volume units. Reporting the ratio makes the approximation error
    visible instead of letting it sit inside a number that looks precise.
    """
    rows = []
    for level in service_levels:
        closed_form = policy.safety_stock(
            planning_unit=planning_unit,
            demand_mean=demand_mean,
            demand_std=demand_std,
            leadtime_mean=leadtime_mean,
            leadtime_std=leadtime_std,
            service_level=level,
        )
        empirical = policy.bootstrap_safety_stock(
            daily_demand=daily_demand,
            leadtime_values=leadtime_values,
            leadtime_probabilities=leadtime_probabilities,
            service_level=level,
            draws=draws,
            rng=rng,
        )
        normal_ss = closed_form.safety_stock
        empirical_ss = empirical["safety_stock"]
        rows.append(
            {
                "planning_unit": planning_unit,
                "cycle_service_level": level,
                "normal_safety_stock": normal_ss,
                "empirical_safety_stock": empirical_ss,
                "difference_units": empirical_ss - normal_ss,
                "normal_over_empirical": normal_ss / empirical_ss if empirical_ss > 0 else np.nan,
                "normal_sigma_ddlt": closed_form.sigma_demand_over_leadtime,
                "empirical_sigma_ddlt": empirical["std_demand_over_leadtime"],
                "empirical_skewness": empirical["skewness"],
                "empirical_excess_kurtosis": empirical["excess_kurtosis"],
                "empirical_zero_share": empirical["zero_share"],
            }
        )
    return pd.DataFrame(rows)
