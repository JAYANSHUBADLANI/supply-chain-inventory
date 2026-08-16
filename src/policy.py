"""Safety stock, reorder point and order quantity.

The time base here is one day. Lead time in this extract is natively measured in whole
days over a support of two to six, so working in days keeps the lead-time distribution
exact instead of forcing a fractional week that would have to be interpolated. Demand at
daily granularity is available with complete coverage for every unit carried forward, and
the audit found no day of week effect to distort it.

Safety stock uses the combined form that carries both sources of uncertainty:

    SS = z * sqrt(L * sigma_d^2 + d^2 * sigma_L^2)

The first term under the root is demand uncertainty accumulated over the lead time. The
second is the demand that arrives while the lead time itself is varying. The widely quoted
simplification drops the second term, which is safe only when lead time is effectively
fixed. On this data it is not, and the decomposition returned alongside every result
records which term is actually driving the buffer, because that ratio differs by an order
of magnitude between the high volume units and the tail and it is what makes a single
blanket policy the wrong answer.

Every cost figure below rests on assumed parameters. The extract carries selling prices
and revenue but no holding rate, no ordering cost and no stockout penalty, so those come
from config and are assumptions, not observations.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class SafetyStockResult:
    """A safety stock figure with the variance decomposition that produced it."""

    planning_unit: str
    service_level: float
    z: float
    demand_mean: float
    demand_std: float
    leadtime_mean: float
    leadtime_std: float
    demand_variance_term: float
    leadtime_variance_term: float
    total_variance: float
    sigma_demand_over_leadtime: float
    safety_stock: float

    @property
    def leadtime_share_of_variance(self) -> float:
        if self.total_variance == 0:
            return float("nan")
        return self.leadtime_variance_term / self.total_variance

    @property
    def demand_share_of_variance(self) -> float:
        if self.total_variance == 0:
            return float("nan")
        return self.demand_variance_term / self.total_variance

    @property
    def dominant_driver(self) -> str:
        return "lead time" if self.leadtime_share_of_variance >= 0.5 else "demand"

    @property
    def demand_only_safety_stock(self) -> float:
        """What the simplified formula would return if lead-time variance were ignored."""
        return self.z * np.sqrt(self.demand_variance_term)

    @property
    def understatement_vs_demand_only(self) -> float:
        """How far the simplified formula falls short, as a multiple."""
        simple = self.demand_only_safety_stock
        return float(self.safety_stock / simple) if simple > 0 else float("nan")

    def to_dict(self) -> Dict:
        payload = asdict(self)
        payload.update(
            {
                "leadtime_share_of_variance": self.leadtime_share_of_variance,
                "demand_share_of_variance": self.demand_share_of_variance,
                "dominant_driver": self.dominant_driver,
                "demand_only_safety_stock": self.demand_only_safety_stock,
                "understatement_vs_demand_only": self.understatement_vs_demand_only,
            }
        )
        return payload


def z_for_service_level(service_level: float) -> float:
    """Standard normal quantile for a target cycle service level."""
    if not 0 < service_level < 1:
        raise ValueError(f"service level must be inside (0, 1), got {service_level}")
    return float(stats.norm.ppf(service_level))


def safety_stock(
    planning_unit: str,
    demand_mean: float,
    demand_std: float,
    leadtime_mean: float,
    leadtime_std: float,
    service_level: float,
) -> SafetyStockResult:
    """Combined demand and lead-time variability safety stock.

    Demand mean and standard deviation are per period, lead time mean and standard
    deviation are in the same period units. Mixing the two time bases is the single
    easiest way to get this formula wrong by a factor of the period length, so both are
    required in the caller's units rather than converted here.
    """
    if demand_mean < 0 or demand_std < 0 or leadtime_mean < 0 or leadtime_std < 0:
        raise ValueError("demand and lead-time parameters must be non negative")

    z = z_for_service_level(service_level)
    demand_term = leadtime_mean * demand_std ** 2
    leadtime_term = (demand_mean ** 2) * (leadtime_std ** 2)
    total = demand_term + leadtime_term
    sigma_ddlt = float(np.sqrt(total))

    return SafetyStockResult(
        planning_unit=planning_unit,
        service_level=service_level,
        z=z,
        demand_mean=demand_mean,
        demand_std=demand_std,
        leadtime_mean=leadtime_mean,
        leadtime_std=leadtime_std,
        demand_variance_term=float(demand_term),
        leadtime_variance_term=float(leadtime_term),
        total_variance=float(total),
        sigma_demand_over_leadtime=sigma_ddlt,
        safety_stock=float(z * sigma_ddlt),
    )


def expected_demand_over_leadtime(demand_mean: float, leadtime_mean: float) -> float:
    return demand_mean * leadtime_mean


def reorder_point(demand_mean: float, leadtime_mean: float, safety_stock_units: float) -> float:
    """Expected demand during lead time plus the buffer."""
    return expected_demand_over_leadtime(demand_mean, leadtime_mean) + safety_stock_units


def economic_order_quantity(
    annual_demand: float, ordering_cost: float, annual_holding_cost_per_unit: float
) -> float:
    """Classic EOQ. Every input is an assumption except the demand rate."""
    if annual_holding_cost_per_unit <= 0:
        raise ValueError("annual holding cost per unit must be positive")
    if ordering_cost < 0 or annual_demand < 0:
        raise ValueError("ordering cost and annual demand must be non negative")
    return float(np.sqrt(2.0 * annual_demand * ordering_cost / annual_holding_cost_per_unit))


def standard_normal_loss(z: float) -> float:
    """Expected shortage per unit of sigma for a normal demand-over-lead-time."""
    return float(stats.norm.pdf(z) - z * stats.norm.sf(z))


def expected_fill_rate(sigma_ddlt: float, z: float, order_quantity: float) -> float:
    """Item fill rate implied by a cycle service level and an order quantity.

    Cycle service level answers how often a replenishment cycle ends without a stockout.
    Fill rate answers what share of demanded units ship from stock, which is the number a
    commercial conversation actually cares about. The two differ, and the gap widens as
    the order quantity shrinks, so quoting one while meaning the other overstates service.
    """
    if order_quantity <= 0:
        raise ValueError("order quantity must be positive")
    shortage_per_cycle = sigma_ddlt * standard_normal_loss(z)
    return float(np.clip(1.0 - shortage_per_cycle / order_quantity, 0.0, 1.0))


def bootstrap_safety_stock(
    daily_demand: np.ndarray,
    leadtime_values: np.ndarray,
    leadtime_probabilities: np.ndarray,
    service_level: float,
    draws: int,
    rng: np.random.Generator,
) -> Dict:
    """Safety stock from the empirical demand-over-lead-time distribution.

    The normal approximation behind the closed form assumes demand over lead time is
    symmetric and light tailed. For a unit selling seventy a day that is close enough. For
    a unit selling under one a day, where three quarters of days are zero, it is not, and
    the closed form will quote a buffer that the actual distribution does not support.

    This resamples lead times from their observed mass function, then resamples that many
    days of actual historical demand, and reads the required buffer straight off the
    resulting quantile. No distributional assumption is imposed.
    """
    if len(daily_demand) == 0:
        raise ValueError("daily demand series is empty")

    lead_draws = rng.choice(leadtime_values, size=draws, p=leadtime_probabilities)
    max_lead = int(lead_draws.max())
    totals = np.zeros(draws)
    if max_lead > 0:
        sampled = rng.choice(daily_demand, size=(draws, max_lead), replace=True)
        mask = np.arange(max_lead)[None, :] < lead_draws[:, None]
        totals = (sampled * mask).sum(axis=1)

    mean_ddlt = float(totals.mean())
    quantile = float(np.quantile(totals, service_level))
    return {
        "service_level": service_level,
        "draws": draws,
        "mean_demand_over_leadtime": mean_ddlt,
        "std_demand_over_leadtime": float(totals.std(ddof=1)),
        "quantile_demand_over_leadtime": quantile,
        "safety_stock": max(quantile - mean_ddlt, 0.0),
        "skewness": float(stats.skew(totals)),
        "excess_kurtosis": float(stats.kurtosis(totals)),
        "zero_share": float((totals == 0).mean()),
    }


def annual_holding_cost(units: float, unit_price: float, costs) -> float:
    """Cost of carrying a given number of units for a year, under the stated assumptions."""
    return float(units * costs.annual_holding_cost(unit_price))


def total_annual_policy_cost(
    annual_demand: float,
    order_quantity: float,
    safety_stock_units: float,
    unit_price: float,
    costs,
) -> Dict:
    """Ordering plus holding cost for a policy, split so each part is visible."""
    if order_quantity <= 0:
        raise ValueError("order quantity must be positive")
    orders_per_year = annual_demand / order_quantity
    ordering = orders_per_year * costs.ordering_cost_per_order
    cycle_stock_holding = annual_holding_cost(order_quantity / 2.0, unit_price, costs)
    safety_stock_holding = annual_holding_cost(safety_stock_units, unit_price, costs)
    return {
        "orders_per_year": float(orders_per_year),
        "ordering_cost": float(ordering),
        "cycle_stock_holding_cost": float(cycle_stock_holding),
        "safety_stock_holding_cost": float(safety_stock_holding),
        "total_annual_cost": float(ordering + cycle_stock_holding + safety_stock_holding),
    }
