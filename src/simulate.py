"""Period by period simulation of an (s, Q) policy against the actual order stream.

The policy parameters are fitted on an early window and the simulation runs on a later
window the fitting never saw. Without that split the comparison would be circular: any
policy whose safety stock is derived from the variance of the same demand it is then
scored against will look calibrated whether or not the method generalises.

Unmet demand is treated as lost rather than backordered, which is the right assumption for
a retail order stream and the conservative one for measuring service. Lead times are drawn
from the empirical mass function, and the same draws are reused across every policy under
comparison so that one policy cannot win on a luckier sequence of deliveries. That common
random numbers detail matters more than it sounds: without it the differences between
policies on a 280 day window are comfortably inside the noise.
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class SimulationResult:
    """Realised service and cost for one planning unit under one policy."""

    planning_unit: str
    policy: str
    reorder_point: float
    order_quantity: float
    safety_stock: float
    days: int
    units_demanded: float
    units_shipped: float
    units_lost: float
    stockout_days: int
    orders_placed: int
    cycles: int
    cycles_with_stockout: int
    mean_on_hand: float
    holding_cost: float
    ordering_cost: float
    total_cost: float

    @property
    def fill_rate(self) -> float:
        if self.units_demanded == 0:
            return float("nan")
        return self.units_shipped / self.units_demanded

    @property
    def stockout_day_rate(self) -> float:
        return self.stockout_days / self.days if self.days else float("nan")

    @property
    def realised_cycle_service_level(self) -> float:
        if self.cycles == 0:
            return float("nan")
        return 1.0 - self.cycles_with_stockout / self.cycles

    def to_dict(self) -> Dict:
        payload = asdict(self)
        payload.update(
            {
                "fill_rate": self.fill_rate,
                "stockout_day_rate": self.stockout_day_rate,
                "realised_cycle_service_level": self.realised_cycle_service_level,
            }
        )
        return payload


def simulate_unit(
    planning_unit: str,
    policy_name: str,
    demand: np.ndarray,
    leadtimes: np.ndarray,
    reorder_point: float,
    order_quantity: float,
    safety_stock: float,
    daily_holding_cost: float,
    ordering_cost: float,
    warmup_days: int = 0,
) -> SimulationResult:
    """Run one unit through one policy on a fixed demand and lead-time realisation.

    Order of operations within a day is receipt, then demand, then the replenishment
    decision. Receiving before serving demand is the standard convention and avoids
    recording a stockout on a day when the delivery had in fact already landed.

    The inventory position that the reorder point is compared against includes stock on
    order, not just stock on hand. Comparing on hand alone would re-order on every day of
    the lead time and pile up a pipeline several times the intended size.
    """
    if order_quantity <= 0:
        raise ValueError("order quantity must be positive")
    if len(demand) != len(leadtimes):
        raise ValueError("demand and lead-time arrays must be the same length")

    days = len(demand)
    on_hand = reorder_point + order_quantity / 2.0
    pipeline: Dict[int, float] = {}

    units_demanded = units_shipped = units_lost = 0.0
    stockout_days = orders_placed = 0
    cycles = cycles_with_stockout = 0
    on_hand_trace = np.zeros(days)
    cycle_open = False
    cycle_had_stockout = False

    for day in range(days):
        arrival = pipeline.pop(day, 0.0)
        on_hand += arrival

        today = demand[day]
        shipped = min(on_hand, today)
        lost = today - shipped
        on_hand -= shipped

        counted = day >= warmup_days
        if counted:
            units_demanded += today
            units_shipped += shipped
            units_lost += lost
            if lost > 0:
                stockout_days += 1
                cycle_had_stockout = True

        on_hand_trace[day] = on_hand

        on_order = sum(pipeline.values())
        position = on_hand + on_order
        if position <= reorder_point:
            lead = int(leadtimes[day])
            pipeline[day + lead] = pipeline.get(day + lead, 0.0) + order_quantity
            if counted:
                orders_placed += 1
                if cycle_open:
                    cycles += 1
                    cycles_with_stockout += int(cycle_had_stockout)
                cycle_open = True
                cycle_had_stockout = False

    if cycle_open:
        cycles += 1
        cycles_with_stockout += int(cycle_had_stockout)

    counted_trace = on_hand_trace[warmup_days:]
    mean_on_hand = float(counted_trace.mean()) if len(counted_trace) else 0.0
    holding = mean_on_hand * daily_holding_cost * len(counted_trace)
    ordering = orders_placed * ordering_cost

    return SimulationResult(
        planning_unit=planning_unit,
        policy=policy_name,
        reorder_point=float(reorder_point),
        order_quantity=float(order_quantity),
        safety_stock=float(safety_stock),
        days=len(counted_trace),
        units_demanded=float(units_demanded),
        units_shipped=float(units_shipped),
        units_lost=float(units_lost),
        stockout_days=stockout_days,
        orders_placed=orders_placed,
        cycles=cycles,
        cycles_with_stockout=cycles_with_stockout,
        mean_on_hand=mean_on_hand,
        holding_cost=float(holding),
        ordering_cost=float(ordering),
        total_cost=float(holding + ordering),
    )


def common_leadtime_draws(
    units: List[str],
    days: int,
    leadtime_values: np.ndarray,
    leadtime_probabilities: np.ndarray,
    seed: int,
) -> Dict[str, np.ndarray]:
    """One fixed lead-time realisation per unit, shared by every policy compared.

    Generated once and handed to each policy in turn. Any difference in outcome between
    policies is then attributable to the policy rather than to the draw.
    """
    rng = np.random.default_rng(seed)
    return {
        unit: rng.choice(leadtime_values, size=days, p=leadtime_probabilities)
        for unit in units
    }


def aggregate(results: List[SimulationResult]) -> pd.DataFrame:
    """Roll unit level results up to one row per policy.

    Fill rate is recomputed from summed units rather than averaged across units. Averaging
    a ratio across units of wildly different size would let a product selling under one a
    day carry the same weight as one selling seventy, which would flatter any policy that
    happens to do well on the tail.
    """
    frame = pd.DataFrame([r.to_dict() for r in results])
    grouped = frame.groupby("policy").agg(
        units=("planning_unit", "nunique"),
        units_demanded=("units_demanded", "sum"),
        units_shipped=("units_shipped", "sum"),
        units_lost=("units_lost", "sum"),
        stockout_days=("stockout_days", "sum"),
        simulated_days=("days", "sum"),
        cycles=("cycles", "sum"),
        cycles_with_stockout=("cycles_with_stockout", "sum"),
        total_safety_stock=("safety_stock", "sum"),
        mean_on_hand=("mean_on_hand", "sum"),
        holding_cost=("holding_cost", "sum"),
        ordering_cost=("ordering_cost", "sum"),
        total_cost=("total_cost", "sum"),
    )
    grouped["fill_rate"] = grouped["units_shipped"] / grouped["units_demanded"]
    grouped["stockout_day_rate"] = grouped["stockout_days"] / grouped["simulated_days"]
    grouped["realised_cycle_service_level"] = (
        1.0 - grouped["cycles_with_stockout"] / grouped["cycles"]
    )
    return grouped


MIN_CYCLES_FOR_SERVICE_ESTIMATE = 30


def segment_breakdown(results: List[SimulationResult], segments: pd.Series) -> pd.DataFrame:
    """Policy outcomes split by segment, since the aggregate hides where each one wins.

    Cycle service level is a proportion measured over completed replenishment cycles, so a
    segment that only completes a handful of them in the test window cannot support the
    estimate however precise the number looks. Slow moving units order rarely by
    construction, which is exactly the case where a reader is most likely to take a cycle
    service figure at face value, so the count and a sufficiency flag travel with it.
    """
    frame = pd.DataFrame([r.to_dict() for r in results])
    frame["segment"] = frame["planning_unit"].map(segments)
    grouped = frame.groupby(["policy", "segment"]).agg(
        units=("planning_unit", "nunique"),
        units_demanded=("units_demanded", "sum"),
        units_shipped=("units_shipped", "sum"),
        units_lost=("units_lost", "sum"),
        cycles=("cycles", "sum"),
        cycles_with_stockout=("cycles_with_stockout", "sum"),
        total_safety_stock=("safety_stock", "sum"),
        holding_cost=("holding_cost", "sum"),
        total_cost=("total_cost", "sum"),
    )
    grouped["fill_rate"] = grouped["units_shipped"] / grouped["units_demanded"]
    grouped["realised_cycle_service_level"] = (
        1.0 - grouped["cycles_with_stockout"] / grouped["cycles"]
    )
    grouped["cycles_per_unit"] = grouped["cycles"] / grouped["units"]
    grouped["cycle_service_estimable"] = grouped["cycles"] >= MIN_CYCLES_FOR_SERVICE_ESTIMATE
    grouped.loc[~grouped["cycle_service_estimable"], "realised_cycle_service_level"] = np.nan
    return grouped
