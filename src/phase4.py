"""Phase 4: backtest the policy on a held-out window against two alternatives.

Three policies are compared on identical demand and identical lead-time realisations.

The segmented policy is the one this project argues for: safety stock from the combined
formula carrying both demand and lead-time variance, with the service target set by the
unit's ABC tier.

The demand-only policy is the widely quoted simplification that drops the lead-time
variance term. It is included because it is what most implementations actually use, so it
is the comparison that says whether the extra term earns its place.

The flat baseline holds a fixed percentage of expected lead-time demand as buffer,
identical treatment for every unit. It is the do-nothing-clever option.

Every parameter for all three is estimated on the fit window alone.
"""

from typing import Dict

import numpy as np
import pandas as pd

from src import data_io, figures, leadtime, policy, simulate

DAYS_PER_YEAR = 365.0


def run(config, force: bool = False) -> Dict:
    reports = config.path("reports_dir")
    tables = reports / "tables"
    interim = config.path("interim_dir")

    orders = data_io.load_cached(interim / "clean_orders.parquet")
    grid = data_io.load_cached(interim / "policy_inputs.parquet")
    if orders is None or grid is None:
        raise FileNotFoundError("phase 2 outputs missing, run: python run.py 2")

    costs = config.costs
    bt = config.backtest
    lt_cfg = config.leadtime

    proxy_frame = orders if lt_cfg.proxy_mode is None else orders.loc[
        orders["Shipping Mode"] == lt_cfg.proxy_mode
    ]
    lead = leadtime.pooled_parameters(proxy_frame)
    distribution = leadtime.empirical_distribution(proxy_frame)
    lead_values = distribution.index.values.astype(float)
    lead_probs = distribution.values

    daily = data_io.daily_demand_panel(orders, config)
    units = [u for u in grid.index if u in daily.index]
    prices = data_io.unit_prices(orders, config)

    total_days = daily.shape[1]
    test_days = bt.test_days
    if test_days >= total_days:
        raise ValueError(
            f"test window of {test_days} days does not fit inside {total_days} days of history"
        )
    split = total_days - test_days
    fit_window = daily.iloc[:, :split]
    test_window = daily.iloc[:, split:]

    parameters = _fit_parameters(
        units, fit_window, grid, prices, lead, costs, bt.baseline_safety_stock_pct_of_mean_demand
    )
    data_io.write_table(parameters, tables / "backtest_policy_parameters.csv")

    draws = simulate.common_leadtime_draws(
        units=units,
        days=test_days,
        leadtime_values=lead_values,
        leadtime_probabilities=lead_probs,
        seed=bt.leadtime_seed,
    )

    results = []
    for unit in units:
        demand = test_window.loc[unit].values.astype(float)
        row = parameters.loc[unit]
        unit_price = float(prices.loc[unit])
        daily_holding = costs.annual_holding_cost(unit_price) / DAYS_PER_YEAR
        for policy_name in ("segmented", "demand_only", "flat_baseline"):
            results.append(
                simulate.simulate_unit(
                    planning_unit=unit,
                    policy_name=policy_name,
                    demand=demand,
                    leadtimes=draws[unit],
                    reorder_point=float(row[f"{policy_name}_reorder_point"]),
                    order_quantity=float(row["order_quantity"]),
                    safety_stock=float(row[f"{policy_name}_safety_stock"]),
                    daily_holding_cost=daily_holding,
                    ordering_cost=costs.ordering_cost_per_order,
                    warmup_days=bt.warmup_days,
                )
            )

    unit_results = pd.DataFrame([r.to_dict() for r in results])
    data_io.write_table(unit_results, tables / "backtest_unit_results.csv", index=False)

    summary = simulate.aggregate(results)
    data_io.write_table(summary, tables / "backtest_summary.csv")

    breakdown = simulate.segment_breakdown(results, grid["segment"])
    data_io.write_table(breakdown, tables / "backtest_by_segment.csv")

    figures.backtest_service_cost(summary, reports / "figures" / "backtest_service_cost.png")

    comparison = _comparison(summary)
    thin = breakdown[~breakdown["cycle_service_estimable"]]
    thin_segments = sorted(thin.index.get_level_values("segment").unique().tolist())

    findings = {
        "split": {
            "total_days": int(total_days),
            "fit_days": int(split),
            "test_days": int(test_days),
            "fit_window": [str(fit_window.columns.min().date()), str(fit_window.columns.max().date())],
            "test_window": [str(test_window.columns.min().date()), str(test_window.columns.max().date())],
            "warmup_days_excluded_from_metrics": bt.warmup_days,
        },
        "leadtime_proxy": lead.to_dict(),
        "policies": {
            "segmented": "combined demand and lead-time variance, ABC tiered service level",
            "demand_only": "demand variance only, ABC tiered service level",
            "flat_baseline": (
                f"{bt.baseline_safety_stock_pct_of_mean_demand:.0%} of expected lead-time "
                f"demand, identical for every unit"
            ),
        },
        "summary": summary.reset_index().to_dict(orient="records"),
        "comparison": comparison,
        "by_segment": breakdown.reset_index().to_dict(orient="records"),
        "segments_without_estimable_cycle_service": {
            "segments": thin_segments,
            "minimum_cycles_required": simulate.MIN_CYCLES_FOR_SERVICE_ESTIMATE,
            "reason": (
                "slow moving units order rarely, so the test window completes too few "
                "replenishment cycles to estimate a cycle service level; fill rate is the "
                "reliable service measure for these segments"
            ),
        },
    }
    data_io.write_json(findings, reports / "phase4_findings.json")

    return {
        "fit_window": f"{fit_window.columns.min().date()} to {fit_window.columns.max().date()}",
        "test_window": f"{test_window.columns.min().date()} to {test_window.columns.max().date()}",
        "units_simulated": len(units),
        "fill_rate": {p: round(float(summary.loc[p, "fill_rate"]), 5) for p in summary.index},
        "realised_cycle_service_level": {
            p: round(float(summary.loc[p, "realised_cycle_service_level"]), 4) for p in summary.index
        },
        "total_safety_stock_units": {
            p: round(float(summary.loc[p, "total_safety_stock"]), 1) for p in summary.index
        },
        "holding_cost": {p: round(float(summary.loc[p, "holding_cost"]), 2) for p in summary.index},
        "units_lost": {p: round(float(summary.loc[p, "units_lost"]), 1) for p in summary.index},
        "comparison": comparison,
        "segments_without_estimable_cycle_service": thin_segments,
    }


def _fit_parameters(
    units,
    fit_window: pd.DataFrame,
    grid: pd.DataFrame,
    prices: pd.Series,
    lead,
    costs,
    baseline_pct: float,
) -> pd.DataFrame:
    """Estimate every policy's parameters on the fit window and nothing else."""
    rows = []
    for unit in units:
        series = fit_window.loc[unit].values.astype(float)
        demand_mean = float(series.mean())
        demand_std = float(series.std(ddof=1))
        service_level = float(grid.loc[unit, "target_service_level"])
        unit_price = float(prices.loc[unit])

        combined = policy.safety_stock(
            planning_unit=unit,
            demand_mean=demand_mean,
            demand_std=demand_std,
            leadtime_mean=lead.mean_days,
            leadtime_std=lead.std_days,
            service_level=service_level,
        )
        demand_only_ss = combined.demand_only_safety_stock
        flat_ss = baseline_pct * demand_mean * lead.mean_days

        order_quantity = policy.economic_order_quantity(
            annual_demand=demand_mean * DAYS_PER_YEAR,
            ordering_cost=costs.ordering_cost_per_order,
            annual_holding_cost_per_unit=costs.annual_holding_cost(unit_price),
        )
        expected_ldt = policy.expected_demand_over_leadtime(demand_mean, lead.mean_days)

        rows.append(
            {
                "planning_unit": unit,
                "segment": grid.loc[unit, "segment"],
                "target_service_level": service_level,
                "fit_demand_mean": demand_mean,
                "fit_demand_std": demand_std,
                "expected_leadtime_demand": expected_ldt,
                "order_quantity": order_quantity,
                "segmented_safety_stock": combined.safety_stock,
                "segmented_reorder_point": expected_ldt + combined.safety_stock,
                "demand_only_safety_stock": demand_only_ss,
                "demand_only_reorder_point": expected_ldt + demand_only_ss,
                "flat_baseline_safety_stock": flat_ss,
                "flat_baseline_reorder_point": expected_ldt + flat_ss,
                "leadtime_share_of_variance": combined.leadtime_share_of_variance,
            }
        )
    return pd.DataFrame(rows).set_index("planning_unit")


def _comparison(summary: pd.DataFrame) -> Dict:
    """Segmented policy against each alternative, on service and on what it cost."""
    out = {}
    base = summary.loc["segmented"]
    for other in summary.index:
        if other == "segmented":
            continue
        rival = summary.loc[other]
        out[f"segmented_vs_{other}"] = {
            "fill_rate_gain_pp": round(
                float(base["fill_rate"] - rival["fill_rate"]) * 100, 4
            ),
            "units_lost_avoided": round(float(rival["units_lost"] - base["units_lost"]), 1),
            "extra_safety_stock_units": round(
                float(base["total_safety_stock"] - rival["total_safety_stock"]), 1
            ),
            "extra_holding_cost": round(
                float(base["holding_cost"] - rival["holding_cost"]), 2
            ),
            "extra_total_cost": round(float(base["total_cost"] - rival["total_cost"]), 2),
        }
    return out
