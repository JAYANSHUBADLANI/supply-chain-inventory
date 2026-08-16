"""Phase 3: safety stock, reorder point, EOQ and the service level tradeoff.

Runs on a daily time base against the units Phase 2 carried forward. Produces the policy
table, the variance decomposition that shows which uncertainty is actually driving the
buffer in each segment, a sensitivity sweep over the assumed lead-time variability, and
for one representative unit per segment both a cost/service curve and a check of the
closed form against the empirical distribution.
"""

from typing import Dict

import numpy as np
import pandas as pd

from src import data_io, figures, leadtime, policy, tradeoff

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
    policy_cfg = config.policy
    lt_cfg = config.leadtime

    proxy_frame = orders if lt_cfg.proxy_mode is None else orders.loc[
        orders["Shipping Mode"] == lt_cfg.proxy_mode
    ]
    lead = leadtime.pooled_parameters(proxy_frame)
    lead_distribution = leadtime.empirical_distribution(proxy_frame)
    lead_values = lead_distribution.index.values.astype(float)
    lead_probs = lead_distribution.values

    daily = data_io.daily_demand_panel(orders, config)
    units = [u for u in grid.index if u in daily.index]
    prices = data_io.unit_prices(orders, config)

    rows = []
    for unit in units:
        series = daily.loc[unit].values.astype(float)
        demand_mean = float(series.mean())
        demand_std = float(series.std(ddof=1))
        service_level = float(grid.loc[unit, "target_service_level"])
        unit_price = float(prices.loc[unit])

        ss = policy.safety_stock(
            planning_unit=unit,
            demand_mean=demand_mean,
            demand_std=demand_std,
            leadtime_mean=lead.mean_days,
            leadtime_std=lead.std_days,
            service_level=service_level,
        )
        annual_demand = demand_mean * DAYS_PER_YEAR
        eoq = policy.economic_order_quantity(
            annual_demand=annual_demand,
            ordering_cost=costs.ordering_cost_per_order,
            annual_holding_cost_per_unit=costs.annual_holding_cost(unit_price),
        )
        cost_block = policy.total_annual_policy_cost(
            annual_demand=annual_demand,
            order_quantity=eoq,
            safety_stock_units=ss.safety_stock,
            unit_price=unit_price,
            costs=costs,
        )
        record = ss.to_dict()
        record.update(
            {
                "segment": grid.loc[unit, "segment"],
                "abc_class": grid.loc[unit, "abc_class"],
                "xyz_class": grid.loc[unit, "xyz_class"],
                "unit_price": unit_price,
                "annual_demand": annual_demand,
                "expected_demand_over_leadtime": policy.expected_demand_over_leadtime(
                    demand_mean, lead.mean_days
                ),
                "reorder_point": policy.reorder_point(
                    demand_mean, lead.mean_days, ss.safety_stock
                ),
                "economic_order_quantity": eoq,
                "days_of_cover_at_eoq": eoq / demand_mean if demand_mean > 0 else np.nan,
                "expected_fill_rate": policy.expected_fill_rate(
                    ss.sigma_demand_over_leadtime, ss.z, eoq
                ),
            }
        )
        record.update(cost_block)
        rows.append(record)

    policy_table = pd.DataFrame(rows).set_index("planning_unit")
    data_io.write_table(policy_table, tables / "policy_table.csv")

    decomposition = (
        policy_table.groupby("segment")
        .agg(
            units=("safety_stock", "size"),
            median_leadtime_share=("leadtime_share_of_variance", "median"),
            median_demand_share=("demand_share_of_variance", "median"),
            total_safety_stock=("safety_stock", "sum"),
            total_safety_stock_holding=("safety_stock_holding_cost", "sum"),
            median_understatement=("understatement_vs_demand_only", "median"),
        )
        .round(4)
    )
    data_io.write_table(decomposition, tables / "variance_decomposition_by_segment.csv")

    sensitivity = _sensitivity(policy_table, lead, lt_cfg.sensitivity_multipliers, costs)
    data_io.write_table(sensitivity, tables / "leadtime_sensitivity.csv", index=False)

    representatives = _representatives(policy_table, policy_cfg.representative_segments)
    curves, comparisons = [], []
    rng = np.random.default_rng(policy_cfg.bootstrap_seed)
    for segment, unit in representatives.items():
        row = policy_table.loc[unit]
        curve = tradeoff.service_level_curve(
            planning_unit=unit,
            demand_mean=row["demand_mean"],
            demand_std=row["demand_std"],
            leadtime_mean=lead.mean_days,
            leadtime_std=lead.std_days,
            unit_price=row["unit_price"],
            annual_demand=row["annual_demand"],
            order_quantity=row["economic_order_quantity"],
            service_levels=policy_cfg.service_level_sweep,
            costs=costs,
        )
        curve.insert(0, "segment", segment)
        curves.append(curve)

        comparison = tradeoff.normal_versus_empirical(
            planning_unit=unit,
            daily_demand=daily.loc[unit].values.astype(float),
            leadtime_values=lead_values,
            leadtime_probabilities=lead_probs,
            demand_mean=row["demand_mean"],
            demand_std=row["demand_std"],
            leadtime_mean=lead.mean_days,
            leadtime_std=lead.std_days,
            service_levels=policy_cfg.service_level_sweep,
            draws=policy_cfg.bootstrap_draws,
            rng=rng,
        )
        comparison.insert(0, "segment", segment)
        comparisons.append(comparison)

    curve_table = pd.concat(curves, ignore_index=True)
    comparison_table = pd.concat(comparisons, ignore_index=True)
    data_io.write_table(curve_table, tables / "service_level_curve.csv", index=False)
    data_io.write_table(comparison_table, tables / "normal_versus_empirical.csv", index=False)

    knees = {
        segment: tradeoff.knee_point(curve_table[curve_table["segment"] == segment])
        for segment in representatives
    }

    basis = tradeoff.differentiation_basis(policy_table)
    data_io.write_table(basis, tables / "differentiation_basis.csv")

    charts = reports / "figures"
    figures.variance_decomposition(decomposition, charts / "variance_decomposition.png")
    figures.normal_versus_empirical(comparison_table, charts / "normal_versus_empirical.png")
    figures.leadtime_sensitivity(sensitivity, charts / "leadtime_sensitivity.png")
    for segment in representatives:
        figures.service_level_curve(
            curve_table, segment, charts / f"service_level_curve_{segment}.png"
        )

    data_io.cache_frame(policy_table, interim / "policy_table.parquet")

    findings = {
        "leadtime_proxy": lead.to_dict(),
        "leadtime_proxy_mode": lt_cfg.proxy_mode,
        "cost_assumptions": {
            "annual_holding_rate": costs.annual_holding_rate,
            "ordering_cost_per_order": costs.ordering_cost_per_order,
            "unit_cost_as_share_of_price": costs.unit_cost_as_share_of_price,
            "note": "assumed, not observed; the extract carries no cost structure",
        },
        "variance_decomposition": decomposition.reset_index().to_dict(orient="records"),
        "representatives": representatives,
        "knee_points": knees,
        "knee_caveat": (
            "safety stock is proportional to the normal quantile of the target, so the "
            "marginal cost curve has one shape for every unit and the knee cannot "
            "differentiate segments; see differentiation_basis"
        ),
        "differentiation_basis": basis.reset_index().to_dict(orient="records"),
        "sensitivity": sensitivity.to_dict(orient="records"),
    }
    data_io.write_json(findings, reports / "phase3_findings.json")

    ax_units = policy_table[policy_table["segment"] == "AX"]
    cz_units = policy_table[policy_table["segment"] == "CZ"]
    return {
        "units_with_policy": len(policy_table),
        "leadtime_proxy": f"{lead.mean_days:.3f} +/- {lead.std_days:.3f} days ({lt_cfg.proxy_mode})",
        "median_leadtime_share_AX": round(float(ax_units["leadtime_share_of_variance"].median()), 4),
        "median_leadtime_share_CZ": round(float(cz_units["leadtime_share_of_variance"].median()), 4),
        "median_understatement_AX": round(float(ax_units["understatement_vs_demand_only"].median()), 3),
        "median_understatement_CZ": round(float(cz_units["understatement_vs_demand_only"].median()), 3),
        "total_safety_stock_units": round(float(policy_table["safety_stock"].sum()), 1),
        "total_annual_holding_cost": round(
            float(policy_table["safety_stock_holding_cost"].sum()), 2
        ),
        "representatives": representatives,
        "knee_service_levels": {s: k["knee_service_level"] for s, k in knees.items()},
        "buffer_cost_per_1000_revenue_by_segment": basis[
            "median_buffer_cost_per_1000_revenue"
        ].to_dict(),
    }


def _representatives(policy_table: pd.DataFrame, segments) -> Dict[str, str]:
    """Highest revenue unit in each segment named in config, for the detailed curves."""
    chosen = {}
    for segment in segments:
        block = policy_table[policy_table["segment"] == segment]
        if len(block) == 0:
            continue
        chosen[segment] = str(block["annual_demand"].idxmax())
    return chosen


def _sensitivity(policy_table: pd.DataFrame, lead, multipliers, costs) -> pd.DataFrame:
    """How the whole book's safety stock responds to the assumed lead-time variability.

    sigma_L is a proxy parameter rather than a measurement, so the defensible claim is not
    a point estimate but the range of answers it produces. A multiplier of zero is the
    deterministic lead-time case, which is what the simplified formula assumes.
    """
    rows = []
    for multiplier in multipliers:
        scaled = lead.scaled(multiplier)
        total_ss, total_cost, shares = 0.0, 0.0, []
        for unit, row in policy_table.iterrows():
            result = policy.safety_stock(
                planning_unit=str(unit),
                demand_mean=row["demand_mean"],
                demand_std=row["demand_std"],
                leadtime_mean=scaled.mean_days,
                leadtime_std=scaled.std_days,
                service_level=row["service_level"],
            )
            total_ss += result.safety_stock
            total_cost += policy.annual_holding_cost(
                result.safety_stock, row["unit_price"], costs
            )
            shares.append(result.leadtime_share_of_variance)
        rows.append(
            {
                "sigma_multiplier": multiplier,
                "leadtime_std_days": scaled.std_days,
                "total_safety_stock_units": total_ss,
                "total_annual_holding_cost": total_cost,
                "median_leadtime_share_of_variance": float(np.nanmedian(shares)),
            }
        )
    frame = pd.DataFrame(rows)
    base = frame.loc[frame["sigma_multiplier"] == 1.0, "total_safety_stock_units"]
    if len(base) == 1:
        frame["safety_stock_vs_base"] = frame["total_safety_stock_units"] / float(base.iloc[0])
    return frame
