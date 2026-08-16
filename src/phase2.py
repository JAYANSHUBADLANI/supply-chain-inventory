"""Phase 2: demand characterisation, lead-time parameters and the ABC/XYZ grid.

ABC runs over every planning unit that appears in the clean window, because the revenue
Pareto of the catalogue is a fact about the catalogue and excluding the tail would flatter
it. XYZ runs only where a coefficient of variation can actually be estimated, and only
units whose variance estimate survives the observation and intermittency checks are
carried into the policy stage.
"""

from typing import Dict

import pandas as pd

from src import data_io, demand, leadtime, segmentation

SENSITIVITY_MULTIPLIERS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]


def run(config, force: bool = False) -> Dict:
    reports = config.path("reports_dir")
    tables = reports / "tables"
    interim = config.path("interim_dir")

    panel = data_io.load_cached(interim / "weekly_demand_panel.parquet")
    profile = data_io.load_cached(interim / "planning_unit_profile.parquet")
    orders = data_io.load_cached(interim / "clean_orders.parquet")
    if panel is None or profile is None or orders is None:
        raise FileNotFoundError("phase 1 outputs missing, run: python run.py 1")

    demand_cfg = config.demand
    seg_cfg = config.segmentation

    eligible = profile.index[profile["weeks_covered"] >= demand_cfg.min_weeks_covered]
    stats = demand.demand_statistics(panel, min_observations=demand_cfg.min_weeks_covered)
    data_io.write_table(stats, tables / "demand_statistics.csv")
    demand_ledger = demand.exclusion_ledger(stats)

    pooled = leadtime.pooled_parameters(orders)
    by_mode = leadtime.parameters_by_mode(orders)
    mode_table = pd.DataFrame([p.to_dict() for p in by_mode])
    data_io.write_table(mode_table, tables / "leadtime_parameters_by_mode.csv", index=False)

    schedule_gap = leadtime.scheduled_versus_real(orders)
    data_io.write_table(schedule_gap, tables / "scheduled_versus_real.csv")

    distribution = leadtime.empirical_distribution(orders)
    data_io.write_table(distribution.to_frame(), tables / "leadtime_distribution.csv")

    sensitivity = leadtime.sensitivity_grid(pooled, SENSITIVITY_MULTIPLIERS)
    data_io.write_table(
        pd.DataFrame([p.to_dict() for p in sensitivity]),
        tables / "leadtime_sensitivity_grid.csv",
        index=False,
    )

    revenue = profile["revenue"]
    cv = stats["coefficient_of_variation"]
    grid = segmentation.build_grid(
        revenue=revenue,
        coefficient_of_variation=cv,
        abc_thresholds=seg_cfg.abc_thresholds,
        xyz_thresholds=seg_cfg.xyz_thresholds,
        service_levels=seg_cfg.service_levels,
    )
    grid = grid.join(
        stats[
            [
                "observations",
                "weekly_mean",
                "weekly_std",
                "zero_week_share",
                "variance_estimate_trustworthy",
            ]
        ]
    )
    grid["carried_to_policy"] = grid["variance_estimate_trustworthy"].fillna(False)
    data_io.write_table(grid, tables / "abc_xyz_grid.csv")

    summary = segmentation.grid_summary(grid)
    data_io.write_table(summary, tables / "abc_xyz_summary.csv", index=False)

    concentration = segmentation.concentration_profile(revenue)
    degeneracy = segmentation.degeneracy_flags(grid)

    policy_units = grid.index[grid["carried_to_policy"]]
    policy_inputs = grid.loc[policy_units].join(
        stats.loc[policy_units, ["weekly_variance"]]
    )
    data_io.cache_frame(policy_inputs, interim / "policy_inputs.parquet")
    data_io.cache_frame(stats, interim / "demand_statistics.parquet")

    findings = {
        "demand": {
            "ledger": demand_ledger,
            "eligible_by_coverage": int(len(eligible)),
            "median_cv_carried": float(cv.loc[policy_units].median()),
            "cv_range_carried": [
                float(cv.loc[policy_units].min()),
                float(cv.loc[policy_units].max()),
            ],
        },
        "leadtime": {
            "pooled_proxy": pooled.to_dict(),
            "by_mode": [p.to_dict() for p in by_mode],
            "scheduled_versus_real": schedule_gap.reset_index().to_dict(orient="records"),
            "sensitivity_multipliers": SENSITIVITY_MULTIPLIERS,
        },
        "segmentation": {
            "concentration": concentration,
            "degeneracy": degeneracy,
            "service_levels": seg_cfg.service_levels,
            "cells": summary.to_dict(orient="records"),
        },
    }
    data_io.write_json(findings, reports / "phase2_findings.json")

    return {
        "planning_units_profiled": len(stats),
        "carried_to_policy": len(policy_units),
        "carried_revenue_share": round(
            float(revenue.loc[policy_units].sum() / revenue.sum()), 4
        ),
        "demand_cv_range": [
            round(float(cv.loc[policy_units].min()), 3),
            round(float(cv.loc[policy_units].max()), 3),
        ],
        "leadtime_proxy_mean_days": round(pooled.mean_days, 3),
        "leadtime_proxy_std_days": round(pooled.std_days, 3),
        "leadtime_proxy_cv": round(pooled.coefficient_of_variation, 3),
        "units_for_80pct_revenue": concentration["units_for_80pct_revenue"],
        "share_of_units_for_80pct_revenue": concentration["share_of_units_for_80pct_revenue"],
        "populated_grid_cells": degeneracy["populated_cells"],
        "a_tier_collapses_to_one_xyz_class": degeneracy["a_tier_collapses_to_one_xyz_class"],
    }
