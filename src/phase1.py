"""Phase 1: clean the extract and audit it before any policy work starts.

The audit runs against the full raw extract, not the cleaned window, because the point
is to find the boundary rather than assume it. The window cut is then applied using what
the audit found.
"""

from pathlib import Path
from typing import Dict

import pandas as pd

from src import audit, data_io, figures


def run(config, force: bool = False) -> Dict:
    reports = config.path("reports_dir")
    tables = reports / "tables"
    interim = config.path("interim_dir")

    frame = data_io.load_raw(config)

    monthly = audit.structural_break_scan(frame, config)
    data_io.write_table(monthly, tables / "monthly_profile.csv")

    audit_block = config.section("audit")
    collapse = audit.locate_quantity_collapse(monthly, audit_block["quantity_collapse_threshold"])

    boundary = config.section("data")["clean_window_end"]
    turnover = audit.catalogue_turnover(frame, config, boundary)

    mode_tests = audit.leadtime_by_mode(
        frame, audit_block["leadtime_uniformity_alpha"], audit_block["min_rows_for_mode_test"]
    )
    mode_table = pd.DataFrame([t.to_dict() for t in mode_tests])
    data_io.write_table(mode_table, tables / "leadtime_by_mode.csv", index=False)

    volumes = frame["Shipping Mode"].value_counts()
    top_two = list(volumes.head(2).index)
    homogeneity = audit.mode_pair_homogeneity(frame, top_two[0], top_two[1])

    regional = audit.leadtime_regional_signal(frame, within_mode=volumes.idxmax())
    derived = audit.derived_column_check(frame)

    kept, ledger = data_io.apply_exclusions(frame, config)
    data_io.write_json(ledger.to_dict(), reports / "exclusion_ledger.json")

    panel = data_io.weekly_demand_panel(kept, config)
    coverage = data_io.coverage(panel)
    gaps = data_io.interior_gaps(panel)
    prices = data_io.unit_prices(kept, config)
    revenue = data_io.revenue(kept, config)

    profile = pd.concat([coverage, gaps, prices, revenue], axis=1)
    profile["revenue_share"] = (profile["revenue"] / profile["revenue"].sum()).round(5)
    profile = profile.sort_values("revenue", ascending=False)
    data_io.write_table(profile, tables / "planning_unit_profile.csv")

    demand_cfg = config.demand
    core = profile.index[
        (profile["weeks_covered"] >= demand_cfg.core_min_weeks_covered) & (profile["interior_gaps"] == 0)
    ]
    eligible = profile.index[profile["weeks_covered"] >= demand_cfg.min_weeks_covered]

    concurrency = audit.market_concurrency(kept)
    data_io.write_table(concurrency, tables / "market_concurrency.csv")

    autocorr = audit.demand_autocorrelation(panel.loc[core])

    data_io.cache_frame(panel, interim / "weekly_demand_panel.parquet")
    data_io.cache_frame(profile, interim / "planning_unit_profile.parquet")
    data_io.cache_frame(
        kept[["order_date", "week", "Product Name", "Market", "Order Region",
              "Shipping Mode", "Order Item Quantity", "Sales", "Product Price",
              audit.REAL_LEADTIME, audit.SCHEDULED_LEADTIME]],
        interim / "clean_orders.parquet",
    )

    charts = reports / "figures"
    figures.leadtime_by_mode(frame, charts / "leadtime_by_mode.png")
    figures.demand_series(panel, list(core), charts / "weekly_demand_core.png")

    findings = {
        "leadtime": {
            "modes": [t.to_dict() for t in mode_tests],
            "degenerate_modes": [t.shipping_mode for t in mode_tests if t.is_degenerate],
            "uniform_modes": [t.shipping_mode for t in mode_tests if t.reads_as_uniform],
            "top_mode_homogeneity": homogeneity,
            "regional_signal": regional,
            "verdict": _leadtime_verdict(mode_tests, homogeneity, regional),
        },
        "structural_break": {
            "quantity_collapse": collapse,
            "catalogue_turnover": turnover,
        },
        "derived_columns": derived,
        "market_concurrency": {
            "markets": int(len(concurrency)),
            "total_weeks": int(concurrency["total_weeks"].iloc[0]),
            "max_share_of_weeks_active": float(concurrency["share_of_weeks_active"].max()),
            "all_markets_concurrent": bool((concurrency["share_of_weeks_active"] > 0.95).all()),
        },
        "aggregate_demand": autocorr,
    }
    data_io.write_json(findings, reports / "audit_findings.json")

    return {
        "raw_rows": ledger.raw_rows,
        "kept_rows": ledger.kept_rows,
        "clean_window": f"{ledger.clean_window_start} to {ledger.clean_window_end}",
        "weeks": ledger.weeks_kept,
        "products_in_window": ledger.products_kept,
        "eligible_planning_units": len(eligible),
        "core_planning_units": len(core),
        "core_revenue_share": round(float(profile.loc[core, "revenue"].sum() / profile["revenue"].sum()), 4),
        "leadtime_verdict": findings["leadtime"]["verdict"],
        "quantity_collapse_from": collapse.get("first_collapsed_month"),
        "max_abs_demand_autocorrelation": round(autocorr["max_abs_autocorrelation"], 4),
    }


def _leadtime_verdict(mode_tests, homogeneity: Dict, regional: Dict) -> str:
    """One line summary of whether the lead-time column can support empirical estimation."""
    degenerate = [t.shipping_mode for t in mode_tests if t.is_degenerate]
    uniform = [t.shipping_mode for t in mode_tests if t.reads_as_uniform]
    if not degenerate and not uniform and regional["regional_signal_detected"]:
        return "lead time carries usable structure"
    parts = []
    if degenerate:
        parts.append(f"{len(degenerate)} mode(s) have zero realised variance")
    if uniform:
        parts.append(f"{len(uniform)} mode(s) are indistinguishable from a discrete uniform draw")
    if homogeneity["same_distribution"]:
        parts.append(
            f"{homogeneity['left_mode']} and {homogeneity['right_mode']} share one distribution "
            f"despite scheduling {homogeneity['left_scheduled']:.0f} and "
            f"{homogeneity['right_scheduled']:.0f} days"
        )
    if not regional["regional_signal_detected"]:
        parts.append(
            f"region explains {regional['variance_explained_pct']:.2f}% of lead-time variance"
        )
    return "generated, not observed: " + "; ".join(parts)
