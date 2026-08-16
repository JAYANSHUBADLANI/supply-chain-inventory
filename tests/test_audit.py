"""Tests that the audit detectors fire on known-synthetic input and stay quiet otherwise.

The audit is the load bearing part of this project: its conclusion is what redirected the
design away from treating the lead-time column as observed. A detector that only ever
returns the answer I already believed would be worthless, so each one is tested against
both a case it should flag and a case it should not.
"""

import numpy as np
import pandas as pd
import pytest

from src import audit
from tests.conftest import make_orders


class TestLeadTimeByMode:
    def test_flags_constant_leadtime_as_degenerate(self, degenerate_leadtime_orders):
        results = audit.leadtime_by_mode(degenerate_leadtime_orders, alpha=0.01, min_rows=100)
        assert len(results) == 1
        assert results[0].is_degenerate
        assert results[0].observed_std == 0.0
        assert results[0].support == [2]

    def test_flags_uniform_draw_as_uniform(self, uniform_leadtime_orders):
        results = audit.leadtime_by_mode(uniform_leadtime_orders, alpha=0.01, min_rows=100)
        assert all(r.reads_as_uniform for r in results)
        assert not any(r.is_degenerate for r in results)

    def test_does_not_flag_centred_distribution_as_uniform(self, structured_leadtime_orders):
        results = audit.leadtime_by_mode(structured_leadtime_orders, alpha=0.01, min_rows=100)
        assert not results[0].reads_as_uniform
        assert not results[0].is_degenerate

    def test_late_share_uses_the_mode_own_schedule(self, degenerate_leadtime_orders):
        results = audit.leadtime_by_mode(degenerate_leadtime_orders, alpha=0.01, min_rows=100)
        assert results[0].late_share == 1.0

    def test_small_modes_are_not_tested_for_uniformity(self):
        frame = make_orders(
            n_per_mode={"Same Day": 50},
            real_leadtime={"Same Day": lambda rng, n: rng.integers(0, 2, n)},
            scheduled={"Same Day": 0},
        )
        results = audit.leadtime_by_mode(frame, alpha=0.01, min_rows=500)
        assert not results[0].reads_as_uniform
        assert np.isnan(results[0].uniform_p_value)


class TestModePairHomogeneity:
    def test_identifies_two_modes_drawn_from_one_law(self, uniform_leadtime_orders):
        result = audit.mode_pair_homogeneity(
            uniform_leadtime_orders, "Standard Class", "Second Class"
        )
        assert result["same_distribution"]
        assert result["left_scheduled"] != result["right_scheduled"]

    def test_separates_genuinely_different_modes(self):
        frame = make_orders(
            n_per_mode={"Standard Class": 4000, "Second Class": 4000},
            real_leadtime={
                "Standard Class": lambda rng, n: rng.integers(4, 7, n),
                "Second Class": lambda rng, n: rng.integers(2, 5, n),
            },
            scheduled={"Standard Class": 4, "Second Class": 2},
            seed=4,
        )
        result = audit.mode_pair_homogeneity(frame, "Standard Class", "Second Class")
        assert not result["same_distribution"]


class TestRegionalSignal:
    def test_reports_no_signal_when_region_is_irrelevant(self, uniform_leadtime_orders):
        frame = pd.concat([uniform_leadtime_orders] * 4, ignore_index=True)
        result = audit.leadtime_regional_signal(frame, within_mode="Standard Class")
        assert not result["regional_signal_detected"]
        assert result["eta_squared"] < 0.01

    def test_detects_a_real_regional_effect(self):
        rng = np.random.default_rng(11)
        rows = []
        for region, centre in [("Near", 2), ("Far", 6)]:
            for _ in range(2000):
                rows.append(
                    {
                        "Shipping Mode": "Standard Class",
                        "Days for shipping (real)": int(np.clip(rng.normal(centre, 0.7), 0, 9)),
                        "Order Region": region,
                    }
                )
        frame = pd.DataFrame(rows)
        result = audit.leadtime_regional_signal(frame, within_mode="Standard Class")
        assert result["regional_signal_detected"]
        assert result["eta_squared"] > 0.5

    def test_effect_size_is_reported_independently_of_significance(self):
        rng = np.random.default_rng(12)
        rows = []
        for region, centre in [("A", 4.00), ("B", 4.03)]:
            for _ in range(40000):
                rows.append(
                    {
                        "Shipping Mode": "Standard Class",
                        "Days for shipping (real)": float(rng.normal(centre, 1.4)),
                        "Order Region": region,
                    }
                )
        frame = pd.DataFrame(rows)
        result = audit.leadtime_regional_signal(frame, within_mode="Standard Class")
        assert result["eta_squared"] < 0.01
        assert not result["regional_signal_detected"]


class TestDerivedColumnCheck:
    def test_identifies_a_reconstructable_column(self, uniform_leadtime_orders):
        result = audit.derived_column_check(uniform_leadtime_orders)
        assert result["is_derived"]
        assert result["agreement"] == pytest.approx(1.0)

    def test_does_not_flag_an_independent_column(self, uniform_leadtime_orders):
        frame = uniform_leadtime_orders.copy()
        rng = np.random.default_rng(5)
        frame["Late_delivery_risk"] = rng.integers(0, 2, len(frame))
        result = audit.derived_column_check(frame)
        assert not result["is_derived"]


class TestStructuralBreak:
    def _monthly(self, share_by_month):
        return pd.DataFrame(
            {"share_quantity_one": list(share_by_month.values()),
             "order_lines": [5000] * len(share_by_month),
             "distinct_products": [50] * len(share_by_month)},
            index=pd.PeriodIndex(list(share_by_month), freq="M"),
        )

    def test_locates_the_first_collapsed_month(self):
        monthly = self._monthly(
            {"2017-07": 0.56, "2017-08": 0.57, "2017-09": 0.58, "2017-10": 0.96, "2017-11": 1.0}
        )
        result = audit.locate_quantity_collapse(monthly, threshold=0.95)
        assert result["collapse_detected"]
        assert result["first_collapsed_month"] == "2017-10"
        assert result["months_affected"] == 2
        assert result["persists_to_end"]

    def test_reports_nothing_when_quantity_keeps_varying(self):
        monthly = self._monthly({"2017-07": 0.5, "2017-08": 0.52, "2017-09": 0.51})
        result = audit.locate_quantity_collapse(monthly, threshold=0.95)
        assert not result["collapse_detected"]
        assert result["first_collapsed_month"] is None

    def test_catalogue_turnover_counts_shared_products(self, config):
        frame = pd.DataFrame(
            {
                "order_date": pd.to_datetime(
                    ["2017-01-01", "2017-01-02", "2017-11-01", "2017-11-02"]
                ),
                "Product Name": ["old_a", "old_b", "old_a", "new_c"],
            }
        )
        result = audit.catalogue_turnover(frame, config, "2017-10-01")
        assert result["products_before"] == 2
        assert result["products_after"] == 2
        assert result["shared"] == 1
        assert result["after_only"] == 1


class TestMarketConcurrency:
    def test_flags_time_sliced_markets(self):
        weeks = pd.date_range("2015-01-05", periods=20, freq="W-MON")
        rows = [{"Market": "LATAM", "week": w} for w in weeks[:10]]
        rows += [{"Market": "Europe", "week": w} for w in weeks[10:]]
        result = audit.market_concurrency(pd.DataFrame(rows))
        assert (result["share_of_weeks_active"] == 0.5).all()

    def test_concurrent_markets_score_full_coverage(self):
        weeks = pd.date_range("2015-01-05", periods=20, freq="W-MON")
        rows = [{"Market": m, "week": w} for w in weeks for m in ("LATAM", "Europe")]
        result = audit.market_concurrency(pd.DataFrame(rows))
        assert (result["share_of_weeks_active"] == 1.0).all()


class TestDemandAutocorrelation:
    def test_independent_draws_show_no_persistence(self):
        rng = np.random.default_rng(7)
        panel = pd.DataFrame(rng.normal(500, 50, (3, 200)))
        result = audit.demand_autocorrelation(panel)
        assert result["max_abs_autocorrelation"] < 0.2

    def test_a_trending_series_shows_persistence(self):
        panel = pd.DataFrame([np.arange(200, dtype=float) + 100])
        result = audit.demand_autocorrelation(panel)
        assert result["autocorrelation"]["lag_1"] > 0.9
