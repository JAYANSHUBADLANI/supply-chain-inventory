"""Tests for the demand statistics, chiefly which weeks count as observations."""

import numpy as np
import pandas as pd
import pytest

from src import demand


def panel_from(rows: dict, weeks: int = 10) -> pd.DataFrame:
    index = pd.date_range("2015-01-05", periods=weeks, freq="W-MON")
    return pd.DataFrame(rows, index=index).T


class TestActiveSpan:
    def test_span_runs_from_first_to_last_active_week(self):
        row = pd.Series(
            [np.nan, np.nan, 5.0, np.nan, 7.0, np.nan],
            index=pd.date_range("2015-01-05", periods=6, freq="W-MON"),
        )
        span = demand.active_span(row)
        assert span.first_week == pd.Timestamp("2015-01-19")
        assert span.last_week == pd.Timestamp("2015-02-02")
        assert span.weeks == 3
        assert span.active_weeks == 2
        assert span.zero_weeks == 1

    def test_a_never_active_unit_returns_no_span(self):
        row = pd.Series([np.nan] * 4, index=pd.date_range("2015-01-05", periods=4, freq="W-MON"))
        assert demand.active_span(row) is None


class TestDemandSeries:
    def test_leading_and_trailing_absence_is_excluded_not_zeroed(self):
        row = pd.Series(
            [np.nan, 10.0, np.nan, 20.0, np.nan],
            index=pd.date_range("2015-01-05", periods=5, freq="W-MON"),
        )
        series = demand.demand_series(row)
        assert len(series) == 3
        assert series.tolist() == [10.0, 0.0, 20.0]

    def test_a_product_launched_late_is_not_penalised_for_prior_weeks(self):
        early = pd.Series([5.0] * 10, index=pd.date_range("2015-01-05", periods=10, freq="W-MON"))
        late = early.copy()
        late.iloc[:5] = np.nan
        assert demand.demand_series(late).mean() == demand.demand_series(early).mean()


class TestDemandStatistics:
    def test_uses_the_sample_standard_deviation(self):
        panel = panel_from({"a": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]})
        stats = demand.demand_statistics(panel, min_observations=5)
        expected = pd.Series([10, 20, 30, 40, 50, 60, 70, 80, 90, 100]).std(ddof=1)
        assert stats.loc["a", "weekly_std"] == pytest.approx(expected)

    def test_variance_is_the_square_of_the_reported_deviation(self):
        panel = panel_from({"a": [4, 9, 2, 7, 5, 8, 3, 6, 5, 4]})
        stats = demand.demand_statistics(panel, min_observations=5)
        row = stats.loc["a"]
        assert row["weekly_variance"] == pytest.approx(row["weekly_std"] ** 2)

    def test_thin_samples_are_flagged_not_dropped(self):
        panel = panel_from({"thin": [5, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 5]})
        stats = demand.demand_statistics(panel, min_observations=50)
        assert "thin" in stats.index
        assert not stats.loc["thin", "sufficient_observations"]
        assert not stats.loc["thin", "variance_estimate_trustworthy"]

    def test_intermittent_units_are_flagged(self):
        values = [5.0] * 10
        panel = panel_from({"holed": values})
        panel.iloc[0, 2] = np.nan
        panel.iloc[0, 5] = np.nan
        stats = demand.demand_statistics(panel, min_observations=5)
        assert stats.loc["holed", "zero_week_share"] == pytest.approx(0.2)
        assert stats.loc["holed", "intermittent"]
        assert not stats.loc["holed", "variance_estimate_trustworthy"]

    def test_steady_unit_with_enough_history_is_trustworthy(self):
        panel = panel_from({"steady": [10, 11, 9, 10, 12, 8, 10, 11, 9, 10]})
        stats = demand.demand_statistics(panel, min_observations=5)
        assert stats.loc["steady", "variance_estimate_trustworthy"]

    def test_week_slice_restricts_the_estimation_window(self):
        panel = panel_from({"a": [10] * 5 + [100] * 5})
        full = demand.demand_statistics(panel, min_observations=1)
        early = demand.demand_statistics(panel, min_observations=1, week_slice=slice(0, 5))
        assert early.loc["a", "weekly_mean"] == pytest.approx(10.0)
        assert full.loc["a", "weekly_mean"] == pytest.approx(55.0)


class TestExclusionLedger:
    def test_ledger_accounts_for_every_planning_unit(self):
        panel = panel_from(
            {
                "steady": [10, 11, 9, 10, 12, 8, 10, 11, 9, 10],
                "thin": [5, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 5],
                "holed": [5, 5, np.nan, 5, 5, np.nan, 5, 5, 5, 5],
            }
        )
        stats = demand.demand_statistics(panel, min_observations=8)
        ledger = demand.exclusion_ledger(stats)
        assert ledger["planning_units_seen"] == 3
        assert (
            ledger["excluded_thin_sample"]
            + ledger["excluded_intermittent"]
            + ledger["carried_forward"]
            == 3
        )

    def test_ledger_raises_when_categories_overlap(self):
        stats = pd.DataFrame(
            {
                "sufficient_observations": [True],
                "intermittent": [False],
                "variance_estimate_trustworthy": [False],
                "total_units": [100.0],
            },
            index=["a"],
        )
        with pytest.raises(ValueError, match="does not reconcile"):
            demand.exclusion_ledger(stats)
