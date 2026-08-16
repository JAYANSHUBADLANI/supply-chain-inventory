"""Tests for exclusion accounting and the shape of the weekly demand panel."""

import numpy as np
import pandas as pd
import pytest

from src import data_io
from src.data_io import ExclusionLedger


class TestExclusionLedger:
    def test_reconciling_ledger_passes(self):
        ledger = ExclusionLedger(
            raw_rows=100, excluded_cancelled_or_fraud=10, excluded_after_clean_window=20,
            kept_rows=70, clean_window_start="2015-01-01", clean_window_end="2017-09-30",
            products_kept=5, weeks_kept=144,
        )
        ledger.check()

    def test_non_reconciling_ledger_raises(self):
        ledger = ExclusionLedger(
            raw_rows=100, excluded_cancelled_or_fraud=10, excluded_after_clean_window=20,
            kept_rows=69, clean_window_start="2015-01-01", clean_window_end="2017-09-30",
            products_kept=5, weeks_kept=144,
        )
        with pytest.raises(ValueError, match="does not reconcile"):
            ledger.check()

    def test_exclusions_account_for_every_row(self, config):
        frame = pd.DataFrame(
            {
                "Order Status": ["COMPLETE", "CANCELED", "SUSPECTED_FRAUD", "COMPLETE"],
                "order_date": pd.to_datetime(
                    ["2016-05-01", "2016-05-02", "2016-05-03", "2017-12-01"]
                ),
                "Product Name": ["a", "b", "c", "d"],
                "Order Item Quantity": [1, 1, 1, 1],
            }
        )
        kept, ledger = data_io.apply_exclusions(frame, config)
        assert ledger.raw_rows == 4
        assert ledger.excluded_cancelled_or_fraud == 2
        assert ledger.excluded_after_clean_window == 1
        assert ledger.kept_rows == 1
        assert len(kept) == 1

    def test_cancelled_rows_are_removed_before_the_window_cut(self, config):
        frame = pd.DataFrame(
            {
                "Order Status": ["CANCELED"],
                "order_date": pd.to_datetime(["2017-12-01"]),
                "Product Name": ["a"],
                "Order Item Quantity": [1],
            }
        )
        _, ledger = data_io.apply_exclusions(frame, config)
        assert ledger.excluded_cancelled_or_fraud == 1
        assert ledger.excluded_after_clean_window == 0


class TestPanel:
    def _frame(self):
        weeks = pd.date_range("2015-01-05", periods=6, freq="W-MON")
        rows = []
        for i, week in enumerate(weeks):
            rows.append({"Product Name": "steady", "week": week, "Order Item Quantity": 10})
            if i in (0, 5):
                rows.append({"Product Name": "bursty", "week": week, "Order Item Quantity": 4})
        return pd.DataFrame(rows)

    def test_absent_weeks_are_null_not_zero(self, config):
        panel = data_io.weekly_demand_panel(self._frame(), config)
        assert panel.loc["bursty"].isna().sum() == 4
        assert panel.loc["steady"].isna().sum() == 0

    def test_coverage_counts_active_weeks(self, config):
        panel = data_io.weekly_demand_panel(self._frame(), config)
        cov = data_io.coverage(panel)
        assert cov["steady"] == 6
        assert cov["bursty"] == 2

    def test_interior_gaps_ignore_leading_and_trailing_absence(self, config):
        weeks = pd.date_range("2015-01-05", periods=6, freq="W-MON")
        rows = [
            {"Product Name": "late_start", "week": w, "Order Item Quantity": 5}
            for w in weeks[3:]
        ]
        rows += [
            {"Product Name": "holed", "week": w, "Order Item Quantity": 5}
            for w in [weeks[0], weeks[2], weeks[5]]
        ]
        panel = data_io.weekly_demand_panel(pd.DataFrame(rows), config)
        gaps = data_io.interior_gaps(panel)
        assert gaps["late_start"] == 0
        assert gaps["holed"] == 3

    def test_a_continuous_series_has_no_interior_gaps(self, config):
        panel = data_io.weekly_demand_panel(self._frame(), config)
        gaps = data_io.interior_gaps(panel)
        assert gaps["steady"] == 0
