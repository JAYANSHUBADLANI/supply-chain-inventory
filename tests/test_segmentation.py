"""Tests for the ABC and XYZ cuts and the honesty checks on the resulting grid."""

import numpy as np
import pandas as pd
import pytest

from src import segmentation

SERVICE_LEVELS = {"A": 0.98, "B": 0.95, "C": 0.90}


class TestABC:
    def test_boundary_unit_belongs_to_the_tier_it_completes(self):
        revenue = pd.Series({"a": 80.0, "b": 15.0, "c": 5.0})
        result = segmentation.abc_classify(revenue, [0.80, 0.95])
        assert result.loc["a", "abc_class"] == "A"
        assert result.loc["b", "abc_class"] == "B"
        assert result.loc["c", "abc_class"] == "C"

    def test_a_single_dominant_unit_does_not_absorb_the_whole_catalogue(self):
        revenue = pd.Series({"whale": 900.0, "b": 60.0, "c": 30.0, "d": 10.0})
        result = segmentation.abc_classify(revenue, [0.80, 0.95])
        assert result.loc["whale", "abc_class"] == "A"
        assert (result["abc_class"] == "A").sum() == 1

    def test_output_is_ordered_by_revenue(self):
        revenue = pd.Series({"small": 1.0, "big": 100.0, "mid": 10.0})
        result = segmentation.abc_classify(revenue, [0.80, 0.95])
        assert list(result.index) == ["big", "mid", "small"]

    def test_cumulative_share_reaches_one(self):
        revenue = pd.Series({chr(97 + i): float(100 - i) for i in range(20)})
        result = segmentation.abc_classify(revenue, [0.80, 0.95])
        assert result["cumulative_revenue_share"].iloc[-1] == pytest.approx(1.0)

    def test_invalid_thresholds_are_rejected(self):
        revenue = pd.Series({"a": 1.0})
        with pytest.raises(ValueError, match="increasing"):
            segmentation.abc_classify(revenue, [0.95, 0.80])


class TestXYZ:
    def test_cuts_at_the_configured_boundaries(self):
        cv = pd.Series({"steady": 0.10, "variable": 0.30, "erratic": 0.90})
        result = segmentation.xyz_classify(cv, [0.25, 0.50])
        assert result["steady"] == "X"
        assert result["variable"] == "Y"
        assert result["erratic"] == "Z"

    def test_boundary_value_falls_into_the_higher_bucket(self):
        cv = pd.Series({"on_edge": 0.25})
        assert segmentation.xyz_classify(cv, [0.25, 0.50])["on_edge"] == "Y"

    def test_invalid_thresholds_are_rejected(self):
        cv = pd.Series({"a": 0.1})
        with pytest.raises(ValueError, match="increasing"):
            segmentation.xyz_classify(cv, [0.5, 0.25])


class TestGrid:
    def _inputs(self):
        revenue = pd.Series({"a": 800.0, "b": 150.0, "c": 40.0, "d": 10.0})
        cv = pd.Series({"a": 0.10, "b": 0.30, "c": 0.80, "d": 0.90})
        return revenue, cv

    def test_service_level_follows_the_abc_tier(self):
        revenue, cv = self._inputs()
        grid = segmentation.build_grid(revenue, cv, [0.80, 0.95], [0.25, 0.50], SERVICE_LEVELS)
        assert grid.loc["a", "target_service_level"] == 0.98
        assert grid.loc[grid["abc_class"] == "C", "target_service_level"].eq(0.90).all()

    def test_segment_label_joins_both_axes(self):
        revenue, cv = self._inputs()
        grid = segmentation.build_grid(revenue, cv, [0.80, 0.95], [0.25, 0.50], SERVICE_LEVELS)
        assert grid.loc["a", "segment"] == "AX"

    def test_missing_service_level_raises(self):
        revenue, cv = self._inputs()
        with pytest.raises(ValueError, match="no service level configured"):
            segmentation.build_grid(revenue, cv, [0.80, 0.95], [0.25, 0.50], {"A": 0.98})

    def test_units_without_a_variability_estimate_are_marked_not_dropped(self):
        revenue = pd.Series({"a": 800.0, "b": 200.0})
        cv = pd.Series({"a": 0.10})
        grid = segmentation.build_grid(revenue, cv, [0.80, 0.95], [0.25, 0.50], SERVICE_LEVELS)
        assert len(grid) == 2
        assert grid.loc["b", "segment"].endswith("?")

    def test_summary_covers_every_unit(self):
        revenue, cv = self._inputs()
        grid = segmentation.build_grid(revenue, cv, [0.80, 0.95], [0.25, 0.50], SERVICE_LEVELS)
        summary = segmentation.grid_summary(grid)
        assert summary["units"].sum() == len(grid)
        assert summary["revenue_share"].sum() == pytest.approx(1.0, abs=1e-4)


class TestConcentration:
    def test_textbook_pareto_puts_a_fifth_of_units_at_four_fifths_of_revenue(self):
        revenue = pd.Series({f"u{i}": v for i, v in enumerate([40.0] * 2 + [2.5] * 8)})
        profile = segmentation.concentration_profile(revenue)
        assert profile["share_of_units_for_80pct_revenue"] == pytest.approx(0.2)

    def test_extreme_concentration_is_reported_as_such(self):
        revenue = pd.Series({f"u{i}": v for i, v in enumerate([1000.0] + [1.0] * 99)})
        profile = segmentation.concentration_profile(revenue)
        assert profile["units_for_80pct_revenue"] == 1
        assert profile["share_of_units_for_80pct_revenue"] < 0.05


class TestDegeneracy:
    def test_flags_an_a_tier_that_sits_in_one_variability_bucket(self):
        revenue = pd.Series({"a": 500.0, "b": 400.0, "c": 100.0})
        cv = pd.Series({"a": 0.10, "b": 0.11, "c": 0.80})
        grid = segmentation.build_grid(revenue, cv, [0.80, 0.95], [0.25, 0.50], SERVICE_LEVELS)
        flags = segmentation.degeneracy_flags(grid)
        assert flags["a_tier_collapses_to_one_xyz_class"]
        assert flags["a_tier_xyz_classes"] == ["X"]

    def test_does_not_flag_a_spread_a_tier(self):
        revenue = pd.Series({"a": 500.0, "b": 400.0, "c": 100.0})
        cv = pd.Series({"a": 0.10, "b": 0.80, "c": 0.30})
        grid = segmentation.build_grid(revenue, cv, [0.80, 0.95], [0.25, 0.50], SERVICE_LEVELS)
        flags = segmentation.degeneracy_flags(grid)
        assert not flags["a_tier_collapses_to_one_xyz_class"]

    def test_counts_populated_cells_out_of_nine(self):
        revenue = pd.Series({"a": 500.0, "b": 400.0, "c": 100.0})
        cv = pd.Series({"a": 0.10, "b": 0.80, "c": 0.30})
        flags = segmentation.degeneracy_flags(
            segmentation.build_grid(revenue, cv, [0.80, 0.95], [0.25, 0.50], SERVICE_LEVELS)
        )
        assert flags["possible_cells"] == 9
        assert 1 <= flags["populated_cells"] <= 9
