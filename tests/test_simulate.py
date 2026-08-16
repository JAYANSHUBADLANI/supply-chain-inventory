"""Tests for the inventory simulation.

The simulation is where a subtle bug produces the most convincing wrong answer, because
almost any ordering loop will yield plausible looking fill rates. The cases below pin the
mechanics that actually matter: that stock on order counts toward the reorder decision,
that receipts land before demand is served, that a policy with more buffer cannot lose more
sales on the same realisation, and that the warmup period is genuinely excluded.
"""

import numpy as np
import pandas as pd
import pytest

from src import simulate


def constant_run(demand_per_day=10.0, days=200, lead=4, **kwargs):
    demand = np.full(days, demand_per_day)
    leadtimes = np.full(days, lead)
    defaults = dict(
        planning_unit="u",
        policy_name="p",
        demand=demand,
        leadtimes=leadtimes,
        reorder_point=60.0,
        order_quantity=100.0,
        safety_stock=20.0,
        daily_holding_cost=0.01,
        ordering_cost=250.0,
    )
    defaults.update(kwargs)
    return simulate.simulate_unit(**defaults)


class TestMechanics:
    def test_ample_stock_ships_everything(self):
        result = constant_run(reorder_point=500.0, order_quantity=2000.0)
        assert result.units_lost == 0.0
        assert result.fill_rate == pytest.approx(1.0)

    def test_no_replenishment_eventually_stocks_out(self):
        demand = np.full(100, 10.0)
        result = simulate.simulate_unit(
            planning_unit="u", policy_name="p", demand=demand,
            leadtimes=np.full(100, 999), reorder_point=0.0, order_quantity=1.0,
            safety_stock=0.0, daily_holding_cost=0.0, ordering_cost=0.0,
        )
        assert result.units_lost > 0
        assert result.stockout_days > 0

    def test_pipeline_counts_toward_the_reorder_decision(self):
        result = constant_run(demand_per_day=10.0, days=100, lead=4,
                              reorder_point=60.0, order_quantity=100.0)
        expected_orders = 100 * 10.0 / 100.0
        assert result.orders_placed == pytest.approx(expected_orders, abs=2)

    def test_ignoring_the_pipeline_would_overorder(self):
        """A correct loop orders about once per Q of demand, not once per day."""
        result = constant_run(demand_per_day=10.0, days=100, order_quantity=100.0)
        assert result.orders_placed < 20

    def test_receipts_land_before_demand_is_served(self):
        demand = np.array([0.0, 0.0, 100.0])
        result = simulate.simulate_unit(
            planning_unit="u", policy_name="p", demand=demand,
            leadtimes=np.array([2, 2, 2]), reorder_point=1000.0, order_quantity=100.0,
            safety_stock=0.0, daily_holding_cost=0.0, ordering_cost=0.0,
        )
        assert result.units_lost == 0.0

    def test_demand_and_leadtime_arrays_must_align(self):
        with pytest.raises(ValueError, match="same length"):
            simulate.simulate_unit(
                planning_unit="u", policy_name="p", demand=np.zeros(10),
                leadtimes=np.zeros(5), reorder_point=1.0, order_quantity=1.0,
                safety_stock=0.0, daily_holding_cost=0.0, ordering_cost=0.0,
            )

    def test_zero_order_quantity_is_rejected(self):
        with pytest.raises(ValueError, match="order quantity"):
            constant_run(order_quantity=0.0)


class TestMonotonicity:
    def test_more_buffer_never_loses_more_sales_on_the_same_realisation(self):
        rng = np.random.default_rng(0)
        demand = np.clip(rng.normal(10, 4, 300), 0, None)
        leadtimes = rng.integers(2, 7, 300)
        losses = []
        for rop in (30.0, 60.0, 90.0, 120.0):
            result = simulate.simulate_unit(
                planning_unit="u", policy_name="p", demand=demand, leadtimes=leadtimes,
                reorder_point=rop, order_quantity=100.0, safety_stock=rop,
                daily_holding_cost=0.01, ordering_cost=250.0,
            )
            losses.append(result.units_lost)
        assert losses == sorted(losses, reverse=True)

    def test_more_buffer_costs_more_to_hold(self):
        rng = np.random.default_rng(1)
        demand = np.clip(rng.normal(10, 4, 300), 0, None)
        leadtimes = rng.integers(2, 7, 300)
        costs = []
        for rop in (30.0, 90.0, 150.0):
            result = simulate.simulate_unit(
                planning_unit="u", policy_name="p", demand=demand, leadtimes=leadtimes,
                reorder_point=rop, order_quantity=100.0, safety_stock=rop,
                daily_holding_cost=0.01, ordering_cost=250.0,
            )
            costs.append(result.holding_cost)
        assert costs == sorted(costs)

    def test_longer_lead_times_reduce_service_at_a_fixed_reorder_point(self):
        rng = np.random.default_rng(2)
        demand = np.clip(rng.normal(10, 4, 300), 0, None)
        short = simulate.simulate_unit(
            planning_unit="u", policy_name="p", demand=demand,
            leadtimes=np.full(300, 2), reorder_point=40.0, order_quantity=100.0,
            safety_stock=0.0, daily_holding_cost=0.0, ordering_cost=0.0,
        )
        long = simulate.simulate_unit(
            planning_unit="u", policy_name="p", demand=demand,
            leadtimes=np.full(300, 8), reorder_point=40.0, order_quantity=100.0,
            safety_stock=0.0, daily_holding_cost=0.0, ordering_cost=0.0,
        )
        assert long.units_lost >= short.units_lost


class TestWarmup:
    def test_warmup_days_are_excluded_from_counted_demand(self):
        full = constant_run(days=100, warmup_days=0)
        warmed = constant_run(days=100, warmup_days=30)
        assert warmed.days == 70
        assert warmed.units_demanded == pytest.approx(full.units_demanded * 0.70)

    def test_warmup_does_not_change_the_demand_stream_itself(self):
        warmed = constant_run(days=100, warmup_days=30, demand_per_day=10.0)
        assert warmed.units_demanded == pytest.approx(700.0)


class TestCommonRandomNumbers:
    def test_same_seed_reproduces_the_draws(self):
        values, probs = np.array([2.0, 3.0, 4.0]), np.array([0.3, 0.4, 0.3])
        first = simulate.common_leadtime_draws(["a", "b"], 50, values, probs, seed=7)
        second = simulate.common_leadtime_draws(["a", "b"], 50, values, probs, seed=7)
        for unit in ("a", "b"):
            assert np.array_equal(first[unit], second[unit])

    def test_different_units_get_different_draws(self):
        values, probs = np.array([2.0, 3.0, 4.0]), np.array([0.3, 0.4, 0.3])
        draws = simulate.common_leadtime_draws(["a", "b"], 200, values, probs, seed=7)
        assert not np.array_equal(draws["a"], draws["b"])

    def test_draws_stay_inside_the_support(self):
        values, probs = np.array([2.0, 3.0, 4.0]), np.array([0.3, 0.4, 0.3])
        draws = simulate.common_leadtime_draws(["a"], 500, values, probs, seed=7)
        assert set(np.unique(draws["a"])).issubset({2.0, 3.0, 4.0})


class TestAggregation:
    def _results(self):
        return [
            simulate.SimulationResult(
                planning_unit="big", policy="p", reorder_point=0, order_quantity=1,
                safety_stock=10, days=100, units_demanded=10000, units_shipped=9900,
                units_lost=100, stockout_days=5, orders_placed=10, cycles=10,
                cycles_with_stockout=1, mean_on_hand=50, holding_cost=100,
                ordering_cost=50, total_cost=150,
            ),
            simulate.SimulationResult(
                planning_unit="small", policy="p", reorder_point=0, order_quantity=1,
                safety_stock=2, days=100, units_demanded=100, units_shipped=50,
                units_lost=50, stockout_days=40, orders_placed=2, cycles=2,
                cycles_with_stockout=1, mean_on_hand=5, holding_cost=10,
                ordering_cost=10, total_cost=20,
            ),
        ]

    def test_fill_rate_is_unit_weighted_not_averaged_across_products(self):
        summary = simulate.aggregate(self._results())
        assert summary.loc["p", "fill_rate"] == pytest.approx(9950 / 10100)

    def test_naive_average_of_ratios_would_differ(self):
        summary = simulate.aggregate(self._results())
        naive = (0.99 + 0.50) / 2
        assert summary.loc["p", "fill_rate"] != pytest.approx(naive)

    def test_cycle_service_pools_cycles(self):
        summary = simulate.aggregate(self._results())
        assert summary.loc["p", "realised_cycle_service_level"] == pytest.approx(1 - 2 / 12)


class TestSegmentBreakdown:
    def test_thin_cycle_counts_suppress_the_service_estimate(self):
        results = [
            simulate.SimulationResult(
                planning_unit=f"u{i}", policy="p", reorder_point=0, order_quantity=1,
                safety_stock=1, days=100, units_demanded=100, units_shipped=100,
                units_lost=0, stockout_days=0, orders_placed=1, cycles=1,
                cycles_with_stockout=0, mean_on_hand=5, holding_cost=1,
                ordering_cost=1, total_cost=2,
            )
            for i in range(3)
        ]
        segments = pd.Series({f"u{i}": "CZ" for i in range(3)})
        breakdown = simulate.segment_breakdown(results, segments)
        row = breakdown.loc[("p", "CZ")]
        assert not row["cycle_service_estimable"]
        assert np.isnan(row["realised_cycle_service_level"])
        assert row["fill_rate"] == pytest.approx(1.0)

    def test_sufficient_cycles_keep_the_service_estimate(self):
        results = [
            simulate.SimulationResult(
                planning_unit=f"u{i}", policy="p", reorder_point=0, order_quantity=1,
                safety_stock=1, days=100, units_demanded=100, units_shipped=100,
                units_lost=0, stockout_days=0, orders_placed=20, cycles=20,
                cycles_with_stockout=1, mean_on_hand=5, holding_cost=1,
                ordering_cost=1, total_cost=2,
            )
            for i in range(3)
        ]
        segments = pd.Series({f"u{i}": "AX" for i in range(3)})
        breakdown = simulate.segment_breakdown(results, segments)
        row = breakdown.loc[("p", "AX")]
        assert row["cycle_service_estimable"]
        assert row["realised_cycle_service_level"] == pytest.approx(1 - 3 / 60)
