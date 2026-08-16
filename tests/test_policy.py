"""Tests for the inventory policy formulas.

The safety stock formula is the piece of this project most likely to be wrong in a way
that still produces a plausible looking number, so it is checked against hand computable
cases and against its own limiting behaviour rather than only for not crashing.
"""

import numpy as np
import pytest
from scipy import stats

from src import policy


class TestServiceLevelQuantile:
    def test_median_service_level_needs_no_buffer(self):
        assert policy.z_for_service_level(0.5) == pytest.approx(0.0)

    def test_known_quantiles(self):
        assert policy.z_for_service_level(0.95) == pytest.approx(1.6449, abs=1e-4)
        assert policy.z_for_service_level(0.99) == pytest.approx(2.3263, abs=1e-4)

    def test_higher_service_costs_more_buffer(self):
        assert policy.z_for_service_level(0.99) > policy.z_for_service_level(0.95)

    @pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
    def test_out_of_range_service_levels_are_rejected(self, bad):
        with pytest.raises(ValueError, match="inside"):
            policy.z_for_service_level(bad)


class TestSafetyStock:
    def test_matches_a_hand_computed_case(self):
        result = policy.safety_stock("u", demand_mean=100, demand_std=20, leadtime_mean=4,
                                     leadtime_std=1, service_level=0.95)
        expected_variance = 4 * 400 + 10000 * 1
        assert result.total_variance == pytest.approx(expected_variance)
        assert result.safety_stock == pytest.approx(
            stats.norm.ppf(0.95) * np.sqrt(expected_variance)
        )

    def test_zero_lead_time_variance_reduces_to_the_simple_formula(self):
        result = policy.safety_stock("u", demand_mean=100, demand_std=20, leadtime_mean=4,
                                     leadtime_std=0, service_level=0.95)
        assert result.leadtime_variance_term == 0.0
        assert result.safety_stock == pytest.approx(
            stats.norm.ppf(0.95) * np.sqrt(4) * 20
        )
        assert result.understatement_vs_demand_only == pytest.approx(1.0)

    def test_zero_demand_variance_leaves_only_the_lead_time_term(self):
        result = policy.safety_stock("u", demand_mean=100, demand_std=0, leadtime_mean=4,
                                     leadtime_std=1, service_level=0.95)
        assert result.demand_variance_term == 0.0
        assert result.safety_stock == pytest.approx(stats.norm.ppf(0.95) * 100)

    def test_variance_shares_sum_to_one(self):
        result = policy.safety_stock("u", demand_mean=70, demand_std=16, leadtime_mean=4,
                                     leadtime_std=1.4, service_level=0.98)
        assert result.leadtime_share_of_variance + result.demand_share_of_variance == pytest.approx(1.0)

    def test_high_volume_low_variability_unit_is_lead_time_driven(self):
        result = policy.safety_stock("fast", demand_mean=70, demand_std=16, leadtime_mean=4,
                                     leadtime_std=1.42, service_level=0.98)
        assert result.dominant_driver == "lead time"
        assert result.leadtime_share_of_variance > 0.8

    def test_low_volume_erratic_unit_is_demand_driven(self):
        result = policy.safety_stock("slow", demand_mean=0.87, demand_std=1.86, leadtime_mean=4,
                                     leadtime_std=1.42, service_level=0.90)
        assert result.dominant_driver == "demand"
        assert result.demand_share_of_variance > 0.8

    def test_ignoring_lead_time_variance_understates_a_fast_mover(self):
        result = policy.safety_stock("fast", demand_mean=70, demand_std=16, leadtime_mean=4,
                                     leadtime_std=1.42, service_level=0.98)
        assert result.understatement_vs_demand_only > 2.0
        assert result.safety_stock > result.demand_only_safety_stock

    def test_safety_stock_rises_with_the_target(self):
        levels = [0.80, 0.90, 0.95, 0.99]
        values = [
            policy.safety_stock("u", 100, 20, 4, 1, level).safety_stock for level in levels
        ]
        assert values == sorted(values)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"demand_mean": -1, "demand_std": 20, "leadtime_mean": 4, "leadtime_std": 1},
            {"demand_mean": 100, "demand_std": -1, "leadtime_mean": 4, "leadtime_std": 1},
            {"demand_mean": 100, "demand_std": 20, "leadtime_mean": -1, "leadtime_std": 1},
            {"demand_mean": 100, "demand_std": 20, "leadtime_mean": 4, "leadtime_std": -1},
        ],
    )
    def test_negative_parameters_are_rejected(self, kwargs):
        with pytest.raises(ValueError, match="non negative"):
            policy.safety_stock("u", service_level=0.95, **kwargs)


class TestReorderPoint:
    def test_is_lead_time_demand_plus_buffer(self):
        assert policy.reorder_point(70, 4, 100) == pytest.approx(380)

    def test_with_no_buffer_is_just_lead_time_demand(self):
        assert policy.reorder_point(70, 4, 0) == pytest.approx(280)


class TestEOQ:
    def test_matches_a_hand_computed_case(self):
        assert policy.economic_order_quantity(1000, 50, 4) == pytest.approx(
            np.sqrt(2 * 1000 * 50 / 4)
        )

    def test_quadruples_demand_doubles_the_quantity(self):
        small = policy.economic_order_quantity(1000, 50, 4)
        large = policy.economic_order_quantity(4000, 50, 4)
        assert large / small == pytest.approx(2.0)

    def test_higher_holding_cost_shrinks_the_order(self):
        assert policy.economic_order_quantity(1000, 50, 8) < policy.economic_order_quantity(
            1000, 50, 4
        )

    def test_zero_holding_cost_is_rejected(self):
        with pytest.raises(ValueError, match="holding cost"):
            policy.economic_order_quantity(1000, 50, 0)


class TestFillRate:
    def test_loss_function_at_zero(self):
        assert policy.standard_normal_loss(0.0) == pytest.approx(0.3989, abs=1e-4)

    def test_loss_function_decreases_in_z(self):
        assert policy.standard_normal_loss(2.0) < policy.standard_normal_loss(1.0)

    def test_fill_rate_exceeds_cycle_service_level(self):
        result = policy.safety_stock("u", 70, 16, 4, 1.42, 0.95)
        fill = policy.expected_fill_rate(result.sigma_demand_over_leadtime, result.z, 1000)
        assert fill > 0.95

    def test_smaller_orders_widen_the_gap_to_cycle_service(self):
        result = policy.safety_stock("u", 70, 16, 4, 1.42, 0.95)
        big = policy.expected_fill_rate(result.sigma_demand_over_leadtime, result.z, 2000)
        small = policy.expected_fill_rate(result.sigma_demand_over_leadtime, result.z, 200)
        assert big > small

    def test_zero_order_quantity_is_rejected(self):
        with pytest.raises(ValueError, match="order quantity"):
            policy.expected_fill_rate(100, 1.6, 0)


class TestBootstrapSafetyStock:
    def _leadtime(self):
        return np.array([2.0, 3.0, 4.0, 5.0, 6.0]), np.full(5, 0.2)

    def test_recovers_the_normal_answer_on_well_behaved_demand(self):
        rng = np.random.default_rng(3)
        daily = rng.normal(70, 16, 4000)
        values, probs = self._leadtime()
        empirical = policy.bootstrap_safety_stock(daily, values, probs, 0.95, 40000, rng)
        closed = policy.safety_stock("u", 70, 16, 4.0, np.std([2, 3, 4, 5, 6]), 0.95)
        assert empirical["safety_stock"] == pytest.approx(closed.safety_stock, rel=0.15)

    def test_zero_inflated_demand_produces_a_skewed_distribution(self):
        rng = np.random.default_rng(4)
        daily = np.where(rng.random(4000) < 0.75, 0.0, rng.poisson(3.5, 4000))
        values, probs = self._leadtime()
        empirical = policy.bootstrap_safety_stock(daily, values, probs, 0.95, 40000, rng)
        assert empirical["skewness"] > 0.5

    def test_higher_service_level_requires_more_buffer(self):
        rng = np.random.default_rng(5)
        daily = rng.normal(70, 16, 2000)
        values, probs = self._leadtime()
        low = policy.bootstrap_safety_stock(daily, values, probs, 0.90, 20000, rng)
        high = policy.bootstrap_safety_stock(daily, values, probs, 0.99, 20000, rng)
        assert high["safety_stock"] > low["safety_stock"]

    def test_deterministic_lead_time_of_zero_gives_no_demand(self):
        rng = np.random.default_rng(6)
        daily = rng.normal(70, 16, 1000)
        empirical = policy.bootstrap_safety_stock(
            daily, np.array([0.0]), np.array([1.0]), 0.95, 1000, rng
        )
        assert empirical["mean_demand_over_leadtime"] == 0.0
        assert empirical["safety_stock"] == 0.0

    def test_empty_demand_series_is_rejected(self):
        values, probs = self._leadtime()
        with pytest.raises(ValueError, match="empty"):
            policy.bootstrap_safety_stock(
                np.array([]), values, probs, 0.95, 100, np.random.default_rng(0)
            )


class TestCosts:
    class _Costs:
        annual_holding_rate = 0.25
        ordering_cost_per_order = 250.0
        unit_cost_as_share_of_price = 0.65

        def annual_holding_cost(self, unit_price):
            return unit_price * self.unit_cost_as_share_of_price * self.annual_holding_rate

    def test_holding_cost_scales_with_units(self):
        costs = self._Costs()
        assert policy.annual_holding_cost(100, 60.0, costs) == pytest.approx(
            100 * 60.0 * 0.65 * 0.25
        )

    def test_total_cost_splits_into_named_parts(self):
        costs = self._Costs()
        block = policy.total_annual_policy_cost(25000, 1000, 200, 60.0, costs)
        assert block["total_annual_cost"] == pytest.approx(
            block["ordering_cost"]
            + block["cycle_stock_holding_cost"]
            + block["safety_stock_holding_cost"]
        )

    def test_cycle_stock_is_half_the_order_quantity(self):
        costs = self._Costs()
        block = policy.total_annual_policy_cost(25000, 1000, 0, 60.0, costs)
        assert block["cycle_stock_holding_cost"] == pytest.approx(
            policy.annual_holding_cost(500, 60.0, costs)
        )

    def test_zero_order_quantity_is_rejected(self):
        with pytest.raises(ValueError, match="order quantity"):
            policy.total_annual_policy_cost(25000, 0, 200, 60.0, self._Costs())
