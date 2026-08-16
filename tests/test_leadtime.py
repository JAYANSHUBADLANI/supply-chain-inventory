"""Tests for lead-time parameter construction and its provenance flags.

The proxy flag is not decoration. Every downstream consumer of a lead-time parameter can
read whether it came from an observation or from a substitution, and the write-up depends
on that being carried rather than reconstructed by hand.
"""

import numpy as np
import pandas as pd
import pytest

from src import leadtime
from tests.conftest import make_orders


class TestParameters:
    def test_days_convert_to_weeks(self):
        params = leadtime.LeadTimeParameters(
            label="t", source="s", is_proxy=True, rows=10, mean_days=7.0, std_days=3.5
        )
        assert params.mean_weeks == pytest.approx(1.0)
        assert params.std_weeks == pytest.approx(0.5)

    def test_coefficient_of_variation_is_scale_free(self):
        params = leadtime.LeadTimeParameters(
            label="t", source="s", is_proxy=True, rows=10, mean_days=4.0, std_days=2.0
        )
        assert params.coefficient_of_variation == pytest.approx(0.5)
        assert params.std_weeks / params.mean_weeks == pytest.approx(0.5)

    def test_scaling_changes_variability_and_leaves_the_mean(self):
        base = leadtime.LeadTimeParameters(
            label="base", source="s", is_proxy=True, rows=10, mean_days=4.0, std_days=2.0
        )
        scaled = base.scaled(0.5)
        assert scaled.mean_days == base.mean_days
        assert scaled.std_days == pytest.approx(1.0)

    def test_scaling_to_zero_gives_a_deterministic_lead_time(self):
        base = leadtime.LeadTimeParameters(
            label="base", source="s", is_proxy=True, rows=10, mean_days=4.0, std_days=2.0
        )
        assert base.scaled(0.0).std_days == 0.0

    def test_proxy_flag_survives_scaling(self):
        base = leadtime.LeadTimeParameters(
            label="base", source="s", is_proxy=True, rows=10, mean_days=4.0, std_days=2.0
        )
        assert base.scaled(1.5).is_proxy

    def test_serialised_form_carries_the_source(self):
        base = leadtime.pooled_parameters(
            make_orders(
                n_per_mode={"Standard Class": 200},
                real_leadtime={"Standard Class": lambda rng, n: rng.integers(2, 7, n)},
            )
        )
        payload = base.to_dict()
        assert payload["is_proxy"]
        assert "no supplier or purchase order data" in payload["source"]
        assert "mean_weeks" in payload


class TestPooledAndByMode:
    def test_pooled_uses_every_row(self):
        frame = make_orders(
            n_per_mode={"Standard Class": 300, "First Class": 200},
            real_leadtime={
                "Standard Class": lambda rng, n: rng.integers(2, 7, n),
                "First Class": lambda rng, n: np.full(n, 2),
            },
        )
        assert leadtime.pooled_parameters(frame).rows == 500

    def test_by_mode_splits_and_recovers_a_constant_mode(self):
        frame = make_orders(
            n_per_mode={"Standard Class": 300, "First Class": 200},
            real_leadtime={
                "Standard Class": lambda rng, n: rng.integers(2, 7, n),
                "First Class": lambda rng, n: np.full(n, 2),
            },
        )
        by_mode = {p.label: p for p in leadtime.parameters_by_mode(frame)}
        assert by_mode["First Class"].std_days == 0.0
        assert by_mode["Standard Class"].std_days > 0.0


class TestEmpiricalDistribution:
    def test_probabilities_sum_to_one(self):
        frame = make_orders(
            n_per_mode={"Standard Class": 1000},
            real_leadtime={"Standard Class": lambda rng, n: rng.integers(2, 7, n)},
        )
        assert leadtime.empirical_distribution(frame).sum() == pytest.approx(1.0)

    def test_sampling_reproduces_the_support(self):
        frame = make_orders(
            n_per_mode={"Standard Class": 2000},
            real_leadtime={"Standard Class": lambda rng, n: rng.integers(2, 7, n)},
        )
        dist = leadtime.empirical_distribution(frame)
        draws = leadtime.sample_leadtimes(dist, 5000, np.random.default_rng(0))
        assert set(np.unique(draws)).issubset(set(dist.index))

    def test_sampling_recovers_the_mean(self):
        frame = make_orders(
            n_per_mode={"Standard Class": 4000},
            real_leadtime={"Standard Class": lambda rng, n: rng.integers(2, 7, n)},
        )
        dist = leadtime.empirical_distribution(frame)
        draws = leadtime.sample_leadtimes(dist, 40000, np.random.default_rng(1))
        assert draws.mean() == pytest.approx(frame["Days for shipping (real)"].mean(), abs=0.05)


class TestScheduledVersusReal:
    def test_bias_is_realised_minus_scheduled(self):
        frame = make_orders(
            n_per_mode={"Standard Class": 1000},
            real_leadtime={"Standard Class": lambda rng, n: np.full(n, 6)},
            scheduled={"Standard Class": 4},
        )
        result = leadtime.scheduled_versus_real(frame)
        assert result.loc["Standard Class", "bias_days"] == pytest.approx(2.0)
        assert result.loc["Standard Class", "late_share"] == pytest.approx(1.0)

    def test_a_mode_delivering_early_shows_negative_bias(self):
        frame = make_orders(
            n_per_mode={"Standard Class": 1000},
            real_leadtime={"Standard Class": lambda rng, n: np.full(n, 2)},
            scheduled={"Standard Class": 4},
        )
        result = leadtime.scheduled_versus_real(frame)
        assert result.loc["Standard Class", "bias_days"] == pytest.approx(-2.0)
        assert result.loc["Standard Class", "late_share"] == 0.0


class TestSensitivityGrid:
    def test_grid_spans_the_requested_multipliers(self):
        base = leadtime.LeadTimeParameters(
            label="base", source="s", is_proxy=True, rows=10, mean_days=4.0, std_days=2.0
        )
        grid = leadtime.sensitivity_grid(base, [0.0, 0.5, 1.0, 2.0])
        assert [p.std_days for p in grid] == [0.0, 1.0, 2.0, 4.0]
        assert all(p.mean_days == 4.0 for p in grid)
