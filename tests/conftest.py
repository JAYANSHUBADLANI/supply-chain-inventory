"""Synthetic fixtures so the audit detectors can be tested without the raw extract."""

import numpy as np
import pandas as pd
import pytest

from src.config import Config


@pytest.fixture
def config():
    return Config.load()


def make_orders(
    n_per_mode=None,
    real_leadtime=None,
    scheduled=None,
    start="2015-01-05",
    weeks=20,
    products=("A", "B"),
    markets=("LATAM", "Europe"),
    regions=("Central America", "Western Europe"),
    quantity=2,
    seed=0,
):
    """Build an order-line frame with controllable lead-time behaviour per mode."""
    rng = np.random.default_rng(seed)
    n_per_mode = n_per_mode or {"Standard Class": 400}
    scheduled = scheduled or {mode: 4 for mode in n_per_mode}
    rows = []
    week_index = pd.date_range(start, periods=weeks, freq="W-MON")
    for mode, count in n_per_mode.items():
        draws = real_leadtime[mode](rng, count)
        for i in range(count):
            week = week_index[i % weeks]
            rows.append(
                {
                    "Shipping Mode": mode,
                    "Days for shipping (real)": int(draws[i]),
                    "Days for shipment (scheduled)": scheduled[mode],
                    "Late_delivery_risk": int(draws[i] > scheduled[mode]),
                    "Order Status": "COMPLETE",
                    "order_date": week + pd.Timedelta(days=int(rng.integers(0, 7))),
                    "week": week,
                    "Product Name": products[i % len(products)],
                    "Market": markets[i % len(markets)],
                    "Order Region": regions[i % len(regions)],
                    "Order Item Quantity": quantity,
                    "Sales": 100.0,
                    "Product Price": 50.0,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def uniform_leadtime_orders():
    """Two modes drawing from the same uniform law but promising different windows."""
    return make_orders(
        n_per_mode={"Standard Class": 4000, "Second Class": 4000},
        real_leadtime={
            "Standard Class": lambda rng, n: rng.integers(2, 7, n),
            "Second Class": lambda rng, n: rng.integers(2, 7, n),
        },
        scheduled={"Standard Class": 4, "Second Class": 2},
        seed=1,
    )


@pytest.fixture
def structured_leadtime_orders():
    """A mode whose lead time is genuinely centred and not uniform over its support."""
    return make_orders(
        n_per_mode={"Standard Class": 4000},
        real_leadtime={
            "Standard Class": lambda rng, n: np.clip(
                np.round(rng.normal(4, 0.8, n)), 2, 6
            ).astype(int)
        },
        scheduled={"Standard Class": 4},
        seed=2,
    )


@pytest.fixture
def degenerate_leadtime_orders():
    """A mode that always delivers on exactly the same day."""
    return make_orders(
        n_per_mode={"First Class": 1000},
        real_leadtime={"First Class": lambda rng, n: np.full(n, 2)},
        scheduled={"First Class": 1},
        seed=3,
    )
