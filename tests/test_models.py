"""Tests for the three pricing models."""

import numpy as np
import pandas as pd
import pytest

from parking_pricing.config import PricingConfig
from parking_pricing.models import (
    demand_bounds,
    mean_competitor_price,
    model1_baseline_linear,
    model2_demand_based,
    model3_competitive,
)

CFG = PricingConfig()


def occupancy_frame(rates):
    return pd.DataFrame({"occupancy_rate": rates})


def test_model1_increases_with_occupancy():
    """The submitted formula decreased with occupancy; this is the fix."""
    price = model1_baseline_linear(occupancy_frame([0.0, 0.5, 1.0]), CFG)
    assert price.is_monotonic_increasing


def test_model1_empty_lot_is_cheapest():
    price = model1_baseline_linear(occupancy_frame([0.0, 1.0]), CFG)
    assert price.iloc[0] < price.iloc[-1]


def test_model1_stays_in_band():
    """Guards the original blow-up to 327 against a base of 10."""
    price = model1_baseline_linear(occupancy_frame(np.linspace(0, 1, 101)), CFG)
    assert price.min() >= CFG.price_floor
    assert price.max() <= CFG.price_ceiling


def test_demand_bounds_match_weights():
    low, high = demand_bounds(CFG.model2)
    assert low == pytest.approx(-0.7)
    assert high == pytest.approx(1.0 + 0.5 + 1.5 + 0.8)


def demand_frame(n=3, **overrides):
    base = {
        "Occupancy": [50] * n,
        "Capacity": [100] * n,
        "QueueLength": [5] * n,
        "TrafficLevel": [1] * n,
        "IsSpecialDay": [0] * n,
        "VehicleTypeWeight": [1.5] * n,
    }
    base.update(overrides)
    return pd.DataFrame(base)


def test_model2_price_in_band():
    df = demand_frame(3, QueueLength=[0, 5, 10], TrafficLevel=[0, 1, 2])
    out = model2_demand_based(df, CFG)
    assert out["price"].between(CFG.price_floor, CFG.price_ceiling).all()


def test_model2_demand_norm_in_unit_range():
    df = demand_frame(2, Occupancy=[0, 100], IsSpecialDay=[0, 1])
    out = model2_demand_based(df, CFG)
    assert out["demand_norm"].between(0.0, 1.0).all()


def test_model2_higher_occupancy_raises_price():
    df = demand_frame(2, Occupancy=[10, 90])
    out = model2_demand_based(df, CFG)
    assert out["price"].iloc[1] > out["price"].iloc[0]


def test_model2_is_deterministic_per_row():
    """Normalisation uses weight-derived bounds, not the observed min and max, so
    a row price must not depend on which other rows share the batch."""
    row = demand_frame(1, Occupancy=[70])
    alone = model2_demand_based(row, CFG)["price"].iloc[0]
    batch = demand_frame(3, Occupancy=[0, 70, 100])
    in_batch = model2_demand_based(batch, CFG)["price"].iloc[1]
    assert alone == pytest.approx(in_batch)


@pytest.fixture
def two_lot_prices():
    ts = pd.to_datetime(["2016-10-04 08:00", "2016-10-04 08:00"])
    return pd.DataFrame(
        {
            "lot_id": ["LOT_01", "LOT_02"],
            "Timestamp": ts,
            "occupancy_rate": [0.5, 0.5],
            "price": [12.0, 8.0],
        }
    )


def test_mean_competitor_price_excludes_self(two_lot_prices):
    groups = {"LOT_01": ["LOT_02"], "LOT_02": ["LOT_01"]}
    comp = mean_competitor_price(two_lot_prices, groups)
    assert comp.tolist() == [8.0, 12.0]


def test_lot_without_competitors_falls_back_to_own_price(two_lot_prices):
    groups = {"LOT_01": [], "LOT_02": []}
    comp = mean_competitor_price(two_lot_prices, groups)
    assert comp.tolist() == [12.0, 8.0]
    assert not comp.isna().any()


def test_model3_is_noop_without_competitors(two_lot_prices):
    groups = {"LOT_01": [], "LOT_02": []}
    out = model3_competitive(two_lot_prices, groups, CFG)
    assert out["price"].tolist() == pytest.approx(two_lot_prices["price"].tolist())


def test_model3_pulls_toward_competitor(two_lot_prices):
    groups = {"LOT_01": ["LOT_02"], "LOT_02": ["LOT_01"]}
    out = model3_competitive(two_lot_prices, groups, CFG)
    # LOT_01 at 12 with a rival at 8 moves down; LOT_02 moves up.
    assert out["price"].iloc[0] < 12.0
    assert out["price"].iloc[1] > 8.0


def test_model3_discounts_full_lot_undercut_by_rival(two_lot_prices):
    full = two_lot_prices.copy()
    full["occupancy_rate"] = [1.0, 0.5]
    groups = {"LOT_01": ["LOT_02"], "LOT_02": ["LOT_01"]}
    out = model3_competitive(full, groups, CFG)
    assert out["is_full"].tolist() == [True, False]
    # Pull alone gives 12 + 0.3*(8-12) = 10.8; the full-lot discount takes 10%.
    assert out["price"].iloc[0] == pytest.approx(10.8 * 0.9)


def test_model3_price_in_band(two_lot_prices):
    extreme = two_lot_prices.copy()
    extreme["price"] = [100.0, 0.01]
    groups = {"LOT_01": ["LOT_02"], "LOT_02": ["LOT_01"]}
    out = model3_competitive(extreme, groups, CFG)
    assert out["price"].between(CFG.price_floor, CFG.price_ceiling).all()
