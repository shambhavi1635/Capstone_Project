"""End-to-end tests over the committed dataset."""

import pandas as pd
import pytest

from parking_pricing.config import PricingConfig
from parking_pricing.data import load_dataset, lot_locations
from parking_pricing.pipeline import OUTPUT_COLUMNS, daily_summary, run_pipeline

CFG = PricingConfig()

EXPECTED_LOTS = 14


@pytest.fixture(scope="module")
def prices():
    return run_pipeline()


def test_dataset_loads_with_expected_shape():
    df = load_dataset()
    assert df["lot_id"].nunique() == EXPECTED_LOTS
    assert df.groupby("lot_id")["Timestamp"].is_monotonic_increasing.all()


def test_dataset_has_one_row_per_lot_timestamp():
    """The source file has 152 rows sharing a (lot, timestamp) with another row;
    collapsing each such group to one reading removes 86 rows."""
    df = load_dataset()
    assert not df.duplicated(subset=["lot_id", "Timestamp"]).any()
    assert df.attrs["duplicates_dropped"] == 86
    assert len(df) == 18_282


def test_missing_dataset_raises_clear_error():
    with pytest.raises(FileNotFoundError, match="Dataset not found"):
        load_dataset("does/not/exist.csv")


def test_lot_locations_is_one_row_per_lot():
    lots = lot_locations(load_dataset())
    assert len(lots) == EXPECTED_LOTS
    assert not lots["lot_id"].duplicated().any()


def test_pipeline_prices_one_row_per_lot_timestamp(prices):
    """The original collapsed all lots into a single price per day."""
    assert prices["lot_id"].nunique() == EXPECTED_LOTS
    assert not prices.duplicated(subset=["lot_id", "Timestamp"]).any()
    assert len(prices) == len(load_dataset())


def test_pipeline_output_columns(prices):
    assert list(prices.columns) == OUTPUT_COLUMNS


def test_all_model_prices_stay_in_band(prices):
    for col in ["price_model1", "price_model2", "price_model3"]:
        assert prices[col].between(CFG.price_floor, CFG.price_ceiling).all(), col


def test_no_missing_prices(prices):
    for col in ["price_model1", "price_model2", "price_model3"]:
        assert not prices[col].isna().any(), col


def test_competitor_counts_are_plausible(prices):
    """The original reported 1312 competitors -- the number of timesteps per lot.

    With 14 lots, no lot can have more than 13.
    """
    assert prices["competitors_within_radius"].max() <= EXPECTED_LOTS - 1


def test_nearest_competitor_distance_is_never_zero(prices):
    distances = prices["nearest_competitor_m"].dropna()
    assert (distances > 0).all()


def test_isolated_lot_model3_equals_model2(prices):
    """A lot with no rival within the radius must be untouched by Model 3."""
    alone = prices[prices["competitors_within_radius"] == 0]
    assert not alone.empty
    pd.testing.assert_series_equal(
        alone["price_model3"], alone["price_model2"], check_names=False
    )


def test_pipeline_is_reproducible():
    """No unseeded randomness: two runs must agree exactly."""
    first = run_pipeline()
    second = run_pipeline()
    pd.testing.assert_frame_equal(first, second)


def test_daily_summary_shape(prices):
    daily = daily_summary(prices)
    assert set(daily.columns) == {
        "lot_id",
        "day",
        "occupancy_rate",
        "price_model1",
        "price_model2",
        "price_model3",
    }
    assert len(daily) == prices.groupby(["lot_id", "day"]).ngroups


def test_config_band_is_respected_when_widened():
    """A custom config must actually change the clamp, not be ignored."""
    wide = PricingConfig(base_price=10.0, min_multiplier=0.1, max_multiplier=10.0)
    out = run_pipeline(cfg=wide)
    assert out["price_model1"].max() <= wide.price_ceiling
    assert out["price_model1"].max() > 0
