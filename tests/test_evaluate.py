"""Tests for the model-comparison metrics."""

import numpy as np
import pandas as pd
import pytest

from parking_pricing.config import PricingConfig
from parking_pricing.evaluate import (
    MODELS,
    band_saturation,
    compare_models,
    competitive_effect,
    format_report,
    occupancy_responsiveness,
    price_volatility,
)
from parking_pricing.pipeline import run_pipeline

CFG = PricingConfig()


@pytest.fixture(scope="module")
def prices():
    return run_pipeline()


def frame(lot_ids, values, occupancy=None):
    n = len(values)
    return pd.DataFrame(
        {
            "lot_id": lot_ids,
            "Timestamp": pd.date_range("2016-10-04", periods=n, freq="30min"),
            "price_model1": values,
            "occupancy_rate": occupancy if occupancy is not None else np.linspace(0, 1, n),
            "Occupancy": [10] * n,
        }
    )


def test_volatility_is_zero_for_a_flat_price():
    df = frame(["LOT_01"] * 4, [10.0, 10.0, 10.0, 10.0])
    assert price_volatility(df, "price_model1") == 0.0


def test_volatility_measures_step_size():
    df = frame(["LOT_01"] * 4, [10.0, 12.0, 10.0, 12.0])
    assert price_volatility(df, "price_model1") == pytest.approx(2.0)


def test_volatility_does_not_cross_lot_boundaries():
    """A jump between two different lots is not a price change."""
    same = frame(["LOT_01"] * 4, [10.0, 11.0, 10.0, 11.0])
    split = frame(["LOT_01", "LOT_01", "LOT_02", "LOT_02"], [10.0, 11.0, 50.0, 51.0])
    assert price_volatility(split, "price_model1") == pytest.approx(
        price_volatility(same, "price_model1")
    )


def test_band_saturation_counts_both_edges():
    df = frame(["LOT_01"] * 4, [CFG.price_floor, 10.0, 12.0, CFG.price_ceiling])
    assert band_saturation(df, "price_model1", CFG) == pytest.approx(0.5)


def test_band_saturation_is_zero_inside_the_band():
    df = frame(["LOT_01"] * 3, [8.0, 10.0, 12.0])
    assert band_saturation(df, "price_model1", CFG) == 0.0


def test_responsiveness_is_one_when_price_tracks_occupancy():
    df = frame(["LOT_01"] * 5, [6.0, 7.0, 8.0, 9.0, 10.0], occupancy=np.linspace(0, 1, 5))
    assert occupancy_responsiveness(df, "price_model1") == pytest.approx(1.0)


def test_responsiveness_is_negative_for_the_submitted_model1_shape():
    """The original priced an empty lot highest; that shape must score negative."""
    df = frame(["LOT_01"] * 5, [10.0, 9.0, 8.0, 7.0, 6.0], occupancy=np.linspace(0, 1, 5))
    assert occupancy_responsiveness(df, "price_model1") == pytest.approx(-1.0)


def test_compare_models_covers_every_model(prices):
    table = compare_models(prices, CFG)
    assert list(table.index) == [m.replace("price_", "") for m in MODELS]
    assert not table.isna().any().any()


def test_all_models_respond_positively_to_occupancy(prices):
    table = compare_models(prices, CFG)
    assert (table["occupancy_corr"] > 0).all()


def test_competitive_effect_is_exactly_zero_for_isolated_lots(prices):
    """The control group: no rival within the radius means no adjustment."""
    effect = competitive_effect(prices)
    assert effect.loc["isolated", "max_abs_shift"] == 0.0
    assert effect.loc["has rivals", "mean_abs_shift"] > 0.0


def test_competitive_effect_partitions_every_lot(prices):
    effect = competitive_effect(prices)
    assert effect["lots"].sum() == prices["lot_id"].nunique()
    assert effect["rows"].sum() == len(prices)


def test_report_renders(prices):
    report = format_report(prices, CFG)
    assert "Model comparison" in report
    assert "Competitive effect" in report
    for model in MODELS:
        assert model.replace("price_", "") in report
