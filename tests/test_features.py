"""Tests for the geospatial and demand features."""

import numpy as np
import pandas as pd
import pytest

from parking_pricing.features import (
    add_demand_features,
    competitor_groups,
    haversine_matrix,
    proximity_features,
)


def test_haversine_diagonal_is_zero():
    lat = np.array([26.1, 26.2, 26.3])
    lon = np.array([91.7, 91.8, 91.9])
    d = haversine_matrix(lat, lon)
    assert np.allclose(np.diag(d), 0.0)


def test_haversine_is_symmetric():
    lat = np.array([26.1, 26.2, 20.0])
    lon = np.array([91.7, 91.8, 78.0])
    d = haversine_matrix(lat, lon)
    assert np.allclose(d, d.T)


def test_haversine_known_distance():
    """One degree of latitude at the equator is ~111.2 km."""
    d = haversine_matrix(np.array([0.0, 1.0]), np.array([0.0, 0.0]))
    assert d[0, 1] == pytest.approx(111_195, rel=1e-3)


@pytest.fixture
def lots():
    # Two lots ~122 m apart, plus one far away.
    return pd.DataFrame(
        {
            "lot_id": ["LOT_01", "LOT_02", "LOT_03"],
            "Latitude": [26.1400, 26.1411, 20.0000],
            "Longitude": [91.7300, 91.7300, 78.0000],
            "Capacity": [100, 200, 300],
        }
    )


def test_lot_is_never_its_own_competitor(lots):
    """The bug this guards: the original counted a lot's repeated rows."""
    out = proximity_features(lots, radius_m=500)
    assert out["competitors_within_radius"].tolist() == [1, 1, 0]
    assert (out["nearest_competitor_m"] > 0).all()


def test_isolated_lot_has_no_competitors(lots):
    groups = competitor_groups(lots, radius_m=500)
    assert groups["LOT_03"] == []
    assert groups["LOT_01"] == ["LOT_02"]


def test_proximity_rejects_per_timestamp_frame(lots):
    """Passing the un-deduplicated frame is the original bug; it must raise."""
    repeated = pd.concat([lots, lots], ignore_index=True)
    with pytest.raises(ValueError, match="one row per lot"):
        proximity_features(repeated)


def test_demand_features_are_normalised():
    df = pd.DataFrame(
        {
            "Occupancy": [0, 50, 100],
            "Capacity": [100, 100, 100],
            "QueueLength": [0, 5, 10],
            "TrafficLevel": [0, 1, 2],
            "IsSpecialDay": [0, 0, 1],
            "VehicleTypeWeight": [1.0, 1.5, 2.0],
        }
    )
    out = add_demand_features(df)
    for col in ["occupancy_rate", "queue_norm", "traffic_norm", "special_day", "vehicle_norm"]:
        assert out[col].between(0.0, 1.0).all(), col
    assert out["queue_norm"].tolist() == [0.0, 0.5, 1.0]
    assert out["vehicle_norm"].tolist() == [0.0, 0.5, 1.0]


def test_demand_features_clip_out_of_range_input():
    """Occupancy above capacity appears in the real data; the rate must stay <= 1."""
    df = pd.DataFrame(
        {
            "Occupancy": [150],
            "Capacity": [100],
            "QueueLength": [99],
            "TrafficLevel": [5],
            "IsSpecialDay": [1],
            "VehicleTypeWeight": [3.0],
        }
    )
    out = add_demand_features(df)
    assert out["occupancy_rate"].iloc[0] == 1.0
    assert out["queue_norm"].iloc[0] == 1.0
    assert out["vehicle_norm"].iloc[0] == 1.0
