"""Geospatial and demand features.

Proximity features are computed once per *lot*, never per row. The original
implementation looped over all 18k rows, so a lot's 1312 repeated timestamps
counted as its own neighbours -- producing ``competitor_counts == 1312`` and a
nearest-competitor distance of a few metres for every lot.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import COMPETITOR_RADIUS_M, EARTH_RADIUS_M


def haversine_matrix(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Full pairwise great-circle distance matrix in metres.

    With 14 lots this is a 14x14 matrix, so the straightforward vectorised form
    is both clearer and faster than a spatial index.
    """
    phi = np.radians(np.asarray(lat, dtype=float))[:, None]
    lam = np.radians(np.asarray(lon, dtype=float))[:, None]

    dphi = phi - phi.T
    dlam = lam - lam.T

    a = np.sin(dphi / 2) ** 2 + np.cos(phi) * np.cos(phi.T) * np.sin(dlam / 2) ** 2
    # Clip guards against tiny negative values from floating-point error.
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def proximity_features(
    lots: pd.DataFrame, radius_m: float = COMPETITOR_RADIUS_M
) -> pd.DataFrame:
    """Per-lot competitor geography.

    Args:
        lots: one row per lot, as returned by :func:`~parking_pricing.data.lot_locations`.
        radius_m: lots closer than this count as competitors.

    Returns:
        ``lots`` plus ``nearest_competitor_m`` (NaN when a lot is alone) and
        ``competitors_within_radius``. The diagonal is excluded, so a lot is
        never its own competitor.
    """
    if lots["lot_id"].duplicated().any():
        raise ValueError(
            "proximity_features expects one row per lot; got duplicate lot_ids. "
            "Pass lot_locations(df), not the per-timestamp frame."
        )

    dist = haversine_matrix(lots["Latitude"].to_numpy(), lots["Longitude"].to_numpy())
    np.fill_diagonal(dist, np.inf)

    out = lots.copy()
    nearest = dist.min(axis=1)
    out["nearest_competitor_m"] = np.where(np.isinf(nearest), np.nan, nearest)
    out["competitors_within_radius"] = (dist < radius_m).sum(axis=1).astype(int)
    return out


def competitor_groups(
    lots: pd.DataFrame, radius_m: float = COMPETITOR_RADIUS_M
) -> dict[str, list[str]]:
    """Map each lot_id to the lot_ids of its competitors within ``radius_m``."""
    dist = haversine_matrix(lots["Latitude"].to_numpy(), lots["Longitude"].to_numpy())
    np.fill_diagonal(dist, np.inf)
    ids = lots["lot_id"].tolist()
    return {
        lot: [ids[j] for j in np.flatnonzero(dist[i] < radius_m)]
        for i, lot in enumerate(ids)
    }


def add_demand_features(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise the raw demand drivers onto a common 0-1 scale.

    Model 2 sums weighted drivers, so leaving them on their native scales
    (queue 0-10, traffic 0-2, weight 1.0-2.0) would let unit choice decide the
    weighting. Normalising makes the configured weights mean what they say.
    """
    out = df.copy()

    out["queue_norm"] = (out["QueueLength"] / 10.0).clip(0.0, 1.0)
    out["traffic_norm"] = (out["TrafficLevel"] / 2.0).clip(0.0, 1.0)
    out["special_day"] = out["IsSpecialDay"].astype(float).clip(0.0, 1.0)
    # Vehicle weights run 1.0 (small) to 2.0 (EV); map that span onto 0-1.
    out["vehicle_norm"] = ((out["VehicleTypeWeight"] - 1.0) / 1.0).clip(0.0, 1.0)

    if "occupancy_rate" not in out.columns:
        out["occupancy_rate"] = (out["Occupancy"] / out["Capacity"]).clip(0.0, 1.0)

    return out
