"""Loading and cleaning the parking dataset.

The dataset holds one row per (parking lot, timestamp). Lots are identified by
their coordinates, which the raw file repeats on every row; `load_dataset`
collapses that into a stable ``lot_id`` so downstream code can group by lot.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

#: Columns the pipeline needs from the raw CSV.
REQUIRED_COLUMNS = [
    "Latitude",
    "Longitude",
    "Timestamp",
    "Occupancy",
    "Capacity",
    "QueueLength",
    "TrafficLevel",
    "IsSpecialDay",
    "VehicleTypeWeight",
]

DEFAULT_DATASET = Path(__file__).resolve().parents[2] / "data" / "parking_dataset.csv"


def assign_lot_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Give each distinct coordinate pair a stable ``lot_id``.

    Lots are numbered by latitude then longitude so the labels do not depend on
    row order in the source file.
    """
    lots = (
        df[["Latitude", "Longitude"]]
        .drop_duplicates()
        .sort_values(["Latitude", "Longitude"])
        .reset_index(drop=True)
    )
    lots["lot_id"] = [f"LOT_{i + 1:02d}" for i in range(len(lots))]
    return df.merge(lots, on=["Latitude", "Longitude"], how="left")


def load_dataset(path: str | Path | None = None) -> pd.DataFrame:
    """Read the dataset, validate it, and return it sorted by lot then time.

    Raises:
        FileNotFoundError: if the CSV is missing.
        ValueError: if required columns are absent, or occupancy/capacity are
            unusable (non-positive capacity makes occupancy rate meaningless).
    """
    path = Path(path) if path is not None else DEFAULT_DATASET
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Expected the committed copy at "
            f"data/parking_dataset.csv, or pass --data explicitly."
        )

    df = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset {path} is missing columns: {missing}")

    df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    numeric = ["Latitude", "Longitude", "Occupancy", "Capacity", "QueueLength",
               "TrafficLevel", "IsSpecialDay", "VehicleTypeWeight"]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    before = len(df)
    df = df.dropna(subset=numeric + ["Timestamp"])
    dropped = before - len(df)

    if (df["Capacity"] <= 0).any():
        raise ValueError("Dataset contains rows with Capacity <= 0")

    df = assign_lot_ids(df)

    # The source file holds 152 duplicate readings: same lot, same timestamp,
    # identical Occupancy and Capacity, but differing synthetic QueueLength
    # (the original script drew randoms per row, so repeated readings diverged).
    # The real measurements agree, so collapsing them loses no observation and
    # leaves a well-formed panel of one row per lot per timestamp.
    # `kind="stable"` matters: with an unstable sort, which member of a duplicate
    # group counts as "first" can vary by platform and NumPy version, so the kept
    # QueueLength -- and every price derived from it -- would differ between
    # machines.
    before_dedup = len(df)
    df = df.sort_values(["lot_id", "Timestamp"], kind="stable").drop_duplicates(
        subset=["lot_id", "Timestamp"], keep="first"
    )
    duplicates_dropped = before_dedup - len(df)

    df["day"] = df["Timestamp"].dt.normalize()

    # Occupancy above capacity shows up in this dataset; clip so the rate stays
    # a true 0-1 fraction rather than silently exceeding 1.
    df["occupancy_rate"] = (df["Occupancy"] / df["Capacity"]).clip(0.0, 1.0)

    df = df.reset_index(drop=True)
    df.attrs["rows_dropped"] = dropped
    df.attrs["duplicates_dropped"] = duplicates_dropped
    return df


def lot_locations(df: pd.DataFrame) -> pd.DataFrame:
    """One row per lot: its id, coordinates and capacity.

    This is the frame proximity features must be built from. Computing them on
    the full per-timestamp frame instead is what made the original
    ``competitor_counts`` equal to the number of timesteps per lot.
    """
    return (
        df.groupby("lot_id", as_index=False)
        .agg(
            Latitude=("Latitude", "first"),
            Longitude=("Longitude", "first"),
            Capacity=("Capacity", "max"),
        )
        .sort_values("lot_id")
        .reset_index(drop=True)
    )
