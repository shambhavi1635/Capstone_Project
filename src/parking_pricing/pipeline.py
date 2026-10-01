"""Orchestration: dataset in, per-lot per-timestep prices out."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import PricingConfig
from .data import load_dataset, lot_locations
from .features import add_demand_features, competitor_groups, proximity_features
from .models import model1_baseline_linear, model2_demand_based, model3_competitive

OUTPUT_COLUMNS = [
    "lot_id",
    "Timestamp",
    "day",
    "Occupancy",
    "Capacity",
    "occupancy_rate",
    "nearest_competitor_m",
    "competitors_within_radius",
    "demand",
    "demand_norm",
    "price_model1",
    "price_model2",
    "mean_competitor_price",
    "is_full",
    "price_model3",
]


def run_pipeline(
    data_path: str | Path | None = None,
    cfg: PricingConfig | None = None,
) -> pd.DataFrame:
    """Run all three models over the dataset.

    Returns one tidy frame with a row per (lot, timestamp) and a price column
    per model, so the three are directly comparable without a merge.
    """
    cfg = cfg or PricingConfig()

    df = load_dataset(data_path)
    lots = proximity_features(lot_locations(df), cfg.competitor_radius_m)
    groups = competitor_groups(lots, cfg.competitor_radius_m)

    df = df.merge(
        lots[["lot_id", "nearest_competitor_m", "competitors_within_radius"]],
        on="lot_id",
        how="left",
    )
    df = add_demand_features(df)

    df["price_model1"] = model1_baseline_linear(df, cfg)

    m2 = model2_demand_based(df, cfg)
    df["demand"] = m2["demand"]
    df["demand_norm"] = m2["demand_norm"]
    df["price_model2"] = m2["price"]

    m3 = model3_competitive(
        df[["lot_id", "Timestamp", "occupancy_rate"]].assign(price=df["price_model2"]),
        groups,
        cfg,
    )
    df["mean_competitor_price"] = m3["mean_competitor_price"]
    df["is_full"] = m3["is_full"]
    df["price_model3"] = m3["price"]

    return df[OUTPUT_COLUMNS].sort_values(["lot_id", "Timestamp"]).reset_index(drop=True)


def daily_summary(prices: pd.DataFrame) -> pd.DataFrame:
    """Mean price per lot per day, for plotting and for sanity checks."""
    return (
        prices.groupby(["lot_id", "day"], as_index=False)
        .agg(
            occupancy_rate=("occupancy_rate", "mean"),
            price_model1=("price_model1", "mean"),
            price_model2=("price_model2", "mean"),
            price_model3=("price_model3", "mean"),
        )
        .sort_values(["lot_id", "day"])
        .reset_index(drop=True)
    )


def write_outputs(prices: pd.DataFrame, out_dir: str | Path = "outputs") -> dict[str, Path]:
    """Write the tidy prices and the daily summary; return the paths written."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "prices": out_dir / "prices.csv",
        "daily": out_dir / "daily_summary.csv",
    }
    prices.to_csv(paths["prices"], index=False)
    daily_summary(prices).to_csv(paths["daily"], index=False)
    return paths
