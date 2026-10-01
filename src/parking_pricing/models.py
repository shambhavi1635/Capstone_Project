"""The three pricing models.

All three share two invariants the original did not have:

* **Price is per lot per timestep.** The original aggregated into a daily
  tumbling window keyed only on the calendar day, collapsing all 14 lots into
  one number per day.
* **Price stays inside a band** around the base price. The original Model 1
  divided by minimum occupancy and reached 327 against a base of 10.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Model1Config, Model2Config, Model3Config, PricingConfig
from .features import add_demand_features, competitor_groups


def clamp_price(price: pd.Series | np.ndarray, cfg: PricingConfig) -> pd.Series:
    """Clip prices into ``[price_floor, price_ceiling]``."""
    return pd.Series(np.asarray(price, dtype=float)).clip(
        cfg.price_floor, cfg.price_ceiling
    )


def model1_baseline_linear(
    df: pd.DataFrame, cfg: PricingConfig | None = None
) -> pd.Series:
    """Model 1 -- price rises linearly with how full the lot is.

    ``price = base * (1 + alpha * occupancy_rate)``

    The submitted formula was ``10 + 5 * (capacity / min_occupancy)``, which is
    *decreasing* in occupancy: an almost-empty lot priced highest. This is the
    corrected monotone form; see docs/model-changes.md.
    """
    cfg = cfg or PricingConfig()
    m1: Model1Config = cfg.model1
    raw = cfg.base_price * (1.0 + m1.alpha * df["occupancy_rate"].to_numpy())
    return clamp_price(raw, cfg).set_axis(df.index)


def raw_demand(df: pd.DataFrame, m2: Model2Config) -> pd.Series:
    """Weighted sum of the normalised demand drivers."""
    weights = m2.weights()
    missing = [c for c in weights if c not in df.columns]
    if missing:
        raise ValueError(
            f"Demand features missing: {missing}. Call add_demand_features first."
        )
    total = np.zeros(len(df), dtype=float)
    for column, weight in weights.items():
        total += weight * df[column].to_numpy(dtype=float)
    return pd.Series(total, index=df.index)


def demand_bounds(m2: Model2Config) -> tuple[float, float]:
    """Theoretical min/max of :func:`raw_demand` given the weights.

    Derived from the weights rather than from the observed data, so a streaming
    row is normalised identically to a batch row -- the same input always yields
    the same price, regardless of what else is in the window.
    """
    values = list(m2.weights().values())
    return sum(w for w in values if w < 0), sum(w for w in values if w > 0)


def model2_demand_based(
    df: pd.DataFrame, cfg: PricingConfig | None = None
) -> pd.DataFrame:
    """Model 2 -- price driven by a normalised demand function.

    Returns a frame with ``demand`` (raw), ``demand_norm`` (0-1) and ``price``.
    Mid demand maps to exactly the base price, so the model is neutral by
    default and the configured weights describe deviations from it.
    """
    cfg = cfg or PricingConfig()
    m2: Model2Config = cfg.model2

    featured = add_demand_features(df)
    demand = raw_demand(featured, m2)

    low, high = demand_bounds(m2)
    span = high - low
    norm = (demand - low) / span if span else pd.Series(0.5, index=df.index)
    norm = norm.clip(0.0, 1.0)

    price = cfg.base_price * (1.0 + m2.lambda_ * (norm - 0.5))

    return pd.DataFrame(
        {
            "demand": demand,
            "demand_norm": norm,
            "price": clamp_price(price, cfg).set_axis(df.index),
        },
        index=df.index,
    )


def mean_competitor_price(
    priced: pd.DataFrame, groups: dict[str, list[str]]
) -> pd.Series:
    """Mean price of each lot's competitors at the same timestamp.

    Args:
        priced: must carry ``lot_id``, ``Timestamp`` and ``price``.
        groups: lot_id -> competitor lot_ids, from
            :func:`~parking_pricing.features.competitor_groups`.

    Returns:
        A series aligned to ``priced.index``. Lots with no competitor within the
        radius fall back to their own price, which makes the Model 3 adjustment
        a no-op for them rather than injecting a NaN.
    """
    wide = priced.pivot_table(
        index="Timestamp", columns="lot_id", values="price", aggfunc="mean"
    )

    means = {}
    for lot in wide.columns:
        rivals = [r for r in groups.get(lot, []) if r in wide.columns]
        means[lot] = wide[rivals].mean(axis=1) if rivals else wide[lot]
    mean_wide = pd.DataFrame(means)
    mean_wide.index.name = "Timestamp"
    mean_wide.columns.name = "lot_id"

    # Look the value up per (Timestamp, lot_id) pair and restore the row order.
    long = mean_wide.stack().rename("mean_competitor_price").reset_index()
    merged = priced[["Timestamp", "lot_id"]].merge(
        long, on=["Timestamp", "lot_id"], how="left"
    )
    out = merged["mean_competitor_price"].to_numpy()
    return pd.Series(out, index=priced.index).fillna(priced["price"])


def model3_competitive(
    priced: pd.DataFrame,
    groups: dict[str, list[str]],
    cfg: PricingConfig | None = None,
) -> pd.DataFrame:
    """Model 3 -- Model 2's price nudged toward the local market.

    Two effects, both bounded:

    * the price is pulled a fraction ``competitor_pull`` of the way toward the
      mean competitor price;
    * a lot that is effectively full while rivals are cheaper sheds
      ``full_lot_discount``, diverting demand instead of letting a queue build.
    """
    cfg = cfg or PricingConfig()
    m3: Model3Config = cfg.model3

    comp = mean_competitor_price(priced, groups)
    base = priced["price"].to_numpy(dtype=float)
    comp_arr = comp.to_numpy(dtype=float)

    adjusted = base + m3.competitor_pull * (comp_arr - base)

    is_full = priced["occupancy_rate"].to_numpy(dtype=float) >= m3.full_threshold
    undercut = comp_arr < base
    adjusted = np.where(is_full & undercut, adjusted * (1.0 - m3.full_lot_discount), adjusted)

    return pd.DataFrame(
        {
            "mean_competitor_price": comp,
            "is_full": is_full,
            "price": clamp_price(adjusted, cfg).set_axis(priced.index),
        },
        index=priced.index,
    )
