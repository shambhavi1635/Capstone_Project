"""Comparing the three models.

There is no ground-truth price in this dataset, so none of these metrics scores
*accuracy*. They characterise behaviour: how far each model moves, how steadily,
how well it tracks occupancy, and how often it runs into its own limits. Those
are the questions you can honestly answer without a label.
"""

from __future__ import annotations

import pandas as pd

from .config import PricingConfig

MODELS = ["price_model1", "price_model2", "price_model3"]


def price_volatility(prices: pd.DataFrame, model: str) -> float:
    """Mean absolute price change between consecutive timesteps, within a lot.

    A pricing policy that jumps around is hard to post on a sign and harder for
    a driver to trust, so lower is usually better -- but a model that never
    moves is not pricing dynamically at all. Read it next to the price range.
    """
    return (
        prices.sort_values(["lot_id", "Timestamp"])
        .groupby("lot_id")[model]
        .diff()
        .abs()
        .mean()
    )


def band_saturation(prices: pd.DataFrame, model: str, cfg: PricingConfig) -> float:
    """Fraction of prices resting on the floor or the ceiling.

    High saturation means the band, not the model, is setting the price -- the
    signal is being clipped away rather than expressed.
    """
    series = prices[model]
    at_edge = (series <= cfg.price_floor + 1e-9) | (series >= cfg.price_ceiling - 1e-9)
    return float(at_edge.mean())


def occupancy_responsiveness(prices: pd.DataFrame, model: str) -> float:
    """Spearman correlation between price and occupancy.

    Rank-based, so it rewards getting the *ordering* right without assuming the
    relationship is linear. A dynamic pricing model should be strongly positive;
    the submitted Model 1 would score negative here.
    """
    return float(prices[model].corr(prices["occupancy_rate"], method="spearman"))


def revenue_proxy(prices: pd.DataFrame, model: str) -> float:
    """Sum of price x occupancy.

    .. warning::
       This is **not** a revenue forecast. It holds occupancy fixed while price
       changes, so it ignores the elasticity that is the entire point of dynamic
       pricing -- raising every price would "improve" it. Use it only to compare
       the scale at which models operate, never to pick a winner.
    """
    return float((prices[model] * prices["Occupancy"]).sum())


def compare_models(
    prices: pd.DataFrame, cfg: PricingConfig | None = None
) -> pd.DataFrame:
    """One row per model, one column per behavioural metric."""
    cfg = cfg or PricingConfig()

    rows = []
    for model in MODELS:
        series = prices[model]
        rows.append(
            {
                "model": model.replace("price_", ""),
                "mean_price": series.mean(),
                "min_price": series.min(),
                "max_price": series.max(),
                "price_std": series.std(),
                "volatility": price_volatility(prices, model),
                "band_saturation": band_saturation(prices, model, cfg),
                "occupancy_corr": occupancy_responsiveness(prices, model),
                "revenue_proxy": revenue_proxy(prices, model),
            }
        )
    return pd.DataFrame(rows).set_index("model")


def competitive_effect(prices: pd.DataFrame) -> pd.DataFrame:
    """How much Model 3 moved away from Model 2, split by competition.

    Lots with no rival inside the radius must show an effect of exactly zero --
    that is the control group, and it is what makes the rest interpretable.
    """
    delta = prices["price_model3"] - prices["price_model2"]
    grouped = prices.assign(
        delta=delta,
        has_rivals=prices["competitors_within_radius"] > 0,
    ).groupby("has_rivals")

    return pd.DataFrame(
        {
            "lots": grouped["lot_id"].nunique(),
            "rows": grouped.size(),
            "mean_shift": grouped["delta"].mean(),
            "mean_abs_shift": grouped["delta"].apply(lambda s: s.abs().mean()),
            "max_abs_shift": grouped["delta"].apply(lambda s: s.abs().max()),
        }
    ).rename(index={True: "has rivals", False: "isolated"})


def format_report(prices: pd.DataFrame, cfg: PricingConfig | None = None) -> str:
    """A plain-text summary, for the CLI."""
    comparison = compare_models(prices, cfg)
    effect = competitive_effect(prices)

    lines = [
        "Model comparison",
        "-" * 72,
        comparison.drop(columns="revenue_proxy").round(3).to_string(),
        "",
        "  volatility       mean absolute price change between timesteps",
        "  band_saturation  fraction of prices pinned to the floor or ceiling",
        "  occupancy_corr   Spearman correlation of price with occupancy",
        "",
        "Competitive effect (Model 3 vs Model 2)",
        "-" * 72,
        effect.round(4).to_string(),
        "",
        "  Isolated lots must show a shift of exactly 0 -- they are the control.",
    ]
    return "\n".join(lines)
