"""Figures for the three pricing models.

Colour roles follow a validated categorical palette: slots 1-3 (blue, orange,
aqua) clear the all-pairs colour-vision-deficiency and normal-vision separation
floors on this surface. Aqua sits below 3:1 contrast against the surface, so
every series also carries a direct end-of-line label -- identity is never
conveyed by colour alone.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Render to file; never require a display.

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- Design tokens -----------------------------------------------------------

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e6e5e0"

#: Categorical slots 1-3, in fixed order. Never cycled, never reassigned by rank.
SERIES_COLORS = {
    "price_model1": "#2a78d6",  # blue
    "price_model2": "#eb6834",  # orange
    "price_model3": "#1baf7a",  # aqua
}

SERIES_LABELS = {
    "price_model1": "Model 1 · rule-based",
    "price_model2": "Model 2 · demand",
    "price_model3": "Model 3 · competitive",
}

#: Single hue, light to dark, for magnitude.
SEQUENTIAL_HUE = "#2a78d6"

LINE_WIDTH = 2.0

#: Models 2 and 3 track each other closely, so the topmost is dashed and the one
#: beneath shows through the gaps. Identity still rests on colour *and* label.
SERIES_DASHES = {
    "price_model1": (None, None),
    "price_model2": (None, None),
    "price_model3": (5, 3),
}

MODELS = list(SERIES_COLORS)


def _direct_labels(ax, points, min_gap_frac: float = 0.045) -> None:
    """Annotate each line end, nudging labels apart when they would overprint.

    Args:
        points: (label, x, y) per series, in draw order.
        min_gap_frac: minimum vertical separation, as a fraction of the y-range.
    """
    lo, hi = ax.get_ylim()
    min_gap = (hi - lo) * min_gap_frac

    ordered = sorted(points, key=lambda p: p[2])
    placed: list[float] = []
    for _, _, y in ordered:
        if placed and y - placed[-1] < min_gap:
            placed.append(placed[-1] + min_gap)
        else:
            placed.append(y)

    for (label, x, y), y_text in zip(ordered, placed):
        ax.annotate(
            label,
            xy=(x, y),
            xytext=(8, (y_text - y) * 72 / (hi - lo) * ax.get_figure().get_figheight() / 2),
            textcoords="offset points",
            color=TEXT_SECONDARY, fontsize=9, va="center",
        )


def _style_axes(ax, xlabel: str, ylabel: str) -> None:
    """Recessive grid and axes; text in ink tokens, never a series colour."""
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=1.0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9, length=0)
    ax.set_xlabel(xlabel, color=TEXT_SECONDARY, fontsize=10, labelpad=8)
    ax.set_ylabel(ylabel, color=TEXT_SECONDARY, fontsize=10, labelpad=8)


def _title(ax, title: str, subtitle: str | None = None) -> None:
    ax.set_title(title, color=TEXT_PRIMARY, fontsize=14, loc="left", pad=20 if subtitle else 12)
    if subtitle:
        ax.text(
            0.0, 1.02, subtitle,
            transform=ax.transAxes, color=TEXT_SECONDARY, fontsize=10, va="bottom",
        )


def _new_figure(width=11.0, height=5.5):
    fig, ax = plt.subplots(figsize=(width, height), facecolor=SURFACE)
    return fig, ax


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_price_over_time(daily: pd.DataFrame, out_dir: Path) -> Path:
    """Daily mean price per model, averaged across lots."""
    series = daily.groupby("day", as_index=False)[MODELS].mean()

    fig, ax = _new_figure()
    for model in MODELS:
        ax.plot(
            series["day"], series[model],
            color=SERIES_COLORS[model], linewidth=LINE_WIDTH,
            label=SERIES_LABELS[model], solid_capstyle="round",
            dashes=SERIES_DASHES[model] if SERIES_DASHES[model][0] else (None, None),
        )

    _style_axes(ax, "Date", "Mean price")
    # Direct labels are the relief that lets a low-contrast hue carry identity.
    _direct_labels(
        ax,
        [
            (SERIES_LABELS[m].split(" · ")[0], series["day"].iloc[-1], series[m].iloc[-1])
            for m in MODELS
        ],
    )
    _title(
        ax,
        "Daily mean price by model",
        f"Averaged across {daily['lot_id'].nunique()} lots · all prices held inside the configured band",
    )
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.legend(
        frameon=False, loc="upper left", fontsize=9,
        labelcolor=TEXT_SECONDARY, ncols=3, bbox_to_anchor=(0.0, -0.16),
    )
    fig.subplots_adjust(right=0.88)
    return _save(fig, out_dir / "price_over_time.png")


def plot_price_vs_occupancy(prices: pd.DataFrame, out_dir: Path) -> Path:
    """Mean price against occupancy, binned.

    This is the figure that shows the Model 1 correction: the submitted formula
    sloped downward here, pricing an empty lot highest.
    """
    bins = np.linspace(0.0, 1.0, 21)
    binned = prices.assign(
        occ_bin=pd.cut(prices["occupancy_rate"], bins, include_lowest=True)
    )
    grouped = binned.groupby("occ_bin", observed=True)[MODELS].mean()
    centres = [interval.mid for interval in grouped.index]

    fig, ax = _new_figure(width=10.0, height=5.5)
    for model in MODELS:
        ax.plot(
            centres, grouped[model],
            color=SERIES_COLORS[model], linewidth=LINE_WIDTH,
            marker="o", markersize=5, markeredgecolor=SURFACE, markeredgewidth=1.5,
            label=SERIES_LABELS[model],
            dashes=SERIES_DASHES[model] if SERIES_DASHES[model][0] else (None, None),
        )

    _style_axes(ax, "Occupancy rate", "Mean price")
    _direct_labels(
        ax,
        [
            (SERIES_LABELS[m].split(" · ")[0], centres[-1], grouped[m].iloc[-1])
            for m in MODELS
        ],
    )
    _title(
        ax,
        "Price response to occupancy",
        "Every model now rises with occupancy — the submitted Model 1 sloped the other way",
    )
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend(
        frameon=False, loc="upper left", fontsize=9,
        labelcolor=TEXT_SECONDARY, ncols=3, bbox_to_anchor=(0.0, -0.16),
    )
    fig.subplots_adjust(right=0.86)
    return _save(fig, out_dir / "price_vs_occupancy.png")


def plot_price_by_lot(prices: pd.DataFrame, out_dir: Path) -> Path:
    """Mean competitive price per lot -- magnitude, so one hue light to dark."""
    by_lot = (
        prices.groupby("lot_id", as_index=False)
        .agg(price=("price_model3", "mean"), rivals=("competitors_within_radius", "first"))
        .sort_values("price")
    )

    lo, hi = by_lot["price"].min(), by_lot["price"].max()
    span = hi - lo or 1.0
    base = matplotlib.colors.to_rgb(SEQUENTIAL_HUE)
    # Light-to-dark ramp in a single hue: lift the lightest end toward the surface.
    colors = [
        tuple(c + (1.0 - c) * 0.55 * (1.0 - (p - lo) / span) for c in base)
        for p in by_lot["price"]
    ]

    # The rival count rides in the tick label rather than inside the bar, where
    # muted ink on a saturated fill would fall under the contrast floor.
    labels = [
        f"{lot}   {n} rival{'' if n == 1 else 's'}"
        for lot, n in zip(by_lot["lot_id"], by_lot["rivals"])
    ]

    fig, ax = _new_figure(width=9.5, height=6.0)
    ax.barh(
        labels, by_lot["price"],
        color=colors, height=0.72, linewidth=1.5, edgecolor=SURFACE,
    )
    for label, price in zip(labels, by_lot["price"]):
        ax.annotate(
            f"{price:.2f}",
            xy=(price, label), xytext=(6, 0), textcoords="offset points",
            color=TEXT_PRIMARY, fontsize=9, va="center",
        )

    _style_axes(ax, "Mean price (Model 3) · lots labelled with rivals within 500 m", "")
    _title(
        ax,
        "Mean competitive price by lot",
        "Per-lot pricing — the submitted pipeline produced one price per day for all lots combined",
    )
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, by_lot["price"].max() * 1.18)
    return _save(fig, out_dir / "price_by_lot.png")


def render_all(
    prices: pd.DataFrame, daily: pd.DataFrame, out_dir: str | Path = "outputs"
) -> list[Path]:
    """Render every figure; return the paths written."""
    out_dir = Path(out_dir)
    return [
        plot_price_over_time(daily, out_dir),
        plot_price_vs_occupancy(prices, out_dir),
        plot_price_by_lot(prices, out_dir),
    ]
