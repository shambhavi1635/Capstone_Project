"""Smoke tests for figure rendering."""

import matplotlib
import pytest

from parking_pricing.pipeline import daily_summary, run_pipeline
from parking_pricing.plots import SERIES_COLORS, SERIES_LABELS, render_all


@pytest.fixture(scope="module")
def prices():
    return run_pipeline()


def test_renders_every_figure(prices, tmp_path):
    paths = render_all(prices, daily_summary(prices), tmp_path)
    assert len(paths) == 3
    for path in paths:
        assert path.exists()
        assert path.stat().st_size > 0


def test_backend_is_headless():
    """Figures must render without a display, so CI never needs one."""
    assert matplotlib.get_backend().lower() == "agg"


def test_every_series_has_a_colour_and_a_label():
    """Identity is never carried by colour alone."""
    assert set(SERIES_COLORS) == set(SERIES_LABELS)
    assert len(set(SERIES_COLORS.values())) == len(SERIES_COLORS)
