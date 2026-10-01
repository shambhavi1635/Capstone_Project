"""Dynamic pricing for urban parking lots.

Three pricing models over a 14-lot, 73-day dataset:

* Model 1 -- rule-based, linear in occupancy;
* Model 2 -- demand function over occupancy, queue, traffic, special days and
  vehicle mix;
* Model 3 -- Model 2 adjusted for nearby competitors' prices.

The entry point is :func:`parking_pricing.pipeline.run_pipeline`.
"""

from .config import PricingConfig
from .pipeline import daily_summary, run_pipeline, write_outputs

__version__ = "1.0.0"

__all__ = ["PricingConfig", "run_pipeline", "daily_summary", "write_outputs"]
