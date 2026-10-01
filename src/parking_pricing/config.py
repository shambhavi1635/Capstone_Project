"""Tunable parameters for the three pricing models.

Every magic number in the pipeline lives here so experiments are a one-line
change instead of a hunt through the model code.
"""

from dataclasses import dataclass, field

#: Starting price for every lot, in the dataset's currency unit.
BASE_PRICE = 10.0

#: Prices are never allowed outside this multiple of BASE_PRICE. Keeping the
#: band tight is what stops a single busy hour from producing an absurd fare
#: (the original Model 1 reached 327 for a base of 10).
MIN_PRICE_MULTIPLIER = 0.5
MAX_PRICE_MULTIPLIER = 2.0

#: Two lots are "competitors" when they sit within this many metres.
COMPETITOR_RADIUS_M = 500.0

#: Earth radius in metres, for the haversine distance.
EARTH_RADIUS_M = 6_371_000.0


@dataclass(frozen=True)
class Model1Config:
    """Rule-based linear model: price rises with how full the lot is."""

    #: Price swing at full occupancy, as a fraction of BASE_PRICE.
    alpha: float = 1.0


@dataclass(frozen=True)
class Model2Config:
    """Demand-function model.

    Each coefficient weights one normalised (0-1) demand driver, so the raw
    weights are directly comparable to one another.
    """

    occupancy_weight: float = 1.0
    queue_weight: float = 0.5
    traffic_weight: float = -0.7
    special_day_weight: float = 1.5
    vehicle_weight: float = 0.8

    #: How strongly normalised demand moves price.
    lambda_: float = 1.0

    def weights(self) -> dict[str, float]:
        return {
            "occupancy_rate": self.occupancy_weight,
            "queue_norm": self.queue_weight,
            "traffic_norm": self.traffic_weight,
            "special_day": self.special_day_weight,
            "vehicle_norm": self.vehicle_weight,
        }


@dataclass(frozen=True)
class Model3Config:
    """Competition-aware model layered on top of Model 2."""

    #: Weight on the gap between our price and the local mean competitor price.
    competitor_pull: float = 0.3

    #: When the lot is full and rivals are cheaper, shed this fraction of price
    #: to push demand next door rather than queue people at the gate.
    full_lot_discount: float = 0.1

    #: Occupancy rate at or above which a lot counts as full.
    full_threshold: float = 0.95


@dataclass(frozen=True)
class PricingConfig:
    base_price: float = BASE_PRICE
    min_multiplier: float = MIN_PRICE_MULTIPLIER
    max_multiplier: float = MAX_PRICE_MULTIPLIER
    competitor_radius_m: float = COMPETITOR_RADIUS_M
    model1: Model1Config = field(default_factory=Model1Config)
    model2: Model2Config = field(default_factory=Model2Config)
    model3: Model3Config = field(default_factory=Model3Config)

    @property
    def price_floor(self) -> float:
        return self.base_price * self.min_multiplier

    @property
    def price_ceiling(self) -> float:
        return self.base_price * self.max_multiplier
