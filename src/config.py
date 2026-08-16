"""Configuration loading and the typed parameter blocks the pipeline reads.

Every economic parameter in this project is an assumption rather than an observation.
The DataCo extract carries prices and revenue but no holding cost, no ordering cost and
no stockout penalty, so those live here in one place where they can be found, changed
and cited in the write-up instead of being scattered through the code as literals.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "config.yaml"


@dataclass(frozen=True)
class CostAssumptions:
    """Stated cost parameters. None of these are observed in the source data."""

    annual_holding_rate: float
    ordering_cost_per_order: float
    unit_cost_as_share_of_price: float
    weeks_per_year: int

    def unit_cost(self, unit_price: float) -> float:
        """Assumed cost of goods for one unit, derived from its selling price."""
        return unit_price * self.unit_cost_as_share_of_price

    def weekly_holding_cost(self, unit_price: float) -> float:
        """Cost of carrying one unit for one week."""
        return self.unit_cost(unit_price) * self.annual_holding_rate / self.weeks_per_year

    def annual_holding_cost(self, unit_price: float) -> float:
        """Cost of carrying one unit for one year."""
        return self.unit_cost(unit_price) * self.annual_holding_rate


@dataclass(frozen=True)
class SegmentationConfig:
    """ABC revenue cut points, XYZ variability cut points and per-tier service targets."""

    abc_thresholds: List[float]
    xyz_thresholds: List[float]
    service_levels: Dict[str, float]

    def service_level_for(self, abc_class: str) -> float:
        return self.service_levels[abc_class]


@dataclass(frozen=True)
class DemandConfig:
    frequency: str
    min_weeks_covered: int
    core_min_weeks_covered: int


@dataclass(frozen=True)
class LeadTimeConfig:
    """Which observed shipping mode stands in for replenishment lead time, and its sweep.

    Inbound replenishment of a distribution centre moves on standard freight rather than
    a same day courier, so the default proxy is the standard mode rather than the pooled
    mix across all four. Setting proxy_mode to null pools every mode instead.
    """

    proxy_mode: Optional[str]
    sensitivity_multipliers: List[float]


@dataclass(frozen=True)
class PolicyConfig:
    time_base: str
    bootstrap_draws: int
    bootstrap_seed: int
    service_level_sweep: List[float]
    representative_segments: List[str]


@dataclass(frozen=True)
class BacktestConfig:
    test_weeks: int
    baseline_safety_stock_pct_of_mean_demand: float
    warmup_days: int
    leadtime_seed: int
    unmet_demand: str

    @property
    def test_days(self) -> int:
        return self.test_weeks * 7


class Config:
    """Parsed project configuration with paths resolved against the project root."""

    def __init__(self, raw: dict, root: Path = PROJECT_ROOT):
        self._raw = raw
        self.root = root

    @classmethod
    def load(cls, path: Path = DEFAULT_CONFIG) -> "Config":
        with open(path) as handle:
            return cls(yaml.safe_load(handle))

    def path(self, key: str) -> Path:
        return (self.root / self._raw["paths"][key]).resolve()

    def section(self, name: str) -> dict:
        return self._raw[name]

    @property
    def costs(self) -> CostAssumptions:
        block = self._raw["costs"]
        return CostAssumptions(
            annual_holding_rate=block["annual_holding_rate"],
            ordering_cost_per_order=block["ordering_cost_per_order"],
            unit_cost_as_share_of_price=block["unit_cost_as_share_of_price"],
            weeks_per_year=block["weeks_per_year"],
        )

    @property
    def segmentation(self) -> SegmentationConfig:
        block = self._raw["segmentation"]
        return SegmentationConfig(
            abc_thresholds=block["abc_thresholds"],
            xyz_thresholds=block["xyz_thresholds"],
            service_levels=block["service_levels"],
        )

    @property
    def demand(self) -> DemandConfig:
        block = self._raw["demand"]
        return DemandConfig(
            frequency=block["frequency"],
            min_weeks_covered=block["min_weeks_covered"],
            core_min_weeks_covered=block["core_min_weeks_covered"],
        )

    @property
    def leadtime(self) -> LeadTimeConfig:
        block = self._raw["leadtime"]
        return LeadTimeConfig(
            proxy_mode=block["proxy_mode"],
            sensitivity_multipliers=block["sensitivity_multipliers"],
        )

    @property
    def policy(self) -> PolicyConfig:
        block = self._raw["policy"]
        return PolicyConfig(
            time_base=block["time_base"],
            bootstrap_draws=block["bootstrap_draws"],
            bootstrap_seed=block["bootstrap_seed"],
            service_level_sweep=block["service_level_sweep"],
            representative_segments=block["representative_segments"],
        )

    @property
    def backtest(self) -> BacktestConfig:
        block = self._raw["backtest"]
        return BacktestConfig(
            test_weeks=block["test_weeks"],
            baseline_safety_stock_pct_of_mean_demand=block[
                "baseline_safety_stock_pct_of_mean_demand"
            ],
            warmup_days=block["warmup_days"],
            leadtime_seed=block["leadtime_seed"],
            unmet_demand=block["unmet_demand"],
        )
