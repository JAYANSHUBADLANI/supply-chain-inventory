"""Lead-time parameters for the policy, and an explicit record of where they come from.

Safety stock under lead-time uncertainty needs a mean and a variance of replenishment
lead time: the time from raising a replenishment order on a supplier to that stock being
available to sell. The DataCo extract contains no supplier, no purchase order and no
receipt, so that quantity is not present in the data and cannot be estimated from it.

What the extract does contain is outbound fulfilment time, the interval from a customer
order to that order shipping. This module uses that as a stated proxy. The two are
different quantities and the substitution is a modelling assumption, not a measurement,
which is why every object returned from here carries the proxy flag and the source
description with it rather than handing back a bare number that would read as observed.

The audit in audit.py establishes that even the proxy is generated rather than real: it
is a discrete uniform draw conditioned on shipping mode and nothing else. That does not
stop it from parameterising a policy, but it does mean the resulting sigma is an assumed
input, so the policy is swept across a range of it rather than fitted to a point estimate.
"""

from dataclasses import dataclass, asdict
from typing import Dict, List

import numpy as np
import pandas as pd

DAYS_PER_WEEK = 7.0


@dataclass(frozen=True)
class LeadTimeParameters:
    """A lead-time mean and standard deviation with its provenance attached."""

    label: str
    source: str
    is_proxy: bool
    rows: int
    mean_days: float
    std_days: float

    @property
    def mean_weeks(self) -> float:
        return self.mean_days / DAYS_PER_WEEK

    @property
    def std_weeks(self) -> float:
        return self.std_days / DAYS_PER_WEEK

    @property
    def coefficient_of_variation(self) -> float:
        return self.std_days / self.mean_days if self.mean_days > 0 else float("nan")

    def scaled(self, std_multiplier: float) -> "LeadTimeParameters":
        """The same lead time with its variability scaled, for sensitivity analysis."""
        return LeadTimeParameters(
            label=f"{self.label} (sigma x{std_multiplier:g})",
            source=f"{self.source}, standard deviation scaled by {std_multiplier:g}",
            is_proxy=True,
            rows=self.rows,
            mean_days=self.mean_days,
            std_days=self.std_days * std_multiplier,
        )

    def to_dict(self) -> Dict:
        payload = asdict(self)
        payload.update(
            {
                "mean_weeks": self.mean_weeks,
                "std_weeks": self.std_weeks,
                "coefficient_of_variation": self.coefficient_of_variation,
            }
        )
        return payload


PROXY_SOURCE = (
    "outbound fulfilment time (Days for shipping (real)) used as a stated proxy for "
    "replenishment lead time; the extract contains no supplier or purchase order data"
)


def pooled_parameters(frame: pd.DataFrame, column: str = "Days for shipping (real)") -> LeadTimeParameters:
    """Lead-time proxy pooled across shipping modes, weighted by observed volume."""
    values = frame[column]
    return LeadTimeParameters(
        label="pooled across shipping modes",
        source=PROXY_SOURCE,
        is_proxy=True,
        rows=int(len(values)),
        mean_days=float(values.mean()),
        std_days=float(values.std(ddof=1)),
    )


def parameters_by_mode(
    frame: pd.DataFrame, column: str = "Days for shipping (real)"
) -> List[LeadTimeParameters]:
    """Lead-time proxy split by shipping mode, which is the only split that changes it."""
    out = []
    for mode, block in frame.groupby("Shipping Mode"):
        values = block[column]
        out.append(
            LeadTimeParameters(
                label=str(mode),
                source=f"{PROXY_SOURCE}; restricted to shipping mode {mode}",
                is_proxy=True,
                rows=int(len(values)),
                mean_days=float(values.mean()),
                std_days=float(values.std(ddof=1)),
            )
        )
    return out


def empirical_distribution(
    frame: pd.DataFrame, column: str = "Days for shipping (real)"
) -> pd.Series:
    """Probability mass of the lead-time proxy over its integer support."""
    counts = frame[column].value_counts().sort_index()
    return (counts / counts.sum()).rename("probability")


def scheduled_versus_real(frame: pd.DataFrame) -> pd.DataFrame:
    """Where the promised shipment window differs systematically from the realised one.

    Planning against the scheduled number alone would assume a deterministic lead time
    that the data does not deliver, and would be biased as well as overconfident: the
    realised mean sits above the schedule for two of the four modes and the realised
    distribution is identical across the two highest volume modes despite their schedules
    differing by two days.
    """
    grouped = frame.groupby("Shipping Mode")
    out = pd.DataFrame(
        {
            "rows": grouped.size(),
            "scheduled_days": grouped["Days for shipment (scheduled)"].first(),
            "realised_mean_days": grouped["Days for shipping (real)"].mean().round(4),
            "realised_std_days": grouped["Days for shipping (real)"].std(ddof=1).round(4),
        }
    )
    out["bias_days"] = (out["realised_mean_days"] - out["scheduled_days"]).round(4)
    out["late_share"] = grouped.apply(
        lambda b: (b["Days for shipping (real)"] > b["Days for shipment (scheduled)"]).mean(),
        include_groups=False,
    ).round(4)
    out["schedule_understates_variability"] = out["realised_std_days"] > 0
    return out.sort_values("rows", ascending=False)


def sensitivity_grid(base: LeadTimeParameters, multipliers: List[float]) -> List[LeadTimeParameters]:
    """The base proxy rescaled across a range, since sigma_L is assumed and not measured."""
    return [base.scaled(m) for m in multipliers]


def sample_leadtimes(
    distribution: pd.Series, size: int, rng: np.random.Generator
) -> np.ndarray:
    """Draw lead times from the empirical mass function, for the simulation backtest."""
    return rng.choice(distribution.index.values, size=size, p=distribution.values)
