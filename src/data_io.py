"""Raw extract loading, exclusion accounting and the weekly demand panel.

The source file is the DataCo Smart Supply Chain extract. It is not UTF-8: reading it
with the pandas default fails on the third byte of the first data row, so the encoding
is read from config rather than left implicit.

Two exclusions are applied before anything downstream sees the data, and both are
recorded in a ledger that reconciles row for row against the raw count. Cancelled and
suspected fraud lines never shipped and are not demand. The extract also changes
character partway through, which is handled in audit.py and enforced here as a window
cut, because fitting an inventory policy across that boundary would mix two different
data generating processes.
"""

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


@dataclass
class ExclusionLedger:
    """Row level reconciliation from the raw extract to the modelling population."""

    raw_rows: int
    excluded_cancelled_or_fraud: int
    excluded_after_clean_window: int
    kept_rows: int
    clean_window_start: str
    clean_window_end: str
    products_kept: int
    weeks_kept: int

    def to_dict(self) -> Dict:
        return asdict(self)

    def check(self) -> None:
        """Fail loudly if the exclusions do not account for every raw row."""
        total = (
            self.excluded_cancelled_or_fraud
            + self.excluded_after_clean_window
            + self.kept_rows
        )
        if total != self.raw_rows:
            raise ValueError(
                f"exclusion ledger does not reconcile: {total} accounted for against "
                f"{self.raw_rows} raw rows"
            )


def load_raw(config) -> pd.DataFrame:
    """Read the raw CSV and parse the two date columns into real timestamps."""
    block = config.section("data")
    path = config.path("raw_csv")
    if not path.exists():
        raise FileNotFoundError(
            f"raw extract not found at {path}. Set paths.raw_csv in config/config.yaml "
            f"to the location of DataCoSupplyChainDataset.csv"
        )
    frame = pd.read_csv(path, encoding=block["encoding"], low_memory=False)
    frame["order_date"] = pd.to_datetime(frame[block["order_date_column"]])
    frame["ship_date"] = pd.to_datetime(frame[block["ship_date_column"]])
    return frame


def apply_exclusions(frame: pd.DataFrame, config) -> tuple:
    """Drop non-demand rows and everything past the clean window, with a ledger."""
    block = config.section("data")
    raw_rows = len(frame)

    cancelled_mask = frame["Order Status"].isin(block["excluded_order_statuses"])
    after_cancel = frame.loc[~cancelled_mask].copy()

    window_end = pd.Timestamp(block["clean_window_end"])
    window_mask = after_cancel["order_date"] < window_end
    kept = after_cancel.loc[window_mask].copy()

    kept["week"] = kept["order_date"].dt.to_period("W").dt.start_time

    ledger = ExclusionLedger(
        raw_rows=raw_rows,
        excluded_cancelled_or_fraud=int(cancelled_mask.sum()),
        excluded_after_clean_window=int((~window_mask).sum()),
        kept_rows=len(kept),
        clean_window_start=str(kept["order_date"].min().date()),
        clean_window_end=str(kept["order_date"].max().date()),
        products_kept=int(kept[block["planning_unit"]].nunique()),
        weeks_kept=int(kept["week"].nunique()),
    )
    ledger.check()
    return kept, ledger


def calendar_weeks(frame: pd.DataFrame) -> pd.DatetimeIndex:
    """Every week between the first and last order, including fully inactive ones.

    Taking the column index from the weeks that happen to appear in the data would drop
    any week with no activity anywhere, which would then be invisible to the coverage and
    gap checks rather than counted against them.
    """
    weeks = frame["week"]
    return pd.date_range(weeks.min(), weeks.max(), freq="W-MON")


def weekly_demand_panel(frame: pd.DataFrame, config) -> pd.DataFrame:
    """Planning unit by week matrix of order quantity, on a complete calendar index.

    Weeks in which a product recorded no orders are held as NaN rather than zero so that
    coverage can be distinguished from genuine zero demand. Callers that need a demand
    series fill explicitly.
    """
    block = config.section("data")
    unit = block["planning_unit"]
    panel = (
        frame.groupby([unit, "week"])[block["quantity_column"]]
        .sum()
        .unstack()
        .reindex(columns=calendar_weeks(frame))
    )
    panel.index.name = unit
    panel.columns.name = "week"
    return panel


def daily_demand_panel(frame: pd.DataFrame, config) -> pd.DataFrame:
    """Planning unit by calendar day matrix of order quantity.

    The policy math runs on a daily base because lead time is measured in whole days.
    Unlike the weekly panel this fills absent days with zero rather than NaN: coverage has
    already been established at weekly granularity, and a day inside an active unit's span
    with no order is genuine zero demand.
    """
    block = config.section("data")
    unit = block["planning_unit"]
    day = frame["order_date"].dt.normalize()
    days = pd.date_range(day.min(), day.max(), freq="D")
    panel = (
        frame.assign(day=day)
        .groupby([unit, "day"])[block["quantity_column"]]
        .sum()
        .unstack()
        .reindex(columns=days)
        .fillna(0.0)
    )
    panel.index.name = unit
    panel.columns.name = "day"
    return panel


def coverage(panel: pd.DataFrame) -> pd.Series:
    """Number of weeks in which each planning unit recorded at least one order."""
    return panel.notna().sum(axis=1).rename("weeks_covered")


def interior_gaps(panel: pd.DataFrame) -> pd.Series:
    """Weeks with no orders that fall between a unit's first and last active week.

    A unit with zero interior gaps has a continuous series and can be simulated period
    by period. A unit whose activity is a handful of scattered bursts cannot, and this
    is what separates the two cases.
    """
    result = {}
    for unit, row in panel.iterrows():
        active = row.notna().values
        if not active.any():
            result[unit] = 0
            continue
        first = active.argmax()
        last = len(active) - 1 - active[::-1].argmax()
        result[unit] = int((~active[first : last + 1]).sum())
    return pd.Series(result, name="interior_gaps")


def unit_prices(frame: pd.DataFrame, config) -> pd.Series:
    """Median observed selling price per planning unit."""
    unit = config.section("data")["planning_unit"]
    return frame.groupby(unit)["Product Price"].median().rename("unit_price")


def revenue(frame: pd.DataFrame, config) -> pd.Series:
    """Total observed sales value per planning unit over the clean window."""
    unit = config.section("data")["planning_unit"]
    return frame.groupby(unit)["Sales"].sum().rename("revenue")


def cache_frame(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)
    return path


def load_cached(path: Path) -> Optional[pd.DataFrame]:
    if path.exists():
        return pd.read_parquet(path)
    return None


def write_json(payload: Dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, default=str)
    return path


def write_table(frame: pd.DataFrame, path: Path, index: bool = True) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=index)
    return path
