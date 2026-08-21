"""
Data adapter for CatBoost.

Unlike Prophet, CatBoost has no native concept of dates, trend, or
seasonality — it just sees a table of numeric/categorical features and a
target. All of the time-series structure Prophet gets "for free" has to
be engineered here as explicit columns.

--- Redesigned: lag/rolling features removed (real accuracy finding) ---

The original version of this adapter (Phase 3) engineered lag(1/2/3/7/14)
and rolling-window(7/14) features from the target's own history. Because
CatBoostWrapper.predict() is a single batched call (not a per-step
recursive loop), those features had to be "frozen" once at the end of
training history and broadcast unchanged across the entire forecast
horizon (the "direct" multi-horizon strategy).

Real-dataset testing surfaced a serious consequence of that design: on a
144-day held-out test window, CatBoost's MAPE was 52.6% in the first half
of the window and 27.3% in the second half — and a targeted breakdown
showed December_Dip days (a known, flagged demand-drop period in the
source data) at 67-90% MAPE, far above the ~35-40% baseline on ordinary
days. Root cause: a frozen snapshot of "recent history" has no way to
see a sudden target level-shift (a promotional spike or a post-holiday
dip) coming, and keeps predicting the pre-shift baseline for the entire
horizon. A separately-built, dedicated CatBoost notebook against the
same dataset — using an explicit trend index and cyclical (sin/cos)
seasonality instead of lag/rolling features — achieved ~16% MAPE with no
such failure mode, because none of ITS features depend on the target's
own recent history, so none of them need to be frozen in the first
place.

This version adopts that proven approach: an explicit linear trend index
(`time_idx`, an absolute day/week offset from a fixed reference date —
NOT a row-position counter, so it stays continuous and gap-robust across
the train/future boundary), continuous cyclical encodings of
yearly/half-yearly seasonality (and, for daily data, weekly/monthly
seasonality too), plain calendar fields, and independent columns passed
through as-is. None of these require knowledge of the target's own
history, so transform_future() computes every one of them directly from
dates alone — exactly like transform_train() does.

Practical consequence: this adapter is effectively STATELESS again
(matching ProphetAdapter), except for a single `_reference_date`
retained across the train/future boundary purely to keep `time_idx`
continuous. There is no more "frozen feature", no more lag-warmup row
dropping, and transform_future() no longer depends on real target
history at all — only on a reference date established by a prior
transform_train() call.
"""

from typing import List, Optional

import numpy as np
import pandas as pd

from horizon_forecaster.data_adapter.base_adapter import BaseAdapter
from horizon_forecaster.exceptions import InvalidConfigError

_YEARLY_PERIOD_DAYS = 365.25
_YEARLY_PERIOD_WEEKS = 52.0

class CatBoostAdapter(BaseAdapter):
    """
    Reshapes raw user data into a flat feature table CatBoost can train
    and predict on: a trend index, cyclical/calendar features, and
    independent columns passed through as-is.

    Args:
        date_column, target_column, independent_columns: see BaseAdapter.
        frequency: "daily" or "weekly" — controls whether the
            higher-resolution daily-only features (day-of-week,
            weekly/monthly cyclicals) are included, and which period
            length the yearly cyclical uses. Required, since this
            adapter's feature set is genuinely frequency-dependent
            (unlike ProphetAdapter, which has no frequency awareness of
            its own).
    """

    def __init__(
        self,
        date_column: str,
        target_column: str,
        independent_columns: Optional[List[str]] = None,
        frequency: Optional[str] = None,
    ):
        super().__init__(date_column, target_column, independent_columns, frequency=frequency)
        if frequency not in ("daily", "weekly"):
            raise InvalidConfigError(
                f"CatBoostAdapter requires frequency='daily' or 'weekly', got {frequency!r}.",
                param_name="frequency",
                provided_value=frequency,
            )

        # Set by transform_train(); required by transform_future() purely
        # to keep the trend index (time_idx) continuous and gap-robust
        # across the train/future boundary. Unlike the old lag/rolling
        # design, no target history is retained or needed here.
        self._reference_date: Optional[pd.Timestamp] = None

    def transform_train(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        required_columns = [self.date_column, self.target_column] + self.independent_columns
        self._validate_columns_present(raw_df, required_columns)

        df = pd.DataFrame()
        df["ds"] = pd.to_datetime(raw_df[self.date_column])
        df["y"] = raw_df[self.target_column]
        for col in self.independent_columns:
            df[col] = raw_df[col]

        df = df.sort_values("ds").reset_index(drop=True)
        self._reference_date = df["ds"].iloc[0]

        return self._add_features(df)

    def transform_future(
        self, future_dates: pd.Series, future_independent_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        if self._reference_date is None:
            raise InvalidConfigError(
                "CatBoostAdapter.transform_future() called before transform_train(). "
                "The trend index (time_idx) needs a reference date established by "
                "transform_train() first.",
                param_name="transform_future",
                provided_value=None,
            )

        df = pd.DataFrame()
        df["ds"] = pd.to_datetime(future_dates).reset_index(drop=True)

        if self.independent_columns:
            if future_independent_df is None:
                raise InvalidConfigError(
                    f"independent_columns {self.independent_columns} were declared, "
                    f"but no future_independent_df was provided.",
                    param_name="future_independent_df",
                    provided_value=None,
                )
            self._validate_columns_present(future_independent_df, self.independent_columns)
            for col in self.independent_columns:
                df[col] = future_independent_df[col].reset_index(drop=True)

        return self._add_features(df)

    def _add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        ds = df["ds"]

        # Trend: absolute offset from the fixed reference date established
        # at transform_train() time — deliberately NOT a row-position
        # counter, so the trend stays continuous and meaningful across the
        # train/future boundary (and robust to any gaps in the actual date
        # sequence, e.g. a non-contiguous forecast horizon).
        day_offset = (ds - self._reference_date).dt.days
        df["time_idx"] = day_offset if self.frequency == "daily" else day_offset / 7.0

        df["week_of_year"] = ds.dt.isocalendar().week.astype(int)
        df["month"] = ds.dt.month
        df["quarter"] = ds.dt.quarter
        df["year"] = ds.dt.year

        # Yearly/half-yearly cyclicals — valid for both frequencies.
        if self.frequency == "daily":
            cyclical_position = ds.dt.dayofyear
            yearly_period = _YEARLY_PERIOD_DAYS
        else:
            cyclical_position = ds.dt.isocalendar().week.astype(int)
            yearly_period = _YEARLY_PERIOD_WEEKS

        df["yearly_sin"] = np.sin(2 * np.pi * cyclical_position / yearly_period)
        df["yearly_cos"] = np.cos(2 * np.pi * cyclical_position / yearly_period)
        df["half_yearly_sin"] = np.sin(2 * np.pi * cyclical_position / (yearly_period / 2))
        df["half_yearly_cos"] = np.cos(2 * np.pi * cyclical_position / (yearly_period / 2))

        if self.frequency == "daily":
            df["day_of_week"] = ds.dt.dayofweek
            df["day_of_year"] = ds.dt.dayofyear
            df["weekly_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
            df["weekly_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
            df["monthly_sin"] = np.sin(2 * np.pi * ds.dt.day / 30.5)
            df["monthly_cos"] = np.cos(2 * np.pi * ds.dt.day / 30.5)

        return df

