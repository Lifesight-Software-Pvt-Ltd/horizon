"""
Data adapter for CatBoost.

Unlike Prophet, CatBoost has no native concept of dates, trend, or
seasonality — it just sees a table of numeric/categorical features and a
target. All of the time-series structure Prophet gets "for free" has to
be engineered here as explicit columns: calendar features (day-of-week,
month, etc.) and lag/rolling-window features derived from the target's
own history. This is precisely why CatBoost was chosen as the Phase 3
model: it forces base_adapter.py's contract to prove itself against a
genuinely different data shape, not just a second wrapper around the
same ds/y idea.
"""

from typing import List, Optional, Sequence

import pandas as pd

from horizon_forecaster.data_adapter.base_adapter import BaseAdapter
from horizon_forecaster.exceptions import InvalidConfigError

# Generic, frequency-agnostic defaults. Frequency-aware overrides (e.g.
# shorter lag sets for weekly data, where a "lag_14" is 14 WEEKS back,
# not days) belong in config/default_params.yaml's catboost block, added
# in a later Phase 3 step — this adapter itself has no notion of
# "daily" vs "weekly" and just operates on whatever lags/windows it's
# constructed with, exactly like ProphetAdapter has no frequency
# awareness of its own.
DEFAULT_LAGS: Sequence[int] = (1, 2, 3, 7, 14)
DEFAULT_ROLLING_WINDOWS: Sequence[int] = (7, 14)

class CatBoostAdapter(BaseAdapter):
    """
    Reshapes raw user data into a flat feature table CatBoost can train
    and predict on: calendar features + independent columns (passed
    through as-is) + lag features + rolling-window statistics of the
    target.

    Design decision: this adapter is STATEFUL, unlike ProphetAdapter.
    transform_train() retains the historical target series
    (`self._history`) specifically so transform_future() can compute lag
    and rolling features that reach back into real training data rather
    than starting from nothing. This is a deliberate, documented
    deviation from BaseAdapter's implicit stateless-pure-function
    pattern (established by ProphetAdapter in Phase 2) — CatBoost is the
    first model in this project whose future-feature construction
    genuinely requires historical target values, not just historical
    independent variables. Calling transform_future() before
    transform_train() raises InvalidConfigError rather than silently
    operating on no history.

    Design decision: lag/rolling features for the future horizon are
    computed once, "frozen" as of the end of the training history, and
    reused identically for every row in the forecast horizon (only the
    calendar features and independent columns vary per future row).
    This is the standard "direct" multi-horizon strategy (as opposed to
    "recursive", which would feed each step's prediction back in as the
    next step's lag input). Direct was chosen over recursive for v1 to
    avoid compounding-error risk and to keep CatBoostWrapper.predict()
    a single batched call, consistent with every other wrapper's
    predict() signature in this project (one call, not a per-step
    loop). This mirrors the same reasoning already applied to Prophet's
    default future-independent-variable strategy in Phase 2 (reuse
    known history rather than inventing unknown future values) and is
    documented here as a revisitable v1 choice, not a permanent
    architectural decision.

    Args:
        date_column, target_column, independent_columns: see BaseAdapter.
        lags: which lags (in periods) of the target to include as
            features, e.g. (1, 7) means "target 1 period ago" and
            "target 7 periods ago".
        rolling_windows: window sizes (in periods) for rolling
            mean/std features of the target, computed using only prior
            periods (shift(1) before rolling — the current period's own
            target is never included in its own rolling statistics).
    """

    _CALENDAR_FEATURE_NAMES = (
        "day_of_week",
        "day_of_month",
        "day_of_year",
        "week_of_year",
        "month",
        "quarter",
        "is_month_start",
        "is_month_end",
        "is_weekend",
    )

    def __init__(
        self,
        date_column: str,
        target_column: str,
        independent_columns: Optional[List[str]] = None,
        lags: Sequence[int] = DEFAULT_LAGS,
        rolling_windows: Sequence[int] = DEFAULT_ROLLING_WINDOWS,
    ):
        super().__init__(date_column, target_column, independent_columns)
        self.lags = list(lags)
        self.rolling_windows = list(rolling_windows)
        self._engineered_columns = [f"lag_{lag}" for lag in self.lags] + [
            f"rolling_mean_{w}" for w in self.rolling_windows
        ] + [f"rolling_std_{w}" for w in self.rolling_windows]

        # Set by transform_train(); required by transform_future(). Holds
        # the full ("ds", "y") training history, in chronological order,
        # BEFORE any NaN-lag rows are dropped — transform_future() needs
        # the true tail of history, not a truncated one.
        self._history: Optional[pd.DataFrame] = None

    def transform_train(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        required_columns = [self.date_column, self.target_column] + self.independent_columns
        self._validate_columns_present(raw_df, required_columns)

        df = pd.DataFrame()
        df["ds"] = pd.to_datetime(raw_df[self.date_column])
        df["y"] = raw_df[self.target_column]
        for col in self.independent_columns:
            df[col] = raw_df[col]

        df = df.sort_values("ds").reset_index(drop=True)

        # Retain full pre-drop history for transform_future()'s use,
        # before this instance mutates/truncates anything further.
        self._history = df[["ds", "y"]].copy()

        df = self._add_calendar_features(df, date_col="ds")
        df = self._add_lag_and_rolling_features(df, y=df["y"])

        # Rows at the start of the series don't have enough history to
        # compute every lag/rolling feature (e.g. lag_14 is NaN for the
        # first 14 rows). Dropped explicitly and deliberately, rather
        # than left as NaN or silently imputed — CatBoost can technically
        # handle NaN natively, but a lag feature that's NaN because there
        # simply is no history yet is a structurally different kind of
        # "missing" than a genuinely missing real-world value, and
        # conflating the two risks teaching the model a spurious pattern
        # at the start of every series it ever sees.
        before_drop = len(df)
        df = df.dropna(subset=self._engineered_columns).reset_index(drop=True)
        self.n_rows_dropped_for_lag_warmup = before_drop - len(df)

        return df

    def transform_future(
        self, future_dates: pd.Series, future_independent_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        if self._history is None:
            raise InvalidConfigError(
                "CatBoostAdapter.transform_future() called before transform_train(). "
                "Lag/rolling features need real target history, which is only "
                "available after transform_train() has run at least once on this "
                "adapter instance.",
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

        df = self._add_calendar_features(df, date_col="ds")

        # Frozen lag/rolling features (see class docstring): computed
        # once from the tail of the retained training history, then
        # broadcast identically across every row of the forecast
        # horizon.
        frozen_features = self._compute_frozen_lag_and_rolling_features()
        for col, value in frozen_features.items():
            df[col] = value

        return df

    def _add_calendar_features(self, df: pd.DataFrame, date_col: str) -> pd.DataFrame:
        ds = df[date_col]
        df["day_of_week"] = ds.dt.dayofweek
        df["day_of_month"] = ds.dt.day
        df["day_of_year"] = ds.dt.dayofyear
        df["week_of_year"] = ds.dt.isocalendar().week.astype(int)
        df["month"] = ds.dt.month
        df["quarter"] = ds.dt.quarter
        df["is_month_start"] = ds.dt.is_month_start.astype(int)
        df["is_month_end"] = ds.dt.is_month_end.astype(int)
        df["is_weekend"] = (ds.dt.dayofweek >= 5).astype(int)
        return df

    def _add_lag_and_rolling_features(self, df: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        for lag in self.lags:
            df[f"lag_{lag}"] = y.shift(lag)

        # shift(1) before rolling: a period's rolling stats only ever
        # summarize STRICTLY PRIOR periods, never (even partially) its
        # own target value — otherwise this would leak the label into
        # its own features.
        shifted = y.shift(1)
        for window in self.rolling_windows:
            df[f"rolling_mean_{window}"] = shifted.rolling(window).mean()
            df[f"rolling_std_{window}"] = shifted.rolling(window).std()

        return df

    def _compute_frozen_lag_and_rolling_features(self) -> dict:
        """
        Computes each lag/rolling feature exactly once, anchored to the
        end of the retained training history — i.e. "as of the last
        known actual value" — for reuse across every row of the future
        horizon. See class docstring: this is the "direct" (not
        recursive) multi-horizon strategy.
        """
        y = self._history["y"]
        features = {}

        for lag in self.lags:
            if lag > len(y):
                raise InvalidConfigError(
                    f"Cannot compute lag_{lag}: only {len(y)} periods of history "
                    f"are available. Reduce this adapter's `lags` configuration or "
                    f"provide more training history.",
                    param_name="lags",
                    provided_value=lag,
                )
            features[f"lag_{lag}"] = y.iloc[-lag]

        for window in self.rolling_windows:
            trailing = y.iloc[-window:]
            if len(trailing) < window:
                raise InvalidConfigError(
                    f"Cannot compute a {window}-period rolling feature: only "
                    f"{len(trailing)} periods of history are available. Reduce "
                    f"this adapter's `rolling_windows` configuration or provide "
                    f"more training history.",
                    param_name="rolling_windows",
                    provided_value=window,
                )
            features[f"rolling_mean_{window}"] = trailing.mean()
            features[f"rolling_std_{window}"] = trailing.std()

        return features

    @staticmethod
    def _validate_columns_present(df: pd.DataFrame, required_columns: list) -> None:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise InvalidConfigError(
                f"Missing required column(s) in input data: {missing}",
                param_name="raw_df columns",
                provided_value=list(df.columns),
            )
