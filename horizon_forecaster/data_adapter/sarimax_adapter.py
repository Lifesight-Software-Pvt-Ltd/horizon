"""
Data adapter for SARIMAX (statsmodels).

SARIMAX models the target directly as a (seasonal) ARIMA process with an
exogenous regression matrix. Unlike CatBoost, it does NOT want engineered
calendar/trend features -- differencing and the AR/MA/seasonal terms carry
the trend and short-season structure themselves. So this adapter's job is
narrower and specific to how SARIMAX behaves best in practice:

1. Pass the target through as raw "y". The log transform that stabilises
   the variance of a strictly-positive metric is done inside
   SARIMAXWrapper (so "y" keeps its raw meaning across every model, and
   the wrapper's yhat comes back in raw units for MAPE).

2. Build the exogenous matrix:
   - Independent columns are standardised (SARIMAX is sensitive to exog
     scale). A column that looks spend-like -- non-negative and
     non-binary -- is log1p'd first (matching the project owner's proven
     notebook approach); 0/1 flags are left as-is. Detection is automatic
     because the library's `independent_columns` is a flat list with no
     spend/flag metadata.
   - Annual seasonality is added as Fourier terms (a DeterministicProcess),
     for BOTH frequencies. This is the only place yearly seasonality comes
     from for weekly data (a real SARIMAX seasonal_order of s=52 is
     prohibitively slow/unstable), and it complements the cheap weekly
     s=7 seasonal_order the wrapper uses for daily data.

Statefulness: like CatBoostAdapter, transform_train() establishes state
that transform_future() reuses -- here the per-column log mask, the
standardisation mean/std fitted on TRAIN only, and the DeterministicProcess
(whose out_of_sample() continues the Fourier phase seamlessly into the
forecast horizon). transform_future() therefore assumes the future rows
are a contiguous continuation of training (which they are in
forecaster.py: the test split and the forecast horizon both immediately
follow the training window).

Standardisation is implemented manually (store train mean/std) rather than
via scikit-learn, to avoid adding a dependency the project doesn't already
have.
"""

from typing import List, Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.deterministic import DeterministicProcess, Fourier

from horizon_forecaster.data_adapter.base_adapter import BaseAdapter
from horizon_forecaster.exceptions import InvalidConfigError

_YEARLY_PERIOD_DAYS = 365.25
_YEARLY_PERIOD_WEEKS = 52.0
_FOURIER_ORDER_ANNUAL = 2  # 2 sin/cos pairs -> broad annual shape; needs ~1yr+ of data

# Annual Fourier terms are only added when training history covers at least
# this fraction of the annual period. Below it, the terms span less than half
# a cycle and become unconstrained trend-like regressors that EXTRAPOLATE
# wildly out of sample -- the same "don't force yearly seasonality on short
# history" failure mode Prophet's 'auto' candidates guard against (confirmed
# here: on a ~2.5-month daily series, including annual Fourier drove test MAPE
# from ~6% to ~65%). On shorter data we simply omit them and rely on the
# ARIMA structure (+ the weekly s=7 seasonal order, for daily).
_MIN_ANNUAL_COVERAGE = 0.5


class SARIMAXAdapter(BaseAdapter):
    """
    Reshapes raw user data into SARIMAX's (endog "y", exog matrix) form:
    raw target, standardised (and spend-columns log1p'd) independent
    variables, plus annual Fourier seasonal terms.

    Args:
        date_column, target_column, independent_columns: see BaseAdapter.
        frequency: "daily" or "weekly" -- selects the annual Fourier
            period (365.25 days vs 52 weeks). Required.
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
                f"SARIMAXAdapter requires frequency='daily' or 'weekly', got {frequency!r}.",
                param_name="frequency",
                provided_value=frequency,
            )

        # State established by transform_train(), reused by transform_future().
        self._log_columns: Optional[List[str]] = None       # independents to log1p
        self._means: Optional[pd.Series] = None             # train mean per independent (post-log)
        self._stds: Optional[pd.Series] = None              # train std per independent (post-log)
        self._dp: Optional[DeterministicProcess] = None     # annual Fourier process (None if omitted)
        self._fourier_columns: List[str] = []
        self._exog_columns: List[str] = []                  # final exog order (independents + fourier)
        self._is_fit: bool = False                          # set by transform_train()

    # ---------- public API ----------

    def transform_train(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        required_columns = [self.date_column, self.target_column] + self.independent_columns
        self._validate_columns_present(raw_df, required_columns)

        df = pd.DataFrame()
        df["ds"] = pd.to_datetime(raw_df[self.date_column])
        df["y"] = pd.to_numeric(raw_df[self.target_column], errors="coerce")
        for col in self.independent_columns:
            df[col] = pd.to_numeric(raw_df[col], errors="coerce")
        df = df.sort_values("ds").reset_index(drop=True)

        # Independent-variable exog: auto-log then standardise (fit on train).
        if self.independent_columns:
            self._log_columns = self._detect_log_columns(df[self.independent_columns])
            exog_ind = self._apply_log(df[self.independent_columns])
            self._means = exog_ind.mean()
            self._stds = exog_ind.std(ddof=0).replace(0.0, 1.0)  # guard constant columns
            standardized = (exog_ind - self._means) / self._stds
            for col in self.independent_columns:
                df[col] = standardized[col].to_numpy()

        # Annual Fourier seasonal exog -- only when there is enough history to
        # constrain it (see _MIN_ANNUAL_COVERAGE). Otherwise omitted entirely.
        period = _YEARLY_PERIOD_DAYS if self.frequency == "daily" else _YEARLY_PERIOD_WEEKS
        if len(df) >= _MIN_ANNUAL_COVERAGE * period:
            fourier = Fourier(period=period, order=_FOURIER_ORDER_ANNUAL)
            self._dp = DeterministicProcess(
                index=pd.RangeIndex(len(df)), additional_terms=[fourier], drop=True
            )
            fourier_train = self._dp.in_sample()
            self._fourier_columns = [f"fourier_annual_{i}" for i in range(fourier_train.shape[1])]
            for name, values in zip(self._fourier_columns, fourier_train.to_numpy().T):
                df[name] = values
        else:
            self._dp = None
            self._fourier_columns = []

        self._is_fit = True
        self._exog_columns = list(self.independent_columns) + self._fourier_columns
        return df

    def transform_future(
        self, future_dates: pd.Series, future_independent_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        if not self._is_fit:
            raise InvalidConfigError(
                "SARIMAXAdapter.transform_future() called before transform_train(); "
                "the standardisation stats and Fourier process are established there first.",
                param_name="transform_future",
                provided_value=None,
            )

        df = pd.DataFrame()
        df["ds"] = pd.to_datetime(future_dates).reset_index(drop=True)
        n = len(df)

        if self.independent_columns:
            if future_independent_df is None:
                raise InvalidConfigError(
                    f"independent_columns {self.independent_columns} were declared, "
                    f"but no future_independent_df was provided.",
                    param_name="future_independent_df",
                    provided_value=None,
                )
            self._validate_columns_present(future_independent_df, self.independent_columns)
            future_ind = future_independent_df[self.independent_columns].reset_index(drop=True)
            future_ind = future_ind.apply(pd.to_numeric, errors="coerce")
            standardized = (self._apply_log(future_ind) - self._means) / self._stds
            for col in self.independent_columns:
                df[col] = standardized[col].to_numpy()

        if self._dp is not None:
            fourier_future = self._dp.out_of_sample(steps=n)
            for name, values in zip(self._fourier_columns, fourier_future.to_numpy().T):
                df[name] = values

        return df

    # ---------- helpers ----------

    def _detect_log_columns(self, ind_df: pd.DataFrame) -> List[str]:
        """
        A column is log1p'd iff it looks spend-like: non-negative and
        non-binary (more than 2 distinct non-null values). This keeps 0/1
        flags linear while log-compressing spend/impression-style columns,
        without needing per-column metadata the library doesn't carry.
        """
        log_cols = []
        for col in ind_df.columns:
            series = ind_df[col].dropna()
            if len(series) == 0:
                continue
            if series.min() >= 0 and series.nunique() > 2:
                log_cols.append(col)
        return log_cols

    def _apply_log(self, ind_df: pd.DataFrame) -> pd.DataFrame:
        out = ind_df.copy()
        for col in (self._log_columns or []):
            if col in out.columns:
                out[col] = np.log1p(out[col])
        return out

    @staticmethod
    def _validate_columns_present(df: pd.DataFrame, required_columns: list) -> None:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise InvalidConfigError(
                f"Missing required column(s) in input data: {missing}",
                param_name="raw_df columns",
                provided_value=list(df.columns),
            )
