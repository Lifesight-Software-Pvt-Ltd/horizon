"""
Data adapter for SARIMAX (statsmodels).

SARIMAX models the target directly as a (seasonal) ARIMA process with an
exogenous regression matrix, so this adapter's job is narrow: hand over
"ds", a raw "y", and a well-scaled exog matrix built from the user's
independent columns.

What it does NOT do, deliberately:

- No target transform. The log1p/expm1 round trip lives in
  SARIMAXWrapper, so "y" keeps one consistent meaning across every model
  and the wrapper's yhat comes back in raw units for MAPE.
- No seasonal features. Annual Fourier terms used to be generated here,
  but they moved into SARIMAXWrapper so the Fourier order becomes an
  ordinary hyperparameter the grid search can choose per dataset. Real
  data forced this: on a ~3-year daily series annual Fourier was a large
  win, while on a ~2-year weekly series it made things dramatically worse
  (and at high order it diverged outright). No fixed default is right for
  both, so the data has to decide -- which it can only do if the setting
  is reachable from the hyperparameter grid.

Statefulness: transform_train() establishes state that transform_future()
reuses -- the per-column log mask, and the standardisation mean/std/range
fitted on TRAIN only.
"""

from typing import List, Optional

import numpy as np
import pandas as pd

from horizon_forecaster.data_adapter.base_adapter import BaseAdapter
from horizon_forecaster.exceptions import InvalidConfigError


class SARIMAXAdapter(BaseAdapter):
    """
    Reshapes raw user data into SARIMAX's (endog "y", exog matrix) form:
    raw target plus standardised (and spend-like columns log1p'd)
    independent variables.

    Args:
        date_column, target_column, independent_columns: see BaseAdapter.
        frequency: "daily" or "weekly". Required for validation, though
            this adapter has no frequency-dependent feature logic left --
            seasonality is the wrapper's concern now.
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
        self._log_columns: Optional[List[str]] = None   # independents to log1p
        self._means: Optional[pd.Series] = None         # train mean per independent (post-log)
        self._stds: Optional[pd.Series] = None          # train std per independent (post-log)
        self._std_min: Optional[pd.Series] = None       # per-col min of standardised TRAIN exog
        self._std_max: Optional[pd.Series] = None       # per-col max of standardised TRAIN exog
        self._exog_columns: List[str] = []
        self._is_fit: bool = False

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

        if self.independent_columns:
            self._log_columns = self._detect_log_columns(df[self.independent_columns])
            exog_ind = self._apply_log(df[self.independent_columns])
            self._means = exog_ind.mean()
            self._stds = exog_ind.std(ddof=0).replace(0.0, 1.0)  # guard constant columns
            standardized = (exog_ind - self._means) / self._stds
            # Record the standardised training range per column. transform_future()
            # clamps future exog to this range so an out-of-range test value (e.g.
            # a spend spike far above anything seen in training) can't extrapolate
            # the fitted coefficient into a log-space blow-up. SARIMAX is linear in
            # the exog, so extrapolating a coefficient beyond the trained range is
            # unreliable anyway -- confirmed on real data, where a test-window spend
            # spike made the forecast diverge until this clamp went in.
            self._std_min = standardized.min()
            self._std_max = standardized.max()
            for col in self.independent_columns:
                df[col] = standardized[col].to_numpy()

        self._exog_columns = list(self.independent_columns)
        self._is_fit = True
        return df

    def transform_future(
        self, future_dates: pd.Series, future_independent_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        if not self._is_fit:
            raise InvalidConfigError(
                "SARIMAXAdapter.transform_future() called before transform_train(); "
                "the standardisation stats are established there first.",
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
            future_ind = future_independent_df[self.independent_columns].reset_index(drop=True)
            future_ind = future_ind.apply(pd.to_numeric, errors="coerce")
            standardized = (self._apply_log(future_ind) - self._means) / self._stds
            # Clamp to the standardised training range (see transform_train): no
            # exog extrapolation beyond what the model was trained on.
            standardized = standardized.clip(lower=self._std_min, upper=self._std_max, axis=1)
            for col in self.independent_columns:
                df[col] = standardized[col].to_numpy()

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
