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

import warnings
from typing import List, Optional

import numpy as np
import pandas as pd

from horizon_forecaster.data_adapter.base_adapter import BaseAdapter
from horizon_forecaster.exceptions import InvalidConfigError

# Days in a mean year, for building the annual seasonal basis used by the
# collinearity check. Matches SARIMAXWrapper's _ANNUAL_PERIOD_DAYS so the
# check measures overlap with the very Fourier terms the wrapper generates.
_ANNUAL_PERIOD_DAYS = 365.25

# An independent column is flagged as annually collinear when a sin/cos
# annual basis explains at least this fraction of its variance (R²). 0.30 is
# deliberately permissive: the warning is diagnostic, not an error, and the
# point is to surface a real overlap the user may not know about. Calibrated
# on real data where a stockout-weighting column (corr with the annual sine
# ~0.58, so R² >= ~0.33) collided with the annual Fourier grid dimension,
# while spend columns (annual R² well below 0.30) did not.
_ANNUAL_COLLINEARITY_R2_THRESHOLD = 0.30


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
            # Warn on annual collinearity BEFORE standardising (scale-free
            # anyway) so the user learns why a seasonal driver can make the
            # higher-Fourier grid candidates unstable. See the helper.
            self._warn_on_annual_collinearity(df["ds"], exog_ind)
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

    def _annual_collinearity_r2(self, ds: pd.Series, values: np.ndarray) -> float:
        """
        Fraction of a column's variance explained by an annual sin/cos basis
        (R² of an OLS fit on [1, sin(2πt/P), cos(2πt/P)], t in days). Phase-
        invariant: a seasonal driver is flagged whatever its peak week. Returns
        0.0 for degenerate input (constant column, <3 usable points).
        """
        v = np.asarray(values, dtype=float)
        finite = np.isfinite(v)
        if finite.sum() < 3:
            return 0.0
        v = v[finite]
        if np.std(v) < 1e-12:
            return 0.0

        ds = pd.to_datetime(pd.Series(ds).reset_index(drop=True))
        t_days = (ds - ds.min()).dt.total_seconds().to_numpy() / 86400.0
        t_days = t_days[finite]
        angle = 2.0 * np.pi * t_days / _ANNUAL_PERIOD_DAYS
        design = np.column_stack([np.ones_like(angle), np.sin(angle), np.cos(angle)])

        coef, *_ = np.linalg.lstsq(design, v, rcond=None)
        ss_res = float(np.sum((v - design @ coef) ** 2))
        ss_tot = float(np.sum((v - v.mean()) ** 2))
        if ss_tot < 1e-12:
            return 0.0
        return max(0.0, 1.0 - ss_res / ss_tot)

    def _warn_on_annual_collinearity(self, ds: pd.Series, exog_df: pd.DataFrame) -> None:
        """
        Flag independent columns that strongly track the annual cycle.

        Such a column explains much of the same variance as SARIMAX's annual
        Fourier terms, so on short histories the two compete: a good
        low-Fourier fit can collapse once the column is added, and the grid
        search is pushed toward a lower `fourier_order_annual`. Confirmed on
        real weekly data, where a stockout-weighting column (which is itself
        seasonal) cut the best achievable MAPE roughly in half versus the same
        run without it -- not a bug, an inherent identifiability limit on ~100
        rows. This surfaces the cause instead of leaving it buried in the grid
        summary. Diagnostic only: nothing is dropped or altered.
        """
        flagged = {
            col: r2
            for col in exog_df.columns
            if (r2 := self._annual_collinearity_r2(ds, exog_df[col].to_numpy()))
            >= _ANNUAL_COLLINEARITY_R2_THRESHOLD
        }
        if not flagged:
            return

        listed = ", ".join(
            f"{col} (annual R²={r2:.2f})"
            for col, r2 in sorted(flagged.items(), key=lambda kv: -kv[1])
        )
        warnings.warn(
            f"SARIMAX: independent column(s) strongly track the annual seasonal cycle: "
            f"{listed}. These explain much of the same variance as SARIMAX's annual Fourier "
            f"terms, so on short histories the two compete and the higher fourier_order_annual "
            f"grid candidates can become unstable (a good low-Fourier fit can collapse once "
            f"such a column is added). This is expected, not an error -- the grid search will "
            f"tend to pick a lower Fourier order when these columns are present. If a run "
            f"without them scores better, this overlap is the likely reason; the column may "
            f"still be causally real, it is just hard to separate from seasonality on this "
            f"much history.",
            stacklevel=4,
        )

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
