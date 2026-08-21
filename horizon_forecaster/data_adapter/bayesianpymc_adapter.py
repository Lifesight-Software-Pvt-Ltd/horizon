"""
Data adapter for the Bayesian PyMC model.

Like SARIMAX, the PyMC wrapper models the raw target directly with a
regression component built from the user's independent variables, so this
adapter's job is narrow: hand over "ds", a raw "y", and a well-scaled exog
matrix. Everything Bayesian -- the target standardisation, the Fourier
seasonal basis, the priors and the sampler -- lives in
BayesianPyMCWrapper.

What it does NOT do, deliberately:

- No target transform. The PyMC formulation standardises "y" internally
  (posterior is sampled in z-space, back-transformed to raw units in
  predict()), so "y" keeps one consistent raw meaning across every model
  and the wrapper's yhat comes back in raw units for MAPE. Mirrors how the
  SARIMAX log1p/expm1 round trip lives in its wrapper, not its adapter.
- No seasonal features. The annual (and, for daily data, weekly) Fourier
  basis is generated in the wrapper from absolute date offsets, exactly as
  SARIMAX does -- so the Fourier order is an ordinary hyperparameter the
  grid search can vary per dataset, and a gap in the date sequence cannot
  shift the phase (a positional index would).

Statefulness: transform_train() establishes state that transform_future()
reuses -- the standardisation mean/std fitted on TRAIN only, plus the
per-column standardised training range used to clamp future exog.

Unlike SARIMAXAdapter, this adapter does NOT log1p spend-like columns: the
reference PyMC notebook standardised raw spend directly, and the model's
priors regularise the coefficients, so the extra transform isn't needed
here. The wrapper's own spend-like detection (for choosing an informative
vs. a zero-centred coefficient prior) keys off the number of distinct
values in the standardised column it receives, so no metadata has to cross
the adapter/wrapper boundary.
"""

from typing import List, Optional

import pandas as pd

from horizon_forecaster.data_adapter.base_adapter import BaseAdapter
from horizon_forecaster.exceptions import InvalidConfigError


class BayesianPyMCAdapter(BaseAdapter):
    """
    Reshapes raw user data into the PyMC wrapper's (raw "y", standardised
    exog) form: the raw target plus standardised independent variables.

    Args:
        date_column, target_column, independent_columns: see BaseAdapter.
        frequency: "daily" or "weekly". Required for validation, though
            this adapter has no frequency-dependent feature logic --
            seasonality is the wrapper's concern.
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
                f"BayesianPyMCAdapter requires frequency='daily' or 'weekly', got {frequency!r}.",
                param_name="frequency",
                provided_value=frequency,
            )

        # State established by transform_train(), reused by transform_future().
        self._means: Optional[pd.Series] = None    # train mean per independent
        self._stds: Optional[pd.Series] = None      # train std per independent
        self._std_min: Optional[pd.Series] = None   # per-col min of standardised TRAIN exog
        self._std_max: Optional[pd.Series] = None   # per-col max of standardised TRAIN exog
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
            exog_ind = df[self.independent_columns]
            self._means = exog_ind.mean()
            self._stds = exog_ind.std(ddof=0).replace(0.0, 1.0)  # guard constant columns
            standardized = (exog_ind - self._means) / self._stds
            # Record the standardised training range per column. transform_future()
            # clamps future exog to this range so an out-of-range test value (e.g. a
            # spend spike far above anything seen in training) can't extrapolate a
            # fitted coefficient into an implausible forecast. The PyMC regression is
            # linear in the exog, so extrapolating a coefficient beyond the trained
            # range is unreliable regardless of the prior -- the same guard SARIMAX
            # adopted after real data showed a test-window spend spike diverging.
            self._std_min = standardized.min()
            self._std_max = standardized.max()
            for col in self.independent_columns:
                df[col] = standardized[col].to_numpy()

        self._is_fit = True
        return df

    def transform_future(
        self, future_dates: pd.Series, future_independent_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        if not self._is_fit:
            raise InvalidConfigError(
                "BayesianPyMCAdapter.transform_future() called before transform_train(); "
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
            standardized = (future_ind - self._means) / self._stds
            # Clamp to the standardised training range (see transform_train): no exog
            # extrapolation beyond what the model was trained on.
            standardized = standardized.clip(lower=self._std_min, upper=self._std_max, axis=1)
            for col in self.independent_columns:
                df[col] = standardized[col].to_numpy()

        return df

    # ---------- helpers ----------

