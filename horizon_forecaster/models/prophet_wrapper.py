"""
Prophet model wrapper.

Expects train_df already shaped by ProphetAdapter (columns: "ds", "y",
plus any independent/regressor columns under their original names).
"""

from typing import Any, Dict, Optional

import pandas as pd

from horizon_forecaster.exceptions import (
    InsufficientDataError,
    ModelConvergenceError,
    ModelPredictionError,
)
from horizon_forecaster.models.base import BaseModel


class ProphetWrapper(BaseModel):
    """
    Wraps Facebook/Meta Prophet to conform to the BaseModel contract.
    """

    model_name = "prophet"

    # Frequency-aware fallback floors, used only if min_rows_required is
    # not explicitly provided at construction time. These mirror
    # config/default_params.yaml's wrapper_config values, kept here as a
    # safety default so this wrapper still behaves sensibly if used
    # standalone (e.g. in tests) without going through the config loader.
    #
    # Daily: 14 days = 2 full weeks, the earliest point a weekly pattern
    #   can even theoretically repeat once.
    # Weekly: 8 weeks (~2 months), below which there isn't enough span to
    #   estimate a trend + any real pattern, as opposed to fitting noise.
    _FREQUENCY_MIN_ROWS_DEFAULTS = {"daily": 14, "weekly": 8}
    _FALLBACK_MIN_ROWS = 10  # used only for an unrecognized frequency string

    def __init__(
        self,
        frequency: str,
        params: Optional[Dict[str, Any]] = None,
        min_rows_required: Optional[int] = None,
        random_seed: int = 42,
    ):
        """
        Args:
            frequency: "daily" or "weekly".
            params: hyperparameters passed directly to Prophet's
                constructor (e.g. yearly_seasonality, seasonality_mode).
                Should NOT contain wrapper-level settings like
                min_rows_required or random_seed — those are explicit
                constructor arguments precisely so they never need to be
                filtered out of this dict before being passed to
                Prophet(**params).
            min_rows_required: minimum training rows required to fit.
                If None, falls back to a frequency-aware default.
            random_seed: seed for Prophet's uncertainty interval
                simulation, which otherwise draws from numpy's global
                random state and produces non-reproducible yhat_lower/
                yhat_upper across repeated predict() calls (confirmed via
                direct testing during development).
        """
        super().__init__(frequency, params)
        if min_rows_required is None:
            min_rows_required = self._FREQUENCY_MIN_ROWS_DEFAULTS.get(
                frequency, self._FALLBACK_MIN_ROWS
            )
        self.min_rows_required = min_rows_required
        self.random_seed = random_seed

    def fit(self, train_df: pd.DataFrame, target_column: str = "y") -> "ProphetWrapper":
        """
        Fit Prophet on already-adapted data.

        Note: `target_column` is accepted for interface consistency with
        BaseModel, but Prophet's adapter always produces a "y" column, so
        this wrapper does not use the argument to rename anything — it
        simply validates that "y" is present.
        """
        if "ds" not in train_df.columns or "y" not in train_df.columns:
            raise ModelPredictionError(
                "ProphetWrapper.fit() expects train_df with 'ds' and 'y' "
                "columns (i.e. data already shaped by ProphetAdapter).",
                model_name=self.model_name,
            )

        if len(train_df) < self.min_rows_required:
            raise InsufficientDataError(
                f"Prophet requires at least {self.min_rows_required} rows to fit reliably.",
                model_name=self.model_name,
                required_periods=self.min_rows_required,
                available_periods=len(train_df),
            )

        regressor_columns = [c for c in train_df.columns if c not in ("ds", "y")]

        try:
            # Imported here (not at module level) so that importing this
            # module doesn't require Prophet's heavier dependency chain
            # (cmdstanpy) unless this specific wrapper is actually used.
            from prophet import Prophet

            model = Prophet(**self.params)
            for col in regressor_columns:
                model.add_regressor(col)
            model.fit(train_df)
        except Exception as e:
            raise ModelConvergenceError(
                f"Prophet failed to fit ({type(e).__name__}: {e}).",
                model_name=self.model_name,
                original_exception=e,
            )

        self._fitted_model = model
        self._regressor_columns = regressor_columns
        self.is_fitted = True
        return self

    def predict(self, horizon: int, future_exog: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Predict using an already-adapted future_exog DataFrame (from
        ProphetAdapter.transform_future()), containing "ds" and any
        required regressor columns.

        Unlike BaseModel's generic Optional[future_exog], Prophet always
        requires a future dataframe to predict against — a None here is
        treated as a caller error (forecaster.py failed to build it),
        not a valid "predict with no context" request.

        Returns ["ds", "yhat", "yhat_lower", "yhat_upper"], or just
        ["ds", "yhat"] when params set uncertainty_samples=0, as
        HorizonForecaster's config does.
        """
        self._check_is_fitted()

        if future_exog is None:
            raise ModelPredictionError(
                "Prophet requires future_exog (a DataFrame with 'ds' and any "
                "required regressor columns) to predict — received None.",
                model_name=self.model_name,
            )

        if "ds" not in future_exog.columns:
            raise ModelPredictionError(
                "future_exog must contain a 'ds' column.",
                model_name=self.model_name,
            )

        if len(future_exog) != horizon:
            raise ModelPredictionError(
                f"future_exog has {len(future_exog)} rows but horizon={horizon} was requested.",
                model_name=self.model_name,
            )

        missing_regressors = [c for c in self._regressor_columns if c not in future_exog.columns]
        if missing_regressors:
            raise ModelPredictionError(
                f"future_exog is missing required regressor column(s): {missing_regressors}",
                model_name=self.model_name,
            )

        # Prophet's uncertainty interval simulation (yhat_lower/yhat_upper)
        # draws from numpy's global random state without a fixed seed by
        # default, meaning repeated predict() calls on the SAME fitted
        # model return slightly different bounds each time (confirmed via
        # direct testing). self.random_seed (explicit constructor arg,
        # default 42) makes results reproducible across repeated calls
        # and across save/load cycles.
        import numpy as np

        np.random.seed(self.random_seed)

        try:
            forecast = self._fitted_model.predict(future_exog)
        except Exception as e:
            raise ModelPredictionError(
                f"Prophet failed during prediction ({type(e).__name__}: {e}).",
                model_name=self.model_name,
                original_exception=e,
            )

        # With uncertainty_samples=0 (HorizonForecaster's config, which
        # replaces every model's interval with a calibrated band) Prophet
        # skips its interval simulation and returns no bound columns.
        bounds = [c for c in ("yhat_lower", "yhat_upper") if c in forecast.columns]
        return forecast[["ds", "yhat", *bounds]].copy()

    def save(self, path: str) -> None:
        """
        Overrides BaseModel's generic pickle default. Prophet's own
        documentation warns against raw pickling of its fitted models,
        since the underlying Stan/cmdstanpy backend can fail to
        deserialize reliably across environments or library versions.
        This uses Prophet's own recommended model_to_json(), plus a
        small JSON envelope for the wrapper-level state (regressor
        columns, frequency, params, min_rows_required, random_seed)
        needed to fully reconstruct this wrapper, not just the
        underlying Prophet model object.
        """
        import json
        from prophet.serialize import model_to_json

        self._check_is_fitted()

        envelope = {
            "frequency": self.frequency,
            "params": self.params,
            "min_rows_required": self.min_rows_required,
            "random_seed": self.random_seed,
            "regressor_columns": self._regressor_columns,
            "prophet_model_json": model_to_json(self._fitted_model),
        }
        with open(path, "w") as f:
            json.dump(envelope, f)

    @classmethod
    def load(cls, path: str) -> "ProphetWrapper":
        """
        Reload a ProphetWrapper previously saved with save(), using
        Prophet's own model_from_json() rather than generic unpickling.
        """
        import json
        from prophet.serialize import model_from_json

        with open(path, "r") as f:
            envelope = json.load(f)

        instance = cls(
            frequency=envelope["frequency"],
            params=envelope["params"],
            min_rows_required=envelope["min_rows_required"],
            random_seed=envelope["random_seed"],
        )
        instance._fitted_model = model_from_json(envelope["prophet_model_json"])
        instance._regressor_columns = envelope["regressor_columns"]
        instance.is_fitted = True
        return instance
