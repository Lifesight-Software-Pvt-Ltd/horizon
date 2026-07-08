"""
Prophet model wrapper.

Expects train_df already shaped by ProphetAdapter (columns: "ds", "y",
plus any independent/regressor columns under their original names).
"""

from typing import Optional

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

    # Prophet can technically fit on very few rows, but forecasts become
    # unreliable well before that floor. This is a conservative sanity
    # floor, not a statistically derived minimum — revisit once
    # config/default_params.yaml daily/weekly defaults are finalized.
    MIN_ROWS_REQUIRED = 10

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

        if len(train_df) < self.MIN_ROWS_REQUIRED:
            raise InsufficientDataError(
                f"Prophet requires at least {self.MIN_ROWS_REQUIRED} rows to fit reliably.",
                model_name=self.model_name,
                required_periods=self.MIN_ROWS_REQUIRED,
                available_periods=len(train_df),
            )

        regressor_columns = [c for c in train_df.columns if c not in ("ds", "y")]

        try:
            # Imported here (not at module level) so that importing this
            # module doesn't require Prophet's heavier dependency chain
            # (cmdstanpy) unless this specific wrapper is actually used.
            from prophet import Prophet

            # "random_seed" is a wrapper-level concept (controls the
            # reproducibility of predict()'s uncertainty simulation, see
            # predict() below) — it is NOT a real Prophet constructor
            # argument, so it must be filtered out before being passed
            # through, or Prophet(**self.params) would raise a TypeError.
            prophet_kwargs = {k: v for k, v in self.params.items() if k != "random_seed"}

            model = Prophet(**prophet_kwargs)
            for col in regressor_columns:
                model.add_regressor(col)
            model.fit(train_df)
        except Exception as e:
            raise ModelConvergenceError(
                "Prophet failed to fit.",
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

        missing_regressors = [
            c for c in self._regressor_columns if c not in future_exog.columns
        ]
        if missing_regressors:
            raise ModelPredictionError(
                f"future_exog is missing required regressor column(s): {missing_regressors}",
                model_name=self.model_name,
            )

        # Prophet's uncertainty interval simulation (yhat_lower/yhat_upper)
        # draws from numpy's global random state without a fixed seed by
        # default, meaning repeated predict() calls on the SAME fitted
        # model return slightly different bounds each time (confirmed via
        # direct testing). A fixed, configurable seed makes results
        # reproducible across repeated calls and across save/load cycles.
        # Default of 42 can be overridden via params={"random_seed": ...}.
        import numpy as np
        np.random.seed(self.params.get("random_seed", 42))

        try:
            forecast = self._fitted_model.predict(future_exog)
        except Exception as e:
            raise ModelPredictionError(
                "Prophet failed during prediction.",
                model_name=self.model_name,
                original_exception=e,
            )

        return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()

    def save(self, path: str) -> None:
        """
        Overrides BaseModel's generic pickle default. Prophet's own
        documentation warns against raw pickling of its fitted models,
        since the underlying Stan/cmdstanpy backend can fail to
        deserialize reliably across environments or library versions.
        This uses Prophet's own recommended model_to_json(), plus a
        small JSON envelope for the wrapper-level state (regressor
        columns, frequency, params) needed to fully reconstruct this
        wrapper, not just the underlying Prophet model object.
        """
        import json
        from prophet.serialize import model_to_json

        self._check_is_fitted()

        envelope = {
            "frequency": self.frequency,
            "params": self.params,
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

        instance = cls(frequency=envelope["frequency"], params=envelope["params"])
        instance._fitted_model = model_from_json(envelope["prophet_model_json"])
        instance._regressor_columns = envelope["regressor_columns"]
        instance.is_fitted = True
        return instance