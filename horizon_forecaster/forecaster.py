"""
HorizonForecaster: the main orchestration class for horizon_forecaster.

This is the first place that wires together data adapters, model
wrappers, and config loading — everything built so far (Phase 1's
contracts, Phase 2's Prophet adapter/wrapper/config) was built precisely
so this file could stay thin, since all the hard decisions were already
made and tested in isolation before reaching here.

Current scope (Phase 2, steps 5-6): single model (Prophet), evaluate()
and forecast(). No ensemble/multi-model selection yet — that's Phase 4.
"""

import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import yaml

from horizon_forecaster.data_adapter.prophet_adapter import ProphetAdapter
from horizon_forecaster.evaluation.metrics import MapeResult, mape
from horizon_forecaster.exceptions import InvalidConfigError, UnsupportedFrequencyError
from horizon_forecaster.models.base import BaseModel
from horizon_forecaster.models.prophet_wrapper import ProphetWrapper
from horizon_forecaster.utils import (
    clean_zero_variance,
    infer_frequency,
    validate_frequency,
    validate_training_size,
)

_CONFIG_PATH = Path(__file__).parent / "config" / "default_params.yaml"
_FREQUENCY_OFFSETS = {"daily": "D", "weekly": "7D"}

_HORIZON_WARNING_MULTIPLE = 2

@dataclass
class EvaluationResult:
    """
    Structured result from HorizonForecaster.evaluate().

    Attributes:
        model_name: which model this result is for (e.g. "prophet").
        mape: MapeResult (value + zero-actual exclusion metadata), not a
            bare float.
        n_train: number of rows used for training.
        n_test: number of rows held out for evaluation.
        test_predictions: DataFrame with columns ["ds", "y", "yhat"]
            (plus "yhat_lower"/"yhat_upper" if available).
        fitted_model: the wrapper instance fitted on the TRAIN SPLIT ONLY
            (using the winning hyperparameters, if grid search was used).
        grid_search_summary: if hyperparameters were selected via grid
            search (no explicit model_params override given), a list of
            every candidate tried, each as {"params": {...}, "mape": float},
            sorted best-to-worst. None if an explicit override was used.
    """

    model_name: str
    mape: MapeResult
    n_train: int
    n_test: int
    test_predictions: pd.DataFrame
    fitted_model: BaseModel
    grid_search_summary: Optional[List[dict]] = None

@dataclass
class ForecastResult:
    """
    Structured result from HorizonForecaster.forecast().

    Attributes:
        model_name: which model produced this forecast.
        forecast: DataFrame with columns ["ds", "yhat"] (plus
            "yhat_lower"/"yhat_upper" if available).
        horizon: the number of periods forecasted.
        used_default_future_strategy: True if future independent-variable
            values were auto-generated rather than caller-supplied.
    """

    model_name: str
    forecast: pd.DataFrame
    horizon: int
    used_default_future_strategy: bool

class HorizonForecaster:
    """
    Main orchestrator. Takes raw user data once at construction, and
    exposes evaluate() (train/test split + MAPE) as a first step, with a
    separate forecast() step (full-data refit + real future prediction)
    to follow.

    Args:
        raw_df: user's raw DataFrame containing at least date_column and
            target_column, plus any of independent_columns that apply.
        date_column: name of the date column in raw_df.
        target_column: name of the target/dependent variable column.
        frequency: "daily" or "weekly".
        training_size: percentage (50-100) or fraction (0.5-1.0) of data
            to use for training; the remainder is held out for
            evaluation. Defaults to 80 (an 80/20 split).
        independent_columns: names of independent variable columns
            (spend, impressions, promos, etc.) to use, if any.
        model_params: optional dict of Prophet hyperparameter overrides.
            If given, this SKIPS the default hyperparameter grid search
            entirely and uses exactly these values (merged on top of
            this frequency's base_hyperparameters). If omitted (the
            default), evaluate() automatically tries every candidate in
            config/default_params.yaml's hyperparameter_grid on the same
            train/test split and selects whichever achieves the lowest
            MAPE — see EvaluationResult.grid_search_summary for the full
            comparison.
    """

    def __init__(
        self,
        raw_df: pd.DataFrame,
        date_column: str,
        target_column: str,
        frequency: str,
        training_size: float = 80,
        independent_columns: Optional[List[str]] = None,
        model_params: Optional[dict] = None,
    ):
        self.frequency = validate_frequency(frequency)
        self.training_fraction = validate_training_size(training_size)
        self.date_column = date_column
        self.target_column = target_column

        requested_independent_columns = independent_columns or []
        cleaned_df, dropped_columns = clean_zero_variance(
            raw_df, exclude_columns=[date_column, target_column]
        )
        self.dropped_zero_variance_columns = dropped_columns

        self.independent_columns = [
            c for c in requested_independent_columns if c not in dropped_columns
        ]

        self.raw_df = cleaned_df.sort_values(date_column).reset_index(drop=True)
        self._warn_if_declared_frequency_mismatches_data()

        self._config = self._load_config()
        base_hyperparameters = self._config.get("base_hyperparameters", {})

        # Explicit override -> skip grid search entirely, exactly one
        # fixed hyperparameter set, resolved immediately.
        # No override -> grid search mode; hyperparameters are resolved
        # LAZILY (on the first evaluate() or forecast() call, whichever
        # comes first), since selecting them requires actually fitting
        # every candidate against the train/test split.
        self._use_grid_search = model_params is None and "hyperparameter_grid" in self._config
        if self._use_grid_search:
            self._selected_hyperparameters: Optional[dict] = None
        else:
            overrides = model_params or {}
            self._selected_hyperparameters = {**base_hyperparameters, **overrides}

        self._grid_search_result: Optional[EvaluationResult] = None
        self._forecast_model: Optional[BaseModel] = None

    def _load_config(self) -> dict:
        """Load this frequency's Prophet config block."""
        with open(_CONFIG_PATH) as f:
            full_config = yaml.safe_load(f)
        return full_config["prophet"][self.frequency]

    def _warn_if_declared_frequency_mismatches_data(self) -> None:
        try:
            inferred = infer_frequency(self.raw_df[self.date_column])
        except (UnsupportedFrequencyError, InvalidConfigError):
            warnings.warn(
                f"Could not confirm that '{self.date_column}' has a clean, "
                f"regular daily or weekly spacing. Proceeding with the "
                f"declared frequency ('{self.frequency}') as-is, but verify "
                f"this matches your data's actual granularity.",
                stacklevel=2,
            )
            return

        if inferred != self.frequency:
            warnings.warn(
                f"Declared frequency is '{self.frequency}', but the data's "
                f"most common date spacing looks like '{inferred}' data. "
                f"This can cause the model's seasonality assumptions and "
                f"future forecast date spacing to be inconsistent with the "
                f"data it was actually trained on. If this is intentional, "
                f"this warning can be ignored.",
                stacklevel=2,
            )

    def _chronological_split(self):
        """Shared by evaluate() and the grid search: split, validate, adapt."""
        n_total = len(self.raw_df)
        n_train = int(n_total * self.training_fraction)
        n_test = n_total - n_train

        if n_test == 0:
            raise InvalidConfigError(
                f"training_size of {self.training_fraction * 100:.0f}% leaves no rows for "
                f"evaluation ({n_total} total rows -> {n_train} train, 0 test). "
                f"Use a smaller training_size, or a larger dataset.",
                param_name="training_size",
                provided_value=self.training_fraction,
            )

        train_raw = self.raw_df.iloc[:n_train]
        test_raw = self.raw_df.iloc[n_train:]

        adapter = ProphetAdapter(
            date_column=self.date_column,
            target_column=self.target_column,
            independent_columns=self.independent_columns,
        )
        adapted_train = adapter.transform_train(train_raw)
        adapted_test = adapter.transform_train(test_raw)

        return n_train, n_test, adapted_train, adapted_test

    def _fit_predict_mape(self, params: dict, adapted_train, adapted_test):
        """Fit one candidate hyperparameter set, predict on test, score it."""
        model = ProphetWrapper(
            frequency=self.frequency,
            params=params,
            min_rows_required=self._config["wrapper_config"]["min_rows_required"],
            random_seed=self._config["wrapper_config"]["random_seed"],
        )
        model.fit(adapted_train, target_column="y")

        future_exog = adapted_test.drop(columns=["y"])
        forecast_df = model.predict(horizon=len(adapted_test), future_exog=future_exog)
        mape_result = mape(adapted_test["y"].values, forecast_df["yhat"].values)
        return model, forecast_df, mape_result

    def _run_hyperparameter_grid_search(self) -> EvaluationResult:
        """
        Fits and evaluates EVERY candidate in this frequency's
        hyperparameter_grid on the same chronological train/test split,
        and returns the EvaluationResult of whichever candidate achieved
        the lowest test-set MAPE. Caches the winning hyperparameters
        (for forecast()'s later full-data refit) and the full comparison.
        """
        n_train, n_test, adapted_train, adapted_test = self._chronological_split()
        base = self._config.get("base_hyperparameters", {})

        candidates = []
        for grid_entry in self._config["hyperparameter_grid"]:
            full_params = {**base, **grid_entry}
            model, forecast_df, mape_result = self._fit_predict_mape(
                full_params, adapted_train, adapted_test
            )
            candidates.append((full_params, model, forecast_df, mape_result))

        def sort_key(candidate):
            val = candidate[3].value
            return float("inf") if val != val else val  # NaN sorts last

        candidates.sort(key=sort_key)
        best_params, best_model, best_forecast_df, best_mape = candidates[0]

        test_predictions = adapted_test[["ds", "y"]].copy()
        test_predictions["yhat"] = best_forecast_df["yhat"].values
        if "yhat_lower" in best_forecast_df.columns:
            test_predictions["yhat_lower"] = best_forecast_df["yhat_lower"].values
            test_predictions["yhat_upper"] = best_forecast_df["yhat_upper"].values

        grid_search_summary = [
            {"params": params, "mape": m.value} for params, _, _, m in candidates
        ]

        self._selected_hyperparameters = best_params
        result = EvaluationResult(
            model_name=best_model.model_name,
            mape=best_mape,
            n_train=n_train,
            n_test=n_test,
            test_predictions=test_predictions,
            fitted_model=best_model,
            grid_search_summary=grid_search_summary,
        )
        self._grid_search_result = result
        return result

    def evaluate(self) -> EvaluationResult:
        """
        Chronologically split the data, fit Prophet on the training
        portion only, predict on the held-out test portion, and compute
        MAPE.

        If no explicit model_params override was given at construction,
        this automatically searches this frequency's hyperparameter_grid
        and returns the result of whichever candidate scored best.

        Raises:
            InvalidConfigError: if training_size leaves zero rows for
                the test split.
            InsufficientDataError: propagated from ProphetWrapper.fit().
            ModelConvergenceError / ModelPredictionError: propagated
                from ProphetWrapper if the underlying Prophet fit or
                predict genuinely fails.
        """
        if self._use_grid_search:
            if self._grid_search_result is not None:
                return self._grid_search_result
            return self._run_hyperparameter_grid_search()

        n_train, n_test, adapted_train, adapted_test = self._chronological_split()
        model, forecast_df, mape_result = self._fit_predict_mape(
            self._selected_hyperparameters, adapted_train, adapted_test
        )

        test_predictions = adapted_test[["ds", "y"]].copy()
        test_predictions["yhat"] = forecast_df["yhat"].values
        if "yhat_lower" in forecast_df.columns:
            test_predictions["yhat_lower"] = forecast_df["yhat_lower"].values
            test_predictions["yhat_upper"] = forecast_df["yhat_upper"].values

        return EvaluationResult(
            model_name=model.model_name,
            mape=mape_result,
            n_train=n_train,
            n_test=n_test,
            test_predictions=test_predictions,
            fitted_model=model,
            grid_search_summary=None,
        )

    def forecast(
        self, horizon: int, future_independent_data: Optional[pd.DataFrame] = None
    ) -> ForecastResult:
        """
        Produce a real future forecast, refitting on the FULL dataset
        (not just the train split) the first time this is called.

        If hyperparameters haven't been selected yet (i.e. forecast() is
        called before evaluate() in grid-search mode), the grid search
        runs here first, purely to select hyperparameters — the
        expensive full-data refit itself still only ever happens once.

        Args:
            horizon: number of future periods to forecast.
            future_independent_data: optional DataFrame of real future
                values for this instance's independent_columns, aligned
                to the horizon. If None and independent_columns were
                declared, a default strategy is applied.

        Raises:
            InvalidConfigError: if horizon is not a positive integer.
        """
        if horizon <= 0:
            raise InvalidConfigError(
                f"horizon must be a positive integer, got {horizon}.",
                param_name="horizon",
                provided_value=horizon,
            )

        n_total = len(self.raw_df)
        if horizon > _HORIZON_WARNING_MULTIPLE * n_total:
            warnings.warn(
                f"Forecasting {horizon} periods from only {n_total} periods of history. "
                f"Forecasts this far beyond the available history should be treated with "
                f"caution — uncertainty intervals may understate the true risk of "
                f"extrapolating this far.",
                stacklevel=2,
            )

        adapter = ProphetAdapter(
            date_column=self.date_column,
            target_column=self.target_column,
            independent_columns=self.independent_columns,
        )

        if self._forecast_model is None:
            if self._selected_hyperparameters is None:
                self._run_hyperparameter_grid_search()

            adapted_full = adapter.transform_train(self.raw_df)
            model = ProphetWrapper(
                frequency=self.frequency,
                params=self._selected_hyperparameters,
                min_rows_required=self._config["wrapper_config"]["min_rows_required"],
                random_seed=self._config["wrapper_config"]["random_seed"],
            )
            model.fit(adapted_full, target_column="y")
            self._forecast_model = model

        used_default_strategy = False
        if self.independent_columns and future_independent_data is None:
            future_independent_data = self._build_default_future_independent_data(horizon)
            used_default_strategy = True

        last_date = self.raw_df[self.date_column].max()
        offset = _FREQUENCY_OFFSETS[self.frequency]
        future_dates = pd.Series(
            pd.date_range(
                start=pd.to_datetime(last_date) + pd.tseries.frequencies.to_offset(offset),
                periods=horizon,
                freq=offset,
            )
        )

        future_exog = adapter.transform_future(future_dates, future_independent_data)
        forecast_df = self._forecast_model.predict(horizon=horizon, future_exog=future_exog)

        return ForecastResult(
            model_name=self._forecast_model.model_name,
            forecast=forecast_df,
            horizon=horizon,
            used_default_future_strategy=used_default_strategy,
        )

    def _build_default_future_independent_data(self, horizon: int) -> pd.DataFrame:
        """
        Default future-independent-variable strategy: reuse the actual
        historical values of each independent column, cycling
        (repeating) the full history as many times as needed to cover
        the requested horizon, then taking the trailing `horizon` rows
        of that cycled sequence.
        """
        history = self.raw_df[self.independent_columns].to_numpy()
        n_total = len(history)

        n_repeats = math.ceil(horizon / n_total)
        tiled = np.tile(history, (n_repeats, 1))
        future_values = tiled[-horizon:]

        return pd.DataFrame(future_values, columns=self.independent_columns)