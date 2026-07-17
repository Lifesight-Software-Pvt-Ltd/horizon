"""
HorizonForecaster: the main orchestration class for horizon_forecaster.

Phase 3 extended this from a single hardcoded model (Prophet) to a
registry-driven orchestrator that can run any number of registered
models in one evaluate()/forecast() call, each independent and
parallel. Phase 4 adds the actual ensemble combination on top of that:
evaluate()/forecast() still return the same per-model dict as before
(unchanged, backward compatible), and additionally populate
self.evaluate_ensemble / self.forecast_ensemble (an EnsembleResult --
see evaluation/ensemble.py) as a side effect, mirroring how
self.evaluate_errors/self.forecast_errors already work.

Adding a new model to this orchestrator should mean adding ONE entry to
MODEL_REGISTRY below, nothing else in this file. If a new model ever
forces a change to this file's core logic beyond that, that's a signal
a Phase 1/2 contract (BaseModel/BaseAdapter) had a gap -- fix the
contract, don't special-case the orchestrator.
"""

import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

from horizon_forecaster.data_adapter.base_adapter import BaseAdapter
from horizon_forecaster.data_adapter.catboost_adapter import CatBoostAdapter
from horizon_forecaster.data_adapter.prophet_adapter import ProphetAdapter
from horizon_forecaster.data_adapter.sarimax_adapter import SARIMAXAdapter
from horizon_forecaster.evaluation.ensemble import (
    WEIGHTING_STRATEGIES,
    EnsembleResult,
    build_ensemble,
)
from horizon_forecaster.evaluation.metrics import MapeResult, mape
from horizon_forecaster.exceptions import (
    AllModelsRejectedError,
    HorizonForecasterError,
    InvalidConfigError,
    UnsupportedFrequencyError,
)
from horizon_forecaster.models.base import BaseModel
from horizon_forecaster.models.catboost_wrapper import CatBoostWrapper
from horizon_forecaster.models.prophet_wrapper import ProphetWrapper
from horizon_forecaster.models.sarimax_wrapper import SARIMAXWrapper
from horizon_forecaster.utils import (
    clean_zero_variance,
    infer_frequency,
    validate_frequency,
    validate_mape_threshold,
    validate_training_size,
)

_CONFIG_PATH = Path(__file__).parent / "config" / "default_params.yaml"
_FREQUENCY_OFFSETS = {"daily": "D", "weekly": "7D"}

_HORIZON_WARNING_MULTIPLE = 2

# Every model this orchestrator can run. Only models with a real
# adapter/wrapper implementation belong here. SARIMAX landed in Phase 6
# (Part 1); pymc/neuralprophet are still empty stub files (Phase 6
# Parts 2-3) and are deliberately NOT registered yet.
MODEL_REGISTRY: Dict[str, dict] = {
    "prophet": {"adapter": ProphetAdapter, "wrapper": ProphetWrapper},
    "catboost": {"adapter": CatBoostAdapter, "wrapper": CatBoostWrapper},
    "sarimax": {"adapter": SARIMAXAdapter, "wrapper": SARIMAXWrapper},
}

@dataclass
class EvaluationResult:
    """
    Structured result from evaluating ONE model. HorizonForecaster.evaluate()
    returns a Dict[str, EvaluationResult] keyed by model name -- this
    dataclass itself is unchanged from the single-model Phase 2 version,
    since Phase 4's ensemble logic will consume these per-model results
    directly.

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
    Structured result from forecasting with ONE model. forecast()
    returns a Dict[str, ForecastResult] keyed by model name.

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
    exposes evaluate() (train/test split + MAPE, per model) and
    forecast() (full-data refit + real future prediction, per model).

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
        models: which models to run, e.g. ["prophet", "catboost"]. If
            None (the default), runs every model in MODEL_REGISTRY.
            Raises InvalidConfigError naming any unrecognized model.
        model_params: optional PER-MODEL dict of hyperparameter
            overrides, e.g. {"prophet": {...}, "catboost": {...}}. A
            model with an entry here SKIPS that model's default
            hyperparameter grid search entirely and uses exactly what
            was passed (merged on top of that model's
            base_hyperparameters). A model with no entry (or if
            model_params is omitted entirely) uses its own grid search
            as normal -- this is independent per model.
        mape_threshold: accuracy cutoff (0-100, default 40) a model's
            MAPE must be at or below to be included in the ensemble. A
            model scoring above it is excluded and reported in
            evaluate_ensemble.rejected_models /
            forecast_ensemble.rejected_models. This is an upper bound
            only -- an unusually low MAPE is never rejected. Lower it to
            filter more aggressively; raise it (up to 100) to be more
            permissive.
        ensemble_weighting: "inverse_mape" (default -- models with
            lower MAPE get proportionally more weight) or "equal"
            (every model that passes the threshold gets an equal
            share), passed straight through to
            evaluation/ensemble.py's build_ensemble().

    evaluate()/forecast() ALWAYS return a dict keyed by model name, even
    when only one model is requested -- deliberate, so the return shape
    never depends on how many models were asked for.

    Partial-failure behavior: if some requested models fail (e.g.
    InsufficientDataError) while others succeed, the successful models'
    results are still returned; failures are recorded in
    self.evaluate_errors / self.forecast_errors (Dict[str,
    HorizonForecasterError], repopulated fresh on every call) rather
    than aborting the whole call. If EVERY requested model fails,
    AllModelsRejectedError is raised instead of returning an empty dict
    (a silently-empty successful return would hide a real, total
    failure). This is a much lighter concept than the MAPE-threshold
    ensemble rejection below -- just "did this model produce a result
    or not."

    Ensemble behavior (Phase 4): after evaluate()/forecast() build
    their per-model dict as above (unchanged), each call additionally
    populates self.evaluate_ensemble / self.forecast_ensemble (an
    EnsembleResult, see evaluation/ensemble.py) as a side effect --
    the returned dict's shape/meaning is untouched. forecast() has no
    held-out actuals of its own to score model accuracy, so it always
    reuses evaluate()'s per-model MAPE to decide both the threshold
    rejection and the weights; if evaluate() hasn't been run yet this
    session, forecast() runs it internally once (with a warning, since
    this adds evaluate()'s own train/test-split fit cost on top of
    forecast()'s full-data refit).
    """

    def __init__(
        self,
        raw_df: pd.DataFrame,
        date_column: str,
        target_column: str,
        frequency: str,
        training_size: float = 80,
        independent_columns: Optional[List[str]] = None,
        models: Optional[List[str]] = None,
        model_params: Optional[Dict[str, dict]] = None,
        mape_threshold: float = 40.0,
        ensemble_weighting: str = "inverse_mape",
    ):
        self.frequency = validate_frequency(frequency)
        self.training_fraction = validate_training_size(training_size)
        self.date_column = date_column
        self.target_column = target_column

        self.mape_threshold = validate_mape_threshold(mape_threshold)
        if ensemble_weighting not in WEIGHTING_STRATEGIES:
            raise InvalidConfigError(
                f"ensemble_weighting must be one of {WEIGHTING_STRATEGIES} "
                f"(got '{ensemble_weighting}').",
                param_name="ensemble_weighting",
                provided_value=ensemble_weighting,
            )
        self.ensemble_weighting = ensemble_weighting

        self.models = self._validate_models(models)

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

        # Computed once here, not per-model: which rows go to train/test
        # depends only on training_fraction and row count, never on
        # which model is being evaluated. A training_size that leaves
        # zero test rows is a dataset-level configuration error common
        # to every model, not a single model's data problem -- so it's
        # validated here, at construction, and fails immediately with a
        # direct InvalidConfigError, rather than being discovered inside
        # a per-model try/except during evaluate() (where it would
        # otherwise get caught by the partial-failure handling below and
        # surface as a confusing AllModelsRejectedError instead).
        n_total = len(self.raw_df)
        self._n_train = int(n_total * self.training_fraction)
        self._n_test = n_total - self._n_train
        if self._n_test == 0:
            raise InvalidConfigError(
                f"training_size of {self.training_fraction * 100:.0f}% leaves no rows for "
                f"evaluation ({n_total} total rows -> {self._n_train} train, 0 test). "
                f"Use a smaller training_size, or a larger dataset.",
                param_name="training_size",
                provided_value=self.training_fraction,
            )

        self._configs: Dict[str, dict] = {
            model_name: self._load_config(model_name) for model_name in self.models
        }

        model_params = model_params or {}
        self._use_grid_search: Dict[str, bool] = {}
        self._selected_hyperparameters: Dict[str, Optional[dict]] = {}
        for model_name in self.models:
            config = self._configs[model_name]
            base_hyperparameters = config.get("base_hyperparameters", {})
            override = model_params.get(model_name)

            use_grid_search = override is None and "hyperparameter_grid" in config
            self._use_grid_search[model_name] = use_grid_search
            if use_grid_search:
                self._selected_hyperparameters[model_name] = None
            else:
                self._selected_hyperparameters[model_name] = {**base_hyperparameters, **(override or {})}

        self._grid_search_results: Dict[str, EvaluationResult] = {}
        self._split_cache: Dict[str, tuple] = {}
        self._forecast_models: Dict[str, BaseModel] = {}
        self._forecast_adapters: Dict[str, BaseAdapter] = {}

        self.evaluate_errors: Dict[str, HorizonForecasterError] = {}
        self.forecast_errors: Dict[str, HorizonForecasterError] = {}

        # Phase 4: ensemble side-channel state. evaluate_ensemble/
        # forecast_ensemble are populated by evaluate()/forecast()
        # respectively; None until the corresponding method has run at
        # least once. _evaluate_mape_cache is the per-model MAPE dict
        # forecast() reuses to weight its own ensemble (see forecast()
        # docstring) -- None means evaluate() hasn't run yet this
        # session, {} means it ran but produced zero usable results
        # (AllModelsRejectedError), distinguishing "not tried" from
        # "tried and found nothing" so forecast() doesn't retry forever.
        self.evaluate_ensemble: Optional[EnsembleResult] = None
        self.forecast_ensemble: Optional[EnsembleResult] = None
        self._evaluate_mape_cache: Optional[Dict[str, float]] = None

    def _validate_models(self, models: Optional[List[str]]) -> List[str]:
        if models is None:
            return list(MODEL_REGISTRY.keys())

        if not models:
            raise InvalidConfigError(
                "models list cannot be empty.",
                param_name="models",
                provided_value=models,
            )

        invalid = [m for m in models if m not in MODEL_REGISTRY]
        if invalid:
            raise InvalidConfigError(
                f"Unknown model(s) requested: {invalid}. "
                f"Available models: {list(MODEL_REGISTRY.keys())}.",
                param_name="models",
                provided_value=models,
            )
        return list(models)

    def _load_config(self, model_name: str) -> dict:
        """Load one model's config block for this instance's frequency."""
        with open(_CONFIG_PATH) as f:
            full_config = yaml.safe_load(f)
        try:
            return full_config[model_name][self.frequency]
        except KeyError:
            raise InvalidConfigError(
                f"No config found for model '{model_name}' at frequency "
                f"'{self.frequency}' in {_CONFIG_PATH}.",
                param_name="models",
                provided_value=model_name,
            )

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

    def _prepare_split(self, model_name: str):
        """
        Builds and caches (n_train, n_test, adapted_train,
        adapted_test_features, y_test) for ONE model, using that
        model's own adapter class. Cached per model per instance,
        since the split doesn't depend on hyperparameters -- grid
        search candidates all reuse the same adapted data.

        Critical detail: the test split is shaped via
        adapter.transform_future(), NOT adapter.transform_train().
        Using transform_train() on the test split alone (the Phase 2
        Prophet-only approach) happened to work for ProphetAdapter,
        which is a stateless pure function -- but it would be actively
        WRONG for an adapter that carries state across the train/future
        boundary like CatBoostAdapter: transform_train() resets the
        adapter's retained self._reference_date to the test split's own
        first date, which would restart the trend index (time_idx) at 0
        for the test split instead of letting it continue seamlessly
        from where the training history left off -- silently corrupting
        the very feature the redesign added, and any subsequent use of
        the adapter. transform_future() is the correct call here -- it's
        exactly what a genuine future forecast would do, reusing the
        reference_date established by the train-split fit. The only
        difference from a real future forecast is that the actual y
        values for this "future" period ARE already known (it's
        held-out historical data, being used for evaluation) -- so
        they're read directly from raw_df, never routed through the
        adapter, which by design never sees them.
        """
        if model_name in self._split_cache:
            return self._split_cache[model_name]

        n_train, n_test = self._n_train, self._n_test
        train_raw = self.raw_df.iloc[:n_train]
        test_raw = self.raw_df.iloc[n_train:]

        AdapterClass = MODEL_REGISTRY[model_name]["adapter"]
        adapter = AdapterClass(
            date_column=self.date_column,
            target_column=self.target_column,
            independent_columns=self.independent_columns,
            frequency=self.frequency,
        )
        adapted_train = adapter.transform_train(train_raw)

        future_dates = pd.Series(test_raw[self.date_column].reset_index(drop=True))
        future_independent_data = (
            test_raw[self.independent_columns].reset_index(drop=True)
            if self.independent_columns
            else None
        )
        adapted_test_features = adapter.transform_future(future_dates, future_independent_data)
        y_test = test_raw[self.target_column].to_numpy()

        result = (n_train, n_test, adapted_train, adapted_test_features, y_test)
        self._split_cache[model_name] = result
        return result

    def _fit_predict_mape(self, model_name: str, params: dict, adapted_train, adapted_test_features, y_test):
        """Fit one model with one candidate hyperparameter set, predict on the held-out test split, score it."""
        WrapperClass = MODEL_REGISTRY[model_name]["wrapper"]
        wrapper_config = self._configs[model_name]["wrapper_config"]
        model = WrapperClass(
            frequency=self.frequency,
            params=params,
            min_rows_required=wrapper_config["min_rows_required"],
            random_seed=wrapper_config["random_seed"],
        )
        model.fit(adapted_train, target_column="y")

        forecast_df = model.predict(horizon=len(adapted_test_features), future_exog=adapted_test_features)
        mape_result = mape(y_test, forecast_df["yhat"].values)
        return model, forecast_df, mape_result

    def _build_test_predictions(self, adapted_test_features, y_test, forecast_df) -> pd.DataFrame:
        test_predictions = pd.DataFrame({
            "ds": adapted_test_features["ds"].reset_index(drop=True),
            "y": y_test,
            "yhat": forecast_df["yhat"].values,
        })
        if "yhat_lower" in forecast_df.columns:
            test_predictions["yhat_lower"] = forecast_df["yhat_lower"].values
            test_predictions["yhat_upper"] = forecast_df["yhat_upper"].values
        return test_predictions

    def _run_hyperparameter_grid_search(self, model_name: str) -> EvaluationResult:
        n_train, n_test, adapted_train, adapted_test_features, y_test = self._prepare_split(model_name)
        config = self._configs[model_name]
        base = config.get("base_hyperparameters", {})

        candidates = []
        for grid_entry in config["hyperparameter_grid"]:
            full_params = {**base, **grid_entry}
            model, forecast_df, mape_result = self._fit_predict_mape(
                model_name, full_params, adapted_train, adapted_test_features, y_test
            )
            candidates.append((full_params, model, forecast_df, mape_result))

        def sort_key(candidate):
            val = candidate[3].value
            return float("inf") if val != val else val

        candidates.sort(key=sort_key)
        best_params, best_model, best_forecast_df, best_mape = candidates[0]

        test_predictions = self._build_test_predictions(adapted_test_features, y_test, best_forecast_df)
        grid_search_summary = [{"params": p, "mape": m.value} for p, _, _, m in candidates]

        self._selected_hyperparameters[model_name] = best_params
        result = EvaluationResult(
            model_name=best_model.model_name,
            mape=best_mape,
            n_train=n_train,
            n_test=n_test,
            test_predictions=test_predictions,
            fitted_model=best_model,
            grid_search_summary=grid_search_summary,
        )
        self._grid_search_results[model_name] = result
        return result

    def _evaluate_single_model(self, model_name: str) -> EvaluationResult:
        if self._use_grid_search[model_name]:
            if model_name in self._grid_search_results:
                return self._grid_search_results[model_name]
            return self._run_hyperparameter_grid_search(model_name)

        n_train, n_test, adapted_train, adapted_test_features, y_test = self._prepare_split(model_name)
        model, forecast_df, mape_result = self._fit_predict_mape(
            model_name, self._selected_hyperparameters[model_name], adapted_train, adapted_test_features, y_test
        )
        test_predictions = self._build_test_predictions(adapted_test_features, y_test, forecast_df)

        return EvaluationResult(
            model_name=model.model_name,
            mape=mape_result,
            n_train=n_train,
            n_test=n_test,
            test_predictions=test_predictions,
            fitted_model=model,
            grid_search_summary=None,
        )

    def evaluate(self) -> Dict[str, EvaluationResult]:
        """
        Runs evaluate() independently for every requested model.
        Always returns a dict keyed by model name (even for a single
        model). See class docstring for partial-failure behavior.

        Also populates self.evaluate_ensemble (an EnsembleResult) and
        self._evaluate_mape_cache (reused by forecast(), see its
        docstring) as a side effect -- the returned dict itself is
        unchanged from Phase 3.

        Raises:
            AllModelsRejectedError: if every requested model failed.
                (Note: this is about fit/predict failing outright: a
                model that fits fine but scores outside the ensemble's
                MAPE threshold does NOT raise here -- see
                evaluate_ensemble.rejected_models instead.)
        """
        results: Dict[str, EvaluationResult] = {}
        errors: Dict[str, HorizonForecasterError] = {}

        for model_name in self.models:
            try:
                results[model_name] = self._evaluate_single_model(model_name)
            except HorizonForecasterError as e:
                errors[model_name] = e

        self.evaluate_errors = errors

        if not results:
            self._evaluate_mape_cache = {}
            self.evaluate_ensemble = None
            raise AllModelsRejectedError(
                f"All requested models failed during evaluate(): "
                f"{ {k: str(v) for k, v in errors.items()} }",
                rejected_models=list(errors.keys()),
            )

        self._evaluate_mape_cache = {name: r.mape.value for name, r in results.items()}
        first_result = next(iter(results.values()))
        self.evaluate_ensemble = build_ensemble(
            per_model_mape=self._evaluate_mape_cache,
            per_model_predictions={name: r.test_predictions for name, r in results.items()},
            mape_threshold=self.mape_threshold,
            weighting=self.ensemble_weighting,
            actuals=first_result.test_predictions[["ds", "y"]],
        )

        return results

    def _forecast_single_model(
        self, model_name: str, horizon: int, future_independent_data: Optional[pd.DataFrame]
    ) -> ForecastResult:
        AdapterClass = MODEL_REGISTRY[model_name]["adapter"]

        if model_name not in self._forecast_models:
            if self._selected_hyperparameters[model_name] is None:
                self._run_hyperparameter_grid_search(model_name)

            adapter = AdapterClass(
                date_column=self.date_column,
                target_column=self.target_column,
                independent_columns=self.independent_columns,
                frequency=self.frequency,
            )
            adapted_full = adapter.transform_train(self.raw_df)

            wrapper_config = self._configs[model_name]["wrapper_config"]
            WrapperClass = MODEL_REGISTRY[model_name]["wrapper"]
            model = WrapperClass(
                frequency=self.frequency,
                params=self._selected_hyperparameters[model_name],
                min_rows_required=wrapper_config["min_rows_required"],
                random_seed=wrapper_config["random_seed"],
            )
            model.fit(adapted_full, target_column="y")

            self._forecast_models[model_name] = model
            self._forecast_adapters[model_name] = adapter

        adapter = self._forecast_adapters[model_name]

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
        forecast_df = self._forecast_models[model_name].predict(horizon=horizon, future_exog=future_exog)

        return ForecastResult(
            model_name=self._forecast_models[model_name].model_name,
            forecast=forecast_df,
            horizon=horizon,
            used_default_future_strategy=used_default_strategy,
        )

    def forecast(
        self, horizon: int, future_independent_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, ForecastResult]:
        """
        Runs forecast() independently for every requested model, each
        refitting on the FULL dataset (once, cached) the first time
        it's called. Always returns a dict keyed by model name (even
        for a single model). See class docstring for partial-failure
        behavior.

        Also populates self.forecast_ensemble (an EnsembleResult) as a
        side effect -- the returned dict itself is unchanged from
        Phase 3. Since forecast() has no held-out actuals to score
        accuracy against, the ensemble's weights (and which models pass
        the mape_threshold) always come from evaluate()'s
        MAPE, not anything computed here. If evaluate() hasn't been run
        yet this session, forecast() runs it internally once (see the
        warning below) purely to derive those weights -- a separate
        fit/split from forecast()'s own full-data refit, so it doesn't
        affect the forecast values themselves, only the ensemble
        weighting.

        Args:
            horizon: number of future periods to forecast.
            future_independent_data: optional DataFrame of real future
                values for this instance's independent_columns, aligned
                to the horizon. If None and independent_columns were
                declared, a default strategy is applied.

        Raises:
            InvalidConfigError: if horizon is not a positive integer.
            AllModelsRejectedError: if every requested model failed.
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

        if self._evaluate_mape_cache is None:
            warnings.warn(
                "forecast() is running evaluate() internally, since it hasn't been "
                "called yet this session, to derive per-model MAPE for ensemble "
                "weighting. This adds evaluate()'s own train/test-split fit cost on "
                "top of forecast()'s full-data refit. Call evaluate() explicitly "
                "first if you'd like to see (or avoid paying twice for) that cost "
                "separately.",
                stacklevel=2,
            )
            try:
                self.evaluate()
            except AllModelsRejectedError:
                # evaluate() itself couldn't produce a single result. forecast()
                # still attempts its own full-data fits below (a model that
                # failed on the smaller train/test split may yet succeed with
                # more data) -- it just won't have ensemble weights available.
                self._evaluate_mape_cache = {}

        results: Dict[str, ForecastResult] = {}
        errors: Dict[str, HorizonForecasterError] = {}

        for model_name in self.models:
            try:
                results[model_name] = self._forecast_single_model(model_name, horizon, future_independent_data)
            except HorizonForecasterError as e:
                errors[model_name] = e

        self.forecast_errors = errors

        if not results:
            self.forecast_ensemble = None
            raise AllModelsRejectedError(
                f"All requested models failed during forecast(): "
                f"{ {k: str(v) for k, v in errors.items()} }",
                rejected_models=list(errors.keys()),
            )

        forecast_mape_input = {
            name: self._evaluate_mape_cache[name]
            for name in results
            if name in self._evaluate_mape_cache
        }
        if forecast_mape_input:
            self.forecast_ensemble = build_ensemble(
                per_model_mape=forecast_mape_input,
                per_model_predictions={name: results[name].forecast for name in forecast_mape_input},
                mape_threshold=self.mape_threshold,
                weighting=self.ensemble_weighting,
                actuals=None,
            )
        else:
            self.forecast_ensemble = None

        return results

    def _build_default_future_independent_data(self, horizon: int) -> pd.DataFrame:
        """
        Default future-independent-variable strategy: reuse the actual
        historical values of each independent column, cycling
        (repeating) the full history as many times as needed to cover
        the requested horizon, then taking the trailing `horizon` rows
        of that cycled sequence. Model-agnostic -- operates on
        self.raw_df directly, not through any adapter.
        """
        history = self.raw_df[self.independent_columns].to_numpy()
        n_total = len(history)

        n_repeats = math.ceil(horizon / n_total)
        tiled = np.tile(history, (n_repeats, 1))
        future_values = tiled[-horizon:]

        return pd.DataFrame(future_values, columns=self.independent_columns)