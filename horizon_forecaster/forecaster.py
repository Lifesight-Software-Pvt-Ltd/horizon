"""
HorizonForecaster: the main orchestration class for horizon_forecaster.

A registry-driven orchestrator that runs any number of registered models
in one evaluate()/forecast() call, each independent. evaluate()/
forecast() return a per-model dict keyed by model name, and additionally
populate self.evaluate_ensemble / self.forecast_ensemble (a dict of
EnsembleResult keyed by combination method -- see
evaluation/ensemble.py) as a side effect, mirroring how
self.evaluate_errors/self.forecast_errors already work.

Data is split into three chronological windows: train, validation
(every selection decision) and test (reported, never selected on). See
the HorizonForecaster class docstring for why.

Adding a new model to this orchestrator should mean adding ONE entry to
MODEL_REGISTRY below, nothing else in this file. If a new model ever
forces a change to this file's core logic beyond that, that is a signal
the shared contract (BaseModel/BaseAdapter) has a gap -- fix the
contract, rather than special-casing the orchestrator.
"""

import numbers
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import pandas as pd
import yaml

from horizon_forecaster.data_adapter.base_adapter import BaseAdapter
from horizon_forecaster.data_adapter.bayesianpymc_adapter import BayesianPyMCAdapter
from horizon_forecaster.data_adapter.catboost_adapter import CatBoostAdapter
from horizon_forecaster.data_adapter.neuralprophet_adapter import NeuralProphetAdapter
from horizon_forecaster.data_adapter.prophet_adapter import ProphetAdapter
from horizon_forecaster.data_adapter.sarimax_adapter import SARIMAXAdapter
from horizon_forecaster.evaluation.ensemble import (
    DEFAULT_ENSEMBLE_METHOD,
    ENSEMBLE_METHODS,
    EnsembleResult,
    build_ensembles,
)
from horizon_forecaster.evaluation.metrics import MapeResult, mape
from horizon_forecaster.future_exog import (
    check_future_table,
    describe_generation_risks,
    AGGREGATORS,
    FUTURE_EXOG_STRATEGIES,
    resolve_future_exog,
)
from horizon_forecaster.exceptions import (
    AllModelsRejectedError,
    HorizonForecasterError,
    InsufficientDataError,
    InvalidConfigError,
    ModelConvergenceError,
    UnsupportedFrequencyError,
)
from horizon_forecaster.models.base import BaseModel
from horizon_forecaster.models.bayesianpymc_wrapper import BayesianPyMCWrapper
from horizon_forecaster.models.catboost_wrapper import CatBoostWrapper
from horizon_forecaster.models.neuralprophet_wrapper import NeuralProphetWrapper
from horizon_forecaster.models.prophet_wrapper import ProphetWrapper
from horizon_forecaster.models.sarimax_wrapper import SARIMAXWrapper
from horizon_forecaster.utils import (
    MAX_CONTINUOUS_GAP_DAYS,
    clean_zero_variance,
    describe_date_spacing,
    find_duplicate_dates,
    parse_date_column,
    parse_driver_column,
    resolve_split,
    validate_frequency,
    validate_mape_threshold,
    validate_no_missing_values,
)

_CONFIG_PATH = Path(__file__).parent / "config" / "default_params.yaml"
_FREQUENCY_OFFSETS = {"daily": "D", "weekly": "7D"}


@lru_cache(maxsize=1)
def _default_params() -> dict:
    """
    The parsed default_params.yaml, read once per process.

    It used to be re-read and re-parsed for every requested model, so a
    default five-model construction parsed the same ~600-line file five
    times. The file ships inside the package and never changes at
    runtime, so caching is safe.

    Callers must treat the returned dict as READ-ONLY -- it is shared by
    every HorizonForecaster in the process. Nothing mutates it today:
    _load_config only indexes into it, and the per-candidate merges in
    the grid search build new dicts via {**base, **entry}.
    """
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


# Warn once the requested horizon exceeds this fraction of the available
# history. Deliberately strict: the previous rule only fired when the
# horizon exceeded TWICE the entire history, so forecasting 100 weeks from
# 121 weeks of data passed silently -- exactly the case that produced an
# implausible real-world forecast. Forecasting more than half your history
# ahead is already a stretch and deserves a flag.
_HORIZON_WARNING_FRACTION = 0.5

# Forecasts are flagged when they peak more than this far above (or bottom
# this far below) anything ever observed in the target. A trending series
# legitimately forecasts past its historical max, so a plain
# "outside the historical range" rule would false-positive constantly --
# this margin only trips on implausible departures.
_FORECAST_RANGE_TOLERANCE = 0.5

# Every model this orchestrator can run. Only models with a real
# adapter/wrapper implementation belong here.
MODEL_REGISTRY: Dict[str, dict] = {
    "prophet": {"adapter": ProphetAdapter, "wrapper": ProphetWrapper},
    "catboost": {"adapter": CatBoostAdapter, "wrapper": CatBoostWrapper},
    "sarimax": {"adapter": SARIMAXAdapter, "wrapper": SARIMAXWrapper},
    "neuralprophet": {"adapter": NeuralProphetAdapter, "wrapper": NeuralProphetWrapper},
    "bayesianpymc": {"adapter": BayesianPyMCAdapter, "wrapper": BayesianPyMCWrapper},
}


@dataclass
class _Selection:
    """
    Internal record of the hyperparameter-selection step for ONE model.

    Selection happens on the VALIDATION window and nothing else -- this is
    the whole point of the three-way split. Keeping it in its own record,
    cached separately from EvaluationResult, means forecast() can reuse a
    selection (which is all it needs: the winning params, plus a validation
    MAPE for ensemble screening/weighting) WITHOUT paying for the train+
    validation refit and test prediction that only evaluate() reports.
    """

    params: dict
    validation_mape: MapeResult
    grid_search_summary: Optional[List[dict]] = None


@dataclass
class EvaluationResult:
    """
    Structured result from evaluating ONE model. HorizonForecaster.evaluate()
    returns a Dict[str, EvaluationResult] keyed by model name.

    The two MAPEs here are the heart of the three-way split, and they are
    NOT interchangeable:

        validation_mape  the score every DECISION was made on -- which
                         hyperparameter candidate won, whether the model
                         clears mape_threshold, and its inverse-MAPE
                         weight. Because candidates were selected to
                         minimise it, it is a selection score and is
                         optimistic by construction.

        mape             the score REPORTED, measured on the test window,
                         which no selection path is ever allowed to read.
                         This is the honest out-of-sample estimate.

    The gap between them is itself a useful diagnostic: a large one means
    the grid search found a candidate that suited the validation window
    specifically rather than the series generally.

    Attributes:
        model_name: which model this result is for (e.g. "prophet").
        mape: MapeResult on the TEST window -- the reported score.
        validation_mape: MapeResult on the VALIDATION window -- the
            selection score. Present whether or not a grid search ran,
            since the ensemble needs it either way.
        n_train: rows used to fit during selection.
        n_validation: rows in the validation window.
        n_test: rows in the test window.
        test_predictions: DataFrame with columns ["ds", "y", "yhat"]
            (plus "yhat_lower"/"yhat_upper" if available), over the test
            window.
        fitted_model: the wrapper instance fitted on TRAIN + VALIDATION
            using the winning hyperparameters -- i.e. the model that
            actually produced `test_predictions`. Deliberately not the
            train-only fit used during selection: refitting on everything
            available before scoring test is what makes the reported MAPE
            an estimate of the procedure that produces the real forecast.
        grid_search_summary: if hyperparameters were selected via grid
            search (no explicit model_params override given), a list of
            every candidate tried, each as {"params": {...}, "mape": float}
            where that mape is the VALIDATION score, sorted best-to-worst.
            None if an explicit override was used -- and also None when a
            grid search WAS attempted but every candidate failed on the
            training window, so the first configured candidate was used
            as an unselected fallback. Do not read `is None` as "an
            override was given"; check `model_params` for that.
    """

    model_name: str
    mape: MapeResult
    validation_mape: MapeResult
    n_train: int
    n_validation: int
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
        used_default_future_strategy: True if ANY future independent-variable
            values were auto-generated (via future_exog_strategy) rather than
            caller-supplied.
        future_exog_used: the resolved future independent values actually fed
            to the models (supplied columns + generated columns), with a "ds"
            column, for inspection. None when there are no independent columns.
    """

    model_name: str
    forecast: pd.DataFrame
    horizon: int
    used_default_future_strategy: bool
    future_exog_used: Optional[pd.DataFrame] = None


class HorizonForecaster:
    """
    Main orchestrator. Takes raw user data once at construction, and
    exposes evaluate() (train/validation/test split + MAPE, per model) and
    forecast() (full-data refit + real future prediction, per model).

    Args:
        raw_df: user's raw DataFrame containing at least date_column and
            target_column, plus any of independent_columns that apply.
        date_column: name of the date column in raw_df.
        target_column: name of the target/dependent variable column.
        frequency: "daily" or "weekly".
        validation_size: size of the validation window, used for EVERY
            selection decision (hyperparameter grid search, mape_threshold
            screening, inverse-MAPE weighting). An int is a number of
            periods, a float strictly between 0 and 1 is a fraction of the
            data, and None (the default) uses the frequency default from
            utils.DEFAULT_EVALUATION_WINDOWS -- 84 periods daily, 39
            weekly.
        test_size: size of the test window, which is REPORTED and never
            read by any selection path. Same int/float/None semantics as
            validation_size; the frequency default is 28 periods daily, 13
            weekly. Training takes every row the two windows leave.
        training_size: the percentage route, for callers who would rather
            think in percentages than windows. A percentage (50-100) or
            fraction (0.5-1.0), meaning exactly what it always has; the
            remainder is divided between validation and test, three-quarters
            to validation. Mutually exclusive with validation_size/
            test_size -- passing both raises InvalidConfigError rather
            than letting one silently win. Note this parameter is
            percentages ONLY: training_size=80 is 80%, never 80 periods.
        independent_columns: names of independent variable columns
            (spend, impressions, promos, etc.) to use, if any. Values must
            be numbers; booleans and the on/off words in
            utils.YES_NO_WORDS (yes/no, y/n, t/f, true/false, on/off) are
            read as 1/0, and any other text raises InvalidConfigError
            saying what is unclear (see utils.parse_driver_column).
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
        mape_threshold: degenerate-fit filter (0.01-100, default 40), applied to
            each model's VALIDATION MAPE. A model scoring above it is
            excluded from the combination and reported in
            rejected_models. Pass None to admit every model. Treat this
            as a guard against degenerate fits rather than an accuracy
            screen: the default combination rule already discards the
            worst member on every period, so screening hard AND trimming
            spends the same robustness budget twice -- five members
            screened down to three leaves the median.
        ensemble_method: how surviving models are combined. "trimmed"
            (the default -- discard the highest and lowest member per
            period and average the rest), "inverse_mape" (lower
            validation MAPE gets proportionally more weight), or "equal".
            Accepts a list to build several, or None to build all of
            them; the models are already fitted by then, so extra rules
            are close to free. evaluate_ensemble/forecast_ensemble are
            keyed by method name in every case.
        future_exog_strategy: how forecast() generates future values for the
            independent columns when they aren't supplied. One of
            future_exog.FUTURE_EXOG_STRATEGIES: "carry_forward" (default --
            hold the last value flat), "recent_average", "historical_average"
            (season-aware), "seasonal_naive" (seasonal cycle + drift),
            "linear_trend", or "cycle" (the legacy replay-the-tail behaviour).
            Continuous drivers use the chosen strategy; binary event flags
            default to 0 (except under seasonal_naive). See future_exog.py.
        event_columns: names of the binary 0/1 event/flag columns (promos,
            holidays) among independent_columns, so the strategy defaults them
            to 0 ("no event unless supplied") instead of carrying them forward.
            None (default) auto-detects flags: columns whose values are only
            0 and 1 (any other column, even one with two values such as a
            0/5,000 spend or a 9.99/12.99 price, is a quantity). Passing a
            list disables auto-detection and uses exactly those columns.
        future_exog_aggregator: "mean" (default) or "median", used by the
            recent_average / historical_average / seasonal_naive strategies
            (median is robust to spikes in the driver history).
        recent_average_window: N for the recent_average strategy -- how many
            trailing periods to average. Default 28.
        date_format: optional strftime pattern for the date column, e.g.
            "%d/%m/%Y". None (the default) detects the format: a datetime
            column is used as-is, and text must be in one of
            utils.ACCEPTED_DATE_FORMATS (YYYY-MM-DD, YYYY/MM/DD, DD-MM-YYYY,
            DD/MM/YYYY, MM-DD-YYYY, MM/DD/YYYY) or a full ISO 8601
            timestamp, the same on every row. Text that could be read
            either day-first or month-first raises InvalidConfigError
            rather than being guessed; pass date_format to settle it, or
            for any format outside that list.

    Dates are parsed BEFORE the data is sorted or split, so text dates sort
    by time rather than alphabetically. The date column must then meet the
    acceptance contract checked in _validate_date_spacing(): each date once,
    every step a whole number of periods, more than half of the steps
    exactly one period, and no run of missing dates longer than
    MAX_CONTINUOUS_GAP_DAYS. Anything else raises; shorter gaps only warn.

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

    Evaluation windows: this class splits chronologically into THREE
    windows, not two. Hyperparameter grid search, mape_threshold
    screening and inverse-MAPE weighting all score on the VALIDATION
    window; the winning candidate is then refit on train + validation and
    scored once on the TEST window, which no selection path is ever
    allowed to read. EvaluationResult carries both numbers -- `mape` is
    the reported test-window score, `validation_mape` the selection score
    it was chosen by. Before 1.0 there was one held-out window serving
    both roles, which made every reported MAPE a selection score and
    therefore optimistic.

    Ensemble behavior: after evaluate()/forecast() build their per-model
    dict as above, each call additionally populates
    self.evaluate_ensemble / self.forecast_ensemble as a side effect --
    a Dict[str, EnsembleResult] keyed by combination method (see
    evaluation/ensemble.py), always a dict even when one method was
    requested, matching the convention the per-model returns already
    follow. forecast() has no held-out actuals of its own, so any
    screening or weighting reuses the validation MAPE from selection,
    which it has to perform anyway in order to know which hyperparameters
    to fit with.
    """

    def __init__(
        self,
        raw_df: pd.DataFrame,
        date_column: str,
        target_column: str,
        frequency: str,
        validation_size: Optional[Union[int, float]] = None,
        test_size: Optional[Union[int, float]] = None,
        training_size: Optional[Union[int, float]] = None,
        independent_columns: Optional[List[str]] = None,
        models: Optional[List[str]] = None,
        model_params: Optional[Dict[str, dict]] = None,
        mape_threshold: Optional[float] = 40.0,
        ensemble_method: Union[str, Sequence[str], None] = DEFAULT_ENSEMBLE_METHOD,
        future_exog_strategy: str = "carry_forward",
        event_columns: Optional[List[str]] = None,
        future_exog_aggregator: str = "mean",
        recent_average_window: int = 28,
        date_format: Optional[str] = None,
    ):
        self.frequency = validate_frequency(frequency)
        self.date_column = date_column
        self.target_column = target_column

        self.mape_threshold = validate_mape_threshold(mape_threshold)
        self.ensemble_methods = self._validate_ensemble_methods(ensemble_method)

        # Future-exogenous-data strategy (how forecast() generates future
        # driver values when they aren't supplied). Validated here like
        # ensemble_method; event_columns is validated against the surviving
        # independent columns below, once those are known.
        if future_exog_strategy not in FUTURE_EXOG_STRATEGIES:
            raise InvalidConfigError(
                f"future_exog_strategy must be one of {sorted(FUTURE_EXOG_STRATEGIES)} "
                f"(got '{future_exog_strategy}').",
                param_name="future_exog_strategy",
                provided_value=future_exog_strategy,
            )
        if future_exog_aggregator not in AGGREGATORS:
            raise InvalidConfigError(
                f"future_exog_aggregator must be one of {sorted(AGGREGATORS)} "
                f"(got '{future_exog_aggregator}').",
                param_name="future_exog_aggregator",
                provided_value=future_exog_aggregator,
            )
        if (
            isinstance(recent_average_window, bool)
            or not isinstance(recent_average_window, numbers.Integral)
            or recent_average_window <= 0
        ):
            raise InvalidConfigError(
                "recent_average_window must be a positive integer "
                f"(got {recent_average_window!r}).",
                param_name="recent_average_window",
                provided_value=recent_average_window,
            )
        self.future_exog_strategy = future_exog_strategy
        self.future_exog_aggregator = future_exog_aggregator
        self.recent_average_window = int(recent_average_window)

        self.models = self._validate_models(models)

        # Parse dates first, on a copy, so the caller's DataFrame is never
        # modified. Everything below -- sorting, the window split, frequency
        # inference, every model -- must see real dates: sorting text dates
        # orders them alphabetically, which silently scrambles the windows.
        # A missing date column is left for validate_no_missing_values() to
        # report with the other missing columns.
        self.date_format = date_format
        if date_column in raw_df.columns:
            raw_df = raw_df.copy()
            raw_df[date_column] = parse_date_column(
                raw_df[date_column], date_column, date_format=date_format
            )

        requested_independent_columns = independent_columns or []

        # Drivers too: booleans and the on/off words in YES_NO_WORDS become
        # 1/0, numbers written as text become numbers, and any other text
        # raises a specific error rather than the generic "non-numeric" one.
        # Done on a copy, before the zero-variance and missing-value checks.
        to_parse = [
            c
            for c in requested_independent_columns
            if c in raw_df.columns and c not in (date_column, target_column)
        ]
        if to_parse:
            raw_df = raw_df.copy()
            dates = raw_df[date_column] if date_column in raw_df.columns else None
            for col in to_parse:
                raw_df[col] = parse_driver_column(raw_df[col], col, dates=dates).to_numpy()
        # Kept for forecast(): a future table may name a declared driver that
        # was dropped below, which warns rather than errors.
        self._declared_independent_columns = list(requested_independent_columns)
        cleaned_df, dropped_columns = clean_zero_variance(
            raw_df, exclude_columns=[date_column, target_column]
        )
        self.dropped_zero_variance_columns = dropped_columns

        self.independent_columns = [
            c for c in requested_independent_columns if c not in dropped_columns
        ]

        # Complete-data requirement: a single missing/non-numeric cell in any
        # column used for training/testing (date, target, or a surviving
        # independent column) is a hard failure -- missing data is never
        # silently dropped or imputed. Done here, at construction, alongside
        # the other dataset-level validation.
        validate_no_missing_values(cleaned_df, date_column, target_column, self.independent_columns)
        self._reject_duplicate_dates(cleaned_df)

        # event_columns (explicit 0/1 flags for the future-exog strategy). None
        # means "auto-detect flags"; a list means "these exact columns are the
        # flags, auto-detect nothing else". A listed column dropped for zero
        # variance is silently ignored; anything else unrecognized is an error.
        if event_columns is None:
            self.event_columns: Optional[List[str]] = None
        else:
            unknown_events = [
                c
                for c in event_columns
                if c not in self.independent_columns and c not in dropped_columns
            ]
            if unknown_events:
                raise InvalidConfigError(
                    f"event_columns {unknown_events} are not among the independent_columns "
                    f"{self.independent_columns}.",
                    param_name="event_columns",
                    provided_value=event_columns,
                )
            self.event_columns = [c for c in event_columns if c in self.independent_columns]

        self.raw_df = cleaned_df.sort_values(date_column, kind="stable").reset_index(drop=True)
        self._validate_date_spacing()

        # Computed once here, not per-model: which rows go to train /
        # validation / test depends only on the requested sizes and the row
        # count, never on which model is being evaluated. A configuration
        # that leaves a window empty is a dataset-level error common to
        # every model, not a single model's data problem -- so it's
        # validated here, at construction, and fails immediately with a
        # direct InvalidConfigError, rather than being discovered inside a
        # per-model try/except during evaluate() (where it would otherwise
        # get caught by the partial-failure handling below and surface as a
        # confusing AllModelsRejectedError instead).
        n_total = len(self.raw_df)
        self._n_train, self._n_validation, self._n_test = resolve_split(
            n_total=n_total,
            frequency=self.frequency,
            training_size=training_size,
            validation_size=validation_size,
            test_size=test_size,
        )

        self._configs: Dict[str, dict] = {
            model_name: self._load_config(model_name) for model_name in self.models
        }

        # The split is legal in the abstract (every window has rows), but
        # the models actually requested may each need more training rows
        # than it leaves. Checked here rather than per-model at fit time so
        # the caller gets one clear, actionable message naming the binding
        # model up front -- otherwise every model fails separately inside
        # evaluate()'s partial-failure handling and the real cause (the
        # split, not the data) is buried in an AllModelsRejectedError.
        self._validate_train_size_against_model_floors()

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
                self._selected_hyperparameters[model_name] = {
                    **base_hyperparameters,
                    **(override or {}),
                }

        # Selection (winning params + validation MAPE) is cached separately
        # from the full EvaluationResult, because forecast() needs only the
        # former -- see _Selection's docstring.
        self._selection_cache: Dict[str, _Selection] = {}
        self._evaluation_cache: Dict[str, EvaluationResult] = {}
        self._split_cache: Dict[str, tuple] = {}
        self._refit_cache: Dict[str, tuple] = {}
        # Drivers that vary in each fit window ("train", "train+validation"),
        # computed once per window -- see _drivers_that_vary_in().
        self._window_drivers: Dict[str, List[str]] = {}
        self._forecast_models: Dict[str, BaseModel] = {}
        self._forecast_adapters: Dict[str, BaseAdapter] = {}

        # The future independent values actually used by the most
        # recent forecast() (supplied columns + strategy-generated columns),
        # exposed for inspection. None until forecast() runs with exog.
        self.future_exog_used: Optional[pd.DataFrame] = None

        self.evaluate_errors: Dict[str, HorizonForecasterError] = {}
        self.forecast_errors: Dict[str, HorizonForecasterError] = {}

        # Ensemble side-channel state. evaluate_ensemble/forecast_ensemble
        # are populated by evaluate()/forecast() respectively, each a dict
        # keyed by combination method (see the class docstring); None until
        # the corresponding method has run at least once.
        self.evaluate_ensemble: Optional[Dict[str, EnsembleResult]] = None
        self.forecast_ensemble: Optional[Dict[str, EnsembleResult]] = None

    def _validate_ensemble_methods(
        self, ensemble_method: Union[str, Sequence[str], None]
    ) -> List[str]:
        """
        Resolve `ensemble_method` into the list of rules to build.

        Accepts a single name, a sequence of names, or None meaning "all
        of them". None is genuinely useful rather than indulgent: the
        models have already been fitted by the time any of this runs, so
        every additional rule costs one pass of arithmetic over
        predictions that are already in memory. A caller who wants to see
        how much the choice of rule actually matters on their series
        should not have to re-run the fits three times to find out.
        """
        if ensemble_method is None:
            return list(ENSEMBLE_METHODS)

        requested = [ensemble_method] if isinstance(ensemble_method, str) else list(ensemble_method)

        if not requested:
            raise InvalidConfigError(
                "ensemble_method cannot be an empty list. Pass a method name, a list "
                "of names, or None for all of them.",
                param_name="ensemble_method",
                provided_value=ensemble_method,
            )

        invalid = [m for m in requested if m not in ENSEMBLE_METHODS]
        if invalid:
            raise InvalidConfigError(
                f"Unknown ensemble method(s): {invalid}. "
                f"Available methods: {list(ENSEMBLE_METHODS)}.",
                param_name="ensemble_method",
                provided_value=ensemble_method,
            )

        # De-duplicate while preserving the caller's order.
        return list(dict.fromkeys(requested))

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
        """
        One model's config block for this instance's frequency.

        Reads the parsed YAML from _default_params(), which is cached, so
        constructing a forecaster parses the file once rather than once
        per requested model.
        """
        full_config = _default_params()
        try:
            return full_config[model_name][self.frequency]
        except KeyError as e:
            raise InvalidConfigError(
                f"No config found for model '{model_name}' at frequency "
                f"'{self.frequency}' in {_CONFIG_PATH}.",
                param_name="models",
                provided_value=model_name,
            ) from e

    def _validate_train_size_against_model_floors(self) -> None:
        """
        Warn when the resolved training window is below what one or more
        of the requested models need in order to fit.

        The floor checked is each model's own `wrapper_config.
        min_rows_required` for this frequency, so it stays a property of
        the config rather than a constant duplicated here. Every model
        below its floor is named in a single warning; the rest run
        normally, per the library's contract that one model failing never
        costs you the others.

        This NEVER raises, even when no model can fit. The training window
        does not govern everything: `forecast()` refits on the full
        series, so a 16-row daily series with a 10-row training window can
        still produce a real forecast from a model whose floor is 14.
        Refusing here would block that to pre-empt an error `evaluate()`
        already reports perfectly well, per model, in `evaluate_errors`.
        """
        floors = {
            model_name: self._configs[model_name]["wrapper_config"]["min_rows_required"]
            for model_name in self.models
        }
        n_total = len(self.raw_df)
        split_description = (
            f"{self._n_train} training rows ({n_total} total - "
            f"{self._n_validation} validation - {self._n_test} test)"
        )

        too_small = {name: floor for name, floor in floors.items() if self._n_train < floor}
        if not too_small:
            return

        warnings.warn(
            f"The resolved split leaves {split_description}, which is below the "
            f"minimum for {sorted(too_small)} "
            f"({', '.join(f'{n} needs {f}' for n, f in sorted(too_small.items()))}). "
            f"Those models cannot be SELECTED on the validation window and will "
            f"carry no validation MAPE, so they are excluded from any weighted "
            f"ensemble and cannot be screened against mape_threshold. They may "
            f"still be evaluated: the train+validation refit sees more rows than "
            f"the training window alone, and forecast() refits on the full series "
            f"-- any that do fail are reported in evaluate_errors, and the "
            f"remaining models run normally either way. Use smaller "
            f"validation_size/test_size, drop models via models=, or supply a "
            f"longer series.",
            stacklevel=3,
        )

    def _reject_duplicate_dates(self, df: pd.DataFrame) -> None:
        """
        Raise if any date appears more than once.

        Every model treats a row as one period. A repeated date is two
        periods claiming the same day: Prophet and NeuralProphet refuse it
        outright (so they silently drop out of the run), SARIMAX and CatBoost
        count it as an extra period, and inside the test window it misaligns
        the models' predictions so evaluate() fails late with an unrelated
        error. There is no safe default -- summing, averaging or keeping one
        row are all plausible -- so the caller has to choose.
        """
        duplicates = find_duplicate_dates(df[self.date_column])
        if not duplicates:
            return
        shown = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in duplicates[:5]]
        more = f" and {len(duplicates) - 5} more" if len(duplicates) > 5 else ""
        raise InvalidConfigError(
            f"The date column '{self.date_column}' has {len(duplicates)} date(s) that appear "
            f"more than once: {shown}{more}. Each period must appear exactly once -- "
            f"aggregate the rows for each date (sum or mean) or remove the duplicates "
            f"before fitting.",
            param_name="date_column",
            provided_value=shown,
        )

    def _validate_date_spacing(self) -> None:
        """
        Enforce the date acceptance contract on the sorted, de-duplicated
        dates. Checked in this order, because each answers a narrower
        question than the one before:

          1. More than half of the steps are exactly one period. If not, the
             data is not at the declared frequency at all: either it is
             clearly the OTHER supported frequency (a mis-declared
             `frequency`), or a cadence Horizon does not support (every 2
             days, fortnightly, monthly). Previously both only warned, and
             the run went on to produce forecast dates, window sizes and
             model settings for the wrong frequency -- monthly data declared
             daily finished with a confident 0.67% MAPE and a daily-dated
             forecast.
          2. Every step is a whole number of periods -- weekly dates all on
             the same weekday. Otherwise the weekly cycle and the forecast
             dates are misaligned.
          3. No run of missing dates longer than MAX_CONTINUOUS_GAP_DAYS.

        What passes may still have scattered gaps. Those only warn: they are
        common in real data, and the models still run, but the warning says
        what they affect. Horizon never fills gaps itself.
        """
        dates = self.raw_df[self.date_column]
        if len(dates) < 2:
            return  # resolve_split() reports a series this short

        spacing = describe_date_spacing(dates, self.frequency)
        unit = "day" if self.frequency == "daily" else "week"

        if spacing.one_period_share <= 0.5:
            typical_days = spacing.typical_step / pd.Timedelta(days=1)
            looks_like = {1: "daily", 7: "weekly"}.get(typical_days)
            if looks_like is not None and looks_like != self.frequency:
                raise InvalidConfigError(
                    f"The dates in '{self.date_column}' are mostly {typical_days:g} day(s) "
                    f"apart, which is {looks_like} data, but frequency='{self.frequency}' was "
                    f"given. Use frequency='{looks_like}'.",
                    param_name="frequency",
                    provided_value=self.frequency,
                )
            raise UnsupportedFrequencyError(
                f"The dates in '{self.date_column}' are mostly {typical_days:g} day(s) apart, "
                f"and only {spacing.one_period_share:.0%} of them are one {unit} apart, so "
                f"this is not {self.frequency} data. Horizon supports daily or weekly data "
                f"only: resample yours to one of those first.",
                provided_frequency=f"{typical_days:g} days",
            )

        if spacing.irregular_steps:
            start, end = spacing.irregular_example
            hint = (
                " Weekly dates must all fall on the same weekday."
                if self.frequency == "weekly"
                else " Daily dates must be whole days apart (check for times of day)."
            )
            raise InvalidConfigError(
                f"{spacing.irregular_steps} step(s) in '{self.date_column}' are not a whole "
                f"number of {unit}s, e.g. {start:%Y-%m-%d %H:%M} to {end:%Y-%m-%d %H:%M} "
                f"({(end - start) / pd.Timedelta(days=1):g} days).{hint}",
                param_name="date_column",
                provided_value=[str(start), str(end)],
            )

        period_days = 1 if self.frequency == "daily" else 7
        longest_days = spacing.longest_gap_periods * period_days
        if longest_days > MAX_CONTINUOUS_GAP_DAYS:
            start, end = spacing.longest_gap
            raise InvalidConfigError(
                f"No data in '{self.date_column}' between {start:%Y-%m-%d} and "
                f"{end:%Y-%m-%d}: {spacing.longest_gap_periods} {unit}(s) missing in a row. "
                f"The limit is 4 weeks ({MAX_CONTINUOUS_GAP_DAYS} days) -- a longer hole "
                f"removes a whole stretch of trend and seasonality, and SARIMAX treats the "
                f"dates either side of it as neighbours. If the business was closed then, "
                f"add those dates with a value of 0; otherwise fill the gap, or start the "
                f"data after it.",
                param_name="date_column",
                provided_value=[str(start), str(end)],
            )

        if spacing.missing_periods:
            start, end = spacing.longest_gap
            warnings.warn(
                f"The date column '{self.date_column}' has gaps: {spacing.coverage:.0%} of "
                f"{unit}s are present, with {spacing.missing_periods} {unit}(s) missing across "
                f"{spacing.gap_count} gap(s) (longest: {spacing.longest_gap_periods} {unit}(s) "
                f"between {start:%Y-%m-%d} and {end:%Y-%m-%d}). The run continues, but expect "
                f"lower accuracy: the train/validation/test windows are counted in rows, so "
                f"they cover more time than their size suggests, and SARIMAX treats "
                f"consecutive rows as consecutive periods, so it is affected most. Horizon "
                f"does not fill gaps -- add the missing periods to your data if you can.",
                stacklevel=3,
            )

    def _drivers_that_vary_in(
        self, window: str, fit_rows: pd.DataFrame, score_rows: pd.DataFrame
    ) -> List[str]:
        """
        The declared drivers that take more than one value in a fit window,
        computed once per window (every model shares the windows) and
        warning once about the rest.

        Constant-everywhere drivers are dropped at construction, but a
        driver can still be constant inside ONE fit window: a price change
        or a new promotion that first appears in the validation or test
        window is all one value in the rows the model is fitted on. No model
        can learn an effect from such a column, and they failed in different
        ways when given one: NeuralProphet discards it at fit and then
        rejects it at prediction (so it dropped out of the run entirely),
        and Prophet -- when the constant value is not 0 -- fits an arbitrary
        effect to it (+31,403 on a series around 3,500, where the true
        effect was -400), while CatBoost, SARIMAX and Bayesian PyMC gave it
        no effect. Leaving it out of that fit for every model gives all of
        them the honest answer -- no effect yet -- and keeps the models
        comparable. The change still lands inside the scoring window, so
        the warning says which score includes an effect no model could see.
        """
        if window in self._window_drivers:
            return self._window_drivers[window]

        constant = [c for c in self.independent_columns if fit_rows[c].nunique(dropna=False) <= 1]
        drivers = [c for c in self.independent_columns if c not in constant]
        self._window_drivers[window] = drivers

        # Warn only where it matters for a score: a driver whose first
        # change lands in this window's scoring rows. One that only changes
        # later (in the test window, for the training fit) leaves this score
        # untouched and is reported by the train+validation window instead.
        affected, names = [], []
        for col in constant:
            value = fit_rows[col].iloc[0]
            changed = score_rows[score_rows[col] != value]
            if len(changed):
                names.append(f"'{col}'")
                affected.append(
                    f"'{col}' is always {value:g} there and first changes on "
                    f"{changed[self.date_column].iloc[0]:%Y-%m-%d}"
                )
        if affected:
            one = len(affected) == 1
            if window == "train":
                where = "the training window used for model selection"
                score = (
                    "the validation MAPE -- which picks the hyperparameters, is screened "
                    "against mape_threshold and weights the ensemble -- includes an effect no "
                    f"model could see. {'It is' if one else 'They are'} used in every later "
                    f"fit where {'it changes' if one else 'they change'}"
                )
            else:
                where = "the train+validation window used for the test score"
                score = (
                    "the test MAPE includes an effect no model could see. forecast() refits "
                    f"on the full series, where {'it changes' if one else 'they change'}, "
                    f"and uses {'it' if one else 'them'}"
                )
            start = fit_rows[self.date_column].iloc[0]
            end = fit_rows[self.date_column].iloc[-1]
            warnings.warn(
                f"{', '.join(names)} {'never changes' if one else 'never change'} in {where} "
                f"({start:%Y-%m-%d} to {end:%Y-%m-%d}): {'; '.join(affected)}. No model can "
                f"learn an effect from a column that never changes, so "
                f"{'it is' if one else 'they are'} left out of this fit for every model, "
                f"and {score}.",
                stacklevel=3,
            )
        return drivers

    def _adapt_window(
        self, model_name: str, fit_rows: pd.DataFrame, score_rows: pd.DataFrame, window: str
    ):
        """
        Shape one (fit window, scoring window) pair for ONE model, using
        that model's own adapter class. Drivers that never change in the
        fit window are left out (see _drivers_that_vary_in).

        Returns (adapted_fit, adapted_score_features, y_score).

        Critical detail: the scoring window is shaped via
        adapter.transform_future(), NOT adapter.transform_train().
        Using transform_train() on the scoring window alone happened to
        work for ProphetAdapter,
        which is a stateless pure function -- but it would be actively
        WRONG for an adapter that carries state across the train/future
        boundary like CatBoostAdapter: transform_train() resets the
        adapter's retained self._reference_date to the scoring window's
        own first date, which would restart the trend index (time_idx) at
        0 instead of letting it continue seamlessly from where the fit
        window left off -- silently corrupting the very feature the
        redesign added, and any subsequent use of the adapter.
        transform_future() is the correct call here -- it's exactly what a
        genuine future forecast would do, reusing the reference_date
        established by the fit. The only difference from a real future
        forecast is that the actual y values for this "future" period ARE
        already known (it's held-out historical data, being used for
        scoring) -- so they're read directly from raw_df, never routed
        through the adapter, which by design never sees them.

        A FRESH adapter is built per call. This matters now that there are
        two fit windows (train, and train+validation): a stateful adapter
        fitted on train carries standardisation constants and a reference
        date derived from train alone, and reusing it for the train+
        validation refit would leave the refit describing its own data
        with the wrong constants.
        """
        drivers = self._drivers_that_vary_in(window, fit_rows, score_rows)
        AdapterClass = MODEL_REGISTRY[model_name]["adapter"]
        adapter = AdapterClass(
            date_column=self.date_column,
            target_column=self.target_column,
            independent_columns=drivers,
            frequency=self.frequency,
        )
        adapted_fit = adapter.transform_train(fit_rows)

        future_dates = pd.Series(score_rows[self.date_column].reset_index(drop=True))
        future_independent_data = score_rows[drivers].reset_index(drop=True) if drivers else None
        adapted_score_features = adapter.transform_future(future_dates, future_independent_data)
        y_score = score_rows[self.target_column].to_numpy()

        return adapted_fit, adapted_score_features, y_score

    def _prepare_split(self, model_name: str):
        """
        The SELECTION split for one model: fit on train, score on
        validation. Cached per model per instance, since the split doesn't
        depend on hyperparameters -- every grid candidate reuses the same
        adapted data.

        Returns (adapted_train, adapted_validation_features, y_validation).
        """
        if model_name not in self._split_cache:
            self._split_cache[model_name] = self._adapt_window(
                model_name,
                fit_rows=self.raw_df.iloc[: self._n_train],
                score_rows=self.raw_df.iloc[self._n_train : self._n_train + self._n_validation],
                window="train",
            )
        return self._split_cache[model_name]

    def _prepare_refit(self, model_name: str):
        """
        The REPORTING split for one model: fit on train + validation, score
        on test. Cached per model per instance.

        Returning to the data once more before scoring test is deliberate.
        The alternative -- fitting on train alone and predicting straight
        across validation and test in one span -- is cheaper by one fit per
        model, but it measures a model trained on less data than would ever
        be deployed. Refitting on everything available before the test
        window makes the reported MAPE an estimate of the same procedure
        that produces the real forecast (which refits on the full frame),
        which is the number a user actually wants.

        Returns (adapted_train_validation, adapted_test_features, y_test).
        """
        if model_name not in self._refit_cache:
            split_point = self._n_train + self._n_validation
            self._refit_cache[model_name] = self._adapt_window(
                model_name,
                fit_rows=self.raw_df.iloc[:split_point],
                score_rows=self.raw_df.iloc[split_point:],
                window="train+validation",
            )
        return self._refit_cache[model_name]

    def _fit_predict_mape(
        self, model_name: str, params: dict, adapted_fit, adapted_score_features, y_score
    ):
        """Fit one model with one hyperparameter set, predict the scoring window, score it."""
        WrapperClass = MODEL_REGISTRY[model_name]["wrapper"]
        wrapper_config = self._configs[model_name]["wrapper_config"]
        model = WrapperClass(
            frequency=self.frequency,
            params=params,
            min_rows_required=wrapper_config["min_rows_required"],
            random_seed=wrapper_config["random_seed"],
        )
        model.fit(adapted_fit, target_column="y")

        forecast_df = model.predict(
            horizon=len(adapted_score_features), future_exog=adapted_score_features
        )
        mape_result = mape(y_score, forecast_df["yhat"].values)
        return model, forecast_df, mape_result

    def _build_test_predictions(self, adapted_test_features, y_test, forecast_df) -> pd.DataFrame:
        test_predictions = pd.DataFrame(
            {
                "ds": adapted_test_features["ds"].reset_index(drop=True),
                "y": y_test,
                "yhat": forecast_df["yhat"].values,
            }
        )
        if "yhat_lower" in forecast_df.columns:
            test_predictions["yhat_lower"] = forecast_df["yhat_lower"].values
            test_predictions["yhat_upper"] = forecast_df["yhat_upper"].values
        return test_predictions

    def _run_hyperparameter_grid_search(self, model_name: str) -> _Selection:
        """
        Score every hyperparameter candidate on the VALIDATION window and
        keep the winner. Nothing in here ever touches the test window --
        that separation is the entire reason the three-way split exists.
        """
        adapted_train, adapted_validation_features, y_validation = self._prepare_split(model_name)
        config = self._configs[model_name]
        base = config.get("base_hyperparameters", {})

        # A candidate that fails is SKIPPED, not fatal: one unusable
        # hyperparameter combination should not cost you the whole model when
        # others fit fine. Only if every candidate fails does the model itself
        # fail (and then it lands in evaluate_errors like any other failure,
        # leaving the remaining models untouched). This matters in practice --
        # e.g. a SARIMAX order whose forecast diverges, or an invalid CatBoost
        # depth -- and applies to every model, not just the one that surfaced it.
        candidates = []
        failures: Dict[str, str] = {}
        failure_exceptions: List[HorizonForecasterError] = []
        for grid_entry in config["hyperparameter_grid"]:
            full_params = {**base, **grid_entry}
            try:
                model, forecast_df, mape_result = self._fit_predict_mape(
                    model_name,
                    full_params,
                    adapted_train,
                    adapted_validation_features,
                    y_validation,
                )
            except HorizonForecasterError as e:
                failures[str(grid_entry)] = f"{type(e).__name__}: {e}"
                failure_exceptions.append(e)
                continue
            candidates.append((full_params, model, forecast_df, mape_result))

        if not candidates and all(isinstance(e, InsufficientDataError) for e in failure_exceptions):
            # Every candidate failed because the TRAINING WINDOW is below
            # this model's row floor. That is not a hyperparameter problem
            # -- no choice of parameters would have helped -- and, crucially,
            # it does not mean the model cannot be fitted at all: both later
            # fits see more data. The train+validation refit is longer, and
            # forecast() refits on the entire series. A 20-row daily series
            # splits to a 12-row training window, under Prophet's floor of
            # 14, while the 20 rows forecast() actually fits on clear it
            # comfortably.
            #
            # So fall back to the first grid entry rather than failing the
            # model outright. That entry is not arbitrary: per this config's
            # own header, the leading candidates are the settings
            # validated on real data, which makes them the right thing to
            # reach for when there is no evidence to choose on. The
            # validation MAPE is recorded as NaN,
            # which is the honest description -- nothing was validated --
            # and downstream that means the trimmed rule still uses the
            # model while the weighted rules exclude it.
            #
            # Deliberately narrow to InsufficientDataError. A convergence
            # failure across every candidate is a real modelling failure and
            # still raises below; only "the window is too small" is treated
            # as a reason to defer to a default.
            fallback_params = {**base, **config["hyperparameter_grid"][0]}
            warnings.warn(
                f"{model_name}: every hyperparameter candidate failed on the "
                f"{self._n_train}-row training window because the model needs more "
                f"rows than that, so none could be scored. Falling back to the first "
                f"configured candidate without selecting it. Its later fits see more "
                f"data (train+validation for evaluate(), the full series for "
                f"forecast()) and may well succeed, but this model carries no "
                f"validation MAPE -- it will be excluded from any weighted ensemble "
                f"and cannot be screened against mape_threshold. Supply a longer "
                f"series, or smaller validation_size/test_size, to select properly.",
                stacklevel=3,
            )
            self._selected_hyperparameters[model_name] = fallback_params
            return _Selection(
                params=fallback_params,
                validation_mape=MapeResult(
                    value=float("nan"), excluded_zero_actual_count=0, total_count=0
                ),
                grid_search_summary=None,
            )

        if not candidates:
            # When every candidate failed the SAME way, the cause is not
            # hyperparameter-specific -- it's a property of the data or the
            # model (a missing column, a systematically unusable series), and
            # it would be wrong to relabel it. Re-raise the original
            # exception so its real type and structured attributes survive,
            # since callers (including evaluate_errors consumers) reasonably
            # branch on type. Only genuinely mixed failures become the aggregate
            # ModelConvergenceError, where no single type is representative.
            distinct_types = {type(e) for e in failure_exceptions}
            if len(distinct_types) == 1:
                raise failure_exceptions[0]
            raise ModelConvergenceError(
                f"All {len(config['hyperparameter_grid'])} hyperparameter candidates "
                f"failed for {model_name}: {failures}",
                model_name=model_name,
            )

        if failures:
            warnings.warn(
                f"{model_name}: {len(failures)} of "
                f"{len(config['hyperparameter_grid'])} hyperparameter candidates failed "
                f"and were skipped; the best of the {len(candidates)} that succeeded was "
                f"used. Skipped: {failures}",
                stacklevel=3,
            )

        def sort_key(candidate):
            val = candidate[3].value
            return float("inf") if val != val else val

        candidates.sort(key=sort_key)
        best_params, _best_model, _best_forecast_df, best_mape = candidates[0]

        # The candidates' own fitted models are deliberately discarded. Each
        # was fitted on TRAIN only, and every window that gets reported is
        # produced by a later refit on train+validation -- keeping the
        # train-only fits around would just invite something downstream to
        # report from the wrong model.
        grid_search_summary = [{"params": p, "mape": m.value} for p, _, _, m in candidates]

        self._selected_hyperparameters[model_name] = best_params
        return _Selection(
            params=best_params,
            validation_mape=best_mape,
            grid_search_summary=grid_search_summary,
        )

    def _select(self, model_name: str) -> _Selection:
        """
        Resolve one model's hyperparameters and its validation MAPE,
        caching the result.

        This is the ONLY path by which any selection decision is made, and
        it reads the validation window exclusively. Both evaluate() (which
        goes on to refit and report a test score) and forecast() (which
        needs the winning params, plus the validation MAPE for ensemble
        screening and weighting) come through here, so a session never pays
        for selection twice.

        A model given explicit `model_params` skips the grid, but is still
        fitted once on train and scored on validation -- the ensemble needs
        a validation MAPE for it either way, and inventing one from the
        test window would reintroduce exactly the leak the three-way split
        removes.
        """
        if model_name in self._selection_cache:
            return self._selection_cache[model_name]

        if self._use_grid_search[model_name]:
            selection = self._run_hyperparameter_grid_search(model_name)
        else:
            explicit_params = self._selected_hyperparameters[model_name]
            try:
                adapted_train, adapted_validation_features, y_validation = self._prepare_split(
                    model_name
                )
                _model, _forecast_df, validation_mape = self._fit_predict_mape(
                    model_name,
                    explicit_params,
                    adapted_train,
                    adapted_validation_features,
                    y_validation,
                )
            except HorizonForecasterError as e:
                # With explicit params there is nothing to SELECT -- this fit
                # exists only to produce a validation score. If it fails (a
                # short series whose training window is below the model's
                # floor, say), the params are still perfectly usable, and
                # forecast() refits on the full frame where the model may fit
                # fine. So record an undefined score rather than failing the
                # model outright: the trimmed rule needs no MAPE and will
                # still use it, while the weighted rules drop it explicitly.
                # A grid search cannot degrade this way -- with no candidate
                # scores there is no basis for choosing one, so it still
                # raises.
                warnings.warn(
                    f"{model_name} could not be scored on the validation window "
                    f"({type(e).__name__}: {e}). Its explicit model_params are still "
                    f"used, so forecast() can proceed, but it has no validation MAPE "
                    f"-- it will be excluded from any weighted ensemble and cannot be "
                    f"screened against mape_threshold.",
                    stacklevel=3,
                )
                validation_mape = MapeResult(
                    value=float("nan"), excluded_zero_actual_count=0, total_count=0
                )
            selection = _Selection(
                params=explicit_params,
                validation_mape=validation_mape,
                grid_search_summary=None,
            )

        self._selection_cache[model_name] = selection
        return selection

    def _evaluate_single_model(self, model_name: str) -> EvaluationResult:
        """
        Select on validation, refit the winner on train+validation, then
        predict and score the test window.
        """
        if model_name in self._evaluation_cache:
            return self._evaluation_cache[model_name]

        selection = self._select(model_name)

        adapted_train_validation, adapted_test_features, y_test = self._prepare_refit(model_name)
        model, forecast_df, test_mape = self._fit_predict_mape(
            model_name,
            selection.params,
            adapted_train_validation,
            adapted_test_features,
            y_test,
        )
        test_predictions = self._build_test_predictions(adapted_test_features, y_test, forecast_df)

        result = EvaluationResult(
            model_name=model.model_name,
            mape=test_mape,
            validation_mape=selection.validation_mape,
            n_train=self._n_train,
            n_validation=self._n_validation,
            n_test=self._n_test,
            test_predictions=test_predictions,
            fitted_model=model,
            grid_search_summary=selection.grid_search_summary,
        )
        self._evaluation_cache[model_name] = result
        return result

    def evaluate(self) -> Dict[str, EvaluationResult]:
        """
        Runs evaluate() independently for every requested model.
        Always returns a dict keyed by model name (even for a single
        model). See class docstring for partial-failure behavior.

        Each model is selected on the validation window, refit on train +
        validation, then scored on the test window -- so the MAPE on every
        returned EvaluationResult is an out-of-sample estimate, and the
        selection score it was chosen by is kept alongside it as
        `validation_mape`.

        Also populates self.evaluate_ensemble (a dict of EnsembleResult
        keyed by combination method) as a side effect -- the returned dict
        itself is only the per-model results.

        Raises:
            AllModelsRejectedError: if every requested model failed.
                (Note: this is about fit/predict failing outright: a
                model that fits fine but scores outside the ensemble's
                MAPE threshold does NOT raise here -- see
                evaluate_ensemble[method].rejected_models instead, since
                exclusion is decided per combination method.)
        """
        results: Dict[str, EvaluationResult] = {}
        errors: Dict[str, HorizonForecasterError] = {}

        for model_name in self.models:
            try:
                results[model_name] = self._evaluate_single_model(model_name)
            except HorizonForecasterError as e:
                errors[model_name] = e
            except Exception as e:  # noqa: BLE001
                # A model raising an UNEXPECTED (non-HorizonForecaster) error --
                # e.g. a raw numpy LinAlgError from bad input -- must not abort
                # the whole run. Isolate it exactly like a typed failure, but
                # preserve the real type/message (via original_exception) so it
                # stays debuggable rather than silently swallowed.
                errors[model_name] = ModelConvergenceError(
                    f"{model_name} failed with an unexpected " f"{type(e).__name__}: {e}",
                    model_name=model_name,
                    original_exception=e,
                )

        self.evaluate_errors = errors

        if not results:
            self.evaluate_ensemble = None
            error_summary = {k: str(v) for k, v in errors.items()}
            raise AllModelsRejectedError(
                f"All requested models failed during evaluate(): {error_summary}",
                rejected_models=list(errors.keys()),
            )

        # Screening and weighting read the VALIDATION MAPE; the ensemble's
        # own reported score is computed on the TEST window, against test
        # actuals. Mixing those up is exactly the leak the three-way split
        # exists to close, so they are passed as separate arguments.
        first_result = next(iter(results.values()))
        self.evaluate_ensemble = build_ensembles(
            methods=self.ensemble_methods,
            per_model_selection_mape={name: r.validation_mape.value for name, r in results.items()},
            per_model_predictions={name: r.test_predictions for name, r in results.items()},
            mape_threshold=self.mape_threshold,
            actuals=first_result.test_predictions[["ds", "y"]],
        )

        return results

    def _future_dates(self, horizon: int) -> pd.Series:
        """The horizon's future dates, one step past the last observed date."""
        last_date = self.raw_df[self.date_column].max()
        offset = _FREQUENCY_OFFSETS[self.frequency]
        return pd.Series(
            pd.date_range(
                start=pd.to_datetime(last_date) + pd.tseries.frequencies.to_offset(offset),
                periods=horizon,
                freq=offset,
            )
        )

    def _forecast_single_model(
        self,
        model_name: str,
        horizon: int,
        future_independent_data: Optional[pd.DataFrame],
        future_dates: pd.Series,
        used_default_strategy: bool,
        future_exog_used: Optional[pd.DataFrame],
    ) -> ForecastResult:
        AdapterClass = MODEL_REGISTRY[model_name]["adapter"]

        if model_name not in self._forecast_models:
            # Resolve hyperparameters on the validation window if that
            # hasn't happened yet this session. Cached, so a model already
            # selected by evaluate() is not re-searched here.
            selected_params = self._select(model_name).params

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
                params=selected_params,
                min_rows_required=wrapper_config["min_rows_required"],
                random_seed=wrapper_config["random_seed"],
            )
            model.fit(adapted_full, target_column="y")

            self._forecast_models[model_name] = model
            self._forecast_adapters[model_name] = adapter

        adapter = self._forecast_adapters[model_name]

        # future_independent_data and future_dates are resolved ONCE, model-
        # agnostically, in forecast() (see _resolve_future_exog there) and
        # passed in, so every model sees the same assumed future drivers.
        future_exog = adapter.transform_future(future_dates, future_independent_data)
        forecast_df = self._forecast_models[model_name].predict(
            horizon=horizon, future_exog=future_exog
        )

        return ForecastResult(
            model_name=self._forecast_models[model_name].model_name,
            forecast=forecast_df,
            horizon=horizon,
            used_default_future_strategy=used_default_strategy,
            future_exog_used=future_exog_used,
        )

    def _warn_on_forecasts_far_outside_history(self, results: Dict[str, ForecastResult]) -> None:
        """
        Flag any model whose forecast departs implausibly far from every
        value ever observed in the target.

        A trending series legitimately forecasts past its historical
        maximum, so this deliberately does NOT warn on merely leaving the
        historical range -- only on exceeding it by more than
        _FORECAST_RANGE_TOLERANCE. This is model-agnostic: it catches an
        over-forecast regardless of which model produced it, including
        ones driven by auto-generated future independent values.

        The two sides are checked independently -- a wildly oscillating
        forecast can breach both, and reporting only the high side would
        hide half of what went wrong. Note the low-side bound is only
        defined for a strictly positive historical minimum: a series
        containing a zero (sparse conversions, installs) has no
        meaningful "this far below" multiple, so that check is skipped.
        """
        actuals = pd.to_numeric(self.raw_df[self.target_column], errors="coerce").dropna()
        if actuals.empty:
            return

        hist_min, hist_max = float(actuals.min()), float(actuals.max())
        upper = hist_max * (1 + _FORECAST_RANGE_TOLERANCE) if hist_max > 0 else None
        lower = hist_min * (1 - _FORECAST_RANGE_TOLERANCE) if hist_min > 0 else None
        pct = int(_FORECAST_RANGE_TOLERANCE * 100)

        for model_name, result in results.items():
            yhat = pd.to_numeric(result.forecast["yhat"], errors="coerce").dropna()
            if yhat.empty:
                continue

            if upper is not None and float(yhat.max()) > upper:
                warnings.warn(
                    f"{model_name}: forecast peaks at {float(yhat.max()):,.0f}, more than "
                    f"{pct}% above the highest value ever observed ({hist_max:,.0f}). "
                    f"Sanity-check the inputs before trusting this -- especially any "
                    f"auto-generated future independent values, and whether the horizon "
                    f"is too long for the available history.",
                    stacklevel=3,
                )
            if lower is not None and float(yhat.min()) < lower:
                warnings.warn(
                    f"{model_name}: forecast bottoms at {float(yhat.min()):,.0f}, more than "
                    f"{pct}% below the lowest value ever observed ({hist_min:,.0f}). "
                    f"Sanity-check the inputs before trusting this.",
                    stacklevel=3,
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

        Also populates self.forecast_ensemble (a dict of EnsembleResult
        keyed by combination method) as a side effect.

        forecast() has no held-out actuals to score accuracy against, so
        any screening or weighting its ensemble needs comes from the
        validation MAPE established when each model's hyperparameters
        were selected. That selection has to happen anyway -- forecast()
        cannot fit without knowing which parameters won -- so the scores
        come for free and are cached, whether or not evaluate() was ever
        called. (Before 1.0 this method ran the whole of evaluate()
        internally to obtain them, which additionally paid for a test-
        window prediction it then discarded.)

        Args:
            horizon: number of future periods to forecast.
            future_independent_data: optional DataFrame of real future
                values for this instance's independent_columns: one row
                per forecast period, in order. It may hold just some of
                the drivers (the rest are generated via
                future_exog_strategy) and, optionally, the date column or
                a DatetimeIndex, which must then match the forecast dates
                exactly. If None and independent_columns were declared,
                every driver is generated. See
                future_exog.check_future_table for the full contract.

        Raises:
            InvalidConfigError: if horizon is not positive, or the future
                table breaks the contract in check_future_table.
            AllModelsRejectedError: if every requested model failed.
        """
        # Clear the previous call's results first: if this call stops on a
        # bad horizon or future table, these must not still describe the
        # previous forecast as if it were this one.
        self.forecast_ensemble = None
        self.forecast_errors = {}
        self.future_exog_used = None

        # A whole number, checked before anything else: a float such as 2.5
        # used to build two future dates and then fail inside every model as
        # a row-count mismatch, ending in an AllModelsRejectedError that hid
        # the real cause. A whole float (30.0) always worked and still does.
        # bool is excluded because it is an int subclass.
        is_whole = (
            not isinstance(horizon, bool)
            and isinstance(horizon, numbers.Real)
            and float(horizon).is_integer()
        )
        if not is_whole or horizon <= 0:
            raise InvalidConfigError(
                f"horizon must be a positive integer, got {horizon!r}.",
                param_name="horizon",
                provided_value=horizon,
            )
        horizon = int(horizon)

        n_total = len(self.raw_df)
        if horizon > _HORIZON_WARNING_FRACTION * n_total:
            warnings.warn(
                f"Forecasting {horizon} periods from only {n_total} periods of history. "
                f"Forecasts this far beyond the available history should be treated with "
                f"caution — uncertainty intervals may understate the true risk of "
                f"extrapolating this far.",
                stacklevel=2,
            )

        # Resolve the future independent values ONCE, model-agnostically, so
        # every model sees the same assumed future drivers. Partial fill: any
        # column the caller supplied is used as-is; the rest are generated via
        # future_exog_strategy. Exposed as self.future_exog_used.
        future_dates = self._future_dates(horizon)
        if future_independent_data is not None:
            future_independent_data = check_future_table(
                future_independent_data,
                history_df=self.raw_df,
                date_column=self.date_column,
                target_column=self.target_column,
                independent_columns=self.independent_columns,
                declared_columns=self._declared_independent_columns,
                event_columns=self.event_columns,
                future_dates=future_dates,
                frequency=self.frequency,
                date_format=self.date_format,
                models=self.models,
            )
        resolved_exog, self.future_exog_used, used_default, generated = self._resolve_future_exog(
            horizon, future_dates, future_independent_data
        )
        if generated:
            warnings.warn(
                f"future values for {generated} were auto-generated for all {horizon} "
                f"periods via the '{self.future_exog_strategy}' strategy (no values were "
                f"supplied for them). The forecast is only as reliable as those assumed "
                f"inputs -- for an exogenous-variable-driven model this can dominate the "
                f"result. Supply real planned values if you have them (you can supply "
                f"just some columns; the rest are filled), inspect what was assumed via "
                f"forecaster.future_exog_used, and treat this output as a scenario rather "
                f"than a prediction.",
                stacklevel=2,
            )

        if generated:
            for message in describe_generation_risks(
                history_df=self.raw_df,
                date_column=self.date_column,
                independent_columns=self.independent_columns,
                generated=generated,
                future_dates=future_dates,
                frequency=self.frequency,
                strategy=self.future_exog_strategy,
                event_columns=self.event_columns,
            ):
                warnings.warn(message, stacklevel=2)

        results: Dict[str, ForecastResult] = {}
        errors: Dict[str, HorizonForecasterError] = {}

        for model_name in self.models:
            try:
                results[model_name] = self._forecast_single_model(
                    model_name,
                    horizon,
                    resolved_exog,
                    future_dates,
                    used_default,
                    self.future_exog_used,
                )
            except HorizonForecasterError as e:
                errors[model_name] = e
            except Exception as e:  # noqa: BLE001 -- isolate any model failure; see evaluate()
                errors[model_name] = ModelConvergenceError(
                    f"{model_name} failed with an unexpected " f"{type(e).__name__}: {e}",
                    model_name=model_name,
                    original_exception=e,
                )

        self.forecast_errors = errors
        self._warn_on_forecasts_far_outside_history(results)

        if not results:
            self.forecast_ensemble = None
            error_summary = {k: str(v) for k, v in errors.items()}
            raise AllModelsRejectedError(
                f"All requested models failed during forecast(): {error_summary}",
                rejected_models=list(errors.keys()),
            )

        # Screening and weighting reuse the validation MAPE established
        # during selection -- forecast() has no actuals of its own, and
        # every model that got here has been selected, so the scores are
        # already cached. `actuals=None` because no ground truth for the
        # future exists yet, so no ensemble MAPE is computed.
        selection_mape = {
            name: self._selection_cache[name].validation_mape.value for name in results
        }
        self.forecast_ensemble = build_ensembles(
            methods=self.ensemble_methods,
            per_model_selection_mape=selection_mape,
            per_model_predictions={name: results[name].forecast for name in selection_mape},
            mape_threshold=self.mape_threshold,
            actuals=None,
        )

        return results

    def _resolve_future_exog(self, horizon, future_dates, supplied):
        """
        Resolve the future independent values for this forecast, once and
        model-agnostically. Delegates to future_exog.resolve_future_exog, which
        honours PARTIAL FILL (supplied columns kept, the rest generated via
        self.future_exog_strategy) and type-aware flag handling.

        Returns (models_df, used_df, used_default, generated):
            models_df    the independent-columns DataFrame fed to each adapter
                         (None when there are no independent columns).
            used_df      the same, with a leading "ds" column, stored on
                         self.future_exog_used for inspection.
            used_default True if any column was generated rather than supplied.
            generated    the list of generated (vs supplied) columns.
        """
        if not self.independent_columns:
            return None, None, False, []

        models_df, generated = resolve_future_exog(
            history_df=self.raw_df,
            date_column=self.date_column,
            independent_columns=self.independent_columns,
            future_dates=future_dates,
            frequency=self.frequency,
            strategy=self.future_exog_strategy,
            aggregator=self.future_exog_aggregator,
            window=self.recent_average_window,
            event_columns=self.event_columns,
            supplied=supplied,
        )
        used_df = models_df.copy()
        used_df.insert(0, "ds", pd.Series(future_dates).reset_index(drop=True))
        return models_df, used_df, bool(generated), generated
