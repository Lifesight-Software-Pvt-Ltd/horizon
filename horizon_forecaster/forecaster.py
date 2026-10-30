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

import difflib
import numbers
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import yaml

from horizon_forecaster.data_adapter.base_adapter import BaseAdapter
from horizon_forecaster.data_adapter.bayesianpymc_adapter import BayesianPyMCAdapter
from horizon_forecaster.data_adapter.catboost_adapter import CatBoostAdapter
from horizon_forecaster.data_adapter.neuralprophet_adapter import NeuralProphetAdapter
from horizon_forecaster.data_adapter.prophet_adapter import ProphetAdapter
from horizon_forecaster.data_adapter.sarimax_adapter import SARIMAXAdapter
from horizon_forecaster.evaluation.bands import BandCalibration, band, calibrate
from horizon_forecaster.evaluation.ensemble import (
    DEFAULT_ENSEMBLE_METHOD,
    ENSEMBLE_METHODS,
    EnsembleResult,
    build_ensemble,
    build_ensembles,
)
from horizon_forecaster.evaluation.metrics import (
    MapeResult,
    error_correlation,
    mape,
    seasonal_naive,
    wape,
)
from horizon_forecaster.future_exog import (
    check_future_table,
    describe_generation_risks,
    AGGREGATORS,
    FUTURE_EXOG_STRATEGIES,
    YEAR_PERIODS,
    last_year_source_dates,
    resolve_future_exog,
)
from horizon_forecaster.exceptions import (
    AllModelsRejectedError,
    HorizonForecasterError,
    InsufficientDataError,
    InvalidConfigError,
    ModelConvergenceError,
    ModelPredictionError,
    UnsupportedFrequencyError,
)
from horizon_forecaster.models.base import BaseModel
from horizon_forecaster.models.bayesianpymc_wrapper import (
    ANNUAL_MIN_SPAN_DAYS,
    CONVERGENCE_MAX_RHAT,
    CONVERGENCE_MIN_ESS,
    BayesianPyMCWrapper,
    span_days,
)
from horizon_forecaster.models.catboost_wrapper import CatBoostWrapper
from horizon_forecaster.models.neuralprophet_wrapper import NeuralProphetWrapper
from horizon_forecaster.models.prophet_wrapper import ProphetWrapper
from horizon_forecaster.models.sarimax_wrapper import SARIMAXWrapper, unexpected_zero_mask
from horizon_forecaster.utils import (
    as_name_list,
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

    Cached so a default five-model construction does not parse the same
    ~600-line file five times. The file ships inside the package and never
    changes at runtime, so caching is safe.

    Callers must treat the returned dict as READ-ONLY -- it is shared by
    every HorizonForecaster in the process. Nothing mutates it:
    _load_config only indexes into it, and the per-candidate merges in
    the grid search build new dicts via {**base, **entry}.
    """
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


# Warn once the requested horizon exceeds this fraction of the available
# history. Deliberately strict: a rule that fired only above TWICE the
# entire history was tried and let 100 weeks from 121 weeks of data pass
# silently -- exactly the case that produced an implausible real-world
# forecast. Forecasting more than half your history ahead is already a
# stretch and deserves a flag.
_HORIZON_WARNING_FRACTION = 0.5

# How a run is scored when deciding. MAPE is the default metric; "auto"
# keeps it unless the target makes MAPE meaningless, see
# HorizonForecaster._resolve_selection_metric().
SELECTION_METRICS = ("auto", "mape", "wape")
# "auto" switches to WAPE when at least this share of the target is 0
# (MAPE skips every zero actual) or when the target has any negative value.
_WAPE_ZERO_SHARE = 0.10
# At or above this share of zero targets the data is refused: on the series
# tested, every model then did worse than forecasting 0 every day.
_SPARSE_TARGET_ZERO_SHARE = 0.50

# Above this share of zero weeks (where the series is normally above 0),
# SARIMAX is left out of a weekly run. Its fit leaves such weeks out, but on
# weekly data that holds up less well than on daily: across 7 synthetic and
# 6 real weekly series with zero weeks set at random, SARIMAX became
# unusable in a roughly constant share of cases up to 10% and in most of
# them from 15% (on real data: test WAPE 425 against 48, or no forecast).
_SARIMAX_MAX_ZERO_WEEK_SHARE = 0.10

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


def _check_raw_df(raw_df) -> None:
    """raw_df must be a DataFrame with one row of plain column names."""
    if not isinstance(raw_df, pd.DataFrame):
        raise InvalidConfigError(
            f"raw_df must be a pandas DataFrame, got {type(raw_df).__name__}. Wrap it first, "
            f"e.g. pd.DataFrame(...), with the date, target and driver columns named.",
            param_name="raw_df",
            provided_value=type(raw_df).__name__,
        )
    if isinstance(raw_df.columns, pd.MultiIndex):
        raise InvalidConfigError(
            f"raw_df has {raw_df.columns.nlevels} levels of column headers. Flatten them to "
            f"one name per column first, e.g. df.columns = ['_'.join(map(str, c)) for c in "
            f"df.columns].",
            param_name="raw_df",
            provided_value=raw_df.columns.nlevels,
        )
    if raw_df.empty and len(raw_df.columns):
        raise InvalidConfigError(
            "raw_df has no rows.",
            param_name="raw_df",
            provided_value=0,
        )


def _parse_target(values: pd.Series, target_column, dates: Optional[pd.Series]) -> pd.Series:
    """
    Read the target like a driver (numbers written as text become numbers,
    booleans 1/0), but word any failure for a target: the driver messages
    talk about on/off flags and categories, which a target is not.
    """
    try:
        return parse_driver_column(values, target_column, dates=dates)
    except InvalidConfigError as e:
        if not pd.api.types.is_object_dtype(values) and not pd.api.types.is_string_dtype(values):
            raise  # complex numbers, dates: the driver message already fits
        text = pd.Series(values).reset_index(drop=True)
        as_text = text.map(lambda v: v if pd.isna(v) else str(v).strip())
        bad = as_text.notna() & (as_text != "") & pd.to_numeric(as_text, errors="coerce").isna()
        if not bad.any():
            raise
        examples = list(dict.fromkeys(as_text[bad]))[:3]
        where = ""
        if dates is not None:
            when = pd.Series(dates).reset_index(drop=True)[bad].iloc[0]
            if not pd.isna(when):
                where = f" (first on {pd.Timestamp(when):%Y-%m-%d})"
        raise InvalidConfigError(
            f"The target column '{target_column}' has values that aren't numbers, e.g. "
            f"{examples}{where}. Remove thousands separators, currency signs and other text "
            f"before passing the data.",
            param_name="target_column",
            provided_value=examples,
        ) from e


def _check_column_roles(
    raw_df: pd.DataFrame, date_column, target_column, independent_columns: List
) -> None:
    """
    Each used column plays one role, is named once in independent_columns,
    and appears once in raw_df. Unchecked, each of these runs without
    complaint: the date as the target, or the target as its own driver,
    scores a MAPE of 0.0; the date as a driver gave SARIMAX 99.97%; a driver
    listed twice makes every model fail with a raw ValueError.
    """
    if target_column == date_column:
        raise InvalidConfigError(
            f"target_column and date_column are both '{target_column}'. The target is the "
            f"numeric column to forecast; the date column holds each row's date.",
            param_name="target_column",
            provided_value=target_column,
        )
    for role, name in (("target_column", target_column), ("date_column", date_column)):
        if name in independent_columns:
            raise InvalidConfigError(
                f"'{name}' is the {role.replace('_', ' ')}, so it cannot also be in "
                f"independent_columns. "
                + (
                    "A model given the target as a driver just copies it, which scores a "
                    "perfect MAPE and cannot forecast."
                    if role == "target_column"
                    else "Calendar effects are built from the date by every model already."
                ),
                param_name="independent_columns",
                provided_value=list(independent_columns),
            )
    repeated = list(
        dict.fromkeys(c for c in independent_columns if independent_columns.count(c) > 1)
    )
    if repeated:
        raise InvalidConfigError(
            f"independent_columns lists {repeated} more than once. Name each driver once.",
            param_name="independent_columns",
            provided_value=list(independent_columns),
        )
    columns = list(raw_df.columns)
    used = [date_column, target_column, *independent_columns]
    doubled = [c for c in used if columns.count(c) > 1]
    if doubled:
        raise InvalidConfigError(
            f"raw_df has more than one column named {doubled}. Rename or drop the extra "
            f"column(s) so each name is used once.",
            param_name="raw_df",
            provided_value=doubled,
        )


# Settings that only shape a model's own interval. HorizonForecaster replaces
# every model's interval with a calibrated band, so passing one is a mistake.
_NATIVE_BAND_SETTINGS = {
    "prophet": ("interval_width", "uncertainty_samples"),
    "sarimax": ("alpha",),
    "bayesianpymc": ("ci_lower_pct", "ci_upper_pct"),
}


def _check_model_params(model_params) -> Dict[str, Optional[dict]]:
    """
    model_params must map model names to dicts of settings. An unknown model
    name is an error (ignored, a typo such as "neural_prophet" would leave
    the model on its defaults). For wrappers that read their settings one by
    one (SARIMAX, PyMC -- see BaseModel.accepted_params) an unknown setting
    is an error too: ignored, it would still switch off the grid search, so
    a typo would change the result without a word. Names for
    a known model that was not requested are allowed and unused.
    """
    if model_params is None:
        return {}
    if not isinstance(model_params, Mapping):
        raise InvalidConfigError(
            f"model_params must be a dict of {{model name: {{setting: value}}}}, e.g. "
            f"{{'prophet': {{'seasonality_mode': 'multiplicative'}}}} (got "
            f"{type(model_params).__name__}).",
            param_name="model_params",
            provided_value=type(model_params).__name__,
        )
    unknown = [m for m in model_params if m not in MODEL_REGISTRY]
    if unknown:
        hints = []
        for m in unknown:
            close = difflib.get_close_matches(str(m), list(MODEL_REGISTRY), n=1, cutoff=0.6)
            hints.append(f"'{m}'" + (f" (did you mean '{close[0]}'?)" if close else ""))
        raise InvalidConfigError(
            f"model_params names unknown model(s): {', '.join(hints)}. Available models: "
            f"{list(MODEL_REGISTRY)}.",
            param_name="model_params",
            provided_value=unknown,
        )
    for name, settings in model_params.items():
        if settings is None:
            continue
        if not isinstance(settings, Mapping):
            raise InvalidConfigError(
                f"model_params['{name}'] must be a dict of settings, e.g. {{'{name}': "
                f"{{...}}}} (got {type(settings).__name__}).",
                param_name="model_params",
                provided_value={name: settings},
            )
        if name == "neuralprophet":
            missing = [k for k in ("learning_rate", "epochs") if settings.get(k) is None]
            if missing:
                raise InvalidConfigError(
                    f"model_params['neuralprophet'] must set both learning_rate and epochs "
                    f"(missing: {missing}). Without them NeuralProphet runs its learning-rate "
                    f"finder, which fails on torch 2.6 and later and makes two runs on the same "
                    f"data non-comparable. The default grid uses learning_rate 0.01-0.10 and "
                    f"epochs 150-300, e.g. {{'learning_rate': 0.03, 'epochs': 150}}.",
                    param_name="model_params",
                    provided_value={name: missing},
                )
        native = [k for k in _NATIVE_BAND_SETTINGS.get(name, ()) if k in settings]
        if native:
            raise InvalidConfigError(
                f"model_params['{name}'] sets {', '.join(repr(k) for k in native)}, which only "
                f"shaped {name}'s own interval. Horizon gives every model and ensemble a 95% "
                f"band calibrated on its validation errors instead (docs/supported_models.md, "
                f"'Uncertainty bands'), so the setting would have no effect: remove it.",
                param_name="model_params",
                provided_value={name: native},
            )
        accepted = MODEL_REGISTRY[name]["wrapper"].accepted_params
        if accepted is None:
            continue
        bad = [k for k in settings if k not in accepted]
        if bad:
            hints = []
            for k in bad:
                close = difflib.get_close_matches(str(k), sorted(accepted), n=1, cutoff=0.6)
                hints.append(f"'{k}'" + (f" (did you mean '{close[0]}'?)" if close else ""))
            raise InvalidConfigError(
                f"model_params['{name}'] has setting(s) {name} does not use: "
                f"{', '.join(hints)}. Its settings are {sorted(accepted)}.",
                param_name="model_params",
                provided_value={name: bad},
            )
    return dict(model_params)


def _check_finite_predictions(model_name: str, forecast_df: pd.DataFrame) -> None:
    """
    A model that returns NaN or inf predictions has failed, even if it raised
    nothing. Kept, such a model misleads: SARIMAX on an all-negative target
    returns NaN everywhere, scores NaN, and would be listed as a member of
    the trimmed ensemble while the other members did the work.
    """
    yhat = pd.to_numeric(forecast_df["yhat"], errors="coerce").to_numpy(dtype=float)
    bad = ~np.isfinite(yhat)
    if bad.any():
        first = forecast_df["ds"].iloc[int(np.flatnonzero(bad)[0])]
        raise ModelPredictionError(
            f"{model_name} returned {int(bad.sum())} of {len(yhat)} predictions that are not "
            f"finite numbers (NaN or inf), the first for {pd.Timestamp(first):%Y-%m-%d}, so it "
            f"is treated as failed.",
            model_name=model_name,
        )


def _short_error(error: Exception, limit: int = 220) -> str:
    text = f"{type(error).__name__}: {error}"
    return text if len(text) <= limit else text[: limit - 3] + "..."


@dataclass
class _Selection:
    """
    Internal record of the hyperparameter-selection step for ONE model.

    Selection happens on the VALIDATION window and nothing else -- this is
    the whole point of the three-way split. Keeping it in its own record,
    cached separately from EvaluationResult, means forecast() can reuse a
    selection (which is all it needs: the winning params, plus the
    validation scores for ensemble screening/weighting) WITHOUT paying for the train+
    validation refit and test prediction that only evaluate() reports.
    """

    params: dict
    validation_mape: MapeResult
    grid_search_summary: Optional[List[dict]] = None
    validation_wape: float = float("nan")
    # The winning settings' forecast over the validation window; the band
    # is calibrated from its errors (see evaluation/bands.py). None when
    # the model could not be scored on validation.
    validation_predictions: Optional[np.ndarray] = None


@dataclass
class EvaluationResult:
    """
    Structured result from evaluating ONE model. HorizonForecaster.evaluate()
    returns a Dict[str, EvaluationResult] keyed by model name.

    The two MAPEs here are the heart of the three-way split, and they are
    NOT interchangeable (the same holds for `validation_wape` and `wape`,
    and when the forecaster's selection_metric is "wape" it is
    `validation_wape` that every decision was made on):

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
        test_predictions: DataFrame with columns ["ds", "y", "yhat"],
            plus the calibrated 95% band "yhat_lower"/"yhat_upper" (see
            `band`), over the test window.
        fitted_model: the wrapper instance fitted on TRAIN + VALIDATION
            using the winning hyperparameters -- i.e. the model that
            actually produced `test_predictions`. Deliberately not the
            train-only fit used during selection: refitting on everything
            available before scoring test is what makes the reported MAPE
            an estimate of the procedure that produces the real forecast.
        wape: WAPE on the TEST window, next to `mape`. WAPE divides the total
            absolute error by the total actual, so unlike MAPE it counts the
            rows whose actual is 0.
        validation_wape: WAPE on the VALIDATION window, next to
            `validation_mape`. Which of the two drove the decisions is the
            forecaster's `selection_metric`.
        grid_search_summary: if hyperparameters were selected via grid
            search (no explicit model_params override given), a list of
            every candidate tried, each as {"params": {...}, "mape": float,
            "wape": float} where both are VALIDATION scores, sorted
            best-to-worst on the forecaster's selection_metric.
            None if an explicit override was used -- and also None when a
            grid search WAS attempted but every candidate failed on the
            training window, so the first configured candidate was used
            as an unselected fallback. Do not read `is None` as "an
            override was given"; check `model_params` for that.
        band: how the 95% band in `test_predictions` was calibrated
            (evaluation/bands.py). Its `width` is None, and the band
            columns are absent, when there were too few validation errors.
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
    wape: float = float("nan")
    validation_wape: float = float("nan")
    band: Optional[BandCalibration] = None


@dataclass
class ForecastResult:
    """
    Structured result from forecasting with ONE model. forecast()
    returns a Dict[str, ForecastResult] keyed by model name.

    Attributes:
        model_name: which model produced this forecast.
        forecast: DataFrame with columns ["ds", "yhat"], plus the calibrated
            95% band "yhat_lower"/"yhat_upper" (see `band`).
        horizon: the number of periods forecasted.
        used_default_future_strategy: True if ANY future independent-variable
            values were auto-generated (via future_exog_strategy) rather than
            caller-supplied.
        future_exog_used: the resolved future independent values actually fed
            to the models (supplied columns + generated columns), with a "ds"
            column, for inspection. None when there are no independent columns.
        band: how the 95% band in `forecast` was calibrated (the same
            calibration as the model's evaluate() band); its `width` is None,
            and the band columns are absent, when there were too few
            validation errors.
    """

    model_name: str
    forecast: pd.DataFrame
    horizon: int
    used_default_future_strategy: bool
    future_exog_used: Optional[pd.DataFrame] = None
    band: Optional[BandCalibration] = None


class HorizonForecaster:
    """
    Main orchestrator. Takes raw user data once at construction, and
    exposes evaluate() (train/validation/test split + MAPE and WAPE, per
    model) and forecast() (full-data refit + real future prediction, per
    model).

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
            fraction (0.5-1.0) of the rows to train on; the
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
            each model's VALIDATION score (MAPE, or WAPE when
            selection_metric resolves to it). A model scoring above it is
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
            validation score gets proportionally more weight), or "equal".
            Accepts a list to build several, or None to build all of
            them; the models are already fitted by then, so extra rules
            are close to free. evaluate_ensemble/forecast_ensemble are
            keyed by method name in every case.
        future_exog_strategy: how forecast() generates future values for the
            independent columns when they aren't supplied. "auto" (default)
            is "last_year" when the history covers a year (364 days, or 52
            weeks) and "carry_forward" otherwise; the strategy used is
            stored in self.future_exog_strategy. Or one of
            future_exog.FUTURE_EXOG_STRATEGIES: "last_year" (the same weekday
            364 days earlier, or the same week 52 weeks earlier, so the
            latest year repeats), "carry_forward" (hold the last value flat),
            "recent_average", "historical_average" (season-aware),
            "seasonal_naive" (seasonal cycle + drift), "linear_trend", or
            "cycle" (replay the trailing history). Continuous drivers use the
            chosen strategy; binary event flags are 0, except under
            seasonal_naive and last_year, where they repeat. See
            future_exog.py.
        event_columns: names of the binary 0/1 event/flag columns (promos,
            holidays) among independent_columns, so the strategy sets them to
            0 ("no event unless supplied") instead of carrying them forward;
            seasonal_naive and last_year repeat them instead.
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
        selection_metric: the score every decision is made on -- the grid
            search, the mape_threshold screen, the inverse weights and the
            exported best model, in any case ("MAPE" works). "mape" or
            "wape" forces one; "auto" (the
            default) uses MAPE unless the target is 0 on at least 10% of its
            rows or has a negative value, where MAPE stops meaning anything
            (it skips zero actuals, and explodes near zero), and then uses
            WAPE, with a warning. Both are always reported. mape_threshold
            applies to whichever is used; the two are on the same scale for
            ordinary series. The resolved metric is `self.selection_metric`.
            A target that is 0 on half its rows or more is refused whatever
            this is set to: see _resolve_selection_metric().

    Dates are parsed BEFORE the data is sorted or split, so text dates sort
    by time rather than alphabetically. The date column must then meet the
    acceptance contract checked in _validate_date_spacing(): each date once,
    every step a whole number of periods, more than half of the steps
    exactly one period, and no date missing between the first and the
    last. Anything else raises.

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
    it was chosen by, with `wape` and `validation_wape` alongside (the
    selection score is `validation_wape` when selection_metric is
    "wape"). A single held-out window serving both roles would make every
    reported MAPE a selection score and therefore optimistic.

    One deliberate exception: the data checks that decide the run as a
    whole -- MAPE or WAPE (see selection_metric), leaving SARIMAX out for
    a negative or zero-heavy target (see skipped_models), and dropping a
    declared driver that never changes in the history (see
    dropped_zero_variance_columns) -- read every row, test included, so
    that evaluate() and forecast() always make the same decision and the
    results stay uniform and clear. They read properties of the data,
    never a model's score, so they cannot make a reported score
    optimistic. See docs/design_notes.md, "Three windows, not two".

    Ensemble behavior: after evaluate()/forecast() build their per-model
    dict as above, each call additionally populates
    self.evaluate_ensemble / self.forecast_ensemble as a side effect --
    a Dict[str, EnsembleResult] keyed by combination method (see
    evaluation/ensemble.py), always a dict even when one method was
    requested, matching the convention the per-model returns already
    follow. forecast() has no held-out actuals of its own, so any
    screening or weighting reuses the validation score from selection,
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
        future_exog_strategy: str = "auto",
        event_columns: Optional[List[str]] = None,
        future_exog_aggregator: str = "mean",
        recent_average_window: int = 28,
        date_format: Optional[str] = None,
        selection_metric: str = "auto",
    ):
        self.frequency = validate_frequency(frequency)
        self.date_column = date_column
        self.target_column = target_column

        self.mape_threshold = validate_mape_threshold(mape_threshold)
        self.ensemble_methods = self._validate_ensemble_methods(ensemble_method)
        # Abbreviations, so "MAPE" or " Wape " are accepted too, as frequency
        # is; the stored value is always lowercase.
        if isinstance(selection_metric, str):
            selection_metric = selection_metric.strip().lower()
        if selection_metric not in SELECTION_METRICS:
            raise InvalidConfigError(
                f"selection_metric must be one of {list(SELECTION_METRICS)} "
                f"(got {selection_metric!r}).",
                param_name="selection_metric",
                provided_value=selection_metric,
            )

        # Future-exogenous-data strategy (how forecast() generates future
        # driver values when they aren't supplied). Validated here like
        # ensemble_method; event_columns is validated against the surviving
        # independent columns below, once those are known.
        if future_exog_strategy != "auto" and future_exog_strategy not in FUTURE_EXOG_STRATEGIES:
            raise InvalidConfigError(
                f"future_exog_strategy must be 'auto' or one of "
                f"{sorted(FUTURE_EXOG_STRATEGIES)} (got '{future_exog_strategy}').",
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

        # The data and the column roles, before anything reads a column:
        # a raw_df that is not a DataFrame, a column that plays two roles
        # (the target as a driver, the date as the target), or a name that
        # appears twice would otherwise run to completion with a meaningless
        # score or fail as a raw pandas error deep inside every model.
        _check_raw_df(raw_df)
        requested_independent_columns = (
            as_name_list(independent_columns, "independent_columns") or []
        )
        event_columns = as_name_list(event_columns, "event_columns")
        _check_column_roles(raw_df, date_column, target_column, requested_independent_columns)
        if date_format is not None and not isinstance(date_format, str):
            raise InvalidConfigError(
                f"date_format must be a strptime format string such as '%d/%m/%Y', or None "
                f"(got {date_format!r}).",
                param_name="date_format",
                provided_value=date_format,
            )
        model_params = _check_model_params(model_params)

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

        # The target and the drivers are read the same way: booleans and the
        # on/off words in YES_NO_WORDS become 1/0, numbers written as text
        # become numbers, and any other text raises a specific error (with
        # the date) rather than the generic "not a number" one. Done on a
        # copy, before the value and zero-variance checks.
        to_parse = [
            c
            for c in [target_column, *requested_independent_columns]
            if c in raw_df.columns and c != date_column
        ]
        if to_parse:
            raw_df = raw_df.copy()
            dates = raw_df[date_column] if date_column in raw_df.columns else None
            for col in to_parse:
                read = _parse_target if col == target_column else parse_driver_column
                raw_df[col] = read(raw_df[col], col, dates=dates).to_numpy()

        # Complete, finite data: a single blank, non-numeric or infinite cell
        # in the date, the target or ANY declared driver is a hard failure --
        # nothing is silently dropped or imputed. This runs BEFORE the
        # zero-variance cleanup below, which would otherwise count an
        # all-blank driver as "constant" and drop it, while a single blank
        # raised.
        validate_no_missing_values(
            raw_df, date_column, target_column, requested_independent_columns
        )
        self._reject_duplicate_dates(raw_df)

        # Models that cannot handle this data are left out up front, with the
        # reason, rather than run to a misleading result. {model: reason}.
        self.skipped_models: Dict[str, str] = {}
        self._skip_models_that_cannot_fit(raw_df[target_column])
        self.selection_metric = self._resolve_selection_metric(
            raw_df[target_column], raw_df[date_column], selection_metric
        )
        self._handle_zeros_for_sarimax(raw_df[target_column], raw_df[date_column])

        # Kept for forecast(): a future table may name a declared driver that
        # was dropped below, which warns rather than errors.
        self._declared_independent_columns = list(requested_independent_columns)

        # A declared driver with the same value in every row cannot be
        # learned (and makes SARIMAX's exog matrix singular), so it is dropped
        # -- with a warning, since the caller asked for it. Only declared
        # drivers are looked at: other columns in raw_df are never used.
        cleaned_df, dropped_columns = clean_zero_variance(
            raw_df,
            exclude_columns=[c for c in raw_df.columns if c not in requested_independent_columns],
        )
        self.dropped_zero_variance_columns = dropped_columns
        if dropped_columns:
            values = ", ".join(f"'{c}' is always {raw_df[c].iloc[0]:g}" for c in dropped_columns)
            warnings.warn(
                f"independent_columns {dropped_columns} never change in your data ({values}), "
                f"so no model can learn from them. They were dropped, and the models run "
                f"without them. If they should vary, check the data.",
                stacklevel=2,
            )

        self.independent_columns = [
            c for c in requested_independent_columns if c not in dropped_columns
        ]

        # event_columns (explicit 0/1 flags for the future-exog strategy). None
        # means "auto-detect flags"; a list means "these exact columns are the
        # flags, auto-detect nothing else". A listed column dropped for zero
        # variance is ignored (the warning above already named it); anything
        # else unrecognized is an error.
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
        if self.future_exog_strategy == "auto":
            # Every period is present by now, so the row count is the span.
            year = YEAR_PERIODS[self.frequency]
            self.future_exog_strategy = "last_year" if len(self.raw_df) >= year else "carry_forward"

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
        self._band_cache: Dict[str, BandCalibration] = {}
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
        # Set by evaluate(): the per-model results it returned, the
        # seasonal-naive baseline scored on the same windows, and the
        # correlation of the models' test errors (None with one model).
        self.evaluation_results: Optional[Dict[str, EvaluationResult]] = None
        self.naive_baseline: Optional[EvaluationResult] = None
        self.test_error_correlation: Optional[pd.DataFrame] = None
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

    def _resolve_selection_metric(self, target: pd.Series, dates: pd.Series, requested: str) -> str:
        """
        Decide which score drives selection, and refuse a target that is
        mostly zeros.

        At or above _SPARSE_TARGET_ZERO_SHARE (50%) zero targets the data is
        refused, whatever selection_metric says: forecasting 0 every day has a
        WAPE of exactly 100%, and on the series tested above this share
        (70% and 53% zeros) every model scored 103-129%, worse than that. For
        daily data the error says how often the WEEKLY totals are 0, since
        aggregating is usually the fix (those two series: 9% and 6%).

        Otherwise "auto" keeps MAPE, the default metric, unless MAPE stops
        meaning anything: when at least _WAPE_ZERO_SHARE (10%) of the target is
        0 (MAPE skips every zero actual -- on one real series the models' test
        MAPE was 238-420% while their WAPE was 34-41%), or when the target has
        a negative value (MAPE explodes as actuals cross zero). Then WAPE is
        used, with a warning.
        """
        values = pd.to_numeric(target, errors="coerce").to_numpy(dtype=float)
        n = len(values)
        zero_share = float((values == 0).mean()) if n else 0.0
        if zero_share >= _SPARSE_TARGET_ZERO_SHARE:
            unit = "days" if self.frequency == "daily" else "weeks"
            hint = (
                "Horizon has nothing coarser than weekly, so this series cannot be forecast "
                "as it stands."
            )
            if self.frequency == "daily":
                weeks = (
                    pd.DataFrame({"week": pd.to_datetime(dates).dt.to_period("W"), "y": values})
                    .groupby("week")["y"]
                    .sum()
                    .sort_index()
                )
                # Every calendar week inside the data's range, so the first
                # and last (usually partial) weeks are left out. A week with
                # fewer than 7 rows is still summed: this check runs before
                # the date checks, which reject missing dates afterwards.
                complete = weeks.iloc[1:-1]
                if len(complete):
                    weekly_zero = float((complete == 0).mean())
                    hint = (
                        f"Summed to weekly totals it is 0 in {weekly_zero:.0%} of "
                        f"{len(complete)} complete weeks: aggregate the data to weeks (sum "
                        f"the target and any spend columns; average prices and indices) and "
                        f"run with frequency='weekly'."
                    )
            raise InvalidConfigError(
                f"The target '{self.target_column}' is 0 on {zero_share:.0%} of {unit} "
                f"({int((values == 0).sum())} of {n}). At "
                f"{_SPARSE_TARGET_ZERO_SHARE:.0%} or more, every model tested did worse than "
                f"forecasting 0 every period. {hint} Series that are mostly zeros need a "
                f"method built for intermittent demand, which Horizon does not include.",
                param_name="raw_df",
                provided_value=round(zero_share, 4),
            )
        if requested != "auto":
            return requested
        reasons = []
        if zero_share >= _WAPE_ZERO_SHARE:
            unit = "days" if self.frequency == "daily" else "weeks"
            reasons.append(f"is 0 on {zero_share:.0%} of {unit} (MAPE skips every zero actual)")
        n_negative = int((values < 0).sum())
        if n_negative:
            reasons.append(
                f"is negative in {n_negative} of {n} rows (MAPE is meaningless as actuals "
                f"cross zero)"
            )
        if not reasons:
            return "mape"
        warnings.warn(
            f"The target '{self.target_column}' {' and '.join(reasons)}, so model selection, "
            f"the mape_threshold screen, ensemble weights and the best-model choice use WAPE "
            f"instead of MAPE. Both are still reported. Pass selection_metric='mape' to keep "
            f"MAPE.",
            stacklevel=3,
        )
        return "wape"

    @property
    def _metric_label(self) -> str:
        return self.selection_metric.upper()

    def _selection_score(self, selection: "_Selection") -> float:
        """The validation score decisions are made on, by selection_metric."""
        if self.selection_metric == "wape":
            return selection.validation_wape
        return selection.validation_mape.value

    @property
    def _naive_season_days(self) -> int:
        """A week for daily data, 52 weeks for weekly data."""
        return 7 if self.frequency == "daily" else 364

    def _naive_description(self, history_length: int) -> str:
        if self.frequency == "daily":
            return "the same weekday last week" if history_length >= 7 else "the last value"
        return "the same week last year" if history_length >= 52 else "the last value"

    def _seasonal_naive_baseline(self) -> EvaluationResult:
        """
        The seasonal-naive forecast scored on the same windows as the models:
        validation predicted from the training rows, test from training +
        validation, exactly the history each model's fit had. Daily data
        repeats the same weekday last week; weekly data the same week last
        year (the last value when there is less than a year of history).
        """
        y = pd.to_numeric(self.raw_df[self.target_column]).to_numpy(dtype=float)
        ds = pd.to_datetime(self.raw_df[self.date_column]).to_numpy()
        n_train, n_val, n_test = self._n_train, self._n_validation, self._n_test
        val_end = n_train + n_val
        val_y, test_y = y[n_train:val_end], y[val_end:]
        season = self._naive_season_days
        # Matched by date rather than by row position: the same weekday last
        # week, or the same week last year.
        val_pred = seasonal_naive(ds[:n_train], y[:n_train], ds[n_train:val_end], season)
        test_pred = seasonal_naive(ds[:val_end], y[:val_end], ds[val_end:], season)
        test_predictions = pd.DataFrame(
            {
                "ds": ds[val_end:],
                "y": test_y,
                "yhat": test_pred,
            }
        )
        return EvaluationResult(
            model_name="seasonal_naive",
            mape=mape(test_y, test_pred),
            validation_mape=mape(val_y, val_pred),
            n_train=n_train,
            n_validation=n_val,
            n_test=n_test,
            test_predictions=test_predictions,
            fitted_model=None,
            grid_search_summary=None,
            wape=wape(test_y, test_pred),
            validation_wape=wape(val_y, val_pred),
        )

    def _warn_about_results_worse_than_naive(self, results) -> None:
        """
        Name every model and ensemble rule whose test WAPE is worse than the
        seasonal-naive baseline's. Compared on WAPE whatever selection_metric
        is, because WAPE counts every row. On 42 test series the default
        trimmed ensemble lost to this baseline in 13 of the 35 that formed one,
        and some single model did in 32 -- something no score on its own
        reveals.
        """
        base = self.naive_baseline.wape
        if base != base:  # all-zero test window: no baseline to compare with
            return
        worse = [
            f"{name} {r.wape:.2f}"
            for name, r in results.items()
            if r.wape == r.wape and r.wape > base
        ]
        for method, ensemble in (self.evaluate_ensemble or {}).items():
            if (
                ensemble.wape is not None
                and ensemble.wape == ensemble.wape
                and ensemble.wape > base
            ):
                worse.append(f"the '{method}' ensemble {ensemble.wape:.2f}")
        if not worse:
            return
        description = self._naive_description(self._n_train + self._n_validation)
        warnings.warn(
            f"On the test window these do worse than the seasonal-naive baseline, which "
            f"repeats {description} (test WAPE {base:.2f}): {', '.join(worse)}. A forecast "
            f"that cannot beat that adds nothing over it; compare before relying on it. The "
            f"baseline is forecaster.naive_baseline.",
            stacklevel=3,
        )

    def _skip_models_that_cannot_fit(self, target: pd.Series) -> None:
        """
        SARIMAX fits in log1p space, so a target that goes negative breaks it
        without an error: values below -1 become NaN and are skipped as gaps,
        and every forecast comes back positive. On a series that was negative
        on 43% of days it forecast 399 to 1,352 (test MAE twice Prophet's); on
        a day-to-day change series centred on zero it never forecast below 56;
        on an all-negative series it returned NaN. It is left out of any run
        whose target has a negative value, with a warning, or refused when it
        is the only model requested.
        """
        if "sarimax" not in self.models:
            return
        values = pd.to_numeric(target, errors="coerce")
        n_negative = int((values < 0).sum())
        if not n_negative:
            return
        reason = (
            f"the target '{self.target_column}' is negative in {n_negative} of {len(values)} "
            f"rows (minimum {values.min():g}), and SARIMAX fits in log space, where negative "
            f"values cannot be modelled"
        )
        if self.models == ["sarimax"]:
            raise InvalidConfigError(
                f"SARIMAX cannot forecast this series: {reason}. Choose another model (Prophet, "
                f"CatBoost, NeuralProphet and Bayesian PyMC all handle negative values).",
                param_name="models",
                provided_value=self.models,
            )
        self.models = [m for m in self.models if m != "sarimax"]
        self.skipped_models["sarimax"] = reason
        warnings.warn(
            f"SARIMAX is left out of this run: {reason}. The other models "
            f"{self.models} run as usual; see forecaster.skipped_models.",
            stacklevel=3,
        )

    def _handle_zeros_for_sarimax(self, target: pd.Series, dates: pd.Series) -> None:
        """
        Say what SARIMAX does with zeros where the series normally has a value
        (see sarimax_wrapper.unexpected_zero_mask): it leaves them out of its
        fit, which a warning names. On weekly data with more than
        _SARIMAX_MAX_ZERO_WEEK_SHARE of such weeks it is left out of the run
        instead, like a target with negative values. Zeros on closed weekdays
        are not counted and need no warning.
        """
        if "sarimax" not in self.models:
            return
        values = pd.to_numeric(target, errors="coerce").to_numpy(dtype=float)
        mask = unexpected_zero_mask(values, dates, self.frequency)
        n_zero = int(mask.sum())
        if not n_zero:
            return
        n = len(values)
        examples = ", ".join(f"{d:%Y-%m-%d}" for d in pd.to_datetime(pd.Series(dates))[mask][:3])
        more = f" and {n_zero - 3} more" if n_zero > 3 else ""
        if self.frequency == "daily":
            where = (
                f"is 0 on {n_zero} of {n} days that fall on weekdays which otherwise have "
                f"values ({examples}{more})"
            )
            unit = "days"
        else:
            where = f"is 0 in {n_zero} of {n} weeks ({examples}{more})"
            unit = "weeks"
        share = n_zero / n
        if self.frequency == "weekly" and share > _SARIMAX_MAX_ZERO_WEEK_SHARE:
            reason = (
                f"the target '{self.target_column}' {where}, {share:.0%} of the weeks; above "
                f"{_SARIMAX_MAX_ZERO_WEEK_SHARE:.0%} SARIMAX, which fits in log space where each "
                f"zero week is an extreme outlier, failed or forecast far worse than the other "
                f"models in testing"
            )
            if self.models == ["sarimax"]:
                raise InvalidConfigError(
                    f"SARIMAX cannot forecast this series: {reason}. Choose another model "
                    f"(Prophet, CatBoost, NeuralProphet and Bayesian PyMC all handle zero weeks).",
                    param_name="models",
                    provided_value=self.models,
                )
            self.models = [m for m in self.models if m != "sarimax"]
            self.skipped_models["sarimax"] = reason
            warnings.warn(
                f"SARIMAX is left out of this run: {reason}. The other models "
                f"{self.models} run as usual; see forecaster.skipped_models.",
                stacklevel=3,
            )
            return
        caution = (
            " Zero weeks can still degrade its weekly forecast, so compare it with the other "
            "models."
            if self.frequency == "weekly"
            else ""
        )
        warnings.warn(
            f"SARIMAX: the target '{self.target_column}' {where}. SARIMAX fits in log space, "
            f"where such a zero is an extreme outlier, so it leaves these {unit} out of its own "
            f"fit; the other models still see them as 0.{caution}",
            stacklevel=3,
        )

    def _warn_about_lost_models(self, stage: str, results, errors, ensembles) -> None:
        """
        After evaluate() or forecast(): say which requested models are missing
        from the results or from an ensemble, and why. Without the warning
        each of these is visible only by inspecting evaluate_errors /
        forecast_errors or the EnsembleResult fields: a failed model, a model
        screened out by mape_threshold, a trimmed rule that fell back to a
        mean because members were lost, and an ensemble with no members at
        all.
        """
        requested = list(self.models)
        if errors:
            listed = "; ".join(f"{m} ({_short_error(e)})" for m, e in errors.items())
            verb = "is" if len(errors) == 1 else "are"
            warnings.warn(
                f"{len(errors)} of the {len(requested)} requested models failed in {stage}() and "
                f"{verb} in no ensemble: {listed}. Full errors in forecaster.{stage}_errors.",
                stacklevel=3,
            )

        def why(model: str, ensemble) -> str:
            if model in errors:
                return f"{model} failed"
            rejection = ensemble.rejected_models.get(model)
            if rejection is not None and rejection.allowed_max is not None:
                return (
                    f"{model}'s validation {self._metric_label} {rejection.mape_value:.2f} is "
                    f"above mape_threshold {rejection.allowed_max:g}"
                )
            if rejection is not None:
                return f"{model} has no validation {self._metric_label}"
            return f"{model} was not used"

        def names(models) -> str:
            models = list(models)
            return models[0] if len(models) == 1 else ", ".join(models[:-1]) + " and " + models[-1]

        empty, thinned = [], []
        for method, ensemble in (ensembles or {}).items():
            lost = [m for m in requested if m not in ensemble.members]
            if ensemble.combined is None:
                ran = [m for m in requested if m in results]
                over = {
                    m: ensemble.rejected_models[m]
                    for m in ran
                    if m in ensemble.rejected_models
                    and ensemble.rejected_models[m].allowed_max is not None
                }
                if ran and len(over) == len(ran):
                    limit = next(iter(over.values())).allowed_max
                    scores = ", ".join(f"{m} {r.mape_value:.2f}" for m, r in over.items())
                    reasons = (
                        f"every model's validation {self._metric_label} is above "
                        f"mape_threshold {limit:g}: {scores}"
                    )
                else:
                    reasons = "; ".join(why(m, ensemble) for m in ran)
                empty.append(f"'{method}' ({reasons})" if reasons else f"'{method}'")
            elif lost:
                members = names(ensemble.members)
                if not ensemble.degraded:
                    how = f"combined {members}"
                elif ensemble.rule_applied == "single":
                    how = f"is just {members}"
                elif ensemble.rule_applied == "median":
                    how = f"is the median of {members}, not a trimmed mean"
                else:
                    how = f"is a plain {ensemble.rule_applied} of {members}, not a trimmed mean"
                thinned.append(f"'{method}' {how} ({'; '.join(why(m, ensemble) for m in lost)})")
        if thinned:
            warnings.warn(
                f"The {stage}() ensemble lost models: " + "; ".join(thinned) + ".",
                stacklevel=3,
            )
        if empty:
            warnings.warn(
                f"No ensemble could be formed in {stage}() for "
                + "; ".join(empty)
                + ". The ensemble forecast is empty. Raise mape_threshold (or set it to None), "
                "or use the best single model (exporters.choose_best_model).",
                stacklevel=3,
            )

    def _validate_models(self, models: Optional[List[str]]) -> List[str]:
        models = as_name_list(models, "models")
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
        # De-duplicate while preserving the caller's order, as
        # ensemble_method does: a model listed twice is fitted once.
        return list(dict.fromkeys(models))

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
            f"carry no validation {self._metric_label}, so they are excluded from any weighted "
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
             days, fortnightly, monthly). Both raise: with only a warning
             the run goes on to produce forecast dates, window sizes and
             model settings for the wrong frequency -- monthly data declared
             daily finished with a confident 0.67% MAPE and a daily-dated
             forecast.
          2. Every step is a whole number of periods -- weekly dates all on
             the same weekday. Otherwise the weekly cycle and the forecast
             dates are misaligned.
          3. No date missing between the first and the last: every day
             (daily) or every week (weekly) has a row.

        Missing dates always raise (warning on scattered gaps was tried and
        rejected). The models count rows, not dates: a missing day shifts
        every weekly cycle after it, and the train/validation/test windows
        cover more time than their size says. Horizon never fills a gap
        itself, because only the user knows whether a missing day was a
        closed day (0) or lost data. A weekday
        that never appears is named, since that is almost always a closed
        day the export left out.
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

        if spacing.missing_dates:
            raise InvalidConfigError(
                self._missing_dates_message(dates, spacing, unit),
                param_name="date_column",
                provided_value=[f"{d:%Y-%m-%d}" for d in spacing.missing_dates[:5]],
            )

    _WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")

    def _missing_dates_message(self, dates: pd.Series, spacing, unit: str) -> str:
        """The error for rule 3: what is missing, where, and what to add."""

        def count(n: int) -> str:
            return f"{n} {unit}" + ("" if n == 1 else "s")

        missing = spacing.missing_dates
        shown = ", ".join(f"{d:%Y-%m-%d}" for d in missing[:5])
        if len(missing) > 5:
            shown += f" and {len(missing) - 5} more"
        message = (
            f"'{self.date_column}' is missing {count(len(missing))} between its first and "
            f"last dates ({dates.iloc[0]:%Y-%m-%d} to {dates.iloc[-1]:%Y-%m-%d}): {shown}"
        )
        first, last = spacing.longest_gap
        if spacing.gap_count == 1 and len(missing) > 1:
            message += f", all in a row ({first:%Y-%m-%d} to {last:%Y-%m-%d})"
        elif spacing.gap_count > 1 and spacing.longest_gap_periods == 1:
            message += f", in {spacing.gap_count} separate gaps"
        elif spacing.gap_count > 1:
            message += (
                f", in {spacing.gap_count} gaps (the longest is "
                f"{count(spacing.longest_gap_periods)}, {first:%Y-%m-%d} to {last:%Y-%m-%d})"
            )
        message += "."
        span_days = (dates.iloc[-1] - dates.iloc[0]).days + 1
        if self.frequency == "daily" and span_days >= 14:
            present = set(dates.dt.dayofweek)
            absent = [name for i, name in enumerate(self._WEEKDAYS) if i not in present]
            if absent:
                names = (
                    absent[0]
                    if len(absent) == 1
                    else " or ".join([", ".join(absent[:-1]), absent[-1]])
                )
                message += (
                    f" No {names} appears at all: if the business is closed then, add "
                    f"those days with a value of 0."
                )
        message += (
            f" Every {unit} must be present: add the missing dates, with 0 where the "
            f"business was closed and the real values where data was lost, or start the "
            f"data after a long gap."
        )
        return message

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
                    f"the validation {self._metric_label} -- which picks the hyperparameters, "
                    "is screened against mape_threshold and weights the ensemble -- includes "
                    f"an effect no model could see. {'It is' if one else 'They are'} used in "
                    f"every later fit where {'it changes' if one else 'they change'}"
                )
            else:
                where = "the train+validation window used for the test score"
                score = (
                    f"the test {self._metric_label} includes an effect no model could see. "
                    f"forecast() refits on the full series, where "
                    f"{'it changes' if one else 'they change'}, "
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
        Using transform_train() on the scoring window alone would give the
        same result for ProphetAdapter,
        which is a stateless pure function -- but it would be actively
        WRONG for an adapter that carries state across the train/future
        boundary like CatBoostAdapter: transform_train() resets the
        adapter's retained self._reference_date to the scoring window's
        own first date, which would restart the trend index (time_idx) at
        0 instead of letting it continue seamlessly from where the fit
        window left off -- silently corrupting the trend feature, and any
        subsequent use of the adapter.
        transform_future() is the correct call here -- it's exactly what a
        genuine future forecast would do, reusing the reference_date
        established by the fit. The only difference from a real future
        forecast is that the actual y values for this "future" period ARE
        already known (it's held-out historical data, being used for
        scoring) -- so they're read directly from raw_df, never routed
        through the adapter, which by design never sees them.

        A FRESH adapter is built per call. This matters because there are
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
        _check_finite_predictions(model_name, forecast_df)
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
            candidates.append(
                (
                    full_params,
                    model,
                    forecast_df,
                    mape_result,
                    wape(y_validation, forecast_df["yhat"].values),
                )
            )

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
                f"validation {self._metric_label} -- it will be excluded from any weighted "
                f"ensemble and cannot be screened against mape_threshold. Supply a longer "
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
            val = candidate[4] if self.selection_metric == "wape" else candidate[3].value
            return float("inf") if val != val else val

        candidates.sort(key=sort_key)
        best_params, _best_model, best_forecast_df, best_mape, best_wape = candidates[0]

        # The candidates' own fitted models are deliberately discarded. Each
        # was fitted on TRAIN only, and every window that gets reported is
        # produced by a later refit on train+validation -- keeping the
        # train-only fits around would just invite something downstream to
        # report from the wrong model.
        grid_search_summary = [
            {"params": p, "mape": m.value, "wape": w} for p, _, _, m, w in candidates
        ]

        self._selected_hyperparameters[model_name] = best_params
        return _Selection(
            params=best_params,
            validation_mape=best_mape,
            grid_search_summary=grid_search_summary,
            validation_wape=best_wape,
            validation_predictions=best_forecast_df["yhat"].to_numpy(dtype=float),
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
        test window would leak the test window into selection, which the
        three-way split exists to prevent.
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
                validation_wape = wape(y_validation, _forecast_df["yhat"].values)
                validation_predictions = _forecast_df["yhat"].to_numpy(dtype=float)
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
                    f"used, so forecast() can proceed, but it has no validation "
                    f"{self._metric_label} -- it will be excluded from any weighted "
                    f"ensemble and cannot be screened against mape_threshold.",
                    stacklevel=3,
                )
                validation_mape = MapeResult(
                    value=float("nan"), excluded_zero_actual_count=0, total_count=0
                )
                validation_wape = float("nan")
                validation_predictions = None
            selection = _Selection(
                params=explicit_params,
                validation_mape=validation_mape,
                grid_search_summary=None,
                validation_wape=validation_wape,
                validation_predictions=validation_predictions,
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
        calibration = self._band_calibration(model_name)
        forecast_df = self._with_band(forecast_df, calibration)
        test_predictions = self._build_test_predictions(adapted_test_features, y_test, forecast_df)
        test_wape = wape(y_test, forecast_df["yhat"].values)

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
            wape=test_wape,
            validation_wape=selection.validation_wape,
            band=calibration,
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
        # Cleared first, as forecast() clears its own state, so a call that
        # raises cannot leave an earlier call's results looking like its own
        # (the exports read these through forecaster=).
        self.evaluation_results = None
        self.naive_baseline = None
        self.test_error_correlation = None

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
            per_model_selection_mape={
                name: self._selection_score(self._selection_cache[name]) for name in results
            },
            per_model_predictions={name: r.test_predictions for name, r in results.items()},
            mape_threshold=self.mape_threshold,
            actuals=first_result.test_predictions[["ds", "y"]],
            metric=self._metric_label,
        )
        self._add_ensemble_bands(self.evaluate_ensemble, list(results))
        self._warn_about_lost_models("evaluate", results, errors, self.evaluate_ensemble)
        self._warn_about_bands("evaluate", results, self.evaluate_ensemble)

        self.evaluation_results = results
        self.naive_baseline = self._seasonal_naive_baseline()
        self.test_error_correlation = error_correlation(results) if len(results) > 1 else None
        self._warn_about_results_worse_than_naive(results)
        if "bayesianpymc" in results:
            self._warn_about_bayesianpymc("evaluate", results["bayesianpymc"].fitted_model)

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
        _check_finite_predictions(model_name, forecast_df)
        calibration = self._band_calibration(model_name)

        return ForecastResult(
            model_name=self._forecast_models[model_name].model_name,
            forecast=self._with_band(forecast_df, calibration),
            horizon=horizon,
            used_default_future_strategy=used_default_strategy,
            future_exog_used=future_exog_used,
            band=calibration,
        )

    # ---------- calibrated 95% bands (see evaluation/bands.py) ----------

    def _validation_rows(self) -> pd.DataFrame:
        return self.raw_df.iloc[self._n_train : self._n_train + self._n_validation]

    def _validation_zero_days(self) -> np.ndarray:
        """
        Unexpected zeros in the validation window -- a 0 where the series
        normally has a value, the same rule SARIMAX uses -- judged on train +
        validation. They are left out of every band's calibration.
        """
        end = self._n_train + self._n_validation
        history = self.raw_df.iloc[:end]
        mask = unexpected_zero_mask(
            history[self.target_column].to_numpy(dtype=float),
            history[self.date_column],
            self.frequency,
        )
        return mask[self._n_train : end]

    def _band_calibration(self, model_name: str) -> BandCalibration:
        """One model's band calibration, from its selection's validation forecast. Cached."""
        if model_name not in self._band_cache:
            predictions = self._selection_cache[model_name].validation_predictions
            self._band_cache[model_name] = self._calibrate(predictions)
        return self._band_cache[model_name]

    def _calibrate(self, validation_predictions) -> BandCalibration:
        if validation_predictions is None:
            return BandCalibration(
                width=None,
                n_errors=0,
                n_zero_days_left_out=0,
                validation_length=self._n_validation,
                reason="no validation forecast to calibrate from",
            )
        return calibrate(
            self._validation_rows()[self.target_column].to_numpy(dtype=float),
            validation_predictions,
            self._validation_zero_days(),
        )

    def _with_band(self, frame: pd.DataFrame, calibration: BandCalibration) -> pd.DataFrame:
        """
        Replace any model-native bounds on `frame` with the calibrated band.
        Row i of `frame` is i + 1 periods after the last fitted date. The
        lower edge is floored at 0 when the target is never negative,
        wherever the forecast itself is at or above 0.
        """
        out = frame.drop(columns=[c for c in ("yhat_lower", "yhat_upper") if c in frame.columns])
        floor = bool((self.raw_df[self.target_column] >= 0).all())
        edges = band(out["yhat"].to_numpy(dtype=float), calibration, floor)
        if edges is not None:
            out["yhat_lower"], out["yhat_upper"] = edges
        return out

    def _add_ensemble_bands(self, ensembles: Dict[str, EnsembleResult], models) -> None:
        """
        Give each ensemble rule its own band, calibrated on the rule's own
        validation forecast: the members' validation forecasts combined by
        the same rule, scores and threshold, so the members match the
        combination it surrounds. The members' bands are never averaged.
        """
        dates = self._validation_rows()[self.date_column].reset_index(drop=True)
        available = {
            m: pd.DataFrame({"ds": dates, "yhat": self._selection_cache[m].validation_predictions})
            for m in models
            if self._selection_cache[m].validation_predictions is not None
        }
        scores = {m: self._selection_score(self._selection_cache[m]) for m in available}
        for rule, result in ensembles.items():
            if result.combined is None:
                continue
            calibration = None
            if all(m in available for m in result.members):
                validation = build_ensemble(
                    per_model_selection_mape=scores,
                    per_model_predictions=available,
                    mape_threshold=self.mape_threshold,
                    method=rule,
                )
                if validation.combined is not None and list(validation.members) == list(
                    result.members
                ):
                    combined = validation.combined.sort_values("ds")
                    calibration = self._calibrate(combined["yhat"].to_numpy(dtype=float))
            if calibration is None:
                calibration = BandCalibration(
                    width=None,
                    n_errors=0,
                    n_zero_days_left_out=0,
                    validation_length=self._n_validation,
                    reason="a member has no validation forecast to calibrate from",
                )
            result.band = calibration
            result.combined = self._with_band(result.combined, calibration)

    def _warn_about_bands(self, stage: str, results, ensembles, horizon: int = 0) -> None:
        """
        One warning per call for each band issue: outputs with no band (and
        why), unexpected zero days left out of the calibration, and, for
        forecast(), a horizon running past the validation window.
        """
        calibrations = {name: r.band for name, r in results.items() if r.band is not None}
        calibrations.update(
            {
                f"the '{rule}' ensemble": e.band
                for rule, e in (ensembles or {}).items()
                if e.combined is not None and e.band is not None
            }
        )
        missing: Dict[str, List[str]] = {}
        for name, c in calibrations.items():
            if c.width is None:
                missing.setdefault(c.reason, []).append(name)
        if missing:
            # The advice depends on the reason: too few errors needs a longer
            # validation window, while a failed validation fit needs more
            # training rows, which a longer validation window would take away.
            advice = []
            if any("validation errors, fewer than" in why for why in missing):
                advice.append(
                    "A band is calibrated from at least 19 errors on the validation window, "
                    "so a longer validation_size gives such an output one."
                )
            if any("no validation forecast" in why for why in missing):
                advice.append(
                    "A model with no validation forecast could not be scored on the validation "
                    "window (an earlier warning says why), so it has no errors to calibrate "
                    "from, and neither has an ensemble that includes it."
                )
            warnings.warn(
                "No 95% band for "
                + "; ".join(f"{', '.join(names)} ({why})" for why, names in missing.items())
                + ". "
                + " ".join(advice),
                stacklevel=3,
            )
        zero_days = max((c.n_zero_days_left_out for c in calibrations.values()), default=0)
        if zero_days:
            period = "day" if self.frequency == "daily" else "week"
            warnings.warn(
                f"{zero_days} {period}(s) in the validation window are 0 where the series "
                f"normally has a value, and were left out of the 95% bands' calibration, so a "
                f"few outages do not widen the band for every other {period}. The bands "
                f"therefore describe ordinary {period}s and do not cover such outages.",
                stacklevel=3,
            )
        if stage == "forecast" and horizon > self._n_validation:
            unit = "days" if self.frequency == "daily" else "weeks"
            tested = (
                "about 82-88% of actuals from 3 to 16 months ahead on daily data"
                if self.frequency == "daily"
                else "about 57-61% of actuals from 9 months to 2 years ahead on weekly data"
            )
            warnings.warn(
                f"The forecast runs {horizon} {unit} ahead, past the {self._n_validation} "
                f"{unit} of validation errors the 95% bands are calibrated on. After "
                f"{self._n_validation} {unit} each band widens with the square root of the "
                f"distance ahead; in testing it then covered {tested}. Treat the far end of "
                f"the band as a rough range rather than a 95% guarantee.",
                stacklevel=3,
            )

    def _warn_about_bayesianpymc(self, stage: str, model) -> None:
        """
        One warning per call for each of two things Bayesian PyMC's fits
        decide for themselves: annual terms left out of a fit spanning
        fewer than ANNUAL_MIN_SPAN_DAYS, and sampling that did not converge.

        For evaluate() the annual note covers both fits, since the
        validation fit can be short while the test fit is not; the
        convergence check reads the test fit, the one whose numbers are
        reported. For forecast() both read the full-series fit. Grid
        candidates are not checked: they are discarded after scoring.
        """
        if model is None:
            return
        dates = self.raw_df[self.date_column]
        if stage == "evaluate":
            fits = [
                ("validation fit", span_days(dates.iloc[: self._n_train])),
                ("test fit", span_days(dates.iloc[: self._n_train + self._n_validation])),
            ]
            # A training window below the model's row floor means there was
            # no validation fit at all (the floor warning already says so).
            floor = self._configs["bayesianpymc"]["wrapper_config"]["min_rows_required"]
            if self._n_train < floor:
                fits = fits[1:]
        else:
            fits = [("forecast", span_days(dates))]
        short = [(name, days) for name, days in fits if days < ANNUAL_MIN_SPAN_DAYS]
        if len(short) == len(fits):
            spans = ", ".join(f"{name}: {days} days" for name, days in short)
            warnings.warn(
                f"bayesianpymc: fitted without annual (yearly) seasonality, because its data "
                f"spans fewer than {ANNUAL_MIN_SPAN_DAYS} days ({spans}). With under 6 months "
                f"of history it would have to guess the rest of the yearly cycle, and in "
                f"testing that guess made forecasts swing wildly. The rest of the model is "
                f"unchanged.",
                stacklevel=3,
            )
        elif short:
            (_, val_days), (_, test_days) = fits
            warnings.warn(
                f"bayesianpymc: the validation fit spans {val_days} days, fewer than "
                f"{ANNUAL_MIN_SPAN_DAYS}, so it ran without annual (yearly) seasonality, while "
                f"the test fit ({test_days} days) and forecast() include it. Its validation "
                f"score therefore describes a slightly simpler model than the one tested.",
                stacklevel=3,
            )

        stats = getattr(model, "sample_stats", None) or {}
        rhat = stats.get("max_rhat", float("nan"))
        ess = stats.get("min_ess_bulk", float("nan"))
        problems = []
        if rhat == rhat and rhat > CONVERGENCE_MAX_RHAT:
            problems.append(f"largest R-hat {rhat:.3f}, above {CONVERGENCE_MAX_RHAT}")
        if ess == ess and ess < CONVERGENCE_MIN_ESS:
            problems.append(
                f"smallest effective sample size {ess:.0f}, below {CONVERGENCE_MIN_ESS}"
            )
        if problems:
            fit = "test fit" if stage == "evaluate" else "forecast fit"
            warnings.warn(
                f"bayesianpymc: the sampler did not converge cleanly in its {fit} "
                f"({'; '.join(problems)}), so its forecast and band may be unreliable. "
                f"Raise 'draws' and 'tune' in model_params, or rely on the other models.",
                stacklevel=3,
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
        validation score (MAPE or WAPE) established when each model's
        hyperparameters were selected. That selection has to happen anyway -- forecast()
        cannot fit without knowing which parameters won -- so the scores
        come for free and are cached, whether or not evaluate() was ever
        called. Running the whole of evaluate() internally instead would
        also pay for a test-window prediction that is then discarded.

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
        # would build two future dates and then fail inside every model as a
        # row-count mismatch, ending in an AllModelsRejectedError that hides
        # the real cause. A whole float (30.0) is accepted. bool is excluded
        # because it is an int subclass.
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
        # pandas cannot represent dates after 2262-04-11; a horizon reaching
        # past it would otherwise fail with a raw OutOfBoundsDatetime.
        last_date = pd.Timestamp(self.raw_df[self.date_column].max())
        days_per_period = 7 if self.frequency == "weekly" else 1
        if horizon * days_per_period > (pd.Timestamp.max - last_date).days:
            raise InvalidConfigError(
                f"horizon={horizon} would run the forecast past {pd.Timestamp.max:%Y-%m-%d}, "
                f"the last date pandas can represent.",
                param_name="horizon",
                provided_value=horizon,
            )

        n_total = len(self.raw_df)
        if horizon > _HORIZON_WARNING_FRACTION * n_total:
            warnings.warn(
                f"Forecasting {horizon} periods from only {n_total} periods of history. "
                f"Forecasts this far beyond the available history should be treated with "
                f"caution — the 95% bands may understate the true risk of "
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
            copied = ""
            if self.future_exog_strategy == "last_year":
                source = last_year_source_dates(
                    self.raw_df[self.date_column], future_dates, self.frequency
                )
                if source is not None:
                    copied = (
                        f", which copies each from a year earlier: the values of "
                        f"{source.min():%Y-%m-%d} to {source.max():%Y-%m-%d}"
                    )
                    if horizon > YEAR_PERIODS[self.frequency]:
                        copied += ", the latest year of history, repeated"
            warnings.warn(
                f"future values for {generated} were auto-generated for all {horizon} "
                f"periods via the '{self.future_exog_strategy}' strategy{copied} (no values were "
                f"supplied for them). The forecast is only as reliable as those assumed "
                f"inputs -- for an exogenous-variable-driven model this can dominate the "
                f"result. Supply real planned values if you have them (you can supply "
                f"just some columns; the rest are filled), inspect what was assumed via "
                f"forecaster.future_exog_used, and treat this output as a scenario rather "
                f"than a prediction. The 95% bands assume these values are right: in testing, "
                f"band coverage fell by about 9 points when drivers were generated rather "
                f"than known.",
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
        if "bayesianpymc" in results:
            self._warn_about_bayesianpymc("forecast", self._forecast_models["bayesianpymc"])

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
            name: self._selection_score(self._selection_cache[name]) for name in results
        }
        self.forecast_ensemble = build_ensembles(
            methods=self.ensemble_methods,
            per_model_selection_mape=selection_mape,
            per_model_predictions={name: results[name].forecast for name in selection_mape},
            mape_threshold=self.mape_threshold,
            actuals=None,
            metric=self._metric_label,
        )
        self._add_ensemble_bands(self.forecast_ensemble, list(selection_mape))
        self._warn_about_lost_models("forecast", results, errors, self.forecast_ensemble)
        self._warn_about_bands("forecast", results, self.forecast_ensemble, horizon)

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
