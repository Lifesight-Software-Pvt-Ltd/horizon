"""
CSV exporters for horizon_forecaster.

Turns the outputs of HorizonForecaster.forecast() (plus its ensemble)
into tabular CSVs. Like plotting.py, this is a pure, standalone module:
it never imports forecaster.py or any model/adapter class, and reaches
into the result objects only through their public attributes, so writing
a CSV never pulls in the heavy Prophet/CatBoost stack. Type hints for the
result objects are TYPE_CHECKING-only; at runtime the inputs are
duck-typed.

Five deliverables, always written:

1. Final combined forecast -- the ensemble's ["ds", "yhat"] series, plus
   its calibrated 95% band ["yhat_lower", "yhat_upper"] when it has one: the
   single blended number the caller ultimately acts on. One file per
   combination method, since several rules can be built in one run.
2. Best single model's forecast -- exported alongside the combined one,
   because a combination is not guaranteed to beat the best individual
   model, so both candidates ship with the scores needed to choose. The
   model is chosen on the VALIDATION score (MAPE, or WAPE when the
   forecaster selected on it) among the models that pass mape_threshold
   on both validation and test (see choose_best_model).
3. Per-model forecast -- every model's own forecast, stacked into one
   long/tidy table with a "model" column, plus each model's calibrated
   95% band when it has one, so all models sit in one file for easy
   comparison.
4. Model summary -- per model, its validation and test MAPE and WAPE,
   ensemble membership, weight (where the rule has one), status, whether
   it beats the seasonal-naive baseline, which gets a row of its own, and
   how often its 95% band held the test actuals (band_coverage,
   band_test_points).
5. Rejected-models table -- one row per model excluded from a
   combination, with its validation score (MAPE, or WAPE when the
   forecaster selected on it), the threshold it exceeded (if any) and
   the human-readable reason. This is the "why isn't model X in the
   blend?" report, built straight from EnsembleResult.rejected_models.
   Note exclusion is per method: a model with a NaN validation score is
   dropped by the weighted rules but kept by the trimmed one.

With `forecaster=`, a sixth: the correlation of the models' test errors,
which says whether the members make different mistakes (the only way an
ensemble gains). Passing the forecaster also fills in every score, so
callers do not build the score dicts by hand.

Each has a pure "build the DataFrame" function (returns a DataFrame,
writes nothing) plus one convenience writer, export_forecast_csvs(),
that persists them all at once. The builders never raise on empty/None
input -- they return a correctly-columned empty DataFrame -- so a run
where (say) every model was rejected still produces predictable,
header-only files rather than a crash or a missing file.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import pandas as pd

from horizon_forecaster.evaluation.ensemble import as_ensemble_dict
from horizon_forecaster.exceptions import InvalidConfigError
from horizon_forecaster.utils import (
    check_history,
    check_optional_number,
    check_results_dict,
    check_score_dict,
    validate_mape_threshold,
)

if TYPE_CHECKING:  # hints only; never imported at runtime (avoids the heavy stack)
    from horizon_forecaster.evaluation.ensemble import EnsembleResult
    from horizon_forecaster.forecaster import ForecastResult

_FINAL_COLUMNS = ["ds", "yhat"]
_PER_MODEL_BASE_COLUMNS = ["model", "ds", "yhat"]
_REJECTED_COLUMNS = [
    "model",
    "ensemble_method",
    "mape",
    "mape_threshold",
    "reason",
    "selection_metric",
]
_SUMMARY_COLUMNS = [
    "model",
    "validation_mape",
    "mape",
    "ensemble_method",
    "rule_applied",
    "ensemble_weight",
    "status",
    "best_model_check",
    "validation_wape",
    "wape",
    "zero_actuals_excluded",
    "beats_naive",
    "selection_metric",
    "band_coverage",
    "band_test_points",
]

# Matches HorizonForecaster's default mape_threshold. Pass the forecaster's
# own value (forecaster.mape_threshold) when it was changed.
DEFAULT_MAPE_THRESHOLD = 40.0


class _ThresholdDefault:
    """mape_threshold's default on the export functions: the forecaster's own
    threshold when `forecaster=` is given, else DEFAULT_MAPE_THRESHOLD."""

    def __repr__(self) -> str:
        return "the forecaster's mape_threshold, else 40"


_THRESHOLD_DEFAULT = _ThresholdDefault()


def band_coverage(frame: "pd.DataFrame", actuals: "Optional[pd.DataFrame]" = None) -> tuple:
    """
    (percent of actuals inside the 95% band, number of points) on a test
    window, or (None, None) when the frame has no band. `actuals` (columns
    "ds", "y") is merged in by date when `frame` has no "y" column, as for an
    ensemble's combined forecast.
    """
    if frame is None or "yhat_lower" not in frame.columns:
        return (None, None)
    if "y" not in frame.columns:
        if actuals is None:
            return (None, None)
        frame = frame.merge(actuals[["ds", "y"]], on="ds", how="inner")
    if frame.empty:
        return (None, None)
    inside = (frame["y"] >= frame["yhat_lower"]) & (frame["y"] <= frame["yhat_upper"])
    return (float(inside.mean() * 100), int(len(frame)))


def _scores_from_forecaster(forecaster, caller: str) -> dict:
    """
    Everything the exports need from a forecaster after evaluate() and
    forecast(), read through public attributes only (this module never imports
    forecaster.py). Missing pieces come back as None.
    """
    for attr in ("forecast_ensemble", "evaluate_ensemble", "evaluation_results", "mape_threshold"):
        if not hasattr(forecaster, attr):
            raise InvalidConfigError(
                f"{caller}: forecaster= must be the HorizonForecaster the forecasts came from "
                f"(got {type(forecaster).__name__}, which has no '{attr}').",
                param_name="forecaster",
                provided_value=type(forecaster).__name__,
            )
    results = forecaster.evaluation_results
    if results is None:
        warnings.warn(
            f"{caller}: forecaster.evaluate() has not run, so the scores, the baseline and the "
            f"error correlation are left out of the export. Call evaluate() before exporting.",
            stacklevel=3,
        )
        results = {}
    baseline = getattr(forecaster, "naive_baseline", None)
    evaluated = forecaster.evaluate_ensemble or {}
    first = getattr(next(iter(results.values()), None), "test_predictions", None)
    actuals = first[["ds", "y"]] if first is not None and "y" in first else None
    return {
        "per_model_mape": {m: r.mape.value for m, r in results.items()} or None,
        "per_model_validation_mape": {m: r.validation_mape.value for m, r in results.items()}
        or None,
        "per_model_wape": {m: getattr(r, "wape", None) for m, r in results.items()},
        "per_model_validation_wape": {
            m: getattr(r, "validation_wape", None) for m, r in results.items()
        },
        "zero_actuals_excluded": {m: r.mape.excluded_zero_actual_count for m, r in results.items()},
        "ensemble_scores": {
            method: (
                None if e.mape is None else e.mape.value,
                getattr(e, "wape", None),
                None if e.mape is None else e.mape.excluded_zero_actual_count,
            )
            for method, e in evaluated.items()
        },
        "band_coverage": {
            m: band_coverage(getattr(r, "test_predictions", None)) for m, r in results.items()
        },
        "ensemble_band_coverage": {
            method: band_coverage(e.combined, actuals)
            for method, e in evaluated.items()
            if e.combined is not None and actuals is not None
        },
        "baseline": baseline if results else None,
        "selection_metric": getattr(forecaster, "selection_metric", "mape"),
        "error_correlation": (
            getattr(forecaster, "test_error_correlation", None) if results else None
        ),
    }


def build_final_forecast_table(
    ensemble: "Optional[EnsembleResult] | Dict[str, EnsembleResult]",
) -> pd.DataFrame:
    """
    The final blended forecast as a ["ds", "yhat"] table, plus its 95% band
    ["yhat_lower", "yhat_upper"] when it has one.

    Accepts the {method: EnsembleResult} dict a forecaster actually
    produces, a single EnsembleResult, or None. Given several methods it
    tables the FIRST one requested -- a single
    two-column table cannot represent several rules at once, and
    export_forecast_csvs() writes one file per method anyway rather than
    relying on this choice.

    Returns an empty (header-only) DataFrame when no ensemble was formed
    -- ensemble is None or empty, or every model was rejected so
    combined is None -- so a caller always gets a well-shaped table.
    """
    result = next(iter(as_ensemble_dict(ensemble).values()), None)
    if result is None or result.combined is None:
        return pd.DataFrame(columns=_FINAL_COLUMNS)
    columns = _FINAL_COLUMNS + [c for c in ("yhat_lower", "yhat_upper") if c in result.combined]
    return result.combined[columns].reset_index(drop=True)


def build_per_model_forecast_table(
    forecast_results: "Dict[str, ForecastResult]",
) -> pd.DataFrame:
    """
    Every model's forecast stacked into one long table.

    Columns: ["model", "ds", "yhat"], plus the 95% band "yhat_lower"/
    "yhat_upper" when at least one model has one (a model without, e.g. one
    with too few validation errors, gets NaN there). Returns a header-only
    DataFrame when forecast_results is empty.
    """
    check_results_dict(forecast_results, "forecast", "build_per_model_forecast_table")
    if not forecast_results:
        return pd.DataFrame(columns=_PER_MODEL_BASE_COLUMNS)

    frames = []
    for name, result in forecast_results.items():
        fc = result.forecast.copy()
        fc.insert(0, "model", name)
        frames.append(fc)

    combined = pd.concat(frames, ignore_index=True)

    # Stable column order: model, ds, yhat, then any bound columns present.
    ordered = [c for c in _PER_MODEL_BASE_COLUMNS if c in combined.columns]
    ordered += [c for c in ("yhat_lower", "yhat_upper") if c in combined.columns]
    ordered += [c for c in combined.columns if c not in ordered]
    return combined[ordered]


def _finite(value) -> bool:
    return value is not None and value == value


def choose_best_model(
    forecast_results: "Dict[str, ForecastResult]",
    per_model_validation_mape: "Optional[Dict[str, float]]",
    per_model_mape: "Optional[Dict[str, float]]" = None,
    mape_threshold: Optional[float] = DEFAULT_MAPE_THRESHOLD,
    *,
    metric: str = "MAPE",
) -> Tuple[Optional[str], Dict[str, str]]:
    """
    Pick the model exported as "best", and say why each model was or was not.

    Two checks, then one choice:
      1. validation score at or below mape_threshold -- the same screen
         the ensemble uses;
      2. test score at or below mape_threshold, when test scores are given;
      then the lowest VALIDATION score among the models that pass both.

    Choosing the lowest TEST MAPE with no threshold was rejected: it picks
    the export on the window that is supposed to be reported, never
    selected on, and in a real run it exported a model the ensemble had
    rejected, at 103% error. Choosing on test was also measured directly
    and picked worse (docs/design_notes.md, "Window sizes"). Test is kept
    only as a check that rules a model out: a model that passed validation
    but scored above the threshold on test kept doing badly in the periods
    that followed (median 24.8% against 5.6% for the rest). The trade-off,
    stated in the note: the exported model's test MAPE is slightly
    optimistic, because test helped decide which model it is.

    mape_threshold=None skips both checks; a model without a finite
    validation score is never chosen, since there is nothing to choose on.

    `metric` names the score the two dicts hold ("MAPE", or "WAPE" when the
    forecaster's selection_metric is WAPE); only the reasons' wording uses it.

    Returns (best_model_or_None, {model: reason}).
    """
    check_results_dict(forecast_results, "forecast", "choose_best_model")
    check_score_dict(per_model_validation_mape, "per_model_validation_mape", "choose_best_model")
    check_score_dict(per_model_mape, "per_model_mape", "choose_best_model")
    mape_threshold = validate_mape_threshold(mape_threshold)
    checks: Dict[str, str] = {}
    eligible: Dict[str, float] = {}
    validation = per_model_validation_mape or {}
    for name in forecast_results:
        val = validation.get(name)
        if not _finite(val):
            checks[name] = f"not chosen: no validation {metric}"
            continue
        if mape_threshold is not None and val > mape_threshold:
            checks[name] = f"not chosen: validation {metric} {val:.2f} > {mape_threshold:g}"
            continue
        if per_model_mape is not None:
            test = per_model_mape.get(name)
            if not _finite(test):
                checks[name] = f"not chosen: no test {metric}"
                continue
            if mape_threshold is not None and test > mape_threshold:
                checks[name] = f"not chosen: test {metric} {test:.2f} > {mape_threshold:g}"
                continue
        checks[name] = "passed"
        eligible[name] = val

    best = min(eligible, key=eligible.get) if eligible else None
    if best is not None and mape_threshold is None:
        checks[best] = (
            f"exported as best: lowest validation {metric} (mape_threshold is None, no checks)"
        )
    elif best is not None:
        checks[best] = f"exported as best: lowest validation {metric} among the models that passed"
        if per_model_mape is not None:
            checks[best] += (
                f" both checks (test {metric} was used as a check, so it is slightly "
                f"optimistic for this model)"
            )
        else:
            checks[best] += f" the validation check (no test {metric}s were given)"
    return best, checks


def build_best_model_forecast_table(
    forecast_results: "Dict[str, ForecastResult]",
    per_model_mape: "Optional[Dict[str, float]]" = None,
    *,
    per_model_validation_mape: "Optional[Dict[str, float]]" = None,
    mape_threshold: Optional[float] = DEFAULT_MAPE_THRESHOLD,
    metric: str = "MAPE",
) -> pd.DataFrame:
    """
    The best single model's forecast, as a ["model", "ds", "yhat"] table,
    plus its calibrated 95% band when it has one.

    Exists because a weighted ensemble is NOT guaranteed to beat the best
    individual model (see evaluation/ensemble.py) -- exporting only the
    blend can mean shipping the worse forecast. This is written alongside
    the ensemble so the choice can be made per run rather than baked in.

    The model is chosen by choose_best_model(): validation score among the
    models within mape_threshold on validation and on test. `per_model_mape`
    is the test-window MAPE and `per_model_validation_mape` the validation
    MAPE, both from evaluate(). Returns a header-only DataFrame, with a
    warning saying why, when no model qualifies; the ensemble files are
    unaffected.

    When the forecaster selected on WAPE, pass the WAPE dicts in those two
    arguments and metric="WAPE"; export_forecast_csvs(forecaster=...) does
    this for you.
    """
    check_results_dict(forecast_results, "forecast", "build_best_model_forecast_table")
    if not forecast_results:
        return pd.DataFrame(columns=_PER_MODEL_BASE_COLUMNS)
    if per_model_validation_mape is None:
        warnings.warn(
            "The best-model file is written empty: the best model is chosen on validation "
            "MAPE, and per_model_validation_mape was not given. Pass forecaster=forecaster "
            "to the export, or {m: r.validation_mape.value for m, r in results.items()} "
            "from evaluate().",
            stacklevel=3,
        )
        return pd.DataFrame(columns=_PER_MODEL_BASE_COLUMNS)

    best, checks = choose_best_model(
        forecast_results, per_model_validation_mape, per_model_mape, mape_threshold, metric=metric
    )
    if best is None:
        warnings.warn(
            f"No model qualifies as the best single model, so the best-model file is written "
            f"empty (the ensemble files are unaffected): "
            f"{'; '.join(f'{m}: {why}' for m, why in checks.items())}.",
            stacklevel=3,
        )
        return pd.DataFrame(columns=_PER_MODEL_BASE_COLUMNS)

    table = forecast_results[best].forecast.copy()
    table.insert(0, "model", best)
    ordered = [c for c in _PER_MODEL_BASE_COLUMNS if c in table.columns]
    ordered += [c for c in ("yhat_lower", "yhat_upper") if c in table.columns]
    ordered += [c for c in table.columns if c not in ordered]
    return table[ordered]


def build_model_summary_table(
    per_model_mape: "Optional[Dict[str, float]]" = None,
    ensemble: "Optional[EnsembleResult] | Dict[str, EnsembleResult]" = None,
    ensemble_mape: "Optional[float]" = None,
    per_model_validation_mape: "Optional[Dict[str, float]]" = None,
    best_model_checks: "Optional[Dict[str, str]]" = None,
    *,
    per_model_wape: "Optional[Dict[str, float]]" = None,
    per_model_validation_wape: "Optional[Dict[str, float]]" = None,
    zero_actuals_excluded: "Optional[Dict[str, int]]" = None,
    ensemble_scores: "Optional[Dict[str, tuple]]" = None,
    baseline=None,
    selection_metric: Optional[str] = None,
    band_coverage: "Optional[Dict[str, tuple]]" = None,
    ensemble_band_coverage: "Optional[Dict[str, tuple]]" = None,
) -> pd.DataFrame:
    """
    The at-a-glance table for deciding whether to act on a combined
    forecast, a single model, or neither.

    One row per (model, ensemble method) pair, plus one row per method for
    the combined result itself. With the default single method that is the
    familiar "one row per model plus an ensemble row"; asking for several
    methods simply repeats the block per method, so the shape never
    depends on how many were requested. When `baseline` is given, one
    final "seasonal_naive" row with status "baseline" is added.

    Columns: ["model", "validation_mape", "mape", "ensemble_method",
    "rule_applied", "ensemble_weight", "status", "best_model_check",
    "validation_wape", "wape", "zero_actuals_excluded", "beats_naive",
    "selection_metric", "band_coverage", "band_test_points"]. Columns after
    the first eight were added later, at the end, so readers that pick
    columns by name keep working.

    `band_coverage` is the percentage of test-window actuals inside the
    row's 95% band, and `band_test_points` how many points that is: a
    check of the band on this run, noisy on a short window (about +/-8
    points on 28). Empty when the row has no band.

    `best_model_check` says, per model, whether it was exported as the best
    single model and why not otherwise (from choose_best_model); empty on
    the ensemble rows and when no checks were given.

    Both MAPEs are reported because they answer different questions:
    `validation_mape` is what the model was SELECTED and screened on,
    `mape` is its honest out-of-sample score on the test window. A model
    with a large gap between them was flattered by the selection window.

    `wape` and `validation_wape` are the same two windows scored as
    sum|y - yhat| / sum|y| * 100. MAPE leaves out every row whose actual is
    zero; `zero_actuals_excluded` is how many test rows that was, so a MAPE
    computed on a fraction of the window is visible as such. WAPE counts
    every row. `selection_metric` is the score the forecaster selected,
    screened and weighted on ("mape" or "wape").

    `beats_naive` is True where the row's test WAPE is lower than the
    seasonal-naive baseline's -- the same weekday last week for daily data,
    the same week last year for weekly. A forecast that does not beat it
    adds nothing over repeating the past. Empty without a baseline or
    when either WAPE is missing.

    `ensemble_weight` is empty for weightless rules such as the default
    trimmed mean. That is deliberate rather than a gap in the export --
    there is no weight, and filling in an equal share would imply an
    averaging that did not happen.

    Keyword-only extras:
        per_model_wape, per_model_validation_wape: {model: WAPE}, from
            r.wape and r.validation_wape of evaluate()'s results.
        zero_actuals_excluded: {model: count}, from
            r.mape.excluded_zero_actual_count.
        ensemble_scores: {method: (test MAPE, test WAPE, zero actuals
            excluded)} for ensembles that carry no score of their own (the
            forecast() ensemble), from forecaster.evaluate_ensemble.
        baseline: forecaster.naive_baseline (an EvaluationResult).
        selection_metric: forecaster.selection_metric.
        band_coverage: {model: (percent, points)}, from band_coverage() of
            each result's test_predictions.
        ensemble_band_coverage: {method: (percent, points)} for the
            evaluate() ensembles, the same check on the combined forecast.
    """
    caller = "build_model_summary_table"
    check_score_dict(per_model_mape, "per_model_mape", caller)
    check_score_dict(per_model_validation_mape, "per_model_validation_mape", caller)
    check_score_dict(per_model_wape, "per_model_wape", caller)
    check_score_dict(per_model_validation_wape, "per_model_validation_wape", caller)
    check_optional_number(ensemble_mape, "ensemble_mape", caller)
    ensembles = as_ensemble_dict(ensemble)
    per_model_mape = per_model_mape or {}
    per_model_validation_mape = per_model_validation_mape or {}
    per_model_wape = per_model_wape or {}
    per_model_validation_wape = per_model_validation_wape or {}
    zero_actuals_excluded = zero_actuals_excluded or {}
    ensemble_scores = ensemble_scores or {}
    best_model_checks = best_model_checks or {}
    band_coverage = band_coverage or {}
    ensemble_band_coverage = ensemble_band_coverage or {}
    base_wape = _finite_or_none(getattr(baseline, "wape", None))

    def beats(value):
        value = _finite_or_none(value)
        return None if value is None or base_wape is None else bool(value < base_wape)

    def model_row(name, value, method, rule, weight, status):
        return {
            "model": name,
            "validation_mape": per_model_validation_mape.get(name),
            "mape": value,
            "ensemble_method": method,
            "rule_applied": rule,
            "ensemble_weight": weight,
            "status": status,
            "best_model_check": best_model_checks.get(name),
            "validation_wape": per_model_validation_wape.get(name),
            "wape": per_model_wape.get(name),
            "zero_actuals_excluded": zero_actuals_excluded.get(name),
            "beats_naive": beats(per_model_wape.get(name)),
            "selection_metric": selection_metric,
            "band_coverage": band_coverage.get(name, (None, None))[0],
            "band_test_points": band_coverage.get(name, (None, None))[1],
        }

    rows = []
    if not ensembles:
        rows = [
            model_row(name, value, None, None, None, "not_in_ensemble")
            for name, value in per_model_mape.items()
        ]

    for method, result in ensembles.items():
        weights = result.weights or {}
        for name, value in per_model_mape.items():
            if name in result.members:
                status = "in_ensemble"
            elif name in result.rejected_models:
                status = "rejected"
            else:
                status = "not_in_ensemble"
            rule = result.rule_applied if status == "in_ensemble" else None
            rows.append(model_row(name, value, method, rule, weights.get(name), status))

        # The scalar `ensemble_mape` fallback describes ONE combined series,
        # so it is only unambiguous when one method was built. With several,
        # stamping it on every method would report the same score against
        # three materially different series -- and comparing the rules is
        # exactly what this table is for. Leave those blank instead.
        scored_wape, excluded = getattr(result, "wape", None), None
        if result.mape is not None:
            scored = result.mape.value
            excluded = result.mape.excluded_zero_actual_count
        elif method in ensemble_scores:
            scored, scored_wape, excluded = ensemble_scores[method]
        elif len(ensembles) == 1:
            scored = ensemble_mape
        else:
            scored = None

        if scored is not None or result.combined is not None:
            rows.append(
                {
                    "model": "ensemble",
                    "validation_mape": None,
                    "mape": scored,
                    "ensemble_method": method,
                    "rule_applied": result.rule_applied,
                    "ensemble_weight": 1.0,
                    "status": "degraded" if result.degraded else "ensemble",
                    "best_model_check": None,
                    "validation_wape": None,
                    "wape": scored_wape,
                    "zero_actuals_excluded": excluded,
                    "beats_naive": beats(scored_wape),
                    "selection_metric": selection_metric,
                    "band_coverage": ensemble_band_coverage.get(method, (None, None))[0],
                    "band_test_points": ensemble_band_coverage.get(method, (None, None))[1],
                }
            )

    if baseline is not None:
        rows.append(
            {
                "model": getattr(baseline, "model_name", "seasonal_naive"),
                "validation_mape": baseline.validation_mape.value,
                "mape": baseline.mape.value,
                "ensemble_method": None,
                "rule_applied": None,
                "ensemble_weight": None,
                "status": "baseline",
                "best_model_check": None,
                "validation_wape": getattr(baseline, "validation_wape", None),
                "wape": getattr(baseline, "wape", None),
                "zero_actuals_excluded": baseline.mape.excluded_zero_actual_count,
                "beats_naive": None,
                "selection_metric": selection_metric,
            }
        )

    return pd.DataFrame(rows, columns=_SUMMARY_COLUMNS)


def _finite_or_none(value) -> Optional[float]:
    """value as a float, or None when it is missing or NaN."""
    if value is None:
        return None
    value = float(value)
    return None if value != value else value


def build_rejected_models_table(
    ensemble: "Optional[EnsembleResult] | Dict[str, EnsembleResult]",
    *,
    selection_metric: Optional[str] = None,
) -> pd.DataFrame:
    """
    One row per model excluded from a combination, per method.

    Columns: ["model", "ensemble_method", "mape", "mape_threshold",
    "reason", "selection_metric"]. Built from EnsembleResult.rejected_models
    (a dict of ThresholdRejectionError). `mape` is the validation score the
    model was screened on: its MAPE, or its WAPE when `selection_metric` is
    "wape" (the column keeps its name so existing readers still work, and
    `reason` names the metric). It may be NaN for a model with no score
    (e.g. every actual in the window was zero).

    The `ensemble_method` column matters because exclusion is not
    necessarily the same across rules: a model with a NaN validation score
    cannot be weighted, so the weighted rules drop it, while the trimmed
    rule needs no score and keeps it. Returns a header-only DataFrame when
    nothing was rejected anywhere.
    """
    ensembles = as_ensemble_dict(ensemble)

    rows = [
        {
            "model": name,
            "ensemble_method": method,
            "mape": error.mape_value,
            "mape_threshold": error.allowed_max,
            "reason": str(error),
            "selection_metric": selection_metric,
        }
        for method, result in ensembles.items()
        for name, error in result.rejected_models.items()
    ]
    return pd.DataFrame(rows, columns=_REJECTED_COLUMNS)


def export_forecast_csvs(
    forecast_results: "Dict[str, ForecastResult]",
    ensemble: "Optional[EnsembleResult] | Dict[str, EnsembleResult]" = None,
    *,
    forecaster=None,
    per_model_mape: "Optional[Dict[str, float]]" = None,
    per_model_validation_mape: "Optional[Dict[str, float]]" = None,
    ensemble_mape: "Optional[float]" = None,
    mape_threshold: Optional[float] = _THRESHOLD_DEFAULT,
    output_dir: str = ".",
    prefix: str = "forecast",
) -> Dict[str, str]:
    """
    Write the run's CSVs to disk.

    Always written (even when a table is empty, so the output set is
    predictable), into output_dir (created if it doesn't exist):

      {prefix}_final_ensemble_{method}.csv  one per combination method
      {prefix}_final_best_model.csv  the single best model's forecast
      {prefix}_per_model.csv         every model's forecast, stacked
      {prefix}_rejected_models.csv   models excluded, per method
      {prefix}_model_summary.csv     per-model MAPE / WAPE / weight / status

    Written only with forecaster=, when evaluate() ran with two or more
    models:

      {prefix}_error_correlation.csv  correlation of the models' test errors

    Both "final" candidates are written deliberately: a combined ensemble
    is not guaranteed to beat the best individual model, so rather than
    baking that choice in, the run exports both plus the scores needed to
    choose between them.

    The simplest call passes the forecaster after evaluate() and forecast():

        export_forecast_csvs(forecaster.forecast(30), forecaster=forecaster)

    That fills in the ensemble, every score, the baseline row, the
    threshold and the selection metric from the forecaster. An argument
    given explicitly replaces the forecaster's value for that argument.
    Only per_model_validation_mape also changes how the best model is
    chosen: given explicitly, the choice is made on those MAPEs, even if
    the forecaster selected on WAPE. An explicit per_model_mape, ensemble
    or mape_threshold fills the summary and checks without changing the
    metric the choice is made on.

    Args:
        forecast_results: {model_name: ForecastResult} from forecast().
        ensemble: the forecaster's forecast_ensemble ({method:
            EnsembleResult}), a single EnsembleResult, or None. Drives the
            combined-forecast and rejected-models files.
        forecaster: the HorizonForecaster the forecasts came from. Its
            evaluate() results supply the MAPE and WAPE of each model and
            of each ensemble rule, the zero actuals MAPE left out, the
            seasonal-naive baseline and the error correlation. When it
            selected on WAPE, the best model is chosen on WAPE too.
        per_model_mape: {model_name: test-window mape} from evaluate(),
            reported in the summary and used as the second check on the
            best model (it must be within mape_threshold on test too).
        per_model_validation_mape: {model_name: validation-window mape}
            from evaluate(). The best model is the lowest of these among
            the models that pass both checks; without it the best-model
            file is written header-only, with a warning.
        ensemble_mape: fallback for the combined score when an
            EnsembleResult carries none of its own (the forecast()
            context, where no future actuals exist).
        mape_threshold: the threshold both best-model checks use. Defaults
            to forecaster.mape_threshold with forecaster=, else 40; None
            skips the checks.
        output_dir: directory to write into (created if missing).
        prefix: filename prefix for all files.

    Returns:
        {"final_ensemble_<method>": path, "final_best_model": path,
        "per_model": path, "rejected": path, "model_summary": path}, plus
        "error_correlation" when that file was written.
    """
    caller = "export_forecast_csvs"
    check_results_dict(forecast_results, "forecast", caller)
    check_score_dict(per_model_mape, "per_model_mape", caller)
    check_score_dict(per_model_validation_mape, "per_model_validation_mape", caller)
    check_optional_number(ensemble_mape, "ensemble_mape", caller)

    extra = {
        "per_model_wape": None,
        "per_model_validation_wape": None,
        "zero_actuals_excluded": None,
        "ensemble_scores": None,
        "baseline": None,
        "selection_metric": None,
        "band_coverage": None,
        "ensemble_band_coverage": None,
    }
    correlation = None
    # Explicit validation scores win over the forecaster's, the best-model
    # choice included -- even when the forecaster itself selected on WAPE.
    explicit_validation = per_model_validation_mape is not None
    if forecaster is not None:
        scores = _scores_from_forecaster(forecaster, caller)
        if ensemble is None:
            ensemble = forecaster.forecast_ensemble
        if per_model_mape is None:
            per_model_mape = scores["per_model_mape"]
        if per_model_validation_mape is None:
            per_model_validation_mape = scores["per_model_validation_mape"]
        if mape_threshold is _THRESHOLD_DEFAULT:
            mape_threshold = forecaster.mape_threshold
        extra = {key: scores[key] for key in extra}
        correlation = scores["error_correlation"]
    if mape_threshold is _THRESHOLD_DEFAULT:
        mape_threshold = DEFAULT_MAPE_THRESHOLD

    mape_threshold = validate_mape_threshold(mape_threshold)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ensembles = as_ensemble_dict(ensemble)

    # The best model is chosen on whatever the forecaster selected on, so the
    # exported "best" agrees with the models the ensemble screened.
    if (
        extra["selection_metric"] == "wape"
        and extra["per_model_validation_wape"]
        and not explicit_validation
    ):
        choose_val, choose_test, metric = (
            extra["per_model_validation_wape"],
            extra["per_model_wape"],
            "WAPE",
        )
    else:
        choose_val, choose_test, metric = per_model_validation_mape, per_model_mape, "MAPE"
    _, best_model_checks = (
        choose_best_model(forecast_results, choose_val, choose_test, mape_threshold, metric=metric)
        if forecast_results and choose_val is not None
        else (None, {})
    )

    tables = {
        "final_best_model": (
            f"{prefix}_final_best_model.csv",
            build_best_model_forecast_table(
                forecast_results,
                choose_test,
                per_model_validation_mape=choose_val,
                mape_threshold=mape_threshold,
                metric=metric,
            ),
        ),
        "per_model": (f"{prefix}_per_model.csv", build_per_model_forecast_table(forecast_results)),
        "rejected": (
            f"{prefix}_rejected_models.csv",
            build_rejected_models_table(ensembles, selection_metric=extra["selection_metric"]),
        ),
        "model_summary": (
            f"{prefix}_model_summary.csv",
            build_model_summary_table(
                per_model_mape,
                ensembles,
                ensemble_mape,
                per_model_validation_mape,
                best_model_checks=best_model_checks,
                **extra,
            ),
        ),
    }

    # One combined-forecast file per method, always suffixed with the
    # method name even when only one was requested. A filename that
    # silently changed shape depending on how many rules were asked for
    # would be worse than a slightly longer one -- downstream jobs can
    # glob a stable pattern.
    if ensembles:
        for method, result in ensembles.items():
            tables[f"final_ensemble_{method}"] = (
                f"{prefix}_final_ensemble_{method}.csv",
                build_final_forecast_table(result),
            )
    else:
        tables["final_ensemble"] = (
            f"{prefix}_final_ensemble.csv",
            build_final_forecast_table(None),
        )

    written: Dict[str, str] = {}
    for key, (filename, table) in tables.items():
        path = out / filename
        table.to_csv(path, index=False)
        written[key] = str(path)

    if correlation is not None and not correlation.empty:
        path = out / f"{prefix}_error_correlation.csv"
        correlation.rename_axis("model").to_csv(path)
        written["error_correlation"] = str(path)
    return written


def export_forecast(
    forecast_results: "Dict[str, ForecastResult]",
    ensemble: "Optional[EnsembleResult] | Dict[str, EnsembleResult]" = None,
    *,
    forecaster=None,
    per_model_mape: "Optional[Dict[str, float]]" = None,
    per_model_validation_mape: "Optional[Dict[str, float]]" = None,
    ensemble_mape: "Optional[float]" = None,
    mape_threshold: Optional[float] = _THRESHOLD_DEFAULT,
    history=None,
    date_column: str = "ds",
    target_column: str = "y",
    output_dir: str = ".",
    prefix: str = "forecast",
    save_plots: bool = True,
    dpi: int = 100,
) -> Dict[str, str]:
    """
    One-call export of a forecast run: the CSVs AND (by default) the
    forecast charts, all into the same output_dir.

    Writes the CSVs via export_forecast_csvs(), then — unless
    save_plots=False, or there are no forecasts to plot — the per-rule
    overlays and the per-model/per-rule PNGs via
    horizon_forecaster.plotting.save_forecast_plots().
    matplotlib is imported lazily here, so the CSV-only export_forecast_csvs()
    path never pays that import cost.

    Args:
        forecast_results: {model_name: ForecastResult} from forecast().
        ensemble: forecast_ensemble ({method: EnsembleResult}), a single
            EnsembleResult, or None.
        forecaster, per_model_mape, per_model_validation_mape,
        ensemble_mape, mape_threshold: as in export_forecast_csvs().
            Passing forecaster= fills in the rest, including the ensemble
            drawn on the plots.
        history: optional raw history DataFrame drawn behind the forecast
            plots; date_column/target_column name its columns.
        date_column, target_column: column names in `history`.
        output_dir: directory for every file (created if missing).
        prefix: shared filename prefix.
        save_plots: set False to write only the CSVs.
        dpi: resolution for the saved plots.

    Returns:
        A merged dict of every path written: the CSV keys
        ("final_ensemble_<method>", one per combination method -- or a
        bare "final_ensemble" when no ensemble was supplied at all --
        plus "final_best_model", "per_model", "rejected",
        "model_summary", and "error_correlation" when written), and when
        plots were saved, "plot_"-prefixed
        plot keys: one "plot_overlay_<method>" and one
        "plot_ensemble_<method>" per combination method, plus one
        "plot_<model>" per model. Asking for three rules therefore writes
        three overlays and three ensemble figures, each against the same
        models -- not one chart that silently shows only the first rule.
    """
    check_results_dict(forecast_results, "forecast", "export_forecast")
    if save_plots:
        # Before any file is written, so a bad history doesn't leave the
        # CSVs on disk without their charts.
        history = check_history(history, date_column, target_column, "export_forecast")
    if ensemble is None and forecaster is not None:
        ensemble = getattr(forecaster, "forecast_ensemble", None)
    written = export_forecast_csvs(
        forecast_results,
        ensemble,
        forecaster=forecaster,
        per_model_mape=per_model_mape,
        per_model_validation_mape=per_model_validation_mape,
        ensemble_mape=ensemble_mape,
        mape_threshold=mape_threshold,
        output_dir=output_dir,
        prefix=prefix,
    )

    if save_plots and forecast_results:
        # Lazy import: keeps the CSV-only path free of the matplotlib dependency.
        from horizon_forecaster.plotting import save_forecast_plots

        plot_paths = save_forecast_plots(
            forecast_results,
            history=history,
            date_column=date_column,
            target_column=target_column,
            ensemble=ensemble,
            output_dir=output_dir,
            prefix=prefix,
            dpi=dpi,
        )
        written.update({f"plot_{key}": path for key, path in plot_paths.items()})

    return written
