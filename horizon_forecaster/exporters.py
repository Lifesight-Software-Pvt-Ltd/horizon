"""
CSV exporters for horizon_forecaster.

Turns the outputs of HorizonForecaster.forecast() (plus its ensemble)
into tabular CSVs. Like plotting.py, this is a pure, standalone module:
it never imports forecaster.py or any model/adapter class, and reaches
into the result objects only through their public attributes, so writing
a CSV never pulls in the heavy Prophet/CatBoost stack. Type hints for the
result objects are TYPE_CHECKING-only; at runtime the inputs are
duck-typed.

Five deliverables:

1. Final combined forecast -- the ensemble's ["ds", "yhat"] series: the
   single blended number the caller ultimately acts on. One file per
   combination method, since several rules can be built in one run.
2. Best single model's forecast -- exported alongside the combined one,
   because a combination is not guaranteed to beat the best individual
   model, so both candidates ship with the scores needed to choose. The
   model is chosen on VALIDATION MAPE among the models that pass
   mape_threshold on both validation and test (see choose_best_model).
3. Per-model forecast -- every model's own forecast, stacked into one
   long/tidy table with a "model" column (+ bounds where a model
   provides them), so all models sit in one file for easy comparison.
4. Model summary -- per model, its validation and test MAPE, ensemble
   membership, weight (where the rule has one) and status.
5. Rejected-models table -- one row per model excluded from a
   combination, with its MAPE, the threshold it exceeded (if any) and
   the human-readable reason. This is the "why isn't model X in the
   blend?" report, built straight from EnsembleResult.rejected_models.
   Note exclusion is per method: a model with a NaN validation MAPE is
   dropped by the weighted rules but kept by the trimmed one.

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
_REJECTED_COLUMNS = ["model", "ensemble_method", "mape", "mape_threshold", "reason"]
_SUMMARY_COLUMNS = [
    "model",
    "validation_mape",
    "mape",
    "ensemble_method",
    "rule_applied",
    "ensemble_weight",
    "status",
    "best_model_check",
]

# Matches HorizonForecaster's default mape_threshold. Pass the forecaster's
# own value (forecaster.mape_threshold) when it was changed.
DEFAULT_MAPE_THRESHOLD = 40.0


def build_final_forecast_table(
    ensemble: "Optional[EnsembleResult] | Dict[str, EnsembleResult]",
) -> pd.DataFrame:
    """
    The final blended forecast as a ["ds", "yhat"] table.

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
    return result.combined[_FINAL_COLUMNS].reset_index(drop=True)


def build_per_model_forecast_table(
    forecast_results: "Dict[str, ForecastResult]",
) -> pd.DataFrame:
    """
    Every model's forecast stacked into one long table.

    Columns: ["model", "ds", "yhat"], plus "yhat_lower"/"yhat_upper" when
    at least one model supplies them (models that don't, e.g. CatBoost,
    get NaN in those columns). Returns a header-only DataFrame when
    forecast_results is empty.
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
) -> Tuple[Optional[str], Dict[str, str]]:
    """
    Pick the model exported as "best", and say why each model was or was not.

    Two checks, then one choice:
      1. validation MAPE at or below mape_threshold -- the same screen the
         ensemble uses;
      2. test MAPE at or below mape_threshold, when test scores are given;
      then the lowest VALIDATION MAPE among the models that pass both.

    It used to be the lowest TEST MAPE with no threshold at all, so the
    export was chosen on the window that is supposed to be reported, never
    selected on -- and a model the ensemble had rejected was exported at
    103% error in a real run. Choosing on test was also measured directly
    and picked worse (docs/design_notes.md, "Window sizes"). Test is kept
    only as a check that rules a model out: a model that passed validation
    but scored above the threshold on test kept doing badly in the periods
    that followed (median 24.8% against 5.6% for the rest). The trade-off,
    stated in the note: the exported model's test MAPE is slightly
    optimistic, because test helped decide which model it is.

    mape_threshold=None skips both checks; a model without a finite
    validation MAPE is never chosen, since there is nothing to choose on.

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
            checks[name] = "not chosen: no validation MAPE"
            continue
        if mape_threshold is not None and val > mape_threshold:
            checks[name] = f"not chosen: validation MAPE {val:.2f} > {mape_threshold:g}"
            continue
        if per_model_mape is not None:
            test = per_model_mape.get(name)
            if not _finite(test):
                checks[name] = "not chosen: no test MAPE"
                continue
            if mape_threshold is not None and test > mape_threshold:
                checks[name] = f"not chosen: test MAPE {test:.2f} > {mape_threshold:g}"
                continue
        checks[name] = "passed"
        eligible[name] = val

    best = min(eligible, key=eligible.get) if eligible else None
    if best is not None and mape_threshold is None:
        checks[best] = (
            "exported as best: lowest validation MAPE (mape_threshold is None, no checks)"
        )
    elif best is not None:
        checks[best] = "exported as best: lowest validation MAPE among the models that passed"
        if per_model_mape is not None:
            checks[best] += (
                " both checks (test MAPE was used as a check, so it is slightly optimistic "
                "for this model)"
            )
        else:
            checks[best] += " the validation check (no test MAPEs were given)"
    return best, checks


def build_best_model_forecast_table(
    forecast_results: "Dict[str, ForecastResult]",
    per_model_mape: "Optional[Dict[str, float]]" = None,
    *,
    per_model_validation_mape: "Optional[Dict[str, float]]" = None,
    mape_threshold: Optional[float] = DEFAULT_MAPE_THRESHOLD,
) -> pd.DataFrame:
    """
    The best single model's forecast, as a ["model", "ds", "yhat"] table
    (+ bounds where that model provides them).

    Exists because a weighted ensemble is NOT guaranteed to beat the best
    individual model (see evaluation/ensemble.py) -- exporting only the
    blend can mean shipping the worse forecast. This is written alongside
    the ensemble so the choice can be made per run rather than baked in.

    The model is chosen by choose_best_model(): validation MAPE among the
    models within mape_threshold on validation and on test. `per_model_mape`
    is the test-window MAPE and `per_model_validation_mape` the validation
    MAPE, both from evaluate(). Returns a header-only DataFrame, with a
    warning saying why, when no model qualifies; the ensemble files are
    unaffected.
    """
    check_results_dict(forecast_results, "forecast", "build_best_model_forecast_table")
    if not forecast_results:
        return pd.DataFrame(columns=_PER_MODEL_BASE_COLUMNS)
    if per_model_validation_mape is None:
        warnings.warn(
            "The best-model file is written empty: the best model is chosen on validation "
            "MAPE, and per_model_validation_mape was not given. Pass "
            "{m: r.validation_mape.value for m, r in results.items()} from evaluate().",
            stacklevel=3,
        )
        return pd.DataFrame(columns=_PER_MODEL_BASE_COLUMNS)

    best, checks = choose_best_model(
        forecast_results, per_model_validation_mape, per_model_mape, mape_threshold
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
) -> pd.DataFrame:
    """
    The at-a-glance table for deciding whether to act on a combined
    forecast or a single model.

    One row per (model, ensemble method) pair, plus one row per method for
    the combined result itself. With the default single method that is the
    familiar "one row per model plus an ensemble row"; asking for several
    methods simply repeats the block per method, so the shape never
    depends on how many were requested.

    Columns: ["model", "validation_mape", "mape", "ensemble_method",
    "rule_applied", "ensemble_weight", "status", "best_model_check"].

    `best_model_check` says, per model, whether it was exported as the best
    single model and why not otherwise (from choose_best_model); empty on
    the ensemble rows and when no checks were given.

    Both MAPEs are reported because they answer different questions:
    `validation_mape` is what the model was SELECTED and screened on,
    `mape` is its honest out-of-sample score on the test window. A model
    with a large gap between them was flattered by the selection window.

    `ensemble_weight` is empty for weightless rules such as the default
    trimmed mean. That is deliberate rather than a gap in the export --
    there is no weight, and filling in an equal share would imply an
    averaging that did not happen.
    """
    check_score_dict(per_model_mape, "per_model_mape", "build_model_summary_table")
    check_score_dict(
        per_model_validation_mape, "per_model_validation_mape", "build_model_summary_table"
    )
    check_optional_number(ensemble_mape, "ensemble_mape", "build_model_summary_table")
    ensembles = as_ensemble_dict(ensemble)
    per_model_mape = per_model_mape or {}
    per_model_validation_mape = per_model_validation_mape or {}
    best_model_checks = best_model_checks or {}

    if not ensembles:
        rows = [
            {
                "model": name,
                "validation_mape": per_model_validation_mape.get(name),
                "mape": value,
                "ensemble_method": None,
                "rule_applied": None,
                "ensemble_weight": None,
                "status": "not_in_ensemble",
                "best_model_check": best_model_checks.get(name),
            }
            for name, value in per_model_mape.items()
        ]
        return pd.DataFrame(rows, columns=_SUMMARY_COLUMNS)

    rows = []
    for method, result in ensembles.items():
        weights = result.weights or {}
        for name, value in per_model_mape.items():
            if name in result.members:
                status = "in_ensemble"
            elif name in result.rejected_models:
                status = "rejected"
            else:
                status = "not_in_ensemble"
            rows.append(
                {
                    "model": name,
                    "validation_mape": per_model_validation_mape.get(name),
                    "mape": value,
                    "ensemble_method": method,
                    "rule_applied": result.rule_applied if status == "in_ensemble" else None,
                    "ensemble_weight": weights.get(name),
                    "status": status,
                    "best_model_check": best_model_checks.get(name),
                }
            )

        # The scalar `ensemble_mape` fallback describes ONE combined series,
        # so it is only unambiguous when one method was built. With several,
        # stamping it on every method would report the same score against
        # three materially different series -- and comparing the rules is
        # exactly what this table is for. Leave those blank instead.
        if result.mape is not None:
            scored = result.mape.value
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
                }
            )

    return pd.DataFrame(rows, columns=_SUMMARY_COLUMNS)


def build_rejected_models_table(
    ensemble: "Optional[EnsembleResult] | Dict[str, EnsembleResult]",
) -> pd.DataFrame:
    """
    One row per model excluded from a combination, per method.

    Columns: ["model", "ensemble_method", "mape", "mape_threshold",
    "reason"]. Built from EnsembleResult.rejected_models (a dict of
    ThresholdRejectionError). `mape` may be NaN for a model whose MAPE
    itself was NaN (e.g. every actual in the window was zero).

    The `ensemble_method` column matters because exclusion is not
    necessarily the same across rules: a model with a NaN validation MAPE
    cannot be weighted, so the weighted rules drop it, while the trimmed
    rule needs no MAPE and keeps it. Returns a header-only DataFrame when
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
        }
        for method, result in ensembles.items()
        for name, error in result.rejected_models.items()
    ]
    return pd.DataFrame(rows, columns=_REJECTED_COLUMNS)


def export_forecast_csvs(
    forecast_results: "Dict[str, ForecastResult]",
    ensemble: "Optional[EnsembleResult] | Dict[str, EnsembleResult]" = None,
    *,
    per_model_mape: "Optional[Dict[str, float]]" = None,
    per_model_validation_mape: "Optional[Dict[str, float]]" = None,
    ensemble_mape: "Optional[float]" = None,
    mape_threshold: Optional[float] = DEFAULT_MAPE_THRESHOLD,
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
      {prefix}_model_summary.csv     per-model MAPEs / weight / status

    Both "final" candidates are written deliberately: a combined ensemble
    is not guaranteed to beat the best individual model, so rather than
    baking that choice in, the run exports both plus the scores needed to
    choose between them.

    Args:
        forecast_results: {model_name: ForecastResult} from forecast().
        ensemble: the forecaster's forecast_ensemble ({method:
            EnsembleResult}), a single EnsembleResult, or None. Drives the
            combined-forecast and rejected-models files.
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
        mape_threshold: the threshold both best-model checks use. Pass the
            forecaster's own (forecaster.mape_threshold) if you changed it;
            None skips the checks.
        output_dir: directory to write into (created if missing).
        prefix: filename prefix for all files.

    Returns:
        {"final_ensemble_<method>": path, "final_best_model": path,
        "per_model": path, "rejected": path, "model_summary": path}.
    """
    check_results_dict(forecast_results, "forecast", "export_forecast_csvs")
    check_score_dict(per_model_mape, "per_model_mape", "export_forecast_csvs")
    check_score_dict(per_model_validation_mape, "per_model_validation_mape", "export_forecast_csvs")
    check_optional_number(ensemble_mape, "ensemble_mape", "export_forecast_csvs")
    mape_threshold = validate_mape_threshold(mape_threshold)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ensembles = as_ensemble_dict(ensemble)
    _, best_model_checks = (
        choose_best_model(
            forecast_results, per_model_validation_mape, per_model_mape, mape_threshold
        )
        if forecast_results and per_model_validation_mape is not None
        else (None, {})
    )

    tables = {
        "final_best_model": (
            f"{prefix}_final_best_model.csv",
            build_best_model_forecast_table(
                forecast_results,
                per_model_mape,
                per_model_validation_mape=per_model_validation_mape,
                mape_threshold=mape_threshold,
            ),
        ),
        "per_model": (f"{prefix}_per_model.csv", build_per_model_forecast_table(forecast_results)),
        "rejected": (f"{prefix}_rejected_models.csv", build_rejected_models_table(ensembles)),
        "model_summary": (
            f"{prefix}_model_summary.csv",
            build_model_summary_table(
                per_model_mape,
                ensembles,
                ensemble_mape,
                per_model_validation_mape,
                best_model_checks=best_model_checks,
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
    return written


def export_forecast(
    forecast_results: "Dict[str, ForecastResult]",
    ensemble: "Optional[EnsembleResult] | Dict[str, EnsembleResult]" = None,
    *,
    per_model_mape: "Optional[Dict[str, float]]" = None,
    per_model_validation_mape: "Optional[Dict[str, float]]" = None,
    ensemble_mape: "Optional[float]" = None,
    mape_threshold: Optional[float] = DEFAULT_MAPE_THRESHOLD,
    history=None,
    date_column: str = "ds",
    target_column: str = "y",
    output_dir: str = ".",
    prefix: str = "forecast",
    save_plots: bool = True,
    dpi: int = 100,
) -> Dict[str, str]:
    """
    One-call export of a forecast run: the five CSVs AND (by default) the
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
        per_model_mape, per_model_validation_mape, ensemble_mape,
        mape_threshold: as in export_forecast_csvs(). The best-model file
            needs per_model_validation_mape, and uses mape_threshold for
            its two checks -- pass forecaster.mape_threshold if you changed
            it.
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
        "model_summary"), and when plots were saved, "plot_"-prefixed
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
    written = export_forecast_csvs(
        forecast_results,
        ensemble,
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
