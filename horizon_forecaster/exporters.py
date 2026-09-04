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
   model, so both candidates ship with the scores needed to choose.
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

from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional

import pandas as pd

from horizon_forecaster.evaluation.ensemble import as_ensemble_dict

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
]


def build_final_forecast_table(
    ensemble: "Optional[EnsembleResult] | Dict[str, EnsembleResult]",
) -> pd.DataFrame:
    """
    The final blended forecast as a ["ds", "yhat"] table.

    Accepts the {method: EnsembleResult} dict a forecaster actually
    produces, a single EnsembleResult, or None. Given several methods it
    tables the FIRST one, matching plotting._primary_ensemble -- a single
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


def build_best_model_forecast_table(
    forecast_results: "Dict[str, ForecastResult]",
    per_model_mape: "Optional[Dict[str, float]]" = None,
) -> pd.DataFrame:
    """
    The single best-scoring model's forecast, as a ["model", "ds", "yhat"]
    table (+ bounds where that model provides them).

    Exists because a weighted ensemble is NOT guaranteed to beat the best
    individual model (see evaluation/ensemble.py) -- exporting only the
    blend can mean shipping the worse forecast. This is written alongside
    the ensemble so the choice can be made per run rather than baked in.

    `per_model_mape` is {model_name: mape} from evaluate(); the lowest
    finite MAPE wins. Returns a header-only DataFrame when no scores (or
    no forecasts) are available.
    """
    if not forecast_results or not per_model_mape:
        return pd.DataFrame(columns=_PER_MODEL_BASE_COLUMNS)

    scored = {
        name: value
        for name, value in per_model_mape.items()
        if name in forecast_results and value == value  # drop NaN
    }
    if not scored:
        return pd.DataFrame(columns=_PER_MODEL_BASE_COLUMNS)

    best = min(scored, key=scored.get)
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
    "rule_applied", "ensemble_weight", "status"].

    Both MAPEs are reported because they answer different questions:
    `validation_mape` is what the model was SELECTED and screened on,
    `mape` is its honest out-of-sample score on the test window. A model
    with a large gap between them was flattered by the selection window.

    `ensemble_weight` is empty for weightless rules such as the default
    trimmed mean. That is deliberate rather than a gap in the export --
    there is no weight, and filling in an equal share would imply an
    averaging that did not happen.
    """
    ensembles = as_ensemble_dict(ensemble)
    per_model_mape = per_model_mape or {}
    per_model_validation_mape = per_model_validation_mape or {}

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
            used to pick the best model and populate the summary. Without
            it those two files are written header-only.
        per_model_validation_mape: {model_name: validation-window mape},
            reported alongside so the selection-vs-reported gap is visible.
        ensemble_mape: fallback for the combined score when an
            EnsembleResult carries none of its own (the forecast()
            context, where no future actuals exist).
        output_dir: directory to write into (created if missing).
        prefix: filename prefix for all files.

    Returns:
        {"final_ensemble_<method>": path, "final_best_model": path,
        "per_model": path, "rejected": path, "model_summary": path}.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ensembles = as_ensemble_dict(ensemble)

    tables = {
        "final_best_model": (
            f"{prefix}_final_best_model.csv",
            build_best_model_forecast_table(forecast_results, per_model_mape),
        ),
        "per_model": (f"{prefix}_per_model.csv", build_per_model_forecast_table(forecast_results)),
        "rejected": (f"{prefix}_rejected_models.csv", build_rejected_models_table(ensembles)),
        "model_summary": (
            f"{prefix}_model_summary.csv",
            build_model_summary_table(
                per_model_mape, ensembles, ensemble_mape, per_model_validation_mape
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
        ensemble: forecast_ensemble (EnsembleResult), or None.
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
    written = export_forecast_csvs(
        forecast_results,
        ensemble,
        per_model_mape=per_model_mape,
        per_model_validation_mape=per_model_validation_mape,
        ensemble_mape=ensemble_mape,
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
