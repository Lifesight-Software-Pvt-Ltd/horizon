"""
CSV exporters for horizon_forecaster (Phase 5, output layer).

Turns the outputs of HorizonForecaster.forecast() (plus its ensemble)
into tabular CSVs. Like plotting.py, this is a pure, standalone module:
it never imports forecaster.py or any model/adapter class, and reaches
into the result objects only through their public attributes, so writing
a CSV never pulls in the heavy Prophet/CatBoost stack. Type hints for the
result objects are TYPE_CHECKING-only; at runtime the inputs are
duck-typed.

Three deliverables, matching the Phase 5 spec:

1. Final forecast  -- the ensemble's combined ["ds", "yhat"] series: the
   single blended number the caller ultimately acts on.
2. Per-model forecast -- every model's own forecast, stacked into one
   long/tidy table with a "model" column (+ bounds where a model
   provides them), so all models sit in one file for easy comparison.
3. Rejected-models table -- one row per model excluded from the ensemble
   by the MAPE threshold, with its MAPE, the threshold it exceeded, and
   the human-readable reason. This is the "why isn't model X in the
   blend?" report, built straight from EnsembleResult.rejected_models.

Each of these has a pure "build the DataFrame" function (returns a
DataFrame, writes nothing) plus one convenience writer, export_forecast_csvs(),
that persists all three at once. The builders never raise on empty/None
input -- they return a correctly-columned empty DataFrame -- so a run
where (say) every model was rejected still produces predictable,
header-only files rather than a crash or a missing file.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional

import pandas as pd

if TYPE_CHECKING:  # hints only; never imported at runtime (avoids the heavy stack)
    from horizon_forecaster.evaluation.ensemble import EnsembleResult
    from horizon_forecaster.forecaster import ForecastResult

_FINAL_COLUMNS = ["ds", "yhat"]
_PER_MODEL_BASE_COLUMNS = ["model", "ds", "yhat"]
_REJECTED_COLUMNS = ["model", "mape", "mape_threshold", "reason"]
_SUMMARY_COLUMNS = ["model", "mape", "ensemble_weight", "status"]


def build_final_forecast_table(ensemble: "Optional[EnsembleResult]") -> pd.DataFrame:
    """
    The final blended forecast as a ["ds", "yhat"] table.

    Returns an empty (header-only) DataFrame when no ensemble was formed
    -- ensemble is None, or every model was rejected so ensemble.combined
    is None -- so a caller always gets a well-shaped table.
    """
    if ensemble is None or ensemble.combined is None:
        return pd.DataFrame(columns=_FINAL_COLUMNS)
    return ensemble.combined[_FINAL_COLUMNS].reset_index(drop=True)


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
    ensemble: "Optional[EnsembleResult]" = None,
    ensemble_mape: "Optional[float]" = None,
) -> pd.DataFrame:
    """
    One row per model with its MAPE, ensemble weight and status, plus an
    "ensemble" row when its own score is supplied -- the at-a-glance table
    for deciding whether to act on the blend or a single model.

    Columns: ["model", "mape", "ensemble_weight", "status"], where status
    is "in_ensemble", "rejected" (excluded by the MAPE threshold), or
    "not_in_ensemble".
    """
    weights = ensemble.weights if ensemble is not None else {}
    rejected = ensemble.rejected_models if ensemble is not None else {}

    rows = []
    for name, value in (per_model_mape or {}).items():
        if name in weights:
            status = "in_ensemble"
        elif name in rejected:
            status = "rejected"
        else:
            status = "not_in_ensemble"
        rows.append({
            "model": name,
            "mape": value,
            "ensemble_weight": weights.get(name),
            "status": status,
        })

    if ensemble_mape is not None:
        rows.append({
            "model": "ensemble",
            "mape": ensemble_mape,
            "ensemble_weight": 1.0,
            "status": "ensemble",
        })

    return pd.DataFrame(rows, columns=_SUMMARY_COLUMNS)


def build_rejected_models_table(ensemble: "Optional[EnsembleResult]") -> pd.DataFrame:
    """
    One row per model excluded from the ensemble by the MAPE threshold.

    Columns: ["model", "mape", "mape_threshold", "reason"]. Built from
    EnsembleResult.rejected_models (a dict of ThresholdRejectionError).
    `mape` may be NaN for a model whose MAPE itself was NaN (e.g. every
    test-set actual was zero). Returns a header-only DataFrame when there
    is no ensemble or nothing was rejected.
    """
    if ensemble is None or not ensemble.rejected_models:
        return pd.DataFrame(columns=_REJECTED_COLUMNS)

    rows = [
        {
            "model": name,
            "mape": error.mape_value,
            "mape_threshold": error.allowed_max,
            "reason": str(error),
        }
        for name, error in ensemble.rejected_models.items()
    ]
    return pd.DataFrame(rows, columns=_REJECTED_COLUMNS)


def export_forecast_csvs(
    forecast_results: "Dict[str, ForecastResult]",
    ensemble: "Optional[EnsembleResult]" = None,
    *,
    per_model_mape: "Optional[Dict[str, float]]" = None,
    ensemble_mape: "Optional[float]" = None,
    output_dir: str = ".",
    prefix: str = "forecast",
) -> Dict[str, str]:
    """
    Write the run's CSVs to disk.

    Five files, always written (even when a table is empty, so the output
    set is predictable), into output_dir (created if it doesn't exist):

      {prefix}_final_ensemble.csv    the ensemble's blended forecast
      {prefix}_final_best_model.csv  the single lowest-MAPE model's forecast
      {prefix}_per_model.csv         every model's forecast, stacked
      {prefix}_rejected_models.csv   models excluded by the MAPE threshold
      {prefix}_model_summary.csv     per-model MAPE / weight / status

    Both "final" candidates are written deliberately: a weighted ensemble
    is not guaranteed to beat the best individual model, so rather than
    baking that choice in, the run exports both plus the scores needed to
    choose between them.

    Args:
        forecast_results: {model_name: ForecastResult} from forecast().
        ensemble: the forecaster's forecast_ensemble (EnsembleResult), or
            None. Drives the ensemble and rejected-models files.
        per_model_mape: {model_name: mape} from evaluate(), used to pick
            the best model and populate the summary. Without it those two
            files are written header-only.
        ensemble_mape: the ensemble's own MAPE (from evaluate()), added as
            an "ensemble" row in the summary for direct comparison.
        output_dir: directory to write into (created if missing).
        prefix: filename prefix for all files.

    Returns:
        {"final_ensemble": path, "final_best_model": path, "per_model":
        path, "rejected": path, "model_summary": path}.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tables = {
        "final_ensemble": (f"{prefix}_final_ensemble.csv", build_final_forecast_table(ensemble)),
        "final_best_model": (
            f"{prefix}_final_best_model.csv",
            build_best_model_forecast_table(forecast_results, per_model_mape),
        ),
        "per_model": (f"{prefix}_per_model.csv", build_per_model_forecast_table(forecast_results)),
        "rejected": (f"{prefix}_rejected_models.csv", build_rejected_models_table(ensemble)),
        "model_summary": (
            f"{prefix}_model_summary.csv",
            build_model_summary_table(per_model_mape, ensemble, ensemble_mape),
        ),
    }

    written: Dict[str, str] = {}
    for key, (filename, table) in tables.items():
        path = out / filename
        table.to_csv(path, index=False)
        written[key] = str(path)
    return written


def export_forecast(
    forecast_results: "Dict[str, ForecastResult]",
    ensemble: "Optional[EnsembleResult]" = None,
    *,
    per_model_mape: "Optional[Dict[str, float]]" = None,
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
    save_plots=False, or there are no forecasts to plot — the overlay and
    per-model PNGs via horizon_forecaster.plotting.save_forecast_plots().
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
        A merged dict of every path written: the CSV keys ("final_ensemble",
        "final_best_model", "per_model", "rejected", "model_summary") plus,
        when plots were saved, "plot_"-prefixed plot keys ("plot_overlay",
        "plot_ensemble", and one "plot_<model>" per model).
    """
    written = export_forecast_csvs(
        forecast_results,
        ensemble,
        per_model_mape=per_model_mape,
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
