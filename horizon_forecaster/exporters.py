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
    output_dir: str = ".",
    prefix: str = "forecast",
) -> Dict[str, str]:
    """
    Write the final, per-model, and rejected-models CSVs to disk.

    All three files are always written (even when a table is empty, so the
    set of output files is predictable), into output_dir (created if it
    doesn't exist), named "{prefix}_final.csv", "{prefix}_per_model.csv",
    and "{prefix}_rejected_models.csv".

    Args:
        forecast_results: {model_name: ForecastResult} from forecast().
        ensemble: the forecaster's forecast_ensemble (EnsembleResult), or
            None. Drives the final-forecast and rejected-models files.
        output_dir: directory to write into (created if missing).
        prefix: filename prefix for all three files.

    Returns:
        {"final": path, "per_model": path, "rejected": path} -- the string
        paths actually written.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tables = {
        "final": (f"{prefix}_final.csv", build_final_forecast_table(ensemble)),
        "per_model": (f"{prefix}_per_model.csv", build_per_model_forecast_table(forecast_results)),
        "rejected": (f"{prefix}_rejected_models.csv", build_rejected_models_table(ensemble)),
    }

    written: Dict[str, str] = {}
    for key, (filename, table) in tables.items():
        path = out / filename
        table.to_csv(path, index=False)
        written[key] = str(path)
    return written
