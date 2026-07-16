import math

import pandas as pd
import pytest
from types import SimpleNamespace

from horizon_forecaster.exporters import (
    build_final_forecast_table,
    build_per_model_forecast_table,
    build_rejected_models_table,
    export_forecast,
    export_forecast_csvs,
)
from horizon_forecaster.evaluation.ensemble import EnsembleResult
from horizon_forecaster.exceptions import ThresholdRejectionError


def _ds(n=3):
    return pd.date_range("2023-01-01", periods=n, freq="D")


def make_forecast_result(yhats, *, lower=None, upper=None):
    n = len(yhats)
    cols = {"ds": _ds(n), "yhat": yhats}
    if lower is not None:
        cols["yhat_lower"] = lower
        cols["yhat_upper"] = upper
    return SimpleNamespace(forecast=pd.DataFrame(cols))


def make_ensemble(*, combined=True, rejected=None, n=3):
    combined_df = pd.DataFrame({"ds": _ds(n), "yhat": [12.0, 20.0, 32.0][:n]}) if combined else None
    return EnsembleResult(
        combined=combined_df,
        weights={"prophet": 1.0} if combined else {},
        rejected_models=rejected or {},
        weighting="inverse_mape",
        mape=None,
    )


# ---------- final forecast table ----------

def test_build_final_forecast_table_returns_ds_yhat():
    ens = make_ensemble()
    table = build_final_forecast_table(ens)
    assert list(table.columns) == ["ds", "yhat"]
    assert list(table["yhat"]) == [12.0, 20.0, 32.0]


def test_build_final_forecast_table_none_ensemble_is_empty_but_shaped():
    table = build_final_forecast_table(None)
    assert list(table.columns) == ["ds", "yhat"]
    assert len(table) == 0


def test_build_final_forecast_table_combined_none_is_empty():
    table = build_final_forecast_table(make_ensemble(combined=False))
    assert len(table) == 0


# ---------- per-model forecast table ----------

def test_build_per_model_table_stacks_all_models():
    results = {
        "prophet": make_forecast_result([1.0, 2.0, 3.0]),
        "catboost": make_forecast_result([4.0, 5.0, 6.0]),
    }
    table = build_per_model_forecast_table(results)
    assert len(table) == 6
    assert set(table["model"]) == {"prophet", "catboost"}
    assert list(table.columns[:3]) == ["model", "ds", "yhat"]


def test_build_per_model_table_keeps_bounds_and_nans_the_rest():
    results = {
        "prophet": make_forecast_result([1.0, 2.0, 3.0], lower=[0.5, 1.5, 2.5], upper=[1.5, 2.5, 3.5]),
        "catboost": make_forecast_result([4.0, 5.0, 6.0]),  # no bounds
    }
    table = build_per_model_forecast_table(results)
    assert "yhat_lower" in table.columns and "yhat_upper" in table.columns
    assert table.loc[table["model"] == "catboost", "yhat_lower"].isna().all()
    assert not table.loc[table["model"] == "prophet", "yhat_lower"].isna().any()


def test_build_per_model_table_empty_is_shaped():
    table = build_per_model_forecast_table({})
    assert list(table.columns) == ["model", "ds", "yhat"]
    assert len(table) == 0


# ---------- rejected models table ----------

def test_build_rejected_models_table_content():
    err = ThresholdRejectionError(
        "catboost's MAPE (90.00%) exceeds the mape_threshold of 40%.",
        model_name="catboost",
        mape_value=90.0,
        allowed_max=40.0,
    )
    table = build_rejected_models_table(make_ensemble(rejected={"catboost": err}))
    assert list(table.columns) == ["model", "mape", "mape_threshold", "reason"]
    row = table.iloc[0]
    assert row["model"] == "catboost"
    assert row["mape"] == 90.0
    assert row["mape_threshold"] == 40.0
    assert "exceeds the mape_threshold" in row["reason"]


def test_build_rejected_models_table_handles_nan_mape():
    err = ThresholdRejectionError(
        "nan model", model_name="m", mape_value=float("nan"), allowed_max=40.0
    )
    table = build_rejected_models_table(make_ensemble(rejected={"m": err}))
    assert math.isnan(table.iloc[0]["mape"])


def test_build_rejected_models_table_empty_when_none_rejected():
    assert len(build_rejected_models_table(make_ensemble())) == 0
    assert len(build_rejected_models_table(None)) == 0


# ---------- export_forecast_csvs ----------

def test_export_writes_three_readable_files(tmp_path):
    results = {"prophet": make_forecast_result([1.0, 2.0, 3.0])}
    err = ThresholdRejectionError("bad", model_name="catboost", mape_value=90.0, allowed_max=40.0)
    ens = make_ensemble(rejected={"catboost": err})

    written = export_forecast_csvs(results, ens, output_dir=str(tmp_path))

    assert set(written) == {"final", "per_model", "rejected"}
    for path in written.values():
        assert pd.read_csv(path) is not None  # re-reads without error

    final = pd.read_csv(written["final"])
    assert list(final["yhat"]) == [12.0, 20.0, 32.0]
    rejected = pd.read_csv(written["rejected"])
    assert rejected.iloc[0]["model"] == "catboost"


def test_export_creates_missing_nested_dir(tmp_path):
    target = tmp_path / "a" / "b"
    written = export_forecast_csvs({"prophet": make_forecast_result([1.0])}, make_ensemble(), output_dir=str(target))
    assert (target / "forecast_final.csv").exists()
    assert written["final"].endswith("forecast_final.csv")


def test_export_uses_custom_prefix(tmp_path):
    written = export_forecast_csvs(
        {"prophet": make_forecast_result([1.0])}, make_ensemble(),
        output_dir=str(tmp_path), prefix="run42",
    )
    assert (tmp_path / "run42_final.csv").exists()
    assert (tmp_path / "run42_per_model.csv").exists()
    assert (tmp_path / "run42_rejected_models.csv").exists()


def test_export_still_writes_all_files_when_empty(tmp_path):
    # No ensemble, no forecasts: three header-only files should still exist.
    written = export_forecast_csvs({}, None, output_dir=str(tmp_path))
    for path in written.values():
        df = pd.read_csv(path)
        assert len(df) == 0


# ---------- export_forecast bundle (CSVs + plots) ----------

def test_export_forecast_writes_csvs_and_plots(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    results = {"prophet": make_forecast_result([1.0, 2.0, 3.0])}
    written = export_forecast(results, make_ensemble(), output_dir=str(tmp_path))
    # CSV keys + plot keys
    assert {"final", "per_model", "rejected"} <= set(written)
    assert {"plot_overlay", "plot_grid"} <= set(written)
    for name in ("forecast_final.csv", "forecast_overlay.png", "forecast_grid.png"):
        assert (tmp_path / name).exists()


def test_export_forecast_save_plots_false_writes_only_csvs(tmp_path):
    results = {"prophet": make_forecast_result([1.0, 2.0, 3.0])}
    written = export_forecast(results, make_ensemble(), output_dir=str(tmp_path), save_plots=False)
    assert set(written) == {"final", "per_model", "rejected"}
    assert not list(tmp_path.glob("*.png"))


def test_export_forecast_empty_still_writes_csvs_no_plots(tmp_path):
    # No forecasts: CSVs written (header-only), plots skipped rather than raising.
    written = export_forecast({}, None, output_dir=str(tmp_path))
    assert set(written) == {"final", "per_model", "rejected"}
    assert not list(tmp_path.glob("*.png"))
