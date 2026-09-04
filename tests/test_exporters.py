import math

import pandas as pd
from types import SimpleNamespace

from horizon_forecaster.exporters import (
    _REJECTED_COLUMNS,
    _SUMMARY_COLUMNS,
    build_best_model_forecast_table,
    build_final_forecast_table,
    build_model_summary_table,
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
        method="inverse_mape",
        rule_applied="inverse_mape" if combined else "none",
        members=["prophet"] if combined else [],
        weights={"prophet": 1.0} if combined else None,
        rejected_models=rejected or {},
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
        "prophet": make_forecast_result(
            [1.0, 2.0, 3.0], lower=[0.5, 1.5, 2.5], upper=[1.5, 2.5, 3.5]
        ),
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


# ---------- best-model forecast table ----------


def test_build_best_model_table_picks_lowest_mape():
    results = {
        "prophet": make_forecast_result([1.0, 2.0, 3.0]),
        "catboost": make_forecast_result([4.0, 5.0, 6.0]),
    }
    table = build_best_model_forecast_table(results, {"prophet": 29.7, "catboost": 21.3})
    assert set(table["model"]) == {"catboost"}  # lower MAPE wins
    assert list(table["yhat"]) == [4.0, 5.0, 6.0]


def test_build_best_model_table_ignores_nan_scores():
    results = {"a": make_forecast_result([1.0]), "b": make_forecast_result([2.0])}
    table = build_best_model_forecast_table(results, {"a": float("nan"), "b": 40.0})
    assert set(table["model"]) == {"b"}


def test_build_best_model_table_empty_without_scores():
    results = {"a": make_forecast_result([1.0])}
    assert len(build_best_model_forecast_table(results, None)) == 0
    assert len(build_best_model_forecast_table({}, {"a": 10.0})) == 0


# ---------- model summary table ----------


def test_build_model_summary_reports_mape_weight_and_status():
    err = ThresholdRejectionError("bad", model_name="sarimax", mape_value=1403.0, allowed_max=30.0)
    ens = make_ensemble(rejected={"sarimax": err})  # weights = {"prophet": 1.0}
    table = build_model_summary_table(
        {"prophet": 29.71, "sarimax": 1403.0}, ens, ensemble_mape=22.40
    )
    assert list(table.columns) == _SUMMARY_COLUMNS
    by_model = table.set_index("model")
    assert by_model.loc["prophet", "status"] == "in_ensemble"
    assert by_model.loc["prophet", "ensemble_weight"] == 1.0
    assert by_model.loc["sarimax", "status"] == "rejected"
    assert by_model.loc["ensemble", "mape"] == 22.40


def test_build_model_summary_empty_is_shaped():
    table = build_model_summary_table(None, None)
    assert list(table.columns) == _SUMMARY_COLUMNS
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
    assert list(table.columns) == _REJECTED_COLUMNS
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


def test_export_writes_all_five_readable_files(tmp_path):
    results = {"prophet": make_forecast_result([1.0, 2.0, 3.0])}
    err = ThresholdRejectionError("bad", model_name="catboost", mape_value=90.0, allowed_max=40.0)
    ens = make_ensemble(rejected={"catboost": err})

    written = export_forecast_csvs(results, ens, output_dir=str(tmp_path))

    assert set(written) == {
        "final_ensemble_inverse_mape",
        "final_best_model",
        "per_model",
        "rejected",
        "model_summary",
    }
    for path in written.values():
        # pd.read_csv never returns None -- it raises or returns a frame.
        # Assert the file has real columns instead of merely not crashing.
        assert not pd.read_csv(path).columns.empty

    final = pd.read_csv(written["final_ensemble_inverse_mape"])
    assert list(final["yhat"]) == [12.0, 20.0, 32.0]
    rejected = pd.read_csv(written["rejected"])
    assert rejected.iloc[0]["model"] == "catboost"


def test_export_creates_missing_nested_dir(tmp_path):
    target = tmp_path / "a" / "b"
    written = export_forecast_csvs(
        {"prophet": make_forecast_result([1.0])}, make_ensemble(), output_dir=str(target)
    )
    assert (target / "forecast_final_ensemble_inverse_mape.csv").exists()
    assert written["final_ensemble_inverse_mape"].endswith(
        "forecast_final_ensemble_inverse_mape.csv"
    )


def test_export_uses_custom_prefix(tmp_path):
    export_forecast_csvs(
        {"prophet": make_forecast_result([1.0])},
        make_ensemble(),
        output_dir=str(tmp_path),
        prefix="run42",
    )
    assert (tmp_path / "run42_final_ensemble_inverse_mape.csv").exists()
    assert (tmp_path / "run42_per_model.csv").exists()
    assert (tmp_path / "run42_rejected_models.csv").exists()


def test_export_still_writes_all_files_when_empty(tmp_path):
    # No ensemble, no forecasts: all five header-only files should still exist.
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
    # CSV keys + plot keys (overlay + one dedicated figure per model + ensemble)
    assert {
        "final_ensemble_inverse_mape",
        "final_best_model",
        "per_model",
        "rejected",
        "model_summary",
    } <= set(written)
    assert {"plot_overlay", "plot_prophet", "plot_ensemble"} <= set(written)
    for name in (
        "forecast_final_ensemble_inverse_mape.csv",
        "forecast_overlay.png",
        "forecast_prophet.png",
        "forecast_ensemble.png",
    ):
        assert (tmp_path / name).exists()


def test_export_forecast_save_plots_false_writes_only_csvs(tmp_path):
    results = {"prophet": make_forecast_result([1.0, 2.0, 3.0])}
    written = export_forecast(results, make_ensemble(), output_dir=str(tmp_path), save_plots=False)
    assert set(written) == {
        "final_ensemble_inverse_mape",
        "final_best_model",
        "per_model",
        "rejected",
        "model_summary",
    }
    assert not list(tmp_path.glob("*.png"))


def test_export_forecast_empty_still_writes_csvs_no_plots(tmp_path):
    # No forecasts and no ensemble: CSVs written (header-only), plots
    # skipped rather than raising. With no ensemble there is no method to
    # name, so the combined file keeps the plain unsuffixed name.
    written = export_forecast({}, None, output_dir=str(tmp_path))
    assert set(written) == {
        "final_ensemble",
        "final_best_model",
        "per_model",
        "rejected",
        "model_summary",
    }
    assert not list(tmp_path.glob("*.png"))


# ---------- the DEFAULT (weightless) rule, and multi-method runs ----------


def make_trimmed_ensemble(*, degraded=False, n=3):
    """
    The default rule's result shape: no weights at all.

    Every other fixture in this file builds an `inverse_mape` ensemble with
    a populated `weights` dict, so without this the weightless branches --
    a blank `ensemble_weight` column, the `degraded` status, `rule_applied`
    -- were exported by code no test ever ran.
    """
    return EnsembleResult(
        combined=pd.DataFrame({"ds": _ds(n), "yhat": [12.0, 20.0, 32.0][:n]}),
        method="trimmed",
        rule_applied="mean" if degraded else "trimmed",
        members=["prophet", "catboost"] if degraded else ["prophet", "catboost", "sarimax"],
        degraded=degraded,
        weights=None,
        mape=None,
    )


def test_summary_leaves_ensemble_weight_blank_for_a_weightless_rule():
    table = build_model_summary_table(
        {"prophet": 10.0, "catboost": 12.0, "sarimax": 14.0},
        make_trimmed_ensemble(),
        ensemble_mape=9.0,
        per_model_validation_mape={"prophet": 8.0, "catboost": 11.0, "sarimax": 13.0},
    )
    members = table[table["model"] != "ensemble"]
    assert members["ensemble_weight"].isna().all()  # None, not a fabricated share
    assert (members["status"] == "in_ensemble").all()
    assert (members["rule_applied"] == "trimmed").all()
    # Both windows are reported, and they are distinct columns.
    assert members.set_index("model").loc["prophet", "validation_mape"] == 8.0
    assert members.set_index("model").loc["prophet", "mape"] == 10.0


def test_summary_marks_a_degraded_ensemble_row():
    table = build_model_summary_table(
        {"prophet": 10.0, "catboost": 12.0},
        make_trimmed_ensemble(degraded=True),
        ensemble_mape=9.0,
    )
    row = table[table["model"] == "ensemble"].iloc[0]
    assert row["status"] == "degraded"
    assert row["rule_applied"] == "mean"


def test_export_writes_one_final_file_per_ensemble_method(tmp_path):
    """
    `ensemble_method=None` builds all three rules, and each needs its own
    file -- a single `final_ensemble.csv` would silently keep only one.
    """
    results = {"prophet": make_forecast_result([1.0, 2.0, 3.0])}
    ensembles = {
        "trimmed": make_trimmed_ensemble(),
        "inverse_mape": make_ensemble(),
        "equal": make_ensemble(),
    }
    written = export_forecast_csvs(
        results,
        ensembles,
        per_model_mape={"prophet": 10.0},
        ensemble_mape=9.0,
        output_dir=str(tmp_path),
    )

    for method in ("trimmed", "inverse_mape", "equal"):
        key = f"final_ensemble_{method}"
        assert key in written
        assert (tmp_path / f"forecast_final_ensemble_{method}.csv").exists()
    # The summary repeats the per-model block once per method, so a reader
    # can compare the rules side by side in one file.
    summary = pd.read_csv(written["model_summary"])
    assert set(summary["ensemble_method"]) == {"trimmed", "inverse_mape", "equal"}
    assert (summary["model"] == "ensemble").sum() == 3


def test_rejected_table_records_which_method_dropped_the_model():
    """
    Exclusion is per method: a NaN validation MAPE is dropped by the
    weighted rules but kept by the trimmed one, so the reason must say
    which rule it applied to.
    """
    err = ThresholdRejectionError(
        "nan", model_name="sarimax", mape_value=float("nan"), allowed_max=40.0
    )
    ensembles = {
        "trimmed": make_trimmed_ensemble(),
        "inverse_mape": make_ensemble(rejected={"sarimax": err}),
    }
    table = build_rejected_models_table(ensembles)
    assert list(table["ensemble_method"]) == ["inverse_mape"]
    assert table.iloc[0]["model"] == "sarimax"


# ---------- the {method: EnsembleResult} dict a forecaster actually produces ----------
#
# `HorizonForecaster.forecast_ensemble` is a Dict[str, EnsembleResult], never a
# bare one. Before 1.0 the fixtures in this file only ever built a bare result,
# and the shape a real run produces reached several entry points untested --
# which is exactly how export_forecast(save_plots=True) shipped raising
# AttributeError on an ordinary run. These pin the dict shape at the two
# entry points that had no direct coverage.


def test_build_final_forecast_table_accepts_the_forecaster_dict():
    # Regression: this builder is deliverable #1 in the module docstring and
    # was the last consumer not normalising its input, so a caller holding
    # forecaster.forecast_ensemble and calling it directly got
    # "AttributeError: 'dict' object has no attribute 'combined'".
    table = build_final_forecast_table({"trimmed": make_trimmed_ensemble()})
    assert list(table.columns) == ["ds", "yhat"]
    assert list(table["yhat"]) == [12.0, 20.0, 32.0]


def test_build_final_forecast_table_empty_dict_is_empty_but_shaped():
    table = build_final_forecast_table({})
    assert list(table.columns) == ["ds", "yhat"]
    assert len(table) == 0


def test_export_forecast_accepts_the_forecaster_dict(tmp_path):
    # The documented one-call export, given the real dict shape AND asked for
    # plots -- the precise combination that shipped broken.
    import matplotlib

    matplotlib.use("Agg")
    results = {"prophet": make_forecast_result([1.0, 2.0, 3.0])}
    written = export_forecast(
        results,
        {"trimmed": make_trimmed_ensemble()},
        per_model_mape={"prophet": 10.0},
        output_dir=str(tmp_path),
    )
    assert "final_ensemble_trimmed" in written
    assert (tmp_path / "forecast_final_ensemble_trimmed.csv").exists()
    assert "plot_ensemble" in written
    # The combined series must actually reach the file, not just its header.
    assert list(pd.read_csv(written["final_ensemble_trimmed"])["yhat"]) == [12.0, 20.0, 32.0]


def test_scalar_ensemble_mape_is_not_stamped_on_every_method(tmp_path):
    """
    `ensemble_mape` is a single fallback score describing ONE combined
    series. In a multi-method run the rules produce materially different
    series, so repeating one number against all three would fabricate the
    side-by-side comparison this table exists to support.
    """
    ensembles = {
        "trimmed": make_trimmed_ensemble(),
        "inverse_mape": make_ensemble(),
    }
    table = build_model_summary_table(
        {"prophet": 10.0},
        ensembles,
        ensemble_mape=9.0,
    )
    ensemble_rows = table[table["model"] == "ensemble"]
    # Both rules are still reported -- only the borrowed score is withheld.
    assert set(ensemble_rows["ensemble_method"]) == {"trimmed", "inverse_mape"}
    assert ensemble_rows["mape"].isna().all()

    # With a single method the fallback is unambiguous, so it IS used.
    single = build_model_summary_table(
        {"prophet": 10.0},
        {"trimmed": make_trimmed_ensemble()},
        ensemble_mape=9.0,
    )
    assert single[single["model"] == "ensemble"].iloc[0]["mape"] == 9.0
