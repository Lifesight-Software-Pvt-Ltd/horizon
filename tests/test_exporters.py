import math

import pandas as pd
import pytest
from types import SimpleNamespace

from horizon_forecaster.exporters import (
    choose_best_model,
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
from horizon_forecaster.evaluation.metrics import MapeResult
from horizon_forecaster.exceptions import InvalidConfigError, ThresholdRejectionError


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


def _two_models():
    return {
        "prophet": make_forecast_result([1.0, 2.0, 3.0]),
        "catboost": make_forecast_result([4.0, 5.0, 6.0]),
    }


def test_best_model_is_the_lowest_validation_mape_not_the_lowest_test_mape():
    table = build_best_model_forecast_table(
        _two_models(),
        {"prophet": 12.0, "catboost": 20.0},  # test: prophet better
        per_model_validation_mape={"prophet": 9.0, "catboost": 6.0},  # validation: catboost
    )
    assert set(table["model"]) == {"catboost"}
    assert list(table["yhat"]) == [4.0, 5.0, 6.0]


def test_best_model_must_pass_the_threshold_on_validation():
    # The real-run failure: a model the ensemble rejected was exported at 103%.
    best, checks = choose_best_model(
        _two_models(), {"prophet": 45.0, "catboost": 30.0}, {"prophet": 5.0, "catboost": 103.0}
    )
    assert best is None
    assert checks["prophet"] == "not chosen: validation MAPE 45.00 > 40"
    assert checks["catboost"] == "not chosen: test MAPE 103.00 > 40"


def test_best_model_must_pass_the_threshold_on_test_too():
    best, checks = choose_best_model(
        _two_models(), {"prophet": 5.0, "catboost": 9.0}, {"prophet": 55.0, "catboost": 12.0}
    )
    assert best == "catboost"
    assert checks["prophet"] == "not chosen: test MAPE 55.00 > 40"
    assert "test MAPE was used as a check" in checks["catboost"]


def test_best_model_uses_the_given_threshold_and_none_skips_the_checks():
    results, val, test = _two_models(), {"prophet": 5.0, "catboost": 9.0}, {"prophet": 55.0}
    assert choose_best_model(results, val, test, mape_threshold=60)[0] == "prophet"
    assert choose_best_model(results, val, test, mape_threshold=None)[0] == "prophet"


def test_best_model_without_test_scores_applies_only_the_validation_check():
    best, checks = choose_best_model(_two_models(), {"prophet": 5.0, "catboost": 9.0}, None)
    assert best == "prophet"
    assert "no test MAPEs were given" in checks["prophet"]


def test_best_model_never_picks_a_model_without_a_validation_mape():
    best, checks = choose_best_model(
        _two_models(),
        {"prophet": float("nan"), "catboost": 30.0},
        {"prophet": 1.0, "catboost": 2.0},
    )
    assert best == "catboost"
    assert checks["prophet"] == "not chosen: no validation MAPE"


def test_best_model_table_is_empty_with_a_warning_when_no_model_qualifies():
    with pytest.warns(UserWarning, match="No model qualifies .* catboost: not chosen"):
        table = build_best_model_forecast_table(
            _two_models(),
            {"prophet": 50.0, "catboost": 60.0},
            per_model_validation_mape={"prophet": 5.0, "catboost": 6.0},
        )
    assert len(table) == 0 and list(table.columns) == ["model", "ds", "yhat"]


def test_best_model_table_is_empty_with_a_warning_without_validation_scores():
    with pytest.warns(UserWarning, match="per_model_validation_mape was not given"):
        table = build_best_model_forecast_table(_two_models(), {"prophet": 5.0})
    assert len(table) == 0
    assert len(build_best_model_forecast_table({}, {"a": 10.0})) == 0


def test_summary_reports_each_models_best_model_check():
    _, checks = choose_best_model(
        _two_models(), {"prophet": 5.0, "catboost": 9.0}, {"prophet": 55.0, "catboost": 12.0}
    )
    table = build_model_summary_table(
        {"prophet": 55.0, "catboost": 12.0},
        per_model_validation_mape={"prophet": 5.0, "catboost": 9.0},
        best_model_checks=checks,
    )
    by_model = table.set_index("model")["best_model_check"]
    assert by_model["prophet"] == "not chosen: test MAPE 55.00 > 40"
    assert by_model["catboost"].startswith("exported as best")


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
    # CSV keys + plot keys (one overlay and one ensemble figure per
    # combination rule, plus one dedicated figure per model)
    assert {
        "final_ensemble_inverse_mape",
        "final_best_model",
        "per_model",
        "rejected",
        "model_summary",
    } <= set(written)
    assert {
        "plot_overlay_inverse_mape",
        "plot_prophet",
        "plot_ensemble_inverse_mape",
    } <= set(written)
    for name in (
        "forecast_final_ensemble_inverse_mape.csv",
        "forecast_overlay_inverse_mape.png",
        "forecast_prophet.png",
        "forecast_ensemble_inverse_mape.png",
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
# bare one. Fixtures that only ever build a bare result would leave the shape a
# real run produces untested at several entry points -- and
# export_forecast(save_plots=True) raising AttributeError on an ordinary run
# would go unnoticed. These pin the dict shape at the two
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
    assert "plot_ensemble_trimmed" in written
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


def test_export_applies_both_best_model_checks_with_the_given_threshold(tmp_path):
    results = _two_models()
    written = export_forecast_csvs(
        results,
        make_ensemble(),
        per_model_mape={"prophet": 35.0, "catboost": 12.0},
        per_model_validation_mape={"prophet": 5.0, "catboost": 9.0},
        mape_threshold=30.0,
        output_dir=str(tmp_path),
    )
    best = pd.read_csv(written["final_best_model"])
    assert set(best["model"]) == {"catboost"}  # prophet fails the test check at 30
    summary = pd.read_csv(written["model_summary"]).drop_duplicates("model").set_index("model")
    assert summary.loc["prophet", "best_model_check"] == "not chosen: test MAPE 35.00 > 30"
    assert summary.loc["catboost", "best_model_check"].startswith("exported as best")


# ---------- argument checks ----------
#
# Unchecked, a wrong argument surfaces as a raw AttributeError/KeyError deep
# inside, or -- for the scores -- as Python reprs written into the CSV.


def _evaluation_like():
    return {"prophet": SimpleNamespace(test_predictions=pd.DataFrame(), mape=None)}


def test_export_names_the_expected_results_when_given_evaluate_output(tmp_path):
    with pytest.raises(
        InvalidConfigError,
        match=r"forecaster\.forecast.*look like the results of forecaster\.evaluate",
    ):
        export_forecast(_evaluation_like(), output_dir=str(tmp_path), save_plots=False)


def test_export_rejects_a_list_of_results(tmp_path):
    with pytest.raises(InvalidConfigError, match="got list"):
        export_forecast_csvs(list(_two_models().values()), output_dir=str(tmp_path))


def test_export_rejects_an_ensemble_that_is_not_an_ensemble_result(tmp_path):
    with pytest.raises(InvalidConfigError, match="ensemble must be"):
        export_forecast_csvs(_two_models(), "trimmed", output_dir=str(tmp_path))
    with pytest.raises(InvalidConfigError, match="are not EnsembleResults"):
        export_forecast_csvs(_two_models(), {"trimmed": "x"}, output_dir=str(tmp_path))


def test_export_rejects_scores_that_are_not_numbers(tmp_path):
    with pytest.raises(InvalidConfigError, match="per_model_mape must map each model to a number"):
        export_forecast_csvs(
            _two_models(),
            make_ensemble(),
            per_model_mape={"prophet": SimpleNamespace(value=1.0), "catboost": 2.0},
            output_dir=str(tmp_path),
        )
    with pytest.raises(InvalidConfigError, match="ensemble_mape must be a number"):
        export_forecast_csvs(
            _two_models(), make_ensemble(), ensemble_mape="5", output_dir=str(tmp_path)
        )


def test_export_rejects_a_text_mape_threshold(tmp_path):
    # Unchecked, it would write a model_summary.csv with no model rows.
    with pytest.raises(InvalidConfigError, match="mape_threshold must be a number"):
        export_forecast_csvs(
            _two_models(), make_ensemble(), mape_threshold="40", output_dir=str(tmp_path)
        )


def test_export_checks_history_columns_before_writing_anything(tmp_path):
    history = pd.DataFrame({"date": _ds(3), "orders": [1.0, 2.0, 3.0]})
    with pytest.raises(InvalidConfigError, match=r"history has no column\(s\) \['ds', 'y'\]"):
        export_forecast(_two_models(), make_ensemble(), history=history, output_dir=str(tmp_path))
    assert list(tmp_path.iterdir()) == []


def test_export_forecast_plots_a_history_with_text_dates(tmp_path):
    history = pd.DataFrame(
        {"date": ["2022-12-29", "2022-12-30", "2022-12-31"], "orders": [1.0, 2.0, 3.0]}
    )
    written = export_forecast(
        _two_models(),
        make_ensemble(),
        history=history,
        date_column="date",
        target_column="orders",
        output_dir=str(tmp_path),
    )
    assert any(k.startswith("plot_") for k in written)


# ---------- WAPE, the seasonal-naive baseline and forecaster= ----------


def _score(value, excluded=0, total=3):
    return MapeResult(value=value, excluded_zero_actual_count=excluded, total_count=total)


def _eval(mape, val_mape, wape, val_wape, excluded=0):
    return SimpleNamespace(
        mape=_score(mape, excluded),
        validation_mape=_score(val_mape),
        wape=wape,
        validation_wape=val_wape,
    )


def _fake_forecaster(selection_metric="mape", evaluated=True, mape_threshold=40.0):
    """The public attributes the exports read, as evaluate() + forecast() leave them."""
    ensemble = make_ensemble()
    ensemble.method = "trimmed"
    scored = EnsembleResult(
        combined=ensemble.combined,
        method="trimmed",
        rule_applied="inverse_mape",
        members=["prophet"],
        weights={"prophet": 1.0},
        rejected_models={},
        mape=_score(12.0, excluded=1),
        wape=15.0,
    )
    results = {
        # catboost wins on MAPE, prophet on WAPE: the choice follows the metric.
        "prophet": _eval(20.0, 12.0, 18.0, 10.0, excluded=1),
        "catboost": _eval(10.0, 5.0, 30.0, 25.0, excluded=1),
    }
    baseline = SimpleNamespace(
        model_name="seasonal_naive",
        mape=_score(22.0, excluded=1),
        validation_mape=_score(21.0),
        wape=25.0,
        validation_wape=24.0,
    )
    corr = pd.DataFrame(
        [[1.0, 0.4], [0.4, 1.0]], index=["prophet", "catboost"], columns=["prophet", "catboost"]
    )
    return SimpleNamespace(
        forecast_ensemble={"trimmed": ensemble},
        evaluate_ensemble={"trimmed": scored} if evaluated else None,
        evaluation_results=results if evaluated else None,
        naive_baseline=baseline if evaluated else None,
        test_error_correlation=corr if evaluated else None,
        mape_threshold=mape_threshold,
        selection_metric=selection_metric,
    )


def _two_forecasts():
    return {
        "prophet": make_forecast_result([1.0, 2.0, 3.0]),
        "catboost": make_forecast_result([4.0, 5.0, 6.0]),
    }


def test_summary_keeps_the_original_columns_first():
    assert _SUMMARY_COLUMNS[:8] == [
        "model",
        "validation_mape",
        "mape",
        "ensemble_method",
        "rule_applied",
        "ensemble_weight",
        "status",
        "best_model_check",
    ]
    assert _SUMMARY_COLUMNS[8:] == [
        "validation_wape",
        "wape",
        "zero_actuals_excluded",
        "beats_naive",
        "selection_metric",
        "band_coverage",
        "band_test_points",
    ]


def test_export_with_forecaster_fills_every_score_and_the_baseline(tmp_path):
    forecaster = _fake_forecaster()
    written = export_forecast_csvs(_two_forecasts(), forecaster=forecaster, output_dir=tmp_path)
    summary = pd.read_csv(written["model_summary"]).set_index("model")
    assert summary.loc["prophet", "wape"] == 18.0
    assert summary.loc["prophet", "validation_wape"] == 10.0
    assert summary.loc["catboost", "mape"] == 10.0
    assert summary.loc["catboost", "zero_actuals_excluded"] == 1
    # The ensemble row takes evaluate()'s test scores for the same rule.
    assert summary.loc["ensemble", "mape"] == 12.0
    assert summary.loc["ensemble", "wape"] == 15.0
    # Baseline row, and beats_naive against its test WAPE of 25.
    assert summary.loc["seasonal_naive", "status"] == "baseline"
    assert summary.loc["seasonal_naive", "wape"] == 25.0
    assert bool(summary.loc["prophet", "beats_naive"]) is True
    assert bool(summary.loc["catboost", "beats_naive"]) is False
    assert bool(summary.loc["ensemble", "beats_naive"]) is True
    assert set(summary["selection_metric"]) == {"mape"}


def test_export_with_forecaster_chooses_the_best_model_on_mape_by_default(tmp_path):
    written = export_forecast_csvs(
        _two_forecasts(), forecaster=_fake_forecaster(), output_dir=tmp_path
    )
    assert set(pd.read_csv(written["final_best_model"])["model"]) == {"catboost"}


def test_export_with_a_wape_forecaster_chooses_the_best_model_on_wape(tmp_path):
    written = export_forecast_csvs(
        _two_forecasts(), forecaster=_fake_forecaster("wape"), output_dir=tmp_path
    )
    assert set(pd.read_csv(written["final_best_model"])["model"]) == {"prophet"}
    summary = pd.read_csv(written["model_summary"]).set_index("model")
    assert "lowest validation WAPE" in summary.loc["prophet", "best_model_check"]
    assert set(summary["selection_metric"]) == {"wape"}


def test_export_uses_the_forecasters_threshold_unless_one_is_given(tmp_path):
    # Both models' validation MAPE (12, 5) is above 4: nobody qualifies.
    with pytest.warns(UserWarning, match="No model qualifies"):
        written = export_forecast_csvs(
            _two_forecasts(),
            forecaster=_fake_forecaster(mape_threshold=4.0),
            output_dir=tmp_path,
        )
    assert pd.read_csv(written["final_best_model"]).empty
    written = export_forecast_csvs(
        _two_forecasts(),
        forecaster=_fake_forecaster(mape_threshold=4.0),
        mape_threshold=40.0,
        output_dir=tmp_path,
    )
    assert not pd.read_csv(written["final_best_model"]).empty


def test_explicit_scores_win_over_the_forecasters(tmp_path):
    written = export_forecast_csvs(
        _two_forecasts(),
        forecaster=_fake_forecaster(),
        per_model_mape={"prophet": 1.0, "catboost": 2.0},
        per_model_validation_mape={"prophet": 1.0, "catboost": 2.0},
        output_dir=tmp_path,
    )
    summary = pd.read_csv(written["model_summary"]).set_index("model")
    assert summary.loc["prophet", "mape"] == 1.0
    assert set(pd.read_csv(written["final_best_model"])["model"]) == {"prophet"}


def test_export_writes_the_error_correlation(tmp_path):
    written = export_forecast_csvs(
        _two_forecasts(), forecaster=_fake_forecaster(), output_dir=tmp_path
    )
    corr = pd.read_csv(written["error_correlation"], index_col="model")
    assert corr.loc["prophet", "catboost"] == 0.4


def test_export_without_forecaster_writes_no_error_correlation(tmp_path):
    with pytest.warns(UserWarning, match="per_model_validation_mape was not given"):
        written = export_forecast_csvs(_two_forecasts(), make_ensemble(), output_dir=tmp_path)
    assert "error_correlation" not in written
    summary = pd.read_csv(written["model_summary"])
    assert "seasonal_naive" not in set(summary["model"])
    assert summary["beats_naive"].isna().all()


def test_export_before_evaluate_warns_and_still_writes(tmp_path):
    with pytest.warns(UserWarning, match=r"forecaster.evaluate\(\) has not run"):
        written = export_forecast_csvs(
            _two_forecasts(), forecaster=_fake_forecaster(evaluated=False), output_dir=tmp_path
        )
    assert "error_correlation" not in written
    assert (tmp_path / "forecast_final_ensemble_trimmed.csv").exists()


def test_export_rejects_something_that_is_not_a_forecaster(tmp_path):
    with pytest.raises(InvalidConfigError, match="forecaster= must be the HorizonForecaster"):
        export_forecast_csvs(_two_forecasts(), forecaster={"a": 1}, output_dir=tmp_path)


def test_rejected_table_names_the_selection_metric(tmp_path):
    forecaster = _fake_forecaster("wape")
    forecaster.forecast_ensemble["trimmed"].rejected_models = {
        "catboost": ThresholdRejectionError(
            "catboost's validation WAPE 45.00 is above 40",
            model_name="catboost",
            mape_value=45.0,
            allowed_max=40.0,
        )
    }
    written = export_forecast_csvs(_two_forecasts(), forecaster=forecaster, output_dir=tmp_path)
    rejected = pd.read_csv(written["rejected"])
    assert list(rejected.columns) == _REJECTED_COLUMNS
    assert rejected.loc[0, "selection_metric"] == "wape"


def test_export_forecast_passes_the_forecaster_through(tmp_path):
    written = export_forecast(
        _two_forecasts(), forecaster=_fake_forecaster(), output_dir=tmp_path, save_plots=False
    )
    assert "error_correlation" in written
    assert "final_ensemble_trimmed" in written


def test_explicit_validation_mapes_choose_the_best_model_even_under_wape(tmp_path):
    written = export_forecast_csvs(
        _two_forecasts(),
        forecaster=_fake_forecaster("wape"),  # on WAPE alone prophet would be chosen
        per_model_validation_mape={"prophet": 9.0, "catboost": 1.0},
        output_dir=tmp_path,
    )
    assert set(pd.read_csv(written["final_best_model"])["model"]) == {"catboost"}
