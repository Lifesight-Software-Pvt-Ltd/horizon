"""
Calibrated 95% bands (evaluation/bands.py) and how HorizonForecaster applies
them to every model and ensemble rule.

The unit tests need nothing fitted. The forecaster tests share one fitted
run (module fixture) of three fast models on 300 days of synthetic data, so
the suite pays for the fits once.
"""

import math
import warnings

import numpy as np
import pandas as pd
import pytest

from horizon_forecaster.evaluation.bands import (
    MIN_CALIBRATION_ERRORS,
    band,
    calibrate,
    conformal_width,
    coverage,
    widening,
)
from horizon_forecaster.exceptions import InvalidConfigError
from horizon_forecaster.exporters import build_final_forecast_table, build_model_summary_table
from horizon_forecaster.forecaster import HorizonForecaster

# --------------------------------------------------------------------------
# the method itself
# --------------------------------------------------------------------------


def test_width_is_the_conformal_rank_of_the_absolute_errors():
    errors = np.arange(1, 85, dtype=float)  # 84 errors, 1..84
    # ceil(0.95 * 85) = 81 -> the 81st smallest
    assert conformal_width(errors) == 81.0
    assert conformal_width(-errors) == 81.0  # absolute values


def test_nineteen_errors_is_the_minimum_and_uses_the_largest():
    assert conformal_width(np.arange(1, 20, dtype=float)) == 19.0
    assert conformal_width(np.arange(1, 19, dtype=float)) is None
    assert MIN_CALIBRATION_ERRORS == 19


def test_calibrate_leaves_out_unexpected_zero_days():
    y = np.full(30, 100.0)
    yhat = y + np.arange(30)  # errors 0..29
    zero_days = np.zeros(30, dtype=bool)
    zero_days[-2:] = True  # the two largest errors are outage days
    c = calibrate(y, yhat, zero_days)
    assert (c.n_errors, c.n_zero_days_left_out, c.validation_length) == (28, 2, 30)
    assert c.width == conformal_width(np.arange(28.0))


def test_calibrate_gives_no_band_below_nineteen_errors():
    c = calibrate(np.ones(18), np.ones(18))
    assert c.width is None
    assert "18 validation errors" in c.reason


def test_width_is_flat_to_the_validation_length_then_grows_with_the_square_root():
    m = widening(np.array([1, 84, 85, 168, 336]), 84)
    assert np.allclose(m, [1, 1, math.sqrt(85 / 84), math.sqrt(2), 2])


def test_band_floors_at_zero_only_when_asked():
    c = calibrate(np.zeros(20), np.full(20, 10.0))  # width 10
    lower, upper = band(np.array([5.0, 50.0]), c, floor_at_zero=True)
    assert lower.tolist() == [0.0, 40.0] and upper.tolist() == [15.0, 60.0]
    lower, _ = band(np.array([5.0]), c, floor_at_zero=False)
    assert lower.tolist() == [-5.0]


def test_no_band_without_a_width():
    assert band(np.array([1.0]), calibrate(np.ones(5), np.ones(5)), True) is None


def test_coverage_counts_points_inside():
    assert coverage([1, 5, 9], [0, 6, 0], [2, 8, 10]) == (pytest.approx(200 / 3), 3)


# --------------------------------------------------------------------------
# through the forecaster
# --------------------------------------------------------------------------

_MODELS = ["prophet", "sarimax", "catboost"]
_PARAMS = {
    "prophet": {"seasonality_mode": "additive"},
    "sarimax": {"p": 1, "q": 1, "P": 0, "Q": 0},
    "catboost": {"iterations": 100, "depth": 4},
}


def _series(n=300, zeros=(), seed=3):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    y = 1000 + 0.5 * t + 60 * np.sin(2 * np.pi * t / 7) + rng.normal(0, 25, n)
    df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=n), "sales": y})
    df.loc[list(zeros), "sales"] = 0.0
    return df


def _run(df, horizon=30, **kw):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        f = HorizonForecaster(
            df, "date", "sales", "daily", models=_MODELS, model_params=_PARAMS, **kw
        )
        results = f.evaluate()
        forecasts = f.forecast(horizon)
    return f, results, forecasts, [str(w.message) for w in caught]


@pytest.fixture(scope="module")
def run():
    # 300 days: 188 train, 84 validation, 28 test; 120-day forecast.
    return _run(_series(), horizon=120, mape_threshold=None)


def test_every_model_and_ensemble_rule_has_a_band_in_evaluate_and_forecast(run):
    f, results, forecasts, _ = run
    for name in _MODELS:
        assert {"yhat_lower", "yhat_upper"} <= set(results[name].test_predictions.columns)
        assert {"yhat_lower", "yhat_upper"} <= set(forecasts[name].forecast.columns)
        assert results[name].band.width > 0
        assert forecasts[name].band is results[name].band  # one calibration per model
    for ensembles in (f.evaluate_ensemble, f.forecast_ensemble):
        for e in ensembles.values():
            assert {"yhat_lower", "yhat_upper"} <= set(e.combined.columns)
            assert e.band.width > 0


def test_a_models_band_is_its_own_validation_errors_at_the_conformal_rank(run):
    f, results, _, _ = run
    sel = f._selection_cache["catboost"]
    rows = f.raw_df.iloc[f._n_train : f._n_train + f._n_validation]
    expected = conformal_width(rows["sales"].to_numpy() - sel.validation_predictions)
    assert results["catboost"].band.width == pytest.approx(expected)
    tp = results["catboost"].test_predictions
    assert np.allclose(tp["yhat_upper"] - tp["yhat"], expected)  # flat inside 28 days


def test_the_ensemble_band_comes_from_the_ensembles_own_validation_forecast(run):
    f, _, _, _ = run
    e = f.evaluate_ensemble["trimmed"]
    rows = f.raw_df.iloc[f._n_train : f._n_train + f._n_validation]
    members = np.vstack([f._selection_cache[m].validation_predictions for m in e.members])
    middle = np.sort(members, axis=0)[1]  # trimmed of three members = the middle value
    expected = conformal_width(rows["sales"].to_numpy() - middle)
    assert e.band.width == pytest.approx(expected)


def test_the_forecast_band_widens_past_the_validation_length(run):
    _, _, forecasts, _ = run
    fc = forecasts["prophet"].forecast
    half = (fc["yhat_upper"] - fc["yhat"]).to_numpy()
    q = forecasts["prophet"].band.width
    assert np.allclose(half[:84], q)
    assert half[84] == pytest.approx(q * math.sqrt(85 / 84))
    assert half[119] == pytest.approx(q * math.sqrt(120 / 84))


def test_a_horizon_past_the_validation_length_warns_once(run):
    *_, messages = run
    hits = [m for m in messages if "past the 84 days of validation errors" in m]
    assert len(hits) == 1
    assert "82-88% of actuals from 3 to 16 months ahead" in hits[0]


def test_model_summary_reports_each_bands_test_coverage(run):
    f, results, forecasts, _ = run
    from horizon_forecaster.exporters import _scores_from_forecaster

    scores = _scores_from_forecaster(f, "test")
    table = build_model_summary_table(
        scores["per_model_mape"],
        f.forecast_ensemble,
        band_coverage=scores["band_coverage"],
        ensemble_band_coverage=scores["ensemble_band_coverage"],
    )
    row = table[table.model == "sarimax"].iloc[0]
    tp = results["sarimax"].test_predictions
    inside = ((tp.y >= tp.yhat_lower) & (tp.y <= tp.yhat_upper)).mean() * 100
    assert row.band_coverage == pytest.approx(inside)
    assert row.band_test_points == 28
    assert table[table.model == "ensemble"].band_test_points.iloc[0] == 28


def test_the_final_ensemble_table_carries_the_band(run):
    f, *_ = run
    table = build_final_forecast_table(f.forecast_ensemble)
    assert list(table.columns) == ["ds", "yhat", "yhat_lower", "yhat_upper"]


def test_unexpected_zero_days_are_left_out_and_named_once_per_call():
    # Days 200-271 are validation (188 train): two outage days in it.
    f, results, _, messages = _run(_series(zeros=(200, 230)), mape_threshold=None)
    assert results["sarimax"].band.n_zero_days_left_out == 2
    assert results["sarimax"].band.n_errors == 82
    assert sum("2 day(s) in the validation window are 0" in m for m in messages) == 2  # eval + fc


def test_unexpected_zero_weeks_are_named_as_weeks():
    # 160 weeks: 108 train, 39 validation (rows 108-146), 13 test.
    rng = np.random.default_rng(3)
    df = pd.DataFrame(
        {
            "date": pd.date_range("2022-01-03", periods=160, freq="7D"),
            "sales": 1000 + rng.normal(0, 25, 160),
        }
    )
    df.loc[120, "sales"] = 0.0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        f = HorizonForecaster(
            df,
            "date",
            "sales",
            "weekly",
            models=["catboost"],
            model_params={"catboost": _PARAMS["catboost"]},
            mape_threshold=None,
        )
        f.evaluate()
    messages = [str(w.message) for w in caught]
    assert any("1 week(s) in the validation window are 0" in m for m in messages)
    assert not any("day(s) in the validation window" in m for m in messages)


def test_too_few_validation_errors_means_no_band_and_a_warning():
    f, results, forecasts, messages = _run(
        _series(), validation_size=14, test_size=14, mape_threshold=None
    )
    assert "yhat_lower" not in results["catboost"].test_predictions.columns
    assert "yhat_lower" not in forecasts["catboost"].forecast.columns
    assert "yhat_lower" not in f.forecast_ensemble["trimmed"].combined.columns
    hit = [m for m in messages if m.startswith("No 95% band for")]
    assert hit and "14 validation errors, fewer than the 19 a 95% band needs" in hit[0]


def test_an_ensemble_member_without_a_validation_forecast_leaves_the_ensemble_without_a_band(run):
    f, *_ = run
    ens = {k: v for k, v in f.evaluate_ensemble.items()}
    kept = f._selection_cache["catboost"].validation_predictions
    try:
        f._selection_cache["catboost"].validation_predictions = None
        f._add_ensemble_bands(ens, list(_MODELS))
        assert ens["trimmed"].band.width is None
        assert "yhat_lower" not in ens["trimmed"].combined.columns
    finally:
        f._selection_cache["catboost"].validation_predictions = kept


def test_generated_drivers_warning_says_the_bands_assume_them():
    df = _series()
    df["spend"] = np.random.default_rng(1).uniform(50, 150, len(df))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        f = HorizonForecaster(
            df,
            "date",
            "sales",
            "daily",
            independent_columns=["spend"],
            models=["catboost"],
            model_params={"catboost": {"iterations": 50}},
        )
        f.forecast(10)
    text = " ".join(str(w.message) for w in caught)
    assert "The 95% bands assume these values are right" in text


@pytest.mark.parametrize(
    "model, setting",
    [
        ("prophet", {"interval_width": 0.95}),
        ("prophet", {"uncertainty_samples": 0}),
        ("sarimax", {"alpha": 0.1}),
        ("bayesianpymc", {"ci_lower_pct": 2.5}),
    ],
)
def test_model_native_interval_settings_are_refused(model, setting):
    with pytest.raises(InvalidConfigError, match="calibrated on its validation errors"):
        HorizonForecaster(_series(), "date", "sales", "daily", model_params={model: setting})
