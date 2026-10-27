"""
Tests for the Bayesian PyMC model.

Split into three tiers so the fast, deterministic logic is always covered
even where the heavy dependency is absent:

1. Adapter tests -- pure pandas/numpy, no pymc needed.
2. Wrapper geometry/guard tests -- Fourier basis, time index, and the
   contract's error paths (missing columns, predict-before-fit, too few
   rows). These exercise everything up to, but not including, sampling, so
   they need no pymc either.
3. Real-sampling integration tests -- a genuine (tiny) NUTS run, guarded by
   importorskip("pymc"). They confirm the
   fit -> predict -> back-transform path, the credible band, save/load, and
   end-to-end wiring through HorizonForecaster.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from horizon_forecaster.data_adapter.bayesianpymc_adapter import BayesianPyMCAdapter
from horizon_forecaster.exceptions import (
    InsufficientDataError,
    InvalidConfigError,
    ModelPredictionError,
)
from horizon_forecaster.forecaster import MODEL_REGISTRY
from horizon_forecaster.models.bayesianpymc_wrapper import (
    ANNUAL_MIN_SPAN_DAYS,
    BayesianPyMCWrapper,
    span_days,
)

# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


def make_raw(n=60, freq="D"):
    rng = np.random.default_rng(0)
    dates = pd.date_range("2022-01-01", periods=n, freq=freq)
    trend = np.arange(n) * 1.5 + 100.0
    season = 10.0 * np.sin(2 * np.pi * np.arange(n) / 7.0)
    return pd.DataFrame(
        {
            "date": dates,
            "orders": trend + season + rng.normal(0, 3, n),
            "spend": np.abs(rng.normal(100, 20, n)) + 1.0,  # continuous driver
            "promo": rng.integers(0, 2, n),  # binary flag
        }
    )


# --------------------------------------------------------------------------
# 1. adapter
# --------------------------------------------------------------------------


def test_adapter_requires_valid_frequency():
    with pytest.raises(InvalidConfigError):
        BayesianPyMCAdapter("date", "orders", frequency="monthly")


def test_adapter_emits_ds_y_and_independents_only():
    adapter = BayesianPyMCAdapter("date", "orders", ["spend", "promo"], "daily")
    out = adapter.transform_train(make_raw())
    assert list(out.columns) == ["ds", "y", "spend", "promo"]


def test_adapter_keeps_y_raw():
    """y must stay in raw units (standardisation is the wrapper's job)."""
    raw = make_raw()
    adapter = BayesianPyMCAdapter("date", "orders", ["spend"], "daily")
    out = adapter.transform_train(raw)
    assert np.allclose(np.sort(out["y"].to_numpy()), np.sort(raw["orders"].to_numpy()))


def test_adapter_standardizes_independents_on_train():
    adapter = BayesianPyMCAdapter("date", "orders", ["spend"], "daily")
    out = adapter.transform_train(make_raw())
    assert abs(out["spend"].mean()) < 1e-9
    assert abs(out["spend"].std(ddof=0) - 1.0) < 1e-6


def test_adapter_future_reuses_train_stats_and_clamps_range():
    adapter = BayesianPyMCAdapter("date", "orders", ["spend"], "daily")
    adapter.transform_train(make_raw())
    fut_dates = pd.Series(pd.date_range("2022-04-01", periods=4, freq="D"))
    # A spend value far above anything in training must be clamped, not
    # extrapolated, so it cannot exceed the standardised training max.
    fut_ind = pd.DataFrame({"spend": [1e9, 1e9, 1e9, 1e9]})
    out = adapter.transform_future(fut_dates, fut_ind)
    assert list(out.columns) == ["ds", "spend"]
    assert out["spend"].max() <= float(adapter._std_max["spend"]) + 1e-9


def test_adapter_future_requires_independent_df_when_declared():
    adapter = BayesianPyMCAdapter("date", "orders", ["spend"], "daily")
    adapter.transform_train(make_raw())
    with pytest.raises(InvalidConfigError):
        adapter.transform_future(pd.Series(pd.date_range("2022-04-01", periods=3)), None)


def test_adapter_future_before_train_errors():
    adapter = BayesianPyMCAdapter("date", "orders", ["spend"], "daily")
    with pytest.raises(InvalidConfigError):
        adapter.transform_future(
            pd.Series(pd.date_range("2022-04-01", periods=3)),
            pd.DataFrame({"spend": [1.0, 2.0, 3.0]}),
        )


def test_adapter_missing_column_errors():
    adapter = BayesianPyMCAdapter("date", "orders", ["spend"], "daily")
    with pytest.raises(InvalidConfigError):
        adapter.transform_train(make_raw().drop(columns=["spend"]))


def test_adapter_no_independents_ok():
    adapter = BayesianPyMCAdapter("date", "orders", [], "weekly")
    out = adapter.transform_train(make_raw(freq="7D"))
    assert list(out.columns) == ["ds", "y"]


# --------------------------------------------------------------------------
# 2. wrapper geometry & guards (no sampling)
# --------------------------------------------------------------------------


def _geom_wrapper(frequency="daily", n_fourier=4, use_short=None):
    w = BayesianPyMCWrapper(frequency=frequency, params={"n_fourier": n_fourier})
    if use_short is not None:
        w.use_short_seasonal = use_short
    w._reference_date = pd.Timestamp("2022-01-01")
    return w


def test_fourier_basis_shape_and_layout():
    w = _geom_wrapper(n_fourier=3)
    ds = pd.Series(pd.date_range("2022-01-01", periods=10, freq="D"))
    ff_long, ff_short = w._fourier_terms(ds)
    assert ff_long.shape == (10, 6)  # 2 * n_fourier
    assert ff_short is not None and ff_short.shape == (10, 6)
    # first columns are the cosines; at day 0 all cosines == 1, all sines == 0
    assert np.allclose(ff_long[0, :3], 1.0)
    assert np.allclose(ff_long[0, 3:], 0.0)


def test_short_seasonal_absent_for_weekly():
    w = _geom_wrapper(frequency="weekly")
    ds = pd.Series(pd.date_range("2022-01-01", periods=10, freq="7D"))
    _, ff_short = w._fourier_terms(ds)
    assert ff_short is None


def test_weekly_order_is_capped_at_three_while_the_annual_order_follows_n_fourier():
    w = _geom_wrapper(n_fourier=4)
    ds = pd.Series(pd.date_range("2022-01-01", periods=10, freq="D"))
    ff_long, ff_short = w._fourier_terms(ds)
    assert ff_long.shape == (10, 8)
    assert ff_short.shape == (10, 6)


def test_weekly_order_four_only_mirrors_order_three_on_whole_days():
    """Why the cap loses nothing: the order-4 columns are the order-3 ones, sine negated."""
    day = np.arange(30, dtype=float)
    basis = BayesianPyMCWrapper._fourier_basis(day, 7.0, 4)  # cos1..cos4, sin1..sin4
    assert np.allclose(basis[:, 3], basis[:, 2])
    assert np.allclose(basis[:, 7], -basis[:, 6])


def test_no_annual_columns_when_the_fit_turned_annual_terms_off():
    w = _geom_wrapper(n_fourier=4)
    w.annual_terms = False
    ds = pd.Series(pd.date_range("2022-01-01", periods=10, freq="D"))
    ff_long, ff_short = w._fourier_terms(ds)
    assert ff_long.shape == (10, 0)
    assert ff_short.shape == (10, 6)


def test_span_days_counts_calendar_days_inclusive():
    assert span_days(pd.Series(pd.date_range("2022-01-01", periods=183, freq="D"))) == 183
    unordered = pd.Series(pd.to_datetime(["2022-07-01", "2022-01-03", "2022-03-15"]))
    assert span_days(unordered) == 180  # first to last date, not the row count
    assert span_days(pd.Series([], dtype="datetime64[ns]")) == 0


def test_time_steps_units_by_frequency():
    ds = pd.Series(pd.date_range("2022-01-01", periods=3, freq="7D"))
    w_daily = _geom_wrapper(frequency="daily")
    w_weekly = _geom_wrapper(frequency="weekly")
    assert np.allclose(w_daily._time_steps(ds), [0, 7, 14])  # days
    assert np.allclose(w_weekly._time_steps(ds), [0, 1, 2])  # weeks


def test_fit_requires_ds_and_y():
    w = BayesianPyMCWrapper(frequency="daily")
    with pytest.raises(ModelPredictionError):
        w.fit(pd.DataFrame({"foo": [1, 2, 3]}))


def test_fit_too_few_rows_raises_insufficient():
    w = BayesianPyMCWrapper(frequency="daily", min_rows_required=35)
    tiny = pd.DataFrame(
        {
            "ds": pd.date_range("2022-01-01", periods=5, freq="D"),
            "y": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    with pytest.raises(InsufficientDataError):
        w.fit(tiny)


def test_predict_before_fit_raises():
    w = BayesianPyMCWrapper(frequency="daily")
    with pytest.raises(ModelPredictionError):
        w.predict(
            horizon=3, future_exog=pd.DataFrame({"ds": pd.date_range("2022-01-01", periods=3)})
        )


# --------------------------------------------------------------------------
# registry / config wiring (no sampling)
# --------------------------------------------------------------------------


def test_registered_in_model_registry():
    assert "bayesianpymc" in MODEL_REGISTRY
    assert MODEL_REGISTRY["bayesianpymc"]["adapter"] is BayesianPyMCAdapter
    assert MODEL_REGISTRY["bayesianpymc"]["wrapper"] is BayesianPyMCWrapper


def test_config_has_both_frequencies():
    import yaml
    from horizon_forecaster.forecaster import _CONFIG_PATH

    with open(_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    for freq in ("daily", "weekly"):
        block = cfg["bayesianpymc"][freq]
        assert block["hyperparameter_grid"]
        assert "min_rows_required" in block["wrapper_config"]
        assert "random_seed" in block["wrapper_config"]


# --------------------------------------------------------------------------
# forecaster warnings about annual terms and convergence (no sampling: the
# fitted model is a stand-in carrying only sample_stats)
# --------------------------------------------------------------------------


class _FittedStub:
    def __init__(self, max_rhat=float("nan"), min_ess_bulk=float("nan")):
        self.sample_stats = {"max_rhat": max_rhat, "min_ess_bulk": min_ess_bulk}


def _pymc_forecaster(n):
    from horizon_forecaster.forecaster import HorizonForecaster

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return HorizonForecaster(
            raw_df=make_raw(n),
            date_column="date",
            target_column="orders",
            frequency="daily",
            models=["bayesianpymc"],
            model_params={"bayesianpymc": {"n_fourier": 2}},
        )


def _pymc_warnings(hf, stage, model=None):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        hf._warn_about_bayesianpymc(stage, model or _FittedStub())
    return [str(w.message) for w in caught]


def test_evaluate_on_a_short_series_names_both_fits_without_annual_terms():
    (message,) = _pymc_warnings(_pymc_forecaster(150), "evaluate")  # 83 / 133 days
    assert message.startswith("bayesianpymc: fitted without annual (yearly) seasonality")
    assert "(validation fit: 83 days, test fit: 133 days)" in message


def test_evaluate_says_when_only_the_validation_fit_is_short():
    (message,) = _pymc_warnings(_pymc_forecaster(250), "evaluate")  # 138 / 222 days
    assert "the validation fit spans 138 days, fewer than 183" in message
    assert "the test fit (222 days) and forecast() include it" in message


def test_forecast_on_a_short_series_says_it_has_no_annual_terms():
    (message,) = _pymc_warnings(_pymc_forecaster(150), "forecast")
    assert "fitted without annual (yearly) seasonality" in message
    assert "(forecast: 150 days)" in message


def test_no_validation_fit_is_described_when_the_training_window_is_below_the_floor():
    from horizon_forecaster.forecaster import HorizonForecaster

    def forecaster(n, validation, test):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return HorizonForecaster(
                raw_df=make_raw(n),
                date_column="date",
                target_column="orders",
                frequency="daily",
                models=["bayesianpymc"],
                validation_size=validation,
                test_size=test,
            )

    # 26 training rows, under PyMC's floor of 35: there is no validation fit,
    # and the test fit spans 194 days, so there is nothing to say.
    assert _pymc_warnings(forecaster(250, 168, 56), "evaluate") == []
    # 30 training rows; the test fit spans 130 days and is the only fit named.
    (message,) = _pymc_warnings(forecaster(150, 100, 20), "evaluate")
    assert "(test fit: 130 days)" in message
    assert "validation fit" not in message


def test_no_annual_note_from_exactly_the_minimum_span():
    n = ANNUAL_MIN_SPAN_DAYS + 84 + 28  # the training window spans exactly 183 days
    assert _pymc_warnings(_pymc_forecaster(n), "evaluate") == []
    assert _pymc_warnings(_pymc_forecaster(ANNUAL_MIN_SPAN_DAYS), "forecast") == []


@pytest.mark.parametrize(
    "stats, expected",
    [
        ({"max_rhat": 1.02, "min_ess_bulk": 150.0}, None),
        ({"max_rhat": 1.06, "min_ess_bulk": 500.0}, "largest R-hat 1.060, above 1.05"),
        ({"min_ess_bulk": 80.0}, "smallest effective sample size 80, below 100"),
        ({}, None),  # one chain: R-hat is NaN and is not reported
    ],
)
def test_convergence_warning_only_past_the_loose_thresholds(stats, expected):
    messages = _pymc_warnings(_pymc_forecaster(400), "evaluate", _FittedStub(**stats))
    if expected is None:
        assert messages == []
    else:
        (message,) = messages
        assert "did not converge cleanly in its test fit" in message
        assert expected in message


# --------------------------------------------------------------------------
# 3. real sampling integration (skipped where pymc is not installed)
# --------------------------------------------------------------------------

pytest.importorskip("pymc", reason="pymc is not installed; sampling tests are skipped")

# Tiny sampler settings: correctness of the plumbing, not posterior quality.
_FAST = {"draws": 80, "tune": 80, "chains": 1, "n_fourier": 2}


def _split(raw, n_train):
    return raw.iloc[:n_train].reset_index(drop=True), raw.iloc[n_train:].reset_index(drop=True)


def test_fit_predict_shape_and_bounds():
    raw = make_raw(60)
    train, test = _split(raw, 48)
    adapter = BayesianPyMCAdapter("date", "orders", ["spend", "promo"], "daily")
    adapted_train = adapter.transform_train(train)
    fut = adapter.transform_future(pd.Series(test["date"]), test[["spend", "promo"]])

    w = BayesianPyMCWrapper(frequency="daily", params=_FAST)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w.fit(adapted_train)
    out = w.predict(horizon=len(fut), future_exog=fut)

    assert list(out.columns) == ["ds", "yhat", "yhat_lower", "yhat_upper"]
    assert len(out) == len(test)
    assert (out["yhat"] >= 0).all()
    assert (out["yhat_lower"] <= out["yhat_upper"] + 1e-9).all()
    assert np.isfinite(out["yhat"].to_numpy()).all()
    # Bounds are a posterior-PREDICTIVE band, so the likelihood's sigma/nu
    # draws must be retained (not just the mean-function coefficients).
    assert {"sigma", "nu"}.issubset(w._post)
    assert (out["yhat_upper"] > out["yhat_lower"]).all()  # non-degenerate band


def test_fit_predict_no_independents():
    raw = make_raw(60)
    train, test = _split(raw, 48)
    adapter = BayesianPyMCAdapter("date", "orders", [], "daily")
    adapted_train = adapter.transform_train(train)
    fut = adapter.transform_future(pd.Series(test["date"]), None)

    w = BayesianPyMCWrapper(frequency="daily", params=_FAST)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w.fit(adapted_train)
    out = w.predict(horizon=len(fut), future_exog=fut)
    assert len(out) == len(test)
    assert (out["yhat"] >= 0).all()


def _fit_no_drivers(n):
    raw = make_raw(n)
    train, test = _split(raw, n - 10)
    adapter = BayesianPyMCAdapter("date", "orders", [], "daily")
    w = BayesianPyMCWrapper(frequency="daily", params=_FAST)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w.fit(adapter.transform_train(train))
    return w, w.predict(horizon=10, future_exog=adapter.transform_future(test["date"], None))


def test_a_fit_spanning_under_six_months_samples_no_annual_terms():
    w, out = _fit_no_drivers(60)
    assert w.annual_terms is False
    assert w._post["beta_long"].shape[1] == 0
    assert w._post["beta_short"].shape[1] == 4  # weekly order 2 (n_fourier=2)
    assert np.isfinite(out["yhat"].to_numpy()).all()


def test_a_fit_spanning_six_months_or_more_keeps_annual_terms():
    w, out = _fit_no_drivers(ANNUAL_MIN_SPAN_DAYS + 10)
    assert w.annual_terms is True
    assert w._post["beta_long"].shape[1] == 4
    assert np.isfinite(out["yhat"].to_numpy()).all()
    assert {"max_rhat", "min_ess_bulk", "num_divergences"} <= set(w.sample_stats)


def test_predict_horizon_mismatch_raises():
    raw = make_raw(60)
    train, _ = _split(raw, 48)
    adapter = BayesianPyMCAdapter("date", "orders", ["spend", "promo"], "daily")
    adapted_train = adapter.transform_train(train)
    w = BayesianPyMCWrapper(frequency="daily", params=_FAST)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w.fit(adapted_train)
    fut = adapter.transform_future(
        pd.Series(pd.date_range("2022-03-01", periods=3, freq="D")),
        pd.DataFrame({"spend": [100.0, 100.0, 100.0], "promo": [0, 1, 0]}),
    )
    with pytest.raises(ModelPredictionError):
        w.predict(horizon=5, future_exog=fut)


def test_save_load_roundtrip_reproduces_prediction(tmp_path):
    raw = make_raw(60)
    train, test = _split(raw, 48)
    adapter = BayesianPyMCAdapter("date", "orders", ["spend"], "daily")
    adapted_train = adapter.transform_train(train)
    fut = adapter.transform_future(pd.Series(test["date"]), test[["spend"]])

    w = BayesianPyMCWrapper(frequency="daily", params=_FAST)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w.fit(adapted_train)
    before = w.predict(horizon=len(fut), future_exog=fut)["yhat"].to_numpy()

    path = tmp_path / "pymc_model.pkl"
    w.save(str(path))
    reloaded = BayesianPyMCWrapper.load(str(path))
    after = reloaded.predict(horizon=len(fut), future_exog=fut)["yhat"].to_numpy()

    # predict() uses only the pickled numpy posterior, so it must match exactly.
    assert np.allclose(before, after)


def test_end_to_end_through_forecaster():
    from horizon_forecaster.forecaster import HorizonForecaster

    raw = make_raw(60)
    hf = HorizonForecaster(
        raw_df=raw,
        date_column="date",
        target_column="orders",
        frequency="daily",
        independent_columns=["spend", "promo"],
        models=["bayesianpymc"],
        model_params={"bayesianpymc": _FAST},  # skip grid search -> fast
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = hf.evaluate()
        forecasts = hf.forecast(horizon=5)

    assert "bayesianpymc" in results
    assert results["bayesianpymc"].mape.value >= 0
    assert "bayesianpymc" in forecasts
    assert len(forecasts["bayesianpymc"].forecast) == 5
    assert (forecasts["bayesianpymc"].forecast["yhat"] >= 0).all()
