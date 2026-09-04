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
from horizon_forecaster.models.bayesianpymc_wrapper import BayesianPyMCWrapper

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
# 3. real sampling integration (local/CI gate)
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
