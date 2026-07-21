import numpy as np
import pandas as pd
import pytest

from horizon_forecaster.data_adapter.sarimax_adapter import SARIMAXAdapter
from horizon_forecaster.models.sarimax_wrapper import SARIMAXWrapper
from horizon_forecaster.exceptions import (
    InsufficientDataError,
    ModelConvergenceError,
    ModelPredictionError,
)

# A single grid candidate's worth of params, as the forecaster would merge them.
BASE = {"p": 1, "q": 1, "P": 0, "Q": 0, "use_seasonal": False, "fourier_order_annual": 0}


def make_adapted(n=90, with_exog=True, seed=0, test_size=14, freq="D"):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=n, freq=freq)
    trend = 100 + np.arange(n)
    season = 15 * np.sin(2 * np.pi * np.arange(n) / 7)
    spend = np.abs(rng.normal(200, 40, n)) + 1.0
    y = np.clip(trend + season + 0.05 * spend + rng.normal(0, 8, n), 1, None)
    raw = pd.DataFrame({"date": dates, "orders": y, "spend": spend, "promo": rng.integers(0, 2, n)})

    cols = ["spend", "promo"] if with_exog else []
    adapter = SARIMAXAdapter("date", "orders", cols, "daily")
    train, test = raw.iloc[: n - test_size], raw.iloc[n - test_size:]
    adapted_train = adapter.transform_train(train)
    adapted_future = adapter.transform_future(
        pd.Series(test["date"].to_numpy()), test[cols] if cols else None
    )
    return adapted_train, adapted_future, test["orders"].to_numpy()


@pytest.fixture(scope="module")
def fitted():
    train, future, y_test = make_adapted()
    model = SARIMAXWrapper("daily", dict(BASE), min_rows_required=35, random_seed=42)
    model.fit(train)
    return model, train, future, y_test


# ---------- fit / predict basics ----------

def test_fit_uses_grid_order_and_adf_derived_d(fitted):
    model, *_ = fitted
    assert model.is_fitted
    # p and q come from the grid params; d is derived from the data by ADF
    assert model._order[0] == BASE["p"]
    assert model._order[2] == BASE["q"]
    assert model._d in (0, 1)


def test_predict_shape_bounds_and_nonnegative(fitted):
    model, _, future, _ = fitted
    out = model.predict(horizon=len(future), future_exog=future)
    assert list(out.columns) == ["ds", "yhat", "yhat_lower", "yhat_upper"]
    assert len(out) == len(future)
    assert (out["yhat"] >= 0).all()
    assert (out["yhat_lower"] <= out["yhat_upper"]).all()


def test_prediction_interval_is_a_stable_band_not_exploding(fitted):
    model, _, future, _ = fitted
    out = model.predict(horizon=len(future), future_exog=future)
    ratio = (out["yhat_upper"] / out["yhat"]).to_numpy()
    assert (ratio < 20).all()
    assert ratio.max() / ratio.min() < 1.5


def test_predict_produces_reasonable_mape(fitted):
    from horizon_forecaster.evaluation.metrics import mape
    model, _, future, y_test = fitted
    out = model.predict(horizon=len(future), future_exog=future)
    assert mape(y_test, out["yhat"].to_numpy()).value < 30


# ---------- annual Fourier (now the wrapper's job) ----------

def test_fourier_order_zero_adds_no_exog_columns():
    train, future, _ = make_adapted(with_exog=False)
    model = SARIMAXWrapper("daily", {**BASE, "fourier_order_annual": 0}, 35)
    model.fit(train)
    assert model._build_exog(train) is None  # no independents, no Fourier -> no exog


def test_fourier_order_adds_two_columns_per_harmonic():
    train, _, _ = make_adapted(with_exog=False)
    model = SARIMAXWrapper("daily", {**BASE, "fourier_order_annual": 3}, 35)
    model.fit(train)
    exog = model._build_exog(train)
    assert exog.shape == (len(train), 6)  # sin+cos per harmonic


def test_fourier_is_phase_aligned_by_absolute_date_not_position():
    """
    Terms are built from day offsets against the fit-time reference date, so
    a gap in the future dates must NOT shift the phase (a positional index
    would). Two future frames covering the same dates -- one contiguous, one
    with rows removed -- must agree on the shared dates.
    """
    train, _, _ = make_adapted(with_exog=False)
    model = SARIMAXWrapper("daily", {**BASE, "fourier_order_annual": 2}, 35)
    model.fit(train)

    contiguous = pd.DataFrame({"ds": pd.date_range("2022-05-01", periods=10, freq="D")})
    gapped = contiguous.iloc[[0, 3, 7]].reset_index(drop=True)
    full_terms = model._fourier_terms(contiguous["ds"])
    gapped_terms = model._fourier_terms(gapped["ds"])
    np.testing.assert_allclose(full_terms[[0, 3, 7]], gapped_terms, rtol=1e-12)


def test_fourier_terms_are_bounded():
    train, _, _ = make_adapted(with_exog=False)
    model = SARIMAXWrapper("daily", {**BASE, "fourier_order_annual": 4}, 35)
    model.fit(train)
    terms = model._fourier_terms(train["ds"])
    assert terms.min() >= -1.0000001 and terms.max() <= 1.0000001


# ---------- error contract ----------

def test_predict_before_fit_raises():
    model = SARIMAXWrapper("daily", dict(BASE), min_rows_required=35)
    with pytest.raises(ModelPredictionError):
        model.predict(5, future_exog=None)


def test_insufficient_data_raises():
    train, _, _ = make_adapted()
    model = SARIMAXWrapper("daily", dict(BASE), min_rows_required=35)
    with pytest.raises(InsufficientDataError):
        model.fit(train.iloc[:20])


def test_predict_horizon_mismatch_raises(fitted):
    model, _, future, _ = fitted
    with pytest.raises(ModelPredictionError):
        model.predict(horizon=len(future) + 3, future_exog=future)


def test_predict_missing_exog_column_raises(fitted):
    model, _, future, _ = fitted
    with pytest.raises(ModelPredictionError):
        model.predict(horizon=len(future), future_exog=future.drop(columns=["spend"]))


def test_fit_failure_raises_convergence(monkeypatch):
    train, _, _ = make_adapted()
    model = SARIMAXWrapper("daily", dict(BASE), min_rows_required=35)

    def boom(*args, **kwargs):
        raise RuntimeError("forced fit failure")

    monkeypatch.setattr(model, "_fit_one", boom)
    with pytest.raises(ModelConvergenceError):
        model.fit(train)


def test_divergence_guard_rejects_clip_saturated_forecast():
    train, future, _ = make_adapted()
    model = SARIMAXWrapper("daily", dict(BASE), min_rows_required=35)
    model.fit(train)
    model._clip_anchor_log = -5.0  # force the clip cap far below any real forecast
    with pytest.raises(ModelPredictionError):
        model.predict(horizon=len(future), future_exog=future)


# ---------- back-transform behaviour ----------

def test_bias_correction_lifts_point_forecast():
    model = SARIMAXWrapper("daily", {**BASE, "log_clip_headroom": 100.0})
    model._resid_var = 0.5
    base = model._back_transform(np.array([2.0]), 0.0, 10.0)
    lifted = model._back_transform(np.array([2.0]), 0.5, 10.0)
    assert lifted[0] > base[0]


def test_bias_correction_can_be_disabled():
    off = SARIMAXWrapper("daily", {**BASE, "log_clip_headroom": 100.0, "bias_correction": False})
    on = SARIMAXWrapper("daily", {**BASE, "log_clip_headroom": 100.0})
    args = (np.array([2.0]), 0.5, 10.0)
    assert off._back_transform(*args)[0] < on._back_transform(*args)[0]
    assert off._back_transform(*args)[0] == pytest.approx(np.expm1(2.0))


def test_clip_prevents_explosion():
    model = SARIMAXWrapper("daily", {**BASE, "log_clip_headroom": 1.0})
    out = model._back_transform(np.array([50.0]), 0.0, 5.0)
    assert out[0] <= np.expm1(6.0) + 1e-6


def test_clip_anchor_uses_quantile_not_raw_max():
    train, _, _ = make_adapted()
    train = train.copy()
    train.loc[train.index[0], "y"] = train["y"].max() * 100
    model = SARIMAXWrapper("daily", dict(BASE), min_rows_required=35)
    model.fit(train)
    assert model._clip_anchor_log < float(np.log1p(train["y"].max()))


def test_adf_differencing_order_flags_trend():
    model = SARIMAXWrapper("daily", dict(BASE))
    rng = np.random.default_rng(0)
    trending = np.log1p(100 + np.arange(80) * 2.0 + rng.normal(0, 5, 80))
    assert model._adf_differencing_order(trending) == 1


# ---------- persistence ----------

def test_save_load_roundtrip_preserves_prediction(tmp_path, fitted):
    model, _, future, _ = fitted
    path = str(tmp_path / "sarimax.pkl")
    model.save(path)
    loaded = SARIMAXWrapper.load(path)
    before = model.predict(len(future), future)["yhat"].to_numpy()
    after = loaded.predict(len(future), future)["yhat"].to_numpy()
    np.testing.assert_allclose(before, after, rtol=1e-6)


# ---------- integration through the orchestrator ----------

def test_sarimax_uses_the_standard_grid_search_like_other_models():
    from horizon_forecaster.forecaster import HorizonForecaster, MODEL_REGISTRY

    assert "sarimax" in MODEL_REGISTRY

    rng = np.random.default_rng(5)
    n = 150
    dates = pd.date_range("2022-01-01", periods=n, freq="D")
    y = np.clip(
        200 + 1.2 * np.arange(n) + 20 * np.sin(2 * np.pi * np.arange(n) / 7) + rng.normal(0, 12, n),
        1, None,
    )
    df = pd.DataFrame({"date": dates, "orders": y, "spend": np.abs(rng.normal(300, 60, n)) + 1})

    f = HorizonForecaster(
        df, date_column="date", target_column="orders", frequency="daily",
        independent_columns=["spend"], models=["sarimax"], mape_threshold=100,
    )
    assert f._use_grid_search["sarimax"] is True  # no longer self-tuning

    result = f.evaluate()["sarimax"]
    # SARIMAX now reports its grid like every other model
    assert result.grid_search_summary is not None
    assert len(result.grid_search_summary) >= 1
    assert "yhat_lower" in result.test_predictions.columns

    forecasts = f.forecast(horizon=10)
    assert len(forecasts["sarimax"].forecast) == 10
