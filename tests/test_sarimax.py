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


def make_adapted(n=90, with_exog=True, seed=0, test_size=14):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=n, freq="D")
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
    model = SARIMAXWrapper(frequency="daily", params={}, min_rows_required=35, random_seed=42)
    model.fit(train)
    return model, train, future, y_test


# ---------- fit / predict basics ----------

def test_fit_sets_is_fitted_and_selects_order(fitted):
    model, *_ = fitted
    assert model.is_fitted
    assert model._order is not None and model._seasonal_order is not None


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
    # Constant-variance band: the upper bound stays a bounded multiple of the
    # point forecast at every step (no horizon blow-up), and the ratio is
    # ~constant across steps rather than ballooning far out.
    assert (ratio < 20).all()
    assert ratio.max() / ratio.min() < 1.5


def test_predict_produces_reasonable_mape(fitted):
    from horizon_forecaster.evaluation.metrics import mape
    model, _, future, y_test = fitted
    out = model.predict(horizon=len(future), future_exog=future)
    # sanity floor: a working fit on this clean series should be well under 30%
    assert mape(y_test, out["yhat"].to_numpy()).value < 30


# ---------- error contract ----------

def test_predict_before_fit_raises():
    model = SARIMAXWrapper("daily", {}, min_rows_required=35)
    with pytest.raises(ModelPredictionError):
        model.predict(5, future_exog=None)


def test_insufficient_data_raises():
    train, _, _ = make_adapted()
    model = SARIMAXWrapper("daily", {}, min_rows_required=35)
    with pytest.raises(InsufficientDataError):
        model.fit(train.iloc[:20])


def test_predict_horizon_mismatch_raises(fitted):
    model, _, future, _ = fitted
    with pytest.raises(ModelPredictionError):
        model.predict(horizon=len(future) + 3, future_exog=future)


def test_predict_missing_exog_column_raises(fitted):
    model, _, future, _ = fitted
    bad = future.drop(columns=["spend"])
    with pytest.raises(ModelPredictionError):
        model.predict(horizon=len(future), future_exog=bad)


def test_all_candidates_failing_raises_convergence(monkeypatch):
    train, _, _ = make_adapted()
    model = SARIMAXWrapper("daily", {}, min_rows_required=35)

    def boom(*args, **kwargs):
        raise RuntimeError("forced fit failure")

    monkeypatch.setattr(model, "_fit_one", boom)
    with pytest.raises(ModelConvergenceError):
        model.fit(train)


# ---------- the improvement mechanics ----------

def test_bias_correction_lifts_point_forecast():
    model = SARIMAXWrapper("daily", {"log_clip_headroom": 100.0})
    base = model._back_transform(np.array([2.0, 2.0]), resid_var=0.0, train_max_log=10.0)
    lifted = model._back_transform(np.array([2.0, 2.0]), resid_var=0.5, train_max_log=10.0)
    assert (lifted > base).all()


def test_clip_prevents_explosion():
    model = SARIMAXWrapper("daily", {"log_clip_headroom": 1.0})
    out = model._back_transform(np.array([50.0]), resid_var=0.0, train_max_log=5.0)
    # capped at expm1(5 + 1) -> must not explode toward expm1(50)
    assert out[0] <= np.expm1(6.0) + 1e-6


def test_adf_differencing_order_flags_trend():
    model = SARIMAXWrapper("daily", {})
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


# ---------- integration through the orchestrator (Prophet-free) ----------

def test_sarimax_flows_through_forecaster_and_ensemble():
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
        independent_columns=["spend"], models=["catboost", "sarimax"], mape_threshold=100,
    )
    results = f.evaluate()
    assert set(results) == {"catboost", "sarimax"}
    # SARIMAX self-tunes: no forecaster-level grid search
    assert results["sarimax"].grid_search_summary is None
    # first bounds-emitting model to reach test_predictions
    assert "yhat_lower" in results["sarimax"].test_predictions.columns
    assert f.evaluate_ensemble is not None and f.evaluate_ensemble.combined is not None

    forecasts = f.forecast(horizon=10)
    assert set(forecasts) == {"catboost", "sarimax"}
    assert len(forecasts["sarimax"].forecast) == 10
