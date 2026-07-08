import pandas as pd
import pytest

from horizon_forecaster.exceptions import (
    InsufficientDataError,
    ModelConvergenceError,
    ModelPredictionError,
)
from horizon_forecaster.models.prophet_wrapper import ProphetWrapper

def make_train_df(n=30):
    return pd.DataFrame({
        "ds": pd.date_range("2026-01-01", periods=n, freq="D"),
        "y": [10 + i * 0.5 for i in range(n)],
    })

def make_train_df_with_regressor(n=30):
    df = make_train_df(n)
    df["spend"] = [100 + i * 2 for i in range(n)]
    return df

# ---------- fit() ----------

def test_fit_succeeds_on_sufficient_data():
    model = ProphetWrapper(frequency="daily")
    result = model.fit(make_train_df(), target_column="y")
    assert result is model
    assert model.is_fitted is True

def test_fit_raises_insufficient_data_error_below_min_rows():
    model = ProphetWrapper(frequency="daily")
    small_df = make_train_df(n=5)
    with pytest.raises(InsufficientDataError) as exc_info:
        model.fit(small_df, target_column="y")
    assert model.is_fitted is False
    assert exc_info.value.available_periods == 5

def test_fit_raises_exactly_one_row_below_min_rows_boundary():
    model = ProphetWrapper(frequency="daily")
    df = make_train_df(n=model.MIN_ROWS_REQUIRED - 1)
    with pytest.raises(InsufficientDataError):
        model.fit(df, target_column="y")

def test_fit_succeeds_at_exact_min_rows_boundary():
    model = ProphetWrapper(frequency="daily")
    df = make_train_df(n=model.MIN_ROWS_REQUIRED)
    model.fit(df, target_column="y")
    assert model.is_fitted is True

def test_fit_raises_model_prediction_error_when_missing_ds_or_y_columns():
    model = ProphetWrapper(frequency="daily")
    bad_df = pd.DataFrame({"date": pd.date_range("2026-01-01", periods=15), "value": range(15)})
    with pytest.raises(ModelPredictionError):
        model.fit(bad_df, target_column="y")

def test_fit_registers_regressor_columns():
    model = ProphetWrapper(frequency="daily")
    model.fit(make_train_df_with_regressor(), target_column="y")
    assert model._regressor_columns == ["spend"]

def test_fit_with_no_regressor_columns():
    model = ProphetWrapper(frequency="daily")
    model.fit(make_train_df(), target_column="y")
    assert model._regressor_columns == []

def test_fit_accepts_param_overrides():
    model = ProphetWrapper(frequency="daily", params={"yearly_seasonality": False})
    model.fit(make_train_df(), target_column="y")
    assert model.is_fitted is True

def test_fit_raises_model_convergence_error_on_genuine_prophet_failure():
    """
    Duplicate ds timestamps cause a genuine Prophet/Stan optimization
    failure (confirmed empirically, not simulated) — this exercises the
    real exception-translation path in fit(), not a fabricated one.
    """
    model = ProphetWrapper(frequency="daily")
    bad_df = pd.DataFrame({
        "ds": [pd.Timestamp("2026-01-01")] * 15,  # all identical -> invalid for Prophet
        "y": list(range(15)),
    })
    with pytest.raises(ModelConvergenceError) as exc_info:
        model.fit(bad_df, target_column="y")
    assert model.is_fitted is False
    assert exc_info.value.original_exception is not None

# ---------- predict() ----------

def test_predict_before_fit_raises():
    model = ProphetWrapper(frequency="daily")
    future = pd.DataFrame({"ds": pd.date_range("2026-02-01", periods=5)})
    with pytest.raises(ModelPredictionError):
        model.predict(horizon=5, future_exog=future)

def test_predict_with_none_future_exog_raises():
    model = ProphetWrapper(frequency="daily")
    model.fit(make_train_df(), target_column="y")
    with pytest.raises(ModelPredictionError):
        model.predict(horizon=5, future_exog=None)

def test_predict_returns_expected_shape_and_columns():
    model = ProphetWrapper(frequency="daily")
    model.fit(make_train_df(), target_column="y")
    future = pd.DataFrame({"ds": pd.date_range("2026-01-31", periods=7, freq="D")})
    forecast = model.predict(horizon=7, future_exog=future)
    assert list(forecast.columns) == ["ds", "yhat", "yhat_lower", "yhat_upper"]
    assert len(forecast) == 7

def test_predict_raises_when_horizon_does_not_match_future_exog_length():
    model = ProphetWrapper(frequency="daily")
    model.fit(make_train_df(), target_column="y")
    future = pd.DataFrame({"ds": pd.date_range("2026-01-31", periods=7, freq="D")})
    with pytest.raises(ModelPredictionError):
        model.predict(horizon=10, future_exog=future)  # mismatch: 7 rows vs horizon=10

def test_predict_raises_when_future_exog_missing_ds_column():
    model = ProphetWrapper(frequency="daily")
    model.fit(make_train_df(), target_column="y")
    future = pd.DataFrame({"date": pd.date_range("2026-01-31", periods=5)})
    with pytest.raises(ModelPredictionError):
        model.predict(horizon=5, future_exog=future)

def test_predict_raises_when_missing_required_regressor_in_future_exog():
    model = ProphetWrapper(frequency="daily")
    model.fit(make_train_df_with_regressor(), target_column="y")
    # future_exog missing "spend", which was registered as a regressor during fit
    future = pd.DataFrame({"ds": pd.date_range("2026-01-31", periods=5, freq="D")})
    with pytest.raises(ModelPredictionError):
        model.predict(horizon=5, future_exog=future)

def test_predict_succeeds_with_required_regressor_present():
    model = ProphetWrapper(frequency="daily")
    model.fit(make_train_df_with_regressor(), target_column="y")
    future = pd.DataFrame({
        "ds": pd.date_range("2026-01-31", periods=5, freq="D"),
        "spend": [160, 162, 164, 166, 168],
    })
    forecast = model.predict(horizon=5, future_exog=future)
    assert len(forecast) == 5
    assert not forecast["yhat"].isna().any()

def test_predict_raises_model_prediction_error_on_genuine_prophet_failure():
    """
    NaN values in a registered regressor column cause a genuine Prophet
    predict() failure (confirmed empirically) — this exercises the real
    exception-translation path, not a fabricated one. Note this NaN
    passes our own wrapper-level validation (the column IS present,
    just contains NaN values), so it genuinely reaches Prophet's own
    internal predict() call before failing.
    """
    import numpy as np

    model = ProphetWrapper(frequency="daily")
    model.fit(make_train_df_with_regressor(), target_column="y")
    future = pd.DataFrame({
        "ds": pd.date_range("2026-01-31", periods=5, freq="D"),
        "spend": [np.nan] * 5,
    })
    with pytest.raises(ModelPredictionError) as exc_info:
        model.predict(horizon=5, future_exog=future)
    assert exc_info.value.original_exception is not None

def test_fit_predict_chaining_end_to_end():
    future = pd.DataFrame({"ds": pd.date_range("2026-01-31", periods=3, freq="D")})
    forecast = (
        ProphetWrapper(frequency="daily")
        .fit(make_train_df(), target_column="y")
        .predict(horizon=3, future_exog=future)
    )
    assert len(forecast) == 3

# ---------- save() / load() ----------

def test_save_raises_if_called_before_fit(tmp_path):
    model = ProphetWrapper(frequency="daily")
    with pytest.raises(ModelPredictionError):
        model.save(str(tmp_path / "model.json"))

def test_save_and_load_roundtrip_preserves_prediction(tmp_path):
    save_path = str(tmp_path / "prophet_model.json")

    original = ProphetWrapper(frequency="daily")
    original.fit(make_train_df(), target_column="y")
    original.save(save_path)

    reloaded = ProphetWrapper.load(save_path)
    assert reloaded.is_fitted is True
    assert reloaded.frequency == "daily"
    assert reloaded._regressor_columns == []

    future = pd.DataFrame({"ds": pd.date_range("2026-01-31", periods=5, freq="D")})
    original_forecast = original.predict(horizon=5, future_exog=future)
    reloaded_forecast = reloaded.predict(horizon=5, future_exog=future)

    # Same model, same input -> same output
    pd.testing.assert_frame_equal(
        original_forecast.reset_index(drop=True),
        reloaded_forecast.reset_index(drop=True),
    )

def test_save_and_load_preserves_regressor_columns(tmp_path):
    save_path = str(tmp_path / "prophet_model_with_regressor.json")

    original = ProphetWrapper(frequency="daily")
    original.fit(make_train_df_with_regressor(), target_column="y")
    original.save(save_path)

    reloaded = ProphetWrapper.load(save_path)
    assert reloaded._regressor_columns == ["spend"]

    future = pd.DataFrame({
        "ds": pd.date_range("2026-01-31", periods=3, freq="D"),
        "spend": [160, 162, 164],
    })
    forecast = reloaded.predict(horizon=3, future_exog=future)
    assert len(forecast) == 3

def test_loaded_model_is_a_fresh_instance_not_shared_state(tmp_path):
    """
    Confirms reloading produces a genuinely independent instance, not
    something accidentally still referencing the original in-memory
    object (which would defeat the purpose of cross-session persistence).
    """
    save_path = str(tmp_path / "prophet_model.json")
    original = ProphetWrapper(frequency="daily")
    original.fit(make_train_df(), target_column="y")
    original.save(save_path)

    reloaded = ProphetWrapper.load(save_path)
    assert reloaded is not original
    assert reloaded._fitted_model is not original._fitted_model