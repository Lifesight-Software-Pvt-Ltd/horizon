import pandas as pd
import pytest

from horizon_forecaster.exceptions import (
    InsufficientDataError,
    ModelConvergenceError,
    ModelPredictionError,
)
from horizon_forecaster.models.prophet_wrapper import ProphetWrapper


def make_train_df(n=30):
    return pd.DataFrame(
        {
            "ds": pd.date_range("2026-01-01", periods=n, freq="D"),
            "y": [10 + i * 0.5 for i in range(n)],
        }
    )


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
    df = make_train_df(n=model.min_rows_required - 1)
    with pytest.raises(InsufficientDataError):
        model.fit(df, target_column="y")


def test_fit_succeeds_at_exact_min_rows_boundary():
    model = ProphetWrapper(frequency="daily")
    df = make_train_df(n=model.min_rows_required)
    model.fit(df, target_column="y")
    assert model.is_fitted is True


def test_min_rows_required_is_frequency_aware_by_default():
    daily_model = ProphetWrapper(frequency="daily")
    weekly_model = ProphetWrapper(frequency="weekly")
    assert daily_model.min_rows_required == 14
    assert weekly_model.min_rows_required == 8
    assert daily_model.min_rows_required != weekly_model.min_rows_required


def test_min_rows_required_can_be_explicitly_overridden():
    model = ProphetWrapper(frequency="daily", min_rows_required=100)
    assert model.min_rows_required == 100
    df = make_train_df(n=50)
    with pytest.raises(InsufficientDataError):
        model.fit(df, target_column="y")  # 50 < explicit override of 100


def test_random_seed_defaults_to_42_and_can_be_overridden():
    default_model = ProphetWrapper(frequency="daily")
    custom_model = ProphetWrapper(frequency="daily", random_seed=123)
    assert default_model.random_seed == 42
    assert custom_model.random_seed == 123


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
    bad_df = pd.DataFrame(
        {
            "ds": [pd.Timestamp("2026-01-01")] * 15,  # all identical -> invalid for Prophet
            "y": list(range(15)),
        }
    )
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


def test_predict_without_uncertainty_samples_returns_only_the_point_forecast():
    # HorizonForecaster's config sets uncertainty_samples=0: Prophet then
    # skips its interval simulation and returns no bound columns, and the
    # point forecast is unchanged.
    future = pd.DataFrame({"ds": pd.date_range("2026-01-31", periods=7, freq="D")})
    with_interval = ProphetWrapper(frequency="daily").fit(make_train_df(), "y")
    without = ProphetWrapper(frequency="daily", params={"uncertainty_samples": 0})
    forecast = without.fit(make_train_df(), "y").predict(horizon=7, future_exog=future)
    assert list(forecast.columns) == ["ds", "yhat"]
    assert forecast["yhat"].tolist() == pytest.approx(
        with_interval.predict(horizon=7, future_exog=future)["yhat"].tolist()
    )


def test_the_config_turns_off_prophets_own_interval():
    from horizon_forecaster.forecaster import _default_params

    for frequency in ("daily", "weekly"):
        base = _default_params()["prophet"][frequency]["base_hyperparameters"]
        assert base["uncertainty_samples"] == 0


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
    future = pd.DataFrame(
        {
            "ds": pd.date_range("2026-01-31", periods=5, freq="D"),
            "spend": [160, 162, 164, 166, 168],
        }
    )
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
    future = pd.DataFrame(
        {
            "ds": pd.date_range("2026-01-31", periods=5, freq="D"),
            "spend": [np.nan] * 5,
        }
    )
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

    future = pd.DataFrame(
        {
            "ds": pd.date_range("2026-01-31", periods=3, freq="D"),
            "spend": [160, 162, 164],
        }
    )
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


def test_save_and_load_preserves_wrapper_level_config():
    """
    min_rows_required and random_seed are wrapper-level config, not
    Prophet hyperparameters — confirm they survive the save/load
    roundtrip too, not just the underlying Prophet model.
    """
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = os.path.join(tmp_dir, "model.json")
        original = ProphetWrapper(frequency="weekly", min_rows_required=12, random_seed=99)
        original.fit(make_train_df(n=20), target_column="y")
        original.save(save_path)

        reloaded = ProphetWrapper.load(save_path)
        assert reloaded.min_rows_required == 12
        assert reloaded.random_seed == 99
        assert reloaded.frequency == "weekly"


def test_fit_failure_message_names_the_cause():
    # evaluate_errors must get the cause, not "Prophet failed to fit." alone.
    model = ProphetWrapper(frequency="daily", params={"not_a_prophet_setting": 1})
    with pytest.raises(ModelConvergenceError, match=r"Prophet failed to fit \(TypeError: "):
        model.fit(make_train_df(), target_column="y")
