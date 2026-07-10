import warnings

import pandas as pd
import pytest

from horizon_forecaster.exceptions import (
    InsufficientDataError,
    InvalidConfigError,
    UnsupportedFrequencyError,
)
from horizon_forecaster.forecaster import HorizonForecaster, EvaluationResult, ForecastResult

def make_raw_df(n=60, freq="D", with_spend=False, with_zero_variance_col=False):
    df = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=n, freq=freq),
        "orders": [100 + i * 2 for i in range(n)],
    })
    if with_spend:
        df["spend"] = [50 + i for i in range(n)]
    if with_zero_variance_col:
        df["constant_col"] = 5  # same value every row -> zero variance
    return df

# ---------- constructor validation ----------

def test_constructor_validates_frequency():
    df = make_raw_df()
    with pytest.raises(UnsupportedFrequencyError):
        HorizonForecaster(df, date_column="date", target_column="orders", frequency="monthly")

def test_constructor_validates_training_size():
    df = make_raw_df()
    with pytest.raises(InvalidConfigError):
        HorizonForecaster(
            df, date_column="date", target_column="orders", frequency="daily", training_size=30
        )

def test_constructor_default_training_size_is_80_percent():
    df = make_raw_df()
    forecaster = HorizonForecaster(df, date_column="date", target_column="orders", frequency="daily")
    assert forecaster.training_fraction == 0.8

def test_constructor_drops_zero_variance_independent_columns():
    df = make_raw_df(with_zero_variance_col=True)
    forecaster = HorizonForecaster(
        df,
        date_column="date",
        target_column="orders",
        frequency="daily",
        independent_columns=["constant_col"],
    )
    assert "constant_col" in forecaster.dropped_zero_variance_columns
    # Critically: also removed from independent_columns, so the adapter's
    # hard-error never fires against a column WE dropped, not the user.
    assert "constant_col" not in forecaster.independent_columns

def test_constructor_sorts_data_chronologically():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2026-01-03", "2026-01-01", "2026-01-02"]),
        "orders": [30, 10, 20],
    })
    forecaster = HorizonForecaster(df, date_column="date", target_column="orders", frequency="daily")
    assert forecaster.raw_df["orders"].tolist() == [10, 20, 30]

# ---------- config loading ----------

def test_config_loaded_correctly_for_daily():
    df = make_raw_df(n=60)
    forecaster = HorizonForecaster(df, date_column="date", target_column="orders", frequency="daily")
    assert forecaster._config["wrapper_config"]["min_rows_required"] == 14
    assert forecaster._config["hyperparameters"]["weekly_seasonality"] is True

def test_config_loaded_correctly_for_weekly():
    df = make_raw_df(n=60, freq="7D")
    forecaster = HorizonForecaster(df, date_column="date", target_column="orders", frequency="weekly")
    assert forecaster._config["wrapper_config"]["min_rows_required"] == 8
    assert forecaster._config["hyperparameters"]["weekly_seasonality"] is False

# ---------- evaluate() ----------

def test_evaluate_returns_structured_result():
    df = make_raw_df(n=60)
    forecaster = HorizonForecaster(df, date_column="date", target_column="orders", frequency="daily")
    result = forecaster.evaluate()

    assert isinstance(result, EvaluationResult)
    assert result.model_name == "prophet"
    assert result.n_train == 48  # 60 * 0.8
    assert result.n_test == 12
    assert len(result.test_predictions) == 12
    assert result.fitted_model.is_fitted is True

def test_evaluate_produces_low_mape_on_clean_trend():
    """
    A clean, noise-free linear trend should be near-trivially predictable
    -> confirms the whole wired pipeline (split -> adapt -> fit ->
    predict -> MAPE) produces a numerically sensible result, not just
    code that runs without error.
    """
    df = make_raw_df(n=60)
    forecaster = HorizonForecaster(df, date_column="date", target_column="orders", frequency="daily")
    result = forecaster.evaluate()
    assert result.mape.value < 5.0  # should be very low on a clean trend

def test_evaluate_test_predictions_have_expected_columns():
    df = make_raw_df(n=60)
    forecaster = HorizonForecaster(df, date_column="date", target_column="orders", frequency="daily")
    result = forecaster.evaluate()
    assert set(["ds", "y", "yhat", "yhat_lower", "yhat_upper"]).issubset(
        set(result.test_predictions.columns)
    )

def test_evaluate_works_with_independent_columns():
    df = make_raw_df(n=60, with_spend=True)
    forecaster = HorizonForecaster(
        df,
        date_column="date",
        target_column="orders",
        frequency="daily",
        independent_columns=["spend"],
    )
    result = forecaster.evaluate()
    assert result.fitted_model._regressor_columns == ["spend"]

def test_evaluate_raises_invalid_config_error_when_training_size_leaves_no_test_rows():
    # A tiny dataset where int(n * 0.99...) could plausibly consume all rows
    # is contrived; instead directly force the scenario via a very small n
    # combined with a training_size that floors to 100% of available rows.
    df = make_raw_df(n=1)
    forecaster = HorizonForecaster(
        df, date_column="date", target_column="orders", frequency="daily", training_size=100
    )
    with pytest.raises(InvalidConfigError):
        forecaster.evaluate()

def test_evaluate_propagates_insufficient_data_error_on_tiny_dataset():
    df = make_raw_df(n=10)  # 80% train -> 8 rows, below daily floor of 14
    forecaster = HorizonForecaster(df, date_column="date", target_column="orders", frequency="daily")
    with pytest.raises(InsufficientDataError):
        forecaster.evaluate()

def test_evaluate_weekly_frequency_end_to_end():
    df = make_raw_df(n=40, freq="7D")
    forecaster = HorizonForecaster(df, date_column="date", target_column="orders", frequency="weekly")
    result = forecaster.evaluate()
    assert result.model_name == "prophet"
    assert result.mape.value < 10.0

# ---------- forecast() ----------

def test_forecast_returns_structured_result():
    df = make_raw_df(n=60)
    forecaster = HorizonForecaster(df, date_column="date", target_column="orders", frequency="daily")
    result = forecaster.forecast(horizon=10)

    assert isinstance(result, ForecastResult)
    assert result.model_name == "prophet"
    assert result.horizon == 10
    assert len(result.forecast) == 10
    assert result.used_default_future_strategy is False  # no independent columns at all

def test_forecast_continues_trend_correctly():
    df = make_raw_df(n=60)  # orders = 100 + i*2, so trend is +2/period
    forecaster = HorizonForecaster(df, date_column="date", target_column="orders", frequency="daily")
    result = forecaster.forecast(horizon=5)
    yhats = result.forecast["yhat"].tolist()
    diffs = [round(yhats[i + 1] - yhats[i], 4) for i in range(len(yhats) - 1)]
    assert all(d == 2.0 for d in diffs)

def test_forecast_raises_on_non_positive_horizon():
    df = make_raw_df(n=60)
    forecaster = HorizonForecaster(df, date_column="date", target_column="orders", frequency="daily")
    with pytest.raises(InvalidConfigError):
        forecaster.forecast(horizon=0)
    with pytest.raises(InvalidConfigError):
        forecaster.forecast(horizon=-5)

def test_forecast_refits_only_once_and_caches_across_calls():
    """
    Critical design contract: the full-data refit must happen exactly
    once per instance, regardless of how many times forecast() is
    called afterward (even with different horizons).
    """
    df = make_raw_df(n=60)
    forecaster = HorizonForecaster(df, date_column="date", target_column="orders", frequency="daily")

    assert forecaster._forecast_model is None
    forecaster.forecast(horizon=5)
    model_after_first_call = forecaster._forecast_model
    assert model_after_first_call is not None

    forecaster.forecast(horizon=10)  # different horizon
    model_after_second_call = forecaster._forecast_model

    assert model_after_first_call is model_after_second_call  # same object, not refit

def test_forecast_uses_default_strategy_when_independent_data_not_supplied():
    df = make_raw_df(n=60, with_spend=True)
    forecaster = HorizonForecaster(
        df,
        date_column="date",
        target_column="orders",
        frequency="daily",
        independent_columns=["spend"],
    )
    result = forecaster.forecast(horizon=5)
    assert result.used_default_future_strategy is True
    assert len(result.forecast) == 5

def test_forecast_uses_supplied_future_independent_data_when_provided():
    df = make_raw_df(n=60, with_spend=True)
    forecaster = HorizonForecaster(
        df,
        date_column="date",
        target_column="orders",
        frequency="daily",
        independent_columns=["spend"],
    )
    future_spend = pd.DataFrame({"spend": [500, 510, 520, 530, 540]})
    result = forecaster.forecast(horizon=5, future_independent_data=future_spend)
    assert result.used_default_future_strategy is False
    assert len(result.forecast) == 5

def test_default_future_strategy_reuses_last_n_periods_when_horizon_fits_within_history():
    df = make_raw_df(n=60, with_spend=True)  # spend = 50 + i, last values: ..., 107, 108, 109
    forecaster = HorizonForecaster(
        df,
        date_column="date",
        target_column="orders",
        frequency="daily",
        independent_columns=["spend"],
    )
    default_future = forecaster._build_default_future_independent_data(horizon=3)
    # last 3 actual spend values in a 60-row series starting at 50: 107, 108, 109
    assert default_future["spend"].tolist() == [107, 108, 109]

def test_default_future_strategy_cycles_when_horizon_exceeds_history():
    df = make_raw_df(n=20, with_spend=True)  # spend = 50..69 (20 values)
    forecaster = HorizonForecaster(
        df,
        date_column="date",
        target_column="orders",
        frequency="daily",
        independent_columns=["spend"],
    )
    # horizon of 45 > 20 rows of history -> must cycle, not crash
    default_future = forecaster._build_default_future_independent_data(horizon=45)
    assert len(default_future) == 45
    assert not default_future["spend"].isna().any()

def test_forecast_warns_when_horizon_far_exceeds_history():
    df = make_raw_df(n=20, with_spend=True)
    forecaster = HorizonForecaster(
        df,
        date_column="date",
        target_column="orders",
        frequency="daily",
        independent_columns=["spend"],
    )
    with pytest.warns(UserWarning, match="Forecasting 45 periods from only 20 periods"):
        forecaster.forecast(horizon=45)

def test_forecast_does_not_warn_when_horizon_is_reasonable():
    df = make_raw_df(n=60)
    forecaster = HorizonForecaster(df, date_column="date", target_column="orders", frequency="daily")
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning becomes a test failure
        forecaster.forecast(horizon=10)  # well within 2x of 60 rows of history

def test_forecast_weekly_frequency_produces_correct_future_dates():
    df = make_raw_df(n=40, freq="7D")
    forecaster = HorizonForecaster(df, date_column="date", target_column="orders", frequency="weekly")
    result = forecaster.forecast(horizon=3)
    last_train_date = df["date"].max()
    expected_first_future_date = last_train_date + pd.Timedelta(days=7)
    assert result.forecast["ds"].iloc[0] == expected_first_future_date