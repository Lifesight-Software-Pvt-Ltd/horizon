import pandas as pd
import pytest

from horizon_forecaster.exceptions import (
    InsufficientDataError,
    InvalidConfigError,
    UnsupportedFrequencyError,
)
from horizon_forecaster.forecaster import HorizonForecaster, EvaluationResult

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