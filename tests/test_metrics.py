import math

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from horizon_forecaster.evaluation.metrics import (
    MapeResult,
    error_correlation,
    mae,
    mape,
    rmse,
    seasonal_naive,
    wape,
)

# ---------- mape: basic correctness ----------


def test_mape_basic_correctness():
    y_true = [100, 200, 300]
    y_pred = [110, 190, 300]
    result = mape(y_true, y_pred)
    # errors: 10/100=10%, 10/200=5%, 0/300=0% -> mean = 5%
    assert math.isclose(result.value, 5.0, rel_tol=1e-6)
    assert result.excluded_zero_actual_count == 0
    assert result.total_count == 3


def test_mape_perfect_prediction_is_zero():
    y_true = [50, 60, 70]
    y_pred = [50, 60, 70]
    result = mape(y_true, y_pred)
    assert result.value == 0.0


# ---------- mape: zero-actual handling (the critical edge case) ----------


def test_mape_excludes_zero_actual_rows_by_default():
    y_true = [0, 100, 200]
    y_pred = [5, 110, 190]
    result = mape(y_true, y_pred, zero_handling="exclude")
    # only rows with y_true=100 and y_true=200 should count
    # errors: 10/100=10%, 10/200=5% -> mean = 7.5%
    assert math.isclose(result.value, 7.5, rel_tol=1e-6)
    assert result.excluded_zero_actual_count == 1
    assert result.total_count == 3
    assert math.isclose(result.excluded_fraction, 1 / 3, rel_tol=1e-6)


def test_mape_all_zero_actuals_returns_nan_not_crash():
    y_true = [0, 0, 0]
    y_pred = [5, 10, 15]
    result = mape(y_true, y_pred, zero_handling="exclude")
    assert math.isnan(result.value)
    assert result.excluded_zero_actual_count == 3


def test_mape_epsilon_strategy_does_not_exclude_rows():
    y_true = [0, 100]
    y_pred = [5, 110]
    result = mape(y_true, y_pred, zero_handling="epsilon", epsilon=1e-9)
    assert result.excluded_zero_actual_count == 0
    assert result.total_count == 2
    # the zero-actual row dominates the mean massively due to tiny epsilon
    assert result.value > 1000  # sanity check it's huge, not silently ignored


def test_mape_invalid_zero_handling_raises():
    with pytest.raises(ValueError):
        mape([1, 2], [1, 2], zero_handling="not_a_real_strategy")


# ---------- mape: shape validation ----------


def test_mape_raises_on_mismatched_shapes():
    with pytest.raises(ValueError):
        mape([1, 2, 3], [1, 2])


# ---------- mape: float conversion ----------


def test_mape_result_supports_float_conversion():
    result = MapeResult(value=12.5, excluded_zero_actual_count=0, total_count=10)
    assert float(result) == 12.5


def test_mape_result_excluded_fraction_zero_total_count():
    result = MapeResult(value=0.0, excluded_zero_actual_count=0, total_count=0)
    assert result.excluded_fraction == 0.0


# ---------- wape ----------


def test_wape_basic_correctness():
    y_true = [100, 200, 300]
    y_pred = [110, 190, 300]
    # sum|err| = 10+10+0 = 20, sum|actual| = 600 -> 20/600 = 3.333...%
    result = wape(y_true, y_pred)
    assert math.isclose(result, 20 / 600 * 100, rel_tol=1e-6)


def test_wape_handles_zero_actual_rows_without_crashing():
    y_true = [0, 100, 200]
    y_pred = [5, 110, 190]
    result = wape(y_true, y_pred)
    # sum|err| = 5+10+10=25, sum|actual|=300 -> 25/300*100
    assert math.isclose(result, 25 / 300 * 100, rel_tol=1e-6)


def test_wape_all_zero_actuals_returns_nan():
    result = wape([0, 0], [1, 1])
    assert math.isnan(result)


def test_wape_raises_on_mismatched_shapes():
    with pytest.raises(ValueError):
        wape([1, 2, 3], [1, 2])


# ---------- rmse ----------


def test_rmse_basic_correctness():
    y_true = [3, -0.5, 2, 7]
    y_pred = [2.5, 0.0, 2, 8]
    expected = math.sqrt(np.mean([(3 - 2.5) ** 2, (-0.5 - 0) ** 2, 0, (7 - 8) ** 2]))
    assert math.isclose(rmse(y_true, y_pred), expected, rel_tol=1e-6)


def test_rmse_perfect_prediction_is_zero():
    assert rmse([1, 2, 3], [1, 2, 3]) == 0.0


def test_rmse_raises_on_mismatched_shapes():
    with pytest.raises(ValueError):
        rmse([1, 2, 3], [1, 2])


# ---------- mae ----------


def test_mae_basic_correctness():
    y_true = [3, -0.5, 2, 7]
    y_pred = [2.5, 0.0, 2, 8]
    expected = np.mean([0.5, 0.5, 0, 1])
    assert math.isclose(mae(y_true, y_pred), expected, rel_tol=1e-6)


def test_mae_raises_on_mismatched_shapes():
    with pytest.raises(ValueError):
        mae([1, 2, 3], [1, 2])


# ---------- seasonal_naive ----------


def test_seasonal_naive_repeats_the_same_weekday_last_week():
    dates = pd.date_range("2024-01-01", periods=14)  # Mon 1 Jan .. Sun 14 Jan
    future = pd.date_range("2024-01-15", periods=9)
    pred = seasonal_naive(dates, np.arange(14.0), future, 7)
    assert list(pred) == [7, 8, 9, 10, 11, 12, 13, 7, 8]


def test_seasonal_naive_matches_by_date_on_weekday_only_data():
    # Mon-Fri only: "7 rows back" would be a different weekday.
    dates = pd.bdate_range("2024-01-01", periods=10)  # two weeks, Mon..Fri
    values = np.arange(10.0)
    future = pd.bdate_range("2024-01-15", periods=5)
    assert list(seasonal_naive(dates, values, future, 7)) == [5, 6, 7, 8, 9]


def test_seasonal_naive_repeats_the_same_week_last_year():
    dates = pd.date_range("2023-01-02", periods=60, freq="W-MON")
    future = pd.date_range(dates[-1] + pd.Timedelta(days=7), periods=3, freq="W-MON")
    pred = seasonal_naive(dates, np.arange(60.0), future, 364)
    assert list(pred) == [8, 9, 10]  # 52 weeks before each future week


def test_seasonal_naive_falls_back_to_the_last_value_without_a_season_of_history():
    dates = pd.date_range("2024-01-01", periods=2)
    future = pd.date_range("2024-01-03", periods=3)
    assert list(seasonal_naive(dates, [4.0, 5.0], future, 7)) == [5, 5, 5]


def test_seasonal_naive_refuses_an_empty_history():
    with pytest.raises(ValueError):
        seasonal_naive([], [], pd.date_range("2024-01-01", periods=3), 7)


# ---------- error_correlation ----------


def _eval_result(errors):
    ds = pd.date_range("2024-01-01", periods=len(errors))
    y = np.full(len(errors), 100.0)
    return SimpleNamespace(
        test_predictions=pd.DataFrame({"ds": ds, "y": y, "yhat": y - np.asarray(errors)})
    )


def test_error_correlation_compares_signed_errors():
    corr = error_correlation(
        {
            "a": _eval_result([1, -2, 3, -4]),
            "b": _eval_result([2, -4, 6, -8]),  # the same mistakes, larger
            "c": _eval_result([-1, 2, -3, 4]),  # the opposite mistakes
        }
    )
    assert list(corr.columns) == ["a", "b", "c"]
    assert math.isclose(corr.loc["a", "b"], 1.0)
    assert math.isclose(corr.loc["a", "c"], -1.0)


def test_error_correlation_is_empty_for_a_single_model():
    assert error_correlation({"a": _eval_result([1, 2, 3])}).empty
