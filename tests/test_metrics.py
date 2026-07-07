import math

import numpy as np
import pytest

from horizon_forecaster.evaluation.metrics import mape, wape, rmse, mae, MapeResult

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