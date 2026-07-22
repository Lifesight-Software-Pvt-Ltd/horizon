import warnings
import numpy as np
import pandas as pd
import pytest

from horizon_forecaster.data_adapter.sarimax_adapter import SARIMAXAdapter
from horizon_forecaster.exceptions import InvalidConfigError


def make_raw(n=40, freq="D"):
    rng = np.random.default_rng(0)
    dates = pd.date_range("2022-01-01", periods=n, freq=freq)
    return pd.DataFrame({
        "date": dates,
        "orders": np.arange(n) + 50.0,
        "spend": np.abs(rng.normal(100, 20, n)) + 1.0,  # spend-like: positive, continuous
        "promo": rng.integers(0, 2, n),                 # binary flag
    })


def test_requires_valid_frequency():
    with pytest.raises(InvalidConfigError):
        SARIMAXAdapter("date", "orders", frequency="monthly")


def test_transform_train_emits_ds_y_and_independents_only():
    """
    Annual Fourier used to be generated here; it now lives in the wrapper so
    the grid search can choose its order. The adapter must emit nothing but
    ds, y and the (transformed) independent columns.
    """
    adapter = SARIMAXAdapter("date", "orders", ["spend", "promo"], "daily")
    out = adapter.transform_train(make_raw(220))
    assert list(out.columns) == ["ds", "y", "spend", "promo"]
    assert not any(c.startswith("fourier") for c in out.columns)
    assert adapter._exog_columns == ["spend", "promo"]


def test_auto_log_detects_spend_not_flag():
    adapter = SARIMAXAdapter("date", "orders", ["spend", "promo"], "daily")
    adapter.transform_train(make_raw())
    assert "spend" in adapter._log_columns
    assert "promo" not in adapter._log_columns


def test_independents_are_standardized_on_train():
    adapter = SARIMAXAdapter("date", "orders", ["spend"], "daily")
    out = adapter.transform_train(make_raw())
    assert abs(out["spend"].mean()) < 1e-9
    assert abs(out["spend"].std(ddof=0) - 1.0) < 1e-6


def test_transform_future_reuses_train_stats_and_shape():
    adapter = SARIMAXAdapter("date", "orders", ["spend", "promo"], "daily")
    adapter.transform_train(make_raw(220))
    fut_dates = pd.Series(pd.date_range("2022-09-01", periods=5, freq="D"))
    fut_ind = pd.DataFrame({"spend": [100.0] * 5, "promo": [0, 1, 0, 1, 0]})
    out = adapter.transform_future(fut_dates, fut_ind)
    assert list(out.columns) == ["ds", "spend", "promo"]
    assert len(out) == 5
    # constant future spend -> constant standardized value (train stats reused)
    assert out["spend"].nunique() == 1


def test_transform_future_clamps_out_of_range_exog():
    adapter = SARIMAXAdapter("date", "orders", ["spend"], "daily")
    adapter.transform_train(make_raw(60))  # train spend ~ normal(100, 20)
    fut_dates = pd.Series(pd.date_range("2022-03-05", periods=3, freq="D"))
    fut = pd.DataFrame({"spend": [1e6, 1e6, 1e6]})  # wildly out of training range
    out = adapter.transform_future(fut_dates, fut)
    assert (out["spend"] <= adapter._std_max["spend"] + 1e-9).all()
    assert abs(out["spend"].iloc[0] - adapter._std_max["spend"]) < 1e-9


def test_transform_future_before_train_raises():
    adapter = SARIMAXAdapter("date", "orders", ["spend"], "daily")
    with pytest.raises(InvalidConfigError):
        adapter.transform_future(
            pd.Series(pd.date_range("2022-01-01", periods=3, freq="D")),
            pd.DataFrame({"spend": [1, 2, 3]}),
        )


def test_transform_future_missing_independent_df_raises():
    adapter = SARIMAXAdapter("date", "orders", ["spend"], "daily")
    adapter.transform_train(make_raw())
    with pytest.raises(InvalidConfigError):
        adapter.transform_future(pd.Series(pd.date_range("2022-02-10", periods=3, freq="D")), None)


def test_transform_train_missing_column_raises():
    adapter = SARIMAXAdapter("date", "orders", ["spend"], "daily")
    with pytest.raises(InvalidConfigError):
        adapter.transform_train(make_raw().drop(columns=["spend"]))


def _two_years_weekly(n=104):
    return pd.date_range("2022-01-02", periods=n, freq="W")


def test_annual_collinearity_warning_fires_for_a_seasonal_driver():
    """
    A column that tracks the annual cycle (like a seasonal stockout weighting)
    explains the same variance as the wrapper's annual Fourier terms, so the
    user is warned that the higher-Fourier grid candidates may destabilise.
    """
    dates = _two_years_weekly()
    t = np.arange(len(dates))
    seasonal = 5 + 3 * np.sin(2 * np.pi * t / 52.0)  # non-negative, non-binary -> log1p'd
    raw = pd.DataFrame({"date": dates, "orders": 100 + t + seasonal, "oos": seasonal})

    adapter = SARIMAXAdapter("date", "orders", ["oos"], "weekly")
    with pytest.warns(UserWarning, match=r"track the annual seasonal cycle.*oos"):
        adapter.transform_train(raw)


def test_annual_collinearity_warning_silent_for_a_nonseasonal_driver():
    """
    A spend-like column dominated by trend + noise, not the annual cycle,
    must NOT trip the warning -- otherwise it would cry wolf on ordinary
    spend inputs. (A plain trend is not sinusoidal, so its annual R² is low.)
    """
    dates = _two_years_weekly()
    t = np.arange(len(dates))
    rng = np.random.default_rng(0)
    spend = 200 + 3.0 * t + rng.normal(0, 40, len(dates))  # trend + noise, no annual term
    raw = pd.DataFrame({"date": dates, "orders": 100 + t, "spend": spend})

    adapter = SARIMAXAdapter("date", "orders", ["spend"], "weekly")
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any collinearity warning would raise
        adapter.transform_train(raw)  # must not raise


def test_annual_collinearity_r2_high_for_pure_sine_low_for_trend():
    """
    The scorer itself: ~1.0 on a clean annual sine, and for a linear ramp
    comfortably below the warning threshold. (A ramp isn't sinusoidal, but
    over a non-integer number of years it retains a small projection onto the
    annual basis -- what matters is that it stays well under the threshold, so
    ordinary trending spend never trips the warning.)
    """
    from horizon_forecaster.data_adapter.sarimax_adapter import (
        _ANNUAL_COLLINEARITY_R2_THRESHOLD,
    )

    adapter = SARIMAXAdapter("date", "orders", ["x"], "weekly")
    dates = _two_years_weekly()
    t = np.arange(len(dates))
    sine = np.sin(2 * np.pi * t / 52.0)
    ramp = t.astype(float)
    assert adapter._annual_collinearity_r2(pd.Series(dates), sine) > 0.95
    assert (
        adapter._annual_collinearity_r2(pd.Series(dates), ramp)
        < _ANNUAL_COLLINEARITY_R2_THRESHOLD
    )


def test_annual_collinearity_r2_is_zero_for_constant_column():
    adapter = SARIMAXAdapter("date", "orders", ["x"], "weekly")
    dates = _two_years_weekly()
    assert adapter._annual_collinearity_r2(pd.Series(dates), np.full(len(dates), 7.0)) == 0.0


def test_works_with_no_independent_columns():
    adapter = SARIMAXAdapter("date", "orders", [], "weekly")
    out = adapter.transform_train(make_raw(30, freq="W"))
    assert list(out.columns) == ["ds", "y"]
    assert adapter._exog_columns == []
    future = adapter.transform_future(pd.Series(pd.date_range("2022-09-01", periods=4, freq="W")))
    assert list(future.columns) == ["ds"]
    assert len(future) == 4
