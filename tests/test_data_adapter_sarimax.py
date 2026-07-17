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


def _fourier_cols(df):
    return [c for c in df.columns if c.startswith("fourier_annual_")]


def test_requires_valid_frequency():
    with pytest.raises(InvalidConfigError):
        SARIMAXAdapter("date", "orders", frequency="monthly")


def test_transform_train_has_ds_y_exog_and_fourier():
    # >= half a year of daily data -> annual Fourier is included
    adapter = SARIMAXAdapter("date", "orders", ["spend", "promo"], "daily")
    out = adapter.transform_train(make_raw(220))
    assert "ds" in out.columns and "y" in out.columns
    assert "spend" in out.columns and "promo" in out.columns
    assert len(_fourier_cols(out)) == 4  # Fourier order 2 -> 4 columns


def test_short_daily_omits_annual_fourier():
    # < half a year of daily data -> annual Fourier omitted (extrapolation guard)
    adapter = SARIMAXAdapter("date", "orders", ["spend"], "daily")
    out = adapter.transform_train(make_raw(60))
    assert _fourier_cols(out) == []
    assert adapter._exog_columns == ["spend"]


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
    assert out.columns[0] == "ds"
    assert len(out) == 5
    assert len(_fourier_cols(out)) == 4
    # constant future spend -> constant standardized value (train stats reused)
    assert out["spend"].nunique() == 1


def test_transform_future_clamps_out_of_range_exog():
    adapter = SARIMAXAdapter("date", "orders", ["spend"], "daily")
    adapter.transform_train(make_raw(60))  # train spend ~ normal(100, 20)
    fut_dates = pd.Series(pd.date_range("2022-03-05", periods=3, freq="D"))
    fut = pd.DataFrame({"spend": [1e6, 1e6, 1e6]})  # wildly out of training range
    out = adapter.transform_future(fut_dates, fut)
    # clamped down to the standardised training max, not extrapolated
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


def test_weekly_fourier_and_no_independents():
    adapter = SARIMAXAdapter("date", "orders", [], "weekly")
    out = adapter.transform_train(make_raw(30, freq="W"))
    fcols = _fourier_cols(out)
    assert len(fcols) == 4
    # with no independents, exog is exactly the Fourier columns
    assert adapter._exog_columns == fcols


def test_no_independents_transform_future_ok():
    adapter = SARIMAXAdapter("date", "orders", [], "daily")
    adapter.transform_train(make_raw(220))
    out = adapter.transform_future(pd.Series(pd.date_range("2022-09-01", periods=4, freq="D")))
    assert len(out) == 4
    assert len(_fourier_cols(out)) == 4
