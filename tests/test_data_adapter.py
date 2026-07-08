import pandas as pd
import pytest

from horizon_forecaster.exceptions import InvalidConfigError
from horizon_forecaster.data_adapter.prophet_adapter import ProphetAdapter

# =====================================================================
# ProphetAdapter
# =====================================================================

def make_raw_df():
    return pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=5, freq="D"),
        "orders": [10, 20, 15, 25, 30],
        "spend": [100, 150, 120, 180, 200],
        "promo_flag": [0, 1, 0, 1, 0],
    })

def test_prophet_adapter_transform_train_renames_date_and_target():
    raw_df = make_raw_df()
    adapter = ProphetAdapter(
        date_column="date", target_column="orders", independent_columns=["spend"]
    )
    result = adapter.transform_train(raw_df)

    assert list(result.columns) == ["ds", "y", "spend"]
    assert result["y"].tolist() == [10, 20, 15, 25, 30]
    assert pd.api.types.is_datetime64_any_dtype(result["ds"])

def test_prophet_adapter_transform_train_includes_multiple_independent_columns():
    raw_df = make_raw_df()
    adapter = ProphetAdapter(
        date_column="date",
        target_column="orders",
        independent_columns=["spend", "promo_flag"],
    )
    result = adapter.transform_train(raw_df)
    assert "spend" in result.columns
    assert "promo_flag" in result.columns

def test_prophet_adapter_transform_train_works_with_no_independent_columns():
    raw_df = make_raw_df()
    adapter = ProphetAdapter(date_column="date", target_column="orders")
    result = adapter.transform_train(raw_df)
    assert list(result.columns) == ["ds", "y"]

def test_prophet_adapter_transform_train_raises_on_declared_but_missing_independent_column():
    raw_df = make_raw_df()
    adapter = ProphetAdapter(
        date_column="date",
        target_column="orders",
        independent_columns=["spend", "clicks"],  # "clicks" not in raw_df
    )
    with pytest.raises(InvalidConfigError) as exc_info:
        adapter.transform_train(raw_df)
    assert "clicks" in str(exc_info.value)

def test_prophet_adapter_transform_train_sorts_by_date():
    raw_df = pd.DataFrame({
        "date": pd.to_datetime(["2026-01-03", "2026-01-01", "2026-01-02"]),
        "orders": [30, 10, 20],
    })
    adapter = ProphetAdapter(date_column="date", target_column="orders")
    result = adapter.transform_train(raw_df)
    assert result["ds"].tolist() == list(pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]))
    assert result["y"].tolist() == [10, 20, 30]

def test_prophet_adapter_transform_train_raises_on_missing_date_column():
    raw_df = pd.DataFrame({"orders": [1, 2, 3]})
    adapter = ProphetAdapter(date_column="date", target_column="orders")
    with pytest.raises(InvalidConfigError):
        adapter.transform_train(raw_df)

def test_prophet_adapter_transform_train_raises_on_missing_target_column():
    raw_df = pd.DataFrame({"date": pd.date_range("2026-01-01", periods=3)})
    adapter = ProphetAdapter(date_column="date", target_column="orders")
    with pytest.raises(InvalidConfigError):
        adapter.transform_train(raw_df)

def test_prophet_adapter_transform_future_basic():
    future_dates = pd.Series(pd.date_range("2026-02-01", periods=3, freq="D"))
    adapter = ProphetAdapter(date_column="date", target_column="orders")
    result = adapter.transform_future(future_dates)
    assert list(result.columns) == ["ds"]
    assert len(result) == 3

def test_prophet_adapter_transform_future_includes_provided_independent_data():
    future_dates = pd.Series(pd.date_range("2026-02-01", periods=3, freq="D"))
    future_independent_df = pd.DataFrame({"spend": [100, 110, 120]})
    adapter = ProphetAdapter(
        date_column="date", target_column="orders", independent_columns=["spend"]
    )
    result = adapter.transform_future(future_dates, future_independent_df)
    assert "spend" in result.columns
    assert result["spend"].tolist() == [100, 110, 120]

def test_prophet_adapter_transform_future_raises_when_independent_declared_but_df_is_none():
    """
    By the time transform_future() is called, forecaster.py must already
    have resolved future independent-variable values (either real user
    data or a fallback strategy). A None arriving here with
    independent_columns declared indicates forecaster.py failed to do
    that — a real bug worth surfacing loudly, not masking.
    """
    future_dates = pd.Series(pd.date_range("2026-02-01", periods=3, freq="D"))
    adapter = ProphetAdapter(
        date_column="date", target_column="orders", independent_columns=["spend"]
    )
    with pytest.raises(InvalidConfigError):
        adapter.transform_future(future_dates, future_independent_df=None)

def test_prophet_adapter_transform_future_raises_on_missing_independent_column_in_df():
    future_dates = pd.Series(pd.date_range("2026-02-01", periods=3, freq="D"))
    future_independent_df = pd.DataFrame({"clicks": [1, 2, 3]})  # "spend" missing
    adapter = ProphetAdapter(
        date_column="date", target_column="orders", independent_columns=["spend"]
    )
    with pytest.raises(InvalidConfigError) as exc_info:
        adapter.transform_future(future_dates, future_independent_df)
    assert "spend" in str(exc_info.value)

def test_prophet_adapter_transform_future_with_no_independent_columns_declared():
    """
    If the adapter has no independent_columns declared at all, passing
    future_independent_df=None is perfectly valid — there's nothing to
    resolve.
    """
    future_dates = pd.Series(pd.date_range("2026-02-01", periods=2, freq="D"))
    adapter = ProphetAdapter(date_column="date", target_column="orders")
    result = adapter.transform_future(future_dates, future_independent_df=None)
    assert list(result.columns) == ["ds"]