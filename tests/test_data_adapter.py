import pandas as pd
import pytest

from horizon_forecaster.exceptions import InvalidConfigError
from horizon_forecaster.data_adapter.prophet_adapter import ProphetAdapter
from horizon_forecaster.data_adapter.catboost_adapter import CatBoostAdapter

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

# =====================================================================
# CatBoostAdapter
# =====================================================================

def make_catboost_raw_df(n=30, with_spend=False):
    df = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=n, freq="D"),
        "orders": [100 + i * 2 for i in range(n)],
    })
    if with_spend:
        df["spend"] = [50 + i for i in range(n)]
    return df

# ---------- transform_train ----------

def test_catboost_adapter_transform_train_drops_warmup_rows_and_has_no_nans():
    """
    The first max(lags, rolling_windows) rows can't have every lag/rolling
    feature computed (not enough history yet) and are deliberately
    dropped, rather than left as NaN or silently imputed.
    """
    df = make_catboost_raw_df(n=30)
    adapter = CatBoostAdapter(date_column="date", target_column="orders")
    result = adapter.transform_train(df)

    expected_warmup = max(max(adapter.lags), max(adapter.rolling_windows))
    assert adapter.n_rows_dropped_for_lag_warmup == expected_warmup
    assert len(result) == 30 - expected_warmup
    assert result.isna().sum().sum() == 0

def test_catboost_adapter_transform_train_lag_values_are_correct():
    df = make_catboost_raw_df(n=30)
    orders = df["orders"].tolist()
    adapter = CatBoostAdapter(date_column="date", target_column="orders")
    result = adapter.transform_train(df)

    warmup = adapter.n_rows_dropped_for_lag_warmup
    # First surviving row's lag_1/lag_7 should equal the actual orders
    # value 1 / 7 periods before it in the original (pre-drop) series.
    assert result.iloc[0]["lag_1"] == orders[warmup - 1]
    assert result.iloc[0]["lag_7"] == orders[warmup - 7]

def test_catboost_adapter_transform_train_rolling_mean_and_std_are_correct():
    df = make_catboost_raw_df(n=30)
    y = df["orders"]
    adapter = CatBoostAdapter(date_column="date", target_column="orders")
    result = adapter.transform_train(df)

    warmup = adapter.n_rows_dropped_for_lag_warmup
    expected_mean = y.shift(1).rolling(7).mean().iloc[warmup]
    expected_std = y.shift(1).rolling(7).std().iloc[warmup]
    assert result.iloc[0]["rolling_mean_7"] == pytest.approx(expected_mean)
    assert result.iloc[0]["rolling_std_7"] == pytest.approx(expected_std)

def test_catboost_adapter_transform_train_rolling_features_exclude_current_row():
    """
    shift(1) before rolling: a row's own target value must never leak
    into its own rolling statistics.
    """
    df = make_catboost_raw_df(n=30)
    adapter = CatBoostAdapter(date_column="date", target_column="orders", lags=(1,), rolling_windows=(1,))
    result = adapter.transform_train(df)
    # rolling_mean_1 (window=1, shifted by 1) should equal lag_1 exactly,
    # since a 1-period rolling mean of a single strictly-prior value IS
    # that value.
    assert (result["rolling_mean_1"] == result["lag_1"]).all()

def test_catboost_adapter_transform_train_calendar_features_present_and_correct():
    df = make_catboost_raw_df(n=30)
    adapter = CatBoostAdapter(date_column="date", target_column="orders")
    result = adapter.transform_train(df)

    for col in adapter._CALENDAR_FEATURE_NAMES:
        assert col in result.columns
    first_ds = result.iloc[0]["ds"]
    assert result.iloc[0]["day_of_week"] == first_ds.dayofweek
    assert result.iloc[0]["month"] == first_ds.month
    assert result.iloc[0]["is_weekend"] == int(first_ds.dayofweek >= 5)

def test_catboost_adapter_transform_train_includes_independent_columns():
    df = make_catboost_raw_df(n=30, with_spend=True)
    adapter = CatBoostAdapter(date_column="date", target_column="orders", independent_columns=["spend"])
    result = adapter.transform_train(df)
    assert "spend" in result.columns

def test_catboost_adapter_transform_train_works_with_no_independent_columns():
    df = make_catboost_raw_df(n=30)
    adapter = CatBoostAdapter(date_column="date", target_column="orders")
    result = adapter.transform_train(df)
    assert "spend" not in result.columns

def test_catboost_adapter_transform_train_raises_on_declared_but_missing_independent_column():
    df = make_catboost_raw_df(n=30)  # no "spend" column
    adapter = CatBoostAdapter(date_column="date", target_column="orders", independent_columns=["spend"])
    with pytest.raises(InvalidConfigError) as exc_info:
        adapter.transform_train(df)
    assert "spend" in str(exc_info.value)

def test_catboost_adapter_transform_train_raises_on_missing_date_column():
    df = pd.DataFrame({"orders": list(range(30))})
    adapter = CatBoostAdapter(date_column="date", target_column="orders")
    with pytest.raises(InvalidConfigError):
        adapter.transform_train(df)

def test_catboost_adapter_transform_train_raises_on_missing_target_column():
    df = pd.DataFrame({"date": pd.date_range("2026-01-01", periods=30)})
    adapter = CatBoostAdapter(date_column="date", target_column="orders")
    with pytest.raises(InvalidConfigError):
        adapter.transform_train(df)

def test_catboost_adapter_transform_train_sorts_by_date_before_computing_lags():
    n = 30
    shuffled_dates = list(pd.date_range("2026-01-01", periods=n, freq="D"))
    # reverse order on input; correctness of lag_1 after sort proves the
    # adapter sorts BEFORE computing lag/rolling features, not after.
    df = pd.DataFrame({
        "date": list(reversed(shuffled_dates)),
        "orders": list(reversed([100 + i * 2 for i in range(n)])),
    })
    adapter = CatBoostAdapter(date_column="date", target_column="orders")
    result = adapter.transform_train(df)
    assert result["ds"].is_monotonic_increasing
    warmup = adapter.n_rows_dropped_for_lag_warmup
    assert result.iloc[0]["lag_1"] == 100 + (warmup - 1) * 2

def test_catboost_adapter_transform_train_raises_when_history_shorter_than_requested_lag():
    df = make_catboost_raw_df(n=10)
    adapter = CatBoostAdapter(date_column="date", target_column="orders", lags=(1, 30))
    # transform_train itself won't raise (dropna just yields 0 rows), but
    # the shortage becomes an explicit error the moment transform_future
    # tries to anchor a lag_30 that doesn't exist in history.
    adapter.transform_train(df)
    with pytest.raises(InvalidConfigError) as exc_info:
        adapter.transform_future(pd.Series(pd.date_range("2026-02-01", periods=3)))
    assert "lag_30" in str(exc_info.value)

def test_catboost_adapter_custom_lags_and_rolling_windows_are_respected():
    df = make_catboost_raw_df(n=30)
    adapter = CatBoostAdapter(date_column="date", target_column="orders", lags=(1, 2), rolling_windows=(3,))
    result = adapter.transform_train(df)
    assert set(["lag_1", "lag_2", "rolling_mean_3", "rolling_std_3"]).issubset(result.columns)
    assert "lag_7" not in result.columns
    assert adapter.n_rows_dropped_for_lag_warmup == 3  # max(1, 2, 3)

# ---------- transform_future ----------

def test_catboost_adapter_transform_future_raises_if_called_before_transform_train():
    adapter = CatBoostAdapter(date_column="date", target_column="orders")
    future_dates = pd.Series(pd.date_range("2026-03-01", periods=3))
    with pytest.raises(InvalidConfigError):
        adapter.transform_future(future_dates)

def test_catboost_adapter_transform_future_basic_shape():
    df = make_catboost_raw_df(n=30)
    adapter = CatBoostAdapter(date_column="date", target_column="orders")
    adapter.transform_train(df)

    future_dates = pd.Series(pd.date_range(df["date"].max() + pd.Timedelta(days=1), periods=5, freq="D"))
    result = adapter.transform_future(future_dates)
    assert len(result) == 5
    assert "lag_1" in result.columns
    assert "rolling_mean_7" in result.columns
    assert result.isna().sum().sum() == 0

def test_catboost_adapter_transform_future_lag_and_rolling_features_are_frozen_across_horizon():
    """
    Direct (not recursive) multi-horizon strategy: every future row gets
    the SAME lag/rolling feature values, anchored to the end of training
    history — only calendar/independent features vary per row.
    """
    df = make_catboost_raw_df(n=30)
    adapter = CatBoostAdapter(date_column="date", target_column="orders")
    adapter.transform_train(df)

    future_dates = pd.Series(pd.date_range(df["date"].max() + pd.Timedelta(days=1), periods=5, freq="D"))
    result = adapter.transform_future(future_dates)
    assert result["lag_1"].nunique() == 1
    assert result["lag_7"].nunique() == 1
    assert result["rolling_mean_7"].nunique() == 1
    assert result["rolling_std_7"].nunique() == 1
    # but calendar features are NOT frozen - they vary per actual future date
    assert result["day_of_week"].nunique() > 1 or len(result) == 1

def test_catboost_adapter_transform_future_lag_values_match_end_of_training_history():
    df = make_catboost_raw_df(n=30)
    orders = df["orders"].tolist()
    adapter = CatBoostAdapter(date_column="date", target_column="orders")
    adapter.transform_train(df)

    future_dates = pd.Series(pd.date_range(df["date"].max() + pd.Timedelta(days=1), periods=3, freq="D"))
    result = adapter.transform_future(future_dates)
    assert result.iloc[0]["lag_1"] == orders[-1]
    assert result.iloc[0]["lag_7"] == orders[-7]

def test_catboost_adapter_transform_future_includes_provided_independent_data():
    df = make_catboost_raw_df(n=30, with_spend=True)
    adapter = CatBoostAdapter(date_column="date", target_column="orders", independent_columns=["spend"])
    adapter.transform_train(df)

    future_dates = pd.Series(pd.date_range(df["date"].max() + pd.Timedelta(days=1), periods=3, freq="D"))
    future_spend = pd.DataFrame({"spend": [500, 510, 520]})
    result = adapter.transform_future(future_dates, future_spend)
    assert result["spend"].tolist() == [500, 510, 520]

def test_catboost_adapter_transform_future_raises_when_independent_declared_but_df_is_none():
    df = make_catboost_raw_df(n=30, with_spend=True)
    adapter = CatBoostAdapter(date_column="date", target_column="orders", independent_columns=["spend"])
    adapter.transform_train(df)

    future_dates = pd.Series(pd.date_range(df["date"].max() + pd.Timedelta(days=1), periods=3, freq="D"))
    with pytest.raises(InvalidConfigError):
        adapter.transform_future(future_dates, future_independent_df=None)

def test_catboost_adapter_transform_future_raises_on_missing_independent_column_in_df():
    df = make_catboost_raw_df(n=30, with_spend=True)
    adapter = CatBoostAdapter(date_column="date", target_column="orders", independent_columns=["spend"])
    adapter.transform_train(df)

    future_dates = pd.Series(pd.date_range(df["date"].max() + pd.Timedelta(days=1), periods=3, freq="D"))
    future_df_missing = pd.DataFrame({"clicks": [1, 2, 3]})
    with pytest.raises(InvalidConfigError) as exc_info:
        adapter.transform_future(future_dates, future_df_missing)
    assert "spend" in str(exc_info.value)

def test_catboost_adapter_transform_future_with_no_independent_columns_declared():
    df = make_catboost_raw_df(n=30)
    adapter = CatBoostAdapter(date_column="date", target_column="orders")
    adapter.transform_train(df)

    future_dates = pd.Series(pd.date_range(df["date"].max() + pd.Timedelta(days=1), periods=2, freq="D"))
    result = adapter.transform_future(future_dates, future_independent_df=None)
    assert "spend" not in result.columns
    assert len(result) == 2