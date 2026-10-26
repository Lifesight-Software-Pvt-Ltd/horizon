import pandas as pd
import pytest

from horizon_forecaster.exceptions import InvalidConfigError
from horizon_forecaster.data_adapter.prophet_adapter import ProphetAdapter
from horizon_forecaster.data_adapter.catboost_adapter import CatBoostAdapter

# =====================================================================
# ProphetAdapter
# =====================================================================


def make_raw_df():
    return pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=5, freq="D"),
            "orders": [10, 20, 15, 25, 30],
            "spend": [100, 150, 120, 180, 200],
            "promo_flag": [0, 1, 0, 1, 0],
        }
    )


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
    raw_df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-03", "2026-01-01", "2026-01-02"]),
            "orders": [30, 10, 20],
        }
    )
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
#
# A trend index (time_idx) + cyclical (sin/cos) seasonality, with no
# lag/rolling-window features (a real accuracy finding) — see
# catboost_adapter.py's module docstring for the full story (a frozen
# lag/rolling snapshot produced 60-90% MAPE on a real dataset's sudden
# demand-shift period; a trend+cyclical feature
# set doesn't have this failure mode since nothing needs to be frozen).
# =====================================================================


def make_catboost_raw_df(n=30, with_spend=False, freq="D"):
    df = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=n, freq=freq),
            "orders": [100 + i * 2 for i in range(n)],
        }
    )
    if with_spend:
        df["spend"] = [50 + i for i in range(n)]
    return df


_DAILY_EXPECTED_COLUMNS = {
    "ds",
    "y",
    "time_idx",
    "week_of_year",
    "month",
    "quarter",
    "year",
    "yearly_sin",
    "yearly_cos",
    "half_yearly_sin",
    "half_yearly_cos",
    "day_of_week",
    "day_of_year",
    "weekly_sin",
    "weekly_cos",
    "monthly_sin",
    "monthly_cos",
}
_WEEKLY_EXPECTED_COLUMNS = {
    "ds",
    "y",
    "time_idx",
    "week_of_year",
    "month",
    "quarter",
    "year",
    "yearly_sin",
    "yearly_cos",
    "half_yearly_sin",
    "half_yearly_cos",
}

# ---------- constructor / frequency validation ----------


def test_catboost_adapter_requires_valid_frequency():
    with pytest.raises(InvalidConfigError):
        CatBoostAdapter(date_column="date", target_column="orders", frequency="monthly")


def test_catboost_adapter_requires_frequency_to_be_provided():
    with pytest.raises(InvalidConfigError):
        CatBoostAdapter(date_column="date", target_column="orders")  # frequency=None default


# ---------- transform_train ----------


def test_catboost_adapter_transform_train_daily_produces_expected_columns():
    df = make_catboost_raw_df(n=30)
    adapter = CatBoostAdapter(date_column="date", target_column="orders", frequency="daily")
    result = adapter.transform_train(df)
    assert set(result.columns) == _DAILY_EXPECTED_COLUMNS


def test_catboost_adapter_transform_train_weekly_excludes_daily_only_features():
    df = make_catboost_raw_df(n=30, freq="7D")
    adapter = CatBoostAdapter(date_column="date", target_column="orders", frequency="weekly")
    result = adapter.transform_train(df)
    assert set(result.columns) == _WEEKLY_EXPECTED_COLUMNS
    assert "day_of_week" not in result.columns


def test_catboost_adapter_transform_train_drops_no_rows():
    """
    None of these features need history to warm up (unlike lag/rolling
    features) -- every row of the input should survive.
    """
    df = make_catboost_raw_df(n=30)
    adapter = CatBoostAdapter(date_column="date", target_column="orders", frequency="daily")
    result = adapter.transform_train(df)
    assert len(result) == 30
    assert result.isna().sum().sum() == 0


def test_catboost_adapter_transform_train_time_idx_starts_at_zero():
    df = make_catboost_raw_df(n=30)
    adapter = CatBoostAdapter(date_column="date", target_column="orders", frequency="daily")
    result = adapter.transform_train(df)
    assert result["time_idx"].tolist() == list(range(30))


def test_catboost_adapter_transform_train_time_idx_weekly_measured_in_weeks():
    df = make_catboost_raw_df(n=10, freq="7D")
    adapter = CatBoostAdapter(date_column="date", target_column="orders", frequency="weekly")
    result = adapter.transform_train(df)
    assert result["time_idx"].tolist() == [float(i) for i in range(10)]


def test_catboost_adapter_transform_train_calendar_features_are_correct():
    df = make_catboost_raw_df(n=30)
    adapter = CatBoostAdapter(date_column="date", target_column="orders", frequency="daily")
    result = adapter.transform_train(df)
    first_ds = result.iloc[0]["ds"]
    assert result.iloc[0]["day_of_week"] == first_ds.dayofweek
    assert result.iloc[0]["month"] == first_ds.month
    assert result.iloc[0]["year"] == first_ds.year


def test_catboost_adapter_transform_train_cyclical_features_are_bounded():
    """
    sin/cos features should always fall within [-1, 1] and never be
    NaN -- a basic sanity check that the cyclical encoding math is
    correct across a full year of dates, not just a handful.
    """
    df = make_catboost_raw_df(n=400)
    adapter = CatBoostAdapter(date_column="date", target_column="orders", frequency="daily")
    result = adapter.transform_train(df)
    cyclical_cols = [
        "yearly_sin",
        "yearly_cos",
        "half_yearly_sin",
        "half_yearly_cos",
        "weekly_sin",
        "weekly_cos",
        "monthly_sin",
        "monthly_cos",
    ]
    for col in cyclical_cols:
        assert result[col].between(-1, 1).all(), col
        assert result[col].notna().all(), col


def test_catboost_adapter_transform_train_includes_independent_columns():
    df = make_catboost_raw_df(n=30, with_spend=True)
    adapter = CatBoostAdapter(
        date_column="date", target_column="orders", independent_columns=["spend"], frequency="daily"
    )
    result = adapter.transform_train(df)
    assert "spend" in result.columns


def test_catboost_adapter_transform_train_works_with_no_independent_columns():
    df = make_catboost_raw_df(n=30)
    adapter = CatBoostAdapter(date_column="date", target_column="orders", frequency="daily")
    result = adapter.transform_train(df)
    assert "spend" not in result.columns


def test_catboost_adapter_transform_train_raises_on_declared_but_missing_independent_column():
    df = make_catboost_raw_df(n=30)  # no "spend" column
    adapter = CatBoostAdapter(
        date_column="date", target_column="orders", independent_columns=["spend"], frequency="daily"
    )
    with pytest.raises(InvalidConfigError) as exc_info:
        adapter.transform_train(df)
    assert "spend" in str(exc_info.value)


def test_catboost_adapter_transform_train_raises_on_missing_date_column():
    df = pd.DataFrame({"orders": list(range(30))})
    adapter = CatBoostAdapter(date_column="date", target_column="orders", frequency="daily")
    with pytest.raises(InvalidConfigError):
        adapter.transform_train(df)


def test_catboost_adapter_transform_train_raises_on_missing_target_column():
    df = pd.DataFrame({"date": pd.date_range("2026-01-01", periods=30)})
    adapter = CatBoostAdapter(date_column="date", target_column="orders", frequency="daily")
    with pytest.raises(InvalidConfigError):
        adapter.transform_train(df)


def test_catboost_adapter_transform_train_sorts_by_date():
    n = 30
    shuffled_dates = list(pd.date_range("2026-01-01", periods=n, freq="D"))
    df = pd.DataFrame(
        {
            "date": list(reversed(shuffled_dates)),
            "orders": list(reversed([100 + i * 2 for i in range(n)])),
        }
    )
    adapter = CatBoostAdapter(date_column="date", target_column="orders", frequency="daily")
    result = adapter.transform_train(df)
    assert result["ds"].is_monotonic_increasing
    assert result["time_idx"].tolist() == list(range(n))


# ---------- transform_future ----------


def test_catboost_adapter_transform_future_raises_if_called_before_transform_train():
    adapter = CatBoostAdapter(date_column="date", target_column="orders", frequency="daily")
    future_dates = pd.Series(pd.date_range("2026-03-01", periods=3))
    with pytest.raises(InvalidConfigError):
        adapter.transform_future(future_dates)


def test_catboost_adapter_transform_future_basic_shape():
    df = make_catboost_raw_df(n=30)
    adapter = CatBoostAdapter(date_column="date", target_column="orders", frequency="daily")
    adapter.transform_train(df)

    future_dates = pd.Series(
        pd.date_range(df["date"].max() + pd.Timedelta(days=1), periods=5, freq="D")
    )
    result = adapter.transform_future(future_dates)
    assert len(result) == 5
    assert set(result.columns) == (_DAILY_EXPECTED_COLUMNS - {"y"})
    assert result.isna().sum().sum() == 0


def test_catboost_adapter_transform_future_time_idx_continues_seamlessly_from_training():
    """
    The core property of the trend-index design: time_idx must continue
    from exactly where training left off, and must vary per row -- unlike
    lag/rolling features, which are frozen (identical) across the whole
    forecast horizon.
    """
    df = make_catboost_raw_df(n=30)
    adapter = CatBoostAdapter(date_column="date", target_column="orders", frequency="daily")
    train_result = adapter.transform_train(df)

    future_dates = pd.Series(
        pd.date_range(df["date"].max() + pd.Timedelta(days=1), periods=5, freq="D")
    )
    result = adapter.transform_future(future_dates)
    assert result["time_idx"].iloc[0] == train_result["time_idx"].iloc[-1] + 1
    assert result["time_idx"].tolist() == [30, 31, 32, 33, 34]
    assert result["time_idx"].nunique() == 5


def test_catboost_adapter_transform_future_calendar_and_cyclical_features_vary_per_row():
    df = make_catboost_raw_df(n=30)
    adapter = CatBoostAdapter(date_column="date", target_column="orders", frequency="daily")
    adapter.transform_train(df)

    future_dates = pd.Series(
        pd.date_range(df["date"].max() + pd.Timedelta(days=1), periods=10, freq="D")
    )
    result = adapter.transform_future(future_dates)
    assert result["day_of_week"].nunique() > 1
    assert result["yearly_sin"].nunique() > 1


def test_catboost_adapter_transform_future_handles_a_non_contiguous_horizon():
    """
    Because none of these features depend on target history, a forecast
    horizon that isn't immediately contiguous with training (e.g. a
    gap) still produces correct, well-defined time_idx values -- there's
    no "must be the very next period" constraint, which lag/rolling
    features would impose (they need real, unbroken target history right
    up to the forecast start).
    """
    df = make_catboost_raw_df(n=30)
    adapter = CatBoostAdapter(date_column="date", target_column="orders", frequency="daily")
    adapter.transform_train(df)

    far_future_dates = pd.Series(
        pd.date_range(df["date"].max() + pd.Timedelta(days=10), periods=3, freq="D")
    )
    result = adapter.transform_future(far_future_dates)
    assert result["time_idx"].tolist() == [39, 40, 41]


def test_catboost_adapter_transform_future_includes_provided_independent_data():
    df = make_catboost_raw_df(n=30, with_spend=True)
    adapter = CatBoostAdapter(
        date_column="date", target_column="orders", independent_columns=["spend"], frequency="daily"
    )
    adapter.transform_train(df)

    future_dates = pd.Series(
        pd.date_range(df["date"].max() + pd.Timedelta(days=1), periods=3, freq="D")
    )
    future_spend = pd.DataFrame({"spend": [500, 510, 520]})
    result = adapter.transform_future(future_dates, future_spend)
    assert result["spend"].tolist() == [500, 510, 520]


def test_catboost_adapter_transform_future_raises_when_independent_declared_but_df_is_none():
    df = make_catboost_raw_df(n=30, with_spend=True)
    adapter = CatBoostAdapter(
        date_column="date", target_column="orders", independent_columns=["spend"], frequency="daily"
    )
    adapter.transform_train(df)

    future_dates = pd.Series(
        pd.date_range(df["date"].max() + pd.Timedelta(days=1), periods=3, freq="D")
    )
    with pytest.raises(InvalidConfigError):
        adapter.transform_future(future_dates, future_independent_df=None)


def test_catboost_adapter_transform_future_raises_on_missing_independent_column_in_df():
    df = make_catboost_raw_df(n=30, with_spend=True)
    adapter = CatBoostAdapter(
        date_column="date", target_column="orders", independent_columns=["spend"], frequency="daily"
    )
    adapter.transform_train(df)

    future_dates = pd.Series(
        pd.date_range(df["date"].max() + pd.Timedelta(days=1), periods=3, freq="D")
    )
    future_df_missing = pd.DataFrame({"clicks": [1, 2, 3]})
    with pytest.raises(InvalidConfigError) as exc_info:
        adapter.transform_future(future_dates, future_df_missing)
    assert "spend" in str(exc_info.value)


def test_catboost_adapter_transform_future_with_no_independent_columns_declared():
    df = make_catboost_raw_df(n=30)
    adapter = CatBoostAdapter(date_column="date", target_column="orders", frequency="daily")
    adapter.transform_train(df)

    future_dates = pd.Series(
        pd.date_range(df["date"].max() + pd.Timedelta(days=1), periods=2, freq="D")
    )
    result = adapter.transform_future(future_dates, future_independent_df=None)
    assert "spend" not in result.columns
    assert len(result) == 2
