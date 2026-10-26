"""
Unit tests for the future-exogenous-data strategies
(horizon_forecaster/future_exog.py). Pure pandas/numpy -- no models.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from horizon_forecaster.exceptions import InvalidConfigError
from horizon_forecaster.future_exog import (
    FUTURE_EXOG_STRATEGIES,
    check_future_table,
    describe_generation_risks,
    resolve_future_exog,
)


def _hist(cols, n=30, freq="daily", start="2022-01-03"):  # 2022-01-03 is a Monday
    step = "D" if freq == "daily" else "7D"
    df = pd.DataFrame({"date": pd.date_range(start, periods=n, freq=step)})
    for name, vals in cols.items():
        df[name] = vals
    return df


def _future(df, horizon, freq="daily"):
    step = pd.Timedelta(days=1 if freq == "daily" else 7)
    start = df["date"].max() + step
    return pd.Series(pd.date_range(start, periods=horizon, freq=("D" if freq == "daily" else "7D")))


def _resolve(df, cols, horizon, **kw):
    fut = _future(df, horizon, kw.get("frequency", "daily"))
    return resolve_future_exog(df, "date", cols, fut, kw.pop("frequency", "daily"), **kw)


# --------------------------------------------------------------------------
# carry_forward
# --------------------------------------------------------------------------


def test_carry_forward_holds_last_value():
    df = _hist({"spend": np.arange(30.0)})
    out, gen = _resolve(df, ["spend"], 5, strategy="carry_forward")
    assert list(out["spend"]) == [29.0] * 5
    assert gen == ["spend"]
    assert len(out) == 5


def test_flag_defaults_to_zero():
    df = _hist({"promo": ([0, 1] * 15)})
    out, _ = _resolve(df, ["promo"], 4, strategy="carry_forward")
    assert list(out["promo"]) == [0.0] * 4


# --------------------------------------------------------------------------
# recent_average
# --------------------------------------------------------------------------


def test_recent_average_mean_of_last_n():
    df = _hist({"spend": np.arange(30.0)})
    out, _ = _resolve(df, ["spend"], 3, strategy="recent_average", window=10)
    assert out["spend"].iloc[0] == pytest.approx(np.mean(np.arange(20, 30)))


def test_recent_average_median_robust_to_spike():
    # varied history (so it's continuous, not a 2-value flag); the last-10
    # window is nine 10s plus one big spike.
    vals = np.concatenate([np.arange(1, 21.0), np.full(9, 10.0), [10000.0]])
    df = _hist({"spend": vals})
    out_med, _ = _resolve(
        df, ["spend"], 3, strategy="recent_average", aggregator="median", window=10
    )
    out_mean, _ = _resolve(
        df, ["spend"], 3, strategy="recent_average", aggregator="mean", window=10
    )
    assert out_med["spend"].iloc[0] == pytest.approx(10.0)  # median ignores the spike
    assert out_mean["spend"].iloc[0] > 1000  # mean is dragged up by it


# --------------------------------------------------------------------------
# historical_average (season-aware)
# --------------------------------------------------------------------------


def test_historical_average_is_season_aware():
    # value depends only on day-of-week: Mon=100, Tue=110, ...
    dow = np.tile(np.arange(7), 4)  # 28 days from a Monday
    df = _hist({"spend": 100 + dow * 10.0}, n=28)
    out, _ = _resolve(df, ["spend"], 7, strategy="historical_average")
    # future starts on a Monday -> should reproduce the dow pattern exactly
    assert list(np.round(out["spend"], 1)) == [100, 110, 120, 130, 140, 150, 160]


# --------------------------------------------------------------------------
# seasonal_naive (+ drift)
# --------------------------------------------------------------------------


def test_seasonal_naive_has_drift_and_varies_by_phase():
    day = np.arange(28.0)
    dow = np.tile(np.arange(7), 4)
    df = _hist({"spend": 100 + 2 * day + dow * 5.0}, n=28)  # upward trend + weekly shape
    out, _ = _resolve(df, ["spend"], 7, strategy="seasonal_naive")
    fut = out["spend"].to_numpy()
    assert len(set(np.round(fut, 2))) > 1  # varies by phase
    assert fut.mean() > df["spend"].to_numpy()[-7:].mean()  # drift carried upward


def test_seasonal_naive_falls_back_under_one_cycle():
    df = _hist({"spend": np.arange(5.0)}, n=5)  # < 7 days
    out, _ = _resolve(df, ["spend"], 3, strategy="seasonal_naive")
    assert list(out["spend"]) == [4.0] * 3  # carry_forward fallback


def test_seasonal_naive_flag_follows_cycle_not_zero():
    # a flag that is 1 only on Mondays -> future Monday should be 1
    dow = np.tile(np.arange(7), 4)
    df = _hist({"holiday": (dow == 0).astype(float)}, n=28)
    out, _ = _resolve(df, ["holiday"], 7, strategy="seasonal_naive")
    assert out["holiday"].iloc[0] == 1.0  # future Monday
    assert out["holiday"].iloc[1:].sum() == 0.0  # other days 0


# --------------------------------------------------------------------------
# linear_trend + non-negativity
# --------------------------------------------------------------------------


def test_linear_trend_extends_the_line():
    df = _hist({"spend": 3.0 * np.arange(30)})  # slope 3
    out, _ = _resolve(df, ["spend"], 4, strategy="linear_trend")
    assert list(np.round(out["spend"], 1)) == [90, 93, 96, 99]


def test_non_negative_driver_is_clipped_at_zero():
    df = _hist({"spend": (30 - np.arange(30.0))})  # declines 30..1, all >= 0
    out, _ = _resolve(df, ["spend"], 5, strategy="linear_trend")
    assert (out["spend"] >= 0).all()
    assert out["spend"].iloc[-1] == 0.0  # projected negative -> clipped


def test_driver_with_negatives_is_not_clipped():
    df = _hist({"net": (10 - np.arange(30.0))})  # goes negative in history
    out, _ = _resolve(df, ["net"], 5, strategy="linear_trend")
    assert (out["net"] < 0).any()  # negatives preserved


# --------------------------------------------------------------------------
# cycle -- replays the trailing history verbatim
# --------------------------------------------------------------------------


def test_cycle_replays_tail_within_history():
    df = _hist({"spend": np.arange(50, 110.0)}, n=60)  # 50..109
    out, _ = _resolve(df, ["spend"], 3, strategy="cycle")
    assert list(out["spend"]) == [107, 108, 109]


def test_cycle_tiles_when_horizon_exceeds_history():
    df = _hist({"spend": np.arange(50, 70.0)}, n=20)
    out, _ = _resolve(df, ["spend"], 45, strategy="cycle")
    assert len(out) == 45 and not out["spend"].isna().any()


def test_cycle_is_exempt_from_flag_handling():
    df = _hist({"promo": ([0, 1] * 15)})  # binary, but cycle tiles as-is
    out, _ = _resolve(df, ["promo"], 4, strategy="cycle")
    assert set(np.unique(out["promo"])) <= {0.0, 1.0}
    assert out["promo"].sum() > 0  # NOT forced to zero


# --------------------------------------------------------------------------
# flag detection: explicit event_columns vs auto-detect
# --------------------------------------------------------------------------


def test_explicit_event_columns_override_autodetect():
    df = _hist({"spend": np.arange(30.0), "promo": ([0, 1] * 15)})
    # declare spend a flag (it isn't) and DON'T list promo -> promo now continuous
    out, _ = _resolve(df, ["spend", "promo"], 3, strategy="carry_forward", event_columns=["spend"])
    assert list(out["spend"]) == [0.0] * 3  # treated as flag -> 0
    assert list(out["promo"]) == [df["promo"].iloc[-1]] * 3  # treated as continuous -> carried


def test_autodetect_flags_when_event_columns_none():
    df = _hist({"promo": ([0, 1] * 15)})
    out, _ = _resolve(df, ["promo"], 3, strategy="carry_forward", event_columns=None)
    assert list(out["promo"]) == [0.0] * 3


# --------------------------------------------------------------------------
# partial fill
# --------------------------------------------------------------------------


def test_partial_fill_keeps_supplied_generates_rest():
    df = _hist({"spend": np.arange(30.0), "promo": ([0, 1] * 15)})
    supplied = pd.DataFrame({"spend": [999.0] * 5})
    out, gen = _resolve(df, ["spend", "promo"], 5, strategy="carry_forward", supplied=supplied)
    assert list(out["spend"]) == [999.0] * 5  # supplied kept
    assert list(out["promo"]) == [0.0] * 5  # generated
    assert gen == ["promo"]  # only promo generated


def test_no_supplied_generates_all():
    df = _hist({"spend": np.arange(30.0)})
    out, gen = _resolve(df, ["spend"], 5, strategy="carry_forward")
    assert gen == ["spend"]


# --------------------------------------------------------------------------
# robustness & validation
# --------------------------------------------------------------------------


def test_nan_history_produces_no_nan_output():
    vals = np.arange(30.0)
    vals[-1] = np.nan
    vals[5] = np.nan
    df = _hist({"spend": vals})
    for strat in (
        "carry_forward",
        "recent_average",
        "historical_average",
        "seasonal_naive",
        "linear_trend",
    ):
        out, _ = _resolve(df, ["spend"], 6, strategy=strat)
        assert np.isfinite(out["spend"].to_numpy()).all(), strat


def test_horizon_longer_than_history_is_fine():
    df = _hist({"spend": np.arange(10.0)}, n=10)
    for strat in FUTURE_EXOG_STRATEGIES:
        out, _ = _resolve(df, ["spend"], 40, strategy=strat)
        assert len(out) == 40, strat


def test_weekly_frequency_carry_forward():
    df = _hist({"spend": np.arange(60.0)}, n=60, freq="weekly")
    out, _ = _resolve(df, ["spend"], 4, strategy="carry_forward", frequency="weekly")
    assert list(out["spend"]) == [59.0] * 4


def test_unknown_strategy_raises():
    df = _hist({"spend": np.arange(30.0)})
    with pytest.raises(InvalidConfigError):
        _resolve(df, ["spend"], 5, strategy="does_not_exist")


def test_unknown_aggregator_raises():
    df = _hist({"spend": np.arange(30.0)})
    with pytest.raises(InvalidConfigError):
        _resolve(df, ["spend"], 5, strategy="recent_average", aggregator="mode")


# --------------------------------------------------------------------------
# check_future_table: the contract for a caller-supplied future table
# --------------------------------------------------------------------------

# 60 daily rows, 2024-01-01 .. 2024-02-29, so the 5-period forecast runs
# 2024-03-01 .. 2024-03-05 -- every day and month is <= 12, which makes
# DD/MM and MM/DD text dates ambiguous on their own.
_CHECK_HIST = pd.DataFrame(
    {
        "date": pd.date_range("2024-01-01", periods=60, freq="D"),
        "sales": np.arange(60) + 100.0,
        "spend": np.linspace(100.0, 200.0, 60),
        "promo": [0, 1] * 30,
        "tv": np.linspace(5.0, 10.0, 60),  # in the data, not declared as a driver
    }
)
_CHECK_FUTURE = pd.Series(pd.date_range("2024-03-01", periods=5, freq="D"))


def _check(table, hist=None, declared=("spend", "promo"), active=None, events=("promo",), **kw):
    hist = _CHECK_HIST if hist is None else hist
    return check_future_table(
        table,
        history_df=hist,
        date_column="date",
        target_column="sales",
        independent_columns=list(declared if active is None else active),
        declared_columns=list(declared),
        event_columns=None if events is None else list(events),
        future_dates=kw.pop("future_dates", _CHECK_FUTURE),
        frequency="daily",
        **kw,
    )


def _table(rows=5, **columns):
    data = {"spend": [150.0] * rows, "promo": [0] * rows}
    data.update(columns)
    return pd.DataFrame(data)


def test_check_accepts_a_valid_table_and_returns_floats_on_a_positional_index():
    out = _check(_table().set_index(pd.RangeIndex(500, 505)))
    assert list(out.columns) == ["spend", "promo"]
    assert list(out.index) == [0, 1, 2, 3, 4]
    assert out["promo"].dtype == float


def test_check_accepts_a_partial_table():
    out = _check(_table()[["spend"]])
    assert list(out.columns) == ["spend"]


@pytest.mark.parametrize("bad", [{"spend": [150.0] * 5}, pd.Series([150.0] * 5, name="spend")])
def test_check_rejects_anything_but_a_dataframe(bad):
    with pytest.raises(InvalidConfigError, match="must be a pandas DataFrame"):
        _check(bad)


def test_check_rejects_a_table_when_no_drivers_were_declared():
    with pytest.raises(InvalidConfigError, match="no independent_columns were declared"):
        _check(_table(), declared=())


def test_check_rejects_duplicate_column_names():
    table = _table()
    table.columns = ["spend", "spend"]
    with pytest.raises(InvalidConfigError, match="more than one column named"):
        _check(table)


def test_check_rejects_a_misspelled_driver_and_suggests_the_real_one():
    with pytest.raises(InvalidConfigError, match="did you mean 'spend'"):
        _check(_table().rename(columns={"spend": "Spend"}))


def test_check_rejects_a_column_that_is_in_the_data_but_not_a_driver():
    with pytest.raises(InvalidConfigError, match="in your data, but not in independent_columns"):
        _check(_table(tv=[7.0] * 5))


def test_check_allows_and_ignores_the_date_and_target_columns():
    out = _check(_table(date=_CHECK_FUTURE, sales=[1.0] * 5))
    assert list(out.columns) == ["spend", "promo"]


@pytest.mark.parametrize("rows", [0, 3, 7])
def test_check_rejects_a_row_count_different_from_the_horizon(rows):
    with pytest.raises(InvalidConfigError, match=rf"has {rows} row\(s\), but horizon=5"):
        _check(_table(rows))


def test_check_accepts_matching_dates_as_a_column_or_an_index():
    _check(_table(date=_CHECK_FUTURE))
    _check(_table().set_index(pd.DatetimeIndex(_CHECK_FUTURE)))
    _check(_table(date=_CHECK_FUTURE.dt.strftime("%Y-%m-%d")))


@pytest.mark.parametrize("fmt", ["%d/%m/%Y", "%m/%d/%Y"])
def test_check_settles_ambiguous_text_dates_against_the_forecast_dates(fmt):
    # On their own these could be day-first or month-first; only one reading
    # gives the forecast dates, so that one is used.
    _check(_table(date=_CHECK_FUTURE.dt.strftime(fmt)))


def test_check_rejects_dates_that_do_not_match_the_forecast():
    shifted = _CHECK_FUTURE + pd.Timedelta(days=7)
    with pytest.raises(InvalidConfigError, match="Row 1 .* is dated 2024-03-08, .* 2024-03-01"):
        _check(_table(date=shifted))


def test_check_rejects_dates_out_of_order():
    with pytest.raises(InvalidConfigError, match="not in order"):
        _check(_table(date=_CHECK_FUTURE[::-1].reset_index(drop=True)))


def test_check_rejects_a_blank_date():
    dates = list(_CHECK_FUTURE[:-1]) + [None]
    with pytest.raises(InvalidConfigError, match="blank"):
        _check(_table(date=dates))


def test_check_reports_every_bad_value_in_one_error():
    table = _table().astype({"spend": object})
    table.loc[1, "spend"] = None
    table.loc[2, "spend"] = np.inf
    with pytest.raises(InvalidConfigError) as caught:
        _check(table)
    message = str(caught.value)
    assert "'spend' is blank for 2024-03-02" in message
    assert "'spend' is infinite for 2024-03-03" in message


def test_check_rejects_text_among_numbers_naming_the_date():
    table = _table().astype({"spend": object})
    table.loc[2, "spend"] = "1,200"
    with pytest.raises(
        InvalidConfigError,
        match=r"'spend' has values that aren't numbers, e\.g\. \['1,200'\] on 2024-03-03",
    ):
        _check(table)


def test_check_reads_yes_no_words_as_a_flag():
    out = _check(_table(promo=["No", "Yes", "n", "Y", "OFF"]))
    assert list(out["promo"]) == [0.0, 1.0, 0.0, 1.0, 0.0]


def test_check_rejects_an_unclear_text_pair():
    with pytest.raises(InvalidConfigError, match="can't tell which one means"):
        _check(_table(promo=["Promo", "None", "Promo", "None", "None"]))


def test_check_rejects_infinite_values():
    with pytest.raises(InvalidConfigError, match="'spend' is infinite"):
        _check(_table(spend=[150.0, np.inf, 150.0, 150.0, 150.0]))


def test_check_rejects_negative_values_for_a_driver_that_is_never_negative():
    with pytest.raises(InvalidConfigError, match="never negative in your data"):
        _check(_table(spend=[150.0, -5.0, 150.0, 150.0, 150.0]))


def test_check_allows_negative_values_when_the_history_has_them():
    hist = _CHECK_HIST.assign(spend=np.linspace(-50.0, 50.0, 60))
    _check(_table(spend=[-10.0] * 5), hist=hist)


def test_check_rejects_a_flag_value_never_seen_in_the_history():
    with pytest.raises(InvalidConfigError, match="'promo' is an on/off flag"):
        _check(_table(promo=[0, 2, 0, 0, 0]))


def test_check_rejects_an_unseen_value_for_an_auto_detected_flag():
    with pytest.raises(InvalidConfigError, match="'promo' is an on/off flag"):
        _check(_table(promo=[0, 2, 0, 0, 0]), events=None)


def test_check_reads_booleans_as_zero_and_one():
    out = _check(_table(promo=[False, True, False, False, False]))
    assert list(out["promo"]) == [0.0, 1.0, 0.0, 0.0, 0.0]


def test_check_does_not_treat_a_two_level_continuous_driver_as_a_flag():
    # Two price points in the history, auto-detection on: a new price is a
    # scenario, not an error.
    hist = _CHECK_HIST.assign(spend=[9.99, 12.99] * 30)
    with pytest.warns(UserWarning, match="outside anything in your data"):
        _check(_table(spend=[14.99] * 5), hist=hist, events=None)


def test_check_warns_and_drops_a_supplied_driver_that_was_dropped_as_constant():
    with pytest.warns(UserWarning, match="dropped when the forecaster was created"):
        out = _check(_table(), active=["promo"])
    assert list(out.columns) == ["promo"]


def test_check_warns_on_out_of_range_values_naming_each_models_behaviour():
    with pytest.warns(UserWarning, match="outside anything in your data") as caught:
        _check(_table(spend=[1000.0] * 5), models=["prophet", "sarimax"])
    message = str(caught[0].message)
    assert "'spend' goes from 1000 to 1000; your data has 100 to 200" in message
    assert "Prophet extends" in message and "SARIMAX caps" in message
    assert "understated a linear effect by about a third at 1.5x" in message
    assert "overstated an effect with diminishing returns" in message


def test_check_out_of_range_warning_leaves_out_the_ensemble_note_for_one_model():
    with pytest.warns(UserWarning, match="outside anything in your data") as caught:
        _check(_table(spend=[1000.0] * 5), models=["prophet"])
    assert "trimmed ensemble" not in str(caught[0].message)


def test_check_out_of_range_warning_leaves_out_the_ensemble_note_below_the_minimum():
    # The note is about values above the highest in the history; a value below
    # the lowest still warns, without it.
    with pytest.warns(UserWarning, match="outside anything in your data") as caught:
        _check(_table(spend=[50.0] * 5), models=["prophet", "sarimax"])
    assert "trimmed ensemble" not in str(caught[0].message)


def test_check_does_not_warn_for_in_range_values():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _check(_table())


# --------------------------------------------------------------------------
# flag detection: only 0/1 columns
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "values, last",
    [
        ([0.0, 5000.0] * 15, 5000.0),  # TV spend: on or off, but a quantity
        ([9.99, 12.99] * 15, 12.99),  # two price points
        ([0.0, 20.0] * 15, 20.0),  # discount %
        ([1.0, 0.5] * 15, 0.5),  # not 0/1
    ],
)
def test_two_value_quantities_are_not_treated_as_flags(values, last):
    df = _hist({"x": values})
    out, _ = _resolve(df, ["x"], 4)
    assert list(out["x"]) == [last] * 4  # carried forward, not zeroed


def test_zero_one_columns_are_still_detected_as_flags():
    df = _hist({"promo": [0, 1] * 15})
    out, _ = _resolve(df, ["promo"], 4)
    assert list(out["promo"]) == [0.0] * 4


# --------------------------------------------------------------------------
# describe_generation_risks: spikes held by carry_forward, missed events
# --------------------------------------------------------------------------


def _risk_hist(n=60, spend=None, flag=None, end="2025-09-28"):
    dates = pd.date_range(end=end, periods=n, freq="W-SUN")
    df = pd.DataFrame({"date": dates, "spend": [1000.0] * n if spend is None else spend})
    if flag is not None:
        df["event"] = flag
    return df


def _risks(df, cols, generated=None, horizon=8, strategy="carry_forward", events=None):
    fut = pd.Series(
        pd.date_range(df["date"].max() + pd.Timedelta(days=7), periods=horizon, freq="7D")
    )
    return describe_generation_risks(
        df,
        "date",
        cols,
        cols if generated is None else generated,
        fut,
        "weekly",
        strategy,
        event_columns=events,
    )


def test_risk_warns_when_carry_forward_holds_a_last_period_spike():
    spend = [1000.0] * 59 + [2600.0]
    (message,) = _risks(_risk_hist(spend=spend), ["spend"])
    assert "'spend' is 2,600, 2.6x its median over the previous 13 weeks (1,000)" in message
    assert "all 8 forecast weeks" in message and "recent_average" in message


def test_risk_warns_on_a_last_period_dip_and_on_a_usually_zero_driver():
    (dip,) = _risks(_risk_hist(spend=[1000.0] * 59 + [400.0]), ["spend"])
    assert "0.4x its median" in dip
    (zero,) = _risks(_risk_hist(spend=[0.0] * 59 + [5000.0]), ["spend"])
    assert "while its median over the previous 13 weeks is 0" in zero


def test_risk_flags_an_event_in_the_last_period_when_values_are_carried():
    flag = [0] * 59 + [1]
    (message,) = _risks(_risk_hist(flag=flag), ["spend", "event"], events=["event"])
    assert "'event' is 1, so it was an event week" in message


def test_risk_is_quiet_for_a_normal_last_period_or_another_strategy():
    assert _risks(_risk_hist(), ["spend"]) == []
    spike = _risk_hist(spend=[1000.0] * 59 + [2600.0])
    assert _risks(spike, ["spend"], strategy="recent_average") == []
    assert _risks(spike, ["spend"], generated=[]) == []  # supplied, not generated
    assert _risks(spike, ["spend"], strategy="cycle") == []


def test_risk_warns_when_a_generated_flag_misses_last_years_event():
    dates = pd.date_range(end="2025-10-26", periods=60, freq="W-SUN")
    flag = [1 if d == pd.Timestamp("2024-11-24") else 0 for d in dates]
    df = _risk_hist(flag=flag, end="2025-10-26")
    (message,) = _risks(df, ["spend", "event"], generated=["event"], events=["event"])
    assert "'event' was 1 on 2024-11-24 (now 2025-11-23)" in message
    assert "set to 0 throughout the forecast" in message
    (seasonal,) = _risks(
        df, ["spend", "event"], generated=["event"], strategy="seasonal_naive", events=["event"]
    )
    assert "by week of the year" in seasonal


def test_risk_does_not_ask_for_events_that_were_supplied_or_out_of_range():
    dates = pd.date_range(end="2025-10-26", periods=60, freq="W-SUN")
    flag = [1 if d == pd.Timestamp("2024-11-24") else 0 for d in dates]
    df = _risk_hist(flag=flag, end="2025-10-26")
    assert _risks(df, ["spend", "event"], generated=["spend"], events=["event"]) == []
    assert _risks(df, ["spend", "event"], generated=["event"], horizon=2, events=["event"]) == []
