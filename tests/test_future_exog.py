"""
Unit tests for the future-exogenous-data strategies
(horizon_forecaster/future_exog.py). Pure pandas/numpy -- no models.
"""

import numpy as np
import pandas as pd
import pytest

from horizon_forecaster.exceptions import InvalidConfigError
from horizon_forecaster.future_exog import (
    FUTURE_EXOG_STRATEGIES,
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
# cycle (legacy) -- must reproduce the old replay-the-tail behaviour
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
