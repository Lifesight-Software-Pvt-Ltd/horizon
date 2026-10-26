import math
import warnings

import pandas as pd
import pytest

from horizon_forecaster.exceptions import (
    AllModelsRejectedError,
    InsufficientDataError,
    InvalidConfigError,
    ModelConvergenceError,
    ModelPredictionError,
    UnsupportedFrequencyError,
)
from horizon_forecaster.forecaster import (
    HorizonForecaster,
    EvaluationResult,
    ForecastResult,
    MODEL_REGISTRY,
)

# Tests that run the DEFAULT model set (every registered model) would
# otherwise sample Bayesian PyMC at its production draws (2000 tune + 2000
# draws x 2 chains, per grid candidate) -- minutes per test. These settings
# make its NUTS run trivially cheap while still exercising the real fit ->
# predict path through the orchestrator; the same way prophet/catboost/
# neuralprophet are given fast, deterministic params in these tests. Passed
# as a model_params override, so PyMC also skips its own grid search here.
_FAST_PYMC = {"draws": 40, "tune": 40, "chains": 1, "n_fourier": 1}


def make_raw_df(n=60, freq="D", with_spend=False, with_zero_variance_col=False):
    df = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=n, freq=freq),
            "orders": [100 + i * 2 for i in range(n)],
        }
    )
    if with_spend:
        df["spend"] = [50 + i for i in range(n)]
    if with_zero_variance_col:
        df["constant_col"] = 5  # same value every row -> zero variance
    return df


def make_noisy_raw_df(n=90, freq="D", seed=7):
    """
    Used by the ensemble tests below. make_raw_df()'s perfectly clean,
    noiseless linear trend lets both models fit it almost exactly (MAPE
    well under 1%). A little real noise lifts MAPE into a realistic,
    few-percent range -- which the rejection test below depends on, since
    it needs a real fit to score ABOVE a deliberately low mape_threshold
    to be excluded (a sub-0.5% fit would pass the 0.5 threshold the
    rejection test uses). Kept separate from make_raw_df() so other tests'
    assumptions aren't disturbed.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    trend = 100 + 2 * np.arange(n)
    noise = rng.normal(0, 3, n)
    return pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=n, freq=freq),
            "orders": trend + noise,
        }
    )


# ---------- constructor validation ----------


def test_constructor_validates_frequency():
    df = make_raw_df()
    with pytest.raises(UnsupportedFrequencyError):
        HorizonForecaster(
            df, date_column="date", target_column="orders", frequency="monthly", models=["prophet"]
        )


def test_constructor_validates_training_size():
    df = make_raw_df()
    with pytest.raises(InvalidConfigError):
        HorizonForecaster(
            df,
            date_column="date",
            target_column="orders",
            frequency="daily",
            training_size=30,
            models=["prophet"],
        )


def test_constructor_defaults_to_frequency_sized_evaluation_windows():
    # 60 daily rows is below the ~249 needed for the full 84/28 defaults,
    # so they scale down proportionally (see utils._MAX_DEFAULTED_
    # EVALUATION_SHARE) while keeping the 3:1 validation:test ratio.
    df = make_raw_df()
    with pytest.warns(UserWarning, match="too short for the default"):
        forecaster = HorizonForecaster(
            df,
            date_column="date",
            target_column="orders",
            frequency="daily",
            models=["prophet"],
        )
    assert (forecaster._n_train, forecaster._n_validation, forecaster._n_test) == (33, 20, 7)


def test_constructor_uses_full_default_windows_on_a_long_series():
    df = make_raw_df(n=400)
    forecaster = HorizonForecaster(
        df,
        date_column="date",
        target_column="orders",
        frequency="daily",
        models=["prophet"],
    )
    assert (forecaster._n_validation, forecaster._n_test) == (84, 28)
    assert forecaster._n_train == 400 - 84 - 28


def test_constructor_rejects_mixing_the_percentage_and_window_routes():
    df = make_raw_df(n=400)
    with pytest.raises(InvalidConfigError, match="cannot be combined"):
        HorizonForecaster(
            df,
            date_column="date",
            target_column="orders",
            frequency="daily",
            models=["prophet"],
            training_size=80,
            test_size=10,
        )


def test_constructor_drops_zero_variance_independent_columns():
    df = make_raw_df(with_zero_variance_col=True)
    forecaster = HorizonForecaster(
        df,
        date_column="date",
        target_column="orders",
        frequency="daily",
        independent_columns=["constant_col"],
        models=["prophet"],
    )
    assert "constant_col" in forecaster.dropped_zero_variance_columns
    # Critically: also removed from independent_columns, so the adapter's
    # hard-error never fires against a column WE dropped, not the user.
    assert "constant_col" not in forecaster.independent_columns


def test_constructor_sorts_data_chronologically():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-03", "2026-01-01", "2026-01-02"]),
            "orders": [30, 10, 20],
        }
    )
    forecaster = HorizonForecaster(
        df, date_column="date", target_column="orders", frequency="daily", models=["prophet"]
    )
    assert forecaster.raw_df["orders"].tolist() == [10, 20, 30]


def test_constructor_sorts_text_dates_by_time_not_alphabetically():
    """
    MM/DD/YYYY text spanning a year end sorts wrongly as text: "01/05/2023"
    comes before "12/20/2022". The dates must be parsed before sorting, or
    the train/validation/test windows mix up the calendar.
    """
    dates = pd.date_range("2022-12-20", periods=40, freq="D")
    df = pd.DataFrame({"date": dates.strftime("%m/%d/%Y"), "orders": range(40)})
    forecaster = HorizonForecaster(
        df,
        date_column="date",
        target_column="orders",
        frequency="daily",
        models=["prophet"],
        validation_size=10,
        test_size=5,
    )
    assert list(forecaster.raw_df["date"]) == list(dates)
    assert forecaster.raw_df["orders"].tolist() == list(range(40))


def test_constructor_does_not_modify_the_callers_dataframe():
    dates = pd.date_range("2022-12-20", periods=40, freq="D")
    df = pd.DataFrame({"date": dates.strftime("%d/%m/%Y"), "orders": range(40)})
    before = df.copy()
    HorizonForecaster(
        df,
        date_column="date",
        target_column="orders",
        frequency="daily",
        models=["prophet"],
        validation_size=10,
        test_size=5,
    )
    pd.testing.assert_frame_equal(df, before)


def test_constructor_refuses_ambiguous_text_dates_until_date_format_is_given():
    # 1-12 February: every value fits both DD/MM/YYYY and MM/DD/YYYY.
    df = pd.DataFrame(
        {
            "date": [f"{day:02d}/02/2026" for day in range(1, 13)],
            "orders": range(10, 130, 10),
        }
    )
    kwargs = dict(date_column="date", target_column="orders", frequency="daily")
    with pytest.raises(InvalidConfigError, match="day-first or month-first"):
        HorizonForecaster(df, models=["prophet"], **kwargs)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # tiny series: its warnings are not the point
        forecaster = HorizonForecaster(df, models=["prophet"], date_format="%d/%m/%Y", **kwargs)
    assert forecaster.raw_df["date"].iloc[0] == pd.Timestamp("2026-02-01")


def test_constructor_rejects_duplicate_dates():
    df = make_raw_df(n=60)
    df = pd.concat([df, df.iloc[[5, 6]]], ignore_index=True)
    with pytest.raises(InvalidConfigError, match="appear more than once"):
        HorizonForecaster(
            df, date_column="date", target_column="orders", frequency="daily", models=["prophet"]
        )


def _spacing_df(dates):
    return pd.DataFrame({"date": list(dates), "orders": range(len(dates))})


def _build(df, frequency, **kwargs):
    return HorizonForecaster(
        df,
        date_column="date",
        target_column="orders",
        frequency=frequency,
        models=["prophet"],
        **kwargs,
    )


_DAILY = pd.date_range("2024-01-01", periods=400, freq="D")
_WEEKLY = pd.date_range("2022-01-02", periods=120, freq="7D")


def test_constructor_accepts_clean_daily_and_weekly_dates_without_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _build(_spacing_df(_DAILY), "daily")
        _build(_spacing_df(_WEEKLY), "weekly")


@pytest.mark.parametrize(
    "dates, frequency, use",
    [(_WEEKLY, "daily", "weekly"), (_DAILY, "weekly", "daily")],
)
def test_constructor_rejects_a_frequency_the_dates_contradict(dates, frequency, use):
    with pytest.raises(InvalidConfigError, match=f"Use frequency='{use}'"):
        _build(_spacing_df(dates), frequency)


@pytest.mark.parametrize(
    "dates, frequency, typical",
    [
        (_DAILY[::2], "daily", "2 day"),  # every other day
        (_WEEKLY[::2], "weekly", "14 day"),  # fortnightly
        (pd.date_range("2018-01-01", periods=60, freq="MS"), "weekly", "31 day"),  # monthly
        (pd.date_range("2018-01-01", periods=60, freq="MS"), "daily", "31 day"),
    ],
)
def test_constructor_rejects_spacing_it_does_not_support(dates, frequency, typical):
    with pytest.raises(UnsupportedFrequencyError, match=f"mostly {typical}"):
        _build(_spacing_df(dates), frequency)


def test_constructor_rejects_dates_with_no_regular_spacing():
    """
    Four dates with gaps of 2, 7 and 15 days have no frequency at all; with
    only a warning, the run would go on to produce settings and forecast
    dates for one anyway.
    """
    dates = pd.to_datetime(["2026-01-01", "2026-01-03", "2026-01-10", "2026-01-25"])
    with pytest.raises(UnsupportedFrequencyError, match="resample"):
        _build(_spacing_df(dates), "weekly")


def test_constructor_rejects_weekly_dates_on_different_weekdays():
    dates = _WEEKLY[:60].append(_WEEKLY[60:] + pd.Timedelta(days=1))  # Sundays, then Mondays
    with pytest.raises(InvalidConfigError, match="same weekday"):
        _build(_spacing_df(dates), "weekly")


def test_constructor_rejects_daily_dates_with_inconsistent_times():
    dates = _DAILY[:200].append(_DAILY[200:] + pd.Timedelta(hours=3))
    with pytest.raises(InvalidConfigError, match="whole days apart"):
        _build(_spacing_df(dates), "daily")


@pytest.mark.parametrize(
    "dates, frequency",
    [
        (_DAILY.delete([200]), "daily"),
        (_DAILY.delete(range(100, 128)), "daily"),
        (_DAILY.delete(range(100, 129)), "daily"),
        (_WEEKLY.delete([60]), "weekly"),
        (_WEEKLY.delete(range(50, 54)), "weekly"),
        (_WEEKLY.delete(range(50, 55)), "weekly"),
    ],
)
def test_constructor_rejects_any_missing_date(dates, frequency):
    """
    One missing day or week is enough to raise, scattered or not.
    """
    with pytest.raises(InvalidConfigError, match="Every (day|week) must be present"):
        _build(_spacing_df(dates), frequency)


def test_the_missing_date_error_names_the_dates_and_the_longest_gap():
    # Every 4th day removed from the first 400: 100 single-day gaps, then a
    # 3-day hole, so the longest gap is not the first one.
    dates = _DAILY.delete(list(range(3, 300, 4)) + [350, 351, 352])
    with pytest.raises(InvalidConfigError) as caught:
        _build(_spacing_df(dates), "daily")
    message = str(caught.value)
    assert "'date' is missing 78 days between its first and last dates" in message
    assert "2024-01-04, 2024-01-08, 2024-01-12, 2024-01-16, 2024-01-20 and 73 more" in message
    assert "in 76 gaps (the longest is 3 days, 2024-12-16 to 2024-12-18)" in message
    assert "with 0 where the business was closed" in message
    assert caught.value.provided_value == [
        "2024-01-04",
        "2024-01-08",
        "2024-01-12",
        "2024-01-16",
        "2024-01-20",
    ]


def test_the_missing_date_error_for_one_run_gives_its_first_and_last_date():
    with pytest.raises(
        InvalidConfigError, match=r"5 weeks .* all in a row \(2022-12-11 to 2023-01-08\)"
    ):
        _build(_spacing_df(_WEEKLY.delete(range(49, 54))), "weekly")


def test_the_missing_date_error_for_one_day_is_singular():
    with pytest.raises(InvalidConfigError, match=r"missing 1 day between .*: 2024-07-19\. Every"):
        _build(_spacing_df(_DAILY.delete([200])), "daily")


@pytest.mark.parametrize(
    "keep, named",
    [
        (lambda d: d.dayofweek < 5, "No Saturday or Sunday appears at all"),
        (lambda d: d.dayofweek != 6, "No Sunday appears at all"),
        (lambda d: d.dayofweek < 4, "No Friday, Saturday or Sunday appears at all"),
    ],
)
def test_the_missing_date_error_names_weekdays_that_never_appear(keep, named):
    with pytest.raises(InvalidConfigError, match="add those days with a value of 0") as caught:
        _build(_spacing_df(_DAILY[keep(_DAILY)]), "daily")
    assert named in str(caught.value)


def test_the_missing_date_error_names_no_weekday_when_every_weekday_appears():
    with pytest.raises(InvalidConfigError) as caught:
        _build(_spacing_df(_DAILY.delete([200, 201])), "daily")
    assert "appears at all" not in str(caught.value)


def test_complete_weekday_data_with_zeros_on_closed_days_is_accepted():
    """What the error asks for: the closed days added, with a value of 0."""
    df = _spacing_df(_DAILY)
    df.loc[_DAILY.dayofweek >= 5, "orders"] = 0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _build(df, "daily", selection_metric="mape")


def test_constructor_raises_immediately_when_training_size_leaves_no_test_rows():
    """
    training_size leaving zero test rows is a dataset-level configuration
    problem, identical for every model -- validated at construction time
    (fails fast, before any model even runs), not discovered later inside
    evaluate()'s per-model try/except.
    """
    df = make_raw_df(n=1)
    with pytest.raises(InvalidConfigError):
        HorizonForecaster(
            df,
            date_column="date",
            target_column="orders",
            frequency="daily",
            training_size=100,
            models=["prophet"],
        )


# ---------- models= parameter ----------


def test_models_defaults_to_every_registered_model():
    df = make_raw_df(n=60)
    forecaster = HorizonForecaster(
        df, date_column="date", target_column="orders", frequency="daily"
    )
    assert set(forecaster.models) == set(MODEL_REGISTRY.keys())
    assert set(forecaster.models) == {
        "prophet",
        "catboost",
        "sarimax",
        "neuralprophet",
        "bayesianpymc",
    }


def test_models_can_be_restricted_to_a_subset():
    df = make_raw_df(n=60)
    forecaster = HorizonForecaster(
        df, date_column="date", target_column="orders", frequency="daily", models=["prophet"]
    )
    assert forecaster.models == ["prophet"]


def test_models_raises_on_unknown_model_name():
    df = make_raw_df(n=60)
    with pytest.raises(InvalidConfigError) as exc_info:
        HorizonForecaster(
            df,
            date_column="date",
            target_column="orders",
            frequency="daily",
            models=["prophet", "nonexistent_model"],
        )
    assert "nonexistent_model" in str(exc_info.value)


def test_models_raises_on_empty_list():
    df = make_raw_df(n=60)
    with pytest.raises(InvalidConfigError):
        HorizonForecaster(
            df, date_column="date", target_column="orders", frequency="daily", models=[]
        )


# ---------- config loading ----------


def test_config_loaded_correctly_for_daily():
    df = make_raw_df(n=60)
    forecaster = HorizonForecaster(
        df, date_column="date", target_column="orders", frequency="daily", models=["prophet"]
    )
    assert forecaster._configs["prophet"]["wrapper_config"]["min_rows_required"] == 14
    assert forecaster._configs["prophet"]["base_hyperparameters"]["daily_seasonality"] is False
    assert len(forecaster._configs["prophet"]["hyperparameter_grid"]) == 6


def test_config_loaded_correctly_for_weekly():
    df = make_raw_df(n=60, freq="7D")
    forecaster = HorizonForecaster(
        df, date_column="date", target_column="orders", frequency="weekly", models=["prophet"]
    )
    assert forecaster._configs["prophet"]["wrapper_config"]["min_rows_required"] == 8
    assert len(forecaster._configs["prophet"]["hyperparameter_grid"]) == 6


def test_config_loaded_independently_per_model():
    df = make_raw_df(n=60)
    forecaster = HorizonForecaster(
        df, date_column="date", target_column="orders", frequency="daily"
    )
    assert forecaster._configs["prophet"]["wrapper_config"]["min_rows_required"] == 14
    assert forecaster._configs["catboost"]["wrapper_config"]["min_rows_required"] == 14
    assert len(forecaster._configs["catboost"]["hyperparameter_grid"]) == 18
    # prophet and catboost have genuinely different hyperparameter keys
    assert "seasonality_mode" in forecaster._configs["prophet"]["hyperparameter_grid"][0]
    assert "iterations" in forecaster._configs["catboost"]["hyperparameter_grid"][0]


def test_model_params_override_skips_grid_search_entirely():
    df = make_raw_df(n=60)
    forecaster = HorizonForecaster(
        df,
        date_column="date",
        target_column="orders",
        frequency="daily",
        models=["prophet"],
        model_params={"prophet": {"seasonality_mode": "multiplicative", "yearly_seasonality": 20}},
    )
    assert forecaster._use_grid_search["prophet"] is False
    # Resolved immediately at construction, not lazily
    assert forecaster._selected_hyperparameters["prophet"]["seasonality_mode"] == "multiplicative"
    assert forecaster._selected_hyperparameters["prophet"]["yearly_seasonality"] == 20
    # base_hyperparameters (not part of the override) still applied
    assert forecaster._selected_hyperparameters["prophet"]["daily_seasonality"] is False


def test_model_params_override_actually_reaches_the_fitted_prophet_model():
    df = make_raw_df(n=60)
    forecaster = HorizonForecaster(
        df,
        date_column="date",
        target_column="orders",
        frequency="daily",
        models=["prophet"],
        model_params={"prophet": {"seasonality_mode": "multiplicative"}},
    )
    result = forecaster.evaluate()["prophet"]
    assert result.fitted_model.params["seasonality_mode"] == "multiplicative"
    assert result.grid_search_summary is None  # no search happened, nothing to report


def test_no_model_params_uses_grid_search_mode():
    df = make_raw_df(n=60)
    forecaster = HorizonForecaster(
        df, date_column="date", target_column="orders", frequency="daily", models=["prophet"]
    )
    assert forecaster._use_grid_search["prophet"] is True
    # Not resolved yet — lazy, until evaluate()/forecast() actually runs it
    assert forecaster._selected_hyperparameters["prophet"] is None


def test_model_params_are_independent_per_model():
    """
    Giving prophet an explicit override should not affect catboost's own
    (independent) decision to use grid search, and vice versa.
    """
    df = make_raw_df(n=60)
    forecaster = HorizonForecaster(
        df,
        date_column="date",
        target_column="orders",
        frequency="daily",
        model_params={"prophet": {"seasonality_mode": "multiplicative"}},  # catboost gets no entry
    )
    assert forecaster._use_grid_search["prophet"] is False
    assert forecaster._use_grid_search["catboost"] is True


# ---------- hyperparameter grid search ----------


def test_grid_search_evaluates_all_six_candidates():
    df = make_raw_df(n=60)
    forecaster = HorizonForecaster(
        df, date_column="date", target_column="orders", frequency="daily", models=["prophet"]
    )
    result = forecaster.evaluate()["prophet"]
    assert result.grid_search_summary is not None
    assert len(result.grid_search_summary) == 6


def test_grid_search_summary_is_sorted_best_to_worst():
    df = make_raw_df(n=60)
    forecaster = HorizonForecaster(
        df, date_column="date", target_column="orders", frequency="daily", models=["prophet"]
    )
    result = forecaster.evaluate()["prophet"]
    mapes = [c["mape"] for c in result.grid_search_summary]
    assert mapes == sorted(mapes)


def test_grid_search_winner_matches_evaluate_result():
    df = make_raw_df(n=60)
    forecaster = HorizonForecaster(
        df, date_column="date", target_column="orders", frequency="daily", models=["prophet"]
    )
    result = forecaster.evaluate()["prophet"]
    # The grid summary holds VALIDATION scores -- that is the window
    # candidates were chosen on. result.mape is the test-window score and
    # is a different number by design; if these two ever match again, the
    # selection/reporting separation has regressed.
    assert result.validation_mape.value == result.grid_search_summary[0]["mape"]


def test_grid_search_runs_only_once_across_repeated_evaluate_calls():
    df = make_raw_df(n=60)
    forecaster = HorizonForecaster(
        df, date_column="date", target_column="orders", frequency="daily", models=["prophet"]
    )
    result1 = forecaster.evaluate()["prophet"]
    result2 = forecaster.evaluate()["prophet"]
    # Same cached result object, not recomputed
    assert result1 is result2


def test_forecast_reuses_grid_search_winner_from_evaluate():
    df = make_raw_df(n=60)
    forecaster = HorizonForecaster(
        df, date_column="date", target_column="orders", frequency="daily", models=["prophet"]
    )
    forecaster.evaluate()
    winning_params = forecaster._selected_hyperparameters["prophet"]

    forecaster.forecast(horizon=5)
    assert forecaster._forecast_models["prophet"].params == winning_params


def test_forecast_runs_grid_search_itself_when_called_before_evaluate():
    """
    forecast() must be independently capable of selecting hyperparameters
    if evaluate() was never called first in this session.
    """
    df = make_raw_df(n=60)
    forecaster = HorizonForecaster(
        df, date_column="date", target_column="orders", frequency="daily", models=["prophet"]
    )
    assert forecaster._selected_hyperparameters["prophet"] is None

    forecaster.forecast(horizon=5)
    assert forecaster._selected_hyperparameters["prophet"] is not None
    assert "prophet" in forecaster._selection_cache


# ---------- evaluate() ----------


def test_evaluate_returns_structured_result():
    df = make_raw_df(n=60)
    forecaster = HorizonForecaster(
        df, date_column="date", target_column="orders", frequency="daily", models=["prophet"]
    )
    results = forecaster.evaluate()
    result = results["prophet"]

    assert isinstance(results, dict)
    assert isinstance(result, EvaluationResult)
    assert result.model_name == "prophet"
    # 60 daily rows is below the ~249 the full 84/28 defaults need, so they
    # scale to 45% of the data (20 + 7), leaving 33 to train on.
    assert (result.n_train, result.n_validation, result.n_test) == (33, 20, 7)
    assert len(result.test_predictions) == 7
    assert result.fitted_model.is_fitted is True
    # The two scores must come from DIFFERENT windows -- checking they are
    # merely "not None" cannot fail, since both are non-optional fields.
    assert result.validation_mape.total_count == result.n_validation
    assert result.mape.total_count == result.n_test


def test_evaluate_produces_low_mape_on_clean_trend():
    """
    A clean, noise-free linear trend should be near-trivially predictable
    -> confirms the whole wired pipeline (split -> adapt -> fit ->
    predict -> MAPE) produces a numerically sensible result, not just
    code that runs without error.
    """
    df = make_raw_df(n=60)
    forecaster = HorizonForecaster(
        df, date_column="date", target_column="orders", frequency="daily", models=["prophet"]
    )
    result = forecaster.evaluate()["prophet"]
    assert result.mape.value < 5.0  # should be very low on a clean trend


def test_evaluate_test_predictions_have_expected_columns():
    df = make_raw_df(n=60)
    forecaster = HorizonForecaster(
        df, date_column="date", target_column="orders", frequency="daily", models=["prophet"]
    )
    result = forecaster.evaluate()["prophet"]
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
        models=["prophet"],
    )
    result = forecaster.evaluate()["prophet"]
    assert result.fitted_model._regressor_columns == ["spend"]


def test_evaluate_all_models_fail_raises_all_models_rejected_error():
    """
    A single requested model failing is treated identically to multiple
    models failing: if NOTHING succeeded, AllModelsRejectedError is
    raised rather than returning a silently-empty dict. The real
    underlying exception is still available via evaluate_errors.
    """
    df = make_raw_df(n=10)  # 6 train + 3 validation rows: below the daily floor of 14 even refit
    forecaster = HorizonForecaster(
        df, date_column="date", target_column="orders", frequency="daily", models=["prophet"]
    )
    with pytest.raises(AllModelsRejectedError) as exc_info:
        forecaster.evaluate()
    assert exc_info.value.rejected_models == ["prophet"]
    assert isinstance(forecaster.evaluate_errors["prophet"], InsufficientDataError)


def test_evaluate_weekly_frequency_end_to_end():
    df = make_raw_df(n=40, freq="7D")
    forecaster = HorizonForecaster(
        df, date_column="date", target_column="orders", frequency="weekly", models=["prophet"]
    )
    result = forecaster.evaluate()["prophet"]
    assert result.model_name == "prophet"
    assert result.mape.value < 10.0


# ---------- multi-model evaluate()/forecast() ----------


def test_evaluate_runs_both_models_by_default():
    df = make_raw_df(n=90)
    forecaster = HorizonForecaster(
        df,
        date_column="date",
        target_column="orders",
        frequency="daily",
        model_params={
            "prophet": {"seasonality_mode": "additive"},
            "catboost": {"iterations": 100, "depth": 3},
            "bayesianpymc": _FAST_PYMC,
        },
    )
    results = forecaster.evaluate()
    assert set(results.keys()) == set(
        MODEL_REGISTRY.keys()
    )  # every registered model runs by default
    assert isinstance(results["prophet"], EvaluationResult)
    assert isinstance(results["catboost"], EvaluationResult)
    assert isinstance(results["sarimax"], EvaluationResult)
    assert isinstance(results["bayesianpymc"], EvaluationResult)


def test_forecast_runs_both_models_by_default():
    df = make_raw_df(n=90)
    forecaster = HorizonForecaster(
        df,
        date_column="date",
        target_column="orders",
        frequency="daily",
        model_params={
            "prophet": {"seasonality_mode": "additive"},
            "catboost": {"iterations": 100, "depth": 3},
            "bayesianpymc": _FAST_PYMC,
        },
    )
    results = forecaster.forecast(horizon=5)
    assert set(results.keys()) == set(
        MODEL_REGISTRY.keys()
    )  # every registered model runs by default
    for result in results.values():
        assert isinstance(result, ForecastResult)
        assert len(result.forecast) == 5


def test_evaluate_partial_failure_returns_successful_models_and_records_errors():
    """
    CatBoost and Prophet use the exact same raw row-count check, so a
    genuine partial failure here needs a real model-family failure, not a
    data-size coincidence: depth=-1 is a
    confirmed-empirically-real invalid CatBoost hyperparameter (see
    test_catboost.py), used here to make CatBoost fail while Prophet
    (given valid params) succeeds on the same dataset.
    """
    df = make_raw_df(n=60)
    forecaster = HorizonForecaster(
        df,
        date_column="date",
        target_column="orders",
        frequency="daily",
        models=["prophet", "catboost"],  # focused 2-model partial-failure case
        model_params={"prophet": {"seasonality_mode": "additive"}, "catboost": {"depth": -1}},
    )
    results = forecaster.evaluate()
    assert set(results.keys()) == {"prophet"}
    assert set(forecaster.evaluate_errors.keys()) == {"catboost"}
    assert isinstance(forecaster.evaluate_errors["catboost"], ModelConvergenceError)


def test_evaluate_errors_reset_fresh_on_each_call_not_accumulated():
    df = make_raw_df(n=60)
    forecaster = HorizonForecaster(
        df,
        date_column="date",
        target_column="orders",
        frequency="daily",
        model_params={
            "prophet": {"seasonality_mode": "additive"},
            "catboost": {"depth": -1},
            "bayesianpymc": _FAST_PYMC,
        },
    )
    forecaster.evaluate()
    errors_first_call = dict(forecaster.evaluate_errors)
    forecaster.evaluate()
    errors_second_call = dict(forecaster.evaluate_errors)
    assert set(errors_first_call.keys()) == set(errors_second_call.keys()) == {"catboost"}


def test_evaluate_all_models_fail_raises_all_models_rejected_error_multi_model():
    df = make_raw_df(n=10)  # below every model's floor
    forecaster = HorizonForecaster(
        df, date_column="date", target_column="orders", frequency="daily"
    )
    with pytest.raises(AllModelsRejectedError) as exc_info:
        forecaster.evaluate()
    assert set(exc_info.value.rejected_models) == set(MODEL_REGISTRY.keys())


def test_grid_search_is_independent_per_model():
    df = make_raw_df(n=60)
    forecaster = HorizonForecaster(
        df,
        date_column="date",
        target_column="orders",
        frequency="daily",
        model_params={
            "prophet": {"seasonality_mode": "additive"},
            "bayesianpymc": _FAST_PYMC,
        },  # catboost gets no override
    )
    results = forecaster.evaluate()
    assert results["prophet"].grid_search_summary is None
    assert results["catboost"].grid_search_summary is not None
    assert len(results["catboost"].grid_search_summary) == 18


def test_catboost_test_split_features_are_anchored_to_train_history_not_test_split():
    """
    Direct test of the orchestrator's split construction,
    against CatBoostAdapter's lag/rolling-free feature set: the
    test split's features must be built via transform_future() (reusing
    the adapter's retained reference_date from transform_train()), not
    transform_train() on the test split alone -- the latter would reset
    time_idx back to 0 at the start of the test split instead of
    continuing the trend seamlessly from where training left off.
    """
    df = make_raw_df(n=90)
    forecaster = HorizonForecaster(
        df,
        date_column="date",
        target_column="orders",
        frequency="daily",
        models=["catboost"],
        model_params={"catboost": {"iterations": 50, "depth": 3}},
    )
    adapted_train, adapted_test_features, y_test = forecaster._prepare_split("catboost")

    assert adapted_test_features["time_idx"].iloc[0] == adapted_train["time_idx"].iloc[-1] + 1
    # and time_idx must vary per row across the scoring window, not be frozen
    assert adapted_test_features["time_idx"].nunique() == forecaster._n_validation


# ---------- forecast() ----------


def test_forecast_returns_structured_result():
    df = make_raw_df(n=60)
    forecaster = HorizonForecaster(
        df, date_column="date", target_column="orders", frequency="daily", models=["prophet"]
    )
    results = forecaster.forecast(horizon=10)
    result = results["prophet"]

    assert isinstance(results, dict)
    assert isinstance(result, ForecastResult)
    assert result.model_name == "prophet"
    assert result.horizon == 10
    assert len(result.forecast) == 10
    assert result.used_default_future_strategy is False  # no independent columns at all


def test_forecast_continues_trend_correctly():
    """
    orders = 100 + i*2, so trend is +2/period. We check the diffs are
    CLOSE to +2 rather than exactly == 2.0: whichever grid candidate wins
    (picked by lowest test-set MAPE) may be a multiplicative-seasonality
    one, which introduces tiny (sub-0.01) numerical deviation from a pure
    linear trend even on clean data. Exact equality made this test flaky
    depending on which of the 6 real candidates the grid search selects.
    """
    df = make_raw_df(n=60)
    forecaster = HorizonForecaster(
        df, date_column="date", target_column="orders", frequency="daily", models=["prophet"]
    )
    result = forecaster.forecast(horizon=5)["prophet"]
    yhats = result.forecast["yhat"].tolist()
    diffs = [yhats[i + 1] - yhats[i] for i in range(len(yhats) - 1)]
    assert all(1.9 <= d <= 2.1 for d in diffs), diffs


def test_forecast_raises_on_non_positive_horizon():
    df = make_raw_df(n=60)
    forecaster = HorizonForecaster(
        df, date_column="date", target_column="orders", frequency="daily", models=["prophet"]
    )
    with pytest.raises(InvalidConfigError):
        forecaster.forecast(horizon=0)
    with pytest.raises(InvalidConfigError):
        forecaster.forecast(horizon=-5)


@pytest.mark.parametrize("horizon", [2.5, float("nan"), True, "5"])
def test_forecast_raises_on_non_integer_horizon(horizon):
    # A float would build int(horizon) future dates and then fail inside
    # every model as a row-count mismatch, surfacing as AllModelsRejectedError.
    # Raised before any model is fitted, so this costs no fit.
    df = make_raw_df(n=60)
    forecaster = HorizonForecaster(
        df, date_column="date", target_column="orders", frequency="daily", models=["prophet"]
    )
    with pytest.raises(InvalidConfigError, match="positive integer"):
        forecaster.forecast(horizon=horizon)


def test_forecast_refits_only_once_and_caches_across_calls():
    """
    Critical design contract: the full-data refit must happen exactly
    once per instance, regardless of how many times forecast() is
    called afterward (even with different horizons).
    """
    df = make_raw_df(n=60)
    forecaster = HorizonForecaster(
        df, date_column="date", target_column="orders", frequency="daily", models=["prophet"]
    )

    assert "prophet" not in forecaster._forecast_models
    forecaster.forecast(horizon=5)
    model_after_first_call = forecaster._forecast_models["prophet"]
    assert model_after_first_call is not None

    forecaster.forecast(horizon=10)  # different horizon
    model_after_second_call = forecaster._forecast_models["prophet"]

    assert model_after_first_call is model_after_second_call  # same object, not refit


def test_forecast_uses_default_strategy_when_independent_data_not_supplied():
    df = make_raw_df(n=60, with_spend=True)
    forecaster = HorizonForecaster(
        df,
        date_column="date",
        target_column="orders",
        frequency="daily",
        independent_columns=["spend"],
        models=["prophet"],
    )
    result = forecaster.forecast(horizon=5)["prophet"]
    assert result.used_default_future_strategy is True
    assert len(result.forecast) == 5


def test_forecast_uses_supplied_future_independent_data_when_provided():
    df = make_raw_df(n=60, with_spend=True)
    forecaster = HorizonForecaster(
        df,
        date_column="date",
        target_column="orders",
        frequency="daily",
        independent_columns=["spend"],
        models=["prophet"],
    )
    future_spend = pd.DataFrame({"spend": [500, 510, 520, 530, 540]})
    result = forecaster.forecast(horizon=5, future_independent_data=future_spend)["prophet"]
    assert result.used_default_future_strategy is False
    assert len(result.forecast) == 5


def test_forecast_rejects_a_future_table_with_the_wrong_number_of_rows():
    df = make_raw_df(n=60, with_spend=True)
    forecaster = HorizonForecaster(
        df,
        date_column="date",
        target_column="orders",
        frequency="daily",
        independent_columns=["spend"],
        models=["catboost"],
    )
    with pytest.raises(InvalidConfigError, match=r"has 3 row\(s\), but horizon=5"):
        forecaster.forecast(horizon=5, future_independent_data=pd.DataFrame({"spend": [60] * 3}))


def test_a_failed_forecast_clears_the_previous_forecasts_results():
    df = make_raw_df(n=60, with_spend=True)
    forecaster = HorizonForecaster(
        df,
        date_column="date",
        target_column="orders",
        frequency="daily",
        independent_columns=["spend"],
        models=["catboost"],
    )
    forecaster.forecast(horizon=5, future_independent_data=pd.DataFrame({"spend": [60] * 5}))
    assert forecaster.forecast_ensemble is not None
    with pytest.raises(InvalidConfigError):
        forecaster.forecast(horizon=8, future_independent_data=pd.DataFrame({"spend": [60] * 5}))
    assert forecaster.forecast_ensemble is None
    assert forecaster.future_exog_used is None
    assert forecaster.forecast_errors == {}


# ---------- future-exogenous-data strategies ----------
# The strategy internals are unit-tested in tests/test_future_exog.py; these
# check the forecaster-level wiring (resolution, partial fill, exposure,
# validation) without needing a model fit.


def _resolve(forecaster, horizon, supplied=None):
    fut = forecaster._future_dates(horizon)
    return forecaster._resolve_future_exog(horizon, fut, supplied)


def test_cycle_strategy_reuses_last_n_periods_when_horizon_fits_within_history():
    df = make_raw_df(n=60, with_spend=True)  # spend = 50 + i, last values: ..., 107, 108, 109
    forecaster = HorizonForecaster(
        df,
        date_column="date",
        target_column="orders",
        frequency="daily",
        independent_columns=["spend"],
        models=["prophet"],
        future_exog_strategy="cycle",
    )
    models_df, _, _, _ = _resolve(forecaster, horizon=3)
    assert models_df["spend"].tolist() == [107, 108, 109]


def test_cycle_strategy_tiles_when_horizon_exceeds_history():
    df = make_raw_df(n=20, with_spend=True)  # spend = 50..69
    forecaster = HorizonForecaster(
        df,
        date_column="date",
        target_column="orders",
        frequency="daily",
        independent_columns=["spend"],
        models=["prophet"],
        future_exog_strategy="cycle",
    )
    models_df, _, _, _ = _resolve(forecaster, horizon=45)
    assert len(models_df) == 45
    assert not models_df["spend"].isna().any()


def test_default_strategy_is_carry_forward():
    df = make_raw_df(n=60, with_spend=True)  # spend last value = 109
    forecaster = HorizonForecaster(
        df,
        date_column="date",
        target_column="orders",
        frequency="daily",
        independent_columns=["spend"],
        models=["prophet"],
    )
    assert forecaster.future_exog_strategy == "carry_forward"
    models_df, used_df, used_default, generated = _resolve(forecaster, horizon=4)
    assert models_df["spend"].tolist() == [109] * 4  # held flat
    assert used_default is True and generated == ["spend"]
    assert list(used_df.columns) == ["ds", "spend"]  # exposed with a ds column


def test_partial_fill_keeps_supplied_generates_missing():
    df = make_raw_df(n=60, with_spend=True)
    df["promo"] = [i % 2 for i in range(len(df))]  # binary flag
    forecaster = HorizonForecaster(
        df,
        date_column="date",
        target_column="orders",
        frequency="daily",
        independent_columns=["spend", "promo"],
        models=["prophet"],
    )
    supplied = pd.DataFrame({"spend": [777.0] * 5})
    models_df, _, used_default, generated = _resolve(forecaster, horizon=5, supplied=supplied)
    assert models_df["spend"].tolist() == [777.0] * 5  # supplied kept
    assert models_df["promo"].tolist() == [0.0] * 5  # generated flag -> 0
    assert generated == ["promo"] and used_default is True


def test_explicit_event_columns_treated_as_flags():
    df = make_raw_df(n=60, with_spend=True)
    forecaster = HorizonForecaster(
        df,
        date_column="date",
        target_column="orders",
        frequency="daily",
        independent_columns=["spend"],
        models=["prophet"],
        event_columns=["spend"],
    )
    models_df, _, _, _ = _resolve(forecaster, horizon=3)
    assert models_df["spend"].tolist() == [0.0] * 3  # forced to 0 as a declared flag


def test_invalid_future_exog_params_raise():
    df = make_raw_df(n=60, with_spend=True)
    common = dict(
        date_column="date",
        target_column="orders",
        frequency="daily",
        independent_columns=["spend"],
        models=["prophet"],
    )
    with pytest.raises(InvalidConfigError):
        HorizonForecaster(df, future_exog_strategy="nope", **common)
    with pytest.raises(InvalidConfigError):
        HorizonForecaster(df, future_exog_aggregator="mode", **common)
    with pytest.raises(InvalidConfigError):
        HorizonForecaster(df, recent_average_window=0, **common)
    with pytest.raises(InvalidConfigError):
        HorizonForecaster(df, event_columns=["not_a_column"], **common)


# ---------- NaN target handling + failure isolation (robustness) ----------


def test_single_missing_target_cell_hard_fails_at_construction():
    df = make_raw_df(n=100)
    df["orders"] = df["orders"].astype(float)
    df.loc[df.index[3], "orders"] = float("nan")  # exactly ONE missing cell
    with pytest.raises(InvalidConfigError):
        HorizonForecaster(
            df, date_column="date", target_column="orders", frequency="daily", models=["prophet"]
        )


def test_single_missing_independent_cell_hard_fails_at_construction():
    df = make_raw_df(n=100, with_spend=True)
    df["spend"] = df["spend"].astype(float)
    df.loc[df.index[3], "spend"] = float("nan")  # one missing driver cell
    with pytest.raises(InvalidConfigError):
        HorizonForecaster(
            df,
            date_column="date",
            target_column="orders",
            frequency="daily",
            independent_columns=["spend"],
            models=["prophet"],
        )


def test_unexpected_model_error_is_isolated_not_fatal():
    """A model raising a raw (non-HorizonForecaster) error must be isolated in
    evaluate_errors, with the real cause preserved, while other models run."""
    df = make_raw_df(n=60)
    f = HorizonForecaster(
        df,
        date_column="date",
        target_column="orders",
        frequency="daily",
        models=["sarimax", "prophet"],
        model_params={"prophet": {"seasonality_mode": "additive"}},
    )
    real = f._evaluate_single_model

    def patched(model_name):
        if model_name == "prophet":
            raise ValueError("simulated raw crash")  # not a HorizonForecasterError
        return real(model_name)

    f._evaluate_single_model = patched
    results = f.evaluate()

    assert "sarimax" in results  # the healthy model still ran
    err = f.evaluate_errors["prophet"]
    assert isinstance(err, ModelConvergenceError)  # wrapped + isolated
    assert isinstance(err.original_exception, ValueError)  # real cause preserved


def test_forecast_warns_when_horizon_far_exceeds_history():
    df = make_raw_df(n=20, with_spend=True)
    forecaster = HorizonForecaster(
        df,
        date_column="date",
        target_column="orders",
        frequency="daily",
        independent_columns=["spend"],
        models=["prophet"],
    )
    with pytest.warns(UserWarning, match="Forecasting 45 periods from only 20 periods"):
        forecaster.forecast(horizon=45)


def test_forecast_does_not_warn_when_horizon_is_reasonable():
    """
    Isolates specifically the horizon-vs-history warning (see
    test_forecast_warns_when_horizon_far_exceeds_history above) from
    the unrelated "running evaluate() internally" warning --
    evaluate() is called first, outside the strict block, so its
    one-time warning doesn't get mistaken for a horizon warning that
    shouldn't have fired.
    """
    df = make_raw_df(n=60)
    forecaster = HorizonForecaster(
        df, date_column="date", target_column="orders", frequency="daily", models=["prophet"]
    )
    forecaster.evaluate()
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning becomes a test failure
        forecaster.forecast(horizon=10)  # well within 2x of 60 rows of history


def test_forecast_warns_once_horizon_exceeds_half_of_history():
    """
    Half the history is enough to flag. Forecasting 100 periods from 121 is
    the case that produced a real implausible forecast, and a rule that
    fired only above 2x the whole history would stay silent on it.
    """
    df = make_raw_df(n=60)
    forecaster = HorizonForecaster(
        df, date_column="date", target_column="orders", frequency="daily", models=["catboost"]
    )
    with pytest.warns(UserWarning, match="Forecasting 40 periods from only 60 periods"):
        forecaster.forecast(horizon=40)


def test_forecast_warns_when_future_independent_data_is_auto_generated():
    df = make_raw_df(n=60, with_spend=True)
    forecaster = HorizonForecaster(
        df,
        date_column="date",
        target_column="orders",
        frequency="daily",
        independent_columns=["spend"],
        models=["catboost"],
    )
    with pytest.warns(UserWarning, match="auto-generated"):
        forecaster.forecast(horizon=5)


def test_forecast_does_not_warn_about_exog_when_it_is_supplied():
    df = make_raw_df(n=60, with_spend=True)
    forecaster = HorizonForecaster(
        df,
        date_column="date",
        target_column="orders",
        frequency="daily",
        independent_columns=["spend"],
        models=["catboost"],
    )
    forecaster.evaluate()
    future_spend = pd.DataFrame({"spend": [200, 201, 202]})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        forecaster.forecast(horizon=3, future_independent_data=future_spend)
    assert not any("auto-generated" in str(w.message) for w in caught)


def test_warns_when_a_forecast_lands_far_above_history():
    df = make_raw_df(n=60)  # orders top out at 218
    forecaster = HorizonForecaster(
        df, date_column="date", target_column="orders", frequency="daily", models=["catboost"]
    )
    absurd = {
        "catboost": ForecastResult(
            model_name="catboost",
            forecast=pd.DataFrame(
                {"ds": pd.date_range("2026-04-01", periods=2), "yhat": [500.0, 900.0]}
            ),
            horizon=2,
            used_default_future_strategy=False,
        )
    }
    with pytest.warns(UserWarning, match="more than 50% above"):
        forecaster._warn_on_forecasts_far_outside_history(absurd)


def test_no_range_warning_for_a_normally_trending_forecast():
    """
    A trending series legitimately forecasts past its historical max --
    that must NOT warn, or the check would fire on every healthy trend.
    """
    df = make_raw_df(n=60)  # tops out at 218
    forecaster = HorizonForecaster(
        df, date_column="date", target_column="orders", frequency="daily", models=["catboost"]
    )
    reasonable = {
        "catboost": ForecastResult(
            model_name="catboost",
            forecast=pd.DataFrame(
                {"ds": pd.date_range("2026-04-01", periods=2), "yhat": [220.0, 232.0]}
            ),
            horizon=2,
            used_default_future_strategy=False,
        )
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        forecaster._warn_on_forecasts_far_outside_history(reasonable)
    assert not any("above the highest value" in str(w.message) for w in caught)


def test_grid_search_skips_failing_candidates_and_uses_the_survivors():
    """
    One unusable hyperparameter combination must not cost you the whole
    model when others fit fine -- it is skipped, with a warning naming how
    many were dropped.
    """
    df = make_raw_df(n=60)
    forecaster = HorizonForecaster(
        df,
        date_column="date",
        target_column="orders",
        frequency="daily",
        models=["sarimax"],
        # The split these tests were written against: SARIMAX needs 35
        # training rows, which the scaled-down defaults (33 of 60) do not leave.
        validation_size=16,
        test_size=8,
    )
    real = forecaster._fit_predict_mape
    calls = {"n": 0}

    def flaky(model_name, params, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] % 2 == 0:  # fail every other candidate
            raise ModelConvergenceError("forced candidate failure", model_name=model_name)
        return real(model_name, params, *args, **kwargs)

    forecaster._fit_predict_mape = flaky
    with pytest.warns(UserWarning, match="candidates failed"):
        result = forecaster.evaluate()["sarimax"]

    assert result.grid_search_summary  # the survivors were still used
    assert calls["n"] > len(result.grid_search_summary)  # some were genuinely skipped


def test_grid_search_fails_the_model_only_when_every_candidate_fails():
    df = make_raw_df(n=60)
    forecaster = HorizonForecaster(
        df,
        date_column="date",
        target_column="orders",
        frequency="daily",
        models=["sarimax"],
        # The split these tests were written against: SARIMAX needs 35
        # training rows, which the scaled-down defaults (33 of 60) do not leave.
        validation_size=16,
        test_size=8,
    )

    def always_fails(model_name, params, *args, **kwargs):
        raise ModelConvergenceError("forced candidate failure", model_name=model_name)

    forecaster._fit_predict_mape = always_fails
    with pytest.raises(AllModelsRejectedError):
        forecaster.evaluate()
    assert isinstance(forecaster.evaluate_errors["sarimax"], ModelConvergenceError)


def test_grid_starved_of_training_rows_falls_back_instead_of_failing_the_model():
    """
    A training window below the model's row floor fails every candidate,
    but that is a property of the WINDOW, not of any hyperparameter -- and
    it does not mean the model is unusable, because both later fits see
    more data. Here 20 daily rows split to a 12-row training window (under
    Prophet's floor of 14) while forecast() refits on all 20, which clears
    it. So selection defers to the first configured candidate and records
    an undefined validation MAPE, rather than refusing to forecast at all.
    """
    df = make_raw_df(n=20)
    forecaster = HorizonForecaster(
        df,
        date_column="date",
        target_column="orders",
        frequency="daily",
        models=["prophet"],
    )
    with pytest.warns(UserWarning, match="every hyperparameter candidate failed"):
        results = forecaster.forecast(horizon=5)

    assert "prophet" in results
    assert len(results["prophet"].forecast) == 5

    selection = forecaster._selection_cache["prophet"]
    assert math.isnan(selection.validation_mape.value)
    # Nothing was ranked, so there is no ranking to report.
    assert selection.grid_search_summary is None
    # A NaN score cannot be weighted, so the weighted rules must drop it --
    # while the trimmed rule, which needs no MAPE, keeps it.
    forecaster_all = HorizonForecaster(
        df,
        date_column="date",
        target_column="orders",
        frequency="daily",
        models=["prophet"],
        ensemble_method=None,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        forecaster_all.forecast(horizon=5)
    assert forecaster_all.forecast_ensemble["trimmed"].members == ["prophet"]
    assert "prophet" in forecaster_all.forecast_ensemble["inverse_mape"].rejected_models


def test_total_grid_failure_preserves_the_original_exception_type():
    """
    When every candidate fails the SAME way, the cause is not
    hyperparameter-specific -- it belongs to the data or the model. The
    original exception must be re-raised rather than relabelled, so its type
    and structured attributes survive for callers that branch on them.

    Wrapping every total failure in ModelConvergenceError instead would turn
    a plain "not enough rows" into a misleading convergence error and drop
    required_periods/available_periods on the floor.
    """
    df = make_raw_df(n=60)
    forecaster = HorizonForecaster(
        df,
        date_column="date",
        target_column="orders",
        frequency="daily",
        models=["sarimax"],
        # The split these tests were written against: SARIMAX needs 35
        # training rows, which the scaled-down defaults (33 of 60) do not leave.
        validation_size=16,
        test_size=8,
    )

    def always_insufficient(model_name, params, *args, **kwargs):
        raise InsufficientDataError(
            "forced", model_name=model_name, required_periods=14, available_periods=8
        )

    forecaster._fit_predict_mape = always_insufficient
    with pytest.raises(AllModelsRejectedError):
        forecaster.evaluate()

    error = forecaster.evaluate_errors["sarimax"]
    assert isinstance(error, InsufficientDataError)
    assert error.required_periods == 14  # structured attributes survived
    assert error.available_periods == 8


def test_mixed_grid_failures_become_an_aggregate_convergence_error():
    """
    The counterpart: when candidates fail in genuinely different ways no
    single type is representative, so the aggregate ModelConvergenceError is
    correct -- and its message must name every distinct cause.
    """
    df = make_raw_df(n=60)
    forecaster = HorizonForecaster(
        df,
        date_column="date",
        target_column="orders",
        frequency="daily",
        models=["sarimax"],
        # The split these tests were written against: SARIMAX needs 35
        # training rows, which the scaled-down defaults (33 of 60) do not leave.
        validation_size=16,
        test_size=8,
    )
    calls = {"n": 0}

    def alternating_failures(model_name, params, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] % 2 == 0:
            raise ModelPredictionError("forced prediction failure", model_name=model_name)
        raise InsufficientDataError("forced data failure", model_name=model_name)

    forecaster._fit_predict_mape = alternating_failures
    with pytest.raises(AllModelsRejectedError):
        forecaster.evaluate()

    error = forecaster.evaluate_errors["sarimax"]
    assert isinstance(error, ModelConvergenceError)
    assert "InsufficientDataError" in str(error)
    assert "ModelPredictionError" in str(error)


def test_forecast_weekly_frequency_produces_correct_future_dates():
    df = make_raw_df(n=40, freq="7D")
    forecaster = HorizonForecaster(
        df, date_column="date", target_column="orders", frequency="weekly", models=["prophet"]
    )
    result = forecaster.forecast(horizon=3)["prophet"]
    last_train_date = df["date"].max()
    expected_first_future_date = last_train_date + pd.Timedelta(days=7)
    assert result.forecast["ds"].iloc[0] == expected_first_future_date


def test_forecast_all_models_fail_raises_all_models_rejected_error():
    df = make_raw_df(n=10)
    forecaster = HorizonForecaster(
        df, date_column="date", target_column="orders", frequency="daily", models=["prophet"]
    )
    with pytest.raises(AllModelsRejectedError):
        forecaster.forecast(horizon=5)
    assert isinstance(forecaster.forecast_errors["prophet"], InsufficientDataError)


def test_forecast_partial_failure_returns_successful_models():
    """
    See test_evaluate_partial_failure_returns_successful_models_and_records_errors
    for why depth=-1 (a genuine model-family failure) is used here rather
    than a data-size coincidence.
    """
    df = make_raw_df(n=60)
    forecaster = HorizonForecaster(
        df,
        date_column="date",
        target_column="orders",
        frequency="daily",
        models=["prophet", "catboost"],  # focused 2-model partial-failure case
        model_params={"prophet": {"seasonality_mode": "additive"}, "catboost": {"depth": -1}},
    )
    results = forecaster.forecast(horizon=3)
    assert set(results.keys()) == {"prophet"}
    assert set(forecaster.forecast_errors.keys()) == {"catboost"}


# ---------- ensemble ----------


def test_constructor_defaults_for_ensemble_params():
    df = make_raw_df()
    forecaster = HorizonForecaster(
        df, date_column="date", target_column="orders", frequency="daily"
    )
    assert forecaster.mape_threshold == 40.0
    assert forecaster.ensemble_methods == ["trimmed"]
    assert forecaster.evaluate_ensemble is None
    assert forecaster.forecast_ensemble is None


def test_constructor_validates_mape_threshold():
    df = make_raw_df()
    with pytest.raises(InvalidConfigError):
        HorizonForecaster(
            df, date_column="date", target_column="orders", frequency="daily", mape_threshold=150
        )


def test_constructor_validates_ensemble_method():
    df = make_raw_df()
    with pytest.raises(InvalidConfigError):
        HorizonForecaster(
            df,
            date_column="date",
            target_column="orders",
            frequency="daily",
            ensemble_method="bogus",
        )


def test_evaluate_populates_ensemble_with_default_permissive_threshold():
    df = make_noisy_raw_df()
    forecaster = HorizonForecaster(
        df,
        date_column="date",
        target_column="orders",
        frequency="daily",
        models=["prophet", "catboost"],  # 2-model ensemble weighting math
        model_params={
            "prophet": {"seasonality_mode": "additive"},
            "catboost": {"iterations": 100, "depth": 3},
        },
    )
    forecaster.evaluate()
    ensemble = forecaster.evaluate_ensemble["trimmed"]
    assert ensemble is not None
    assert ensemble.combined is not None
    # Two members: trimming is undefined, so it degrades to a plain mean
    # and reports that rather than silently calling itself trimmed.
    assert set(ensemble.members) == {"prophet", "catboost"}
    assert ensemble.rule_applied == "mean"
    assert ensemble.degraded is True
    assert ensemble.weights is None
    assert ensemble.mape is not None  # actuals are available in the evaluate() context
    # The ensemble's own calibrated band, never the members' bands averaged.
    assert set(ensemble.combined.columns) <= {"ds", "yhat", "yhat_lower", "yhat_upper"}
    assert ("yhat_lower" in ensemble.combined) == (ensemble.band.width is not None)


def test_evaluate_ensemble_uses_configured_ensemble_method():
    df = make_noisy_raw_df()
    forecaster = HorizonForecaster(
        df,
        date_column="date",
        target_column="orders",
        frequency="daily",
        models=["prophet", "catboost"],  # 2-model equal-weighting check
        model_params={
            "prophet": {"seasonality_mode": "additive"},
            "catboost": {"iterations": 100, "depth": 3},
        },
        ensemble_method="equal",
    )
    forecaster.evaluate()
    assert list(forecaster.evaluate_ensemble) == ["equal"]
    assert forecaster.evaluate_ensemble["equal"].weights["prophet"] == pytest.approx(0.5)
    assert forecaster.evaluate_ensemble["equal"].weights["catboost"] == pytest.approx(0.5)


def test_evaluate_ensemble_rejects_model_above_threshold():
    """
    A mape_threshold of 0.5 rejects any model
    scoring above 0.5% MAPE from the ensemble -- which, on the noisy
    dataset, a real fit does -- but does NOT affect evaluate()'s own
    return value: the per-model result is still returned, only
    evaluate_ensemble reflects the rejection. This is the key
    distinction from AllModelsRejectedError (see evaluation/ensemble.py's
    module docstring). The noisy dataset is used specifically so the real
    MAPE lands above 0.5% (a clean fit could score under 0.5% and slip
    past this deliberately low threshold).
    """
    df = make_noisy_raw_df()
    forecaster = HorizonForecaster(
        df,
        date_column="date",
        target_column="orders",
        frequency="daily",
        models=["prophet"],
        model_params={"prophet": {"seasonality_mode": "additive"}},
        mape_threshold=0.5,
    )
    results = forecaster.evaluate()
    assert "prophet" in results
    ensemble = forecaster.evaluate_ensemble["trimmed"]
    assert ensemble.combined is None
    assert ensemble.members == []
    assert ensemble.rule_applied == "none"
    # None, not {} -- there is no weighting to report either way now.
    assert ensemble.weights is None
    # Screening must read the VALIDATION score, never the reported one.
    # Without this the wiring could be switched back to the test window --
    # reintroducing exactly the leak the three-way split exists to close --
    # and every other test would still pass.
    rejection = ensemble.rejected_models["prophet"]
    assert rejection.mape_value == results["prophet"].validation_mape.value
    assert rejection.mape_value != results["prophet"].mape.value
    assert forecaster.evaluate_ensemble["trimmed"].mape is None
    assert "prophet" in forecaster.evaluate_ensemble["trimmed"].rejected_models


def test_forecast_screens_on_validation_mape_without_running_a_full_evaluate():
    """
    forecast() needs a validation MAPE to screen and weight, but it does
    NOT need evaluate(): selection has to happen anyway to know which
    hyperparameters to fit with, and that produces the score for free.
    Running the whole of evaluate() internally would also pay for a
    test-window prediction that is then thrown away.
    """
    df = make_noisy_raw_df()
    forecaster = HorizonForecaster(
        df,
        date_column="date",
        target_column="orders",
        frequency="daily",
        model_params={
            "prophet": {"seasonality_mode": "additive"},
            "catboost": {"iterations": 100, "depth": 3},
            "bayesianpymc": _FAST_PYMC,
        },
    )
    assert forecaster._selection_cache == {}
    forecaster.forecast(horizon=5)

    assert forecaster._selection_cache  # populated by selection, not by evaluate()
    assert forecaster.evaluate_ensemble is None  # evaluate() was never run
    assert forecaster.forecast_ensemble is not None
    combined = forecaster.forecast_ensemble["trimmed"].combined
    assert {"ds", "yhat"} <= set(combined.columns)
    assert len(forecaster.forecast_ensemble["trimmed"].combined) == 5
    assert forecaster.forecast_ensemble["trimmed"].mape is None  # no actuals exist for the future


def test_forecast_does_not_rewarn_or_recompute_evaluate_on_second_call():
    df = make_noisy_raw_df()
    forecaster = HorizonForecaster(
        df,
        date_column="date",
        target_column="orders",
        frequency="daily",
        model_params={
            "prophet": {"seasonality_mode": "additive"},
            "catboost": {"iterations": 100, "depth": 3},
            "bayesianpymc": _FAST_PYMC,
        },
    )
    forecaster.forecast(horizon=5)
    selection_after_first_call = dict(forecaster._selection_cache)

    forecaster.forecast(horizon=8)

    # Same selection objects, not re-searched.
    assert forecaster._selection_cache == selection_after_first_call
    for name, selection in selection_after_first_call.items():
        assert forecaster._selection_cache[name] is selection


def test_forecast_skips_internal_evaluate_when_already_called_explicitly():
    df = make_noisy_raw_df()
    forecaster = HorizonForecaster(
        df,
        date_column="date",
        target_column="orders",
        frequency="daily",
        model_params={
            "prophet": {"seasonality_mode": "additive"},
            "catboost": {"iterations": 100, "depth": 3},
            "bayesianpymc": _FAST_PYMC,
        },
    )
    forecaster.evaluate()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # Bayesian PyMC's own notes describe the forecast fit itself (90 days
        # has no annual terms; _FAST_PYMC's 40 draws do not converge), so they
        # are expected here; nothing may warn about selection running again.
        warnings.filterwarnings("ignore", message="bayesianpymc: ")
        forecaster.forecast(horizon=5)  # must not warn -- evaluate() already ran


def test_forecast_succeeds_when_the_model_cannot_be_scored_on_validation():
    """
    n=16 -> 9 training rows, below the daily floor of 14, so the model
    cannot be scored on validation; forecast()'s own full-data refit uses
    all 16 rows, which clears the floor. This confirms forecast() still
    attempts (and succeeds at) its own fit when selection has no
    validation score -- a model failing on the smaller training window
    doesn't block forecast() from succeeding with more data.
    """
    df = make_raw_df(n=16)
    forecaster = HorizonForecaster(
        df,
        date_column="date",
        target_column="orders",
        frequency="daily",
        models=["prophet"],
        model_params={"prophet": {"seasonality_mode": "additive"}},
    )
    with pytest.warns(UserWarning, match="could not be scored on the validation window"):
        results = forecaster.forecast(horizon=3)

    assert "prophet" in results
    # Scored NaN rather than failed: explicit params need no selection, so
    # the model still forecasts -- it just carries no validation MAPE.
    import math

    assert math.isnan(forecaster._selection_cache["prophet"].validation_mape.value)


# ---------- drivers that never change inside one fit window ----------
# A driver can vary over the full history (so it survives the zero-variance
# check) yet be one value throughout a fit window -- a price change that
# first appears in validation. Left in, NeuralProphet crashes on it and
# Prophet invents an effect, so it is left out of that window's fit for every
# model, with a warning naming the score it affects.


def _window_df(change_row, before=0.0, after=1.0, n=104):
    """Weekly data, split 65 train / 26 validation / 13 test by _window_forecaster."""
    import numpy as np

    rng = np.random.default_rng(3)
    drv = np.where(np.arange(n) >= change_row, after, before)
    spend = rng.uniform(800, 1200, n).round()
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-08-04", periods=n, freq="W-SUN"),
            "sales": 3000 + 4 * np.arange(n) + 0.5 * spend - 400 * (drv != before),
            "spend": spend,
            "drv": drv,
        }
    )


def _window_forecaster(df, models=("catboost",)):
    return HorizonForecaster(
        df,
        date_column="date",
        target_column="sales",
        frequency="weekly",
        independent_columns=["spend", "drv"],
        models=list(models),
        validation_size=26,
        test_size=13,
    )


def test_driver_constant_in_training_is_left_out_of_the_selection_fit():
    forecaster = _window_forecaster(_window_df(change_row=75))  # changes in validation
    with pytest.warns(UserWarning, match=r"'drv' never changes in the training window") as w:
        train, validation_features, _ = forecaster._prepare_split("catboost")
    message = str(w[0].message)
    assert "always 0 there and first changes on 2026-01-11" in message
    assert "validation MAPE" in message
    assert "drv" not in train.columns and "drv" not in validation_features.columns
    assert "spend" in train.columns

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # varies in train+validation: no warning
        refit_rows, _, _ = forecaster._prepare_refit("catboost")
    assert "drv" in refit_rows.columns


def test_driver_changing_only_in_test_warns_for_the_test_score_only():
    forecaster = _window_forecaster(_window_df(change_row=95))  # changes in test
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # validation is unaffected: no warning
        train, _, _ = forecaster._prepare_split("catboost")
    assert "drv" not in train.columns
    with pytest.warns(UserWarning, match=r"train\+validation window used for the test score"):
        refit_rows, _, _ = forecaster._prepare_refit("catboost")
    assert "drv" not in refit_rows.columns


def test_window_constant_driver_warns_once_per_window_across_models():
    forecaster = _window_forecaster(_window_df(change_row=75), models=("catboost", "sarimax"))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        forecaster._prepare_split("catboost")
        forecaster._prepare_split("sarimax")
    assert sum("never changes" in str(w.message) for w in caught) == 1


def test_window_constant_driver_with_a_nonzero_value_is_also_left_out():
    forecaster = _window_forecaster(_window_df(change_row=75, before=5.0, after=50.0))
    with pytest.warns(UserWarning, match="always 5 there"):
        train, _, _ = forecaster._prepare_split("catboost")
    assert "drv" not in train.columns


def test_drivers_that_vary_in_training_are_kept_without_a_warning():
    forecaster = _window_forecaster(_window_df(change_row=30))  # changes inside train
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        train, _, _ = forecaster._prepare_split("catboost")
    assert {"spend", "drv"} <= set(train.columns)


def test_neuralprophet_runs_when_a_driver_first_changes_in_validation():
    # The reported failure: NeuralProphet dropped the constant column at fit
    # and rejected it at prediction ("Unexpected column ... in data").
    forecaster = HorizonForecaster(
        _window_df(change_row=75),
        date_column="date",
        target_column="sales",
        frequency="weekly",
        independent_columns=["spend", "drv"],
        models=["neuralprophet"],
        model_params={"neuralprophet": {"learning_rate": 0.03, "epochs": 30}},
        validation_size=26,
        test_size=13,
    )
    with pytest.warns(UserWarning, match="never changes in the training window"):
        results = forecaster.evaluate()
    assert forecaster.evaluate_errors == {}
    assert not math.isnan(results["neuralprophet"].validation_mape.value)


# ---------- event flags and on/off text (item: flags and promo weeks) ----------


def _flag_df(values):
    df = make_raw_df(n=60, with_spend=True)
    df["promo"] = values
    return df


def test_constructor_reads_yes_no_text_as_a_flag_without_changing_the_callers_data():
    df = _flag_df(["Yes" if i % 5 == 0 else "No" for i in range(60)])
    forecaster = HorizonForecaster(
        df, "date", "orders", "daily", independent_columns=["spend", "promo"], models=["catboost"]
    )
    assert set(forecaster.raw_df["promo"]) == {0, 1}
    assert df["promo"].iloc[0] == "Yes"


def test_constructor_rejects_an_unclear_flag_pair_with_a_specific_message():
    df = _flag_df(["Promo" if i % 5 == 0 else "Off-season" for i in range(60)])
    with pytest.raises(InvalidConfigError, match="can't tell which one means"):
        HorizonForecaster(
            df,
            "date",
            "orders",
            "daily",
            independent_columns=["spend", "promo"],
            models=["catboost"],
        )


def test_forecast_warns_when_carry_forward_holds_a_last_day_spike():
    df = make_raw_df(n=60, with_spend=True)
    df.loc[59, "spend"] = 500  # usual level ~80-108
    forecaster = HorizonForecaster(
        df, "date", "orders", "daily", independent_columns=["spend"], models=["catboost"]
    )
    with pytest.warns(UserWarning, match="last day of history .* looks unusual"):
        forecaster.forecast(horizon=5)


# ---------- input checks: values, column roles and arguments ----------
#
# Unchecked, each case here would run to completion with a meaningless
# result, or fail as a raw Python error deep inside the models. All are
# constructor checks, so no model is fitted.


def _construct(df, **kwargs):
    args = dict(
        date_column="date",
        target_column="orders",
        frequency="daily",
        validation_size=16,
        test_size=8,
        models=["sarimax"],
    )
    args.update(kwargs)
    return HorizonForecaster(df, **args)


@pytest.mark.parametrize("bad", [math.inf, -math.inf])
def test_constructor_rejects_infinite_target_naming_the_date(bad):
    df = make_raw_df(with_spend=True)
    df["orders"] = df["orders"].astype(float)
    df.loc[10, "orders"] = bad
    with pytest.raises(InvalidConfigError, match="'orders' is infinite for 2026-01-11") as info:
        _construct(df, independent_columns=["spend"])
    assert info.value.provided_value == {"orders": 1}


@pytest.mark.parametrize("bad", [math.inf, "inf", "-Infinity"])
def test_constructor_rejects_infinite_driver(bad):
    df = make_raw_df(with_spend=True)
    df["spend"] = df["spend"].astype(object)
    df.loc[10, "spend"] = bad
    with pytest.raises(InvalidConfigError, match="'spend' is infinite for 2026-01-11"):
        _construct(df, independent_columns=["spend"])


def test_constructor_rejects_a_driver_that_is_infinite_everywhere():
    # Unchecked, it would be dropped as "zero variance" without a word.
    df = make_raw_df(with_spend=True)
    df["spend"] = math.inf
    with pytest.raises(InvalidConfigError, match="'spend' is infinite in every row"):
        _construct(df, independent_columns=["spend"])


@pytest.mark.parametrize("blank", [float("nan"), None, ""])
def test_constructor_rejects_an_all_blank_driver_like_a_single_blank(blank):
    # Like a single blank; unchecked, an all-blank driver would count as
    # "constant" and vanish from the run.
    df = make_raw_df(with_spend=True)
    df["spend"] = blank
    with pytest.raises(InvalidConfigError, match="'spend' is blank in every row"):
        _construct(df, independent_columns=["spend"])


def test_missing_value_message_names_the_first_dates_and_counts_the_rest():
    df = make_raw_df(with_spend=True)
    df.loc[[3, 4, 5, 6, 7], "spend"] = None
    df.loc[20, "orders"] = None
    with pytest.raises(InvalidConfigError) as info:
        _construct(df, independent_columns=["spend"])
    message = str(info.value)
    assert "'orders' is blank for 2026-01-21" in message
    assert "'spend' is blank for 2026-01-04, 2026-01-05, 2026-01-06 and 2 more" in message
    assert info.value.provided_value == {"orders": 1, "spend": 5}


def test_constructor_warns_when_a_declared_driver_never_changes():
    df = make_raw_df(with_spend=True)
    df["promo"] = 0
    with pytest.warns(UserWarning, match=r"\['promo'\] never change.*'promo' is always 0"):
        forecaster = _construct(df, independent_columns=["spend", "promo"])
    assert forecaster.dropped_zero_variance_columns == ["promo"]
    assert forecaster.independent_columns == ["spend"]


def test_constant_or_blank_columns_that_are_not_drivers_are_left_alone():
    df = make_raw_df(with_spend=True)
    df["notes"] = None
    df["const"] = 1
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no "never change" warning for unused columns
        forecaster = _construct(df, independent_columns=["spend"])
    assert forecaster.dropped_zero_variance_columns == []


def test_constructor_reads_a_target_written_as_text_numbers():
    df = make_raw_df()
    df["orders"] = df["orders"].astype(str)
    forecaster = _construct(df)
    assert forecaster.raw_df["orders"].dtype.kind in "if"


def test_constructor_names_thousands_separators_in_the_target():
    df = make_raw_df()
    df["orders"] = [f"{1000 + i:,}" for i in range(len(df))]
    with pytest.raises(InvalidConfigError, match=r"target column 'orders'.*'1,000'.*2026-01-01"):
        _construct(df)


def test_constructor_rejects_a_complex_target():
    df = make_raw_df()
    df["orders"] = df["orders"].astype(complex)
    with pytest.raises(InvalidConfigError, match="complex numbers"):
        _construct(df)


def test_constructor_rejects_a_driver_holding_dates():
    df = make_raw_df()
    df["launch"] = df["date"]
    with pytest.raises(InvalidConfigError, match="'launch' holds dates"):
        _construct(df, independent_columns=["launch"])


def test_constructor_rejects_the_date_as_target():
    with pytest.raises(InvalidConfigError, match="target_column and date_column are both"):
        _construct(make_raw_df(), target_column="date")


def test_constructor_rejects_the_target_as_a_driver():
    with pytest.raises(InvalidConfigError, match="'orders' is the target column"):
        _construct(make_raw_df(), independent_columns=["orders"])


def test_constructor_rejects_the_date_as_a_driver():
    with pytest.raises(InvalidConfigError, match="'date' is the date column"):
        _construct(make_raw_df(), independent_columns=["date"])


def test_constructor_rejects_a_driver_listed_twice():
    with pytest.raises(InvalidConfigError, match=r"lists \['spend'\] more than once"):
        _construct(make_raw_df(with_spend=True), independent_columns=["spend", "spend"])


def test_constructor_rejects_a_used_column_name_that_appears_twice():
    df = make_raw_df(with_spend=True)
    df = pd.concat([df, df[["spend"]]], axis=1)
    with pytest.raises(InvalidConfigError, match=r"more than one column named \['spend'\]"):
        _construct(df, independent_columns=["spend"])


def test_constructor_rejects_multi_level_column_headers():
    df = make_raw_df()
    df.columns = pd.MultiIndex.from_tuples([(c, "x") for c in df.columns])
    with pytest.raises(InvalidConfigError, match="2 levels of column headers"):
        _construct(df)


@pytest.mark.parametrize("raw", [[1, 2, 3], make_raw_df().to_numpy(), make_raw_df()["orders"]])
def test_constructor_rejects_a_raw_df_that_is_not_a_dataframe(raw):
    with pytest.raises(InvalidConfigError, match="raw_df must be a pandas DataFrame"):
        _construct(raw)


def test_constructor_rejects_an_empty_raw_df():
    with pytest.raises(InvalidConfigError, match="raw_df has no rows"):
        _construct(make_raw_df().iloc[:0])


@pytest.mark.parametrize("freq", [None, 7])
def test_constructor_rejects_a_non_text_frequency(freq):
    with pytest.raises(UnsupportedFrequencyError, match="frequency must be one of"):
        _construct(make_raw_df(), frequency=freq)


def test_constructor_rejects_a_text_mape_threshold():
    with pytest.raises(InvalidConfigError, match="mape_threshold must be a number"):
        _construct(make_raw_df(), mape_threshold="40")


def test_constructor_rejects_a_non_text_date_format():
    with pytest.raises(InvalidConfigError, match="date_format must be a strptime format"):
        _construct(make_raw_df(), date_format=5)


def test_single_names_are_read_as_one_item_lists():
    df = make_raw_df(with_spend=True)
    df["promo"] = [i % 7 == 0 for i in range(len(df))]
    forecaster = _construct(df, models="sarimax", independent_columns="spend", event_columns=None)
    assert forecaster.models == ["sarimax"]
    assert forecaster.independent_columns == ["spend"]
    forecaster = _construct(df, independent_columns=["spend", "promo"], event_columns="promo")
    assert forecaster.event_columns == ["promo"]


def test_a_model_listed_twice_is_run_once():
    assert _construct(make_raw_df(), models=["sarimax", "sarimax"]).models == ["sarimax"]


@pytest.mark.parametrize(
    "params, match",
    [
        (["prophet"], "model_params must be a dict"),
        ({"sarimax": 3}, r"model_params\['sarimax'\] must be a dict"),
        ({"neural_prophet": {"epochs": 3}}, "did you mean 'neuralprophet'"),
        ({"sarimax": {"seasonal_periods": 7}}, "did you mean 'seasonal_period'"),
        ({"bayesianpymc": {"draw": 10}}, "did you mean 'draws'"),
    ],
)
def test_constructor_rejects_bad_model_params(params, match):
    with pytest.raises(InvalidConfigError, match=match):
        _construct(make_raw_df(), model_params=params)


def test_model_params_for_a_known_model_that_was_not_requested_are_allowed():
    _construct(make_raw_df(), model_params={"prophet": {"seasonality_mode": "additive"}})


@pytest.mark.parametrize("model_name", ["sarimax", "bayesianpymc"])
def test_accepted_params_match_what_the_wrapper_reads(model_name):
    """
    accepted_params is the list the constructor checks model_params against,
    so it must name exactly the keys the wrapper's __init__ reads: a key read
    but not listed would be rejected, a key listed but not read would be
    accepted and ignored.
    """

    class _Recorder(dict):
        def __init__(self):
            super().__init__()
            self.read = set()

        def __bool__(self):  # BaseModel keeps `params or {}`; stay truthy
            return True

        def get(self, key, default=None):
            self.read.add(key)
            return default

    wrapper_cls = MODEL_REGISTRY[model_name]["wrapper"]
    recorder = _Recorder()
    wrapper_cls(frequency="daily", params=recorder)
    assert recorder.read == set(wrapper_cls.accepted_params)


@pytest.mark.parametrize("model_name", ["sarimax", "bayesianpymc"])
def test_default_config_only_uses_accepted_params(model_name):
    from horizon_forecaster.forecaster import _default_params

    accepted = MODEL_REGISTRY[model_name]["wrapper"].accepted_params
    for frequency, config in _default_params()[model_name].items():
        keys = set(config.get("base_hyperparameters", {}))
        for candidate in config.get("hyperparameter_grid", []):
            keys |= set(candidate)
        assert keys <= accepted, (frequency, keys - accepted)


def test_forecast_rejects_a_horizon_past_the_last_representable_date():
    forecaster = _construct(make_raw_df())
    with pytest.raises(InvalidConfigError, match="past 2262-04-11"):
        forecaster.forecast(horizon=10**7)


# ---------- models that drop out, and ensembles that lose members ----------
#
# Without the warnings, each of these is visible only by reading
# evaluate_errors / forecast_errors or the EnsembleResult fields.

_QUICK = {"prophet": {"seasonality_mode": "additive"}, "catboost": {"iterations": 50, "depth": 3}}


def _two_model_forecaster(**kwargs):
    args = dict(
        date_column="date",
        target_column="orders",
        frequency="daily",
        validation_size=16,
        test_size=8,
        models=["prophet", "catboost"],
        model_params=dict(_QUICK),
    )
    args.update(kwargs)
    return HorizonForecaster(make_noisy_raw_df(), **args)


def _messages(caught):
    return [str(w.message) for w in caught]


def test_a_failed_model_and_the_thinner_ensemble_are_both_warned_about():
    forecaster = _two_model_forecaster(
        model_params={"prophet": _QUICK["prophet"], "catboost": {"depth": 20}}
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        forecaster.evaluate()
        forecaster.forecast(horizon=5)
    messages = _messages(caught)
    for stage in ("evaluate", "forecast"):
        failed = [
            m for m in messages if m.startswith(f"1 of the 2 requested models failed in {stage}()")
        ]
        assert failed and "catboost (ModelConvergenceError" in failed[0]
        assert "Maximum tree depth" in failed[0]  # the cause, not just "failed to fit"
        assert (
            f"The {stage}() ensemble lost models: 'trimmed' is just prophet (catboost failed)."
            in messages
        )


def test_an_ensemble_with_no_members_gets_its_own_warning():
    forecaster = _two_model_forecaster(mape_threshold=0.01)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        forecaster.evaluate()
    empty = [
        m for m in _messages(caught) if m.startswith("No ensemble could be formed in evaluate()")
    ]
    assert empty
    assert "every model's validation MAPE is above mape_threshold 0.01: prophet" in empty[0]
    assert forecaster.evaluate_ensemble["trimmed"].combined is None


def test_a_model_left_out_by_the_threshold_is_named_with_its_score():
    forecaster = _two_model_forecaster(ensemble_method="inverse_mape")
    scores = {m: r.validation_mape.value for m, r in forecaster.evaluate().items()}
    worse = max(scores, key=scores.get)
    threshold = (min(scores.values()) + max(scores.values())) / 2
    forecaster = _two_model_forecaster(ensemble_method="inverse_mape", mape_threshold=threshold)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        forecaster.evaluate()
    lost = [m for m in _messages(caught) if m.startswith("The evaluate() ensemble lost models")]
    assert (
        lost and f"{worse}'s validation MAPE {scores[worse]:.2f} is above mape_threshold" in lost[0]
    )


def test_no_dropout_warning_when_every_requested_model_is_in_the_ensemble():
    # Two models always give a plain mean instead of a trimmed one: that is
    # what was asked for, not a loss, so it does not warn.
    forecaster = _two_model_forecaster()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        forecaster.evaluate()
        forecaster.forecast(horizon=5)
    assert forecaster.evaluate_ensemble["trimmed"].degraded
    noisy = [
        m for m in _messages(caught) if "failed in" in m or "lost models" in m or "No ensemble" in m
    ]
    assert noisy == []


def test_a_model_that_predicts_nan_is_treated_as_failed(monkeypatch):
    from horizon_forecaster.models.catboost_wrapper import CatBoostWrapper

    real_predict = CatBoostWrapper.predict

    def nan_predict(self, horizon, future_exog=None):
        out = real_predict(self, horizon, future_exog)
        out["yhat"] = float("nan")
        return out

    monkeypatch.setattr(CatBoostWrapper, "predict", nan_predict)
    forecaster = _two_model_forecaster()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = forecaster.evaluate()
        forecaster.forecast(horizon=5)
    assert "catboost" not in results
    assert isinstance(forecaster.evaluate_errors["catboost"], ModelPredictionError)
    assert "not finite numbers (NaN or inf)" in str(forecaster.evaluate_errors["catboost"])
    assert "catboost" in forecaster.forecast_errors
    assert forecaster.evaluate_ensemble["trimmed"].members == ["prophet"]


def test_sarimax_is_left_out_with_a_warning_when_the_target_goes_negative():
    df = make_raw_df()
    df.loc[[5, 6, 7], "orders"] = -20
    with pytest.warns(
        UserWarning,
        match="SARIMAX is left out of this run: the target 'orders' is negative in 3 of 60 rows",
    ):
        forecaster = _construct(df, models=["sarimax", "catboost"])
    assert forecaster.models == ["catboost"]
    assert "negative in 3 of 60 rows" in forecaster.skipped_models["sarimax"]


def test_sarimax_alone_on_a_negative_target_is_refused():
    df = make_raw_df()
    df["orders"] = -df["orders"]
    with pytest.raises(InvalidConfigError, match="SARIMAX cannot forecast this series"):
        _construct(df, models=["sarimax"])


def test_skipped_models_is_empty_when_nothing_is_skipped():
    assert _construct(make_raw_df(), models=["sarimax"]).skipped_models == {}


@pytest.mark.parametrize("settings", [{"epochs": 20}, {"learning_rate": 0.03}, {}])
def test_neuralprophet_model_params_must_pin_learning_rate_and_epochs(settings):
    with pytest.raises(InvalidConfigError, match="must set both learning_rate and epochs"):
        _construct(make_raw_df(), model_params={"neuralprophet": settings})


# ---------- selection metric, seasonal-naive baseline, error correlation ----------


def _with_zeros(share, n=90, seed=3):
    """make_noisy_raw_df() with `share` of its days set to 0, spread at random."""
    import numpy as np

    df = make_noisy_raw_df(n=n)
    rng = np.random.default_rng(seed)
    zero_rows = rng.choice(n, size=int(round(share * n)), replace=False)
    df.loc[zero_rows, "orders"] = 0.0
    return df


def test_constructor_rejects_an_unknown_selection_metric():
    with pytest.raises(InvalidConfigError, match="selection_metric must be one of"):
        _construct(make_raw_df(), selection_metric="rmse")


def test_selection_metric_stays_mape_on_an_ordinary_series_without_a_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        forecaster = _construct(make_noisy_raw_df())
    assert forecaster.selection_metric == "mape"
    assert not [m for m in _messages(caught) if "use WAPE instead of MAPE" in m]


def test_selection_metric_switches_to_wape_at_ten_percent_zeros_with_a_warning():
    with pytest.warns(UserWarning, match=r"is 0 on 20% of days \(MAPE skips every zero actual\)"):
        forecaster = _construct(_with_zeros(0.2))
    assert forecaster.selection_metric == "wape"


def test_selection_metric_stays_mape_just_under_ten_percent_zeros():
    assert _construct(_with_zeros(0.09)).selection_metric == "mape"


def test_selection_metric_switches_to_wape_on_a_negative_target():
    df = make_noisy_raw_df()
    df.loc[[3, 4], "orders"] = -5.0
    with pytest.warns(UserWarning, match="is negative in 2 of 90 rows"):
        forecaster = _construct(df, models=["catboost"])
    assert forecaster.selection_metric == "wape"


def test_an_explicit_selection_metric_is_kept_without_a_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        kept = _construct(_with_zeros(0.2), selection_metric="mape")
        forced = _construct(make_noisy_raw_df(), selection_metric="wape")
    assert kept.selection_metric == "mape"
    assert forced.selection_metric == "wape"
    assert not [m for m in _messages(caught) if "use WAPE instead of MAPE" in m]


def test_a_daily_target_that_is_mostly_zero_is_refused_with_the_weekly_share():
    df = make_noisy_raw_df(n=84)  # 2026-01-01 is a Thursday: 11 complete Mon-Sun weeks
    df.loc[df.index % 7 != 0, "orders"] = 0.0
    with pytest.raises(InvalidConfigError) as caught:
        _construct(df, selection_metric="mape")
    message = str(caught.value)
    assert "is 0 on 86% of days (72 of 84)" in message
    assert "0 in 0% of 11 complete weeks" in message
    assert "frequency='weekly'" in message


def test_a_weekly_target_that_is_mostly_zero_is_refused_without_the_weekly_hint():
    df = make_noisy_raw_df(n=60, freq="W-MON")
    df.loc[df.index % 2 == 0, "orders"] = 0.0
    with pytest.raises(InvalidConfigError, match="is 0 on 50% of weeks") as caught:
        _construct(df, frequency="weekly", validation_size=8, test_size=4)
    assert "nothing coarser than weekly" in str(caught.value)


def test_selection_on_wape_drives_the_grid_search_and_the_screen():
    forecaster = HorizonForecaster(
        _with_zeros(0.2),
        date_column="date",
        target_column="orders",
        frequency="daily",
        validation_size=16,
        test_size=8,
        models=["prophet"],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = forecaster.evaluate()["prophet"]
    summary = result.grid_search_summary
    assert all({"params", "mape", "wape"} <= set(c) for c in summary)
    finite = [c["wape"] for c in summary if c["wape"] == c["wape"]]
    assert finite == sorted(finite)  # ranked on WAPE, the selection metric
    assert math.isclose(result.validation_wape, finite[0])
    assert result.wape == result.wape and result.validation_wape == result.validation_wape


def test_evaluate_reports_wape_alongside_mape():
    forecaster = _two_model_forecaster()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = forecaster.evaluate()
    for result in results.values():
        y, yhat = result.test_predictions["y"], result.test_predictions["yhat"]
        expected = (y - yhat).abs().sum() / y.abs().sum() * 100
        assert math.isclose(result.wape, expected, rel_tol=1e-9)
        assert result.validation_wape > 0
    assert forecaster.evaluate_ensemble["trimmed"].wape is not None


def test_evaluate_builds_the_seasonal_naive_baseline_on_the_test_window():
    forecaster = _two_model_forecaster()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        forecaster.evaluate()
    base = forecaster.naive_baseline
    y = forecaster.raw_df["orders"].to_numpy()
    origin = len(y) - base.n_test  # the last row the test forecast may see is origin - 1
    # The same weekday in the last observed week, repeated past day 7.
    expected = [y[origin - 7 + (step % 7)] for step in range(base.n_test)]
    assert base.model_name == "seasonal_naive"
    assert list(base.test_predictions["yhat"]) == list(expected)
    assert base.wape == base.wape and base.mape.value == base.mape.value


def test_models_worse_than_the_seasonal_naive_baseline_are_named():
    import numpy as np

    # A strong weekly cycle the badly-tuned catboost cannot follow.
    df = make_noisy_raw_df()
    df["orders"] = 100 + 60 * (np.arange(len(df)) % 7 == 0) + df["orders"] * 0.01
    forecaster = _construct(
        df,
        models=["catboost"],
        model_params={"catboost": {"iterations": 2, "depth": 1}},
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        forecaster.evaluate()
    worse = [m for m in _messages(caught) if "do worse than the seasonal-naive baseline" in m]
    assert worse
    assert "the same weekday last week" in worse[0]
    assert "catboost" in worse[0]
    assert "forecaster.naive_baseline" in worse[0]


def test_evaluate_reports_the_correlation_of_the_models_test_errors():
    forecaster = _two_model_forecaster()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        forecaster.evaluate()
    corr = forecaster.test_error_correlation
    assert list(corr.index) == list(corr.columns) == ["prophet", "catboost"]
    assert math.isclose(corr.loc["prophet", "prophet"], 1.0)
    assert -1.0 <= corr.loc["prophet", "catboost"] <= 1.0


def test_results_attributes_are_none_before_evaluate():
    forecaster = _construct(make_raw_df())
    assert forecaster.evaluation_results is None
    assert forecaster.naive_baseline is None
    assert forecaster.test_error_correlation is None


def test_an_all_zero_validation_window_runs_end_to_end():
    # 16 zero days in a 90-day series: 18% zeros, so selection is on WAPE,
    # and every validation score is NaN (nothing to divide by). evaluate()
    # and forecast() must still finish; the trimmed ensemble needs no score.
    df = make_noisy_raw_df()
    df.loc[66:81, "orders"] = 0.0
    forecaster = HorizonForecaster(
        df,
        date_column="date",
        target_column="orders",
        frequency="daily",
        validation_size=16,
        test_size=8,
        models=["prophet", "catboost"],
        model_params=dict(_QUICK),
    )
    assert forecaster.selection_metric == "wape"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = forecaster.evaluate()
        forecast = forecaster.forecast(horizon=5)
    for result in results.values():
        assert math.isnan(result.validation_wape)
        assert math.isnan(result.validation_mape.value)
        assert result.wape == result.wape  # the test window has real values
    assert forecaster.evaluate_ensemble["trimmed"].combined is not None
    assert set(forecast) == {"prophet", "catboost"}


def _weekdays_only(df):
    return df[pd.to_datetime(df["date"]).dt.dayofweek < 5].reset_index(drop=True)


def test_the_seasonal_naive_baseline_repeats_the_same_weekday_with_closed_weekends():
    df = make_noisy_raw_df(n=120)
    df.loc[pd.to_datetime(df["date"]).dt.dayofweek >= 5, "orders"] = 0.0
    forecaster = _construct(df, models=["catboost"], model_params={"catboost": _QUICK["catboost"]})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        forecaster.evaluate()
    base = forecaster.naive_baseline.test_predictions
    history = forecaster.raw_df.iloc[: len(forecaster.raw_df) - len(base)]
    last_week = history[history["date"] > history["date"].iloc[-1] - pd.Timedelta(days=7)]
    by_weekday = dict(zip(last_week["date"].dt.dayofweek, last_week["orders"]))
    expected = [by_weekday[d.dayofweek] for d in pd.to_datetime(base["ds"])]
    assert list(base["yhat"]) == expected


def test_a_mostly_zero_weekday_only_series_still_gets_the_weekly_suggestion():
    df = _weekdays_only(make_noisy_raw_df(n=120))
    df.loc[df.index % 5 != 0, "orders"] = 0.0  # sales on one day a week
    with pytest.raises(InvalidConfigError) as caught:
        _construct(df, selection_metric="mape")
    message = str(caught.value)
    assert "complete weeks" in message
    assert "frequency='weekly'" in message


def test_a_failed_evaluate_clears_the_previous_results(monkeypatch):
    forecaster = _two_model_forecaster()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        forecaster.evaluate()
    assert forecaster.naive_baseline is not None

    def fail(model_name):
        raise ModelConvergenceError(f"{model_name} failed", model_name=model_name)

    monkeypatch.setattr(forecaster, "_evaluate_single_model", fail)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(AllModelsRejectedError):
            forecaster.evaluate()
    assert forecaster.evaluation_results is None
    assert forecaster.naive_baseline is None
    assert forecaster.test_error_correlation is None


@pytest.mark.parametrize("given, stored", [("MAPE", "mape"), ("Wape", "wape"), (" AUTO ", "mape")])
def test_selection_metric_accepts_any_case(given, stored):
    assert _construct(make_noisy_raw_df(), selection_metric=given).selection_metric == stored


# ---------- SARIMAX and unexpected zeros ----------


def _weekly_df(n=60, zero_weeks=()):
    df = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=n, freq="7D"),
            "orders": [1000.0 + i for i in range(n)],
        }
    )
    df.loc[list(zero_weeks), "orders"] = 0.0
    return df


def test_sarimax_warns_that_it_leaves_unexpected_zero_days_out_of_its_fit():
    df = make_raw_df()
    df.loc[[10, 33], "orders"] = 0.0
    with pytest.warns(UserWarning, match="is 0 on 2 of 60 days that fall on weekdays") as caught:
        forecaster = _construct(df, models=["sarimax", "catboost"])
    message = next(str(w.message) for w in caught if "SARIMAX:" in str(w.message))
    assert "leaves these days out of its own fit; the other models still see them as 0" in message
    assert forecaster.models == ["sarimax", "catboost"]


def test_closed_weekday_zeros_do_not_warn_about_sarimax():
    df = make_raw_df()
    df.loc[pd.to_datetime(df["date"]).dt.dayofweek == 6, "orders"] = 0.0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _construct(df, models=["sarimax", "catboost"], selection_metric="mape")
    assert not [w for w in caught if "SARIMAX" in str(w.message)]


def test_weekly_zero_weeks_up_to_ten_percent_warn_and_keep_sarimax():
    with pytest.warns(UserWarning, match=r"is 0 in 6 of 60 weeks .*can still degrade its weekly"):
        forecaster = HorizonForecaster(
            _weekly_df(zero_weeks=range(5, 60, 10)),
            "date",
            "orders",
            "weekly",
            models=["sarimax", "catboost"],
            selection_metric="mape",
        )
    assert "sarimax" in forecaster.models and forecaster.skipped_models == {}


def test_weekly_zero_weeks_above_ten_percent_leave_sarimax_out():
    with pytest.warns(
        UserWarning, match="SARIMAX is left out of this run: the target 'orders' is 0 in 7"
    ):
        forecaster = HorizonForecaster(
            _weekly_df(zero_weeks=range(3, 60, 9)),
            "date",
            "orders",
            "weekly",
            models=["sarimax", "catboost"],
            selection_metric="mape",
        )
    assert forecaster.models == ["catboost"]
    assert "12% of the weeks" in forecaster.skipped_models["sarimax"]


def test_sarimax_alone_on_more_than_ten_percent_zero_weeks_is_refused():
    with pytest.raises(
        InvalidConfigError, match="SARIMAX cannot forecast this series: .* is 0 in 7"
    ):
        HorizonForecaster(
            _weekly_df(zero_weeks=range(3, 60, 9)),
            "date",
            "orders",
            "weekly",
            models=["sarimax"],
            selection_metric="mape",
        )
