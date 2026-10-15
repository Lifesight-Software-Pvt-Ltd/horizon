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
    # Weekly dates on the 1st-12th of each month only: every value fits
    # both DD/MM/YYYY and MM/DD/YYYY.
    df = pd.DataFrame(
        {
            "date": ["01/02/2026", "08/02/2026", "01/03/2026", "08/03/2026"],
            "orders": [10, 20, 30, 40],
        }
    )
    kwargs = dict(date_column="date", target_column="orders", frequency="weekly")
    with pytest.raises(InvalidConfigError, match="day-first or month-first"):
        HorizonForecaster(df, models=["prophet"], **kwargs)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # tiny series: scaling/gap warnings are not the point
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
    Previously this only warned ("irregular spacing should warn, not
    raise"). Four dates with gaps of 2, 7 and 15 days have no frequency at
    all, and the run went on to produce settings and forecast dates for one
    anyway.
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


def test_constructor_accepts_a_continuous_gap_of_exactly_four_weeks_with_a_warning():
    with pytest.warns(UserWarning, match=r"longest: 28 day\(s\)"):
        _build(_spacing_df(_DAILY.delete(range(100, 128))), "daily")
    with pytest.warns(UserWarning, match=r"longest: 4 week\(s\)"):
        _build(_spacing_df(_WEEKLY.delete(range(50, 54))), "weekly")


@pytest.mark.parametrize(
    "dates, frequency, run",
    [
        (_DAILY.delete(range(100, 129)), "daily", "29 day"),
        (_WEEKLY.delete(range(50, 55)), "weekly", "5 week"),
    ],
)
def test_constructor_rejects_a_continuous_gap_longer_than_four_weeks(dates, frequency, run):
    with pytest.raises(InvalidConfigError, match=rf"{run}\(s\) missing in a row"):
        _build(_spacing_df(dates), frequency)


def test_constructor_warns_on_scattered_missing_dates_with_the_share_present():
    # Every 4th day removed: 25% missing, but never more than one in a row.
    dates = _DAILY.delete(range(3, 400, 4))
    with pytest.warns(UserWarning, match=r"75% of days are present"):
        forecaster = _build(_spacing_df(dates), "daily")
    assert len(forecaster.raw_df) == 300


def test_constructor_accepts_weekday_only_daily_data_with_a_warning():
    weekdays = _DAILY[_DAILY.dayofweek < 5]
    with pytest.warns(UserWarning, match="has gaps"):
        _build(_spacing_df(weekdays), "daily")


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
    df = make_raw_df(n=10)  # 80% train -> 8 rows, below daily floor of 14
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
    CatBoostAdapter's redesign removed the lag-warmup mechanism that
    used to give CatBoost a higher EFFECTIVE min-rows floor than
    Prophet's (see CHANGELOG) -- both models now use the exact same raw
    row-count check. So a genuine partial failure here needs a real
    model-family failure, not a data-size coincidence: depth=-1 is a
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
    Direct regression test for the orchestrator's split construction,
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
    The old rule only fired above 2x the ENTIRE history, so forecasting
    100 periods from 121 was silent -- the case that produced a real
    implausible forecast. Now half the history is enough to flag.
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
        # training rows, which the scaled-down defaults (33 of 60) no longer leave.
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
        # training rows, which the scaled-down defaults (33 of 60) no longer leave.
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

    Regression cover: candidate-skipping originally wrapped every total
    failure in ModelConvergenceError, which turned a plain "not enough rows"
    into a misleading convergence error and dropped
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
        # training rows, which the scaled-down defaults (33 of 60) no longer leave.
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
        # training rows, which the scaled-down defaults (33 of 60) no longer leave.
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
    assert set(ensemble.combined.columns) == {"ds", "yhat"}


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
    Before 1.0 this ran the whole of evaluate() internally, additionally
    paying for a test-window prediction it then threw away.
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
    assert set(forecaster.forecast_ensemble["trimmed"].combined.columns) == {"ds", "yhat"}
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
        forecaster.forecast(horizon=5)  # must not warn -- evaluate() already ran


def test_forecast_succeeds_when_the_model_cannot_be_scored_on_validation():
    """
    n=16 -> evaluate()'s 80% train split is 12 rows, below the daily
    floor of 14 (fails outright), but forecast()'s own full-data refit
    uses all 16 rows, which clears the floor. This confirms forecast()
    still attempts (and succeeds at) its own fits even when the
    internal evaluate() call it runs for ensemble weights finds nothing
    at all -- a model failing on the smaller train/test split doesn't
    block forecast() from succeeding with more data.
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
# first appears in validation. NeuralProphet used to crash on it and Prophet
# to invent an effect; it is now left out of that window's fit for every
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
